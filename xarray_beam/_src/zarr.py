# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IO with Zarr via Xarray."""
from __future__ import annotations
import collections
import dataclasses
import logging
from typing import (
    Any,
    AbstractSet,
    Dict,
    Optional,
    Mapping,
    Set,
    Tuple,
    Union,
    MutableMapping,
)

import apache_beam as beam
import dask
import dask.array
import xarray

from xarray_beam._src import core
from xarray_beam._src import rechunk
from xarray_beam._src import threadmap

# pylint: disable=logging-fstring-interpolation

ReadableStore = Union[str, Mapping[str, bytes]]
WritableStore = Union[str, MutableMapping[str, bytes]]


def _infer_chunks(dataset: xarray.Dataset) -> Mapping[str, int]:
  """Infer chunks for an xarray.Dataset loaded from Zarr."""
  # The original Zarr array chunks (as tuples) are stored in the "encoding"
  # dictionary on each DataArray object.

  chunks_sets = collections.defaultdict(set)
  for name, variable in dataset.items():
    # exclude variables that are indexed, which are loaded into memory already
    if name not in dataset.indexes:
      for dim, size in zip(variable.dims, variable.encoding['chunks']):
        chunks_sets[dim].add(size)

  chunks = {}
  for dim, sizes in chunks_sets.items():
    if len(sizes) > 1:
      raise ValueError(
          f'inconsistent chunk sizes on Zarr dataset for dimension {dim!r}: '
          f'{sizes}'
      )
    (chunks[dim],) = sizes
  return chunks


def open_zarr(
    store: ReadableStore, **kwargs: Any
) -> Tuple[xarray.Dataset, Mapping[str, int]]:
  """Returns a lazily indexable xarray.Dataset and chunks from a Zarr store.

  Only Zarr stores with the consistent chunking between non-indexed variables
  (i.e., those for which the ``Dataset.chunks`` property is valid) can be
  opened.

  Args:
    store: Xarray compatible Zarr store to open.
    **kwargs: passed on to xarray.open_zarr. The "chunks" keyword argument is
      not supported.

  Returns:
    (dataset, chunks) pair, consisting of a Dataset with the contents of the
    Zarr store and a mapping from dimensions to integer chunk sizes.
  """
  if 'chunks' in kwargs:
    raise TypeError(
        'xarray_beam.open_zarr does not support the `chunks` argument'
    )
  dataset = xarray.open_zarr(store, **kwargs, chunks=None)
  chunks = _infer_chunks(dataset)
  return dataset, chunks


def _raise_template_error():
  raise ValueError(
      'cannot compute array values of xarray.Dataset objects created directly '
      'or indirectly from xarray_beam.make_template()'
  )


def make_template(
    dataset: xarray.Dataset,
    lazy_vars: Optional[AbstractSet[str]] = None,
) -> xarray.Dataset:
  """Make a lazy Dask xarray.Dataset for use only as a template.

  Lazy variables in an xarray.Dataset can be manipulated with xarray operations,
  but cannot be computed.

  Args:
    dataset: dataset to convert into a template.
    lazy_vars: optional explicit set of variables to make lazy. By default, all
      data variables and coordinates that are not used as an index are made
      lazy, matching xarray.Dataset.chunk.

  Returns:
    Dataset with lazy variables. Lazy variable each use a single Dask chunk.
    Non-lazy variables are loaded in memory as NumPy arrays.
  """
  if lazy_vars is None:
    lazy_vars = set(dataset.keys())
    lazy_vars.update(k for k in dataset.coords if k not in dataset.indexes)

  result = dataset.copy()

  # load non-lazy variables into memory
  result.update(dataset.drop_vars(lazy_vars).compute())

  # override the lazy variables
  delayed = dask.delayed(_raise_template_error)()
  name = 'make_template'
  for k, v in dataset.variables.items():
    if k in lazy_vars:
      result[k].data = dask.array.from_delayed(
          delayed, v.shape, v.dtype, name=name
      )

  return result


def _unchunked_vars(ds: xarray.Dataset) -> Set[str]:
  return {k for k, v in ds.variables.items() if v.chunks is None}


def _chunked_vars(ds: xarray.Dataset) -> Set[str]:
  return set(ds.variables.keys()) - _unchunked_vars(ds)


def _make_template_from_chunked(dataset: xarray.Dataset) -> xarray.Dataset:
  """Create a template with lazy variables already chunked with Dask."""
  return make_template(dataset, lazy_vars=_chunked_vars(dataset))


class _DiscoverTemplate(beam.PTransform):
  """Discover the Zarr template from (Key, xarray.Dataset) pairs."""

  def _make_template_chunk(self, key, chunk):
    # The current rule is that everything that *can* be chunked with Dask
    # *should* be chunked, but conceivably this should be customizable, e.g.,
    # for handling 2D latitude/longitude arrays. It's usually safe to rewrite
    # overlapping arrays multiple times in different chunks (Zarr writes are
    # typically atomic), but this may be wasteful.
    return key, make_template(chunk)

  def _consolidate(self, inputs):
    # don't bother with compatibility checks; we won't be computing the values
    # here anyways
    kwargs = {'compat': 'override'}
    _, template = rechunk.consolidate_fully(
        inputs, combine_kwargs=kwargs, merge_kwargs=kwargs
    )
    return template

  def expand(self, pcoll):
    # TODO(shoyer): can we refactor this logic to do a hierarchical merge (i.e.,
    # with beam.CombineGlobally), rather than combining all templates into a
    # list on a single machine? This would help for scalability.
    return (
        pcoll
        | 'MakeChunk' >> beam.MapTuple(self._make_template_chunk)
        | 'ListChunks' >> beam.combiners.ToList()
        | 'ConsolidateChunks' >> beam.Map(self._consolidate)
    )


def _verify_template_is_lazy(template: xarray.Dataset):
  """Verify that a Dataset is suitable for use as a Zarr template."""
  if not template.chunks:
    # We require at least one chunked variable with Dask. Otherwise, there would
    # be no data to write as part of the Beam pipeline.
    raise ValueError(
        'template does not have any variables chunked with Dask. Convert any '
        'variables that will be written in the pipeline into lazy dask arrays, '
        f'e.g., with xarray_beam.make_template():\n{template}'
    )


def _override_chunks(
    dataset: xarray.Dataset,
    chunks: Mapping[str, int],
) -> xarray.Dataset:
  """Override chunks on a Dataset, for already chunked variables only."""

  def maybe_rechunk(variable):
    if variable.chunks is None:
      return variable
    else:
      relevant_chunks = {k: v for k, v in chunks.items() if k in variable.dims}
      return variable.chunk(relevant_chunks)

  data_vars = {
      k: maybe_rechunk(dataset.variables[k]) for k in dataset.data_vars
  }
  coords = {k: maybe_rechunk(dataset.variables[k]) for k in dataset.coords}
  return xarray.Dataset(data_vars, coords, dataset.attrs)


def _dask_to_zarr_chunksize(dim: str, sizes: Tuple[int, ...]) -> int:
  if not sizes:
    return 0
  # It's OK for the last chunk of Zarr array to have smaller size. Otherwise,
  # there should be (at most) one chunk size.
  size_set = set(sizes[:-1])
  if len(size_set) > 1 or sizes[-1] > sizes[0]:
    raise ValueError(
        'Zarr cannot handle inconsistent chunk sizes along dimension '
        f'{dim!r}: {sizes}'
    )
  return sizes[0]


def _infer_zarr_chunks(dataset: xarray.Dataset) -> Dict[str, int]:
  return {
      dim: _dask_to_zarr_chunksize(dim, sizes)
      for dim, sizes in dataset.chunks.items()
  }


def _setup_zarr(template, store, zarr_chunks):
  """Setup a Zarr store."""
  if zarr_chunks is not None:
    template = _override_chunks(template, zarr_chunks)
  _verify_template_is_lazy(template)
  # inconsistent chunks in encoding can lead to spurious failures in xarray:
  # https://github.com/pydata/xarray/issues/5219
  template2 = template.copy(deep=False)
  for var in template2.variables.values():
    if 'chunks' in var.encoding:
      del var.encoding['chunks']
  logging.info(f'writing Zarr metadata for template:\n{template}')
  template2.to_zarr(store, compute=False, consolidated=True, mode='w')


def _validate_zarr_chunk(key, chunk, template, zarr_chunks):
  """Check a chunk for consistency against the given template."""
  unexpected_indexes = [k for k in chunk.indexes if k not in template.indexes]
  if unexpected_indexes:
    raise ValueError(
        'unexpected new indexes found in chunk but not template: '
        f'{unexpected_indexes}'
    )

  region = core.offsets_to_slices(key.offsets, chunk.sizes)
  for dim, full_index in template.indexes.items():
    if dim in chunk.indexes:
      expected_index = full_index[region[dim]]
      actual_index = chunk.indexes[dim]
      if not expected_index.equals(actual_index):
        raise ValueError(
            f'template and chunk indexes do not match for dim {dim!r}:\n'
            f'{expected_index}\nvs.\n{actual_index}'
        )

  if zarr_chunks is None:
    zarr_chunks = {k: v[0] for k, v in template.chunks.items()}
  for dim, offset in key.offsets.items():
    if dim not in zarr_chunks:
      continue
    if offset % zarr_chunks[dim]:
      raise ValueError(
          f'chunk offset {offset} along dimension {dim!r} is not a multiple of '
          f'zarr chunks {zarr_chunks}'
      )
    if (
        chunk.sizes[dim] % zarr_chunks[dim]
        and offset + chunk.sizes[dim] != template.sizes[dim]
    ):
      raise ValueError(
          f'chunk is smaller than zarr chunks {zarr_chunks} along at least one '
          'dimension, which can lead to a race condition that results in '
          'incomplete writes. Use ConsolidateChunks() or Rechunk() to ensure '
          'appropriate chunk sizes before feeding data into ChunksToZarr(). '
          f'\nkey={key}\nchunk={chunk}'
      )

  # TODO(shoyer): consider verifying "already_written" variables for
  # consistency, maybe with an opt-in flag?
  # Note that variable names, shapes & dtypes are verified in xarray's to_zarr()


def _write_chunk_to_zarr(key, chunk, store, template):
  """Write a single Dataset chunk to Zarr."""
  region = core.offsets_to_slices(key.offsets, chunk.sizes)
  already_written = [
      k for k in chunk.variables if k in _unchunked_vars(template)
  ]
  writable_chunk = chunk.drop_vars(already_written)
  try:
    future = writable_chunk.chunk().to_zarr(
        store, region=region, compute=False, consolidated=True
    )
    future.compute(num_workers=len(writable_chunk))
  except Exception as e:
    raise RuntimeError(
        f'failed to write chunk corresponding to key={key}:\n{writable_chunk}'
    ) from e


class ChunksToZarr(beam.PTransform):
  """Write keyed chunks to a Zarr store in parallel."""

  def __init__(
      self,
      store: WritableStore,
      template: Union[xarray.Dataset, beam.pvalue.AsSingleton, None] = None,
      zarr_chunks: Optional[Mapping[str, int]] = None,
      num_threads: Optional[int] = None,
  ):
    # pyformat: disable
    """Initialize ChunksToZarr.

    Args:
      store: a string corresponding to a Zarr path or an existing Zarr store.
      template: an argument providing a lazy xarray.Dataset already chunked
        using Dask (e.g., as created by `xarray_beam.make_template`) that
        matches the structure of the virtual combined dataset corresponding to
        the chunks fed into this PTransform. One or more variables are expected
        to be "chunked" with Dask, and will only have their metadata written to
        Zarr without array values. Three types of inputs are supported:

        1. If `template` is an xarray.Dataset, the Zarr store is setup eagerly.
        2. If `template` is a beam.pvalue.AsSingleton object representing the
           result of a prior step in a Beam pipeline, the Zarr store is setup as
           part of the pipeline.
        3. Finally, if `template` is None, then the structure of the desired
           Zarr store is discovered automatically by inspecting the inputs into
           ChunksToZarr. This is an easy option, but can be quite expensive/slow
           for large datasets -- Beam runners will typically handle this by
           dumping a temporary copy of the complete dataset to disk. For best
           performance, supply the template explicitly (1 or 2).
      zarr_chunks: chunking scheme to use for Zarr. If set, overrides the
        chunking scheme on already chunked arrays in template.
      num_threads: the number of Dataset chunks to write in parallel per worker.
        More threads can increase throughput, but also increases memory usage
        and makes it harder for Beam runners to shard work. Note that each
        variable in a Dataset is already written in parallel, so this is most
        useful for Datasets with a small number of variables.
    """
    # pyformat: enable
    if isinstance(template, xarray.Dataset):
      _setup_zarr(template, store, zarr_chunks)
      if zarr_chunks is None:
        zarr_chunks = _infer_zarr_chunks(template)
      template = _make_template_from_chunked(template)
    elif isinstance(template, beam.pvalue.AsSingleton):
      pass
    elif template is None:
      pass
    else:
      raise TypeError(
          'template must be an None, an xarray.Dataset, or a '
          f'beam.pvalue.AsSingleton object: {template}'
      )
    self.store = store
    self.template = template
    self.zarr_chunks = zarr_chunks
    self.num_threads = num_threads

  def _validate_zarr_chunk(self, key, chunk, template=None):
    # If template doesn't have a default value, Beam errors with "Side inputs
    # must have defaults for MapTuple". Beam should probably be happy with a
    # keyword-only argument, too, but it doesn't like that yet.
    assert template is not None
    _validate_zarr_chunk(key, chunk, template, self.zarr_chunks)
    return key, chunk

  def _write_chunk_to_zarr(self, key, chunk, template=None):
    assert template is not None
    return _write_chunk_to_zarr(key, chunk, self.store, template)

  def expand(self, pcoll):
    if isinstance(self.template, xarray.Dataset):
      template = self.template
      setup_result = None  # already setup in __init__
    else:
      if isinstance(self.template, beam.pvalue.AsSingleton):
        template = self.template
      else:
        assert self.template is None
        template = beam.pvalue.AsSingleton(
            pcoll | 'DiscoverTemplate' >> _DiscoverTemplate()
        )
      setup_result = beam.pvalue.AsSingleton(
          template.pvalue
          | 'SetupZarr' >> beam.Map(_setup_zarr, self.store, self.zarr_chunks)
      )
    return (
        pcoll
        | 'WaitForSetup' >> beam.Map(lambda x, _: x, setup_result)
        | 'ValidateChunks'
        >> beam.MapTuple(self._validate_zarr_chunk, template=template)
        | 'WriteChunks'
        >> threadmap.ThreadMapTuple(
            self._write_chunk_to_zarr,
            template=template,
            num_threads=self.num_threads,
        )
    )


@dataclasses.dataclass
class DatasetToZarr(beam.PTransform):
  """Write an entire xarray.Dataset to a Zarr store."""

  dataset: xarray.Dataset
  store: WritableStore
  zarr_chunks: Optional[Mapping[str, int]] = None

  def expand(self, pcoll):
    # Unchunked variables will be written eagerly via the template, so there's
    # no need to feed them into the pipeline, too.
    source_dataset = self.dataset.drop_vars(_unchunked_vars(self.dataset))
    return (
        pcoll
        | core.DatasetToChunks(source_dataset)
        | ChunksToZarr(
            self.store, template=self.dataset, zarr_chunks=self.zarr_chunks
        )
    )
