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
import logging
from typing import List, Optional, Mapping, Union, MutableMapping

import apache_beam as beam
import dataclasses
import xarray

from xarray_beam._src import core
from xarray_beam._src import rechunk
from xarray_beam._src import threadmap

# pylint: disable=logging-format-interpolation


class _DiscoverTemplate(beam.PTransform):
  """Discover the Zarr template from (ChunkKey, xarray.Dataset) pairs."""

  def _make_template_chunk(self, key, chunk):
    # Make a lazy Dask xarray.Dataset of all zeros shaped like this chunk.
    # The current rule is that everything that *can* be chunked with Dask
    # *should* be chunked, but conceivably this should be customizable, e.g.,
    # for handling 2D latitude/longitude arrays. It's usually safe to rewrite
    # overlapping arrays multiple times in different chunks (Zarr writes are
    # typically atomic), but this may be wasteful.
    zeros = xarray.zeros_like(chunk.chunk(-1))
    return key, zeros

  def _consolidate_chunks(self, inputs):
    _, template = rechunk.consolidate_chunks(inputs)
    return template

  def expand(self, pcoll):
    return (
        pcoll
        | 'MakeChunk' >> beam.MapTuple(self._make_template_chunk)
        | 'ListChunks' >> beam.combiners.ToList()
        | 'ConsolidateChunks' >> beam.Map(self._consolidate_chunks)
    )


def _verify_template_is_lazy(template: xarray.Dataset):
  """Verify that a Dataset is suitable for use as a Zarr template."""
  if not template.chunks:
    # We require at least one chunked variable with Dask. Otherwise, there would
    # be no data to write as part of the Beam pipeline.
    raise ValueError(
        f'template does not have any variables chunked with Dask:\n{template}'
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
  template2.to_zarr(
      store, compute=False, consolidated=True, mode='w',
  )


def _validate_chunk(key, chunk, template):
  """Check a chunk for consistency against the given template."""
  unexpected_indexes = [k for k in chunk.indexes if k not in template.indexes]
  if unexpected_indexes:
    raise ValueError(
        'unexpected new indexes found in chunk but not template: '
        f'{unexpected_indexes}'
    )
  region = key.to_slices(chunk.sizes)
  for dim, full_index in template.indexes.items():
    if dim in chunk.indexes:
      expected_index = full_index[region[dim]]
      actual_index = chunk.indexes[dim]
      if not expected_index.equals(actual_index):
        raise ValueError(
            f'template and chunk indexes do not match for dim {dim}:\n'
            f'{expected_index}\nvs.\n{actual_index}'
        )
  # TODO(shoyer): consider verifying "already_written" variables for
  # consistency, maybe with an opt-in flag?
  # Note that variable names, shapes & dtypes are verified in xarray's to_zarr()


def _unchunked_vars(ds: xarray.Dataset) -> List[str]:
  return [k for k, v in ds.variables.items() if v.chunks is None]


def _write_chunk_to_zarr(key, chunk, store, template):
  """Write a single Dataset chunk to Zarr."""
  region = key.to_slices(chunk.sizes)
  already_written = [
      k for k in chunk.variables if k in _unchunked_vars(template)
  ]
  writable_chunk = chunk.drop_vars(already_written)
  try:
    future = writable_chunk.chunk().to_zarr(store, region=region, compute=False)
    future.compute(num_workers=len(writable_chunk))
  except Exception as e:
    raise RuntimeError(
        f'failed to write chunk corresponding to key={key}:\n{writable_chunk}'
    ) from e


class ChunksToZarr(beam.PTransform):
  """Write keyed chunks to a Zarr store in parallel."""

  def __init__(
      self,
      store: Union[str, MutableMapping[str, bytes]],
      template: Union[xarray.Dataset, beam.pvalue.AsSingleton, None] = None,
      zarr_chunks: Optional[Mapping[str, int]] = None,
      num_threads: Optional[int] = None,
  ):
    """Initialize ChunksToZarr.

    Args:
      store: a string corresponding to a Zarr path or an existing Zarr store.
      template: an argument providing an xarray.Dataset already chunked using
        Dask that matches the structure of xarray.Dataset "chunks" that will be
        fed into this PTransform. One or more variables are expected to be
        "chunked" with Dask, and will only have their metadata written to Zarr
        without array values. Three types of inputs are supported:
        1. If `template` is an xarray.Dataset, the Zarr store is setup eagerly.
        2. If `template` is a beam.pvalue.AsSingleton object representing the
           result of a prior step in a Beam pipeline, the Zarr store is setup as
           part of the pipeline.
        3. Finally, if `template` is None, then the structure of the desired
           Zarr store is discovered automatically by inspecting the inputs into
           ChunkToZarr. This is an easy option, but can be quite expensive/slow
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
    if isinstance(template, xarray.Dataset):
      _setup_zarr(template, store, zarr_chunks)
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

  def _validate_chunk(self, key, chunk, template=None):
    # If template doesn't have a default value, Beam errors with "Side inputs
    # must have defaults for MapTuple". Beam should probably be happy with a
    # keyword-only argument, too, but it doesn't like that yet.
    assert template is not None
    _validate_chunk(key, chunk, template)
    return key, chunk

  def _write_chunk_to_zarr(self, key, chunk, template=None):
    assert template is not None
    return _write_chunk_to_zarr(key, chunk, self.store, template)

  def expand(self, pcoll):
    if isinstance(self.template, xarray.Dataset):
      template = self.template
      setup_result = None
    else:
      if isinstance(self.template, beam.pvalue.AsSingleton):
        template = self.template
      else:
        assert self.template is None
        template = beam.pvalue.AsSingleton(
            pcoll
            | 'DiscoverTemplate' >> _DiscoverTemplate()
        )
      setup_result = beam.pvalue.AsSingleton(
          template.pvalue
          | 'SetupZarr' >> beam.Map(_setup_zarr, self.store, self.zarr_chunks)
      )
    return (
        pcoll
        | 'WaitForSetup' >> beam.Map(lambda x, _: x, setup_result)
        | 'ValidateChunks' >> beam.MapTuple(
            self._validate_chunk, template=template,
        )
        | 'WriteChunks' >> threadmap.ThreadMapTuple(
            self._write_chunk_to_zarr,
            template=template,
            num_threads=self.num_threads,
        )
    )


@dataclasses.dataclass
class DatasetToZarr(beam.PTransform):
  """Write an entire xarray.Dataset to a Zarr store."""
  dataset: xarray.Dataset
  store: Union[str, MutableMapping[str, bytes]]
  zarr_chunks: Optional[Mapping[str, int]] = None

  def expand(self, pcoll):
    # Unchunked variables will be written eagerly via the template, so there's
    # no need to feed them into the pipeline, too.
    source_dataset = self.dataset.drop_vars(_unchunked_vars(self.dataset))
    return (
        pcoll
        | core.DatasetToChunks(source_dataset)
        | ChunksToZarr(
            self.store, template=self.dataset, zarr_chunks=self.zarr_chunks,
        )
    )
