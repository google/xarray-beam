# Copyright 2023 Google LLC
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
"""A high-level interface for Xarray-Beam datasets.

Usage example:

    import xarray_beam as xbeam

    transform = (
        xbeam.Dataset.from_zarr(input_path)
        .rechunk({'time': -1, 'latitude': 10, 'longitude': 10})
        .map_blocks(lambda x: x.median('time'))
        .to_zarr(output_path)
    )
    with beam.Pipeline() as p:
      p | transform
"""
from __future__ import annotations

import collections
from collections.abc import Mapping
import dataclasses
import functools
import itertools
import math
import operator
import os.path
import tempfile
import textwrap
from typing import Callable, Literal

import apache_beam as beam
import dask
import dask.array.api
import numpy as np
import numpy.typing as npt
import xarray
from xarray_beam._src import combiners
from xarray_beam._src import core
from xarray_beam._src import rechunk
from xarray_beam._src import zarr


def _at_least_two_digits(n: int | float) -> str:
  if isinstance(n, int):
    return str(n)
  elif round(n, 2) < 10:
    return f'{n:.1f}'
  else:
    return f'{n:.0f}'


def _to_human_size(nbytes: int) -> str:
  """Convert a number of bytes to a human-readable string."""
  for unit in ['B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB']:
    if nbytes < 1000:
      return f'{_at_least_two_digits(nbytes)}{unit}'
    nbytes /= 1000
  nbytes *= 1000
  return f'{_at_least_two_digits(nbytes)}EB'


def normalize_chunks(
    chunks: Mapping[str, int | str] | str,
    template: xarray.Dataset,
    split_vars: bool = False,
    previous_chunks: Mapping[str, int] | None = None,
) -> dict[str, int]:
  """Normalize chunks for a xarray.Dataset.

  This function interprets various chunk specifications (e.g., -1, 'auto',
  byte-strings) and returns a dictionary mapping dimension names to
  concrete integer chunk sizes. It uses ``dask.array.api.normalize_chunks``
  under the hood.

  Chunk specifications for each dimension can be one of the following:
    - ``-1``: along this dimension chunks should be the full size of the
      dimension.
    - An integer: the exact chunk size for this dimension.
    - A byte-string (e.g., "64MiB", "1GB"): indicates that dask should pick
      chunk sizes to aim for chunks of approximately this size. If byte limits
      are specified for multiple dimensions, they must be consistent (i.e.,
      parse to the same number of bytes).
    - ``'auto'``: chunks will be automatically determined for all 'auto'
      dimensions to ensure chunks are approximately the target number of bytes
      (defaulting to 128MiB, if no byte limits are specified).

  Args:
    chunks: The desired chunking scheme. Can either be a dictionary mapping
      dimension names to chunk sizes, or a single string chunk specification
      (e.g., 'auto' or '100MiB') to be applied as the default for all
      dimensions. Dimensions not included in the dictionary default to
      previous_chunks (if available) or the full size of the dimension.
    template: An xarray.Dataset providing dimension sizes and dtype information,
      used for calculating chunk sizes in bytes.
    split_vars: If True, chunk size limits are applied per-variable, based on
      the largest variable's dtype. If False, limits are applied to chunks
      containing all variables, based on the sum of dtypes for all variables.
    previous_chunks: If provided, hints to dask that chunks should be multiples
      of ``previous_chunks``, if possible.

  Returns:
    A dictionary mapping all dimension names to integer chunk sizes.
  """
  if isinstance(chunks, str):
    chunks = {k: chunks for k in template.dims}

  string_chunks = {v for v in chunks.values() if isinstance(v, str)}
  string_chunks.discard('auto')
  if len(string_chunks) > 1:
    raise ValueError(
        f'cannot specify multiple distinct chunk sizes in bytes: {chunks}'
    )

  defaults = previous_chunks if previous_chunks else template.sizes
  chunks: dict[str, int | str] = {**defaults, **chunks}  # pytype: disable=annotation-type-mismatch

  dtypes = {
      k: v.dtype for k, v in template.variables.items() if v.chunks is not None
  }
  if not dtypes:
    combined_dtype = np.dtype('uint8')  # dummy dtype, won't be used by dask
  elif split_vars:
    combined_dtype = max(dtypes.values(), key=lambda dtype: dtype.itemsize)
  else:
    combined_dtype = np.dtype(list(dtypes.items()))

  chunks_tuple = tuple(chunks.values())
  shape = tuple(template.sizes[k] for k in chunks)
  prev_chunks_tuple = (
      tuple(previous_chunks[k] for k in chunks) if previous_chunks else None
  )

  # Note: This values are the same as the dask defaults. Set them explicitly
  # here to ensure that Xarray-Beam behavior does not depend on the user's
  # dask configuration.
  with dask.config.set({
      'array.chunk-size': '128MiB',
      'array.chunk-size-tolerance': 1.25,
  }):
    normalized_chunks_tuple = dask.array.api.normalize_chunks(
        chunks_tuple,
        shape,
        dtype=combined_dtype,
        previous_chunks=prev_chunks_tuple,
    )
  return {k: v[0] for k, v in zip(chunks, normalized_chunks_tuple)}


def _infer_new_chunks(
    old_sizes: Mapping[str, int],
    old_chunks: Mapping[str, int],
    new_sizes: Mapping[str, int],
) -> Mapping[str, int]:
  """Compute new chunks based on old and new sizes."""
  new_chunks = {}
  for dim, new_size in new_sizes.items():
    assert isinstance(dim, str)

    if dim not in old_sizes:
      new_chunks[dim] = new_size
    elif new_size == old_sizes[dim]:
      new_chunks[dim] = old_chunks[dim]
    else:
      old_size = old_sizes[dim]
      count, remainder = divmod(old_size, old_chunks[dim])
      if remainder != 0:
        raise ValueError(
            f'cannot infer new chunks for dimension {dim!r} with changed size '
            f'{old_size} -> {new_size}: existing chunks {old_chunks} do not '
            f'evenly divide existing sizes {old_sizes}'
        )
      new_chunks[dim], remainder = divmod(new_size, count)
      if remainder != 0:
        raise ValueError(
            f'cannot infer new chunks for dimension {dim!r} with changed size '
            f'{old_size} -> {new_size}: the {count} chunks along this '
            f'dimension do not evenly divide the new size {new_size}'
        )

  return new_chunks


def _apply_to_each_chunk(
    func: Callable[[xarray.Dataset], xarray.Dataset],
    old_chunks: Mapping[str, int],
    new_chunks: Mapping[str, int],
    key: core.Key,
    chunk: xarray.Dataset,
) -> tuple[core.Key, xarray.Dataset]:
  """Apply a function to each chunk."""
  new_chunk = func(chunk)
  new_offsets = {}
  for dim in new_chunk.dims:
    assert isinstance(dim, str)
    new_offsets[dim] = (
        key.offsets.get(dim, 0) // old_chunks.get(dim, 1) * new_chunks[dim]
    )
  new_vars = set(new_chunk) if key.vars is not None else None
  new_key = core.Key(new_offsets, new_vars)
  return new_key, new_chunk


def _concat_labels(label1: str | None, label2: str) -> str:
  """Concatenate Beam PTransform labels."""
  return f'{label1}|{label2}' if label1 is not None else label2


def _whole_dataset_method(method_name: str):
  """Helper function for defining a method with a fast-path for lazy data."""

  def method(self: Dataset, *args, **kwargs) -> Dataset:
    func = operator.methodcaller(method_name, *args, **kwargs)
    template = zarr.make_template(func(self.template))
    chunks = {k: v for k, v in self.chunks.items() if k in template.dims}

    label = _get_label(method_name)
    if isinstance(self.ptransform, core.DatasetToChunks):
      # Some transformations (e.g., indexing) can be applied much less
      # expensively to xarray.Dataset objects rather than via Xarray-Beam. Try
      # to preserve this option for downstream transformations if possible.
      dataset = func(self.ptransform.dataset)
      ptransform = core.DatasetToChunks(dataset, chunks, self.split_vars)
      ptransform.label = _concat_labels(self.ptransform.label, label)
    else:
      ptransform = self.ptransform | label >> beam.MapTuple(
          functools.partial(_apply_to_each_chunk, func, self.chunks, chunks)
      )
    return Dataset(template, chunks, self.split_vars, ptransform)

  return method


class _CountNamer:

  def __init__(self):
    self._counts = collections.defaultdict(itertools.count)

  def apply(self, name: str) -> str:
    return f'{name}_{next(self._counts[name])}'


_get_label = _CountNamer().apply


@dataclasses.dataclass
class Dataset:
  """Experimental high-level representation of an Xarray-Beam dataset."""

  def __init__(
      self,
      template: xarray.Dataset,
      chunks: Mapping[str, int],
      split_vars: bool,
      ptransform: beam.PTransform,
  ):
    """Low level interface for creating a new Dataset, without validation.

    Most users should use the higher level
    :py:class:`xarray_beam.Dataset.from_xarray` or
    :py:class:`xarray_beam.Dataset.from_zarr` instead.

    Args:
      template: xarray.Dataset describing the structure of this dataset,
        typically as produced by :py:func:`xarray_beam.make_template`.
      chunks: mapping from dimension names to chunk sizes. For normalization,
        use :py:func:`xarray_beam.normalize_chunks`.
      split_vars: whether variables are split between separate elements in the
        ptransform, or all stored in the same element.
      ptransform: Beam PTransform of ``(xbeam.Key, xarray.Dataset)`` tuples with
        this dataset's data.
    """
    self._template = template
    self._chunks = chunks
    self._split_vars = split_vars
    self._ptransform = ptransform

  @property
  def template(self) -> xarray.Dataset:
    """Template describing the structure of this dataset."""
    return self._template

  @property
  def chunks(self) -> Mapping[str, int]:
    """Dictionary mapping from dimension names to chunk sizes."""
    return dict(self._chunks)

  @property
  def split_vars(self) -> bool:
    """Whether variables are split between separate elements in the ptransform."""
    return self._split_vars

  @property
  def ptransform(self) -> beam.PTransform:
    """Beam PTransform of (xbeam.Key, xarray.Dataset) with this dataset's data."""
    return self._ptransform

  @property
  def sizes(self) -> Mapping[str, int]:
    """Size of each dimension on this dataset."""
    return dict(self.template.sizes)  # pytype: disable=bad-return-type

  @property
  def bytes_per_chunk(self) -> int:
    """Estimate of the number of bytes per dataset chunk."""
    variable_sizes = [
        v.dtype.itemsize * math.prod(self.chunks[d] for d in v.dims)
        for v in self.template.values()
    ]
    return max(variable_sizes) if self.split_vars else sum(variable_sizes)

  @property
  def chunk_count(self) -> int:
    """Count the number of chunks in this dataset."""
    if self.split_vars:
      total = 0
      for variable in self.template.values():
        total += math.prod(
            math.ceil(self.sizes[d] / self.chunks[d]) for d in variable.dims
        )
      return total
    else:
      return math.prod(
          math.ceil(self.sizes[d] / self.chunks[d]) for d in self.sizes
      )

  def __repr__(self):
    base = repr(self.template)
    chunks_str = ', '.join(
        [f'{k}: {v}' for k, v in self.chunks.items()]
        + [f'split_vars={self.split_vars}']
    )
    chunk_size = _to_human_size(self.bytes_per_chunk)
    total_size = _to_human_size(self.template.nbytes)
    chunk_count = self.chunk_count
    plural = 's' if chunk_count != 1 else ''
    return (
        '<xarray_beam.Dataset>\n'
        f'PTransform: {self.ptransform}\n'
        f'Chunks:     {chunk_size} ({chunks_str})\n'
        f'Template:   {total_size} ({chunk_count} chunk{plural})\n'
        + textwrap.indent('\n'.join(base.split('\n')[1:]), ' ' * 4)
    )

  @classmethod
  def from_xarray(
      cls,
      source: xarray.Dataset,
      chunks: Mapping[str, int | str] | str,
      *,
      split_vars: bool = False,
      previous_chunks: Mapping[str, int] | None = None,
  ) -> Dataset:
    """Create an xarray_beam.Dataset from an xarray.Dataset.

    Args:
      source: xarray.Dataset to read from.
      chunks: optional mapping from dimension names to chunk sizes, or any value
        that can be passed to :py:func:`xarray_beam.normalize_chunks`.
      split_vars: whether variables are split between separate elements in the
        ptransform, or all stored in the same element.
      previous_chunks: chunks hint used for parsing string values in ``chunks``
        with ``normalize_chunks()``.
    """
    template = zarr.make_template(source)
    chunks = normalize_chunks(chunks, template, split_vars, previous_chunks)
    ptransform = core.DatasetToChunks(source, chunks, split_vars)
    ptransform.label = _get_label('from_xarray')
    return cls(template, dict(chunks), split_vars, ptransform)

  @classmethod
  def from_zarr(
      cls,
      path: str,
      *,
      chunks: Mapping[str, int | str] | str | None = None,
      split_vars: bool = False,
  ) -> Dataset:
    """Create an xarray_beam.Dataset from a Zarr store.

    Args:
      path: Zarr path to read from.
      chunks: optional mapping from dimension names to chunk sizes, or any value
        that can be passed to :py:func:`xarray_beam.normalize_chunks`. If not
        provided, the chunk sizes will be inferred from the Zarr file.
      split_vars: whether variables are split between separate elements in the
        ptransform, or all stored in the same element.

    Returns:
      New Dataset created from the Zarr store.
    """
    source, previous_chunks = zarr.open_zarr(path)
    if chunks is None:
      chunks = previous_chunks
    result = cls.from_xarray(
        source, chunks, split_vars=split_vars, previous_chunks=previous_chunks
    )
    result.ptransform.label = _get_label('from_zarr')
    return result

  def _check_shards_or_chunks(
      self,
      zarr_chunks: Mapping[str, int],
      chunks_name: Literal['shards', 'chunks'],
  ) -> None:
    if any(self.chunks[k] % zarr_chunks[k] for k in self.chunks):
      raise ValueError(
          f'cannot write a dataset with chunks {self.chunks} to Zarr with '
          f'{chunks_name} {zarr_chunks}, which do not divide evenly into '
          f'{chunks_name}'
      )

  def to_zarr(
      self,
      path: str,
      *,
      zarr_chunks_per_shard: Mapping[str, int] | None = None,
      zarr_chunks: Mapping[str, int] | None = None,
      zarr_shards: Mapping[str, int] | None = None,
      zarr_format: int | None = None,
  ) -> beam.PTransform:
    """Write this dataset to a Zarr file.

    The extensive options for controlling chunking and sharding are intended for
    power users:

    * If you are happy with the existing chunk sizes of your dataset
      and just want to write it to disk, you can omit all of them.
    * Consider specifying only ``zarr_chunks_per_shard`` to allow for more
      flexible efficient reading of data from disk. This allows for dividing
      dataset chunks into much smaller Zarr chunks on disk, with each chunk
      stored in a single Zarr shard.

    Args:
      path: path to write to.
      zarr_chunks_per_shard: If provided, write this dataset into Zarr shards,
        each with at most this many Zarr chunks per shard (requires Zarr v3).
      zarr_chunks: Explicit chunk sizes to use for storing data in Zarr, as an
        alternative to specifying ``zarr_chunks_per_shard``. Zarr chunk sizes
        must evenly divide the existing chunk sizes of this dataset.
      zarr_shards: Explicit shards to use for storing data in Zarr, which must
        evenly divide the existing chunk sizes of this dataset, and be even
        multiples of chunk sizes. Requires Zarr v3. By default, Zarr sharding is
        not used unless ``zarr_chunks_per_shard`` is provided, in which case
        Zarr shards default to the chunk sizes of this dataset.
      zarr_format: optional integer specifying the explicit Zarr format to use.
        Defaults to Zarr v3 if using shards, or the default format for your
        installed version of Zarr.

    Returns:
      Beam PTransform that writes the dataset to a Zarr file.
    """
    if zarr_chunks_per_shard is not None:
      if zarr_chunks is not None:
        raise ValueError(
            'cannot supply both zarr_chunks_per_shard and zarr_chunks'
        )
      if zarr_shards is None:
        zarr_shards = {}
      zarr_shards = {**self.chunks, **zarr_shards}
      zarr_chunks = {}
      for dim, existing_chunk_size in zarr_shards.items():
        multiple = zarr_chunks_per_shard.get(dim)
        if multiple is None:
          raise ValueError(
              f'cannot write a dataset with chunks {self.chunks} to Zarr with '
              f'{zarr_chunks_per_shard=}, which does not contain a value for '
              f'dimension {dim!r}'
          )
        zarr_chunks[dim], remainder = divmod(existing_chunk_size, multiple)
        if remainder != 0:
          raise ValueError(
              f'cannot write a dataset with chunks {self.chunks} to Zarr with '
              f'{zarr_chunks_per_shard=}, which do not evenly divide into '
              'chunks'
          )
    elif zarr_chunks is None:
      if zarr_shards is not None:
        raise ValueError('cannot supply zarr_shards without zarr_chunks')
      zarr_chunks = {}

    zarr_chunks = {**self.chunks, **zarr_chunks}
    if zarr_shards is not None:
      zarr_shards = {**self.chunks, **zarr_shards}
      self._check_shards_or_chunks(zarr_shards, 'shards')
    else:
      self._check_shards_or_chunks(zarr_chunks, 'chunks')

    if zarr_shards is not None and zarr_format is None:
      zarr_format = 3  # required for shards

    return self.ptransform | _get_label('to_zarr') >> zarr.ChunksToZarr(
        path,
        self.template,
        zarr_chunks=zarr_chunks,
        zarr_shards=zarr_shards,
        zarr_format=zarr_format,
    )

  def collect_with_direct_runner(self) -> xarray.Dataset:
    """Collect a dataset in memory by writing it to a temp file."""
    # TODO(shoyer): generalize this function to something that support
    # alternative runners can we figure out a suitable temp file location for
    # distributed runners?

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = os.path.join(temp_dir, 'tmp.zarr')
      with beam.Pipeline(runner='DirectRunner') as pipeline:
        pipeline |= self.to_zarr(temp_path)
      return xarray.open_zarr(temp_path).compute()

  def map_blocks(
      self,
      /,
      func: Callable[[xarray.Dataset], xarray.Dataset],
      *,
      template: xarray.Dataset | None = None,
      chunks: Mapping[str, int] | None = None,
  ) -> Dataset:
    """Map a function over the chunks of this dataset.

    Args:
      func: any function that does not change the size of dataset chunks, called
        like ``func(chunk)``, where ``chunk`` is an xarray.Dataset.
      template: new template for the resulting dataset. If not provided, an
        attempt will be made to infer the template by applying ``func`` to the
        existing template, which requires that ``func`` is implemented using
        dask compatible operations.
      chunks: new chunks sizes for the resulting dataset. If not provided, an
        attempt will be made to infer the new chunks based on the existing
        chunks, dimensions sizes and the new template.

    Returns:
      New Dataset with updated chunks.
    """
    if template is None:
      try:
        template = func(self.template)
      except ValueError as e:
        raise ValueError(
            'failed to lazily apply func() to the existing template. Consider '
            'supplying template explicitly or modifying func() to support lazy '
            'dask arrays.'
        ) from e
    template = zarr.make_template(template)  # ensure template is lazy

    if self.split_vars:
      old_vars = {
          k for k, v in self.template.variables.items() if v.chunks is not None
      }
      new_vars = {
          k for k, v in template.variables.items() if v.chunks is not None
      }
      if old_vars != new_vars:
        raise ValueError(
            'cannot use map_blocks on a dataset with split_vars=True if '
            'the transformation returns a different set of variables.\n'
            f'Old split variables: {old_vars}\n'
            f'New split variables: {new_vars}'
        )

    if chunks is None:
      chunks = _infer_new_chunks(
          old_sizes=self.sizes,
          old_chunks=self.chunks,
          new_sizes=template.sizes,
      )  # pytype: disable=wrong-arg-types

    label = _get_label('map_blocks')
    ptransform = self.ptransform | label >> beam.MapTuple(
        functools.partial(_apply_to_each_chunk, func, self.chunks, chunks)
    )
    return type(self)(template, chunks, self.split_vars, ptransform)

  # rechunking methods

  def rechunk(
      self,
      chunks: dict[str, int | str] | str,
      min_mem: int | None = None,
      max_mem: int = 2**30,
  ) -> Dataset:
    """Rechunk this Dataset.

    Args:
      chunks: new chunk sizes, either a dict mapping from dimension name to
        chunk size, or any value that can be passed to
        :py:func:`xarray_beam.normalize_chunks`.
      min_mem: optional minimum memory usage for an intermediate chunk in
        rechunking. Defaults to ``max_mem/100``.
      max_mem: optional maximum memory usage ffor an intermediate chunk in
        rechunking. Defaults to 1GB.

    Returns:
      New Dataset with updated chunks.
    """
    chunks = normalize_chunks(
        chunks,
        self.template,
        split_vars=self.split_vars,
        previous_chunks=self.chunks,
    )
    label = _get_label('rechunk')

    if isinstance(self.ptransform, core.DatasetToChunks) and all(
        chunks[k] % self.chunks[k] == 0 for k in chunks
    ):
      # Rechunking can be performed by re-reading the source dataset with new
      # chunks, rather than using a separate rechunking transform.
      ptransform = core.DatasetToChunks(
          self.ptransform.dataset, chunks, self.split_vars
      )
      ptransform.label = _concat_labels(self.ptransform.label, label)
    else:
      # Need to do a full rechunking.
      if self.split_vars:
        itemsize = max(v.dtype.itemsize for v in self.template.values())
      else:
        itemsize = sum(v.dtype.itemsize for v in self.template.values())
      rechunk_transform = rechunk.Rechunk(
          self.sizes,
          self.chunks,
          chunks,
          itemsize=itemsize,
          min_mem=min_mem,
          max_mem=max_mem,
      )
      ptransform = self.ptransform | label >> rechunk_transform
    return type(self)(self.template, chunks, self.split_vars, ptransform)

  def split_variables(self) -> Dataset:
    """Split variables in this Dataset into separate chunks."""
    split_vars = True
    label = _get_label('split_vars')
    ptransform = self.ptransform | label >> rechunk.SplitVariables()
    return type(self)(self.template, self.chunks, split_vars, ptransform)

  def consolidate_variables(self) -> Dataset:
    """Consolidate variables in this Dataset into a single chunk."""
    split_vars = False
    label = _get_label('consolidate_vars')
    ptransform = self.ptransform | label >> rechunk.ConsolidateVariables()
    return type(self)(self.template, self.chunks, split_vars, ptransform)

  def mean(
      self,
      dim: str | list[str] | tuple[str, ...] | None = None,
      *,
      skipna: bool | None = None,
      dtype: npt.DTypeLike | None = None,
      fanout: int | None = None,
  ) -> Dataset:
    """Compute the mean of this Dataset using Beam combiners.

    This can be significantly faster than using rechunking, but good
    performance may require tuning ``fanout``.

    Args:
      dim: dimension(s) to compute the mean over.
      skipna: whether to skip missing data when computing the mean.
      dtype: the desired dtype of the resulting Dataset.
      fanout: size of an intermediate fanout stage for Beam combiners.

    Returns:
      New Dataset with the mean computed.
    """
    # TODO(shoyer): use heuristics to pick a default fanout size.
    if dim is None:
      dims = list(self.template.dims)
    elif isinstance(dim, str):
      dims = [dim]
    else:
      dims = dim
    template = zarr.make_template(
        self.template.mean(dim=dims, skipna=skipna, dtype=dtype)
    )
    chunks = {k: v for k, v in self.chunks.items() if k not in dims}
    label = _get_label(f"mean_{'_'.join(dims)}")
    ptransform = self.ptransform | label >> combiners.Mean(
        dim=dims, skipna=skipna, dtype=dtype, fanout=fanout
    )
    return type(self)(template, chunks, self.split_vars, ptransform)

  _head = _whole_dataset_method('head')

  def head(self, **indexers_kwargs: int) -> Dataset:
    """Return a Dataset with the first N elements of each dimension."""
    if not isinstance(self.ptransform, core.DatasetToChunks):
      raise ValueError(
          'head() is only supported on untransformed datasets, with '
          'ptransform=DatasetToChunks. This dataset has '
          f'ptransform={self.ptransform}'
      )
    return self._head(**indexers_kwargs)

  # thin wrappers around xarray methods
  __getitem__ = _whole_dataset_method('__getitem__')
  transpose = _whole_dataset_method('transpose')

  def pipe(self, func, *args, **kwargs):
    """Apply a function to this dataset with method-chaining syntax."""
    return func(self, *args, **kwargs)
