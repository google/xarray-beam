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
from typing import Any, Callable, Literal

import apache_beam as beam
import xarray
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
      ptransform = label >> core.DatasetToChunks(
          dataset, chunks, self.split_vars
      )
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

  template: xarray.Dataset
  chunks: dict[str, int]
  split_vars: bool
  ptransform: beam.PTransform

  def __post_init__(self):
    self.chunks = rechunk.normalize_chunks(self.chunks, self.sizes)

  @property
  def bytes_per_chunk(self) -> int:
    """Estimate of the number of bytes per chunk."""
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
            math.ceil(self.sizes[d] / self.chunks[d])
            for d in variable.dims
        )
      return total
    else:
      return math.prod(
          math.ceil(self.sizes[d] / self.chunks[d])
          for d in self.sizes
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
        f'<xarray_beam.Dataset>\n'
        f'PTransform: {self.ptransform}\n'
        f'Chunks:     {chunk_size} ({chunks_str})\n'
        f'Template:   {total_size} ({chunk_count} chunk{plural})\n'
        + textwrap.indent('\n'.join(base.split('\n')[1:]), ' ' * 4)
    )

  @classmethod
  def from_xarray(
      cls,
      source: xarray.Dataset,
      chunks: Mapping[str, int],
      split_vars: bool = False,
  ) -> Dataset:
    """Create an xarray_beam.Dataset from an xarray.Dataset."""
    template = zarr.make_template(source)
    ptransform = core.DatasetToChunks(source, chunks, split_vars)
    ptransform.label = _get_label('from_xarray')
    return cls(template, dict(chunks), split_vars, ptransform)

  @property
  def sizes(self) -> Mapping[str, int]:
    """Size of each dimension on this dataset."""
    return self.template.sizes  # pytype: disable=bad-return-type

  @classmethod
  def from_zarr(cls, path: str, split_vars: bool = False) -> Dataset:
    """Create an xarray_beam.Dataset from a zarr file."""
    source, chunks = zarr.open_zarr(path)
    result = cls.from_xarray(source, chunks, split_vars)
    result.ptransform = _get_label('from_zarr') >> result.ptransform
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
      zarr_chunks: Mapping[str, int] | None = None,
      zarr_shards: Mapping[str, int] | None = None,
      zarr_format: int | None = None,
  ) -> beam.PTransform:
    """Write to a Zarr file."""
    if zarr_chunks is None:
      if zarr_shards is not None:
        raise ValueError('cannot supply zarr_shards without zarr_chunks')
      zarr_chunks = {}

    zarr_chunks = {**self.chunks, **zarr_chunks}
    if zarr_shards is not None:
      zarr_shards = {**self.chunks, **zarr_shards}
      self._check_shards_or_chunks(zarr_shards, 'shards')
    else:
      self._check_shards_or_chunks(zarr_chunks, 'chunks')

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
      func,
      *,
      kwargs: dict[str, Any] | None = None,
      template: xarray.Dataset | None = None,
      chunks: Mapping[str, int] | None = None,
  ) -> Dataset:
    """Map a function over the chunks of this dataset.

    Args:
      func: any function that does not change the size of dataset chunks, called
        like `func(chunk, **kwargs)`, where `chunk` is an xarray.Dataset.
      kwargs: passed on to func, unmodified.
      template: new template for the resulting dataset. If not provided, an
        attempt will be made to infer the template by applying `func` to the
        existing template, which requires that `func` is implemented using dask
        compatible operations.
      chunks: new chunks sizes for the resulting dataset. If not provided, an
        attempt will be made to infer the new chunks based on the existing
        chunks, dimensions sizes and the new template.

    Returns:
      New Dataset with updated chunks.
    """
    if kwargs is not None:
      func = functools.partial(func, **kwargs)

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
      chunks: dict[str, int],
      min_mem: int | None = None,
      max_mem: int = 2**30,
  ) -> Dataset:
    """Rechunk this Dataset.

    Args:
      chunks: new chunk sizes, as a dict mapping from dimension name to chunk
        size. -1 is interpreted as a "full chunk".
      min_mem: optional minimum memory usage for rechunking.
      max_mem: optional maximum memory usage for rechunking.

    Returns:
      New Dataset with updated chunks.
    """
    # TODO(shoyer): support human readable strings for chunksizes like dask,
    # e.g., chunks={"time": "10 MB"}.
    chunks = rechunk.normalize_chunks(chunks, self.sizes)  # pytype: disable=wrong-arg-types
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
    label = _get_label('rechunk')
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

  # TODO(shoyer): implement merge, rename, mean, etc

  # thin wrappers around xarray methods
  __getitem__ = _whole_dataset_method('__getitem__')
  transpose = _whole_dataset_method('transpose')

  def pipe(self, func, *args, **kwargs):
    """Apply a function to this dataset, like xarray.Dataset.pipe()."""
    return func(self, *args, **kwargs)
