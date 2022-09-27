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
"""IO with Pangeo-Forge."""
import contextlib
import tempfile
from typing import (
    Dict,
    Iterator,
    Optional,
    Mapping,
    Tuple,
)

import apache_beam as beam
import fsspec
import xarray
from apache_beam.io.filesystems import FileSystems

from xarray_beam._src import core, rechunk


def _zero_dimensions(dataset: xarray.Dataset) -> Mapping[str, int]:
  return {dim: 0 for dim in dataset.dims.keys()}


def _expand_dimensions_by_key(
    dataset: xarray.Dataset, index: 'FilePatternIndex', pattern: 'FilePattern'
) -> xarray.Dataset:
  """Expand the dimensions of the `Dataset` by offsets found in the `Key`."""
  combine_dims_by_name = {
      combine_dim.name: combine_dim for combine_dim in pattern.combine_dims
  }
  index_by_name = {idx.name: idx for idx in index}

  if not combine_dims_by_name:
    return dataset

  for dim_key in index_by_name.keys():
    # skip expanding dimensions if they already exist
    if dim_key in dataset.dims:
      continue

    try:
      combine_dim = combine_dims_by_name[dim_key]
    except KeyError:
      raise ValueError(
          f'could not find CombineDim named {dim_key!r} in pattern {pattern!r}.'
      )

    dim_val = combine_dim.keys[index_by_name[dim_key].index]
    dataset = dataset.expand_dims(**{dim_key: [dim_val]})

  return dataset


class FilePatternToChunks(beam.PTransform):
  """Open data described by a Pangeo-Forge `FilePattern` into keyed chunks."""

  from pangeo_forge_recipes.patterns import FilePattern, FilePatternIndex

  def __init__(
      self,
      pattern: 'FilePattern',
      chunks: Optional[Mapping[str, int]] = None,
      local_copy: bool = False,
      xarray_open_kwargs: Optional[Dict] = None,
  ):
    """Initialize FilePatternToChunks.

    TODO(#29): Currently, `MergeDim`s are not supported.

    Args:
      pattern: a `FilePattern` describing a dataset.
      chunks: split each open dataset into smaller chunks. If not set, the
        transform will return one file per chunk.
      local_copy: Open files from the pattern with local copies instead of a
        buffered reader.
      xarray_open_kwargs: keyword arguments to pass to `xarray.open_dataset()`.
    """
    self.pattern = pattern
    self.chunks = chunks
    self.local_copy = local_copy
    self.xarray_open_kwargs = xarray_open_kwargs or {}
    self._max_size_idx = {}

    if pattern.merge_dims:
      raise ValueError('patterns with `MergeDim`s are not supported.')

  @contextlib.contextmanager
  def _open_dataset(self, path: str) -> xarray.Dataset:
    """Open as an XArray Dataset, sometimes with local caching."""
    if self.local_copy:
      with tempfile.TemporaryDirectory() as tmpdir:
        local_file = fsspec.open_local(
            f'simplecache::{path}', simplecache={'cache_storage': tmpdir}
        )
        yield xarray.open_dataset(local_file, **self.xarray_open_kwargs)
    else:
      with FileSystems().open(path) as file:
        yield xarray.open_dataset(file, **self.xarray_open_kwargs)

  def _open_chunks(
      self, index: 'FilePatternIndex', path: str
  ) -> Iterator[Tuple[core.Key, xarray.Dataset]]:
    """Open datasets into chunks with XArray."""
    with self._open_dataset(path) as dataset:
      dataset = _expand_dimensions_by_key(dataset, index, self.pattern)

      if not self._max_size_idx:
        self._max_size_idx = dataset.sizes

      base_key = core.Key(_zero_dimensions(dataset)).with_offsets(
          **{
              dim.name: self._max_size_idx[dim.name] * dim.index
              for dim in index
          }
      )

      num_threads = len(dataset.data_vars)

      # If chunks is not set by the user, treat the dataset as a single chunk.
      if self.chunks is None:
        yield base_key, dataset.compute(num_workers=num_threads)
        return

      for new_key, chunk in rechunk.split_chunks(
          base_key, dataset, self.chunks
      ):
        yield new_key, chunk.compute(num_workers=num_threads)

  def expand(self, pcoll):
    return (
        pcoll
        | beam.Create(list(self.pattern.items()))
        | beam.FlatMapTuple(self._open_chunks)
    )
