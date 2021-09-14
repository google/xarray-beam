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
from typing import (
    Dict,
    Iterator,
    Optional,
    Mapping,
    Tuple,
    cast,
)

import apache_beam as beam
import xarray
from apache_beam.io.filesystems import FileSystems

from xarray_beam._src import core, rechunk


def _zero_dimensions(dataset: xarray.Dataset) -> Mapping[str, int]:
  return {dim: 0 for dim in dataset.dims.keys()}


def _expand_dimensions_by_key(
    dataset: xarray.Dataset,
    key: core.Key,
    index: Tuple[int, ...],
    pattern: 'FilePattern'
) -> xarray.Dataset:
  """Expand the dimensions of the `Dataset` by offsets found in the `Key`."""
  combine_dims_by_name = {
    combine_dim.name: (i, combine_dim)
    for i, combine_dim in enumerate(pattern.combine_dims)
  }

  if not combine_dims_by_name:
    return dataset

  for dim_key in key.offsets.keys():
    # skip expanding dimensions if they already exist
    if dim_key in dataset.dims:
      continue

    dim_idx, combine_dim = combine_dims_by_name.get(dim_key, (-1, None))
    if dim_idx == -1:
      raise ValueError(
        f"could not find CombineDim named {dim_key!r} in pattern {pattern!r}."
      )

    dim_val = combine_dim.keys[index[dim_idx]]
    dataset = dataset.expand_dims(**{dim_key: [dim_val]})

  return dataset


class FilePatternToChunks(beam.PTransform):
  """Open data described by a Pangeo-Forge `FilePattern` into keyed chunks."""

  from pangeo_forge_recipes.patterns import FilePattern

  def __init__(
      self,
      pattern: 'FilePattern',
      sub_chunks: Optional[Mapping[str, int]] = None,
      xarray_open_kwargs: Optional[Dict] = None
  ):
    """Initialize FilePatternToChunks.

    TODO(#29): Currently, `MergeDim`s are not supported.

    Args:
      pattern: a `FilePattern` describing a dataset.
      sub_chunks: split each open dataset into smaller chunks. If not set, each
        chunk will open the full dataset.
      xarray_open_kwargs: keyword arguments to pass to `xarray.open_dataset()`.
    """
    self.pattern = pattern
    self.sub_chunks = sub_chunks or -1
    self.xarray_open_kwargs = xarray_open_kwargs or {}

    if pattern.merge_dims:
      raise ValueError("patters with `MergeDim`s are not supported.")

  def _prechunk(self) -> Iterator[Tuple[core.Key, Tuple[int, ...], str]]:
    """Converts `FilePattern` items into keyed indexes."""
    # Default to 1 chunk per item, even though there may be more on reading.
    # We discover the actual items-per-input in the _open_chunks() phase.
    initial_chunks = {k: v or 1
                      for k, v in self.pattern.nitems_per_input.items()}
    dim_sizes = {
      k: v or self.pattern.dims[k]
      for k, v, in self.pattern.concat_sequence_lens.items()
    }
    chunks = core.normalize_expanded_chunks(initial_chunks, dim_sizes)
    for key, (index, path) in zip(core.iter_chunk_keys(chunks),
                                  self.pattern.items()):
      yield key, index, path

  def _open_chunks(self, _) -> Iterator[Tuple[core.Key, xarray.Dataset]]:
    """Open datasets into chunks with XArray."""
    for index, path in self.pattern.items():
      with FileSystems().open(path) as file:
        key = core.Key(index)
        base_key = core.Key(_zero_dimensions(dataset)).with_offsets(
          **key.offsets
        )

        dataset = xarray.open_dataset(
          file, chunks=self.sub_chunks, **self.xarray_open_kwargs
        )
        dataset = _expand_dimensions_by_key(dataset, key, index, self.pattern)


        num_threads = len(dataset.data_vars)

        if self.sub_chunks == -1:
          yield base_key, dataset.compute(num_workers=num_threads)
          return

        for new_key, chunk in rechunk.split_chunks(base_key, dataset,
                                                   self.sub_chunks):
          yield new_key, chunk.compute(num_workers=num_threads)

  def expand(self, pcoll):
    return (
        pcoll
        | beam.Create([None])
        | beam.FlatMap(self._open_chunks)
    )

