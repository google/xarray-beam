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
import shutil
import xarray
from apache_beam.io.filesystems import FileSystems

from xarray_beam._src import core, rechunk


def _zero_dimensions(dataset: xarray.Dataset) -> Mapping[str, int]:
  return {dim: 0 for dim in dataset.dims.keys()}


def _expand_dimensions_by_key(
    dataset: xarray.Dataset,
    index: 'FilePatternIndex',
    pattern: 'FilePattern'
) -> xarray.Dataset:
  """Expand the dimensions of the `Dataset` by offsets found in the `Key`."""
  combine_dims_by_name = {
    combine_dim.name: combine_dim for combine_dim in pattern.combine_dims
  }
  index_by_name = {
    idx.name: idx for idx in index
  }

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
        f"could not find CombineDim named {dim_key!r} in pattern {pattern!r}."
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
      split_vars: bool = False,
      num_threads: Optional[int] = None,
      xarray_open_kwargs: Optional[Dict] = None
  ):
    """Initialize FilePatternToChunks.

    Args:
      pattern: a `FilePattern` describing a dataset.
      chunks: split each open dataset into smaller chunks. If not set, the
        transform will return one file per chunk.
      local_copy: Open files from the pattern with local copies instead of a
        buffered reader.
      split_vars: whether to split the dataset into separate records for each
        data variables or to keep all data variables together. If the pattern
        has merge dimensions (and this flag is false), data will be split
        according to the pattern.
      num_threads: optional number of Dataset chunks to load in parallel per
        worker. More threads can increase throughput, but also increases memory
        usage and makes it harder for Beam runners to shard work. Note that each
        variable in a Dataset is already loaded in parallel, so this is most
        useful for Datasets with a small number of variables.
      xarray_open_kwargs: keyword arguments to pass to `xarray.open_dataset()`.
    """
    self.pattern = pattern
    self.chunks = chunks
    self.local_copy = local_copy
    self.split_vars = split_vars
    self.num_threads = num_threads
    self.xarray_open_kwargs = xarray_open_kwargs or {}

    # cache values so they don't have to be re-computed.
    self._max_sizes = {}
    self._concat_dims = pattern.concat_dims
    self._merge_dims = pattern.merge_dims
    self._dim_keys_by_name = {
      dim.name: dim.keys for dim in pattern.combine_dims
    }

  def _maybe_split_vars(
      self,
      key: core.Key,
      dataset: xarray.Dataset
  ) -> Iterator[Tuple[core.Key, xarray.Dataset]]:
    """If 'split_vars' is enabled, produce a chunk for every variable."""
    if not self.split_vars:
      yield key, dataset
      return

    for k in dataset:
      yield key.replace(vars={k}), dataset[[k]]

  @contextlib.contextmanager
  def _open_dataset(self, path: str) -> xarray.Dataset:
    """Open as an XArray Dataset, optionally with local caching."""
    with FileSystems().open(path) as file:
      if self.local_copy:
        with tempfile.NamedTemporaryFile('wb') as local_file:
          shutil.copyfileobj(file, local_file)
          local_file.flush()
          yield xarray.open_dataset(local_file.name, **self.xarray_open_kwargs)
          return

      yield xarray.open_dataset(file, **self.xarray_open_kwargs)

  def _open_chunks(
      self,
      index: 'FilePatternIndex',
      path: str
  ) -> Iterator[Tuple[core.Key, xarray.Dataset]]:
    """Open datasets into chunks with XArray."""
    with self._open_dataset(path) as dataset:

      # We only want to expand the concat dimensions of the dataset.
      dataset = _expand_dimensions_by_key(
        dataset,
        tuple(dim for dim in index if dim.name in self._concat_dims),
        self.pattern
      )

      if not self._max_sizes:
        self._max_sizes = dataset.sizes

      variables = {self._dim_keys_by_name[dim.name][dim.index]
                   for dim in index if dim.name in self._merge_dims}
      if not variables:
        variables = None

      key = core.Key(_zero_dimensions(dataset), variables).with_offsets(
        **{dim.name: self._max_sizes[dim.name] * dim.index
           for dim in index if dim.name in self._concat_dims}
      )

      num_threads = self.num_threads or len(dataset.data_vars)

      # If 'chunks' is not set by the user, treat the dataset as a single chunk.
      if self.chunks is None:
        yield from self._maybe_split_vars(
          key, dataset.compute(num_workers=num_threads)
        )
        return

      for new_key, chunk in rechunk.split_chunks(key, dataset, self.chunks):
        yield from self._maybe_split_vars(
          new_key, chunk.compute(num_workers=num_threads)
        )

  def expand(self, pcoll):
    return (
        pcoll
        | beam.Create(list(self.pattern.items()))
        | beam.FlatMapTuple(self._open_chunks)
    )
