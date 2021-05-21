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
"""Core data model for xarray-beam."""
import functools
import itertools
from typing import (
    AbstractSet, Dict, List, Iterator, Optional, Mapping, Sequence, Tuple,
    Union,
)
import apache_beam as beam
import numpy as np
import xarray

from xarray_beam._src import threadmap


def _default_base(
    base: Optional[Mapping[str, int]],
    keys: Sequence[str],
) -> Dict[str, int]:
  base = {} if base is None else dict(base)
  for dim in keys:
    base.setdefault(dim, 0)
  return base


@functools.total_ordering
class ChunkKey(Mapping[str, int]):
  """An immutable mapping of dimension names to chunk offsets.

  Values indicate integer offset from the origin for dealing with an individual
  "chunk" of a larger xarray.Dataset.

  ChunkKey is hashable, which makes it suitable for use as a key in Beam
  pipelines. It also has a handful of convenience methods and operators defined
  that are useful for building pipelines.

  Example usage::

    >>> key = ChunkKey({"x": 0, "y": 10})

    >>> key["x"]  # ChunkKey works *like* a dict
    0

    >>> {key: 'foo'}  # but also can be used as a key *in* a dict
    {ChunkKey({'x': 0, 'y': 10}): 'foo'}

    # convert to slices for xarray.Dataset.isel()
    >>> key.to_slices({'x': 5, 'y': 10})
    {'x': slice(0, 5, 1), 'y': slice(10, 20, 1)}

    >>> key | {'z': 100}  # insert or override offsets
    ChunkKey({'x': 0, 'y': 10, 'z': 100})

    >>> key - {'x'}  # remove dimensions
    ChunkKey({'y': 10})
  """

  def __init__(self, offsets: Mapping[str, int]):
    self._offsets = offsets

  def to_slices(
      self,
      sizes: Mapping[str, int],
      base: Optional[Mapping[str, int]] = None,
  ) -> Dict[str, slice]:
    """Convert this ChunkKey into slices with an optional base offset.

    Args:
      sizes: dimension sizes for the corresponding chunk.
      base: optional base-offset to subract from this key. This allows for
        relative indexing, e.g., into a chunk of a larger Dataset.

    Returns:
      Slices suitable for indexing with xarray.Dataset.isel().

    Example usage::

      >>> key = ChunkKey({'x': 100})
      >>> key.to_slices({'x': 10})
      {'x': slice(100, 110, 1)}
      >>> key.to_slices({'x': 10}, base={'x': 100})
      {'x': slice(0, 10, 1)}
    """
    base = _default_base(base, keys=self._offsets)
    slices = {}
    for k, v in self._offsets.items():
      offset = v - base[k]
      size = sizes.get(k)
      if size is not None:
        slices[k] = slice(offset, offset + size, 1)
      else:
        if offset != 0:
          raise ValueError(
              f'dimension {k} has a non-zero offset {offset} but does not '
              f'appear in the dict of known sizes: {sizes}'
          )
        slices[k] = slice(None)
    return slices

  def __or__(self, new_offsets: Mapping[str, int]) -> 'ChunkKey':
    return type(self)({**self._offsets, **new_offsets})

  def __sub__(self, keys: AbstractSet[str]) -> 'ChunkKey':
    if isinstance(keys, str):
      # catch the common error of subtracting a string at runtime
      return NotImplemented
    extra_keys = [k for k in keys if k not in self._offsets]
    if extra_keys:
      raise ValueError(f'Keys {extra_keys} not found in {self}')
    return type(self)(
        {k: v for k, v in self._offsets.items() if k not in keys}
    )

  def __repr__(self) -> str:
    return f'{type(self).__name__}({self._offsets})'

  def __getitem__(self, key: str) -> int:
    return self._offsets[key]

  def __iter__(self) -> Iterator[str]:
    return iter(self._offsets)

  def __len__(self) -> int:
    return len(self._offsets)

  def __hash__(self) -> int:
    return hash(frozenset(self.items()))

  def __lt__(self, other) -> bool:
    if not isinstance(other, ChunkKey):
      return NotImplemented
    if other.keys() != self.keys():
      raise ValueError('Dimensions must match for comparison between ChunkKey '
                       f'objects: {self} vs {other}')
    return sorted(self.items()) < sorted(other.items())

  # Beam uses these methods (also used for pickling) for "deterministic
  # encoding" of groupby keys
  def __getstate__(self):
    return sorted(self.items())

  def __setstate__(self, state):
    self._offsets = dict(state)


def _chunks_to_offsets(
    chunks: Mapping[str, Sequence[int]],
) -> Dict[str, List[int]]:
  return {
      dim: np.concatenate([[0], np.cumsum(sizes)[:-1]]).tolist()
      for dim, sizes in chunks.items()
  }


def iter_chunk_keys(
    chunks: Mapping[str, Tuple[int, ...]],
    base: Optional[Mapping[str, int]] = None
) -> Iterator[ChunkKey]:
  """Iterate over the ChunkKey objects corresponding to the given chunks."""
  base = _default_base(base, keys=chunks)
  relative_offsets = _chunks_to_offsets(chunks)
  chunk_indices = [range(len(sizes)) for sizes in chunks.values()]
  for indices in itertools.product(*chunk_indices):
    bounds = dict(base)
    for dim, index in zip(chunks, indices):
      bounds[dim] += relative_offsets[dim][index]
    yield ChunkKey(bounds)


def compute_offset_index(
    offsets: Mapping[str, Sequence[int]],
) -> Dict[str, Dict[int, int]]:
  """Compute a mapping from chunk offsets to chunk indices."""
  index = {}
  for dim, dim_offsets in offsets.items():
    index[dim] = {}
    for i, offset in enumerate(dim_offsets):
      index[dim][offset] = i
  return index


def normalize_expanded_chunks(
    chunks: Mapping[str, Union[int, Tuple[int, ...]]],
    dim_sizes: Mapping[str, int],
) -> Dict[str, Tuple[int, ...]]:
  # pylint: disable=g-doc-args
  # pylint: disable=g-doc-return-or-yield
  """Normalize a dict of chunks to give the expanded size of each block.

  Example usage::

    >>> normalize_expanded_chunks({'x': 3}, {'x': 10})
    {'x': (3, 3, 3, 1)}
  """
  result = {}
  for dim, dim_size in dim_sizes.items():
    if dim not in chunks or chunks[dim] == -1:
      result[dim] = (dim_size,)
    elif isinstance(chunks[dim], tuple):
      total = sum(chunks[dim])
      if total != dim_size:
        raise ValueError(
            f'sum of provided chunks does not match size of dimension {dim}: '
            f'{total} vs {dim_size}'
        )
      result[dim] = chunks[dim]
    else:
      multiple, remainder = divmod(dim_size, chunks[dim])
      result[dim] = (
          multiple * (chunks[dim],) + ((remainder,) if remainder else ())
      )
  return result


class DatasetToChunks(beam.PTransform):
  """Split an xarray.Dataset into keyed chunks."""

  def __init__(
      self,
      dataset: xarray.Dataset,
      chunks: Optional[Mapping[str, Union[int, Tuple[int, ...]]]] = None,
      num_threads: Optional[int] = None,
  ):
    """Initialize ChunksToZarr.

    Args:
      dataset: dataset to split into (ChunkKey, xarray.Dataset) pairs.
      chunks: optional chunking scheme. Required if the dataset is *not*
        already chunked. If the dataset *is* already chunked with Dask, `chunks`
        takes precedence over the existing chunks.
      num_threads: optional number of Dataset chunks to load in parallel per
        worker. More threads can increase throughput, but also increases memory
        usage and makes it harder for Beam runners to shard work. Note that each
        variable in a Dataset is already loaded in parallel, so this is most
        useful for Datasets with a small number of variables.
    """
    if chunks is None:
      chunks = dataset.chunks
    if chunks is None:
      raise ValueError('dataset must be chunked or chunks must be set')
    chunks = normalize_expanded_chunks(chunks, dataset.sizes)
    self.dataset = dataset
    self.chunks = chunks
    self.num_threads = num_threads
    self.offset_index = compute_offset_index(_chunks_to_offsets(chunks))

  def _key_to_chunk(self, key):
    sizes = {
        dim: self.chunks[dim][self.offset_index[dim][offset]]
        for dim, offset in key.items()
    }
    slices = key.to_slices(sizes)
    chunk = self.dataset.isel(slices)
    # Load the data, using a separate thread for each variable
    num_threads = len(self.dataset.data_vars)
    result = chunk.chunk().compute(num_workers=num_threads)
    return key, result

  def expand(self, pcoll):
    return (
        pcoll
        | beam.Create(iter_chunk_keys(self.chunks))
        | threadmap.ThreadMap(
            self._key_to_chunk, num_threads=self.num_threads
        )
    )
