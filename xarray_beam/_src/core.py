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
import itertools
import math
from typing import (
    AbstractSet,
    Dict,
    List,
    Iterator,
    Optional,
    Mapping,
    Sequence,
    Tuple,
    Union,
)

import apache_beam as beam
import immutabledict
import numpy as np
import xarray

from xarray_beam._src import threadmap

_DEFAULT = object()


class Key:
  """A key for keeping track of chunks of a distributed xarray.Dataset.

  Key object in Xarray-Beam include two components:

  - "offsets": an immutable dict indicating integer offsets (total number of
    array elements) from the origin along each dimension for this chunk.
  - "vars": either an frozenset or None, indicating the subset of Dataset
    variables included in this chunk. None means that all variables are
    included.

  Key objects are "deterministically encoded" by Beam, which makes them suitable
  for use as keys in Beam pipelines, i.e., with beam.GroupByKey. They are also
  immutable and hashable, which makes them usable as keys in Python
  dictionaries.

  Example usage::

    >>> key = xarray_beam.Key(offsets={'x': 10}, vars={'foo'})

    >>> key
    xarray_beam.Key(offsets={'x': 10}, vars={'foo'})

    >>> key.offsets
    immutabledict({'x': 10})

    >>> key.vars
    frozenset({'foo'})

  To replace some offsets::

    >>> key.with_offsets(y=0)  # insert
    xarray_beam.Key(offsets={'x': 10, 'y': 0}, vars={'foo'})

    >>> key.with_offsets(x=20)  # override
    xarray_beam.Key(offsets={'x': 20}, vars={'foo'})

    >>> key.with_offsets(x=None)  # remove
    xarray_beam.Key(offsets={}, vars={'foo'})

  To entirely replace offsets or variables::

    >>> key.replace(offsets={'y': 0})
    xarray_beam.Key(offsets={'y': 0}, vars={'foo'})

    >>> key.replace(vars=None)
    xarray_beam.Key(offsets={'x': 10}, vars=None)
  """

  # pylint: disable=redefined-builtin

  def __init__(
      self,
      offsets: Optional[Mapping[str, int]] = None,
      vars: Optional[AbstractSet[str]] = None,
  ):
    if offsets is None:
      offsets = {}
    if isinstance(vars, str):
      raise TypeError(f'vars must be a set or None, but is {vars!r}')
    self.offsets = immutabledict.immutabledict(offsets)
    self.vars = None if vars is None else frozenset(vars)

  def replace(
      self,
      offsets: Union[Mapping[str, int], object] = _DEFAULT,
      vars: Union[AbstractSet[str], None, object] = _DEFAULT,
  ) -> 'Key':
    if offsets is _DEFAULT:
      offsets = self.offsets
    if vars is _DEFAULT:
      vars = self.vars
    return type(self)(offsets, vars)

  def with_offsets(self, **offsets: Optional[int]) -> 'Key':
    new_offsets = dict(self.offsets)
    for k, v in offsets.items():
      if v is None:
        del new_offsets[k]
      else:
        new_offsets[k] = v
    return self.replace(offsets=new_offsets)

  def __repr__(self) -> str:
    offsets = dict(self.offsets)
    vars = set(self.vars) if self.vars is not None else None
    return f'{type(self).__name__}(offsets={offsets}, vars={vars})'

  def __hash__(self) -> int:
    return hash((self.offsets, self.vars))

  def __eq__(self, other) -> bool:
    if not isinstance(other, Key):
      return NotImplemented
    return self.offsets == other.offsets and self.vars == other.vars

  def __ne__(self, other) -> bool:
    return not self == other

  # Beam uses these methods (also used for pickling) for "deterministic
  # encoding" of groupby keys
  def __getstate__(self):
    offsets_state = sorted(self.offsets.items())
    vars_state = None if self.vars is None else sorted(self.vars)
    return (offsets_state, vars_state)

  def __setstate__(self, state):
    self.__init__(*state)


def offsets_to_slices(
    offsets: Mapping[str, int],
    sizes: Mapping[str, int],
    base: Optional[Mapping[str, int]] = None,
) -> Dict[str, slice]:
  """Convert offsets into slices with an optional base offset.

  Args:
    offsets: integer offsets from the origin along each axis.
    sizes: dimension sizes for the corresponding chunks.
    base: optional base-offset to subract from this key. This allows for
      relative indexing, e.g., into a chunk of a larger Dataset.

  Returns:
    Slices suitable for indexing with xarray.Dataset.isel().

  Raises:
    ValueError: if an offset is specified for a dimension where there is no
      corresponding size specified.

  Example usage::

    >>> offsets_to_slices({'x': 100}, sizes={'x': 10})
    {'x': slice(100, 110, 1)}
    >>> offsets_to_slices({'x': 100}, sizes={'x': 10}, base={'x': 100})
    {'x': slice(0, 10, 1)}
  """
  if base is None:
    base = {}
  slices = {}
  missing_chunk_sizes = [k for k in offsets.keys() if k not in sizes]
  if missing_chunk_sizes:
    raise ValueError(
        'some dimensions have offsets specified but no dimension sizes: '
        f'offsets={offsets} and sizes={sizes}'
    )
  for k, size in sizes.items():
    offset = offsets.get(k, 0) - base.get(k, 0)
    slices[k] = slice(offset, offset + size, 1)
  return slices


def _chunks_to_offsets(
    chunks: Mapping[str, Sequence[int]],
) -> Dict[str, List[int]]:
  return {
      dim: np.concatenate([[0], np.cumsum(sizes)[:-1]]).tolist()
      for dim, sizes in chunks.items()
  }


def iter_chunk_keys(
    offsets: Mapping[str, Sequence[int]],
    vars: Optional[AbstractSet[str]] = None,  # pylint: disable=redefined-builtin
) -> Iterator[Key]:
  """Iterate over the Key objects corresponding to the given chunks."""
  chunk_indices = [range(len(sizes)) for sizes in offsets.values()]
  for indices in itertools.product(*chunk_indices):
    key_offsets = {
        dim: offsets[dim][index] for dim, index in zip(offsets, indices)
    }
    yield Key(key_offsets, vars)


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
      result[dim] = multiple * (chunks[dim],) + (
          (remainder,) if remainder else ()
      )
  return result


class DatasetToChunks(beam.PTransform):
  """Split an xarray.Dataset into keyed chunks."""

  def __init__(
      self,
      dataset: xarray.Dataset,
      chunks: Optional[Mapping[str, Union[int, Tuple[int, ...]]]] = None,
      split_vars: bool = False,
      num_threads: Optional[int] = None,
      shard_keys_threshold: int = 200_000,
  ):
    """Initialize DatasetToChunks.

    Args:
      dataset: dataset to split into (Key, xarray.Dataset) pairs.
      chunks: optional chunking scheme. Required if the dataset is *not* already
        chunked. If the dataset *is* already chunked with Dask, `chunks` takes
        precedence over the existing chunks.
      split_vars: whether to split the dataset into separate records for each
        data variable or to keep all data variables together. This is
        recommended if you don't need perform joint operations on different
        dataset variables and individual variable chunks are sufficiently large.
      num_threads: optional number of Dataset chunks to load in parallel per
        worker. More threads can increase throughput, but also increases memory
        usage and makes it harder for Beam runners to shard work. Note that each
        variable in a Dataset is already loaded in parallel, so this is most
        useful for Datasets with a small number of variables or when using
        split_vars=True.
      shard_keys_threshold: threshold at which to compute keys on Beam workers,
        rather than only on the host process. This is important for scaling
        pipelines to millions of tasks.
    """
    if chunks is None:
      chunks = dataset.chunks
    if chunks is None:
      raise ValueError('dataset must be chunked or chunks must be provided')
    expanded_chunks = normalize_expanded_chunks(chunks, dataset.sizes)
    self.dataset = dataset
    self.expanded_chunks = expanded_chunks
    self.split_vars = split_vars
    self.num_threads = num_threads
    self.shard_keys_threshold = shard_keys_threshold
    # TODO(shoyer): consider recalculating these potentially large properties on
    # each worker, rather than only once on the host.
    self.offsets = _chunks_to_offsets(expanded_chunks)
    self.offset_index = compute_offset_index(self.offsets)
    # We use the simple heuristic of only sharding inputs along the dimension
    # with the most chunks.
    lengths = {k: len(v) for k, v in self.offsets.items()}
    self.sharded_dim = max(lengths, key=lengths.get) if lengths else None
    self.shard_count = self._shard_count()

  def _task_count(self) -> int:
    """Count the number of tasks emitted by this transform."""
    counts = {k: len(v) for k, v in self.expanded_chunks.items()}
    if not self.split_vars:
      return int(np.prod(list(counts.values())))
    total = 0
    for variable in self.dataset.values():
      count_list = [v for k, v in counts.items() if k in variable.dims]
      total += int(np.prod(count_list))
    return total

  def _shard_count(self) -> Optional[int]:
    """Determine the number of times to shard input keys."""
    task_count = self._task_count()
    if task_count <= self.shard_keys_threshold:
      return None  # no sharding

    if not self.split_vars:
      return math.ceil(task_count / self.shard_keys_threshold)

    var_count = sum(
        self.sharded_dim in var.dims for var in self.dataset.values()
    )
    return math.ceil(task_count / (var_count * self.shard_keys_threshold))

  def _iter_all_keys(self) -> Iterator[Key]:
    """Iterate over all Key objects."""
    if not self.split_vars:
      yield from iter_chunk_keys(self.offsets)
    else:
      for name, variable in self.dataset.items():
        relevant_offsets = {
            k: v for k, v in self.offsets.items() if k in variable.dims
        }
        yield from iter_chunk_keys(relevant_offsets, vars={name})

  def _iter_shard_keys(
      self, shard_id: Optional[int], var_name: Optional[str]
  ) -> Iterator[Key]:
    """Iterate over Key objects for a specific shard and variable."""
    if var_name is None:
      offsets = self.offsets
    else:
      offsets = {dim: self.offsets[dim] for dim in self.dataset[var_name].dims}

    if shard_id is None:
      assert self.split_vars
      yield from iter_chunk_keys(offsets, vars={var_name})
    else:
      assert self.split_vars == (var_name is not None)
      dim = self.sharded_dim
      count = math.ceil(len(self.offsets[dim]) / self.shard_count)
      dim_slice = slice(shard_id * count, (shard_id + 1) * count)
      offsets = {**offsets, dim: offsets[dim][dim_slice]}
      vars_ = {var_name} if self.split_vars else None
      yield from iter_chunk_keys(offsets, vars=vars_)

  def _shard_inputs(self) -> List[Tuple[Optional[int], Optional[str]]]:
    """Create inputs for sharded key iterators."""
    if not self.split_vars:
      return [(i, None) for i in range(self.shard_count)]

    inputs = []
    for name, variable in self.dataset.items():
      if self.sharded_dim in variable.dims:
        inputs.extend([(i, name) for i in range(self.shard_count)])
      else:
        inputs.append((None, name))
    return inputs

  def _key_to_chunks(self, key: Key) -> Iterator[Tuple[Key, xarray.Dataset]]:
    """Convert a Key into an in-memory (Key, xarray.Dataset) pair."""
    sizes = {
        dim: self.expanded_chunks[dim][self.offset_index[dim][offset]]
        for dim, offset in key.offsets.items()
    }
    slices = offsets_to_slices(key.offsets, sizes)
    dataset = self.dataset if key.vars is None else self.dataset[list(key.vars)]
    chunk = dataset.isel(slices)
    # Load the data, using a separate thread for each variable
    num_threads = len(self.dataset)
    result = chunk.chunk().compute(num_workers=num_threads)
    yield key, result

  def expand(self, pcoll):
    if self.shard_count is None:
      # Create all keys on the machine launching the Beam pipeline. This is
      # faster if the number of keys is small.
      key_pcoll = pcoll | beam.Create(self._iter_all_keys())
    else:
      # Create keys in separate shards on Beam workers. This is more scalable.
      key_pcoll = (
          pcoll
          | beam.Create(self._shard_inputs())
          | 'GenerateKeys' >> beam.FlatMapTuple(self._iter_shard_keys)
          | beam.Reshuffle()
      )

    return key_pcoll | 'KeyToChunks' >> threadmap.FlatThreadMap(
        self._key_to_chunks, num_threads=self.num_threads
    )


def validate_chunk(key: Key, dataset: xarray.Dataset) -> None:
  """Verify that keys correpond to Dataset properties."""
  missing_keys = [repr(k) for k in key.offsets.keys() if k not in dataset.dims]
  if missing_keys:
    raise ValueError(
        f"Key offset(s) {', '.join(missing_keys)} in {key} not found in Dataset"
        f' dimensions: {dataset!r}'
    )

  if key.vars is None:
    return
  missing_vars = [repr(v) for v in key.vars if v not in dataset.data_vars]
  if missing_vars:
    raise ValueError(
        f"Key var(s) {', '.join(missing_vars)} in {key} not found in Dataset"
        f' data variables: {dataset!r}'
    )


class ValidateEachChunk(beam.PTransform):
  """Check that keys match the dataset for each key, dataset tuple."""

  def _validate(self, key, dataset):
    # Other checks may come later...
    validate_chunk(key, dataset)
    return key, dataset

  def expand(self, pcoll):
    return pcoll | beam.MapTuple(self._validate)
