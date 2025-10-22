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
from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence, Set
import contextlib
from functools import cached_property
import itertools
import math
import time
from typing import Any, Generic, TypeVar

import apache_beam as beam
import immutabledict
import numpy as np
import xarray
from xarray_beam._src import threadmap


T = TypeVar('T')


def export(obj: T) -> T:
  obj.__module__ = 'xarray_beam'
  return obj


def inc_counter(namespace: str | type[Any], name: str, value: int = 1):
  """Increments a Beam counter."""
  return beam.metrics.Metrics.counter(namespace, name).inc(value)


@contextlib.contextmanager
def inc_timer_msec(namespace: str | type[Any], name: str) -> Iterator[None]:
  """Records elapsed time in milliseconds in a Beam counter."""
  start = time.perf_counter()
  yield
  elapsed = time.perf_counter() - start
  inc_counter(namespace, name, round(elapsed * 1000))


_DEFAULT = object()


@export
class Key:
  """Key for keeping track of chunks of a distributed Dataset.

  Key objects in Xarray-Beam include two components:

  - `offsets`: an immutable dict indicating integer offsets (total number of
    array elements) from the origin along each dimension for this chunk.
  - `vars`: either an frozenset or None, indicating the subset of Dataset
    variables included in this chunk. The default value of None means that all
    variables are included.

  Alternatively, `indices` may be specified instead of `offsets`. This is a
  newer data model that is not yet fully supported:

  - `indices`: an immutable dict indicating integer chunk indices from the
    origin along each dimension for this chunk.

  `offsets` and `indices` are mutually exclusive: only one of them may be used
  for any given `Key`. For example, if there are chunks of size 100 along the
  'x' dimension, then ``offsets={'x': 400}`` would correspond to
  ``indices={'x': 4}``.

  Key objects are "deterministically encoded" by Beam, which makes them suitable
  for use as keys in Beam pipelines, i.e., with beam.GroupByKey. They are also
  immutable and hashable, which makes them usable as keys in Python
  dictionaries.

  Example usage::

    >>> key = xarray_beam.Key(offsets={'x': 10}, vars={'foo'})

    >>> key
    Key(offsets={'x': 10}, vars={'foo'})

    >>> key.offsets
    immutabledict({'x': 10})

    >>> key.vars
    frozenset({'foo'})

  To replace some offsets::

    >>> key.with_offsets(y=0)  # insert
    Key(offsets={'x': 10, 'y': 0}, vars={'foo'})

    >>> key.with_offsets(x=20)  # override
    Key(offsets={'x': 20}, vars={'foo'})

    >>> key.with_offsets(x=None)  # remove
    Key(offsets={}, vars={'foo'})

  To entirely replace offsets or variables::

    >>> key.replace(offsets={'y': 0})
    Key(offsets={'y': 0}, vars={'foo'})

    >>> key.replace(vars=None)
    Key(offsets={'x': 10})

  You can use `indices` instead of `offsets` to refer to chunks by index::

    >>> key = xarray_beam.Key(indices={'x': 4}, vars={'bar'})
    >>> key
    Key(indices={'x': 4}, vars={'bar'})
    >>> key.with_indices(x=5)
    Key(indices={'x': 5}, vars={'bar'})

  """

  # pylint: disable=redefined-builtin

  def __init__(
      self,
      offsets: Mapping[str, int] | None = None,
      vars: Set[str] | None = None,
      indices: Mapping[str, int] | None = None,
  ):
    if offsets and indices:
      raise ValueError("offsets and indices are mutually exclusive")
    if offsets is None:
      offsets = {}
    if indices is None:
      indices = {}
    if isinstance(vars, str):
      raise TypeError(f"vars must be a set or None, but is {vars!r}")
    self.offsets = immutabledict.immutabledict(offsets)
    self.indices = immutabledict.immutabledict(indices)
    self.vars = None if vars is None else frozenset(vars)

  def replace(
      self,
      offsets: Mapping[str, int] | object = _DEFAULT,
      vars: Set[str] | None | object = _DEFAULT,
      indices: Mapping[str, int] | object = _DEFAULT,
  ) -> Key:
    """Replace one or more components of this Key with new values."""
    if offsets is _DEFAULT:
      offsets = self.offsets
    if vars is _DEFAULT:
      vars = self.vars
    if indices is _DEFAULT:
      indices = self.indices
    return type(self)(offsets, vars, indices)

  def with_offsets(self, **offsets: int | None) -> Key:
    """Replace some offsets with new values.

    Args:
      **offsets: offsets to override (for integer values) or remove, with values
        of ``None``.

    Returns:
      New Key with the specified offsets.
    """
    if self.indices:
      raise ValueError("cannot call with_offsets on a Key with indices")
    new_offsets = dict(self.offsets)
    for k, v in offsets.items():
      if v is None:
        del new_offsets[k]
      else:
        new_offsets[k] = v
    return self.replace(offsets=new_offsets)

  def with_indices(self, **indices: int | None) -> Key:
    """Replace some indices with new values.

    Args:
      **indices: indices to override (for integer values) or remove, with
        values of ``None``.

    Returns:
      New Key with the specified indices.
    """
    if self.offsets:
      raise ValueError("cannot call with_indices on a Key with offsets")
    new_indices = dict(self.indices)
    for k, v in indices.items():
      if v is None:
        del new_indices[k]
      else:
        new_indices[k] = v
    return self.replace(indices=new_indices)

  def __repr__(self) -> str:
    components = []
    if self.offsets:
      components.append(f"offsets={dict(self.offsets)}")
    if self.indices:
      components.append(f"indices={dict(self.indices)}")
    if self.vars is not None:
      components.append(f"vars={set(self.vars)}")
    return f"{type(self).__name__}({', '.join(components)})"

  def __hash__(self) -> int:
    return hash((self.offsets, self.vars, self.indices))

  def __eq__(self, other) -> bool:
    if not isinstance(other, Key):
      return NotImplemented
    return (
        self.offsets == other.offsets
        and self.indices == other.indices
        and self.vars == other.vars
    )

  def __ne__(self, other) -> bool:
    return not self == other

  # Beam uses these methods (also used for pickling) for "deterministic
  # encoding" of groupby keys
  def __getstate__(self):
    offsets_state = sorted(self.offsets.items()) if self.offsets else None
    vars_state = None if self.vars is None else sorted(self.vars)
    indices_state = sorted(self.indices.items()) if self.indices else None
    return offsets_state, vars_state, indices_state

  def __setstate__(self, state):
    self.__init__(*state)


K = TypeVar("K")


@export
def offsets_to_slices(
    offsets: Mapping[K, int],
    sizes: Mapping[K, int],
    base: Mapping[K, int] | None = None,
) -> dict[K, slice]:
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
        "some dimensions have offsets specified but no dimension sizes: "
        f"offsets={offsets} and sizes={sizes}"
    )
  for k, size in sizes.items():
    offset = offsets.get(k, 0) - base.get(k, 0)
    slices[k] = slice(offset, offset + size, 1)
  return slices


def _chunks_to_offsets(
    chunks: Mapping[str, Sequence[int]],
) -> dict[str, list[int]]:
  return {
      dim: np.concatenate([[0], np.cumsum(sizes)[:-1]]).tolist()
      for dim, sizes in chunks.items()
  }


def iter_chunk_keys(
    offsets: Mapping[str, Sequence[int]],
    vars: Set[str] | None = None,  # pylint: disable=redefined-builtin
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
) -> dict[str, dict[int, int]]:
  """Compute a mapping from chunk offsets to chunk indices."""
  index = {}
  for dim, dim_offsets in offsets.items():
    index[dim] = {}
    for i, offset in enumerate(dim_offsets):
      index[dim][offset] = i
  return index


def dask_to_xbeam_chunks(
    dask_chunks: Mapping[Hashable, tuple[int, ...]],
) -> dict[Hashable, int]:
  """Convert dask chunks to xarray-beam chunks."""
  for dim, dim_chunks in dask_chunks.items():
    if len(dim_chunks) > 1:
      if len(set(dim_chunks[:-1])) > 1:
        raise ValueError(
            f"dimension {dim!r} has inconsistent dask chunks: "
            f"{dim_chunks}. All chunks except for the last must be equal."
        )
      if dim_chunks[-1] > dim_chunks[0]:
        raise ValueError(
            f"dimension {dim!r} has dask chunks where the last chunk "
            f"{dim_chunks[-1]} is larger than preceding chunks "
            f"{dim_chunks[0]}: {dim_chunks}."
        )
  return {k: v[0] for k, v in dask_chunks.items()}


def normalize_expanded_chunks(
    chunks: Mapping[str, int | tuple[int, ...]],
    dim_sizes: Mapping[str, int],
) -> dict[str, tuple[int, ...]]:
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
            f"sum of provided chunks does not match size of dimension {dim}: "
            f"{total} vs {dim_size}"
        )
      result[dim] = chunks[dim]
    else:
      multiple, remainder = divmod(dim_size, chunks[dim])
      result[dim] = multiple * (chunks[dim],) + (
          (remainder,) if remainder else ()
      )
  return result


DatasetOrDatasets = TypeVar(
    "DatasetOrDatasets", xarray.Dataset, list[xarray.Dataset]
)


@export
class DatasetToChunks(beam.PTransform, Generic[DatasetOrDatasets]):
  """Split one or more xarray.Datasets into keyed chunks."""

  def __init__(
      self,
      dataset: DatasetOrDatasets,
      chunks: Mapping[str, int | tuple[int, ...]] | None = None,
      split_vars: bool = False,
      num_threads: int | None = None,
      shard_keys_threshold: int = 200_000,
      tasks_per_shard: int = 10_000,
  ):
    """Initialize DatasetToChunks.

    Args:
      dataset: dataset or datasets to split into (Key, xarray.Dataset) or (Key,
        [xarray.Dataset, ...]) pairs.
      chunks: optional chunking scheme. Required if the dataset is *not* already
        chunked. If the dataset *is* already chunked with Dask, `chunks` takes
        precedence over the existing chunks.
      split_vars: whether to split the dataset into separate records for each
        data variable or to keep all data variables together. This is
        recommended if you don't need to perform joint operations on different
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
      tasks_per_shard: number of tasks to emit per shard. Only used if the
        number of tasks exceeds shard_keys_threshold.
    """
    self.dataset = dataset
    self._validate(dataset, split_vars)
    self.split_vars = split_vars
    self.num_threads = num_threads
    self.shard_keys_threshold = shard_keys_threshold
    self.tasks_per_shard = tasks_per_shard

    if chunks is None:
      dask_chunks = self._first.chunks
      if not dask_chunks:
        raise ValueError("dataset must be chunked or chunks must be provided")
      chunks = dask_to_xbeam_chunks(dask_chunks)

    for k in chunks:
      if k not in self._first.dims:
        raise ValueError(
            f"chunks key {k!r} is not a dimension on the provided dataset(s)"
        )

    self.chunks = chunks

  @property
  def _first(self) -> xarray.Dataset:
    return self._datasets[0]

  @property
  def _datasets(self) -> list[xarray.Dataset]:
    if isinstance(self.dataset, xarray.Dataset):
      return [self.dataset]
    return list(self.dataset)  # pytype: disable=bad-return-type

  @cached_property
  def expanded_chunks(self) -> dict[str, tuple[int, ...]]:
    return normalize_expanded_chunks(self.chunks, self._first.sizes)  # pytype: disable=wrong-arg-types  # always-use-property-annotation

  @cached_property
  def offsets(self) -> dict[str, list[int]]:
    return _chunks_to_offsets(self.expanded_chunks)

  @cached_property
  def offset_index(self) -> dict[str, dict[int, int]]:
    return compute_offset_index(self.offsets)

  def _validate(self, dataset, split_vars):
    """Raise errors if input parameters are invalid."""
    if not isinstance(dataset, xarray.Dataset):
      if not (
          isinstance(dataset, list)
          and all(isinstance(ds, xarray.Dataset) for ds in dataset)
      ):
        raise TypeError(
            "'dataset' must be an 'xarray.Dataset' or 'list[xarray.Dataset]'"
        )
      if not dataset:
        raise ValueError("dataset list cannot be empty")
    for ds in self._datasets[1:]:
      for dim, size in ds.sizes.items():
        if dim not in self._first.dims:
          raise ValueError(
              f"dimension {dim} does not appear on the first dataset"
          )
        if size != self._first.sizes[dim]:
          raise ValueError(
              f"dimension {dim} has an inconsistent size on different datasets"
          )
    if split_vars:
      for ds in self._datasets:
        if not ds.keys() <= self._first.keys():
          raise ValueError(
              "inconsistent data_vars when splitting variables:"
              f" {tuple(ds.keys())} != {tuple(self._first.keys())}"
          )

  def _task_count(self) -> int:
    """Count the number of tasks emitted by this transform."""
    counts = {k: len(v) for k, v in self.expanded_chunks.items()}
    if not self.split_vars:
      return int(np.prod(list(counts.values())))
    total = 0
    for variable in self._first.values():
      count_list = [v for k, v in counts.items() if k in variable.dims]
      total += int(np.prod(count_list))
    return total

  @cached_property
  def sharded_dim(self) -> str | None:
    # We use the simple heuristic of only sharding inputs along the dimension
    # with the most chunks.
    lengths = {
        k: math.ceil(size / self.chunks.get(k, size))
        for k, size in self._first.sizes.items()
    }
    return max(lengths, key=lengths.get) if lengths else None  # pytype: disable=bad-return-type

  @cached_property
  def shard_count(self) -> int | None:
    """Determine the number of times to shard input keys."""
    task_count = self._task_count()
    if task_count <= self.shard_keys_threshold:
      return None  # no sharding
    if not self.split_vars:
      return math.ceil(task_count / self.tasks_per_shard)
    var_count = sum(
        self.sharded_dim in var.dims for var in self._first.values()
    )
    return math.ceil(task_count / (var_count * self.tasks_per_shard))

  def _iter_all_keys(self) -> Iterator[Key]:
    """Iterate over all Key objects."""
    if not self.split_vars:
      yield from iter_chunk_keys(self.offsets)
    else:
      for name, variable in self._first.items():
        relevant_offsets = {
            k: v for k, v in self.offsets.items() if k in variable.dims
        }
        yield from iter_chunk_keys(relevant_offsets, vars={name})  # pytype: disable=wrong-arg-types  # always-use-property-annotation

  def _iter_shard_keys(
      self, shard_id: int | None, var_name: str | None
  ) -> Iterator[Key]:
    """Iterate over Key objects for a specific shard and variable."""
    if var_name is None:
      offsets = self.offsets
    else:
      offsets = {dim: self.offsets[dim] for dim in self._first[var_name].dims}

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

  def _shard_inputs(self) -> list[tuple[int | None, str | None]]:
    """Create inputs for sharded key iterators."""
    if not self.split_vars:
      return [(i, None) for i in range(self.shard_count)]

    inputs = []
    for name, variable in self._first.items():
      if self.sharded_dim in variable.dims:
        inputs.extend([(i, name) for i in range(self.shard_count)])
      else:
        inputs.append((None, name))
    return inputs  # pytype: disable=bad-return-type  # always-use-property-annotation

  def _key_to_chunks(self, key: Key) -> Iterator[tuple[Key, DatasetOrDatasets]]:
    """Convert a Key into an in-memory (Key, xarray.Dataset) pair."""
    with inc_timer_msec(self.__class__, "read-msec"):
      sizes = {
          dim: self.expanded_chunks[dim][self.offset_index[dim][offset]]
          for dim, offset in key.offsets.items()
      }
      slices = offsets_to_slices(key.offsets, sizes)
      results = []
      for ds in self._datasets:
        dataset = ds if key.vars is None else ds[list(key.vars)]
        valid_slices = {k: v for k, v in slices.items() if k in dataset.dims}
        chunk = dataset.isel(valid_slices)
        # Load the data, using a separate thread for each variable
        num_threads = len(dataset)
        result = chunk.chunk().compute(num_workers=num_threads)
        results.append(result)

    inc_counter(self.__class__, "read-chunks")
    inc_counter(
        self.__class__, "read-bytes", sum(result.nbytes for result in results)
    )

    if isinstance(self.dataset, xarray.Dataset):
      yield key, results[0]
    else:
      yield key, results

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
          | "GenerateKeys" >> beam.FlatMapTuple(self._iter_shard_keys)
          | beam.Reshuffle()
      )

    return key_pcoll | "KeyToChunks" >> threadmap.FlatThreadMap(
        self._key_to_chunks, num_threads=self.num_threads
    )


def _ensure_chunk_is_computed(key: Key, dataset: xarray.Dataset) -> None:
  """Ensure that a dataset contains no chunked variables."""
  for var_name, variable in dataset.variables.items():
    if variable.chunks is not None:
      raise ValueError(
          f"Dataset variable {var_name!r} corresponding to key {key} is"
          " chunked with Dask. Datasets passed to validate_chunk must be"
          f" fully computed (not chunked): {dataset}\nThis typically arises"
          " with datasets originating with `xarray.open_zarr()`, which by"
          " default use Dask. If this is the case, you can fix it by passing"
          " `chunks=None` or xarray_beam.open_zarr(). Alternatively, you"
          " can load datasets explicitly into memory with `.compute()`."
      )


@export
def validate_chunk(key: Key, datasets: DatasetOrDatasets) -> None:
  """Verify that a key and dataset(s) are valid for xarray-beam transforms."""
  if isinstance(datasets, xarray.Dataset):
    datasets: list[xarray.Dataset] = [datasets]

  for dataset in datasets:
    # Verify that no variables are chunked with Dask
    _ensure_chunk_is_computed(key, dataset)

    # Validate key offsets
    missing_keys = [
        repr(k) for k in key.offsets.keys() if k not in dataset.dims
    ]
    if missing_keys:
      raise ValueError(
          f"Key offset(s) {', '.join(missing_keys)} in {key} not found in"
          f" Dataset dimensions: {dataset!r}"
      )

    # Validate key vars
    if key.vars is not None:
      missing_vars = [repr(v) for v in key.vars if v not in dataset.data_vars]
      if missing_vars:
        raise ValueError(
            f"Key var(s) {', '.join(missing_vars)} in {key} not found in"
            f" Dataset data variables: {dataset!r}"
        )


@export
class ValidateEachChunk(beam.PTransform):
  """Check that keys and dataset(s) are valid for xarray-beam transforms."""

  def _validate(self, key, dataset):
    validate_chunk(key, dataset)
    return key, dataset

  def expand(self, pcoll):
    return pcoll | beam.MapTuple(self._validate)
