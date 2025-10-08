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
"""Combiners for xarray-beam."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import logging
import math
from typing import Literal

import apache_beam as beam
import numpy.typing as npt
import xarray
from xarray_beam._src import core


# TODO(shoyer): add other combiners: sum, std, var, min, max, etc.

# pylint: disable=logging-fstring-interpolation


DimLike = str | Sequence[str] | None


@dataclasses.dataclass
class _SumAndCount:
  """Calculate the sum and count of an xarray.Dataset."""

  dim: DimLike = None
  skipna: bool = True
  dtype: npt.DTypeLike | None = None

  def __call__(
      self, chunk: xarray.Dataset
  ) -> tuple[xarray.Dataset, xarray.Dataset]:
    if self.dtype is not None:
      chunk = chunk.astype(self.dtype)

    if self.skipna:
      sum_increment = chunk.fillna(0)
      count_increment = chunk.notnull()
    else:
      sum_increment = chunk
      count_increment = xarray.ones_like(chunk)

    if self.dim is not None:
      # unconditionally set skipna=False because we already explictly fill in
      # missing values explicitly above
      sum_increment = sum_increment.sum(self.dim, skipna=False)
      count_increment = count_increment.sum(self.dim)

    return sum_increment, count_increment


@dataclasses.dataclass
class MeanCombineFn(beam.transforms.CombineFn):
  """CombineFn for computing an arithmetic mean of xarray.Dataset objects."""

  sum_and_count: _SumAndCount | None = None
  finalize: bool = True

  def create_accumulator(self):
    return (0, 0)

  def add_input(self, sum_count, element):
    (sum_, count) = sum_count
    if self.sum_and_count is not None:
      sum_increment, count_increment = self.sum_and_count(element)
    else:
      sum_increment, count_increment = element
    new_sum = sum_ + sum_increment
    new_count = count + count_increment
    return new_sum, new_count

  def merge_accumulators(self, accumulators):
    sums, counts = zip(*accumulators)
    return sum(sums), sum(counts)

  def extract_output(self, sum_count):
    if self.finalize:
      (sum_, count) = sum_count
      return sum_ / count
    else:
      return sum_count


@dataclasses.dataclass
class Mean(beam.PTransform):
  """Calculate the mean over one or more distributed dataset dimensions.

  This PTransform expects a PCollection of `(key, chunk)` pairs, and outputs a
  PCollection where chunks with the same key (excluding dimensions in `dim`)
  have been averaged together.

  Args:
    dim: Dimension(s) to average over.
    skipna: If True, skip missing values (NaN) when calculating the mean.
    dtype: Data type to use for sum and count accumulators.
    fanout: If provided, use `CombinePerKey.with_hot_key_fanout` to handle hot
      keys by injecting intermediate merging nodes.
    pre_aggregate: If True, calculate sum and count for each chunk before
      combining. This is usually more efficient.
  """

  dim: str | Sequence[str]
  skipna: bool = True
  dtype: npt.DTypeLike | None = None
  fanout: int | None = None
  pre_aggregate: bool = True

  def _update_key(
      self, key: core.Key, chunk: xarray.Dataset
  ) -> tuple[core.Key, xarray.Dataset]:
    dims = [self.dim] if isinstance(self.dim, str) else self.dim
    new_key = key.with_offsets(**{d: None for d in dims if d in key.offsets})
    return new_key, chunk

  def expand(self, pcoll):
    return (
        pcoll
        | beam.MapTuple(self._update_key)
        | Mean.PerKey(
            self.dim, self.skipna, self.dtype, self.fanout, self.pre_aggregate
        )
    )

  @dataclasses.dataclass
  class Globally(beam.PTransform):
    """Calculate global mean over a pcollection of xarray.Dataset objects."""

    dim: DimLike = None
    skipna: bool = True
    dtype: npt.DTypeLike | None = None
    fanout: int | None = None
    pre_aggregate: bool = True

    def expand(self, pcoll):
      sum_and_count = _SumAndCount(self.dim, self.skipna, self.dtype)
      if self.pre_aggregate:
        pcoll = pcoll | beam.Map(sum_and_count)
        combine_fn = MeanCombineFn(sum_and_count=None)
      else:
        combine_fn = MeanCombineFn(sum_and_count)
      return pcoll | beam.CombineGlobally(combine_fn).with_fanout(self.fanout)

  @dataclasses.dataclass
  class PerKey(beam.PTransform):
    """Calculate per-key mean over a pcollection of (hashable, Dataset)."""

    dim: DimLike = None
    skipna: bool = True
    dtype: npt.DTypeLike | None = None
    fanout: int | None = None
    pre_aggregate: bool = True

    def expand(self, pcoll):
      sum_and_count = _SumAndCount(self.dim, self.skipna, self.dtype)
      if self.pre_aggregate:
        pcoll = pcoll | beam.MapTuple(lambda k, v: (k, sum_and_count(v)))
        combine_fn = MeanCombineFn(sum_and_count=None)
      else:
        combine_fn = MeanCombineFn(sum_and_count)
      return pcoll | beam.CombinePerKey(combine_fn).with_hot_key_fanout(
          self.fanout
      )


def _get_chunk_index(
    key: core.Key,
    dims: Sequence[str],
    chunks: Mapping[str, int],
    sizes: Mapping[str, int],
) -> int:
  """Calculate a flat index from chunk indices."""
  chunk_indices = [key.offsets[d] // chunks[d] for d in dims]
  shape = [math.ceil(sizes[d] / chunks[d]) for d in dims]
  chunk_index = 0
  for i, index in enumerate(chunk_indices):
    chunk_index += index * math.prod(shape[i + 1 :])
  return chunk_index


def _index_to_fanout_bins(
    index: int,
    bins_per_stage: tuple[int, ...],
) -> tuple[int, ...]:
  """Assign a flat index to bins for fanout aggregation."""
  total_bins = math.prod(bins_per_stage)
  bin_id = index % total_bins
  bins = []
  for factor in bins_per_stage:
    bins.append(bin_id % factor)
    bin_id //= factor
  return tuple(bins)


def _complete_fanout_bins(
    fanout: int, stages: int, chunks_count: int
) -> tuple[int, ...]:
  for k in range(stages + 1):
    if fanout**k * (fanout - 1) ** (stages - k) >= chunks_count:
      # all things being equal, prefer higher fanout at earlier stages, because
      # this results in a bit less overhead for writing, and the first stage(s)
      # are more likely to saturate all available workers.
      return (fanout,) * k + (fanout - 1,) * (stages - k)
  raise AssertionError(
      f'invalid fanout/stages/chunks_count: {fanout=}, {stages=},'
      f' {chunks_count=}'
  )


def _all_fanout_schedule_costs(
    chunks_count: int,
    bytes_per_chunk: float,
    max_workers: int,
    cost_per_stage: float = 0.1,
    chunks_per_second: float = 1500,
    bytes_per_second: float = 25_000_000,
) -> dict[tuple[int, ...], float]:
  """Estimate the cost of all fanout schedules, as a runtime in seconds."""
  candidates = {}
  # fanout must always be 2 or larger, so the largest possible number of stages
  # is log_2(chunks_count). This is a small enough set of candidates we can
  # generate them all via brute force.
  for stages in range(1, math.ceil(math.log2(chunks_count)) + 1):
    fanout = math.ceil(chunks_count ** (1 / stages))
    bins = _complete_fanout_bins(fanout, stages, chunks_count)
    cost = 0
    tasks = chunks_count
    for stage_bins in bins:
      tasks = math.ceil(tasks / stage_bins)
      # Our model here is that chunk processing has fixed overhead per chunk and
      # per byte. For simplify, we assume that reading and writing have the same
      # cost.
      chunks = fanout + 1  # one extra chunk for writing
      runtime_per_task = (
          chunks / chunks_per_second
          + bytes_per_chunk * chunks / bytes_per_second
      )
      cost += math.ceil(tasks / max_workers) * runtime_per_task + cost_per_stage
    candidates[bins] = cost
  return candidates


def _optimal_fanout_bins(
    dims: Sequence[str],
    chunks: Mapping[str, int],
    sizes: Mapping[str, int],
    itemsize: int,
) -> tuple[int, ...]:
  """Calculate the optimal fanout schedule for a multi-stage mean."""
  chunks_count = math.prod(math.ceil(sizes[d] / chunks[d]) for d in dims)

  bytes_per_chunk = itemsize * math.prod(
      chunks[d] for d in chunks if d not in dims
  )

  # We don't really know how many workers will be available (in reality the
  # Beam runner will likely adjust this dynamically), but one per 5GB of input
  # data up to a max of 10k is in the right ballpark.
  orig_nbytes = itemsize * math.prod(sizes.values())
  max_workers = max(math.ceil(orig_nbytes / 5e9), 10_000)

  candidates = _all_fanout_schedule_costs(
      chunks_count, bytes_per_chunk, max_workers
  )
  # The dict of candidates is empty if chunks_count=1, in which can there's no
  # need to use a combiner.
  return min(candidates, key=candidates.get) if candidates else ()


@dataclasses.dataclass
class MultiStageMean(beam.PTransform):
  """Calculate the mean over dataset dimensions, via multiple stages.

  This can be much faster more efficient than using Mean(), but requires
  understanding the full dataset structure.
  """

  dims: Sequence[str]
  skipna: bool
  dtype: npt.DTypeLike | None
  chunks: Mapping[str, int]
  sizes: Mapping[str, int]
  itemsize: int
  bins_per_stage: tuple[int, ...] | None = None
  pre_aggregate: bool | None = None

  def __post_init__(self):
    if self.bins_per_stage is None:
      self.bins_per_stage = _optimal_fanout_bins(
          self.dims, self.chunks, self.sizes, self.itemsize
      )
    if self.pre_aggregate is None:
      self.pre_aggregate = (
          math.prod(self.chunks[d] for d in self.dims) > 1
          or not self.bins_per_stage
      )
    stages = len(self.bins_per_stage)
    logging.info(
        f'Dataset mean with {stages} stages '
        f'(bins_per_stage={self.bins_per_stage}) and'
        f' pre_aggregate={self.pre_aggregate}'
    )

  def _finalize_no_combiner(
      self, key: core.Key, sum_count: tuple[xarray.Dataset, xarray.Dataset]
  ) -> tuple[core.Key, xarray.Dataset]:
    key = key.with_offsets(**{d: None for d in self.dims if d in key.offsets})
    sum_, count = sum_count
    return key, sum_ / count

  def _prepare_key(
      self, key: core.Key, chunk: xarray.Dataset
  ) -> tuple[tuple[tuple[int, ...], core.Key], xarray.Dataset]:
    assert self.bins_per_stage  # not empty
    index = _get_chunk_index(key, self.dims, self.chunks, self.sizes)
    # strip the final bin because it isn't needed for the combiner
    bin_ids = _index_to_fanout_bins(index, self.bins_per_stage[:-1])
    key = key.with_offsets(**{d: None for d in self.dims if d in key.offsets})
    return ((bin_ids, key), chunk)

  def _strip_leading_fanout_bin(
      self, bin_key: tuple[tuple[int, ...], core.Key], value: xarray.Dataset
  ) -> tuple[tuple[tuple[int, ...], core.Key], xarray.Dataset]:
    bin_ids, key = bin_key
    return (bin_ids[1:], key), value

  def _strip_fanout_bins(
      self, bin_key: tuple[tuple[int, ...], core.Key], value: xarray.Dataset
  ) -> tuple[core.Key, xarray.Dataset]:
    bin_ids, key = bin_key
    assert not bin_ids
    return key, value

  def expand(self, pcoll):
    sum_and_count = _SumAndCount(self.dims, self.skipna, self.dtype)

    if not self.bins_per_stage:  # no combiner needed
      pcoll |= 'Aggregate' >> beam.MapTuple(lambda k, v: (k, sum_and_count(v)))
      pcoll |= 'Finalize' >> beam.MapTuple(self._finalize_no_combiner)
      return pcoll

    if self.pre_aggregate:
      pcoll |= 'PreAggregate' >> beam.MapTuple(
          lambda k, v: (k, sum_and_count(v))
      )
    pcoll |= 'PrepareKey' >> beam.MapTuple(self._prepare_key)
    for i in range(len(self.bins_per_stage)):
      final_stage = i + 1 >= len(self.bins_per_stage)
      if self.pre_aggregate or i > 0:
        combine_fn = MeanCombineFn(None, finalize=final_stage)
      else:
        combine_fn = MeanCombineFn(sum_and_count, finalize=final_stage)
      pcoll |= f'Combine{i}' >> beam.CombinePerKey(combine_fn)
      if not final_stage:
        pcoll |= f'StripBin{i}' >> beam.MapTuple(self._strip_leading_fanout_bin)
    pcoll |= 'StripFanoutBins' >> beam.MapTuple(self._strip_fanout_bins)
    return pcoll
