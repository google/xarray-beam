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

from collections.abc import Sequence
import dataclasses

import apache_beam as beam
import numpy.typing as npt
import xarray
from xarray_beam._src import core


# TODO(shoyer): add other combiners: sum, std, var, min, max, etc.


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
    (sum_, count) = sum_count
    return sum_ / count


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
