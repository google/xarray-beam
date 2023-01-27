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
import dataclasses
import functools
from typing import Optional, Sequence, Union

import apache_beam as beam
import numpy.typing as npt
import xarray

from xarray_beam._src import core


# TODO(shoyer): add other combiners: sum, std, var, min, max, etc.


DimLike = Optional[Union[str, Sequence[str]]]


@dataclasses.dataclass
class MeanCombineFn(beam.transforms.CombineFn):
  """CombineFn for computing an arithmetic mean of xarray.Dataset objects."""

  dim: DimLike = None
  skipna: bool = True
  dtype: Optional[npt.DTypeLike] = None

  def create_accumulator(self):
    return (0, 0)

  def add_input(self, sum_count, element):
    (sum_, count) = sum_count

    if self.dtype is not None:
      element = element.astype(self.dtype)

    if self.skipna:
      sum_increment = element.fillna(0)
      count_increment = element.notnull()
    else:
      sum_increment = element
      count_increment = xarray.ones_like(element)

    if self.dim is not None:
      sum_increment = sum_increment.sum(self.dim)
      count_increment = count_increment.sum(self.dim)

    new_sum = sum_ + sum_increment
    new_count = count + count_increment

    return new_sum, new_count

  def merge_accumulators(self, accumulators):
    sums, counts = zip(*accumulators)
    return sum(sums), sum(counts)

  def extract_output(self, sum_count):
    (sum_, count) = sum_count
    return sum_ / count

  def for_input_type(self, input_type):
    return self


@dataclasses.dataclass
class Mean(beam.PTransform):
  """Calculate the mean over one or more distributed dataset dimensions."""

  dim: Union[str, Sequence[str]]
  skipna: bool = True
  dtype: Optional[npt.DTypeLike] = None

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
        | Mean.PerKey(self.dim, self.skipna, self.dtype)
    )

  @dataclasses.dataclass
  class Globally(beam.PTransform):
    """Calculate global mean over a pcollection of xarray.Dataset objects."""

    dim: DimLike = None
    skipna: bool = True
    dtype: Optional[npt.DTypeLike] = None

    def expand(self, pcoll):
      combine_fn = MeanCombineFn(self.dim, self.skipna, self.dtype)
      return pcoll | beam.CombineGlobally(combine_fn)

  @dataclasses.dataclass
  class PerKey(beam.PTransform):
    """Calculate per-key mean over a pcollection of (hashable, Dataset)."""

    dim: DimLike = None
    skipna: bool = True
    dtype: Optional[npt.DTypeLike] = None

    def expand(self, pcoll):
      combine_fn = MeanCombineFn(self.dim, self.skipna, self.dtype)
      return pcoll | beam.CombinePerKey(combine_fn)
