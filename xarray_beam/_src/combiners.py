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
import dataclasses
from typing import Any

import apache_beam as beam


# TODO(shoyer): add other combiners: sum, std, var, min, max, etc.


@dataclasses.dataclass
class MeanCombineFn(beam.transforms.CombineFn):
  """CombineFn for computing an arithmetic mean of xarray.Dataset objects."""

  skipna: bool = True
  dtype: Any = None

  def create_accumulator(self):
    return (0, 0)

  def add_input(self, sum_count, element):
    (sum_, count) = sum_count
    if self.dtype is not None:
      element = element.astype(self.dtype)
    if self.skipna:
      new_sum = sum_ + element.fillna(0)
      new_count = count + element.notnull()
    else:
      new_sum = sum_ + element
      new_count = count + 1
    return new_sum, new_count

  def merge_accumulators(self, accumulators):
    sums, counts = zip(*accumulators)
    return sum(sums), sum(counts)

  def extract_output(self, sum_count):
    (sum_, count) = sum_count
    return sum_ / count

  def for_input_type(self, input_type):
    return self


class Mean:
  """Combiners for computing arithmetic means of xarray.Dataset objects."""

  @dataclasses.dataclass
  class Globally(beam.PTransform):
    """Calculate the global mean over a pcollection."""

    skipna: bool = True
    dtype: Any = None

    def expand(self, pcoll):
      combine_fn = MeanCombineFn(self.skipna, self.dtype)
      return pcoll | beam.CombineGlobally(combine_fn)

  @dataclasses.dataclass
  class PerKey(beam.PTransform):
    """Calculate the per-key mean over a pcollection."""

    skipna: bool = True
    dtype: Any = None

    def expand(self, pcoll):
      combine_fn = MeanCombineFn(self.skipna, self.dtype)
      return pcoll | beam.CombinePerKey(combine_fn)
