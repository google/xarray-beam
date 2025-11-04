# Copyright 2025 Google LLC
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
"""Beam source for arbitrary data."""
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Generic, Iterator, TypeVar

import apache_beam as beam
from apache_beam.io import iobase


_T = TypeVar('_T')


@dataclasses.dataclass
class RangeSource(iobase.BoundedSource, Generic[_T]):
  """A Beam BoundedSource for a range of elements.

  This source is defined by a count, size of each element, and a function to
  retrieve an element by index.

  Attributes:
    element_count: number of elements in this source.
    element_size: size of each element in bytes.
    get_element: callable that given an integer index in the range ``[0,
      element_count)`` returns the corresponding element of the source.
  """

  element_count: int
  element_size: int
  get_element: Callable[[int], _T]
  coder: beam.coders.Coder = beam.coders.PickleCoder()

  def __post_init__(self):
    if self.element_count < 0:
      raise ValueError(
          f'element_count must be non-negative: {self.element_count}'
      )
    if self.element_size < 0:
      raise ValueError(
          f'element_size must be non-negative: {self.element_size}'
      )

  def estimate_size(self) -> int:
    """Estimates the size of source in bytes."""
    return self.element_count * self.element_size

  def split(
      self,
      desired_bundle_size: int,
      start_position: int | None = None,
      stop_position: int | None = None,
  ) -> Iterator[iobase.SourceBundle]:
    """Splits the source into a set of bundles."""
    start = start_position if start_position is not None else 0
    stop = stop_position if stop_position is not None else self.element_count

    bundle_size_in_elements = int(
        math.ceil(desired_bundle_size / max(self.element_size, 1))
    )
    for bundle_start in range(start, stop, bundle_size_in_elements):
      bundle_stop = min(bundle_start + bundle_size_in_elements, stop)
      weight = (bundle_stop - bundle_start) * self.element_size
      yield iobase.SourceBundle(weight, self, bundle_start, bundle_stop)

  def get_range_tracker(
      self,
      start_position: int | None,
      stop_position: int | None,
  ) -> beam.io.OffsetRangeTracker:
    """Returns a RangeTracker for a given position range."""
    start = start_position if start_position is not None else 0
    stop = stop_position if stop_position is not None else self.element_count
    return beam.io.OffsetRangeTracker(start, stop)

  def read(
      self, range_tracker: beam.io.OffsetRangeTracker
  ) -> Iterator[_T]:
    """Returns an iterator that reads data from the source."""
    i = range_tracker.start_position()
    while range_tracker.try_claim(i):
      yield self.get_element(i)
      i += 1

  def default_output_coder(self) -> beam.coders.Coder:
    """Coder that should be used for the records returned by the source."""
    return self.coder
