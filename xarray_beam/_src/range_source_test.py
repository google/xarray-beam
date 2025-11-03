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
"""Tests for xarray_beam._src.range_source."""
from __future__ import annotations

from absl.testing import absltest
import apache_beam as beam
from xarray_beam._src import range_source
from xarray_beam._src import test_util


class RangeSourceTest(test_util.TestCase):

  def test_read(self):
    source = range_source.RangeSource(
        element_count=5,
        element_size=1,
        get_element=lambda i: f'elem_{i}',
    )
    result = test_util.EagerPipeline() | beam.io.Read(source)
    self.assertEqual(result, ['elem_0', 'elem_1', 'elem_2', 'elem_3', 'elem_4'])

  def test_estimate_size(self):
    source = range_source.RangeSource(10, 8, lambda i: i)
    self.assertEqual(source.estimate_size(), 80)

  def test_split(self):
    source = range_source.RangeSource(10, 1, lambda i: i)
    splits = list(source.split(desired_bundle_size=3))
    # 10 elements, size 1, bundle size 3 bytes -> 3 elements/bundle
    # bundles: [0,3), [3,6), [6,9), [9,10)
    self.assertEqual(len(splits), 4)
    positions = [(s.start_position, s.stop_position) for s in splits]
    self.assertEqual(positions, [(0, 3), (3, 6), (6, 9), (9, 10)])
    weights = [s.weight for s in splits]
    self.assertEqual(weights, [3, 3, 3, 1])

  def test_read_empty_source(self):
    source = range_source.RangeSource(0, 1, lambda i: i)
    result = test_util.EagerPipeline() | beam.io.Read(source)
    self.assertEqual(result, [])

  def test_nonsplittable_range_is_read(self):
    """Test reading a range that is not splittable."""
    source = range_source.RangeSource(
        element_count=1, get_element=str, element_size=1
    )
    result = test_util.EagerPipeline() | 'Read' >> beam.io.Read(source)
    self.assertEqual(result, ['0'])


if __name__ == '__main__':
  absltest.main()
