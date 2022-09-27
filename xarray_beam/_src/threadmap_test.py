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
"""Tests for threadmap."""

from absl.testing import absltest
import apache_beam as beam

from xarray_beam._src import test_util
from xarray_beam._src import threadmap


# pylint: disable=expression-not-assigned
# pylint: disable=pointless-statement


class ThreadMapTest(test_util.TestCase):

  def test_map(self):
    def f(*args, **kwargs):
      return args, kwargs

    expected = [1, 2, 3] | beam.Map(f, 4, y=5)
    actual = [1, 2, 3] | threadmap.ThreadMap(f, 4, y=5)
    self.assertEqual(expected, actual)

    actual = [1, 2, 3] | threadmap.ThreadMap(f, 4, y=5, num_threads=None)
    self.assertEqual(expected, actual)

  def test_flat_map(self):
    def f(*args, **kwargs):
      return [(args, kwargs)] * 2

    expected = [1, 2, 3] | beam.FlatMap(f, 4, y=5)
    actual = [1, 2, 3] | threadmap.FlatThreadMap(f, 4, y=5)
    self.assertEqual(expected, actual)

    actual = [1, 2, 3] | threadmap.FlatThreadMap(f, 4, y=5, num_threads=None)
    self.assertEqual(expected, actual)

  def test_map_tuple(self):
    def f(a, b, y=None):
      return a, b, y

    expected = [(1, 2), (3, 4)] | beam.MapTuple(f, y=5)
    actual = [(1, 2), (3, 4)] | threadmap.ThreadMapTuple(f, y=5)
    self.assertEqual(expected, actual)

    actual = [(1, 2), (3, 4)] | threadmap.ThreadMapTuple(
        f, y=5, num_threads=None
    )
    self.assertEqual(expected, actual)

  def test_flat_map_tuple(self):
    def f(a, b, y=None):
      return a, b, y

    expected = [(1, 2), (3, 4)] | beam.FlatMapTuple(f, y=5)
    actual = [(1, 2), (3, 4)] | threadmap.FlatThreadMapTuple(f, y=5)
    self.assertEqual(expected, actual)

    actual = [(1, 2), (3, 4)] | threadmap.FlatThreadMapTuple(
        f, y=5, num_threads=None
    )
    self.assertEqual(expected, actual)

  def test_maybe_thread_map(self):
    transform = threadmap.ThreadMap(lambda x: x)
    self.assertIsInstance(transform, threadmap._ThreadMap)

    transform = threadmap.ThreadMap(lambda x: x, num_threads=None)
    self.assertIsInstance(transform, beam.ParDo)

    transform = threadmap.ThreadMap(lambda x: x, num_threads=1)
    self.assertIsInstance(transform, threadmap._ThreadMap)


if __name__ == '__main__':
  absltest.main()
