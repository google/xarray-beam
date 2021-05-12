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
"""Tests for xarray_beam._src.core."""

from absl.testing import absltest
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam
from xarray_beam._src import core
from xarray_beam._src import test_util


# pylint: disable=expression-not-assigned
# pylint: disable=pointless-statement


class ChunkKeyTest(test_util.TestCase):

  def test_mapping(self):
    key = xarray_beam.ChunkKey({'x': 0, 'y': 10})
    self.assertEqual(list(key.keys()), ['x', 'y'])
    self.assertEqual(list(key.values()), [0, 10])
    self.assertIn('x', key)
    self.assertNotIn('z', key)
    self.assertEqual(key['x'], 0)

  def test_immutability(self):
    key = xarray_beam.ChunkKey({'x': 0, 'y': 10})

    with self.assertRaises(TypeError):
      key['z'] = 100

    dict_ = {key: 'foo'}
    self.assertEqual(dict_[key], 'foo')

  def test_to_slices(self):
    key = xarray_beam.ChunkKey({'x': 0, 'y': 10})

    expected = {'x': slice(0, 5, 1), 'y': slice(10, 20, 1)}
    slices = key.to_slices({'x': 5, 'y': 10})
    self.assertEqual(slices, expected)

    slices = key.to_slices({'x': 5, 'y': 10, 'extra_key': 100})
    self.assertEqual(slices, expected)

    expected = {'x': slice(None), 'y': slice(10, 20, 1)}
    slices = key.to_slices({'y': 10})
    self.assertEqual(slices, expected)

    with self.assertRaisesRegex(ValueError, 'non-zero offset'):
      key.to_slices({'x': 5})

  def test_operators(self):
    key = xarray_beam.ChunkKey({'x': 0, 'y': 10})

    expected = xarray_beam.ChunkKey({'x': 0, 'y': 10, 'z': 100})
    actual = key | {'z': 100}
    self.assertEqual(actual, expected)

    expected = xarray_beam.ChunkKey({'y': 10})
    actual = key - {'x'}
    self.assertEqual(actual, expected)

    with self.assertRaises(TypeError):
      key - 'x'

    with self.assertRaisesRegex(ValueError, 'not found'):
      key - {'z'}

  def test_repr(self):
    key = xarray_beam.ChunkKey({'x': 0, 'y': 10})
    expected = "ChunkKey({'x': 0, 'y': 10})"
    self.assertEqual(repr(key), expected)

  def test_comparison(self):
    key = xarray_beam.ChunkKey({'x': 0, 'y': 10})
    with self.assertRaises(TypeError):
      key < 'foo'
    with self.assertRaisesRegex(ValueError, 'Dimensions must match'):
      key < xarray_beam.ChunkKey({'x': 0})
    other = xarray_beam.ChunkKey({'x': 0, 'y': 20})
    self.assertLess(key, other)
    self.assertGreater(other, key)

  def test_use_as_beam_key(self):
    inputs = [
        (xarray_beam.ChunkKey({'x': 0, 'y': 1}), 1),
        (xarray_beam.ChunkKey({'x': 0, 'y': 2}), 2),
        (xarray_beam.ChunkKey({'y': 1, 'x': 0}), 3),
    ]
    expected = [
        (xarray_beam.ChunkKey({'x': 0, 'y': 1}), [1, 3]),
        (xarray_beam.ChunkKey({'x': 0, 'y': 2}), [2]),
    ]
    actual = inputs | beam.GroupByKey()
    self.assertEqual(actual, expected)


class DatasetToChunksTest(test_util.TestCase):

  def test_iter_chunk_keys(self):
    actual = sorted(core.iter_chunk_keys({'x': (3, 3), 'y': (2, 2, 2)}))
    expected = [
        xarray_beam.ChunkKey({'x': 0, 'y': 0}),
        xarray_beam.ChunkKey({'x': 0, 'y': 2}),
        xarray_beam.ChunkKey({'x': 0, 'y': 4}),
        xarray_beam.ChunkKey({'x': 3, 'y': 0}),
        xarray_beam.ChunkKey({'x': 3, 'y': 2}),
        xarray_beam.ChunkKey({'x': 3, 'y': 4}),
    ]
    self.assertEqual(actual, expected)

  def test_iter_chunk_keys_with_base(self):
    actual = sorted(core.iter_chunk_keys({'x': (3, 3)}, base={'x': 30}))
    expected = [
        xarray_beam.ChunkKey({'x': 30}),
        xarray_beam.ChunkKey({'x': 33}),
    ]
    self.assertEqual(actual, expected)

  def test_iter_chunk_keys_with_more_base_dims(self):
    actual = sorted(core.iter_chunk_keys({'x': (3, 3)}, base={'x': 30, 'y': 0}))
    expected = [
        xarray_beam.ChunkKey({'x': 30, 'y': 0}),
        xarray_beam.ChunkKey({'x': 33, 'y': 0}),
    ]
    self.assertEqual(actual, expected)

  def test_compute_offset_index(self):
    actual = core.compute_offset_index({'x': (0, 3), 'y': (0, 2, 4)})
    expected = {'x': {0: 0, 3: 1}, 'y': {0: 0, 2: 1, 4: 2}}
    self.assertEqual(actual, expected)

  def test_normalize_expanded_chunks(self):
    actual = core.normalize_expanded_chunks({}, {'x': 10})
    expected = {'x': (10,)}
    self.assertEqual(actual, expected)

    actual = core.normalize_expanded_chunks({'x': -1}, {'x': 10})
    expected = {'x': (10,)}
    self.assertEqual(actual, expected)

    actual = core.normalize_expanded_chunks({'x': (5, 5)}, {'x': 10})
    expected = {'x': (5, 5)}
    self.assertEqual(actual, expected)

    with self.assertRaisesRegex(
        ValueError, 'sum of provided chunks does not match',
    ):
      core.normalize_expanded_chunks({'x': (5, 5, 5)}, {'x': 10})

    actual = core.normalize_expanded_chunks({'x': 3, 'y': 2}, {'x': 9, 'y': 4})
    expected = {'x': (3, 3, 3), 'y': (2, 2)}
    self.assertEqual(actual, expected)

    actual = core.normalize_expanded_chunks({'x': 3}, {'x': 10})
    expected = {'x': (3, 3, 3, 1)}
    self.assertEqual(actual, expected)

  def test_dataset_to_chunks_multiple(self):
    dataset = xarray.Dataset({'foo': ('x', np.arange(6))})
    expected = [
        (xarray_beam.ChunkKey({'x': 0}), dataset.head(x=3)),
        (xarray_beam.ChunkKey({'x': 3}), dataset.tail(x=3)),
    ]
    actual = (
        test_util.EagerPipeline()
        | xarray_beam.DatasetToChunks(dataset.chunk({'x': 3}))
    )
    self.assertIdenticalChunks(actual, expected)

    actual = (
        test_util.EagerPipeline()
        | xarray_beam.DatasetToChunks(dataset.chunk({'x': 3}), num_threads=2)
    )
    self.assertIdenticalChunks(actual, expected)

    actual = (
        test_util.EagerPipeline()
        | xarray_beam.DatasetToChunks(dataset, chunks={'x': 3})
    )
    self.assertIdenticalChunks(actual, expected)

  def test_dataset_to_chunks_whole(self):
    dataset = xarray.Dataset({'foo': ('x', np.arange(6))})
    expected = [(xarray_beam.ChunkKey({'x': 0}), dataset)]
    actual = (
        test_util.EagerPipeline()
        | xarray_beam.DatasetToChunks(dataset, chunks={'x': -1})
    )
    self.assertIdenticalChunks(actual, expected)

    actual = (
        test_util.EagerPipeline()
        | xarray_beam.DatasetToChunks(dataset, chunks={})
    )
    self.assertIdenticalChunks(actual, expected)


if __name__ == '__main__':
  absltest.main()
