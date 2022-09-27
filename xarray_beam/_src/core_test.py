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
from absl.testing import parameterized
import apache_beam as beam
import immutabledict
import numpy as np
import xarray
import xarray_beam as xbeam
from xarray_beam._src import core
from xarray_beam._src import test_util

# pylint: disable=expression-not-assigned
# pylint: disable=pointless-statement


class KeyTest(test_util.TestCase):

  def test_constructor(self):
    key = xbeam.Key({'x': 0, 'y': 10})
    self.assertIsInstance(key.offsets, immutabledict.immutabledict)
    self.assertEqual(dict(key.offsets), {'x': 0, 'y': 10})
    self.assertIsNone(key.vars)

    key = xbeam.Key(vars={'foo'})
    self.assertEqual(dict(key.offsets), {})
    self.assertIsInstance(key.vars, frozenset)
    self.assertEqual(set(key.vars), {'foo'})

    with self.assertRaisesRegex(TypeError, 'vars must be a set or None'):
      xbeam.Key(vars='foo')

  def test_replace(self):
    key = xbeam.Key({'x': 0}, {'foo'})

    expected = xbeam.Key({'x': 1}, {'foo'})
    actual = key.replace({'x': 1})
    self.assertEqual(expected, actual)

    expected = xbeam.Key({'y': 1}, {'foo'})
    actual = key.replace({'y': 1})
    self.assertEqual(expected, actual)

    expected = xbeam.Key({'x': 0})
    actual = key.replace(vars=None)
    self.assertEqual(expected, actual)

    expected = xbeam.Key({'x': 0}, {'bar'})
    actual = key.replace(vars={'bar'})
    self.assertEqual(expected, actual)

    expected = xbeam.Key({'y': 1}, {'foo'})
    actual = key.replace({'y': 1}, {'foo'})
    self.assertEqual(expected, actual)

    expected = xbeam.Key({'y': 1}, {'bar'})
    actual = key.replace({'y': 1}, {'bar'})
    self.assertEqual(expected, actual)

  def test_with_offsets(self):
    key = xbeam.Key({'x': 0})

    expected = xbeam.Key({'x': 1})
    actual = key.with_offsets(x=1)
    self.assertEqual(expected, actual)

    expected = xbeam.Key({'x': 0, 'y': 1})
    actual = key.with_offsets(y=1)
    self.assertEqual(expected, actual)

    expected = xbeam.Key()
    actual = key.with_offsets(x=None)
    self.assertEqual(expected, actual)

    expected = xbeam.Key({'y': 1, 'z': 2})
    actual = key.with_offsets(x=None, y=1, z=2)
    self.assertEqual(expected, actual)

    key2 = xbeam.Key({'x': 0}, vars={'foo'})
    expected = xbeam.Key({'x': 1}, vars={'foo'})
    actual = key2.with_offsets(x=1)
    self.assertEqual(expected, actual)

  def test_repr(self):
    key = xbeam.Key({'x': 0, 'y': 10})
    expected = "Key(offsets={'x': 0, 'y': 10}, vars=None)"
    self.assertEqual(repr(key), expected)

    key = xbeam.Key(vars={'foo'})
    expected = "Key(offsets={}, vars={'foo'})"
    self.assertEqual(repr(key), expected)

  def test_dict_key(self):
    first = {xbeam.Key({'x': 0, 'y': 10}): 1}
    second = {xbeam.Key({'x': 0, 'y': 10}): 1}
    self.assertEqual(first, second)

  def test_equality(self):
    key = xbeam.Key({'x': 0, 'y': 10})
    self.assertEqual(key, key)
    self.assertNotEqual(key, None)

    key2 = xbeam.Key({'x': 0, 'y': 10}, {'bar'})
    self.assertEqual(key2, key2)
    self.assertNotEqual(key, key2)
    self.assertNotEqual(key2, key)

  def test_offsets_as_beam_key(self):
    inputs = [
        (xbeam.Key({'x': 0, 'y': 1}), 1),
        (xbeam.Key({'x': 0, 'y': 2}), 2),
        (xbeam.Key({'y': 1, 'x': 0}), 3),
    ]
    expected = [
        (xbeam.Key({'x': 0, 'y': 1}), [1, 3]),
        (xbeam.Key({'x': 0, 'y': 2}), [2]),
    ]
    actual = inputs | beam.GroupByKey()
    self.assertEqual(actual, expected)

  def test_vars_as_beam_key(self):
    inputs = [
        (xbeam.Key(vars={'foo'}), 1),
        (xbeam.Key(vars={'bar'}), 2),
        (xbeam.Key(vars={'foo'}), 3),
    ]
    expected = [
        (xbeam.Key(vars={'foo'}), [1, 3]),
        (xbeam.Key(vars={'bar'}), [2]),
    ]
    actual = inputs | beam.GroupByKey()
    self.assertEqual(actual, expected)


class TestOffsetsToSlices(test_util.TestCase):

  def test_offsets_to_slices(self):
    offsets = {'x': 0, 'y': 10}

    expected = {'x': slice(0, 5, 1), 'y': slice(10, 20, 1)}
    slices = core.offsets_to_slices(offsets, {'x': 5, 'y': 10})
    self.assertEqual(slices, expected)

    expected = {
        'x': slice(0, 5, 1),
        'y': slice(10, 20, 1),
        'extra_key': slice(0, 100, 1),
    }
    slices = core.offsets_to_slices(
        offsets, {'x': 5, 'y': 10, 'extra_key': 100}
    )
    self.assertEqual(slices, expected)

    with self.assertRaises(ValueError):
      core.offsets_to_slices(offsets, {'y': 10})

  def test_offsets_to_slices_base(self):
    offsets = {'x': 100, 'y': 210}

    base = {'x': 100, 'y': 200}
    expected = {'x': slice(0, 5, 1), 'y': slice(10, 20, 1)}
    slices = core.offsets_to_slices(offsets, {'x': 5, 'y': 10}, base=base)
    self.assertEqual(slices, expected)

    base = {'x': 100}
    expected = {'x': slice(0, 5, 1), 'y': slice(210, 220, 1)}
    slices = core.offsets_to_slices(offsets, {'x': 5, 'y': 10}, base=base)
    self.assertEqual(slices, expected)


class DatasetToChunksTest(test_util.TestCase):

  def test_iter_chunk_keys(self):
    actual = list(core.iter_chunk_keys({'x': (0, 3), 'y': (0, 2, 4)}))
    expected = [
        xbeam.Key({'x': 0, 'y': 0}),
        xbeam.Key({'x': 0, 'y': 2}),
        xbeam.Key({'x': 0, 'y': 4}),
        xbeam.Key({'x': 3, 'y': 0}),
        xbeam.Key({'x': 3, 'y': 2}),
        xbeam.Key({'x': 3, 'y': 4}),
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
        ValueError,
        'sum of provided chunks does not match',
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
        (xbeam.Key({'x': 0}), dataset.head(x=3)),
        (xbeam.Key({'x': 3}), dataset.tail(x=3)),
    ]
    actual = test_util.EagerPipeline() | xbeam.DatasetToChunks(
        dataset.chunk({'x': 3})
    )
    self.assertIdenticalChunks(actual, expected)

    actual = test_util.EagerPipeline() | xbeam.DatasetToChunks(
        dataset.chunk({'x': 3}), num_threads=2
    )
    self.assertIdenticalChunks(actual, expected)

    actual = test_util.EagerPipeline() | xbeam.DatasetToChunks(
        dataset, chunks={'x': 3}
    )
    self.assertIdenticalChunks(actual, expected)

    actual = test_util.EagerPipeline() | xbeam.DatasetToChunks(
        dataset.chunk({'x': 3}), shard_keys_threshold=1
    )
    self.assertIdenticalChunks(actual, expected)

  def test_dataset_to_chunks_whole(self):
    dataset = xarray.Dataset({'foo': ('x', np.arange(6))})
    expected = [(xbeam.Key({'x': 0}), dataset)]
    actual = test_util.EagerPipeline() | xbeam.DatasetToChunks(
        dataset, chunks={'x': -1}
    )
    self.assertIdenticalChunks(actual, expected)

    actual = test_util.EagerPipeline() | xbeam.DatasetToChunks(
        dataset, chunks={}
    )
    expected = [(xbeam.Key({'x': 0}), dataset)]
    self.assertIdenticalChunks(actual, expected)

  def test_dataset_to_chunks_vars(self):
    dataset = xarray.Dataset(
        {
            'foo': ('x', np.arange(6)),
            'bar': ('x', -np.arange(6)),
        }
    )
    expected = [
        (xbeam.Key({'x': 0}, {'foo'}), dataset.head(x=3)[['foo']]),
        (xbeam.Key({'x': 0}, {'bar'}), dataset.head(x=3)[['bar']]),
        (xbeam.Key({'x': 3}, {'foo'}), dataset.tail(x=3)[['foo']]),
        (xbeam.Key({'x': 3}, {'bar'}), dataset.tail(x=3)[['bar']]),
    ]
    actual = test_util.EagerPipeline() | xbeam.DatasetToChunks(
        dataset, chunks={'x': 3}, split_vars=True
    )
    self.assertIdenticalChunks(actual, expected)

  @parameterized.parameters(
      {'shard_keys_threshold': 1},
      {'shard_keys_threshold': 2},
      {'shard_keys_threshold': 10},
  )
  def test_dataset_to_chunks_split_with_different_dims(
      self, shard_keys_threshold
  ):
    dataset = xarray.Dataset(
        {
            'foo': (('x', 'y'), np.array([[1, 2, 3], [4, 5, 6]])),
            'bar': ('x', np.array([1, 2])),
            'baz': ('z', np.array([1, 2, 3])),
        }
    )
    expected = [
        (xbeam.Key({'x': 0, 'y': 0}, {'foo'}), dataset[['foo']].head(x=1)),
        (xbeam.Key({'x': 0}, {'bar'}), dataset[['bar']].head(x=1)),
        (xbeam.Key({'x': 1, 'y': 0}, {'foo'}), dataset[['foo']].tail(x=1)),
        (xbeam.Key({'x': 1}, {'bar'}), dataset[['bar']].tail(x=1)),
        (xbeam.Key({'z': 0}, {'baz'}), dataset[['baz']]),
    ]
    actual = test_util.EagerPipeline() | xbeam.DatasetToChunks(
        dataset,
        chunks={'x': 1},
        split_vars=True,
        shard_keys_threshold=shard_keys_threshold,
    )
    self.assertIdenticalChunks(actual, expected)

  def test_dataset_to_chunks_empty(self):
    dataset = xarray.Dataset()
    expected = [(xbeam.Key({}), dataset)]
    actual = test_util.EagerPipeline() | xbeam.DatasetToChunks(dataset)
    self.assertIdenticalChunks(actual, expected)

  def test_task_count(self):
    dataset = xarray.Dataset(
        {
            'foo': (('x', 'y'), np.zeros((3, 6))),
            'bar': ('x', np.zeros(3)),
            'baz': ('z', np.zeros(10)),
        }
    )

    to_chunks = xbeam.DatasetToChunks(dataset, chunks={'x': 1})
    self.assertEqual(to_chunks._task_count(), 3)

    to_chunks = xbeam.DatasetToChunks(dataset, chunks={'x': 1}, split_vars=True)
    self.assertEqual(to_chunks._task_count(), 7)

    to_chunks = xbeam.DatasetToChunks(dataset, chunks={'y': 1}, split_vars=True)
    self.assertEqual(to_chunks._task_count(), 8)

    to_chunks = xbeam.DatasetToChunks(dataset, chunks={'z': 1}, split_vars=True)
    self.assertEqual(to_chunks._task_count(), 12)


class ValidateEachChunkTest(test_util.TestCase):

  def test_unmatched_dimension_raises_error(self):
    dataset = xarray.Dataset({'foo': ('x', np.arange(6))})
    with self.assertRaises(ValueError) as e:
      ([(xbeam.Key({'x': 0, 'y': 0}), dataset)] | xbeam.ValidateEachChunk())
    self.assertIn(
        "Key offset(s) 'y' in Key(offsets={'x': 0, 'y': 0}, vars=None) not "
        'found in Dataset dimensions',
        e.exception.args[0],
    )

  def test_unmatched_variables_raises_error(self):
    dataset = xarray.Dataset({'foo': ('x', np.arange(6))})
    with self.assertRaises(ValueError) as e:
      ([(xbeam.Key({'x': 0}, {'bar'}), dataset)] | xbeam.ValidateEachChunk())
    self.assertIn(
        "Key var(s) 'bar' in Key(offsets={'x': 0}, vars={'bar'}) not found in "
        'Dataset data variables',
        e.exception.args[0],
    )

  def test_validate_chunks_compose_in_pipeline(self):
    dataset = xarray.Dataset({'foo': ('x', np.arange(6))})
    expected = [(xbeam.Key({'x': 0}), dataset)]
    actual = (
        test_util.EagerPipeline()
        | xbeam.DatasetToChunks(dataset, chunks={'x': -1})
        | xbeam.ValidateEachChunk()
    )
    self.assertIdenticalChunks(actual, expected)


if __name__ == '__main__':
  absltest.main()
