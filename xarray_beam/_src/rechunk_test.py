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
"""Tests for rechunk."""
import re
import textwrap

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import xarray
import xarray_beam
from xarray_beam._src import rechunk
from xarray_beam._src import test_util

# pylint: disable=expression-not-assigned


class RechunkTest(test_util.TestCase):

  def test_normalize_chunks(self):
    inputs = {'x': 3, 'y': 4}
    dim_sizes = {'x': 30, 'y': 20}
    actual = rechunk.normalize_chunks(inputs, dim_sizes)
    self.assertEqual(inputs, actual)

    inputs = {'x': (3, 3), 'y': 5 * (4,)}
    expected = {'x': 3, 'y': 4}
    actual = rechunk.normalize_chunks(inputs, dim_sizes)
    self.assertEqual(expected, actual)

    inputs = {'x': 3, 'y': -1}
    expected = {'x': 3, 'y': 20}
    actual = rechunk.normalize_chunks(inputs, dim_sizes)
    self.assertEqual(expected, actual)

    expected = {'x': 5}
    actual = rechunk.normalize_chunks({'x': 5}, {'x': 9})
    self.assertEqual(expected, actual)

  def test_normalize_chunks_errors(self):
    with self.assertRaisesRegex(
        ValueError, 'chunks for dimension x are not constant',
    ):
      rechunk.normalize_chunks({'x': (3, 4)}, {'x': 7})

    with self.assertRaisesRegex(
        ValueError,
        'all dimensions used in chunks must also have an indicated size',
    ):
      rechunk.normalize_chunks({'x': 10}, {'y': 10})

  def test_rechunking_plan(self):
    # this trivial case fits entirely into memory
    plan = rechunk.rechunking_plan(
        dim_sizes={'x': 10, 'y': 20},
        source_chunks={'x': 1, 'y': 20},
        target_chunks={'x': 10, 'y': 1},
        itemsize=1,
        max_mem=200,
    )
    expected = [{'x': 10, 'y': 20}] * 3
    self.assertEqual(plan, expected)

    # this harder case doesn't
    read_chunks, _, write_chunks = rechunk.rechunking_plan(
        dim_sizes={'t': 1000, 'x': 200, 'y': 300},
        source_chunks={'t': 1, 'x': 200, 'y': 300},
        target_chunks={'t': 1000, 'x': 20, 'y': 20},
        itemsize=8,
        max_mem=10_000_000,
    )
    self.assertGreater(read_chunks['t'], 1)
    self.assertEqual(read_chunks['x'], 200)
    self.assertEqual(read_chunks['y'], 300)
    self.assertEqual(write_chunks['t'], 1000)
    self.assertGreater(read_chunks['x'], 20)
    self.assertGreater(read_chunks['y'], 20)

  def test_consolidate_and_split_chunks(self):
    consolidated = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(0, 10))})),
        (xarray_beam.ChunkKey({'x': 10}),
         xarray.Dataset({'foo': ('x', np.arange(10, 20))})),
    ]
    split = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(0, 5))})),
        (xarray_beam.ChunkKey({'x': 5}),
         xarray.Dataset({'foo': ('x', np.arange(5, 10))})),
        (xarray_beam.ChunkKey({'x': 10}),
         xarray.Dataset({'foo': ('x', np.arange(10, 15))})),
        (xarray_beam.ChunkKey({'x': 15}),
         xarray.Dataset({'foo': ('x', np.arange(15, 20))})),
    ]
    with self.subTest('ConsolidateChunks'):
      actual = split | xarray_beam.ConsolidateChunks({'x': 10})
      self.assertIdenticalChunks(actual, consolidated)
    with self.subTest('SplitChunks'):
      actual = consolidated | xarray_beam.SplitChunks({'x': 5})
      self.assertIdenticalChunks(actual, split)

  def test_consolidate_and_split_uneven_chunks(self):
    consolidated = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(10))})),
    ]
    split = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(0, 4))})),
        (xarray_beam.ChunkKey({'x': 4}),
         xarray.Dataset({'foo': ('x', np.arange(4, 8))})),
        (xarray_beam.ChunkKey({'x': 8}),
         xarray.Dataset({'foo': ('x', np.arange(8, 10))})),
    ]
    with self.subTest('ConsolidateChunks'):
      actual = split | xarray_beam.ConsolidateChunks({'x': 10})
      self.assertIdenticalChunks(actual, consolidated)
    with self.subTest('SplitChunks'):
      actual = consolidated | xarray_beam.SplitChunks({'x': 4})
      self.assertIdenticalChunks(actual, split)

  def test_consolidate_and_split_only_some_dims(self):
    chunk_data = np.arange(0, 10).reshape(2, 5)
    split = [
        (xarray_beam.ChunkKey({'x': 0, 'y': 0}),
         xarray.Dataset({'foo': (('x', 'y'), chunk_data)})),
        (xarray_beam.ChunkKey({'x': 0, 'y': 5}),
         xarray.Dataset({'foo': (('x', 'y'), chunk_data + 10)})),
    ]
    all_data = np.concatenate([chunk_data, chunk_data + 10], axis=1)
    consolidated = [
        (xarray_beam.ChunkKey({'x': 0, 'y': 0}),
         xarray.Dataset({'foo': (('x', 'y'), all_data)})),
    ]
    with self.subTest('ConsolidateChunks'):
      actual = split | xarray_beam.ConsolidateChunks({'y': 10})
      self.assertIdenticalChunks(actual, consolidated)
    with self.subTest('SplitChunks'):
      actual = consolidated | xarray_beam.SplitChunks({'y': 5})
      self.assertIdenticalChunks(actual, split)

  def test_consolidate_with_minus_one_chunks(self):
    inputs = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(0, 10))})),
        (xarray_beam.ChunkKey({'x': 10}),
         xarray.Dataset({'foo': ('x', np.arange(10, 20))})),
    ]
    expected = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(20))})),
    ]
    actual = inputs | xarray_beam.ConsolidateChunks({'x': -1})
    self.assertIdenticalChunks(actual, expected)

  def test_consolidate_with_unchunked_vars(self):
    inputs = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(0, 10)), 'bar': 1})),
        (xarray_beam.ChunkKey({'x': 10}),
         xarray.Dataset({'foo': ('x', np.arange(10, 20)), 'bar': 1})),
    ]
    expected = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(20)), 'bar': 1})),
    ]
    actual = inputs | xarray_beam.ConsolidateChunks({'x': -1})
    self.assertIdenticalChunks(actual, expected)

    inconsistent_inputs = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(0, 10)), 'bar': 1})),
        (xarray_beam.ChunkKey({'x': 10}),
         xarray.Dataset({'foo': ('x', np.arange(10, 20)), 'bar': 2})),
    ]
    with self.assertRaisesRegex(
        ValueError,
        re.escape(textwrap.dedent("""
            combining nested dataset chunks with offsets {'x': [0, 10]} failed.
            Leading datasets along dimension 'x':
              <xarray.Dataset>
              Dimensions:  (x: 10)
              Dimensions without coordinates: x
              Data variables:
                  foo      (x) int64 0 1 2 3 4 5 6 7 8 9
                  bar      int64 1
              <xarray.Dataset>
              Dimensions:  (x: 10)
              Dimensions without coordinates: x
              Data variables:
                  foo      (x) int64 10 11 12 13 14 15 16 17 18 19
                  bar      int64 2
        """).strip())
    ):
      inconsistent_inputs | xarray_beam.ConsolidateChunks({'x': -1})

  @parameterized.parameters(
      dict(start=0, stop=20, multiple=-1, expected=[(0, 20)]),
      dict(start=0, stop=20, multiple=20, expected=[(0, 20)]),
      dict(start=0, stop=20, multiple=100, expected=[(0, 20)]),
      dict(start=0, stop=20, multiple=10, expected=[(0, 10), (10, 20)]),
      dict(start=0, stop=20, multiple=15, expected=[(0, 15), (15, 20)]),
      dict(
          start=0,
          stop=10,
          multiple=3,
          expected=[(0, 3), (3, 6), (6, 9), (9, 10)],
      ),
      dict(start=5, stop=10, multiple=3, expected=[(5, 6), (6, 9), (9, 10)]),
      dict(start=10, stop=20, multiple=12, expected=[(10, 12), (12, 20)]),
      dict(start=10, stop=20, multiple=100, expected=[(10, 20)]),
    )
  def test_split_chunk_bounds(self, start, stop, multiple, expected):
    actual = rechunk._split_chunk_bounds(start, stop, multiple)
    self.assertEqual(actual, expected)

  def test_split_uneven_chunks(self):
    inputs = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(0, 5))})),
        (xarray_beam.ChunkKey({'x': 5}),
         xarray.Dataset({'foo': ('x', np.arange(5, 10))})),
    ]
    expected = [
        (xarray_beam.ChunkKey({'x': 0}),
         xarray.Dataset({'foo': ('x', np.arange(0, 3))})),
        (xarray_beam.ChunkKey({'x': 3}),
         xarray.Dataset({'foo': ('x', np.arange(3, 5))})),
        (xarray_beam.ChunkKey({'x': 5}),
         xarray.Dataset({'foo': ('x', np.arange(5, 6))})),
        (xarray_beam.ChunkKey({'x': 6}),
         xarray.Dataset({'foo': ('x', np.arange(6, 9))})),
        (xarray_beam.ChunkKey({'x': 9}),
         xarray.Dataset({'foo': ('x', np.arange(9, 10))})),
    ]
    actual = inputs | xarray_beam.SplitChunks({'x': 3})
    self.assertIdenticalChunks(actual, expected)

  def test_in_memory_rechunk_success(self):
    inputs = [
        (xarray_beam.ChunkKey({'x': 100, 'y': 300}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[1, 2, 3]]))})),
        (xarray_beam.ChunkKey({'x': 101, 'y': 300}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[4, 5, 6]]))})),
    ]
    target_chunks = {'x': 2, 'y': 1}
    expected = [
        (xarray_beam.ChunkKey({'x': 100, 'y': 300}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[1], [4]]))})),
        (xarray_beam.ChunkKey({'x': 100, 'y': 301}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[2], [5]]))})),
        (xarray_beam.ChunkKey({'x': 100, 'y': 302}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[3], [6]]))})),
    ]
    actual = list(
        rechunk.in_memory_rechunk(inputs, target_chunks)
    )
    self.assertIdenticalChunks(actual, expected)

  def test_in_memory_rechunk_not_unique(self):
    ds_zeros = xarray.Dataset({'foo': ('x', [0])})
    inputs = [
        (xarray_beam.ChunkKey({'x': 0}), ds_zeros),
        (xarray_beam.ChunkKey({'x': 0}), ds_zeros),
    ]
    target_chunks = {'x': 2}
    with self.assertRaisesRegex(ValueError, 'chunk keys are not unique'):
      list(rechunk.in_memory_rechunk(inputs, target_chunks))

  def test_in_memory_rechunk_missing_keys(self):
    ds_zeros = xarray.Dataset({'foo': (('x', 'y'), [[0]])})
    inputs = [
        (xarray_beam.ChunkKey({'x': 0, 'y': 0}), ds_zeros),
        (xarray_beam.ChunkKey({'x': 1, 'y': 1}), ds_zeros),
    ]
    target_chunks = {'x': 2, 'y': 2}
    with self.assertRaisesRegex(
        ValueError, 'some expected chunk keys are missing',
    ):
      list(rechunk.in_memory_rechunk(inputs, target_chunks))

  def test_rechunk_stage(self):
    inputs = [
        (xarray_beam.ChunkKey({'x': 100, 'y': 300}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[1, 2, 3]]))})),
        (xarray_beam.ChunkKey({'x': 101, 'y': 300}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[4, 5, 6]]))})),
        (xarray_beam.ChunkKey({'x': 100, 'y': 303}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[10, 20, 30]]))})),
        (xarray_beam.ChunkKey({'x': 101, 'y': 303}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[40, 50, 60]]))})),
    ]
    expected = [
        (xarray_beam.ChunkKey({'x': 100, 'y': 300}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[1], [4]]))})),
        (xarray_beam.ChunkKey({'x': 100, 'y': 301}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[2], [5]]))})),
        (xarray_beam.ChunkKey({'x': 100, 'y': 302}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[3], [6]]))})),
        (xarray_beam.ChunkKey({'x': 100, 'y': 303}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[10], [40]]))})),
        (xarray_beam.ChunkKey({'x': 100, 'y': 304}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[20], [50]]))})),
        (xarray_beam.ChunkKey({'x': 100, 'y': 305}),
         xarray.Dataset({'foo': (('x', 'y'), np.array([[30], [60]]))})),
    ]
    actual = inputs | rechunk.RechunkStage(
        source_chunks={'x': 1, 'y': 3},
        target_chunks={'x': 2, 'y': 1},
    )
    self.assertIdenticalChunks(actual, expected)

  def test_rechunk_end_to_end(self):
    data = np.random.RandomState(0).randint(2 ** 30, size=(10, 20, 30))
    ds = xarray.Dataset({'foo': (('time', 'x', 'y'), data)})
    key = xarray_beam.ChunkKey({'time': 0, 'x': 0, 'y': 0})
    time_split = [(key, ds)] | xarray_beam.SplitChunks({'time': 1})
    space_split = [(key, ds)] | xarray_beam.SplitChunks({'x': 5, 'y': 5})
    with self.subTest('time-to-space'):
      actual = time_split | rechunk.Rechunk(
          dim_sizes=ds.sizes,
          source_chunks={'time': 1, 'x': 20, 'y': 30},
          target_chunks={'time': 10, 'x': 5, 'y': 5},
          itemsize=8,
          max_mem=10_000,
      )
      self.assertIdenticalChunks(actual, space_split)
    with self.subTest('space-to-time'):
      actual = space_split | rechunk.Rechunk(
          dim_sizes=ds.sizes,
          source_chunks={'time': 10, 'x': 5, 'y': 5},
          target_chunks={'time': 1, 'x': 20, 'y': 30},
          itemsize=8,
          max_mem=10_000,
      )
      self.assertIdenticalChunks(actual, time_split)

  def test_rechunk_not_all_dimensions(self):
    data = np.random.RandomState(0).randint(2 ** 30, size=(10, 20, 30))
    ds = xarray.Dataset({'foo': (('time', 'x', 'y'), data)})
    key = xarray_beam.ChunkKey({'x': 0, 'y': 0})
    y_split_with_time_key = (
        [(key | {'time': 0}, ds)] | xarray_beam.SplitChunks({'y': 3})
    )
    x_split = [(key, ds)] | xarray_beam.SplitChunks({'x': 2})
    actual = x_split | rechunk.Rechunk(
        dim_sizes=ds.sizes,
        source_chunks={'x': 2, 'y': -1},
        target_chunks={'x': -1, 'y': 3},
        itemsize=8,
        max_mem=10_000,
    )
    self.assertIdenticalChunks(actual, y_split_with_time_key)

    with self.assertRaisesRegex(
        ValueError,
        'source_chunks and target_chunks have different keys',
    ):
      rechunk.Rechunk(
          dim_sizes=ds.sizes,
          source_chunks={'x': 2},
          target_chunks={'y': 3},
          itemsize=8,
          max_mem=10_000,
      )

  @parameterized.parameters(
      dict(size=100, max_mem=50, source_chunks=15, target_chunks=10),
      dict(size=100, max_mem=50, source_chunks=10, target_chunks=15),
      dict(size=100, max_mem=50, source_chunks=12, target_chunks=15),
      dict(size=100, max_mem=20, source_chunks=5, target_chunks=7),
  )
  def test_rechunk_1d(self, size, max_mem, source_chunks, target_chunks):
    data = np.random.RandomState(0).randint(2 ** 30, size=(size,))
    ds = xarray.Dataset({'foo': ('x', data)})
    key = xarray_beam.ChunkKey({'x': 0})
    inputs = [(key, ds)] | xarray_beam.SplitChunks({'x': source_chunks})
    expected = [(key, ds)] | xarray_beam.SplitChunks({'x': target_chunks})
    actual = inputs | rechunk.Rechunk(
        dim_sizes=ds.sizes,
        source_chunks={'x': source_chunks},
        target_chunks={'x': target_chunks},
        itemsize=1,
        max_mem=max_mem,
    )
    self.assertIdenticalChunks(actual, expected)

  def test_rechunk_uneven_2d(self):
    data = np.random.RandomState(0).randint(2 ** 30, size=(100, 100))
    ds = xarray.Dataset({'foo': (('x', 'y'), data)})
    key = xarray_beam.ChunkKey({'x': 0, 'y': 0})
    inputs = [(key, ds)] | xarray_beam.SplitChunks({'x': 12})
    expected = [(key, ds)] | xarray_beam.SplitChunks({'y': 15})
    actual = inputs | rechunk.Rechunk(
        dim_sizes=ds.sizes,
        source_chunks={'x': 12, 'y': -1},
        target_chunks={'x': -1, 'y': 15},
        itemsize=1,
        max_mem=100*100//2,  # half the full size
    )
    self.assertIdenticalChunks(actual, expected)


if __name__ == '__main__':
  absltest.main()
