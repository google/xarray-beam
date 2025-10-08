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
"""Tests for xarray_beam._src.combiners."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import xarray
import xarray_beam as xbeam
from xarray_beam._src import combiners
from xarray_beam._src import test_util


# pylint: disable=expression-not-assigned
# pylint: disable=pointless-statement


class GetChunkIndexTest(test_util.TestCase):

  def test_1d(self):
    dims = ['x']
    chunks = {'x': 2}
    sizes = {'x': 8}
    keys = [xbeam.Key({'x': i}) for i in [0, 2, 4, 6]]
    expected_list = [0, 1, 2, 3]
    actual_list = [
        combiners._get_chunk_index(key, dims, chunks, sizes) for key in keys
    ]
    self.assertEqual(actual_list, expected_list)

  def test_2d(self):
    dims = ['x', 'y']
    chunks = {'x': 2, 'y': 3}
    sizes = {'x': 4, 'y': 6}
    keys = [
        xbeam.Key({'x': 0, 'y': 0}),
        xbeam.Key({'x': 0, 'y': 3}),
        xbeam.Key({'x': 2, 'y': 0}),
        xbeam.Key({'x': 2, 'y': 3}),
    ]
    expected_list = [0, 1, 2, 3]
    actual_list = [
        combiners._get_chunk_index(key, dims, chunks, sizes) for key in keys
    ]
    self.assertEqual(actual_list, expected_list)


class IndexToBinsTest(parameterized.TestCase):

  @parameterized.parameters(
      {'chunk_index': 0, 'bins_per_stage': (), 'expected': ()},
      {'chunk_index': 0, 'bins_per_stage': (4,), 'expected': (0,)},
      {'chunk_index': 1, 'bins_per_stage': (4,), 'expected': (1,)},
      {'chunk_index': 2, 'bins_per_stage': (4,), 'expected': (2,)},
      {'chunk_index': 3, 'bins_per_stage': (4,), 'expected': (3,)},
      {'chunk_index': 0, 'bins_per_stage': (2, 2), 'expected': (0, 0)},
      {'chunk_index': 1, 'bins_per_stage': (2, 2), 'expected': (1, 0)},
      {'chunk_index': 2, 'bins_per_stage': (2, 2), 'expected': (0, 1)},
      {'chunk_index': 3, 'bins_per_stage': (2, 2), 'expected': (1, 1)},
      {'chunk_index': 55, 'bins_per_stage': (10, 10), 'expected': (5, 5)},
  )
  def test_index_to_fanout_bins(self, chunk_index, bins_per_stage, expected):
    actual = combiners._index_to_fanout_bins(chunk_index, bins_per_stage)
    self.assertEqual(actual, expected)


class OptimalFanoutTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'dims': ['x'],
          'chunks': {'x': 1_000_000},
          'sizes': {'x': 1_000_000},
          'itemsize': 4,
          'expected': (),
      },
      {
          'dims': ['x'],
          'chunks': {'x': 10_000},
          'sizes': {'x': 1_000_000},
          'itemsize': 4,
          'expected': (100,),
      },
      {
          'dims': ['x'],
          'chunks': {'x': 1_000},
          'sizes': {'x': 1_000_000},
          'itemsize': 4,
          'expected': (32, 32),
      },
      {
          'dims': ['time'],
          'chunks': {'time': 100, 'x': 1000, 'y': 1000},
          'sizes': {'time': 100_000, 'x': 1000, 'y': 1000},
          'itemsize': 4,
          'expected': (4, 4, 4, 4, 4),
      },
      {
          'dims': ['time'],
          'chunks': {'time': 100, 'y': 1000, 'z': 1000},
          'sizes': {'time': 500, 'x': 1000, 'y': 1000},
          'itemsize': 4,
          'expected': (5,),
      },
      {
          'dims': ['time', 'x', 'y'],
          'chunks': {'time': 100, 'x': 1000, 'y': 1000},
          'sizes': {'time': 100_000, 'x': 1000, 'y': 1000},
          'itemsize': 4,
          'expected': (32, 32),
      },
  )
  def test_optimal_fanout_bins(self, dims, chunks, sizes, itemsize, expected):
    actual = combiners._optimal_fanout_bins(dims, chunks, sizes, itemsize)
    self.assertEqual(actual, expected)


class MultiStageMeanTest(test_util.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_fanout_pre_aggregate', (4,), True),
      ('no_fanout_no_pre_aggregate', (4,), False),
      ('with_fanout_pre_aggregate', (2, 2), True),
      ('with_fanout_no_pre_aggregate', (2, 2), False),
      ('with_too_big_fanout', (3, 2), None),
      ('with_three_stages', (1, 2, 2), None),
  )
  def test_multi_stage_mean(self, bins_per_stage, pre_aggregate):
    sizes = {'x': 8}
    chunks = {'x': 2}
    inputs = [
        (
            xbeam.Key({'x': 0}),
            xarray.Dataset({'z': ('x', np.array([0.0, 1.0]))}),
        ),
        (
            xbeam.Key({'x': 2}),
            xarray.Dataset({'z': ('x', np.array([2.0, 3.0]))}),
        ),
        (
            xbeam.Key({'x': 4}),
            xarray.Dataset({'z': ('x', np.array([4.0, 5.0]))}),
        ),
        (
            xbeam.Key({'x': 6}),
            xarray.Dataset({'z': ('x', np.array([6.0, 7.0]))}),
        ),
    ]
    transform = combiners.MultiStageMean(
        dims=['x'],
        skipna=True,
        dtype=None,
        chunks=chunks,
        sizes=sizes,
        itemsize=8,
        bins_per_stage=bins_per_stage,
        pre_aggregate=pre_aggregate,
    )
    expected = [(xbeam.Key({}), xarray.Dataset({'z': 3.5}))]
    actual = inputs | transform
    self.assertAllCloseChunks(actual, expected)

  def test_multi_stage_mean_no_combiner(self):
    sizes = {'x': 2, 'y': 2}
    chunks = {'x': 2, 'y': 2}
    inputs = [
        (
            xbeam.Key({'x': 0, 'y': 0}),
            xarray.Dataset({'z': (('x', 'y'), np.array([[0.0, 1.0]]))}),
        ),
        (
            xbeam.Key({'x': 1, 'y': 0}),
            xarray.Dataset({'z': (('x', 'y'), np.array([[2.0, 3.0]]))}),
        ),
    ]
    transform = combiners.MultiStageMean(
        dims=['y'],
        skipna=True,
        dtype=None,
        chunks=chunks,
        sizes=sizes,
        itemsize=8,
    )
    expected = [
        (
            xbeam.Key({'x': 0}),
            xarray.Dataset({'z': (('x',), np.array([0.5]))}),
        ),
        (
            xbeam.Key({'x': 1}),
            xarray.Dataset({'z': (('x',), np.array([2.5]))}),
        ),
    ]
    actual = inputs | transform
    self.assertAllCloseChunks(actual, expected)


class MeanTest(test_util.TestCase):

  def test_globally(self):
    nan = np.nan
    data_with_nans = np.array(
        [[1, 2, 3], [4, 5, nan], [6, nan, nan], [nan, nan, nan]]
    )
    dataset = xarray.Dataset({'foo': (('x', 'y'), data_with_nans)})
    inputs_x = [dataset.isel(x=i) for i in range(4)]
    inputs_y = [dataset.isel(y=i) for i in range(3)]

    with self.subTest('skipna-default'):
      expected = dataset.mean('y', skipna=True)
      (actual,) = inputs_y | xbeam.Mean.Globally()
      xarray.testing.assert_allclose(expected, actual)

    with self.subTest('skipna=True'):
      expected = dataset.mean('y', skipna=True)
      (actual,) = inputs_y | xbeam.Mean.Globally(skipna=True)
      xarray.testing.assert_allclose(expected, actual)

      expected = dataset.mean('x', skipna=True)
      (actual,) = inputs_x | xbeam.Mean.Globally(skipna=True)
      xarray.testing.assert_allclose(expected, actual)

    with self.subTest('skipna=False', skipna=False):
      expected = dataset.mean('y', skipna=False)
      (actual,) = inputs_y | xbeam.Mean.Globally(skipna=False)
      xarray.testing.assert_allclose(expected, actual)

      expected = dataset.mean('x', skipna=False)
      (actual,) = inputs_x | xbeam.Mean.Globally(skipna=False)
      xarray.testing.assert_allclose(expected, actual)

    with self.subTest('with-fanout'):
      expected = dataset.mean('y', skipna=True)
      (actual,) = inputs_y | xbeam.Mean.Globally(fanout=2)
      xarray.testing.assert_allclose(expected, actual)

  def test_dim_globally(self):
    inputs = [
        xarray.Dataset({'x': ('time', [1, 2])}),
        xarray.Dataset({'x': ('time', [3])}),
    ]
    expected = xarray.Dataset({'x': 2.0})
    (actual,) = inputs | xbeam.Mean.Globally(dim='time')
    xarray.testing.assert_allclose(expected, actual)

  def test_per_key(self):
    inputs = [
        (0, xarray.Dataset({'x': 1})),
        (0, xarray.Dataset({'x': 2})),
        (1, xarray.Dataset({'x': 3})),
        (1, xarray.Dataset({'x': 4})),
    ]
    expected = [
        (0, xarray.Dataset({'x': 1.5})),
        (1, xarray.Dataset({'x': 3.5})),
    ]
    actual = inputs | xbeam.Mean.PerKey()
    self.assertAllCloseChunks(actual, expected)

  def test_mean_1d(self):
    inputs = [
        (xbeam.Key({'x': 0}), xarray.Dataset({'y': ('x', [1, 2, 3])})),
        (xbeam.Key({'x': 3}), xarray.Dataset({'y': ('x', [4, 5, 6])})),
    ]
    expected = [
        (xbeam.Key({}), xarray.Dataset({'y': 3.5})),
    ]
    actual = inputs | xbeam.Mean('x')
    self.assertAllCloseChunks(actual, expected)
    actual = inputs | xbeam.Mean(['x'])
    self.assertAllCloseChunks(actual, expected)

  def test_mean_many(self):
    inputs = []
    for i in range(0, 100, 10):
      inputs.append(
          (xbeam.Key({'x': i}), xarray.Dataset({'y': ('x', i + np.arange(10))}))
      )
    expected = [
        (xbeam.Key({}), xarray.Dataset({'y': 49.5})),
    ]
    actual = inputs | xbeam.Mean('x', fanout=2)
    self.assertAllCloseChunks(actual, expected)

  def test_mean_nans(self):
    nan = np.nan
    data_with_nans = np.array(
        [[1, 2, 3], [4, 5, nan], [6, nan, nan], [nan, nan, nan]]
    )
    dataset = xarray.Dataset({'foo': (('x', 'y'), data_with_nans)})
    eager = test_util.EagerPipeline()
    chunks = eager | xbeam.DatasetToChunks(dataset, {'x': 1, 'y': 1})

    expected = eager | xbeam.DatasetToChunks(
        dataset.mean('y', skipna=False), {'x': 1}
    )
    actual = chunks | xbeam.Mean('y', skipna=False)
    self.assertAllCloseChunks(actual, expected)

    expected = eager | xbeam.DatasetToChunks(
        dataset.mean('y', skipna=True), {'x': 1}
    )
    actual = chunks | xbeam.Mean('y', skipna=True)
    self.assertAllCloseChunks(actual, expected)

    expected = eager | xbeam.DatasetToChunks(
        dataset.mean('x', skipna=True), {'y': 1}
    )
    actual = chunks | xbeam.Mean('x', skipna=True)
    self.assertAllCloseChunks(actual, expected)

    expected = eager | xbeam.DatasetToChunks(
        dataset.mean('x', skipna=False), {'y': 1}
    )
    actual = chunks | xbeam.Mean('x', skipna=False)
    self.assertAllCloseChunks(actual, expected)

  def test_mean_2d(self):
    inputs = [
        (xbeam.Key({'y': 0}), xarray.Dataset({'z': (('x', 'y'), [[1, 2, 3]])})),
        (xbeam.Key({'y': 3}), xarray.Dataset({'z': (('x', 'y'), [[4, 5, 6]])})),
    ]

    expected = [
        (xbeam.Key({'y': 0}), xarray.Dataset({'z': ('y', [1, 2, 3])})),
        (xbeam.Key({'y': 3}), xarray.Dataset({'z': ('y', [4, 5, 6])})),
    ]
    actual = inputs | xbeam.Mean('x')
    self.assertAllCloseChunks(actual, expected)

    expected = [
        (xbeam.Key({}), xarray.Dataset({'z': (('x',), [3.5])})),
    ]
    actual = inputs | xbeam.Mean('y')
    self.assertAllCloseChunks(actual, expected)

    expected = [
        (xbeam.Key({}), xarray.Dataset({'z': 3.5})),
    ]
    actual = inputs | xbeam.Mean(['x', 'y'])
    self.assertAllCloseChunks(actual, expected)


if __name__ == '__main__':
  absltest.main()
