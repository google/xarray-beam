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
from xarray_beam._src import test_util


# pylint: disable=expression-not-assigned
# pylint: disable=pointless-statement


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
