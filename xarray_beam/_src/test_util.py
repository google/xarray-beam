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
"""Testing utilities for Xarray-Beam."""
import pickle
import tempfile

from absl.testing import parameterized
import apache_beam as beam
import numpy as np
import pandas as pd
import xarray

# pylint: disable=expression-not-assigned


def _write_pickle(value, path):
  with open(path, 'wb') as f:
    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)


class EagerPipeline:
  """A mock Beam pipeline for testing that returns lists of Python objects.

  Example usage:

    >>> EagerPipeline() | beam.Create([1, 2, 3]) | beam.Map(lambda x: x**2)
    [1, 4, 9]
  """

  def __or__(self, ptransform):
    with tempfile.NamedTemporaryFile() as f:
      with beam.Pipeline('DirectRunner') as pipeline:
        (
            pipeline
            | ptransform
            | beam.combiners.ToList()
            | beam.Map(_write_pickle, f.name)
        )
      pipeline.run()
      return pickle.load(f)


class TestCase(parameterized.TestCase):
  """TestCase for use in internal Xarray-Beam tests."""

  def _assert_chunks(self, array_assert_func, actual, expected):
    actual = dict(actual)
    expected = dict(expected)
    self.assertCountEqual(expected, actual, msg='inconsistent keys')
    for key in expected:
      array_assert_func(actual[key], expected[key])

  def assertIdenticalChunks(self, actual, expected):
    self._assert_chunks(xarray.testing.assert_identical, actual, expected)

  def assertAllCloseChunks(self, actual, expected):
    self._assert_chunks(xarray.testing.assert_allclose, actual, expected)


def dummy_era5_surface_dataset(
    variables=2,
    latitudes=73,
    longitudes=144,
    times=365 * 4,
    freq='6H',
):
  """A mock version of the Pangeo ERA5 surface reanalysis dataset."""
  # based on: gs://pangeo-era5/reanalysis/spatial-analysis
  dims = ('time', 'latitude', 'longitude')
  shape = (times, latitudes, longitudes)
  var_names = ['asn', 'd2m', 'e', 'mn2t', 'mx2t', 'ptype'][:variables]
  rng = np.random.default_rng(0)
  data_vars = {
      name: (dims, rng.normal(size=shape).astype(np.float32), {'var_index': i})
      for i, name in enumerate(var_names)
  }

  latitude = np.linspace(90, 90, num=latitudes)
  longitude = np.linspace(0, 360, num=longitudes, endpoint=False)
  time = pd.date_range('1979-01-01T00', periods=times, freq=freq)
  coords = {'time': time, 'latitude': latitude, 'longitude': longitude}

  return xarray.Dataset(data_vars, coords, {'global_attr': 'yes'})
