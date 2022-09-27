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
"""Tests for era5_climatology."""

from absl.testing import absltest
from absl.testing import flagsaver
import xarray

from . import era5_climatology
from xarray_beam._src import test_util


class Era5ClimatologyTest(test_util.TestCase):

  def test(self):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = test_util.dummy_era5_surface_dataset(times=90 * 24, freq='1H')
    input_ds.chunk({'time': 31}).to_zarr(input_path)

    expected = input_ds.groupby('time.month').apply(
        lambda x: x.groupby('time.hour').mean('time')
    )

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
    ):
      era5_climatology.main([])

    actual = xarray.open_zarr(output_path)
    xarray.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
  absltest.main()
