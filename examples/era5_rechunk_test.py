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
"""Tests for era5_rechunk."""

from absl.testing import absltest
from absl.testing import flagsaver
import xarray

from . import era5_rechunk
from xarray_beam._src import test_util


class Era5RechunkTest(test_util.TestCase):

  def test(self):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = test_util.dummy_era5_surface_dataset(times=365)
    input_ds.chunk({'time': 31}).to_zarr(input_path)

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
    ):
      era5_rechunk.main([])

    output_ds = xarray.open_zarr(output_path)
    self.assertEqual(
        {k: v[0] for k, v in output_ds.chunks.items()},
        {'latitude': 5, 'longitude': 5, 'time': 365}
    )
    xarray.testing.assert_identical(input_ds, output_ds)


if __name__ == '__main__':
  absltest.main()
