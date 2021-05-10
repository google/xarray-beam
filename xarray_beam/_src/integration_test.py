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
"""Integration tests for Xarray-Beam."""
from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam
from xarray_beam._src import test_util


# pylint: disable=expression-not-assigned


class IntegrationTest(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'eager', 'template_method': 'eager'},
      {'testcase_name': 'lazy', 'template_method': 'lazy'},
      {'testcase_name': 'infer', 'template_method': 'infer'}
  )
  def test_rechunk_zarr_to_zarr(self, template_method):
    src_dir = self.create_tempdir('source').full_path
    dest_dir = self.create_tempdir('destination').full_path

    source_chunks = {'t': 1, 'x': 100, 'y': 120}
    target_chunks = {'t': -1, 'x': 20, 'y': 20}

    rs = np.random.RandomState(0)
    raw_data = rs.randint(2**30, size=(60, 100, 120))  # 5.76 MB
    dataset = xarray.Dataset({'foo': (('t', 'x', 'y'), raw_data)})
    dataset.chunk(source_chunks).to_zarr(src_dir, consolidated=True)

    on_disk = xarray.open_zarr(src_dir, consolidated=True)
    on_disk_chunked = on_disk.chunk(target_chunks)
    with beam.Pipeline('DirectRunner') as pipeline:
      # make template
      if template_method == 'eager':
        target_template = on_disk_chunked
      elif template_method == 'lazy':
        target_template = beam.pvalue.AsSingleton(
            pipeline | beam.Create([on_disk_chunked])
        )
      elif template_method == 'infer':
        target_template = None
      # run pipeline
      (
          pipeline
          | xarray_beam.DatasetToChunks(on_disk)
          | xarray_beam.Rechunk(
              on_disk.sizes, source_chunks, target_chunks,
              itemsize=8,
              max_mem=10_000_000,  # require two stages
          )
          | xarray_beam.ChunksToZarr(dest_dir, target_template)
      )
    roundtripped = xarray.open_zarr(dest_dir, consolidated=True, chunks=False)

    xarray.testing.assert_identical(roundtripped, dataset)


if __name__ == '__main__':
  absltest.main()
