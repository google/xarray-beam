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
import xarray_beam as xbeam
from xarray_beam._src import test_util


# pylint: disable=expression-not-assigned


class IntegrationTest(test_util.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'eager_unified',
          'template_method': 'eager',
          'split_vars': False,
      },
      {
          'testcase_name': 'eager_split',
          'template_method': 'eager',
          'split_vars': True,
      },
      {
          'testcase_name': 'eager_unified_sharded',
          'template_method': 'eager',
          'split_vars': False,
          'shard_keys_threshold': 20,
      },
      {
          'testcase_name': 'eager_split_sharded',
          'template_method': 'eager',
          'split_vars': True,
          'shard_keys_threshold': 20,
      },
      {
          'testcase_name': 'lazy_unified',
          'template_method': 'lazy',
          'split_vars': False,
      },
      {
          'testcase_name': 'infer_unified',
          'template_method': 'infer',
          'split_vars': False,
      },
      {
          'testcase_name': 'infer_split',
          'template_method': 'infer',
          'split_vars': True,
      },
  )
  def test_rechunk_zarr_to_zarr(
      self, template_method, split_vars, shard_keys_threshold=1_000_000
  ):
    src_dir = self.create_tempdir('source').full_path
    dest_dir = self.create_tempdir('destination').full_path

    source_chunks = {'t': 1, 'x': 100, 'y': 120}
    target_chunks = {'t': -1, 'x': 20, 'y': 20}

    rs = np.random.RandomState(0)
    raw_data = rs.randint(2**30, size=(60, 100, 120))  # 5.76 MB
    dataset = xarray.Dataset(
        {
            'foo': (('t', 'x', 'y'), raw_data),
            'bar': (('t', 'x', 'y'), raw_data - 1),
        }
    )
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
          | xbeam.DatasetToChunks(
              on_disk,
              split_vars=split_vars,
              shard_keys_threshold=shard_keys_threshold,
          )
          | xbeam.Rechunk(
              on_disk.sizes,
              source_chunks,
              target_chunks,
              itemsize=8,
              max_mem=10_000_000,  # require two stages
          )
          | xbeam.ChunksToZarr(dest_dir, target_template)
      )
    roundtripped = xarray.open_zarr(dest_dir, consolidated=True, chunks=False)

    xarray.testing.assert_identical(roundtripped, dataset)

  @parameterized.named_parameters(
      {
          'testcase_name': 'unified_unsharded',
          'split_vars': False,
          'shard_keys_threshold': 1_000_000,
      },
      {
          'testcase_name': 'split_unsharded',
          'split_vars': True,
          'shard_keys_threshold': 1_000_000,
      },
      {
          'testcase_name': 'unified_sharded',
          'split_vars': False,
          'shard_keys_threshold': 3,
      },
      {
          'testcase_name': 'split_sharded',
          'split_vars': True,
          'shard_keys_threshold': 3,
      },
  )
  def test_dataset_to_zarr_with_irregular_variables(
      self, split_vars, shard_keys_threshold
  ):
    dataset = xarray.Dataset(
        {
            'volume1': (
                ('t', 'x', 'y', 'z1'),
                np.arange(240).reshape(10, 2, 3, 4),
            ),
            'volume2': (
                ('t', 'x', 'y', 'z2'),
                np.arange(300).reshape(10, 2, 3, 5),
            ),
            'surface': (('t', 'x', 'y'), np.arange(60).reshape(10, 2, 3)),
            'static': (('x', 'y'), np.arange(6).reshape(2, 3)),
        }
    )
    temp_dir = self.create_tempdir().full_path
    template = dataset.chunk()
    chunks = {'t': 1, 'z1': 2, 'z2': 3}
    (
        test_util.EagerPipeline()
        | xbeam.DatasetToChunks(
            dataset,
            chunks,
            split_vars=split_vars,
            shard_keys_threshold=shard_keys_threshold,
        )
        | xbeam.ChunksToZarr(temp_dir, template, chunks)
    )
    actual = xarray.open_zarr(temp_dir, consolidated=True)
    xarray.testing.assert_identical(actual, dataset)


if __name__ == '__main__':
  absltest.main()
