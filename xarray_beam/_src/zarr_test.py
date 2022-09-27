# pyformat: mode=midnight
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
import dask.array as da
import numpy as np
import xarray
import xarray_beam as xbeam
from xarray_beam._src import test_util


# pylint: disable=expression-not-assigned
# pylint: disable=pointless-statement


class DatasetToZarrTest(test_util.TestCase):

  def test_open_zarr(self):
    source_ds = xarray.Dataset(
        {'foo': ('x', da.arange(0, 60, 10, chunks=2))},
    )
    temp_dir = self.create_tempdir().full_path
    source_ds.to_zarr(temp_dir)
    roundtripped_ds, chunks = xbeam.open_zarr(temp_dir)
    xarray.testing.assert_identical(roundtripped_ds, source_ds)
    self.assertEqual(roundtripped_ds.chunks, {})
    self.assertEqual(chunks, {'x': 2})

  def test_open_zarr_inconsistent(self):
    source_ds = xarray.Dataset(
        {
            'foo': ('x', da.arange(0, 60, 10, chunks=2)),
            'bar': ('x', da.arange(0, 60, 10, chunks=3)),
        },
    )
    temp_dir = self.create_tempdir().full_path
    source_ds.to_zarr(temp_dir)
    with self.assertRaisesRegex(
        ValueError,
        "inconsistent chunk sizes on Zarr dataset for dimension 'x': {2, 3}",
    ):
      xbeam.open_zarr(temp_dir)

  def test_make_template(self):
    source = xarray.Dataset(
        {
            'foo': ('x', np.ones(3)),
            'bar': ('x', np.ones(3)),
        },
    )
    actual = xbeam.make_template(source)
    expected = source * 0
    xarray.testing.assert_identical(actual, expected)
    self.assertEqual(actual.chunks, {'x': (3,)})

  def test_chunks_to_zarr(self):
    dataset = xarray.Dataset(
        {'foo': ('x', np.arange(0, 60, 10))},
        coords={'x': np.arange(6)},
    )
    chunked = dataset.chunk()
    inputs = [
        (xbeam.Key({'x': 0}), dataset),
    ]
    with self.subTest('no template'):
      temp_dir = self.create_tempdir().full_path
      inputs | xbeam.ChunksToZarr(temp_dir)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
    with self.subTest('with template'):
      temp_dir = self.create_tempdir().full_path
      inputs | xbeam.ChunksToZarr(temp_dir, chunked)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
    with self.subTest('with zarr_chunks and with template'):
      temp_dir = self.create_tempdir().full_path
      zarr_chunks = {'x': 3}
      inputs | xbeam.ChunksToZarr(temp_dir, chunked, zarr_chunks)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
      self.assertEqual(result.chunks, {'x': (3, 3)})
    with self.subTest('with zarr_chunks and no template'):
      temp_dir = self.create_tempdir().full_path
      zarr_chunks = {'x': 3}
      inputs | xbeam.ChunksToZarr(temp_dir, zarr_chunks=zarr_chunks)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
      self.assertEqual(result.chunks, {'x': (3, 3)})

    temp_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(
        ValueError,
        'template does not have any variables chunked with Dask',
    ):
      xbeam.ChunksToZarr(temp_dir, dataset)

    temp_dir = self.create_tempdir().full_path
    template = chunked.assign_coords(x=np.zeros(6))
    with self.assertRaisesRegex(
        ValueError,
        'template and chunk indexes do not match',
    ):
      inputs | xbeam.ChunksToZarr(temp_dir, template)

    inputs2 = [
        (xbeam.Key({'x': 0}), dataset.expand_dims(z=[1, 2])),
    ]
    temp_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(
        ValueError,
        'unexpected new indexes found in chunk',
    ):
      inputs2 | xbeam.ChunksToZarr(temp_dir, template)

  def test_multiple_vars_chunks_to_zarr(self):
    dataset = xarray.Dataset(
        {
            'foo': ('x', np.arange(0, 60, 10)),
            'bar': ('x', -np.arange(6)),
        },
        coords={'x': np.arange(6)},
    )
    chunked = dataset.chunk()
    inputs = [
        (xbeam.Key({'x': 0}, {'foo'}), dataset[['foo']]),
        (xbeam.Key({'x': 0}, {'bar'}), dataset[['bar']]),
    ]
    with self.subTest('no template'):
      temp_dir = self.create_tempdir().full_path
      inputs | xbeam.ChunksToZarr(temp_dir)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
    with self.subTest('with template'):
      temp_dir = self.create_tempdir().full_path
      inputs | xbeam.ChunksToZarr(temp_dir, chunked)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)

  @parameterized.named_parameters(
      {
          'testcase_name': 'combined_coords',
          'coords': {'bar': (('x', 'y'), -np.arange(6).reshape(3, 2))},
      },
      {
          'testcase_name': 'separate_coords',
          'coords': {'x': np.arange(3), 'y': np.arange(2)},
      },
  )
  def test_2d_chunks_to_zarr(self, coords):
    dataset = xarray.Dataset(
        {'foo': (('x', 'y'), np.arange(0, 60, 10).reshape(3, 2))},
        coords=coords,
    )
    with self.subTest('partial key'):
      inputs = [(xbeam.Key({'x': 0}), dataset)]
      temp_dir = self.create_tempdir().full_path
      inputs | xbeam.ChunksToZarr(temp_dir)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
    with self.subTest('split along partial key'):
      inputs = [(xbeam.Key({'x': 0}), dataset)]
      temp_dir = self.create_tempdir().full_path
      inputs | xbeam.SplitChunks({'x': 1}) | xbeam.ChunksToZarr(temp_dir)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
    with self.subTest('full key'):
      inputs = [(xbeam.Key({'x': 0, 'y': 0}), dataset)]
      temp_dir = self.create_tempdir().full_path
      inputs | xbeam.ChunksToZarr(temp_dir)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)

  def test_dataset_to_zarr_simple(self):
    dataset = xarray.Dataset(
        {'foo': ('x', np.arange(0, 60, 10))},
        coords={'x': np.arange(6)},
        attrs={'meta': 'data'},
    )
    chunked = dataset.chunk({'x': 3})
    temp_dir = self.create_tempdir().full_path
    test_util.EagerPipeline() | xbeam.DatasetToZarr(chunked, temp_dir)
    actual = xarray.open_zarr(temp_dir, consolidated=True)
    xarray.testing.assert_identical(actual, dataset)

  def test_dataset_to_zarr_unchunked(self):
    dataset = xarray.Dataset(
        {'foo': ('x', np.arange(0, 60, 10))},
    )
    temp_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(
        ValueError, 'template does not have any variables chunked with Dask'
    ):
      test_util.EagerPipeline() | xbeam.DatasetToZarr(dataset, temp_dir)

  def test_validate_zarr_chunk_accepts_partial_key(self):
    dataset = xarray.Dataset(
        {'foo': (('x', 'y'), np.zeros((3, 2)))},
        coords={'x': np.arange(3), 'y': np.arange(2)},
    )
    # Should not raise an exception:
    xbeam._src.zarr._validate_zarr_chunk(
        key=xbeam.Key({'x': 0}),
        chunk=dataset,
        template=dataset.chunk(),
        zarr_chunks=None,
    )

  def test_to_zarr_wrong_multiple_error(self):
    inputs = [
        (xbeam.Key({'x': 3}), xarray.Dataset({'foo': ('x', np.arange(3, 6))})),
    ]
    temp_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(
        ValueError,
        "chunk offset 3 along dimension 'x' is not a multiple of zarr chunks "
        "{'x': 4}",
    ):
      inputs | xbeam.ChunksToZarr(temp_dir, zarr_chunks={'x': 4})

  def test_to_zarr_needs_consolidation_error(self):
    ds = xarray.Dataset({'foo': ('x', np.arange(6))})
    inputs = [
        (xbeam.Key({'x': 0}), ds.head(3)),
        (xbeam.Key({'x': 3}), ds.tail(3)),
    ]
    temp_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(
        ValueError, 'chunk is smaller than zarr chunks'
    ):
      inputs | xbeam.ChunksToZarr(temp_dir, zarr_chunks={'x': 6})
    with self.assertRaisesRegex(
        ValueError, 'chunk is smaller than zarr chunks'
    ):
      inputs | xbeam.ChunksToZarr(temp_dir, template=ds.chunk())


if __name__ == '__main__':
  absltest.main()
