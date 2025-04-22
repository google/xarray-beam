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
import re

from absl.testing import absltest
from absl.testing import parameterized
import dask.array as da
import numpy as np
import pandas as pd
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
    template = xbeam.make_template(source)
    self.assertEqual(list(template.data_vars), ['foo', 'bar'])
    self.assertEqual(template.chunks, {'x': (3,)})
    self.assertEqual(template.sizes, {'x': 3})
    with self.assertRaisesRegex(
        ValueError, 'cannot compute array values of xarray.Dataset objects'
    ):
      template.compute()

  def test_make_template_lazy_vars_on_numpy(self):
    source = xarray.Dataset(
        {
            'foo': ('x', np.ones(3)),
            'bar': ('x', np.ones(3)),
        },
    )
    template = xbeam.make_template(source, lazy_vars={'foo'})
    self.assertEqual(template.foo.chunks, ((3,),))
    self.assertIsNone(template.bar.chunks)

  def test_make_template_lazy_vars_on_dask(self):
    source = xarray.Dataset(
        {
            'foo': ('x', np.ones(3)),
            'bar': ('x', np.ones(3)),
        },
    ).chunk({'x': 2})
    template = xbeam.make_template(source, lazy_vars={'foo'})
    self.assertEqual(template.foo.chunks, ((3,),))  # one chunk
    self.assertIsInstance(template.bar.data, np.ndarray)  # computed

  def test_make_template_from_chunked(self):
    source = xarray.Dataset(
        {
            'foo': ('x', da.ones(3)),
            'bar': ('x', np.ones(3)),
        },
    )
    template = xbeam._src.zarr._make_template_from_chunked(source)
    self.assertEqual(template.foo.chunks, ((3,),))
    self.assertIsNone(template.bar.chunks)

  def test_replace_template_dims_with_coords(self):
    source = xarray.Dataset(
        {'foo': (('x', 'y'), np.zeros((1, 2)))},
        coords={'x': [0], 'y': [10, 20]},
    )
    template = xbeam.make_template(source)
    new_x_coords = pd.date_range('2000-01-01', periods=5)
    new_template = xbeam.replace_template_dims(template, x=new_x_coords)

    self.assertEqual(new_template.sizes, {'x': 5, 'y': 2})
    expected_x_coord = xarray.DataArray(
        new_x_coords, dims='x', coords={'x': new_x_coords}
    )
    xarray.testing.assert_equal(new_template.x, expected_x_coord)
    xarray.testing.assert_equal(new_template.y, source.y)  # Unchanged coord
    self.assertEqual(new_template.foo.shape, (5, 2))
    self.assertIsInstance(new_template.foo.data, da.Array)  # Still lazy

  def test_replace_template_dims_with_size(self):
    source = xarray.Dataset(
        {'foo': (('x', 'y'), np.zeros((1, 2)))},
        coords={'x': [0], 'y': [10, 20]},
    )
    template = xbeam.make_template(source)
    new_template = xbeam.replace_template_dims(template, x=10)

    self.assertEqual(new_template.sizes, {'x': 10, 'y': 2})
    self.assertNotIn(
        'x', new_template.coords
    )  # Coord is dropped when replaced by size
    xarray.testing.assert_equal(new_template.y, source.y)
    self.assertEqual(new_template.foo.shape, (10, 2))
    self.assertIsInstance(new_template.foo.data, da.Array)

  def test_replace_template_dims_multiple(self):
    source = xarray.Dataset(
        {'foo': (('x', 'y'), np.zeros((1, 2)))},
        coords={'x': [0], 'y': [10, 20]},
    )
    template = xbeam.make_template(source)
    new_x_coords = pd.date_range('2000-01-01', periods=5)
    new_template = xbeam.replace_template_dims(template, x=new_x_coords, y=3)

    self.assertEqual(new_template.sizes, {'x': 5, 'y': 3})
    expected_x_coord = xarray.DataArray(
        new_x_coords, dims='x', coords={'x': new_x_coords}
    )
    xarray.testing.assert_equal(new_template.x, expected_x_coord)
    self.assertNotIn('y', new_template.coords)
    self.assertEqual(new_template.foo.shape, (5, 3))
    self.assertIsInstance(new_template.foo.data, da.Array)

  def test_replace_template_dims_multiple_vars(self):
    source = xarray.Dataset(
        {
            'foo': (('x', 'y'), np.zeros((1, 2))),
            'bar': ('x', np.zeros(1)),
            'baz': ('z', np.zeros(3)),  # Unrelated dim
        },
        coords={'x': [0], 'y': [10, 20], 'z': [1, 2, 3]},
    )
    template = xbeam.make_template(source)
    new_template = xbeam.replace_template_dims(template, x=5)

    self.assertEqual(new_template.sizes, {'x': 5, 'y': 2, 'z': 3})
    self.assertNotIn('x', new_template.coords)
    xarray.testing.assert_equal(new_template.y, source.y)
    xarray.testing.assert_equal(new_template.z, source.z)
    self.assertEqual(new_template.foo.shape, (5, 2))
    self.assertEqual(new_template.bar.shape, (5,))
    self.assertEqual(new_template.baz.shape, (3,))  # Unchanged var
    self.assertIsInstance(new_template.foo.data, da.Array)
    self.assertIsInstance(new_template.bar.data, da.Array)
    self.assertIsInstance(new_template.baz.data, da.Array)

  def test_replace_template_dims_error_on_non_template(self):
    source = xarray.Dataset({'foo': ('x', np.zeros(1))})  # Not a template
    with self.assertRaisesRegex(ValueError, 'is not chunked with Dask'):
      xbeam.replace_template_dims(source, x=5)

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
      with self.assertWarnsRegex(FutureWarning, 'No template provided'):
        inputs | xbeam.ChunksToZarr(temp_dir, template=None)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
    with self.subTest('with template'):
      temp_dir = self.create_tempdir().full_path
      inputs | xbeam.ChunksToZarr(temp_dir, chunked)
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
    with self.subTest('with template and needs_setup=False'):
      temp_dir = self.create_tempdir().full_path
      xbeam.setup_zarr(chunked, temp_dir)
      inputs | xbeam.ChunksToZarr(temp_dir, chunked, needs_setup=False)
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
      with self.assertWarnsRegex(FutureWarning, 'No template provided'):
        inputs | xbeam.ChunksToZarr(
            temp_dir, template=None, zarr_chunks=zarr_chunks
        )
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
      with self.assertWarnsRegex(FutureWarning, 'No template provided'):
        inputs | xbeam.ChunksToZarr(temp_dir, template=None)
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
      inputs | xbeam.ChunksToZarr(temp_dir, template=dataset.chunk())
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
    with self.subTest('split along partial key'):
      inputs = [(xbeam.Key({'x': 0}), dataset)]
      temp_dir = self.create_tempdir().full_path
      inputs | xbeam.SplitChunks({'x': 1}) | xbeam.ChunksToZarr(
          temp_dir, template=dataset.chunk({'x': 1})
      )
      result = xarray.open_zarr(temp_dir, consolidated=True)
      xarray.testing.assert_identical(dataset, result)
    with self.subTest('full key'):
      inputs = [(xbeam.Key({'x': 0, 'y': 0}), dataset)]
      temp_dir = self.create_tempdir().full_path
      inputs | xbeam.ChunksToZarr(temp_dir, template=dataset.chunk())
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
        ValueError, 'dataset must be chunked or chunks must be provided'
    ):
      test_util.EagerPipeline() | xbeam.DatasetToZarr(dataset, temp_dir)

  def test_validate_zarr_chunk_accepts_partial_key(self):
    dataset = xarray.Dataset(
        {'foo': (('x', 'y'), np.zeros((3, 2)))},
        coords={'x': np.arange(3), 'y': np.arange(2)},
    )
    # Should not raise an exception:
    xbeam.validate_zarr_chunk(
        key=xbeam.Key({'x': 0}),
        chunk=dataset,
        template=dataset.chunk(),
        zarr_chunks=None,
    )

  def test_to_zarr_wrong_multiple_error(self):
    ds = xarray.Dataset({'foo': ('x', np.arange(6))})
    inputs = [
        (xbeam.Key({'x': 3}), ds.tail(3)),
    ]
    temp_dir = self.create_tempdir().full_path
    with self.assertRaisesRegex(
        ValueError,
        (
            "chunk offset 3 along dimension 'x' is not a multiple of zarr "
            "chunks {'x': 4}"
        ),
    ):
      inputs | xbeam.ChunksToZarr(
          temp_dir, template=ds.chunk(4), zarr_chunks={'x': 4}
      )

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
      inputs | xbeam.ChunksToZarr(
          temp_dir, template=ds.chunk(), zarr_chunks={'x': 6}
      )
    with self.assertRaisesRegex(
        ValueError, 'chunk is smaller than zarr chunks'
    ):
      inputs | xbeam.ChunksToZarr(temp_dir, template=ds.chunk())

  def test_to_zarr_fixed_template(self):
    dataset = xarray.Dataset({'foo': ('x', np.arange(6))})
    template = dataset.chunk({'x': 3})
    inputs = [
        (xbeam.Key({'x': 0}), dataset.head(3)),
        (xbeam.Key({'x': 3}), dataset.tail(3)),
    ]
    temp_dir = self.create_tempdir().full_path
    chunks_to_zarr = xbeam.ChunksToZarr(temp_dir, template)
    self.assertEqual(chunks_to_zarr.template.chunks, {'x': (6,)})
    self.assertEqual(chunks_to_zarr.zarr_chunks, {'x': 3})
    inputs | chunks_to_zarr
    actual = xarray.open_zarr(temp_dir, consolidated=True)
    xarray.testing.assert_identical(actual, dataset)

  def test_infer_zarr_chunks(self):
    dataset = xarray.Dataset({'foo': ('x', np.arange(6))})

    chunks = xbeam._src.zarr._infer_zarr_chunks(dataset)
    self.assertEqual(chunks, {})

    chunks = xbeam._src.zarr._infer_zarr_chunks(dataset.chunk())
    self.assertEqual(chunks, {'x': 6})

    chunks = xbeam._src.zarr._infer_zarr_chunks(dataset.head(0).chunk())
    self.assertEqual(chunks, {'x': 0})

    chunks = xbeam._src.zarr._infer_zarr_chunks(dataset.chunk(3))
    self.assertEqual(chunks, {'x': 3})

    chunks = xbeam._src.zarr._infer_zarr_chunks(dataset.chunk(4))
    self.assertEqual(chunks, {'x': 4})

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "Zarr cannot handle inconsistent chunk sizes along dimension 'x': "
            '(2, 4)'
        ),
    ):
      xbeam._src.zarr._infer_zarr_chunks(dataset.chunk({'x': (2, 4)}))

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            "Zarr cannot handle inconsistent chunk sizes along dimension 'x': "
            '(3, 2, 1)'
        ),
    ):
      xbeam._src.zarr._infer_zarr_chunks(dataset.chunk({'x': (3, 2, 1)}))

  def test_chunks_to_zarr_docs_demo(self):
    # verify that the ChunksToChunk demo from our docs works
    data = np.random.RandomState(0).randn(2920, 25, 53)
    ds = xarray.Dataset({'temperature': (('time', 'lat', 'lon'), data)})
    chunks = {'time': 1000, 'lat': 25, 'lon': 53}
    temp_dir = self.create_tempdir().full_path
    (
        test_util.EagerPipeline()
        | xbeam.DatasetToChunks(ds, chunks)
        | xbeam.ChunksToZarr(
            temp_dir, template=xbeam.make_template(ds), zarr_chunks=chunks
        )
    )
    result = xarray.open_zarr(temp_dir)
    xarray.testing.assert_identical(result, ds)


if __name__ == '__main__':
  absltest.main()
