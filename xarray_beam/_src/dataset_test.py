# Copyright 2023 Google LLC
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
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam as xbeam
from xarray_beam._src import dataset as xbeam_dataset
from xarray_beam._src import test_util


class ToHumanSizeTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='zero', size=0, expected='0B'),
      dict(testcase_name='one_byte', size=1, expected='1B'),
      dict(testcase_name='nine_bytes', size=9, expected='9B'),
      dict(testcase_name='ten_bytes', size=10, expected='10B'),
      dict(testcase_name='ninety_nine_bytes', size=99, expected='99B'),
      dict(testcase_name='one_hundred_bytes', size=100, expected='100B'),
      dict(testcase_name='almost_one_kb', size=999, expected='999B'),
      dict(testcase_name='one_kb', size=1000, expected='1.0kB'),
      dict(testcase_name='round_to_10_kb', size=9996, expected='10kB'),
      dict(testcase_name='100_mb', size=10**8, expected='100MB'),
      dict(testcase_name='one_mb', size=10**6, expected='1.0MB'),
      dict(testcase_name='one_gb', size=10**9, expected='1.0GB'),
      dict(testcase_name='one_tb', size=10**12, expected='1.0TB'),
      dict(testcase_name='one_pb', size=10**15, expected='1.0PB'),
      dict(testcase_name='one_eb', size=10**18, expected='1.0EB'),
      dict(testcase_name='one_thousand_eb', size=10**21, expected='1000EB'),
      dict(testcase_name='ten_thousand_eb', size=10**22, expected='10000EB'),
  )
  def test_to_human_size(self, size, expected):
    self.assertEqual(xbeam_dataset._to_human_size(size), expected)


class DatasetTest(test_util.TestCase):

  def test_repr(self):
    ds = xarray.Dataset({'foo': ('x', np.arange(10))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 5})
    self.assertRegex(
        repr(beam_ds),
        re.escape(
            '<xarray_beam.Dataset>\n'
            'PTransform: <DatasetToChunks>\n'
            'Chunks:     40B (x: 5, split_vars=False)\n'
            'Template:   80B (2 chunks)\n'
            '    Dimensions:'
        ).replace('DatasetToChunks', 'DatasetToChunks.*'),
    )

  def test_from_xarray(self):
    ds = xarray.Dataset({'foo': ('x', np.arange(10))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 5})
    self.assertIsInstance(beam_ds, xbeam.Dataset)
    self.assertEqual(beam_ds.sizes, {'x': 10})
    self.assertEqual(beam_ds.template.keys(), {'foo'})
    self.assertEqual(beam_ds.chunks, {'x': 5})
    self.assertFalse(beam_ds.split_vars)
    self.assertEqual(beam_ds.bytes_per_chunk, 40)
    self.assertEqual(beam_ds.chunk_count, 2)
    self.assertRegex(beam_ds.ptransform.label, r'^from_xarray_\d+$')
    expected = [
        (xbeam.Key({'x': 0}), ds.head(x=5)),
        (xbeam.Key({'x': 5}), ds.tail(x=5)),
    ]
    actual = test_util.EagerPipeline() | beam_ds.ptransform
    self.assertIdenticalChunks(expected, actual)

  def test_from_xarray_minus_chunks_missing(self):
    ds = xarray.Dataset({'foo': ('x', np.arange(10))})
    beam_ds = xbeam.Dataset.from_xarray(ds, chunks={})
    self.assertEqual(beam_ds.chunks, {'x': 10})

  def test_from_xarray_minus_one_chunks(self):
    ds = xarray.Dataset({'foo': ('x', np.arange(10))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': -1})
    self.assertEqual(beam_ds.chunks, {'x': 10})

  def test_collect_with_direct_runner(self):
    ds = xarray.Dataset({'foo': ('x', np.arange(10))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 5})
    collected = beam_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(ds, collected)

  @parameterized.parameters(
      dict(split_vars=False),
      dict(split_vars=True),
  )
  def test_from_zarr(self, split_vars):
    temp_dir = self.create_tempdir().full_path
    ds = xarray.Dataset({'foo': ('x', np.arange(10))})
    ds.chunk({'x': 5}).to_zarr(temp_dir)

    beam_ds = xbeam.Dataset.from_zarr(temp_dir, split_vars)

    self.assertRegex(beam_ds.ptransform.label, r'^from_zarr_\d+$')
    self.assertEqual(beam_ds.chunks, {'x': 5})
    self.assertEqual(beam_ds.split_vars, split_vars)

    collected = beam_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(ds, collected)

  def test_to_zarr(self):
    temp_dir = self.create_tempdir().full_path
    ds = xarray.Dataset({'foo': ('x', np.arange(12))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 6})

    with self.subTest('same_chunks'):
      to_zarr = beam_ds.to_zarr(temp_dir)
      self.assertRegex(to_zarr.label, r'^from_xarray_\d+|to_zarr_\d+$')
      with beam.Pipeline() as p:
        p |= to_zarr
      opened, chunks = xbeam.open_zarr(temp_dir)
      xarray.testing.assert_identical(ds, opened)
      self.assertEqual(chunks, {'x': 6})

    with self.subTest('smaller_chunks'):
      temp_dir = self.create_tempdir().full_path
      with beam.Pipeline() as p:
        p |= beam_ds.to_zarr(temp_dir, zarr_chunks={'x': 3})
      opened, chunks = xbeam.open_zarr(temp_dir)
      xarray.testing.assert_identical(ds, opened)
      self.assertEqual(chunks, {'x': 3})

    with self.subTest('larger_chunks'):
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          "cannot write a dataset with chunks {'x': 6} to Zarr with chunks "
          "{'x': 9}, which do not divide evenly into chunks",
      ):
        beam_ds.to_zarr(temp_dir, zarr_chunks={'x': 9})

    with self.subTest('shards_without_chunks'):
      with self.assertRaisesWithLiteralMatch(
          ValueError, 'cannot supply zarr_shards without zarr_chunks'
      ):
        beam_ds.to_zarr(temp_dir, zarr_shards={'x': -1})

  def test_to_zarr_shards(self):
    temp_dir = self.create_tempdir().full_path
    ds = xarray.Dataset({'foo': ('x', np.arange(12))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 6})

    with self.subTest('same_shards_as_chunks'):
      with beam.Pipeline() as p:
        p |= beam_ds.to_zarr(
            temp_dir, zarr_chunks={'x': 3}, zarr_shards={'x': 6}, zarr_format=3
        )
      opened, chunks = xbeam.open_zarr(temp_dir)
      xarray.testing.assert_identical(ds, opened)
      self.assertEqual(chunks, {'x': 3})
      self.assertEqual(opened['foo'].encoding['shards'], (6,))

    with self.subTest('larger_shards'):
      with self.assertRaisesWithLiteralMatch(
          ValueError,
          "cannot write a dataset with chunks {'x': 6} to Zarr with shards "
          "{'x': 9}, which do not divide evenly into shards",
      ):
        beam_ds.to_zarr(
            temp_dir, zarr_chunks={'x': 3}, zarr_shards={'x': 9}, zarr_format=3
        )

  def test_to_zarr_default_chunks(self):
    temp_dir = self.create_tempdir().full_path
    ds = xarray.Dataset({'foo': (('x', 'y'), np.arange(20).reshape(10, 2))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 4})
    to_zarr = beam_ds.to_zarr(temp_dir, zarr_chunks={'x': 2})

    self.assertRegex(to_zarr.label, r'^from_xarray_\d+|to_zarr_\d+$')
    with beam.Pipeline() as p:
      p |= to_zarr
    opened, chunks = xbeam.open_zarr(temp_dir)
    xarray.testing.assert_identical(ds, opened)
    self.assertEqual(chunks, {'x': 2, 'y': 2})

  @parameterized.named_parameters(
      dict(testcase_name='getitem', call=lambda x: x[['foo']]),
      dict(testcase_name='transpose', call=lambda x: x.transpose()),
  )
  def test_lazy_methods(self, call):
    ds = xarray.Dataset(
        {
            'foo': ('x', np.arange(10)),
            'bar': (('x', 'y'), -np.arange(20).reshape(10, 2)),
        },
        coords={'x': np.arange(0, 100, 10)},
    )
    expected = call(ds)

    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 5, 'y': 1})

    with self.subTest('chunks_from_dataset'):
      result = call(beam_ds)
      actual = result.collect_with_direct_runner()
      xarray.testing.assert_identical(expected, actual)

    with self.subTest('already_transformed'):
      result = beam_ds.map_blocks(lambda x: x).pipe(call)
      actual = result.collect_with_direct_runner()
      xarray.testing.assert_identical(expected, actual)

  def test_head(self):
    ds = xarray.Dataset({'foo': ('x', np.arange(10))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 5})

    head_ds = beam_ds.head(x=2)
    self.assertRegex(head_ds.ptransform.label, r'^from_xarray_\d+|head_\d+$')
    expected = ds.head(x=2)
    actual = head_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(expected, actual)

    with self.assertRaisesRegex(
        ValueError,
        re.escape(
            'head() is only supported on untransformed datasets, with '
            'ptransform=DatasetToChunks. This dataset has ptransform='
        ),
    ):
      beam_ds.map_blocks(lambda x: x).head(x=2)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_chunking',
          old_sizes={'x': 13},
          old_chunks={'x': 13},
          new_sizes={'x': 7},
          expected={'x': 7},
      ),
      dict(
          testcase_name='large_chunking',
          old_sizes={'x': 100},
          old_chunks={'x': 20},
          new_sizes={'x': 10},
          expected={'x': 2},
      ),
      dict(
          testcase_name='new_dims',
          old_sizes={'x': 5},
          old_chunks={'x': 5},
          new_sizes={'y': 10},
          expected={'y': 10},
      ),
  )
  def test_infer_new_chunks(self, old_sizes, old_chunks, new_sizes, expected):
    actual = xbeam_dataset._infer_new_chunks(old_sizes, old_chunks, new_sizes)
    self.assertEqual(actual, expected)

  def test_infer_new_chunks_uneven_chunks_error(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "cannot infer new chunks for dimension 'x' with changed size "
        "10 -> 5: existing chunks {'x': 3} do not evenly divide existing "
        "sizes {'x': 10}",
    ):
      xbeam_dataset._infer_new_chunks(
          old_sizes={'x': 10}, old_chunks={'x': 3}, new_sizes={'x': 5}
      )

  def test_infer_new_chunks_uneven_new_size_error(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "cannot infer new chunks for dimension 'x' with changed size "
        '10 -> 3: the 2 chunks along this dimension do not evenly divide '
        'the new size 3',
    ):
      xbeam_dataset._infer_new_chunks(
          old_sizes={'x': 10}, old_chunks={'x': 5}, new_sizes={'x': 3}
      )

  def test_pipe(self):
    source = xarray.Dataset({'foo': ('x', np.arange(10))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 5})
    mapped_ds = source_ds.pipe(xbeam.Dataset.map_blocks, lambda ds: 2 * ds)
    expected = 2 * source
    actual = mapped_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, expected)


class MapBlocksTest(test_util.TestCase):

  def test_map_blocks(self):
    source = xarray.Dataset({'foo': ('x', np.arange(10))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 5})
    mapped_ds = source_ds.map_blocks(lambda x: 2 * x)
    self.assertRegex(
        mapped_ds.ptransform.label, r'^from_xarray_\d+|map_blocks_\d+$'
    )
    expected = 2 * source
    actual = mapped_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, expected)

  def test_map_blocks_new_vars_and_dims(self):
    source = xarray.Dataset({'foo': ('x', np.arange(10))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 5})
    mapped_ds = source_ds.map_blocks(
        lambda ds: ds.assign(bar=2 * ds.foo.expand_dims('y'))
    )
    self.assertEqual(mapped_ds.chunks, {'x': 5, 'y': 1})
    expected = source.assign(bar=2 * source.foo.expand_dims('y'))
    actual = mapped_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, expected)

  def test_map_blocks_new_size_full_chunks(self):
    source = xarray.Dataset({'foo': (('x', 'y'), np.arange(20).reshape(4, 5))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 2})
    mapped_ds = source_ds.map_blocks(lambda ds: ds.head(y=3))
    expected = source.head(y=3)
    actual = mapped_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, expected)

  def test_map_blocks_new_size_evenly_divided(self):
    source = xarray.Dataset({'foo': ('x', np.arange(8))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 4})
    mapped_ds = source_ds.map_blocks(lambda ds: ds.coarsen(x=2).mean())
    expected = source.coarsen(x=2).mean()
    self.assertEqual(mapped_ds.chunks, {'x': 2})
    actual = mapped_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, expected)

  def test_map_blocks_median(self):
    source = xarray.Dataset({'foo': (('x', 'y'), np.arange(20).reshape(4, 5))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 2})
    mapped_ds = source_ds.map_blocks(lambda ds: ds.median('y'))
    self.assertEqual(mapped_ds.chunks, {'x': 2})
    expected = source.median('y')
    actual = mapped_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, expected)

  def test_map_blocks_template_inference_fails(self):
    source = xarray.Dataset({'foo': ('x', np.arange(10))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 5})
    func = lambda ds: ds.compute()  # something non-lazy

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'failed to lazily apply func() to the existing template. Consider '
        'supplying template explicitly or modifying func() to support lazy '
        'dask arrays.',
    ):
      source_ds.map_blocks(func)

  def test_map_blocks_explicit_template(self):
    source = xarray.Dataset({'foo': ('x', np.arange(10))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 5})
    func = lambda ds: ds.compute()  # something non-lazy
    mapped_ds = source_ds.map_blocks(func, template=source)
    actual = mapped_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, source)

  def test_map_blocks_new_split_vars_fails(self):
    source = xarray.Dataset({'foo': ('x', np.arange(10))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 5}, split_vars=True)
    func = lambda ds: ds.rename({'foo': 'bar'})
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        'cannot use map_blocks on a dataset with split_vars=True if the '
        'transformation returns a different set of variables.\n'
        "Old split variables: {'foo'}\n"
        "New split variables: {'bar'}",
    ):
      source_ds.map_blocks(func)


class RechunkingTest(test_util.TestCase):

  def test_split_variables(self):
    source = xarray.Dataset(
        {'foo': ('x', np.arange(10)), 'bar': ('x', np.arange(10))}
    )
    beam_ds = xbeam.Dataset.from_xarray(source, {'x': 5}, split_vars=False)
    self.assertFalse(beam_ds.split_vars)
    split_ds = beam_ds.split_variables()
    self.assertTrue(split_ds.split_vars)
    self.assertRegex(
        split_ds.ptransform.label, r'^from_xarray_\d+\|split_vars_\d+$'
    )
    actual = split_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, source)

  def test_consolidate_variables(self):
    source = xarray.Dataset(
        {'foo': ('x', np.arange(10)), 'bar': ('x', np.arange(10))}
    )
    beam_ds = xbeam.Dataset.from_xarray(source, {'x': 5}, split_vars=True)
    self.assertTrue(beam_ds.split_vars)
    consolidated_ds = beam_ds.consolidate_variables()
    self.assertFalse(consolidated_ds.split_vars)
    self.assertRegex(
        consolidated_ds.ptransform.label,
        r'^from_xarray_\d+\|consolidate_vars_\d+$',
    )
    actual = consolidated_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, source)

  def test_rechunk(self):
    source_chunks = {'x': 5, 'y': 1}
    target_chunks = {'x': 2, 'y': -1}
    source = xarray.Dataset({'foo': (('x', 'y'), np.arange(40).reshape(10, 4))})
    beam_ds = xbeam.Dataset.from_xarray(source, source_chunks)
    rechunked_ds = beam_ds.rechunk(target_chunks)

    self.assertEqual(rechunked_ds.chunks, {'x': 2, 'y': 4})
    actual = rechunked_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, source)

  def test_rechunk_split_vars(self):
    source = xarray.Dataset({
        'foo': (('x', 'y'), np.arange(20).reshape(10, 2)),
        'bar': ('x', np.arange(10)),
    })
    beam_ds = xbeam.Dataset.from_xarray(
        source, {'x': 5, 'y': 2}, split_vars=True
    )
    rechunked_ds = beam_ds.rechunk({'x': 2, 'y': 1})
    self.assertEqual(rechunked_ds.chunks, {'x': 2, 'y': 1})
    actual = rechunked_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, source)


class EndToEndTest(test_util.TestCase):

  def test_bytes_per_chunk_and_chunk_count(self):
    source_ds = test_util.dummy_era5_surface_dataset(
        variables=2, latitudes=73, longitudes=144, times=365, freq='24H'
    )

    xbeam_ds = xbeam.Dataset.from_xarray(
        source_ds, {'time': 90}, split_vars=False
    )
    self.assertEqual(
        xbeam_ds.chunks, {'time': 90, 'latitude': 73, 'longitude': 144}
    )
    self.assertEqual(xbeam_ds.bytes_per_chunk, 2 * 73 * 144 * 90 * 4)
    self.assertEqual(xbeam_ds.chunk_count, 5)

    xbeam_ds = xbeam.Dataset.from_xarray(
        source_ds, {'time': 90}, split_vars=True
    )
    self.assertEqual(
        xbeam_ds.chunks, {'time': 90, 'latitude': 73, 'longitude': 144}
    )
    self.assertEqual(xbeam_ds.bytes_per_chunk, 73 * 144 * 90 * 4)
    self.assertEqual(xbeam_ds.chunk_count, 5 * 2)

  def test_docstring_example(self):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('output').full_path

    source_ds = test_util.dummy_era5_surface_dataset(
        variables=2, latitudes=73, longitudes=144, times=365, freq='24H'
    )
    source_ds.chunk({'time': 90}).to_zarr(input_path)

    transform = (
        xbeam.Dataset.from_zarr(input_path)
        .rechunk({'time': -1, 'latitude': 10, 'longitude': 10})
        .map_blocks(lambda x: x.median('time'))
        .to_zarr(output_path)
    )
    test_util.EagerPipeline() | transform

    expected = source_ds.median('time')
    actual, chunks = xbeam.open_zarr(output_path)
    xarray.testing.assert_identical(expected, actual)
    self.assertEqual(chunks, {'latitude': 10, 'longitude': 10})

  def test_climatology(self):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('output').full_path

    source_ds = test_util.dummy_era5_surface_dataset(times=365, freq='24H')
    source_ds.chunk({'time': 90}).to_zarr(input_path)

    transform = (
        xbeam.Dataset.from_zarr(input_path)
        .rechunk({'time': -1, 'latitude': 10, 'longitude': 10})
        .map_blocks(lambda x: x.groupby('time.month').mean())
        .to_zarr(output_path)
    )
    test_util.EagerPipeline() | transform

    expected = source_ds.groupby('time.month').mean()
    actual, chunks = xbeam.open_zarr(output_path)
    xarray.testing.assert_identical(expected, actual)
    self.assertEqual(chunks, {'month': 12, 'latitude': 10, 'longitude': 10})

  def test_resample(self):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('output').full_path

    source_ds = test_util.dummy_era5_surface_dataset(
        latitudes=73, longitudes=144, times=365, freq='24H'
    )
    source_ds.chunk({'time': 90}).to_zarr(input_path)

    transform = (
        xbeam.Dataset.from_zarr(input_path)
        .rechunk({'time': -1, 'latitude': 10, 'longitude': 10})
        .map_blocks(lambda x: x.resample(time='10D').mean())
        .rechunk({'time': 20, 'latitude': -1, 'longitude': -1})
        .to_zarr(
            output_path,
            zarr_chunks={'time': 10, 'latitude': -1, 'longitude': -1},
            zarr_shards={'time': 20, 'latitude': -1, 'longitude': -1},
            zarr_format=3,
        )
    )
    test_util.EagerPipeline() | transform

    expected = source_ds.resample(time='10D').mean()
    actual, chunks = xbeam.open_zarr(output_path)
    xarray.testing.assert_identical(expected, actual)
    self.assertEqual(chunks, {'time': 10, 'latitude': 73, 'longitude': 144})


class MeanTest(test_util.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='x', dim='x', skipna=True, fanout=None),
      dict(testcase_name='y', dim='y', skipna=True, fanout=None),
      dict(testcase_name='two_dims', dim=['x', 'y'], skipna=True, fanout=None),
      dict(testcase_name='all_dims', dim=None, skipna=True, fanout=None),
      dict(testcase_name='skipna_false', dim='y', skipna=False, fanout=None),
      dict(testcase_name='with_fanout', dim='y', skipna=True, fanout=2),
  )
  def test_mean(self, dim, skipna, fanout):
    source_ds = xarray.Dataset(
        {'foo': (('x', 'y'), np.array([[1, 2, np.nan], [4, np.nan, 6]]))}
    )
    beam_ds = xbeam.Dataset.from_xarray(source_ds, chunks={'x': 1})
    actual = beam_ds.mean(dim=dim, skipna=skipna, fanout=fanout)
    expected = source_ds.mean(dim=dim, skipna=skipna)
    actual_collected = actual.collect_with_direct_runner()
    xarray.testing.assert_identical(expected, actual_collected)


if __name__ == '__main__':
  absltest.main()
