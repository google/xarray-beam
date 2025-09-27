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


class DatasetTest(test_util.TestCase):

  def test_from_xarray(self):
    ds = xarray.Dataset({'foo': ('x', np.arange(10))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 5})
    self.assertIsInstance(beam_ds, xbeam.Dataset)
    self.assertEqual(beam_ds.sizes, {'x': 10})
    self.assertEqual(beam_ds.template.keys(), {'foo'})
    self.assertEqual(beam_ds.chunks, {'x': 5})
    self.assertFalse(beam_ds.split_vars)
    self.assertRegex(beam_ds.ptransform.label, r'^from_xarray_\d+$')
    self.assertEqual(
        repr(beam_ds).split('\n')[0],
        '<xarray_beam.Dataset[x: 5][split_vars=False]>',
    )
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
    ds = xarray.Dataset({'foo': ('x', np.arange(10))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 5})
    to_zarr = beam_ds.to_zarr(temp_dir)

    self.assertRegex(to_zarr.label, r'^from_xarray_\d+|to_zarr_\d+$')
    with beam.Pipeline() as p:
      p |= to_zarr
    opened = xarray.open_zarr(temp_dir).compute()
    xarray.testing.assert_identical(ds, opened)

  def test_to_zarr_shards(self):
    temp_dir = self.create_tempdir().full_path
    ds = xarray.Dataset({'foo': ('x', np.arange(12))})
    beam_ds = xbeam.Dataset.from_xarray(ds, {'x': 6})
    to_zarr = beam_ds.to_zarr(
        temp_dir, zarr_chunks={'x': 3}, zarr_shards={'x': 6}, zarr_format=3
    )
    with beam.Pipeline() as p:
      p |= to_zarr
    opened = xarray.open_zarr(temp_dir).compute()
    xarray.testing.assert_identical(ds, opened)

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
    result = call(beam_ds)
    actual = result.collect_with_direct_runner()
    xarray.testing.assert_identical(expected, actual)

  def test_apply_to_each_chunk_inconsistent_size(self):
    ds = xarray.Dataset({'foo': ('x', np.arange(10))})
    key = xbeam.Key({'x': 1})
    func = lambda x: x.head(x=4)
    with self.assertRaisesRegex(
        ValueError,
        re.escape("changes size of dimension 'x' with non-zero chunk offset 1"),
    ):
      xbeam_dataset._apply_to_each_chunk(func, key, ds)


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

  def test_map_blocks_new_size(self):
    source = xarray.Dataset({'foo': (('x', 'y'), np.arange(20).reshape(4, 5))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 2})
    mapped_ds = source_ds.map_blocks(lambda x: x.head(y=2))
    expected = source.head(y=2)
    actual = mapped_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, expected)

  def test_map_blocks_median(self):
    source = xarray.Dataset({'foo': (('x', 'y'), np.arange(20).reshape(4, 5))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 2})
    mapped_ds = source_ds.map_blocks(lambda x: x.median('y'))
    expected = source.median('y')
    actual = mapped_ds.collect_with_direct_runner()
    xarray.testing.assert_identical(actual, expected)

  def test_map_blocks_errors(self):
    source = xarray.Dataset({'foo': ('x', np.arange(10))})
    source_ds = xbeam.Dataset.from_xarray(source, {'x': 5})
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "'x' has an inconsistent size between the new and old "
        "template: {'x': 3} vs {'x': 10}",
    ):
      source_ds.map_blocks(lambda x: x.head(x=3))

  def test_map_blocks_template_method_fails(self):
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

    with self.assertRaisesWithLiteralMatch(
        ValueError, "unsupported template_method: 'unknown'"
    ):
      source_ds.map_blocks(func, template_method='unknown')


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

  def test_docstring_example(self):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('output').full_path

    source_ds = test_util.dummy_era5_surface_dataset(times=365, freq='24H')
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


if __name__ == '__main__':
  absltest.main()
