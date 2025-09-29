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
import textwrap

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
      result = beam_ds.map_blocks(lambda x: x).pipe(call, beam_ds)
      actual = result.collect_with_direct_runner()
      xarray.testing.assert_identical(expected, actual)

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
        "10 -> 3: the 2 chunks along this dimension do not evenly divide "
        "the new size 3",
    ):
      xbeam_dataset._infer_new_chunks(
          old_sizes={'x': 10}, old_chunks={'x': 5}, new_sizes={'x': 3}
      )


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
        lambda ds: ds.assign(bar=2*ds.foo.expand_dims('y'))
    )
    self.assertEqual(mapped_ds.chunks, {'x': 5, 'y': 1})
    expected = source.assign(bar=2*source.foo.expand_dims('y'))
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


if __name__ == '__main__':
  absltest.main()
