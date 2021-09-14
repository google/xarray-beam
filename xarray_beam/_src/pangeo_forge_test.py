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
"""Tests for xarray_beam._src.pangeo."""

import contextlib
import itertools
import tempfile

import numpy as np
from pangeo_forge_recipes.patterns import FilePattern, ConcatDim

from xarray_beam._src import core
from xarray_beam._src import test_util
from xarray_beam._src.pangeo_forge import (
    FilePatternToChunks,
    _expand_dimensions_by_key
)


class ExpandDimensionsByKeyTest(test_util.TestCase):

  def setUp(self):
    self.test_data = test_util.dummy_era5_surface_dataset()
    self.level = ConcatDim("level", list(range(91, 100)))
    self.pattern = FilePattern(lambda level: f"gs://dir/{level}.nc", self.level)

  def test_expands_dimensions(self):
    key = core.Key(offsets={"time": 0, "level": 0})

    for i, (index, _) in enumerate(self.pattern.items()):
      actual = _expand_dimensions_by_key(
        self.test_data, key, index, self.pattern
      )

      expected_dims = dict(self.test_data.dims)
      expected_dims.update({"level": 1})

      self.assertEqual(expected_dims, dict(actual.dims))
      self.assertEqual(np.array([self.level.keys[i]]), actual["level"])

  def test_raises_error_when_dataset_is_not_found(self):
    key = core.Key({"time": 0, "boat": 0})
    index = (0,)
    with self.assertRaisesRegex(ValueError, "boat") as e:
      _expand_dimensions_by_key(
        self.test_data, key, index, self.pattern
      )


class FilePatternToChunksTest(test_util.TestCase):

  def setUp(self):
    self.test_data = test_util.dummy_era5_surface_dataset()

  @contextlib.contextmanager
  def pattern_from_testdata(self) -> FilePattern:
    """Produces a FilePattern for a temporary NetCDF file of test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
      target = f'{tmpdir}/era5.nc'
      self.test_data.to_netcdf(target)
      yield FilePattern(lambda: target)

  @contextlib.contextmanager
  def multifile_pattern(
      self,
      time_step: int = 479,
      longitude_step: int = 47
  ) -> FilePattern:
    """Produces a FilePattern for a temporary NetCDF file of test data."""
    time_dim = ConcatDim('time', list(range(0, 360 * 4, time_step)))
    longitude_dim = ConcatDim('longitude', list(range(0, 144, longitude_step)))

    with tempfile.TemporaryDirectory() as tmpdir:
      def make_path(time: int, longitude: int) -> str:
        return f'{tmpdir}/era5-{time}-{longitude}.nc'

      for time in time_dim.keys:
        for long in longitude_dim.keys:
          chunk = self.test_data.isel(
            time=slice(time, time + time_step),
            longitude=slice(long, long + longitude_step)
          )
          chunk.to_netcdf(make_path(time, long))
      yield FilePattern(make_path, time_dim, longitude_dim)

  def test_prechunk_converts_correctly(self):
    pattern = FilePattern(
      lambda time: f"gs://bucket/{time:02d}/{time:02d}.nc",
      ConcatDim("time", list(range(1, 31))),
    )

    transform = FilePatternToChunks(pattern)

    expected = [core.Key({"time": i}) for i in range(0, 30)]
    actual = [key for key, *_ in transform._prechunk()]

    self.assertEqual(expected, actual)

  def test_prechunk_with_two_dims_converts_correctly(self):
    pattern = FilePattern(
      lambda time, level: f"gs://bucket/{time:02d}/{level:02d}.nc",
      ConcatDim("time", list(range(1, 31))),
      ConcatDim("level", list(range(5))),
    )

    transform = FilePatternToChunks(pattern)

    expected = [core.Key({"time": i, "level": j})
                for i, j in itertools.product(range(30), range(5))]
    actual = [key for key, *_ in transform._prechunk()]

    self.assertEqual(expected, actual)

  def test_prechunk_from_pattern_with_nitemsper_converts_correctly(self):
    pattern = FilePattern(
      lambda time, level: f"gs://bucket/{time:02d}/{level:02d}.nc",
      ConcatDim("time", list(range(1, 31)), nitems_per_file=24),
      ConcatDim("level", list(range(5))),
    )

    transform = FilePatternToChunks(pattern)

    expected = [core.Key({"time": i, "level": j})
                for i, j in itertools.product(range(0, 30 * 24, 24), range(5))]
    actual = [key for key, *_ in transform._prechunk()]

    self.assertEqual(expected, actual)

  def test_no_subchunks_returns_single_dataset(self):
    expected = [(core.Key({"time": 0, "latitude": 0, "longitude": 0}),
                 self.test_data)]
    with self.pattern_from_testdata() as pattern:
      actual = test_util.EagerPipeline() | FilePatternToChunks(pattern)

    self.assertIdenticalChunks(actual, expected)

  def test_single_subchunks_returns_multiple_datasets(self):
    with self.pattern_from_testdata() as pattern:
      result = (
          test_util.EagerPipeline()
          | FilePatternToChunks(pattern, sub_chunks={"longitude": 48})
      )

    expected_keys = [core.Key({"time": 0, "latitude": 0, "longitude": i})
                     for i in range(0, 144, 48)]
    expected_sizes = [{"time": 365 * 4, "latitude": 73, "longitude": 48}
                      for _ in range(3)]
    actual_keys = [key for key, _ in result]
    actual_sizes = [dict(ds.sizes) for _, ds in result]

    self.assertEqual(expected_keys, actual_keys)
    self.assertEqual(expected_sizes, actual_sizes)

  def test_multiple_subchunks_returns_multiple_datasets(self):
    with self.pattern_from_testdata() as pattern:
      result = (
          test_util.EagerPipeline()
          | FilePatternToChunks(pattern,
                                sub_chunks={"longitude": 48, "latitude": 24})
      )

    expected_keys = [
      core.Key({"time": 0, "latitude": a, "longitude": o})
      for o, a in itertools.product(range(0, 144, 48), range(0, 73, 24))
    ]
    expected_sizes = [
      {"time": 365 * 4, "longitude": o, "latitude": a}
      for o, a, in itertools.product([48, 48, 48], [24, 24, 24, 1])
    ]
    actual_keys = [key for key, _ in result]
    actual_sizes = [dict(ds.sizes) for _, ds in result]

    self.assertEqual(expected_keys, actual_keys)
    self.assertEqual(expected_sizes, actual_sizes)

  def test_single_subchunks_over_multiple_files_returns_multiple_datasets(self):
    with self.multifile_pattern() as pattern:
      result = (
          test_util.EagerPipeline()
          | FilePatternToChunks(pattern, sub_chunks={"latitude": 24})
      )

    expected_keys = [
      core.Key({"time": t, "latitude": a, "longitude": o})
      for t, o, a in itertools.product(range(4), range(4), range(0, 73, 24))
    ]
    expected_sizes = [
      {"time": t, "latitude": a, "longitude": o}
      for t, o, a, in
      itertools.product([479, 479, 479, 23], [47, 47, 47, 3], [24, 24, 24, 1])
    ]

    actual_keys = [key for key, _ in result]
    actual_sizes = [dict(ds.sizes) for _, ds in result]

    self.assertEqual(expected_keys, actual_keys)
    self.assertEqual(expected_sizes, actual_sizes)
