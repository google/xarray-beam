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
from typing import Dict

import numpy as np
from absl.testing import parameterized
from pangeo_forge_recipes.patterns import (
    FilePattern,
    ConcatDim,
    DimIndex,
    CombineOp,
)

from xarray_beam import split_chunks
from xarray_beam._src import core
from xarray_beam._src import test_util
from xarray_beam._src.pangeo_forge import (
    FilePatternToChunks,
    _expand_dimensions_by_key,
)


class ExpandDimensionsByKeyTest(test_util.TestCase):

  def setUp(self):
    self.test_data = test_util.dummy_era5_surface_dataset()
    self.level = ConcatDim("level", list(range(91, 100)))
    self.pattern = FilePattern(lambda level: f"gs://dir/{level}.nc", self.level)

  def test_expands_dimensions(self):
    for i, (index, _) in enumerate(self.pattern.items()):
      actual = _expand_dimensions_by_key(self.test_data, index, self.pattern)

      expected_dims = dict(self.test_data.dims)
      expected_dims.update({"level": 1})

      self.assertEqual(expected_dims, dict(actual.dims))
      self.assertEqual(np.array([self.level.keys[i]]), actual["level"])

  def test_raises_error_when_dataset_is_not_found(self):
    index = (DimIndex("boat", 0, 1, CombineOp.CONCAT),)
    with self.assertRaisesRegex(ValueError, "boat"):
      _expand_dimensions_by_key(self.test_data, index, self.pattern)


class FilePatternToChunksTest(test_util.TestCase):

  def setUp(self):
    self.test_data = test_util.dummy_era5_surface_dataset()

  @contextlib.contextmanager
  def pattern_from_testdata(self) -> FilePattern:
    """Produces a FilePattern for a temporary NetCDF file of test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
      target = f"{tmpdir}/era5.nc"
      self.test_data.to_netcdf(target)
      yield FilePattern(lambda: target)

  @contextlib.contextmanager
  def multifile_pattern(
      self, time_step: int = 479, longitude_step: int = 47
  ) -> FilePattern:
    """Produces a FilePattern for a temporary NetCDF file of test data."""
    time_dim = ConcatDim("time", list(range(0, 360 * 4, time_step)))
    longitude_dim = ConcatDim("longitude", list(range(0, 144, longitude_step)))

    with tempfile.TemporaryDirectory() as tmpdir:

      def make_path(time: int, longitude: int) -> str:
        return f"{tmpdir}/era5-{time}-{longitude}.nc"

      for time in time_dim.keys:
        for long in longitude_dim.keys:
          chunk = self.test_data.isel(
              time=slice(time, time + time_step),
              longitude=slice(long, long + longitude_step),
          )
          chunk.to_netcdf(make_path(time, long))
      yield FilePattern(make_path, time_dim, longitude_dim)

  def test_returns_single_dataset(self):
    expected = [
        (core.Key({"time": 0, "latitude": 0, "longitude": 0}), self.test_data)
    ]
    with self.pattern_from_testdata() as pattern:
      actual = test_util.EagerPipeline() | FilePatternToChunks(pattern)

    self.assertAllCloseChunks(actual, expected)

  def test_single_subchunks_returns_multiple_datasets(self):
    with self.pattern_from_testdata() as pattern:
      result = test_util.EagerPipeline() | FilePatternToChunks(
          pattern, chunks={"longitude": 48}
      )

    expected = [
        (
            core.Key({"time": 0, "latitude": 0, "longitude": i}),
            self.test_data.isel(longitude=slice(i, i + 48)),
        )
        for i in range(0, 144, 48)
    ]
    self.assertAllCloseChunks(result, expected)

  def test_multiple_subchunks_returns_multiple_datasets(self):
    with self.pattern_from_testdata() as pattern:
      result = test_util.EagerPipeline() | FilePatternToChunks(
          pattern, chunks={"longitude": 48, "latitude": 24}
      )

    expected = [
        (
            core.Key({"time": 0, "longitude": o, "latitude": a}),
            self.test_data.isel(
                longitude=slice(o, o + 48), latitude=slice(a, a + 24)
            ),
        )
        for o, a in itertools.product(range(0, 144, 48), range(0, 73, 24))
    ]

    self.assertAllCloseChunks(result, expected)

  @parameterized.parameters(
      dict(time_step=479, longitude_step=47),
      dict(time_step=365, longitude_step=72),
      dict(time_step=292, longitude_step=71),
      dict(time_step=291, longitude_step=48),
  )
  def test_multiple_datasets_returns_multiple_datasets(
      self, time_step: int, longitude_step: int
  ):
    expected = [
        (
            core.Key({"time": t, "latitude": 0, "longitude": o}),
            self.test_data.isel(
                time=slice(t, t + time_step),
                longitude=slice(o, o + longitude_step),
            ),
        )
        for t, o in itertools.product(
            range(0, 360 * 4, time_step), range(0, 144, longitude_step)
        )
    ]
    with self.multifile_pattern(time_step, longitude_step) as pattern:
      actual = test_util.EagerPipeline() | FilePatternToChunks(pattern)

    self.assertAllCloseChunks(actual, expected)

  @parameterized.parameters(
      dict(time_step=365, longitude_step=72, chunks={"latitude": 36}),
      dict(time_step=365, longitude_step=72, chunks={"longitude": 36}),
      dict(
          time_step=365,
          longitude_step=72,
          chunks={"longitude": 36, "latitude": 66},
      ),
  )
  def test_multiple_datasets_with_subchunks_returns_multiple_datasets(
      self,
      time_step: int,
      longitude_step: int,
      chunks: Dict[str, int],
  ):
    expected = []
    for t, o in itertools.product(
        range(0, 360 * 4, time_step), range(0, 144, longitude_step)
    ):
      expected.extend(
          split_chunks(
              core.Key({"latitude": 0, "longitude": o, "time": t}),
              self.test_data.isel(
                  time=slice(t, t + time_step),
                  longitude=slice(o, o + longitude_step),
              ),
              chunks,
          )
      )
    with self.multifile_pattern(time_step, longitude_step) as pattern:
      actual = test_util.EagerPipeline() | FilePatternToChunks(
          pattern, chunks=chunks
      )

      self.assertAllCloseChunks(actual, expected)
