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
"""A high-level interface for Xarray-Beam datasets.

Usage example (not fully implemented yet!):

    import xarray_beam as xbeam

    transform = (
        xbeam.Dataset.from_zarr(input_path)
        .rechunk({'time': -1, 'latitude': 10, 'longitude': 10})
        .map_blocks(lambda x: x.median('time'))
        .to_zarr(output_path)
    )
    with beam.Pipeline() as p:
      p | transform
"""
from __future__ import annotations

import collections
from collections.abc import Mapping
import dataclasses
import itertools
import os.path
import tempfile

import apache_beam as beam
import xarray
from xarray_beam._src import core
from xarray_beam._src import zarr


class _CountNamer:

  def __init__(self):
    self._counts = collections.defaultdict(itertools.count)

  def apply(self, name: str) -> str:
    return f'{name}_{next(self._counts[name])}'


_get_label = _CountNamer().apply


@dataclasses.dataclass
class Dataset:
  """Experimental high-level representation of an Xarray-Beam dataset."""

  template: xarray.Dataset
  chunks: dict[str, int]
  split_vars: bool
  ptransform: beam.PTransform

  @classmethod
  def from_xarray(
      cls,
      source: xarray.Dataset,
      chunks: Mapping[str, int],
      split_vars: bool = False,
  ) -> Dataset:
    """Create an xarray_beam.Dataset from an xarray.Dataset."""
    template = zarr.make_template(source)
    ptransform = _get_label('from_xarray') >> core.DatasetToChunks(
        source, chunks, split_vars
    )
    return cls(template, dict(chunks), split_vars, ptransform)

  @classmethod
  def from_zarr(cls, path: str, split_vars: bool = False) -> Dataset:
    """Create an xarray_beam.Dataset from a zarr file."""
    source, chunks = zarr.open_zarr(path)
    result = cls.from_xarray(source, chunks, split_vars)
    result.ptransform = _get_label('from_zarr') >> result.ptransform
    return result

  def to_zarr(self, path: str) -> beam.PTransform:
    """Write to a Zarr file."""
    return self.ptransform | _get_label('to_zarr') >> zarr.ChunksToZarr(
        path, self.template, self.chunks
    )

  def collect_with_direct_runner(self) -> xarray.Dataset:
    """Collect a dataset in memory by writing it to a temp file."""
    # TODO(shoyer): generalize this function to something that support
    # alternative runners can we figure out a suitable temp file location for
    # distributed runners?

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_path = os.path.join(temp_dir, 'tmp.zarr')
      with beam.Pipeline(runner='DirectRunner') as pipeline:
        pipeline |= self.to_zarr(temp_path)
      return xarray.open_zarr(temp_path).compute()

  # TODO(shoyer): implement map_blocks, rechunking, merge, rename, mean, etc

  @property
  def sizes(self) -> dict[str, int]:
    """Size of each dimension on this dataset."""
    return dict(self.template.sizes)  # pytype: disable=bad-return-type

  def pipe(self, func, *args, **kwargs):
    return func(*args, **kwargs)

  def __repr__(self):
    base = repr(self.template)
    chunks_str = ', '.join(f'{k}: {v}' for k, v in self.chunks.items())
    return (
        f'<xarray_beam.Dataset[{chunks_str}][split_vars={self.split_vars}]>'
        + '\n'
        + '\n'.join(base.split('\n')[1:])
    )
