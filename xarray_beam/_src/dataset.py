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
import functools
import itertools
import operator
import os.path
import tempfile
import textwrap
from typing import Any, Callable, Literal

import apache_beam as beam
import xarray
from xarray_beam._src import core
from xarray_beam._src import rechunk
from xarray_beam._src import zarr


def _apply_to_each_chunk(
    func: Callable[[xarray.Dataset], xarray.Dataset],
    key: core.Key,
    chunk: xarray.Dataset,
) -> tuple[core.Key, xarray.Dataset]:
  """Apply a function to each chunk."""
  new_chunk = func(chunk)
  new_offsets = {}
  for dim in new_chunk.dims:
    assert isinstance(dim, str)
    new_offsets[dim] = key.offsets.get(dim, 0)
  for dim, size in new_chunk.sizes.items():
    old_size = chunk.sizes.get(dim)
    if old_size is not None and old_size != size and new_offsets[dim] != 0:
      raise ValueError(
          f'applied function {func} changes size of dimension {dim!r} with '
          f'non-zero chunk offset {new_offsets[dim]}:\nChunk key: {key}\n'
          f'Input chunk:\n{textwrap.indent(repr(chunk), prefix="    ")}\n'
          f'Computed chunk:\n{textwrap.indent(repr(new_chunk), prefix="    ")}'
      )
  new_vars = set(new_chunk) if key.vars is not None else None
  new_key = core.Key(new_offsets, new_vars)
  return new_key, new_chunk


def _whole_dataset_method(method_name: str):
  """Helper function for defining a method with a fast-path for lazy data."""

  def method(self: Dataset, *args, **kwargs) -> Dataset:
    func = operator.methodcaller(method_name, *args, **kwargs)
    template = zarr.make_template(func(self.template))
    chunks = {k: v for k, v in self.chunks.items() if k in template.dims}

    label = _get_label(method_name)
    if isinstance(self.ptransform, core.DatasetToChunks):
      dataset = func(self.ptransform.dataset)
      ptransform = label >> core.DatasetToChunks(
          dataset, chunks, self.split_vars
      )
    else:
      ptransform = self.ptransform | label >> beam.MapTuple(
          functools.partial(_apply_to_each_chunk, func)
      )
    return Dataset(template, chunks, self.split_vars, ptransform)

  return method


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

  def __post_init__(self):
    for dim, size in self.template.sizes.items():
      if self.chunks.get(dim, -1) == -1:
        self.chunks[dim] = size

  @classmethod
  def from_xarray(
      cls,
      source: xarray.Dataset,
      chunks: Mapping[str, int],
      split_vars: bool = False,
  ) -> Dataset:
    """Create an xarray_beam.Dataset from an xarray.Dataset."""
    template = zarr.make_template(source)
    ptransform = core.DatasetToChunks(source, chunks, split_vars)
    ptransform.label = _get_label('from_xarray')
    return cls(template, dict(chunks), split_vars, ptransform)

  @classmethod
  def from_zarr(cls, path: str, split_vars: bool = False) -> Dataset:
    """Create an xarray_beam.Dataset from a zarr file."""
    source, chunks = zarr.open_zarr(path)
    result = cls.from_xarray(source, chunks, split_vars)
    result.ptransform.label = _get_label('from_zarr')
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

  def map_blocks(
      self,
      /,
      func,
      *,
      kwargs: dict[str, Any] | None = None,
      template_method: Literal['lazy_func'] = 'lazy_func',
  ) -> Dataset:
    """Map a function over the chunks of this dataset.

    Args:
      func: any function that does not change the size of dataset chunks, called
        like `func(chunk, **kwargs)`, where `chunk` is an xarray.Dataset.
      kwargs: passed on to func, unmodified.
      template_method: method to use for updating the dataset template:  *
        'lazy_func': apply `func` to the existing template. This requires that
        `func` does not need to access any chunk values, which typically means
        implemented using dask compatible operations in Xarray.

    Returns:
      New Dataset with updated chunks.
    """
    if kwargs is not None:
      func = functools.partial(func, **kwargs)

    if template_method == 'lazy_func':
      try:
        template = func(self.template)
      except ValueError as e:
        raise ValueError(
            'failed to lazily apply func() to the existing template. Consider '
            'supplying template explicitly or modifying func() to support lazy '
            'dask arrays.'
        ) from e
      template = zarr.make_template(template)  # ensure template is lazy
    else:
      raise ValueError(f'unsupported template_method: {template_method!r}')

    for dim in template.dims:
      if (
          dim in self.sizes
          and self.chunks[dim] != self.sizes[dim]
          and template.sizes[dim] != self.sizes[dim]
      ):
        raise ValueError(
            f'{dim!r} has an inconsistent size between the new and old '
            f'template: {dict(template.sizes)} vs {dict(self.sizes)}'
        )

    label = _get_label('map_blocks')
    chunks = {}
    for dim, size in template.sizes.items():
      assert isinstance(dim, str)
      chunks[dim] = self.chunks.get(dim, size)
    ptransform = self.ptransform | label >> beam.MapTuple(
        functools.partial(_apply_to_each_chunk, func)
    )
    return type(self)(template, chunks, self.split_vars, ptransform)

  # TODO(shoyer): implement rechunking, merge, rename, mean, etc

  # thin wrappers around xarray methods
  __getitem__ = _whole_dataset_method('__getitem__')
  transpose = _whole_dataset_method('transpose')

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
