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
"""Rechunk a Zarr dataset."""
import types

from absl import app
from absl import flags
import apache_beam as beam
import xarray_beam as xbeam


INPUT_PATH = flags.DEFINE_string('input_path', None, help='input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='output Zarr path')
TARGET_CHUNKS = flags.DEFINE_string(
    'target_chunks',
    '',
    help=(
        'chunks on the input Zarr dataset to change on the outputs, in the '
        'form of a comma separated dimension=size pairs, e.g., '
        "--target_chunks='x=10,y=10'. Omitted dimensions are not changed and a "
        'chunksize of -1 indicates not to chunk a dimension.'
    ),
)
TARGET_SHARDS = flags.DEFINE_string(
    'target_shards',
    None,
    help=(
        'Desired shards for each dimension in the output Zarr dataset, in the '
        'same format as --target_chunks. If omitted, sharding is not used. '
        'Shards should be multiples of corresponding chunk sizes. Only valid '
        'with Zarr v3.'
    ),
)
ZARR_FORMAT = flags.DEFINE_integer(
    'zarr_format',
    None,
    help='Zarr format to use for the output.',
)
RUNNER = flags.DEFINE_string('runner', None, help='beam.runners.Runner')


# pylint: disable=expression-not-assigned


def _try_to_int(chunks_str: str) -> int | str:
  try:
    return int(chunks_str)
  except ValueError:
    return chunks_str


def _parse_chunks_flag(
    chunks_str: str,
) -> dict[str | types.EllipsisType, int | str] | int | str:
  """Parse a string representation of unnormalized chunks."""
  if '=' not in chunks_str:
    return _try_to_int(chunks_str)

  chunks = {}
  parts = chunks_str.split(',')
  for part in parts:
    k, v = part.split('=')
    if k == '...':
      k = ...
    chunks[k] = _try_to_int(v)
  return chunks


def main(argv):
  target_chunks = _parse_chunks_flag(TARGET_CHUNKS.value)

  target_shards = (
      _parse_chunks_flag(TARGET_SHARDS.value)
      if TARGET_SHARDS.value is not None
      else None
  )

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    root |= (
        xbeam.Dataset.from_zarr(INPUT_PATH.value, split_vars=True)
        .rechunk(target_chunks if target_shards is None else target_shards)
        .to_zarr(
            OUTPUT_PATH.value,
            zarr_chunks=target_chunks,
            zarr_shards=target_shards,
            zarr_format=ZARR_FORMAT.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
