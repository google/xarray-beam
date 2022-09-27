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
"""Rechunk the WeatherBench ERA5 dataset from images to time-series."""
from absl import app
from absl import flags
import apache_beam as beam
import xarray_beam as xbeam


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


# pylint: disable=expression-not-assigned


def main(argv):
  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)
  template = xbeam.make_template(source_dataset)
  target_chunks = {'latitude': 5, 'longitude': 5, 'time': -1}

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    (
        root
        # Note: splitting across the 19 variables in this dataset is a critical
        # optimization step here, because it allows rechunking to make use of
        # much larger intermediate chunks.
        | xbeam.DatasetToChunks(source_dataset, source_chunks, split_vars=True)
        | xbeam.Rechunk(
            source_dataset.sizes,
            source_chunks,
            target_chunks,
            itemsize=4,
        )
        | xbeam.ChunksToZarr(OUTPUT_PATH.value, template, target_chunks)
    )


if __name__ == '__main__':
  app.run(main)
