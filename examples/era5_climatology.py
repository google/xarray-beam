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
"""Calculate climatology for the Pangeo ERA5 surface dataset."""
from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray
import xarray_beam


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')

FLAGS = flags.FLAGS

# pylint: disable=expression-not-assigned


def rekey_chunk_on_month_hour(key, dataset):
  month = dataset.time.dt.month.item()
  hour = dataset.time.dt.hour.item()
  new_key = key - {'time'} | {'month': month - 1, 'hour': hour}
  new_dataset = (
      dataset
      .squeeze('time', drop=True)
      .expand_dims(month=[month], hour=[hour])
  )
  return new_key, new_dataset


def main(argv):
  # By passing chunks=None, we use Xarray's lazy-loading instead of Dask. This
  # result is much less data being passed from the launch script to workers.
  source_dataset = xarray.open_zarr(
      INPUT_PATH.value, chunks=None, consolidated=True,
  )

  # This lazy "template" allows us to setup the Zarr outputs before running the
  # pipeline. We don't really need to supply a template here because the outputs
  # are small (the template argument in ChunksToZarr is optional), but it makes
  # the pipeline slightly more efficient.
  template = (
      source_dataset
      .isel(time=0, drop=True)
      .pipe(xarray.zeros_like)  # don't load even time=0 into memory
      .expand_dims(month=np.arange(12)+1, hour=np.arange(24))
      .chunk({'hour': 1, 'month': 1})  # make lazy with dask
      .pipe(xarray.zeros_like)  # compress the dask graph
  )

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    (
        root
        | xarray_beam.DatasetToChunks(source_dataset, {'time': 31})
        | xarray_beam.SplitChunks({'time': 1})
        | beam.MapTuple(rekey_chunk_on_month_hour)
        | xarray_beam.Mean.PerKey(dtype=np.float64)  # avoid overflow
        | beam.MapTuple(lambda k, v: (k, v.astype(np.float32)))
        | xarray_beam.ChunksToZarr(OUTPUT_PATH.value, template)
    )


if __name__ == '__main__':
  app.run(main)
