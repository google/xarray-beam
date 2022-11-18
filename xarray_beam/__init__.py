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
"""Public API for Xarray-Beam."""

# pylint: disable=g-multiple-import
from xarray_beam import Mean
from xarray_beam._src.combiners import (
    MeanCombineFn,
)
from xarray_beam._src.core import (
    Key,
    DatasetToChunks,
    ValidateEachChunk,
    offsets_to_slices,
    validate_chunk
)
from xarray_beam._src.rechunk import (
    ConsolidateChunks,
    ConsolidateVariables,
    SplitChunks,
    SplitVariables,
    Rechunk,
    consolidate_chunks,
    consolidate_variables,
    consolidate_fully,
    split_chunks,
    split_variables,
    in_memory_rechunk,
)
from xarray_beam._src.zarr import (
    open_zarr,
    make_template,
    ChunksToZarr,
    DatasetToZarr,
)

__version__ = '0.5.1'
