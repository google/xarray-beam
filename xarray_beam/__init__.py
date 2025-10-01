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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570
# pylint: disable=g-multiple-import,useless-import-alias,g-importing-member
from xarray_beam._src.combiners import (
    Mean as Mean,
    MeanCombineFn as MeanCombineFn,
)
from xarray_beam._src.core import (
    Key as Key,
    DatasetToChunks as DatasetToChunks,
    ValidateEachChunk as ValidateEachChunk,
    offsets_to_slices as offsets_to_slices,
    validate_chunk as validate_chunk,
)
from xarray_beam._src.dataset import (
    Dataset as Dataset,
    normalize_chunks as normalize_chunks,
)
from xarray_beam._src.rechunk import (
    ConsolidateChunks as ConsolidateChunks,
    ConsolidateVariables as ConsolidateVariables,
    SplitChunks as SplitChunks,
    SplitVariables as SplitVariables,
    Rechunk as Rechunk,
    consolidate_chunks as consolidate_chunks,
    consolidate_variables as consolidate_variables,
    consolidate_fully as consolidate_fully,
    split_chunks as split_chunks,
    split_variables as split_variables,
    in_memory_rechunk as in_memory_rechunk,
)
from xarray_beam._src.zarr import (
    open_zarr as open_zarr,
    make_template as make_template,
    replace_template_dims as replace_template_dims,
    setup_zarr as setup_zarr,
    validate_zarr_chunk as validate_zarr_chunk,
    write_chunk_to_zarr as write_chunk_to_zarr,
    ChunksToZarr as ChunksToZarr,
    DatasetToZarr as DatasetToZarr,
)

__version__ = '0.10.2'  # automatically synchronized to pyproject.toml
