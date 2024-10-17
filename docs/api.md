# API docs

```{eval-rst}
.. currentmodule:: xarray_beam
```

## Core data model

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    Key
```

## Reading and writing data

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    open_zarr
    DatasetToChunks
    ChunksToZarr
    DatasetToZarr
    make_template
    setup_zarr
    validate_zarr_chunk
    write_chunk_to_zarr
```

## Aggregation

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    Mean
    Mean.Globally
    Mean.PerKey
    MeanCombineFn
```

## Rechunking

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    ConsolidateChunks
    ConsolidateVariables
    SplitChunks
    SplitVariables
    Rechunk
```

## Utility transforms

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    ValidateEachChunk
```

## Utility functions

```{eval-rst}
.. autosummary::
    :toctree: _autosummary

    offsets_to_slices
    validate_chunk
    consolidate_chunks
    consolidate_variables
    consolidate_fully
    split_chunks
    split_variables
    in_memory_rechunk
```