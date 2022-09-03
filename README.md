# Xarray-Beam

Xarray-Beam is a Python library for building
[Apache Beam](https://beam.apache.org/) pipelines with
[Xarray](http://xarray.pydata.org/en/stable/) datasets.

The project aims to facilitate data transformations and analysis on large-scale
multi-dimensional labeled arrays, such as:

-   Ad-hoc computation on Xarray data, by dividing a `xarray.Dataset` into many
    smaller pieces ("chunks").
-   Adjusting array chunks, using the
    [Rechunker algorithm](https://rechunker.readthedocs.io/en/latest/algorithm.html).
-   Ingesting large, multi-dimensional array datasets into an analysis-ready,
    cloud-optimized format, namely [Zarr](https://zarr.readthedocs.io/) (see
    also [Pangeo Forge](https://github.com/pangeo-forge/pangeo-forge-recipes)).
-   Calculating statistics (e.g., "climatology") across distributed datasets
    with arbitrary groups.

For more about our approach and how to get started,
**[read the documentation](https://xarray-beam.readthedocs.io/)**!

**Warning: Xarray-Beam is a sharp tool 🔪**

Xarray-Beam is relatively new, and focused on expert users:

-   We use it extensively at Google for processing large-scale weather datasets,
    but there is not yet a vibrant external community.
-   It provides low-level abstractions that facilitate writing very large
    scale data pipelines (e.g., 100+ TB), but by design it requires explicitly
    thinking about how every operation is parallelized.

## Installation

Xarray-Beam requires recent versions of immutabledict, Xarray, Dask, Rechunker,
Zarr, and Apache Beam. For best performance when writing Zarr files, use Xarray
0.19.0 or later.

## Disclaimer

Xarray-Beam is an experiment that we are sharing with the outside world in the
hope that it will be useful. It is not a supported Google product. We welcome
feedback, bug reports and code contributions, but cannot guarantee they will be
addressed.

See the "Contribution guidelines" for more.

## Credits

Contributors:

-   Stephan Hoyer
-   Jason Hickey
-   Cenk Gazen
-   Alex Merose
