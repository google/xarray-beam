# Xarray-Beam: distributed Xarray with Apache Beam

Xarray-Beam is a library for writing [Apache Beam](http://beam.apache.org/) pipelines consisting of [xarray](http://xarray.pydata.org) Dataset objects. This documentation (and Xarray-Beam itself) assumes basic familiarity with both Beam and Xarray.

The documentation includes narrative documentation that will walk you through the basics of writing a pipeline with Xarray-Beam, and also comprehensive API docs.

We recommend reading both, as well as a few [end to end examples](https://github.com/google/xarray-beam/tree/main/examples) to understand what code using Xarray-Beam typically looks like.

## Contents

```{toctree}
:maxdepth: 1
why-xarray-beam.md
data-model.ipynb
read-write.ipynb
aggregation.ipynb
rechunking.ipynb
api.md
```