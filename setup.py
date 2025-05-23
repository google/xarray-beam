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
# ==============================================================================
"""Setup Xarray-Beam."""
import setuptools


base_requires = [
    'apache_beam>=2.31.0',
    'dask',
    'immutabledict',
    'rechunker>=0.5.1',
    'zarr',
    'xarray',
]
docs_requires = [
    'myst-nb',
    'myst-parser',
    'sphinx',
    'sphinx_rtd_theme',
    'scipy',
]
tests_requires = [
    'absl-py',
    'pandas',
    'pytest',
    'scipy',
    'h5netcdf'
]

setuptools.setup(
    name='xarray-beam',
    version='0.8.1',  # keep in sync with __init__.py
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    install_requires=base_requires,
    extras_require={
        'tests': tests_requires,
        'docs': docs_requires,
    },
    url='https://github.com/google/xarray-beam',
    packages=setuptools.find_packages(exclude=['examples']),
    python_requires='>=3',
)
