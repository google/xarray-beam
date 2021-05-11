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


INSTALL_REQUIRES = [
    'apache_beam',
    'dask',
    'rechunker',
    'zarr',
    'xarray',
]

setuptools.setup(
    name='xarray-beam',
    version='0.0.1',
    license='Apache 2.0',
    author='Google LLC',
    author_email='noreply@google.com',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/google/xarray-beam',
    packages=setuptools.find_packages(),
    python_requires='>=3',
)
