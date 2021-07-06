#!/usr/bin/python
#
# -----------------------------------------------------------------------------
# Copyright * 2014, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.
#
# The Crisis Mapping Toolkit (CMT) v1 platform is licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# -----------------------------------------------------------------------------
import os

try:
    # if setuptools is available, use it to take advantage of its dependency
    # handling
    from setuptools import setup                          # pylint: disable=g-import-not-at-top
except ImportError:
    # if setuptools is not available, use distutils (standard library). Users
    # will receive errors for missing packages
    from distutils.core import setup                      # pylint: disable=g-import-not-at-top

try:
    import PyQt4
except ImportError:
    print("""
            WARNING: PyQt4 is required to use the Crisis Mapping Toolkit.
            Please install PyQt4 on your system.
          """)


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='crisismappingtoolkit',
    version=1.0,
    description='Crisis Mapping Toolkit',
    license='Apache2',
    url='https://github.com/bcoltin/CrisisMappingToolkit',
    download_url='',  # package download URL
    packages=['cmt', 'cmt.modis', 'cmt.radar', 'cmt.util'],
    install_requires=[
        'earthengine-api',
        'pillow'
    ],
    classifiers=[
        # Get strings from
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python',
        'Operating System :: OS Independent',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Graphics :: Viewers',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='earth engine image analysis',
    author='Brian Coltin',
    author_email='brian.j.coltin@nasa.gov',
    long_description=read('README.md')
)
