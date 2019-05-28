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

import collections
import cStringIO
import functools
import math
import sys
import os.path
import urllib2
import uuid
import zipfile
import tempfile
from multiprocessing.pool import ThreadPool

import ee

from PIL import ImageQt
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import PyQt4
from PyQt4 import QtCore, QtGui

# The default URL to fetch tiles from.  We could pull this from the EE library,
# however this doesn't have any other dependencies on that yet, so let's not.
BASE_URL = 'https://earthengine.googleapis.com'

TEMP_FILE_DIR = tempfile.gettempdir()

class LocalEEImage(object):
    """Downloads an entire image from Earth Engine and maintains it locally."""

    # Download an image locally. Caches images with the same bbox, scale and image_name
    def __init__(self, eeobject, bbox, scale, bands, image_name=None):
        """Constructs a LocalEEImage.

        Arguments:
        eeobject   -- The ee.Image object to download.
        bbox       -- A four-tuple describing the bounds to download, (lon0, lat0, lon1, lat1).
        scale      -- The scale in meters of the image to download.
        bands      -- A list containing the names of the bands in eeobject to download.
        image_name -- A unique name for this image. If an image of the same name, bbox and
        scale was downloaded previously, a cached image may be used. If a name is not specified,
        no caching is performed.
        """

        image_name = image_name.replace(' ', '') # can't have space in filename
        if image_name == None: # Use a default name if needed
            image_name = uuid.uuid4()
        # If you get an invalid asset ID here that probably means there is a problem with image_name
        url = eeobject.getDownloadUrl({'name' : image_name, 'scale': scale, 'crs': 'EPSG:4326',
                                       'region': apply(ee.Geometry.Rectangle, bbox).toGeoJSONString()})
        tempFile = '%s_%g,%g_%g,%g_%g.zip' % (image_name, bbox[0], bbox[1], bbox[2], bbox[3], scale)
        filename = os.path.join(TEMP_FILE_DIR, tempFile) 
        if (image_name != None) and (not os.path.isfile(filename)):
            print('Downloading image...')
            data = urllib2.urlopen(url)
            with open(filename, 'wb') as fp:
                while True:
                    chunk = data.read(16 * 1024)
                    if not chunk: break
                    fp.write(chunk)
            print('Download complete!')

        # extract the zip file
        z = zipfile.ZipFile(filename, 'r')
        # read the transform from the tfw file
        transform_file = z.open(image_name + '.' + bands[0] + '.tfw', 'r')
        self.transform = [float(line) for line in transform_file]

        # Load each of the bands in to memory
        self.images = dict()
        for b in bands:
            bandfilename   = image_name + '.' + b + '.tif'
            band_file_path = os.path.join(TEMP_FILE_DIR, bandfilename)
            z.extract(bandfilename, TEMP_FILE_DIR)
            self.images[b] = plt.imread(band_file_path)
        
        self.image_name = image_name
        self.bands      = bands
        self.bbox       = bbox
        self.scale      = scale

    def image_to_global(self, r, c):
        '''Convert pixel coordinate to latitude and longitude.'''
        lng = self.transform[0] * c + self.transform[4]
        lat = self.transform[3] * r + self.transform[5]
        return (lng, lat)

    def global_to_image(self, lon, lat):
        '''Convert a latitude and longitude to a pixel coordinate.'''
        c = (lon - self.transform[4]) / self.transform[0]
        r = (lat - self.transform[5]) / self.transform[3]
        return (r, c)

    def get(self, band, r, c):
        '''Fetch the selected pixel from the selected band.'''
        return self.images[band][r, c]
    
    def get(self, r, c):
        '''Fetch the selected pixels across all bands'''
        pixel = []
        for b in self.bands:
            pixel.append(self.images[b][r, c])
        return pixel

    def get_image(self, band_name):
        '''Fetch the selected band as a PIL image.'''
        return self.images[band_name]
    
    def get_band_by_index(self, band_index):
        '''Fetch the selected band as a PIL image.'''
        return self.images[self.bands[band_index]]

    def size(self):
        '''Get the width and height of the image.'''
        return self.images[self.bands[0]].shape

