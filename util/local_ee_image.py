import collections
import cStringIO
import functools
import math
import sys
import os.path
import urllib2
import uuid
import zipfile
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

class LocalTiledImage(object):
    """A class representing a map overlay."""

    TILE_WIDTH = 256
    TILE_HEIGHT = 256
    MAX_TILES = 100                    # The maximum number of tiles to download

    def __init__(self, eeobject, bbox, level, vis_params=None):
        # Flatten any lists to comma separated strings.
        if vis_params:
            vis_params = dict(vis_params)
            for key in vis_params.keys():
                item = vis_params.get(key)
                if (isinstance(item, collections.Iterable) and
                        not isinstance(item, basestring)):
                    vis_params[key] = ','.join([str(x) for x in item])
        self.mapid = eeobject.getMapId(vis_params)
        self.url = (BASE_URL + '/map/' + self.mapid['mapid'] + '/%d/%d/%d?token=' + self.mapid['token'])

        self.bbox = bbox
        self.level = level
        self.__download_image()

    def image_to_global(self, x, y):
        mercator_range = 256.0
        scale = 2 ** self.level
        origin_x = (mercator_range / 2.0) * scale
        origin_y = (mercator_range / 2.0) * scale
        pixels_per_lon_degree = (mercator_range / 360.0) * scale
        pixels_per_lon_radian = (mercator_range / (2 * math.pi)) * scale
        lng = (x - self.origin_x - origin_x) / pixels_per_lon_degree
        latRadians = (y - self.origin_y - origin_y) / -pixels_per_lon_radian
        lat = (2 * math.atan(math.exp(latRadians)) - math.pi / 2) / (math.pi / 180.0)
        return (lng, lat)

    def global_to_image(self, lon, lat):
        # From maps/api/javascript/geometry/mercator_projection.js
        mercator_range = 256.0
        scale = 2 ** self.level
        origin_x = (mercator_range / 2.0) * scale
        origin_y = (mercator_range / 2.0) * scale
        pixels_per_lon_degree = (mercator_range / 360.0) * scale
        pixels_per_lon_radian = (mercator_range / (2 * math.pi)) * scale

        x = origin_x + (lon * pixels_per_lon_degree)
        siny = math.sin(lat * math.pi / 180.0)
        # Prevent sin() overflow.
        e = 1 - 1e-15
        if siny > e:
            siny = e
        elif siny < -e:
            siny = -e
        y = origin_y + (0.5 * math.log((1 + siny) / (1 - siny)) *
                                        -pixels_per_lon_radian)
        return (x, y)

    def get(self, x, y):
        try:
            (tile, tilex, tiley) = self.__get_tile(x, y)
        except:
            return None
        return tile.getpixel(x - tilex * self.TILE_WIDTH, y - tiley * self.TILE_HEIGHT)

    def __get_tile(self, x, y):
        tilex = int(x / self.TILE_WIDTH)
        tiley = int(y / self.TILE_HEIGHT)
        key = (level, tilex, tiley)
        if not self.tiles.has_key(key):
            print >> sys.stderr, "Tile not found."
            return None
        return (self.tiles[key], tilex, tiley)

    def get_tiles(self):
        return self.tiles

    def __download_image(self):
        tile_list = self.create_tile_queue(self.level, self.bbox)
        pool = ThreadPool(10)
        print 'Downloading tiles.'
        tiles = pool.map(self.__download_tile, tile_list)
        print 'Downloaded tiles.'
        self.tiles = dict(zip(tile_list, tiles))
    
    def create_tile_queue(self, level, orig_bbox):
        topleft     = self.global_to_image(orig_bbox[0], orig_bbox[3])
        bottomright = self.global_to_image(orig_bbox[2], orig_bbox[1])
        bbox = [topleft[0], topleft[1], bottomright[0], bottomright[1]]
        print bbox

        tile_list = []
        for y in xrange(int(bbox[1] / self.TILE_HEIGHT), int(bbox[3] / self.TILE_HEIGHT + 1)):
            for x in xrange(int(bbox[0] / self.TILE_WIDTH), int(bbox[2] / self.TILE_WIDTH + 1)):
                tile_list.append((level, x, y))
        return tile_list

    def __download_tile(self, key):       # pylint: disable=g-bad-name
        (level, x, y) = key
        if x >= 0 and y >= 0 and x <= 2 ** level-1 and y <= 2 ** level-1:
            url = self.url % key
            try:
                data = urllib2.urlopen(url).read()
            except urllib2.HTTPError as e:
                print >> sys.stderr, e
            # PhotoImage can't handle alpha on LA images.
            image = Image.open(cStringIO.StringIO(data)).convert('RGBA')
            return image
        else:
            print >> sys.stderr, "Invalid tile."

class LocalEEImage(object):
    """A class representing a map overlay."""

    TILE_WIDTH = 256
    TILE_HEIGHT = 256
    MAX_TILES = 100                    # The maximum number of tiles to download

    # download an image locally. Caches images with the same bbox, scale and image_name
    def __init__(self, eeobject, bbox, scale, bands, image_name=None):
        if image_name == None:
            image_name = uuid.uuid4()
        url = eeobject.getDownloadUrl({'name' : image_name, 'scale': scale, 'crs': 'EPSG:4326',
                  'region': apply(ee.Geometry.Rectangle, bbox).toGeoJSONString()})
        filename = '/tmp/%s_%g,%g_%g,%g_%g.zip' % (image_name, bbox[0], bbox[1], bbox[2], bbox[3], scale)
        if not os.path.isfile(filename):
            print 'Downloading image...'
            data = urllib2.urlopen(url)
            with open(filename, 'wb') as fp:
                while True:
                    chunk = data.read(16 * 1024)
                    if not chunk: break
                    fp.write(chunk)
            print 'Download complete!'

        z = zipfile.ZipFile(filename, 'r')
        transform_file = z.open(image_name + '.' + bands[0] + '.tfw', 'r')
        self.transform = [float(line) for line in transform_file]

        self.images = dict()
        for b in bands:
            bandfilename = image_name + '.' + b + '.tif'
            z.extract(bandfilename, '/tmp')
            self.images[b] = plt.imread('/tmp/' + bandfilename)
        
        self.image_name = image_name
        self.bands = bands
        self.bbox = bbox
        self.scale = scale

    def image_to_global(self, r, c):
        lng = self.transform[0] * c + self.transform[4]
        lat = self.transform[2] * r + self.transform[5]
        return (lng, lat)

    def global_to_image(self, lon, lat):
        c = (lon - self.transform[4]) / self.transform[0]
        r = (lat - self.transform[5]) / self.transform[2]
        return (r, c)

    def get(self, band, r, c):
        return self.images[band][r, c]

    def get_image(self, band):
        return self.images[band]

    def size(self):
        return self.images[self.bands[0]].shape

