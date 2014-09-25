import logging
logging.basicConfig(level=logging.ERROR)
import util.ee_authenticate
util.ee_authenticate.initialize()
import matplotlib
matplotlib.use('tkagg')

import sys
import argparse
import functools
from pprint import pprint

import ee

def get_image_collection(bounds, start_date, end_date):
    ee_bounds = apply(ee.Geometry.Rectangle, bounds)
    ee_points = map(ee.Geometry.Point, [(bounds[0], bounds[1]), (bounds[0], bounds[3]),
                                    (bounds[2], bounds[1]), (bounds[2], bounds[3])])
    collection = ee.ImageCollection('LT5_L1T').filterDate(start_date, end_date).filterBounds(ee_points[0]).filterBounds(ee_points[1]).filterBounds(ee_points[2]).filterBounds(ee_points[3])
    return collection

def detect_clouds(im):
    cloud_mask = im.select(['B3']).gte(35).select(['B3'], ['cloud'])
    NDSI = (im.select(['B2']).subtract(im.select(['B5']))).divide(im.select(['B2']).add(im.select(['B5'])))
    # originally 0.4, but this supposedly misses some clouds, used 0.7 in paper
    cloud_mask = cloud_mask.And(NDSI.lte(0.4))
    # should be 300K temperature, what is this in pixel values?
    cloud_mask = cloud_mask.And(im.select(['B6']).lte(105))
    return cloud_mask

def detect_water(image, clouds):
    # from "Water Body Detection and Delineation with Landsat TM Data" by Frazier and Page
    water = image.select(['B1']).lte(100).And(image.select(['B2']).lte(55)).And(image.select(['B3']).lte(71))
    water = water.select(['B1'], ['water'])
    # originally B7 <= 13
    water = water.And(image.select(['B4']).lte(66)).And(image.select(['B5']).lte(47)).And(image.select(['B7']).lte(22))
    water = water.And(clouds.Not())
    return water

def count_water(bounds, image):
    print 'count'
    clouds = detect_clouds(image)
    water = detect_water(image, clouds)
    cloud_count = clouds.mask(clouds).reduceRegion(ee.Reducer.count(), bounds, 30)
    water_count = water.mask(water).reduceRegion(ee.Reducer.count(), bounds, 30)
    return ee.Feature(None, {'date' : image.get('DATE_ACQUIRED'),
            'water_count' : water_count.get('water'),
            'cloud_count' : cloud_count.get('cloud')})


def measure_clouds(image):
    return ee.Feature(None, {'value' : 5.0})

parser = argparse.ArgumentParser(description='Measure lake water levels.')
parser.add_argument('--date', dest='date', action='store', required=False, default=None)
args = parser.parse_args()

if args.date == None:
    start_date = ee.Date('1984-01-01')
    end_date = ee.Date('2030-01-01')
else:
    start_date = ee.Date(args.date)
    end_date = start_date.advance(1.0, 'month')

#start_date = ee.Date('2011-06-01') # lake high
#start_date = ee.Date('1993-07-01') # lake low
#start_date = ee.Date('1993-06-01') # lake low but some jet streams

bounds = (-119.2, 37.9,-118.85, 38.1)
collection = get_image_collection(bounds, start_date, end_date)
ee_bounds = apply(ee.Geometry.Rectangle, bounds)

# display individual image from a date
if args.date:
    from util.mapclient_qt import centerMap, addToMap
    landsat = ee.Image(collection.first())
    centerMap((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2, 11)
    addToMap(landsat, {'bands': ['B3', 'B2', 'B1']}, 'Landsat 3,2,1 RGB')
    addToMap(landsat, {'bands': ['B7', 'B5', 'B4']}, 'Landsat 7,5,4 RGB', False)
    addToMap(landsat, {'bands': ['B6']}, 'Landsat 6', False)
    clouds = detect_clouds(landsat)
    water = detect_water(landsat, clouds)
    addToMap(clouds.mask(clouds), {'opacity' : 0.5}, 'Cloud Mask')
    addToMap(water.mask(water), {'opacity' : 0.5, 'palette' : '00FFFF'}, 'Water Mask')
    print count_water(ee_bounds, landsat).getInfo()
# compute water levels in all images of area
else:
    results = collection.map(functools.partial(count_water, ee_bounds))
    #print results.toList(1000000).length().getInfo()
    #print collection.iterate(lambda v, l: l.append(v), [])

