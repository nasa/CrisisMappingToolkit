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
	#ee_bounds = apply(ee.Geometry.Rectangle, bounds)
	#ee_points = map(ee.Geometry.Point, [(bounds[0], bounds[1]), (bounds[0], bounds[3]),
	#								(bounds[2], bounds[1]), (bounds[2], bounds[3])])
	ee_bounds = bounds
	ee_points = ee.List(bounds.bounds().coordinates().get(0))
	points = ee_points.getInfo()
	points = map(functools.partial(apply, ee.Geometry.Point), points)
	collection = ee.ImageCollection('LT5_L1T').filterDate(start_date, end_date).filterBounds(points[0]).filterBounds(points[1]).filterBounds(points[2]).filterBounds(points[3])
	return collection

def detect_clouds(im):
	cloud_mask = im.select(['B3']).gte(35).select(['B3'], ['cloud'])
	NDSI = (im.select(['B2']).subtract(im.select(['B5']))).divide(im.select(['B2']).add(im.select(['B5'])))
	# originally 0.4, but this supposedly misses some clouds, used 0.7 in paper
	# be conservative
	#cloud_mask = cloud_mask.And(NDSI.lte(0.7))
	# should be 300K temperature, what is this in pixel values?
	cloud_mask = cloud_mask.And(im.select(['B6']).lte(100))
	return cloud_mask

def detect_water(image, clouds):
	# from "Water Body Detection and Delineation with Landsat TM Data" by Frazier and Page
	water = image.select(['B1']).lte(100).And(image.select(['B2']).lte(55)).And(image.select(['B3']).lte(71))
	water = water.select(['B1'], ['water'])
	# originally B7 <= 13
	water = water.And(image.select(['B4']).lte(66)).And(image.select(['B5']).lte(47)).And(image.select(['B7']).lte(20))
	# original was B1 57, 23, 21, 14
	water = water.And(image.select(['B1']).gte(30))#.And(image.select(['B2']).gte(10)).And(image.select(['B3']).gte(8))
	#water = water.And(image.select(['B4']).gte(5)).And(image.select(['B5']).gte(2)).And(image.select(['B7']).gte(1))
	water = water.And(clouds.Not())
	return water

def count_water(bounds, image):
	clouds = detect_clouds(image)
	water = detect_water(image, clouds)
	cloud_count = clouds.mask(clouds).reduceRegion(ee.Reducer.count(), bounds, 30)
	#addToMap(ee.Algorithms.ConnectedComponentLabeler(water, ee.Kernel.square(1), 256))
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
	#start_date = ee.Date('1984-01-01')
	start_date = ee.Date('1999-09-20')
	end_date = ee.Date('2030-01-01')
else:
	start_date = ee.Date(args.date)
	end_date = start_date.advance(1.0, 'month')

#start_date = ee.Date('2011-06-01') # lake high
#start_date = ee.Date('1993-07-01') # lake low
#start_date = ee.Date('1993-06-01') # lake low but some jet streams

#all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").toList(1000000)
all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").filterMetadata(u'LAKE_NAME', u'equals', u'Mono Lake').toList(1000000)
lake_num = 0
lake = ee.Feature(all_lakes.get(lake_num))
#print lake.get('TYPE').getInfo()
#print lake.get('name').getInfo()
#print lake.get('COUNTRY').getInfo()
#print lake.get('AREA_SKM').getInfo()
ee_bounds = lake.geometry().buffer(1000)

#bounds = (-119.2, 37.9,-118.85, 38.1)
#ee_bounds = apply(ee.Geometry.Rectangle, bounds)
collection = get_image_collection(ee_bounds, start_date, end_date)

# display individual image from a date
if args.date:
	from util.mapclient_qt import centerMap, addToMap
	landsat = ee.Image(collection.first())
	center = ee_bounds.centroid().getInfo()['coordinates']
	centerMap(center[0], center[1], 11)
	addToMap(landsat, {'bands': ['B3', 'B2', 'B1']}, 'Landsat 3,2,1 RGB')
	addToMap(landsat, {'bands': ['B7', 'B5', 'B4']}, 'Landsat 7,5,4 RGB', False)
	addToMap(landsat, {'bands': ['B6']}, 'Landsat 6', False)
	clouds = detect_clouds(landsat)
	water = detect_water(landsat, clouds)
	addToMap(clouds.mask(clouds), {'opacity' : 0.5}, 'Cloud Mask')
	addToMap(water.mask(water), {'opacity' : 0.5, 'palette' : '00FFFF'}, 'Water Mask')
	addToMap(ee.Feature(ee_bounds))
	#print count_water(ee_bounds, landsat).getInfo()
# compute water levels in all images of area
else:
	v = collection.toList(1000000)
	name = lake.get('LAKE_NAME').getInfo()
	country = lake.get('COUNTRY').getInfo()
	area = lake.get('AREA_SKM').getInfo()
	print '%s, %s, %s' % (name, country, area)
	results = []
	for i in range(v.length().getInfo()):
		im = ee.Image(v.get(i))
		r = count_water(ee_bounds, im).getInfo()['properties']
		print '%s: %10d Cloud, %10d Water, %10d Total' % (r['date'], r['cloud_count'], r['water_count'], r['cloud_count'] + r['water_count'])
		results.append(r)
	#v = ee.List(collection.iterate(lambda v, l: ee.List(l).add(count_water(ee_bounds, v)), ee.List([])))
	#print v.get(0).getInfo()
	#all_images = collection.toList(10000000).getInfo()
	#for im in all_images:
	#	print im.keys()
	#	print count_water(bounds, im)
	#results = collection.map(functools.partial(count_water, ee_bounds))
	#print results.toList(1000000).length().getInfo()
	#print collection.iterate(lambda v, l: l.append(v), [])

