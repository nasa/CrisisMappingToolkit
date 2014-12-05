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

import logging
logging.basicConfig(level=logging.ERROR)
try:
    import cmt.ee_authenticate
except:
    import sys
    import os.path
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    import cmt.ee_authenticate
cmt.ee_authenticate.initialize()
import matplotlib
matplotlib.use('tkagg')

import sys
import argparse
import functools
import time
import threading
import os
import os.path
from pprint import pprint

import ee

def get_image_collection(bounds, start_date, end_date):
    #ee_bounds = apply(ee.Geometry.Rectangle, bounds)
    #ee_points = map(ee.Geometry.Point, [(bounds[0], bounds[1]), (bounds[0], bounds[3]),
    #                                (bounds[2], bounds[1]), (bounds[2], bounds[3])])
    ee_bounds = bounds
    ee_points = ee.List(bounds.bounds().coordinates().get(0))
    points = ee_points.getInfo()
    points = map(functools.partial(apply, ee.Geometry.Point), points)
    collection = ee.ImageCollection('LT5_L1T').filterDate(start_date, end_date).filterBounds(points[0]).filterBounds(points[1]).filterBounds(points[2]).filterBounds(points[3])
    return collection

def detect_clouds(im):
    cloud_mask = im.select(['B3']).gte(35).select(['B3'], ['cloud']).And(im.select(['B6']).lte(120))
    NDSI = (im.select(['B2']).subtract(im.select(['B5']))).divide(im.select(['B2']).add(im.select(['B5'])))
    # originally 0.4, but this supposedly misses some clouds, used 0.7 in paper
    # be conservative
    #cloud_mask = cloud_mask.And(NDSI.lte(0.7))
    # should be 300K temperature, what is this in pixel values?
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

def parse_lake_data(filename):
    f = open(filename, 'r')
    # take values with low cloud cover
    f.readline()
    f.readline()
    f.readline()
    results = dict()
    for l in f:
        parts = l.split(',')
        date = parts[0].strip()
        satellite = parts[1].strip()
        cloud = int(parts[2])
        water = int(parts[3])
        sun_elevation = float(parts[4])
        if not results.has_key(satellite):
            results[satellite] = dict()
        results[satellite][date] = (cloud, water, sun_elevation)
    
    f.close()
    return results

def process_lake(lake, ee_lake, start_date, end_date, output_directory):
    name = lake['properties']['LAKE_NAME']
    if name == '':
        return
    output_file_name = os.path.join(output_directory, name + '.txt')
    data = None
    if os.path.exists(output_file_name):
        data = parse_lake_data(output_file_name)
    f = open(output_file_name, 'w')
    country = lake['properties']['COUNTRY']
    area = lake['properties']['AREA_SKM']
    #print '%s, %s, %s' % (name, country, area)
    f.write('# Name     Country    Area in km^2\n')
    f.write('%s, %s, %s\n' % (name, country, area))
    f.write('# Date, Satellite, Cloud Pixels, Water Pixels, Sun Elevation\n')
    if data != None:
        for sat in sorted(data.keys()):
            for date in sorted(data[sat].keys()):
                f.write('%s, %10s, %10d, %10d, %.5g\n' % (date, sat, data[sat][date][0], data[sat][date][1], data[sat][date][2]))
    try:
        ee_bounds = ee_lake.geometry().buffer(1000)
        collection = get_image_collection(ee_bounds, start_date, end_date)
        v = collection.toList(1000000)
    except:
        print >> sys.stderr, 'Failed to allocate memory to expand buffer for lake %s, skipping.' % (name)
        f.close()
        return

    results = []
    all_images = v.getInfo()
    for i in range(len(all_images)):
        if data != None and 'Landsat 5' in data.keys() and all_images[i]['properties']['DATE_ACQUIRED'] in data['Landsat 5']:
            continue
        im = ee.Image(v.get(i))
        sun_elevation = all_images[i]['properties']['SUN_ELEVATION']
        try:
            r = count_water(ee_bounds, im).getInfo()['properties']
        except Exception as e:
            print >> sys.stderr, 'Failure counting water...trying again. ' + str(e)
            time.sleep(5)
            r = count_water(ee_bounds, im).getInfo()['properties']
        output = '%s, %10s, %10d, %10d, %.5g' % (r['date'], 'Landsat 5', r['cloud_count'], r['water_count'], sun_elevation)
        print '%15s %s' % (name, output)
        f.write(output + '\n')
        results.append(r)
    f.close()

global_semaphore = threading.Semaphore(8)
thread_lock = threading.Lock()
total_threads = 0

class LakeThread(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.args = args
        thread_lock.acquire()
        global total_threads
        total_threads += 1
        thread_lock.release()
        self.start()
    def run(self):
        global_semaphore.acquire()
        try:
            apply(process_lake, self.args)
        except Exception as e:
            print >> sys.stderr, e
        global_semaphore.release()
        thread_lock.acquire()
        global total_threads
        total_threads -= 1
        thread_lock.release()

parser = argparse.ArgumentParser(description='Measure lake water levels.')
parser.add_argument('--date', dest='date', action='store', required=False, default=None)
parser.add_argument('--lake', dest='lake', action='store', required=False, default=None)
parser.add_argument('--results_dir', dest='results_dir', action='store', required=False, default='results')
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

#all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").toList(1000000)
if args.lake != None:
    all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").filterMetadata(u'LAKE_NAME', u'equals', args.lake).toList(1000000)
else:
    #bounds = ee.Geometry.Rectangle(-125.29, 32.55, -114.04, 42.02)
    #all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").filterBounds(bounds).toList(1000000)
    all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").toList(1000000)
         #.filterMetadata(u'AREA_SKM', u'less_than', 300.0).toList(100000)#.filterMetadata(
         #u'LAT_DEG', u'less_than',   42.02).filterMetadata( u'LAT_DEG', u'greater_than', 32.55).filterMetadata(
         #u'LONG_DEG', u'less_than', -114.04).filterMetadata(u'LONG_DEG', u'greater_than', -125.29).toList(1000000)
    #pprint(ee.Feature(all_lakes.get(0)).getInfo())

# display individual image from a date
if args.date:
    lake = all_lakes.get(0).getInfo()
    ee_lake = ee.Feature(all_lakes.get(0))
    from cmt.mapclient_qt import centerMap, addToMap
    ee_bounds = ee_lake.geometry().buffer(1000)
    collection = get_image_collection(ee_bounds, start_date, end_date)
    landsat = ee.Image(collection.first())
    #pprint(landsat.getInfo())
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
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    all_lakes_local = all_lakes.getInfo()
    for i in range(len(all_lakes_local)):
        ee_lake = ee.Feature(all_lakes.get(i))
        LakeThread((all_lakes_local[i], ee_lake, start_date, end_date, args.results_dir))
    while True:
        thread_lock.acquire()
        if total_threads == 0:
            thread_lock.release()
            break
        thread_lock.release()
        time.sleep(0.1)

