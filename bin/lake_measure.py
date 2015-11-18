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

from datetime import datetime as dt
import sys
import argparse
import functools
import time
import threading
import os
import os.path
from pprint import pprint
import ctypes

import ee
ee.Initialize()

cloudThresh = 0.35
#snowThresh  = 0.05
collection_dict = {'L8': 'LANDSAT/LC8_L1T_TOA',
                   'L7': 'LANDSAT/LE7_L1T_TOA',
                   'L5': 'LANDSAT/LT5_L1T_TOA'
                   }

sensor_band_dict = ee.Dictionary({'L8': ee.List([1, 2, 3, 4, 5, 9, 6]),
                                  'L7': ee.List([0, 1, 2, 3, 4, 5, 7]),
                                  'L5': ee.List([0, 1, 2, 3, 4, 5, 6]),
                                  })

spacecraft_dict = {'Landsat5': 'L5', 'Landsat7': 'L7', 'LANDSAT_8': 'L8'}
spacecraft_strdict = {'Landsat5': 'Landsat 5', 'Landsat7': 'Landsat 7', 'LANDSAT_8': 'Landsat 8'}

bandNames = ee.List(['blue','green','red','nir','swir1','temp','swir2'])
bandNumbers = [0,1,2,3,4,5,6]
possibleSensors = ee.List(['L5','L7','L8'])

def getCollection(sensor,bounds,startDate,endDate):
    global collection_dict, sensor_band_dict, bandNames
    ee_bounds = bounds
    collectionName = collection_dict.get(sensor)
    # Start with an un-date-confined collection of images
    WOD = ee.ImageCollection(collectionName).filterBounds(ee_bounds)
    # Filter by the dates
    ls = WOD.filterDate(startDate,endDate)
    ls = ls.select(sensor_band_dict.get(sensor),bandNames)
    return ls

def get_image_collection(bounds, start_date, end_date):
    '''Retrieve Landsat 5 imagery for the selected location and dates'''
    # ee_bounds = apply(ee.Geometry.Rectangle, bounds)
    # ee_points = map(ee.Geometry.Point, [(bounds[0], bounds[1]), (bounds[0], bounds[3]),
    #                 (bounds[2], bounds[1]), (bounds[2], bounds[3])])
    global possibleSensors
    l5s = ee.ImageCollection(ee.Algorithms.If(possibleSensors.contains('L5'),getCollection('L5',bounds,start_date, end_date),getCollection('L5',bounds,ee.Date('1000-01-01'),ee.Date('1001-01-01'))))
    #l7s = ee.ImageCollection(ee.Algorithms.If(possibleSensors.contains('L7'),getCollection('L7',bounds,start_date, end_date),getCollection('L7',bounds,ee.Date('1000-01-01'),ee.Date('1001-01-01'))))
    l8s = ee.ImageCollection(ee.Algorithms.If(possibleSensors.contains('L8'),getCollection('L8',bounds,start_date, end_date),getCollection('L8',bounds,ee.Date('1000-01-01'),ee.Date('1001-01-01'))))
    ls = ee.ImageCollection(l5s.merge(l8s))
    #Clips image to rectangle around buffer. Thought this would free-up memory by reducing image size, but it doesn't
    # seem too :(
    #rect = bounds.bounds().getInfo()
    #ls = ls.map(lambda img: img.clip(rect))
    return ls


# def detect_clouds(im):
#     '''Cloud detection algorithm for Landsat 5 data'''
#     cloudThresh = 20 #Lower means more clouds excluded.
#     cloud_mask = ee.Algorithms.Landsat.simpleCloudScore(im).select(['cloud']).gt(cloudThresh)
#     cloud_mask = cloud_mask.Or(im.select(['red']).eq(0))# Check for scan lines in LS7.
#     # originally 0.4, but this supposedly misses some clouds, used 0.7 in paper
#     # be conservative
#     # cloud_mask = cloud_mask.And(NDSI.lte(0.7))
#     # should be 300K temperature, what is this in pixel values?
#     return cloud_mask

def rescale(img, exp, thresholds):
    return img.expression(exp, {'img': img}).subtract(thresholds[0]).divide(thresholds[1] - thresholds[0])

def detect_clouds(img):
  # Compute several indicators of cloudyness and take the minimum of them.
  score = ee.Image(1.0)
  # Clouds are reasonably bright in the blue band.
  score = score.min(rescale(img, 'img.blue', [0.1, 0.3]))

  # Clouds are reasonably bright in all visible bands.
  score = score.min(rescale(img, 'img.red + img.green + img.blue', [0.2, 0.8]))

  # Clouds are reasonably bright in all infrared bands.
  score = score.min(
      rescale(img, 'img.nir + img.swir1 + img.swir2', [0.3, 0.8]))

  # Clouds are reasonably cool in temperature.
  score = score.min(rescale(img, 'img.temp', [300, 290]))

  # However, clouds are not snow.
  ndsi = img.normalizedDifference(['green', 'swir1'])
  return score.min(rescale(ndsi, 'img', [0.8, 0.6]))


def detect_water(image):
    global collection_dict, sensor_band_dict, spacecraft_dict
    shadowSumBands = ee.List(['nir','swir1','swir2'])# Bands for shadow masking
    # Compute several indicators of water and take the minimum of them.
    score = ee.Image(1.0)

    # Set up some params
    darkBands = ['green','red','nir','swir2','swir1']# ,'nir','swir1','swir2']
    brightBand = 'blue'

    # Water tends to be dark
    sum = image.select(shadowSumBands).reduce(ee.Reducer.sum())
    sum = rescale(sum,'img',[0.35,0.2]).clamp(0,1)
    score = score.min(sum)

    # It also tends to be relatively bright in the blue band
    mean = image.select(darkBands).reduce(ee.Reducer.mean())
    std = image.select(darkBands).reduce(ee.Reducer.stdDev())
    z = (image.select([brightBand]).subtract(std)).divide(mean)
    z = rescale(z,'img',[0,1]).clamp(0,1)
    score = score.min(z)

    # Water is at or above freezing

    score = score.min(rescale(image, 'img.temp', [273, 275]))

    # Water is nigh in ndsi (aka mndwi)
    ndsi = image.normalizedDifference(['green', 'swir1'])
    ndsi = rescale(ndsi, 'img', [0.3, 0.8])

    score = score.min(ndsi)
    return score.clamp(0,1)

def count_water_and_clouds(ee_bounds, image, sun_elevation):
    '''Calls the water and cloud detection algorithms on an image and packages the results'''
    image = ee.Image(image)
    clouds  = detect_clouds(image).gt(cloudThresh)
    #snow = detect_snow(image).gt(snowThresh)
    #Function to scale water detection sensitivity based on time of year.
    def scale_waterThresh(sun_angle):
        waterThresh = ((.45/41)*(62-sun_angle))+.05
        return waterThresh
    waterThresh = scale_waterThresh(sun_elevation)

    water  = detect_water(image).gt(waterThresh).And(clouds.Not())#.And(snow.Not())
    cloud_count = clouds.mask(clouds).reduceRegion(
        reducer = ee.Reducer.count(),
        geometry = ee_bounds,
        scale = 30,
        maxPixels = 1000000,
        bestEffort = True
    )
    water_count = water.mask(water).reduceRegion(
        reducer = ee.Reducer.count(),
        geometry = ee_bounds,
        scale = 30,
        maxPixels = 1000000,
        bestEffort = True
    )
    # addToMap(ee.Algorithms.ConnectedComponentLabeler(water, ee.Kernel.square(1), 256))
    return ee.Feature(None, {'date': image.get('DATE_ACQUIRED'),
                             'spacecraft': image.get('SPACECRAFT_ID'),
                             'water': water_count.get('constant'),
                             'cloud': cloud_count.get('constant')})

def parse_lake_data(filename):
    '''Read in an output file generated by this program'''

    f = open(filename, 'r')
    # take values with low cloud cover
    f.readline() # Skip first header line
    f.readline() # Skip nation infor
    f.readline() # Skip second header line
    results = dict()
    for l in f:   # Loop through each line of the file
        parts = l.split(',')
        date = parts[0].strip()  # Image date
        satellite = parts[1].strip()  # Observing satellite
        cloud = int(parts[2])    # Clound count
        water  = int(parts[3])    # Water count
        sun_elevation = float(parts[4])

        if satellite not in results:
            results[satellite] = dict()
        results[satellite][date] = (cloud, water, sun_elevation)

    f.close()
    return results

# --- Global variables that govern the parallel threads ---
NUM_SIMULTANEOUS_THREADS = 8
global_semaphore = threading.Semaphore(NUM_SIMULTANEOUS_THREADS)
thread_lock = threading.Lock()
total_threads = 0
all_threads = dict()

def process_lake(thread, lake, ee_lake, start_date, end_date, output_directory, update_function):
    '''Computes lake statistics over a date range and writes them to a log file'''

    # Extract the lake name (required!)
    name = lake['properties']['LAKE_NAME']
    if name == '':
        return

    # Set the output file path and load the file if it already exists
    output_file_name = os.path.join(output_directory, name + '.txt')
    data = None
    if os.path.exists(output_file_name):
        data = parse_lake_data(output_file_name)

    # Open the output file for writing and fill in the header lines
    f = open(output_file_name, 'w')
    country = lake['properties']['COUNTRY']
    area = lake['properties']['AREA_SKM']
    # print '%s, %s, %s' % (name, country, area)
    f.write('# Name     Country    Area in km^2\n')
    f.write('%s, %s, %s\n' % (name, country, area))
    f.write('# Date, Satellite, Cloud Pixels, Water Pixels, Sun Elevation\n')

    # If the file already existed and we loaded data from it,
    #   re-write the data back in to the new output file.
    if data is not None:
        for sat in sorted(data.keys()):
            for date in sorted(data[sat].keys()):
                f.write('%s, %10s, %10d, %10d, %.5g\n' % (date, sat, data[sat][date][0], data[sat][date][1],
                                                          data[sat][date][2]))

    try:
        # Take the lake boundary and expand it out in all directions by 1000 meters
        ee_bounds = ee_lake.geometry().buffer(1000)
        # Fetch all the landsat 5 imagery covering the lake on the date range
        collection = get_image_collection(ee_bounds, start_date, end_date)
        v = collection.toList(1000000)
    except:
        print >> sys.stderr, 'Failed to allocate memory to expand buffer for lake %s, skipping.' % (name)
        f.close()
        return

    # Iterate through all the images we retrieved
    results = []
    all_images = v.getInfo()
    #print all_images
    for i in range(len(all_images)):
        if thread.aborted:
            break

        # If we already loaded data that contained results for this image, don't re-process it!
        if ((data is not None) and (all_images[i]['properties']['SPACECRAFT_ID'] in data.keys()) and
            (all_images[i]['properties']['DATE_ACQUIRED'] in data[all_images[i]['properties']['SPACECRAFT_ID']])):
            continue

        # Retrieve the image data and fetch the sun elevation (suggests the amount of light present)
        #print v.get(i)
        im = v.get(i)
        sun_elevation = all_images[i]['properties']['SUN_ELEVATION']

        # Call processing algorithms on the lake with second try in case EE chokes.
        r = count_water_and_clouds(ee_bounds, im, sun_elevation).getInfo()['properties']
        try:
            r = count_water_and_clouds(ee_bounds, im, sun_elevation).getInfo()['properties']
        except Exception as e:
            print >> sys.stderr, 'Failure counting water...trying again. ' + str(e)
            time.sleep(5)
            r = count_water_and_clouds(ee_bounds, im, sun_elevation).getInfo()['properties']

        # Write the processing results to a new line in the file
        output = '%s, %10s, %10d, %10d, %.5g'% (r['date'], r['spacecraft'], r['cloud'], r['water'], sun_elevation)
        print '%15s %s' % (name, output)
        f.write(output + '\n')
        results.append(r)
        update_function(name, r['date'], i, len(all_images))

    f.close()  # Finished processing images, close up the file.

class LakeThread(threading.Thread):
    '''Helper class to manage the number of active lake processing threads'''
    def __init__(self, args, update_function=None):
        threading.Thread.__init__(self)
        self.aborted = False
        self.setDaemon(True)
        self.args = args
        # Increase the global thread count by one
        thread_lock.acquire()
        global total_threads
        total_threads += 1
        # Start processing
        self.start()
        all_threads[self] = True
        thread_lock.release()

    def cancel(self):
        self.aborted = True

    def run(self):
        # Wait for an open thread spot, then begin processing.
        global_semaphore.acquire()
        if not self.aborted:
            try:
                apply(process_lake, (self,) + self.args)
            except Exception as e:
                print >> sys.stderr, e
        global_semaphore.release()

        # Finished processing, decrement the global thread count.
        thread_lock.acquire()
        global total_threads
        total_threads -= 1
        del all_threads[self]
        thread_lock.release()


# ======================================================================================================
# main()

def Lake_Level_Cancel():
    thread_lock.acquire()
    for thread in all_threads:
        thread.cancel()
    thread_lock.release()

def Lake_Level_Run(lake, date = None, enddate = None, results_dir = None, update_function = None, complete_function = None):
    if date is None:
        start_date = ee.Date('1984-01-01')
        end_date   = ee.Date('2030-01-01')
    elif enddate != None and date != None:
        start_date = ee.Date(date)
        end_date   = ee.Date(enddate)
        if dt.strptime(date,'%Y-%m-%d') > dt.strptime(enddate,'%Y-%m-%d'):
            ctypes.windll.user32.MessageBoxA(0, "Date range invalid: Start date is after end date. Please adjust date range and retry."
                                             , "Invalid Date Range", 1)
            return
        elif dt.strptime(date,'%Y-%m-%d') == dt.strptime(enddate,'%Y-%m-%d'):
            ctypes.windll.user32.MessageBoxA(0, "Date range invalid: Start date is same as end date. Please adjust date range and retry."
                                             , "Invalid Date Range", 1)
            return
    else:
        start_date = ee.Date(date)
        end_date = start_date.advance(1.0, 'month')

    # start_date = ee.Date('2011-06-01') # lake high
    # start_date = ee.Date('1993-07-01') # lake low
    # start_date = ee.Date('1993-06-01') # lake low but some jet streams

    # --- This is the database containing all the lake locations!
    # all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").toList(1000000)

    if lake is not None:
        all_lakes = ee.FeatureCollection('ft:1igNpJRGtsq2RtJuieuV0DwMg5b7nU8ZHGgLbC7iq', "geometry").filterMetadata(u'LAKE_NAME', u'equals', lake).toList(1000000)
        if all_lakes.size() == 0:
            print 'Lake not found in database. Ending process...'
    else:
        # bounds = ee.Geometry.Rectangle(-125.29, 32.55, -114.04, 42.02)
        # all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").filterBounds(bounds).toList(1000000)
        all_lakes = ee.FeatureCollection('ft:1igNpJRGtsq2RtJuieuV0DwMg5b7nU8ZHGgLbC7iq', "geometry").toList(1000000)
        # .filterMetadata(u'AREA_SKM', u'less_than', 300.0).toList(100000)#.filterMetadata(
        # u'LAT_DEG', u'less_than',   42.02).filterMetadata( u'LAT_DEG', u'greater_than', 32.55).filterMetadata(
        # u'LONG_DEG', u'less_than', -114.04).filterMetadata(u'LONG_DEG', u'greater_than', -125.29).toList(1000000)
        # pprint(ee.Feature(all_lakes.get(0)).getInfo())

    # display individual image from a date
    if enddate != None and date != None:
        # Create output directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Fetch ee information for all of the lakes we loaded from the database
        all_lakes_local = all_lakes.getInfo()
        for i in range(len(all_lakes_local)): # For each lake...
            ee_lake = ee.Feature(all_lakes.get(i)) # Get this one lake
            # Spawn a processing thread for this lake
            LakeThread((all_lakes_local[i], ee_lake, start_date, end_date, results_dir, \
                    functools.partial(update_function, i, len(all_lakes_local))))

        # Wait in this loop until all of the LakeThreads have stopped
        while True:
            thread_lock.acquire()
            if total_threads == 0:
                thread_lock.release()
                break
            thread_lock.release()
            time.sleep(0.1)
    elif date:
        from cmt.mapclient_qt import centerMap, addToMap
        lake       = all_lakes.get(0).getInfo()
        ee_lake    = ee.Feature(all_lakes.get(0))
        ee_bounds  = ee_lake.geometry().buffer(1000)
        collection = get_image_collection(ee_bounds, start_date, end_date)
        landsat    = ee.Image(collection.first())
        #pprint(landsat.getInfo())
        center = ee_bounds.centroid().getInfo()['coordinates']
        centerMap(center[0], center[1], 11)
        addToMap(landsat, {'bands': ['B3', 'B2', 'B1']}, 'Landsat 3,2,1 RGB')
        addToMap(landsat, {'bands': ['B7', 'B5', 'B4']}, 'Landsat 7,5,4 RGB', False)
        addToMap(landsat, {'bands': ['B6'            ]}, 'Landsat 6',         False)
        clouds = detect_clouds(landsat)
        water = detect_water(landsat, clouds)
        addToMap(clouds.mask(clouds), {'opacity' : 0.5}, 'Cloud Mask')
        addToMap(water.mask(water), {'opacity' : 0.5, 'palette' : '00FFFF'}, 'Water Mask')
        addToMap(ee.Feature(ee_bounds))
        #print count_water_and_clouds(ee_bounds, landsat).getInfo()

    # compute water levels in all images of area
    else:
        # Create output directory
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Fetch ee information for all of the lakes we loaded from the database
        all_lakes_local = all_lakes.getInfo()
        for i in range(len(all_lakes_local)): # For each lake...
            ee_lake = ee.Feature(all_lakes.get(i)) # Get this one lake
            # Spawn a processing thread for this lake
            LakeThread((all_lakes_local[i], ee_lake, start_date, end_date, results_dir, \
                    functools.partial(update_function, i, len(all_lakes_local))))

        # Wait in this loop until all of the LakeThreads have stopped
        while True:
            thread_lock.acquire()
            if total_threads == 0:
                thread_lock.release()
                break
            thread_lock.release()
            time.sleep(0.1)
    if complete_function != None:
        complete_function()
#Lake_Level_Run('Fallen Leaf', date = '2014-3-01', enddate = '2014-4-01', results_dir = 'C:\\Projects\\Fall 2015 - Lake Tahoe Water Resources\\Data\\Python Scripts\\UI_Script\\results')
