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

'''
    This file provides a framework for running a process on many lake dates/locations.
'''

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

import sys
import argparse
import time
import threading
import multiprocessing
import os
import functools

import ee


#---------------------------------------------------------------------------

def get_image_collection_landsat5(bounds, start_date, end_date):
    '''Retrieve Landsat 5 imagery for the selected location and dates.'''

    ee_bounds  = bounds
    ee_points  = ee.List(bounds.bounds().coordinates().get(0))
    points     = ee_points.getInfo()
    points     = map(functools.partial(apply, ee.Geometry.Point), points)
    collection = ee.ImageCollection('LT5_L1T').filterDate(start_date, end_date).filterBounds(points[0]).filterBounds(points[1]).filterBounds(points[2]).filterBounds(points[3])
    return collection


def get_image_collection_modis(bounds, start_date, end_date):
    '''Retrieve MODIS imagery for the selected location and dates.'''

    print 'Fetching MODIS data...'

    ee_bounds    = bounds
    ee_points    = ee.List(bounds.bounds().coordinates().get(0))
    points       = ee_points.getInfo()
    points       = map(functools.partial(apply, ee.Geometry.Point), points)
    highResModis = ee.ImageCollection('MOD09GQ').filterDate(start_date, end_date).filterBounds(points[0]).filterBounds(points[1]).filterBounds(points[2]).filterBounds(points[3])
    lowResModis  = ee.ImageCollection('MOD09GA').filterDate(start_date, end_date).filterBounds(points[0]).filterBounds(points[1]).filterBounds(points[2]).filterBounds(points[3])
    #print highResModis.getInfo()
    #print '================================='
    #print lowResModis.getInfo()['bands']
    #print lowResModis.select('sur_refl_b03').getInfo()
    #print lowResModis.select('sur_refl_b06').getInfo()
    #collection   = highResModis.addBands(lowResModis.select('sur_refl_b03'))#.addBands(lowResModis.select('sur_refl_b06'))

    # This set of code is needed to merge the low and high res MODIS bands
    def merge_bands(element):
        # A function to merge the bands together.
        # After a join, results are in 'primary' and 'secondary' properties.       
        return ee.Image.cat(element.get('primary'), element.get('secondary'))
    join          = ee.Join.inner()
    f             = ee.Filter.equals('system:time_start', None, 'system:time_start')
    modisJoined   = ee.ImageCollection(join.apply(lowResModis, highResModis, f));
    roughJoined   = modisJoined.map(merge_bands);
    # Clean up the joined band names
    band_names_in = ['num_observations_1km','state_1km','SensorZenith','SensorAzimuth','Range','SolarZenith','SolarAzimuth','gflags','orbit_pnt',
                     'num_observations_500m','sur_refl_b03','sur_refl_b04','sur_refl_b05','sur_refl_b06','sur_refl_b07',
                     'QC_500m','obscov_500m','iobs_res','q_scan','num_observations', 'sur_refl_b01_1','sur_refl_b02_1','QC_250m','obscov']
    band_names_out = ['num_observations_1km','state_1km','SensorZenith','SensorAzimuth','Range','SolarZenith','SolarAzimuth','gflags','orbit_pnt',
                      'num_observations_500m','sur_refl_b03','sur_refl_b04','sur_refl_b05','sur_refl_b06','sur_refl_b07',
                      'QC_500m','obscov_500m','iobs_res','q_scan','num_observations_250m', 'sur_refl_b01','sur_refl_b02','QC_250m','obscov']
    collection    = roughJoined.select(band_names_in, band_names_out)
    
    return collection



#---------------------------------------------------------------------------

class LakeDataLoggerBase(object):
    '''Log manager class to store the results of lake processing.
        One of these will be used for each lake.
        This class is a dummy base class that should be derived from.'''
    
    def __init__(self, logDirectory, ee_lake):
        '''Initialize with lake information'''
        self.ee_lake          = ee_lake
        self.output_directory = __computeUniquePath(logDirectory, ee_lake)
        
       
    def _computeUniquePath(self, logDirectory, ee_lake):
        '''Returns a unique file path for this lake in the log directory'''
       
        name = ee_lake.getInfo()['properties']['LAKE_NAME']
        if not name:
            raise Exception('Cannot process a lake without a name!')
        safe_name  = name.replace(' ', '_')
        outputPath = os.path.join(logDirectory, safe_name + '.txt')
        return outputPath
       
    def findRecordByDate(self, date):
        '''Searches for a record with a particular date and returns it'''
        return None
    
    def addDataRecord(self, dataRecord):
        '''Adds a new record to the log'''
        return True


def sample_processing_function(bounds, image, image_date):
    '''Returns a dictionary of results.
       This is the type of function that can be passed to "process_lake"'''
    return {'water_count' : 1, 'cloud_count': 2}


def process_lake(lake, ee_lake, start_date, end_date, output_directory,
                 processing_function, logging_class, image_fetching_function):
    '''Computes lake statistics over a date range and writes them to a log file.
        processing_function is called with two arguments: a bounding box and an ee_image.'''
    
    # Extract the lake name (required!)
    name = lake['properties']['LAKE_NAME']
    if name == '':
        return
    
    print 'Processing lake: ' + name
    
    # Set up logging object for this lake
    logger = logging_class(output_directory, ee_lake)

    # Take the lake boundary and expand it out in all directions by 1000 meters
    ee_bounds     = ee_lake.geometry().buffer(1000)
    # Fetch all the landsat 5 imagery covering the lake on the date range
    collection    = image_fetching_function(ee_bounds, start_date, end_date)
    ee_image_list = collection.toList(1000000)
    
    num_images_found = len(ee_image_list.getInfo())
    print 'Found ' + str(num_images_found) + ' images for this lake.'

    # Iterate through all the images we retrieved
    results = []
    all_image_info = ee_image_list.getInfo()
    for i in range(len(all_image_info)):
        
        
        #for v in all_image_info[i]:
        #    print '--------------------------------------------------------'
        #    print v
        #    print all_image_info[i][v]
        
        
        # Extract the date - look for it in several locations
        if 'DATE_ACQUIRED' in all_image_info[i]['properties']: # Landsat 5
            this_date = all_image_info[i]['properties']['DATE_ACQUIRED']
        else:
            # MODIS: The date is stored in the 'id' field in this format: 'MOD09GA/MOD09GA_005_2004_08_15'
            text       = all_image_info[i]['id']
            dateStart1 = text.rfind('MOD09GA_') + len('MOD09GA_')
            dateStart2 = text.find('_', dateStart1) + 1
            this_date  = text[dateStart2:].replace('_', '-')
        
        # If we already loaded data that contained results for this image, don't re-process it!
        # - In the future may need to check more than the date
        if logger.findRecordByDate(this_date):
            continue
        print 'Processing date ' + str(this_date)
        
        # Retrieve the image data and fetch the sun elevation (suggests the amount of light present)
        this_ee_image = ee.Image(ee_image_list.get(i))
        #sun_elevation = all_image_info[i]['properties']['SUN_ELEVATION'] # Easily available in Landsat5
        
        
        # Call processing algorithms on the lake with second try in case EE chokes.
        #try:
        r = processing_function(ee_bounds, this_ee_image, this_date)
        #except Exception as e:
        #    print >> sys.stderr, 'Failure counting water...trying again. ' + str(e)
        #    time.sleep(5)
        #    r = processing_function(ee_bounds, this_ee_image, this_date).getInfo()['properties']
        #r = {'TEST': (1.0, 2.0), 'satellite': 'dummy'} # DEBUG
        print r
        
        # Append some metadata to the record and log it
        #r['sun_elevation'] = sun_elevation
        r['date'] = this_date
        logger.addDataRecord(r)

    print 'Finished processing lake: ' + name

#======================================================================================================
def main(processing_function, logging_class, image_fetching_function=get_image_collection_landsat5):
    '''This main needs to be called from another file with some arguments'''

    parser = argparse.ArgumentParser(description='Measure lake water levels.')
    parser.add_argument('--start-date',  dest='start_date',  action='store', required=False, default=None, help='YYYY-MM-DD start date')
    parser.add_argument('--end-date',    dest='end_date',    action='store', required=False, default=None, help='YYYY-MM-DD end date')
    parser.add_argument('--lake',        dest='lake',        action='store', required=False, default=None, help='Specify a single lake to process')
    parser.add_argument('--results-dir', dest='results_dir', action='store', required=False, default='results')
    parser.add_argument('--max-lakes',   dest='max_lakes',   type=int,     required=False, default=100, help='Limit to this many lakes')
    parser.add_argument('--threads',     dest='num_threads', type=int,     required=False, default=4)
    args = parser.parse_args()
       
    if args.start_date == None: # Use a large date range
        start_date = ee.Date('1984-01-01')
        end_date   = ee.Date('2015-01-01')
    else: # Start date provided
        start_date = ee.Date(args.start_date)
        if args.end_date: # End date also provided
            end_date = ee.Date(args.end_date)
        else: # Use the input date plus one month
            end_date = start_date.advance(1.0, 'month')
    
    # --- This is the database containing all the lake locations!
    if args.lake != None:
        all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").filterMetadata(u'LAKE_NAME', u'equals', args.lake).toList(1000000)
        if not all_lakes:
            raise Exception('Failed to find user specified lake name!')
    else:
        all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").toList(args.max_lakes)
             #.filterMetadata(u'AREA_SKM', u'less_than', 300.0).toList(100000)#.filterMetadata(
             #u'LAT_DEG', u'less_than',   42.02).filterMetadata( u'LAT_DEG', u'greater_than', 32.55).filterMetadata(
             #u'LONG_DEG', u'less_than', -114.04).filterMetadata(u'LONG_DEG', u'greater_than', -125.29).toList(args.max_lakes)
        #pprint(ee.Feature(all_lakes.get(0)).getInfo())
    
    # Fetch ee information for all of the lakes we loaded from the database
    all_lakes_local = all_lakes.getInfo()
    num_lakes       = len(all_lakes_local)
    print 'Found ' + str(num_lakes) + ' lakes.'

    # Create output directory
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    
    # Create processing pool and multiprocessing manager
    num_threads = args.num_threads
    if num_lakes < num_threads:
        num_threads = num_lakes
    print 'Spawning ' + str(num_threads) + ' worker thread(s)'
    pool    = multiprocessing.Pool(processes=num_threads)
    manager = multiprocessing.Manager()
    
    
    lake_results    = []
    for i in range(len(all_lakes_local)): # For each lake...
        # Get this one lake
        ee_lake = ee.Feature(all_lakes.get(i)) 

    
        process_lake(all_lakes_local[i], ee_lake, start_date, end_date, args.results_dir, processing_function, logging_class, image_fetching_function)
    
        
        ## Spawn a processing thread for this lake
        #lake_results.append(pool.apply_async(process_lake, args=(all_lakes_local[i], ee_lake,
        #                                                         start_date, end_date,
        #                                                         args.results_dir,
        #                                                         processing_function, logging_class, image_fetching_function)))
    
    # Wait until all threads have finished
    print 'Waiting for all threads to complete...'
    for r in lake_results:
        r.get()
    
    # Stop the queue and all the threads
    print 'Cleaning up...'
    pool.close()
    pool.join()

