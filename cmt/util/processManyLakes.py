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
import traceback
import ee

#---------------------------------------------------------------------------

class LakeDataLoggerBase(object):
    '''Log manager class to store the results of lake processing.
        One of these will be used for each lake.
        This class is a dummy base class that should be derived from.'''
    
    def __init__(self, logDirectory, ee_lake, lake_name):
        '''Initialize with lake information'''
        self.ee_lake        = ee_lake
        self.base_directory = logDirectory
        self.lake_name = lake_name
       
    def getBaseDirectory(self):
        '''The top level output folder'''
        return self.base_directory
    
    def getLakeDirectory(self):
        raise Exception('Implement me!')
       
    def getLakeName(self):
        return self.lake_name
       
    def computeLakePrefix(self):
        '''Returns a file prefix for this lake in the log directory'''
        return os.path.join(self.base_directory, self.lake_name)
    
    #def findRecordByDate(self, date):
    #    '''Searches for a record with a particular date and returns it'''
    #    return None
    
    def addDataRecord(self, dataRecord):
        '''Adds a new record to the log and optionally save an image'''
        return True


def sample_processing_function(bounds, image, image_date, logger):
    '''Returns a dictionary of results.
       This is the type of function that can be passed to "process_lake"'''
    return {'water_count' : 1, 'cloud_count': 2}


def isLakeInBadList(name, output_directory, date=None):
    '''Check the blacklist to see if we should skip a lake'''

    # - The blacklist is a CSV file containing:
    #     lake name, date
    # - If the date is left blank then applies to all dates.
    # - If no date is passed in than only a blank date will match.

    # Search the entire file for the name
    list_path = os.path.join(output_directory, 'badLakeList.txt')
    try:
        file_handle = open(list_path, 'r')
        found = False
        for line in file_handle:
            parts = line.strip().split(',')
            if name != parts[0]: # Name does not match, keep searching
                continue
            # Name matches, check the date
            if (parts[1]=='') or (parts[1] == date):
                found = True
                break
        file_handle.close()
        return found
    except: # Fail silently
        return False

def addLakeToBadList(name, output_directory, date=None):
    '''Create a blacklist of lakes we will skip'''
    # Just add the name to a plain text file
    list_path   = os.path.join(output_directory, 'badLakeList.txt')
    file_handle = open(list_path, 'a')
    if date: # Write "name, date"
        file_handle.write(name +','+ str(date) +'\n')
    else: # Write "name,"
        file_handle.write(name + ',\n')
    file_handle.close()
    return True

# The maximum lake size Earth Engine can handle in square kilometers
MAX_LAKE_SIZE = 5000
MAX_LATITUDE  = 55 # SRTM90 is not available beyond 60 latitude

def process_lake(lake, ee_lake, start_date, end_date, output_directory,
                 processing_function, logging_class, image_fetching_function):
    '''Computes lake statistics over a date range and writes them to a log file.
        processing_function is called with two arguments: a bounding box and an ee_image.'''
    
    try:
        
        # Extract the lake name
        name = lake['properties']['LAKE_NAME']
        name = name.replace("'","").replace(".","").replace(",","").replace(" ","_") # Strip out weird characters
        if name == '': # Can't proceed without the name!
            print('Skipping lake with no name!')
            print(lake['properties'])
            return False
    
        # Check if the lake is in the bad lake list
        if isLakeInBadList(name, output_directory):
            print('Skipping known bad lake ' + name)
            return False
    
        boundsInfo = ee_lake.geometry().bounds().getInfo()
    
        # Take the lake boundary and expand it out in all directions by 1000 meters
        # - Need to use bounding boxes instead of exact geometeries, otherwise
        #    Earth Engine's memory usage will explode!
        ee_bounds = ee_lake.geometry().bounds().buffer(1000).bounds()
        
        print('Processing lake: ' + name)
        
        # Set up logging object for this lake
        logger = logging_class(output_directory, ee_lake, name)
        
        
        # Fetch all the imagery covering the lake on the date range
        collection    = image_fetching_function(ee_bounds, start_date, end_date)
        ee_image_list = collection.toList(1000000)
        
        num_images_found = len(ee_image_list.getInfo())
        print('Found ' + str(num_images_found) + ' images for this lake.')
    
        # Iterate through all the images we retrieved
        results = []
        all_image_info = ee_image_list.getInfo()
        for i in range(len(all_image_info)):
                        
            # Extract the date for this image
            this_date = get_image_date(all_image_info[i])
            
            if isLakeInBadList(name, output_directory, this_date):
                print('Skipping known bad instance: ' + name +' - '+ this_date)
                continue
            
            print('Processing date ' + str(this_date))
            
            # Retrieve the image data and fetch the sun elevation (suggests the amount of light present)
            this_ee_image = ee.Image(ee_image_list.get(i))
            #sun_elevation = all_image_info[i]['properties']['SUN_ELEVATION'] # Easily available in Landsat5       
            
            # Call processing algorithms on the lake with second try in case EE chokes.
            try:
                result = processing_function(ee_bounds, this_ee_image, this_date, logger)
            except Exception as e:
                print('Processing failed, skipping this date --> ' + str(e))
                traceback.print_exc(file=sys.stdout)
                continue
            
            # Append some metadata to the record and log it
            #r['sun_elevation'] = sun_elevation
            result['date'] = this_date
            logger.addDataRecord(result)

    except Exception as e:
        print('Caught exception processing the lake!')
        print(str(e))
        traceback.print_exc(file=sys.stdout)

    print('Finished processing lake: ' + name)

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
        all_lakes = ee.FeatureCollection('ft:13s-6qZDKWXsLOWyN7Dap5o6Xuh2sehkirzze29o3', "geometry").filterMetadata(
                        u'LAKE_NAME',  u'not_equals',    "").filterMetadata(
                        u'AREA_SKM',   u'less_than',     MAX_LAKE_SIZE).filterMetadata(
                        u'LAT_DEG',    u'less_than',     MAX_LATITUDE).filterMetadata(
                        u'LAT_DEG',    u'greater_than', -MAX_LATITUDE).toList(args.max_lakes)
        #pprint(ee.Feature(all_lakes.get(0)).getInfo())
    
    # Fetch ee information for all of the lakes we loaded from the database
    all_lakes_local = all_lakes.getInfo()
    num_lakes       = len(all_lakes_local)
    print('Found ' + str(num_lakes) + ' lakes.')

    # Create output directory
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    
    # Create processing pool and multiprocessing manager
    num_threads = args.num_threads
    if num_lakes < num_threads:
        num_threads = num_lakes
    print('Spawning ' + str(num_threads) + ' worker thread(s)')
    pool    = multiprocessing.Pool(processes=num_threads)
    manager = multiprocessing.Manager()
    
    
    lake_results = []
    for i in range(len(all_lakes_local)): # For each lake...
        # Get this one lake
        ee_lake = ee.Feature(all_lakes.get(i)) 

        #process_lake(all_lakes_local[i], ee_lake, start_date, end_date, args.results_dir, processing_function, logging_class, image_fetching_function)
        
        # Spawn a processing thread for this lake
        lake_results.append(pool.apply_async(process_lake, args=(all_lakes_local[i], ee_lake,
                                                                 start_date, end_date,
                                                                 args.results_dir,
                                                                 processing_function, logging_class, image_fetching_function)))
        
    # Wait until all threads have finished
    print('Waiting for all threads to complete...')
    for r in lake_results:
        r.get()
    
    # Stop the queue and all the threads
    print('Cleaning up...')
    pool.close()
    pool.join()

