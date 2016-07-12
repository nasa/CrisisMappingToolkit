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

import ee
import os
import json
import threading
import time
import xml.etree.cElementTree as ET
import tempfile
import zipfile
import urllib2

# Location of the sensor config files
SENSOR_FILE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../config/sensors')

TEMP_FILE_DIR = tempfile.gettempdir()


# This is an approximate function
def getExpandingIndices(length, minIndex=0):
    '''Gets a list of indices starting from the middle and expanding outwards
       evenly in both directions.'''
    
    maxIndex = length-1
    if (maxIndex < 0):
        return []
    
    center = (minIndex + maxIndex) / 2
    offset = 1
    output = [center]
    while True:
    
        if len(output)==length:
            return output
    
        thisIndex = center + offset
        output.append(thisIndex)
        
        if offset > 0:
          offset = -offset
        else:
          offset = -offset + 1


def safeRename(collection, inputNames, outputNames):
    '''Tries to rename the given bands of the collection.  Safely ignores
       any bands which are not present.'''

    output = None

    # Try to add each band to the output collection one at a time    
    for (nameIn, nameOut) in zip(inputNames, outputNames):
        try:
            testBand = collection.select([nameIn], [nameOut])
            # Call getInfo() here to force an exception if the band does not exist
            testBand.getInfo()
            if output:
                output = output.combine(testBand)
            else:
                output = testBand
        except:
            pass
       
    return output
        

def safe_get_info(ee_object, max_num_attempts=5):
    '''Keep trying to call getInfo() on an Earth Engine object until it succeeds.'''
    
    # Do not try again if we get one of these exceptions
    fail_messages = ['Too many pixels']
    
    num_attempts = 0
    while True:
        try:
            return ee_object.getInfo()
        except Exception as e:
            # Rethrow the exception if it is not timeout related
            for message in fail_messages:
                if message in str(e):
                    raise e
                    
            print 'Earth Engine Error: %s. Waiting 10s and then retrying.' % (e)
            time.sleep(10)
            num_attempts += 1
        if max_num_attempts and (num_attempts >= max_num_attempts):
            raise Exception('safe_get_info failed to succeed after ' +str(num_attempts)+ ' attempts!')


class waitForEeResult(threading.Thread):
    '''Starts up a thread to run a pair of functions in series'''

    def __init__(self, function, finished_function = None):
        threading.Thread.__init__(self)
        self.function          = function # Main function -> Run this!
        self.finished_function = finished_function # Run this after the main function is finished
        self.setDaemon(True) # Don't hold up the program on this thread
        self.start()
    def run(self):
        self.finished_function(self.function())

def prettyPrintEE(eeObjectInfo):
    '''Convenient function for printing an EE object with tabbed formatting (pass in result of .getInfo())'''
    print(json.dumps(eeObjectInfo, sort_keys=True, indent=2))

def get_permanent_water_mask():
    '''Returns the global permanent water mask'''
    return ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'], ['b1'])


def regionIsInUnitedStates(region):
        '''Returns true if the current region is inside the US.'''
        
        # Extract the geographic boundary of the US.
        nationList = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw')
        nation     = ee.Feature(nationList.filter(ee.Filter.eq('Country', 'United States')).first())
        nationGeo  = ee.Geometry(nation.geometry())
        result     = nationGeo.contains(region, 10)

        return (str(result.getInfo()) == 'True')
    
def getDefinedSensorNames():
        '''Returns the list of known sensor types'''
        # Get a list of xml files from the sensor files directory
        return [f[:-4] for f in os.listdir(SENSOR_FILE_DIR) if f.endswith('.xml')]


def getDateFromSentinel1Info(info):
    '''Finds the date in a Sentinel1 EE object'''
    idString = info['id']
    
    # The string should look something like this:
    # COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20151112T003149_20151112T003214_008564_00C241_135A
    parts = idString.split('_')
    if len(parts) < 6:
        return None
    dateString = parts[5]
    #year  = int(dateString[0:4])
    #month = int(dateString[4:6])
    #day   = int(dateString[6:8])
    #date  = ee.Date.fromYMD(year, month, day)
    return dateString

def getDateFromLandsatInfo(info):
    '''Finds the date in a Landsat EE object'''
    # Landsat images do not have consistent header information so try multiple names here.
    if 'DATE_ACQUIRED' in info['properties']:
        return info['properties']['DATE_ACQUIRED']
    elif 'ACQUISITION_DATE' in info['properties']:
        return info['properties']['ACQUISITION_DATE']
    else:
        return None

def getDateFromModisInfo(info):
    '''Finds the date in a MODIS EE object'''
    
    # MODIS: The date is stored in the 'id' field in this format: 'MOD09GA/MOD09GA_005_2004_08_15'
    text       = info['id']
    dateStart1 = text.rfind('MOD09GA_') + len('MOD09GA_')
    dateStart2 = text.find('_', dateStart1) + 1
    this_date  = text[dateStart2:].replace('_', '-')
    return this_date


# Currently these functions return non-standard strings
def getDateFromImageInfo(info):
    '''Get the date from an Earth Engine image object.'''
    date = getDateFromLandsatInfo(info)
    if not date:
        date = getDateFromSentinel1Info(info)
    if not date:
        date = getDateFromModisInfo(info)
    return date


def unComputeRectangle(eeRect):
    '''"Decomputes" an ee Rectangle object so more functions will work on it'''
    # This function is to work around some dumb EE behavior

    LON = 0 # Helper constants
    LAT = 1    
    rectCoords  = eeRect.getInfo()['coordinates']    # EE object -> dictionary -> string
    minLon      = rectCoords[0][0][LON]           # Exctract the numbers from the string
    minLat      = rectCoords[0][0][LAT]
    maxLon      = rectCoords[0][2][LON]
    maxLat      = rectCoords[0][2][LAT]
    bbox        = [minLon, minLat, maxLon, maxLat]   # Pack in order
    eeRectFixed = apply(ee.Geometry.Rectangle, bbox) # Convert back to EE rectangle object
    return eeRectFixed
    

def writeModisDomainFile(domainName, bounds, image_ee_date, outputPath, 
                         trainingName=None, unfloodedTrainingName=None):
    '''Generates a MODIS domain file for a location using the permanent water mask'''

    # Get the date one day after the provided date
    end_ee_date  = image_ee_date.advance(1.0, 'day')
    startDateStr = image_ee_date.format().getInfo()[:10] # Get just the date in string format
    endDateStr   = end_ee_date.format().getInfo()[:10]

    # Set up the top levels
    root = ET.Element("domain")
    ET.SubElement(root, "name").text = domainName
    sensors = ET.SubElement(root, "sensors")
    
    # Add the appropriate DEM sensor
    demSensor = ET.SubElement(sensors, "sensor")
    if regionIsInUnitedStates(bounds):
        ET.SubElement(demSensor, "name").text = 'NED13'
    else:
        ET.SubElement(demSensor, "name").text = 'SRTM90'
    
    # Add the MODIS sensor info
    modisSensor = ET.SubElement(sensors,     "sensor")
    bands       = ET.SubElement(modisSensor, "bands")
    source      = ET.SubElement(bands,       "source")
    ET.SubElement(modisSensor, "name").text = 'MODIS'
    ET.SubElement(source,      "start_date").text = startDateStr
    ET.SubElement(source,      "end_date").text = endDateStr
    
    # Fill out the bounding box
    bbox    = ET.SubElement(root, "bbox")
    bbox_bl = ET.SubElement(bbox, "bottomleft")
    bbox_tr = ET.SubElement(bbox, "topright")
    rectCoords = bounds.bounds().getInfo()['coordinates']
    ET.SubElement(bbox_bl, "lon").text = str(rectCoords[0][0][0]) # Min lon
    ET.SubElement(bbox_bl, "lat").text = str(rectCoords[0][0][1]) # Min lat
    ET.SubElement(bbox_tr, "lon").text = str(rectCoords[0][2][0]) # Max lon
    ET.SubElement(bbox_tr, "lat").text = str(rectCoords[0][2][1]) # Max lat
    
    # Fill out the truth information
    ET.SubElement(root, "truth").text = 'permanent_water_mask'
    if trainingName:
        ET.SubElement(root, "training_domain").text = trainingName
    if unfloodedTrainingName:
        ET.SubElement(root, "unflooded_training_domain").text = unfloodedTrainingName
    
    tree = ET.ElementTree(root)
    tree.write(outputPath)

    
    
def writeDomainFilePair(domainName, bounds, image_ee_date, training_ee_date, outputFolder):
    '''Write a test/train pair of domain files for a single domain'''
    
    # Set up the names so the two files are associated
    baseName  = domainName +'_'+ str(image_ee_date.format().getInfo()[:10]) # Append the date string
    testName  = baseName   + '_test'
    trainName = baseName   + '_train'
    testPath  = os.path.join(outputFolder, testName  + '.xml')
    trainPath = os.path.join(outputFolder, trainName + '.xml')
    
    # Write the files with only the unflooded training link
    writeModisDomainFile(domainName, bounds, image_ee_date,    testPath,  None, trainName)
    writeModisDomainFile(domainName, bounds, training_ee_date, trainPath, None, None)
    
    return (testPath, trainPath)
    
    
def which(program):
    '''Tests if a given command line tool is available, replicating the "which" function'''
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None





def downloadEeImage(eeObject, bbox, scale, file_path, vis_params=None):
    '''Downloads an Earth Engine image object to the specified path'''

    # For now we require a GDAL installation in order to save images
    if not(which('gdalbuildvrt') and which('gdal_translate')):
        print 'ERROR: Must have GDAL installed in order to save images!'
        return False

    # Get a list of all the band names in the object
    band_names = []
    if vis_params and ('bands' in vis_params): # Band names were specified
        band_names = vis_params['bands']
        if ',' in band_names: # If needed, convert from string to list
            band_names = band_names.replace(' ', '').split(',')
    else: # Grab the first three band names
        if len(eeObject.getInfo()['bands']) > 3:
            print 'Warning: Limiting recorded file to first three band names!'
        for b in eeObject.getInfo()['bands']:
            band_names.append(b['id'])
            if len(band_names) == 3:
                break
            
    if len(band_names) > 3:
        raise Exception('Images with more than three channels are not supported!')
    
    # Handle selected visualization parameters
    if vis_params and ('min' in vis_params) and ('max' in vis_params): # User specified scaling
        download_object = eeObject.visualize(band_names, min=vis_params['min'], max=vis_params['max'])
    elif vis_params and ('gain' in vis_params):
        # Extract the floating point gain values
        gain_text       = vis_params['gain'].replace(' ', '').split(',')
        gain_vals       = [float(x) for x in gain_text]
        download_object = eeObject.visualize(band_names, gain_vals)
    else:
        download_object = eeObject.visualize(band_names)
    
    # Handle input bounds as string or a rect object
    if isinstance(bbox, basestring) or isinstance(bbox, list): 
        eeRect = apply(ee.Geometry.Rectangle, bbox)
    else:
        eeRect = bbox
    eeGeom = eeRect.toGeoJSONString()
    
    # Retrieve a download URL from Earth Engine
    dummy_name = 'EE_image'
    url = download_object.getDownloadUrl({'name' : dummy_name, 'scale': scale, 'crs': 'EPSG:4326', 'region': eeGeom})
    #print 'Got download URL: ' + url
      
    
    # Generate a temporary path for the packed download file
    temp_prefix = 'CMT_temp_download_' + dummy_name
    zip_name    = temp_prefix + '.zip'
    zip_path    = os.path.join(TEMP_FILE_DIR, zip_name) 
    
    # Download the packed file
    print 'Downloading image...'
    data = urllib2.urlopen(url)
    with open(zip_path, 'wb') as fp:
        while True:
            chunk = data.read(16 * 1024)
            if not chunk:
                break
            fp.write(chunk)
    print 'Download complete!'
    
    # Each band get packed seperately in the zip file.
    z = zipfile.ZipFile(zip_path, 'r')
    
    ## All the transforms should be the same so we only read the first one.
    ## - The transform is the six numbers that make up the CRS matrix (pixel to lat/lon conversion)
    #transform_file = z.open(dummy_name + '.' + band_names[0] + '.tfw', 'r')
    #transform = [float(line) for line in transform_file]
    
    # Extract each of the band images into a temporary file
    # - Eventually the download function is supposed to pack everything in to one file!  https://groups.google.com/forum/#!topic/google-earth-engine-developers/PlgCvJz2Zko
    temp_band_files = []
    band_files_string = ''
    #print 'Extracting...'
    if len(band_names) == 1:
        color_names = ['vis-gray']
    else:
        color_names = ['vis-red', 'vis-green', 'vis-blue']
    for b in color_names:
        band_filename  = dummy_name + '.' + b + '.tif'
        extracted_path = os.path.join(TEMP_FILE_DIR, band_filename)
        #print band_filename
        #print extracted_path
        z.extract(band_filename, TEMP_FILE_DIR)
        temp_band_files.append(extracted_path)
        band_files_string += ' ' + extracted_path
        
    # Generate an intermediate vrt file
    vrt_path = os.path.join(TEMP_FILE_DIR, temp_prefix + '.vrt')
    cmd = 'gdalbuildvrt -separate -resolution highest ' + vrt_path +' '+ band_files_string
    print cmd
    os.system(cmd)
    if not os.path.exists(vrt_path):
        raise Exception('Failed to create VRT file!')
    
    # Convert to the output file
    cmd = 'gdal_translate -ot byte '+ vrt_path + ' ' +file_path
    print cmd
    os.system(cmd)
    
    ### Clean up vrt file
    ##os.remove(vrt_path)
    
    # Check for output file
    if not os.path.exists(file_path):
        raise Exception('Failed to create output image file!')
        
    ### Clean up temporary files
    ##for b in temp_band_files:
    ##    os.remove(b)
    ##os.remove(zip_path)
    
    print 'Finished saving ' + file_path
    return True


def safeEeImageDownload(eeObject, bbox, scale, file_path, vis_params=None):
    '''Wraps downloadEeImage to allow multiple attempts.'''
    NUM_ATTEMPTS = 3
    attempt = 0
    while attempt < NUM_ATTEMPTS:
        if attempt < (NUM_ATTEMPTS-1):
            try:
                return downloadEeImage(eeObject, bbox, scale, file_path, vis_params)
            except:
                time.sleep(3)
        else:
            return downloadEeImage(eeObject, bbox, scale, file_path, vis_params)
        attempt += 1
