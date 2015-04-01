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
import xml.etree.cElementTree as ET

# Location of the sensor config files
SENSOR_FILE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../config/sensors')

def safe_get_info(ee_object, max_num_attempts=None):
    '''Keep trying to call getInfo() on an Earth Engine object until it succeeds.'''
    num_attempts = 0
    while True:
        try:
            return ee_object.getInfo()
        except Exception as e:
            print 'Earth Engine Error: %s. Waiting 10s and then retrying.' % (e)
            time.sleep(10)
            num_attempts += 1
        if max_num_attempts and (num_attempts >= max_num_attempts):
            raise Exception('safe_get_info failed to succeed after ' +str(num_attempts)+ ' attempts!')

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
    

def writeModisDomainFile(domainName, bounds, image_ee_date, outputPath, trainingName=None, unfloodedTrainingName=None):
    '''Generates a MODIS domain file for a location using the permanent water mask'''

    # Get the date one day after the provided date
    end_ee_date = image_ee_date.advance(1.0, 'day')
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
    
    
    