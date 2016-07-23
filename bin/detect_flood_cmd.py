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

#import matplotlib
# matplotlib.use('tkagg')

import sys
import os
import ee
import optparse
import traceback
import simplekml
import json

import cmt.domain
import cmt.modis.flood_algorithms
import cmt.radar.flood_algorithms
import cmt.util.landsat_functions
import cmt.util.miscUtilities

from cmt.util.imageRetrievalFunctions import getCloudFreeModis, getCloudFreeLandsat, getNearestSentinel1

manual='''---=== detect_flood_cmd.py ===---
A command line flood detection tool.
Given a date, location, and options, attempts to generate a flood map of
the requested location.

Three sensors are used: MODIS, Landsat, and Sentinel-1 (radar).  MODIS
is usually available but the other two sensors are usually not available.
MODIS and Landsat are also affected by clouds.  Since clouds are common at
flood locations, there is often no usable data for a given date/location.
There are separate flood detection algorithms for each sensor and the 
algorithm for the most accurate available sensor is chosen to generate
the final flood map.  The order of sensor preference is:
1) Sentinel-1
2) Landsat
3) MODIS

Three outputs are produced: A cloud cover geotiff, a flood map geotiff, and
a .kml file containing the same information in a form ready to be loaded
in to Google Earth.  The cloud cover map is generated from MODIS even if
MODIS is not used to generate the flood map.  Some filtering is done on
the .kml output to reduce polygon complexity and the size of the file.

The algorithms used by this tool are all available in other parts of the
Crisis Mapping Toolkit.  They are:
- Sentinel-1 ==> The SAR algorithm from the paper
                 "A fully automated TerraSAR-X based flood service."
- Landsat    ==> Detection algorithm from the LLAMA tool in the CMT.
- MODIS      ==> The Adaboost algorithm from the paper
                 "Automatic boosted flood mapping from satellite data"

'''


# --------------------------------------------------------------
# Functions


def getBestResolution(domain):
    '''Determines the output resolution of our detection algorithm
       based on the input data.'''
    
    MODIS_RESOLUTION   = 250
    LANDSAT_RESOLUTION = 30
    
    # Look through the sensors and pick the one we use with the best resolution
    bestResolution = MODIS_RESOLUTION # We always at least try for MODIS resolution
    for s in domain.sensor_list:
        name = s.sensor_name.lower()
        if name == 'sentinel-1': # Sentinel data comes in multiple resolutions
            sentinelResolution = s.band_resolutions[s.band_names[0]]
            if bestResolution > sentinelResolution:
                bestResolution = sentinelResolution            
        if 'landsat' in name:
            if bestResolution > LANDSAT_RESOLUTION:
                bestResolution = LANDSAT_RESOLUTION
            
    return bestResolution

# May want to move this function
def detect_flood(domain):
    '''Run flood detection using the available sensors'''

    # Currently we run the sensors in order of preference.  Given the rarity of having multiple
    # good sensors at a single date, this should usually be fine.

    cloudCover = ee.Image(0)
    result     = None
    
    # Note that the radar algorithm currently runs at lower than possible resolution to
    #  ensure that Earth Engine can complete the calculations without failing.
    try:
        domain.get_radar() # Catch an exception here if we don't have radar
        print 'Running RADAR-only flood detection...'
        result = cmt.radar.flood_algorithms.detect_flood(domain, cmt.radar.flood_algorithms.MARTINIS_2)[1]
    except LookupError:
        pass
    except Exception as e:
        print 'Caught exception using RADAR, skipping it:'
        traceback.print_exc()

    try:
        cloudCover = cmt.util.landsat_functions.detect_clouds(domain.get_landsat().image).Or(cloudCover)
        if not result:
            print 'Running LANDSAT-only flood detection...'
            result = cmt.util.landsat_functions.detect_water(domain.get_landsat().image)
    except LookupError:
        pass
    except Exception as e:
        print 'Caught exception using LANDSAT, skipping it:'
        traceback.print_exc()

    try:
        cloudCover = cmt.modis.modis_utilities.getModisBadPixelMask(domain.modis.image).Or(cloudCover)
        if not result:
            print 'Running ADABOOST flood detection...'
            result = cmt.modis.flood_algorithms.detect_flood(domain, cmt.modis.flood_algorithms.ADABOOST)[1]
    except LookupError:
        pass
    except Exception as e:
        if str(e) != "'Domain' object has no attribute 'modis'":
            print 'Caught exception using MODIS, skipping it:'
            traceback.print_exc()
        else:
            print 'No MODIS data found for this date/location!'
    

    if not result:
        raise Exception('No data available to detect a flood!')

    return result, cloudCover

    

def getPolygonArea(vertices):
    '''Returns the area of an N-sided polygon'''
    
    def detPair(a, b):
        '''Called for each pair of input coordinets'''
        return a[0]*b[1]-a[1]*b[0]
    
    total = 0.0
    numElements = len(vertices)
    for i in range(0,numElements-1):
        a = vertices[i]
        b = vertices[i+1]
        total += detPair(a,b)
    
    total += detPair(vertices[numElements-1], vertices[0])
    
    return 0.5*abs(total)


# Unused, the EE method works fine!
def filter_coordinates(coordList, height=2000, maxError = 0.000004):
    '''Reduces the number of vertices in a polygon without changing the area more than maxError.
       The height is used to insert an elevation for each 2D coordinate.'''
    
    numCoords = len(coordList)
    if numCoords == 0:
        return []
    
    keptCoords = [(coordList[0][0], coordList[0][1], height)]
    lastIndex = 0
    nextIndex = 2
    while True:
        if nextIndex >= numCoords:
            break
               
        poly = coordList[lastIndex:nextIndex+1]
        area = getPolygonArea(poly)
                
        if area > maxError:
            # lastIndex catches up to the other indices, otherwise it stays put.
            thisIndex = nextIndex-1
            lastIndex = thisIndex
            # Record the current coordinate.
            keptCoords.append((coordList[thisIndex][0], coordList[thisIndex][1], height))
            
        # Advance 
        nextIndex += 1
            
    keptCoords.append((coordList[numCoords-1][0], coordList[numCoords-1][1], height))
    
    return keptCoords        

def clean_coordinates(coordList, height=2000):
    '''Just clean up a coordinate list so it can be passed to simplekml'''
    output = []
    for coord in coordList:
        output.append((coord[0], coord[1], height))
    return output


def addPoly(kmlObject, coordList, height, isWater, color):
    '''Add a polygon to a simplekml object'''
    
    # Separate the outer and inner coordinate lists that were passed in
    outerBounds = clean_coordinates(coordList[0], height)
    innerBounds = []
    for i in range(1,len(coordList)):
        innerBounds.append(clean_coordinates(coordList[i], height))

    # Make a polygon with the provided borders    
    poly = kmlObject.newpolygon(outerboundaryis=outerBounds,
                                innerboundaryis=innerBounds)

    poly.altitudemode = simplekml.AltitudeMode.relativetoground
    poly.extrude=1
    poly.style.linestyle.color = simplekml.Color.blue
    poly.style.polystyle.color = color 

    if isWater:
        poly.style.polystyle.fill  = 1
    else:
        poly.style.polystyle.fill  = 0


def addFeatureSet(kmlObject, featureInfo, activeColor):
    '''Adds a set of features to a KML object'''
    
    height = 1000 # For better Google Earth display
    
    for f in featureInfo:
       
        isWater = f['properties']['label']
        coords  = f['geometry']['coordinates']
        
        if isWater:
            color = activeColor
        else:
            color = '00FFFFFF' # Translucent white
        
        if f['geometry']['type'] == 'MultiPolygon':

            # Multipolygons are basically just multiple polygons, not sure
            #  why they are grouped up.

            #print 'MULTIPOLYGON: ' + str(len(coords))
            #print 'area = ' + str(featureInfo['properties']['area'])
            
            multipoly = kmlObject.newmultigeometry()
            
            for entry in coords:
                addPoly(multipoly, entry, height, isWater, color)
                
        else: # Regular polygon
            addPoly(kmlObject, coords, height, isWater, color)
    return kmlObject
            

def parseKmlDescription(line):
    '''Parses the description line of the output KML into a dictionary'''
    if not ('<description>' in line):
        raise Exception('Incorrect description line passed in: ' + line)
        
    line   = line.replace('&quot;', '"') # Sometimes quotes get written out like this
    start  = line.find('>')
    end    = line.rfind('<')
    s      = line[start+1:end] # Extract the information
    output = json.loads(s) # Parse
    
    return output

    
def writeKmlDescription(floodInfo):
    '''Generate the description line for the KML file containing flood information'''

    #s = " ".join([sensor.sensor_name for sensor in sensorList])
    s = json.dumps(floodInfo)
    return s



def coordListsToKml(resultFeatureInfo, cloudFeatureInfo, kmlPath, floodInfo):
    '''Converts a local coordinate list to KML'''
       
    # Initialize kml document
    kml = simplekml.Kml()
    kml.document.name = 'ASP CMT flood detections - DATE'
    kml.document.description = writeKmlDescription(floodInfo)
    kml.hint = 'target=earth'

    WATER_COLOR = 'FFF0E614' # Solid teal
    CLOUD_COLOR = 'FFFFFFFF' # Solid white
    addFeatureSet(kml, cloudFeatureInfo,  CLOUD_COLOR)
    addFeatureSet(kml, resultFeatureInfo, WATER_COLOR)
    
    # Save kml document
    print 'Saving: ' + kmlPath
    kml.save(kmlPath)


def getBinaryFeatures(binaryImage, bounds, outputResolution):
    '''From a binary image, vectorize to features and call getInfo()'''
    
    print 'Converting to vectors...'
    featureCollection = binaryImage.reduceToVectors(geometry=bounds, scale=outputResolution, eightConnected=False);
    print '\n---Retrieving info---\n'

    # If too many features were detected, keep the N largest ones.
    #MAX_POLYGONS = 100
    MAX_POLYGONS = 1000
    MIN_POLYGONS = 1 # If less than this, must be bad!
    #MIN_AREA = 500000
    MIN_AREA = 0
    
    numFeatures = featureCollection.size().getInfo()
    print 'Detected ' +str(numFeatures)+ ' polygons.'
    if numFeatures < MIN_POLYGONS:
        raise Exception('Unsufficient polygons detected!')

    # Compute the size of each feature
    def addPolySize(feature):
        MAX_ERROR = 100
        return ee.Feature(feature).set({'area':feature.area(MAX_ERROR)})
    featureCollection = featureCollection.map(addPolySize, True)
    
    # Throw out features below a certain size
    featureCollection = featureCollection.filterMetadata('area', 'greater_than', MIN_AREA)
    numFeatures = featureCollection.size().getInfo()

    print 'Reduced to ' +str(numFeatures)+ ' polygons.'
    if numFeatures < MIN_POLYGONS:
        raise Exception('Unsufficient polygons detected!')

    # Get a list of the features with simplified geometry
    def simplifyFeature(feature):
        MAX_ERROR = 100
        return ee.Feature(feature).simplify(MAX_ERROR)
        
    fList = featureCollection.toList(MAX_POLYGONS) # Limit total number of polygons
    fList = fList.map(simplifyFeature)
    numFeatures = fList.size().getInfo()

    # Quit now if we did not find anything
    if numFeatures == 0:
        raise Exception('Unsufficient features detected!')

    print 'Grabbing polygon info...'
    allFeatureInfo = fList.getInfo()
    print '...done'
    
    print 'Restricted to ' +str(numFeatures)+ ' polygons.'
    
    return allFeatureInfo
    

def addEarthEngineImageIds(infoDict):
    '''Takes the ID information from the input dict and adds full, ready-
    to-use Earth Engine image IDs to the dict.'''

    # MODIS
    if 'modis_id' in infoDict:
        # Ex: 1_MYD09GA_005_2016_07_14_MYD09GQ_005_2016_07_14
        line   = infoDict['modis_id']
        start1 = line.find('M')
        start2 = line.rfind('M')
        nameA  = line[start1:start2-1]
        nameQ  = line[start2:]
        if 'Y' in line: # Aqua
            idA = 'MODIS/MYD09GA/'+nameA
            idQ = 'MODIS/MYD09GQ/'+nameQ
        else: # Terra
            idA = 'MODIS/MOD09GA/'+nameA
            idQ = 'MODIS/MOD09GQ/'+nameQ
        infoDict['modis_image_id_A'] = idA
        infoDict['modis_image_id_Q'] = idQ

    # Landsat
    if 'landsat_id' in infoDict:
        if 'LT5' in infoDict['landsat_id']:
            prefix = 'LT5_L1T_TOA/'
        if 'LE7' in infoDict['landsat_id']:
            prefix = 'LE7_L1T_TOA/'
        if 'LC8' in infoDict['landsat_id']:
            prefix = 'LC8_L1T_TOA/'
        infoDict['landsat_image_id'] = prefix + infoDict['landsat_id']    

    # Sentinel-1
    if 'sentinel1_id' in infoDict:
        infoDict['sentinel1_image_id'] = 'COPERNICUS/S1_GRD/' + infoDict['sentinel1_id']

    return infoDict

    

    return infoDict

# --------------------------------------------------------------
def main(argsIn):

    #logger = logging.getLogger() TODO: Switch to using a logger!

    # Be careful passing in negative number arguments!

    try:
          usage = "usage: detect_flood_cmd.py <output_folder> <date: YYYY-MM-DD> <minLon> <minLat> <maxLon> <maxLat> [--help]\n  "
          parser = optparse.OptionParser(usage=usage)

          parser.add_option("--save-inputs", dest="saveInputs", action="store_true", default=False,
                            help="Save the input images to disk for debugging.")
          parser.add_option("--search-days", dest="searchRangeDays",  default=5, type="int",
                            help="The number of days around the requested date so search for input images.")
          parser.add_option("--max-cloud-percentage", dest="maxCloudPercentage",  default=0.05, type="float",
                            help="Only allow images with this percentage of cloud cover.")
          parser.add_option("--min-sensor-coverage", dest="minCoverage",  default=0.80, type="float",
                           help="Only use sensor images that cover this percentage of the target region.")         
          parser.add_option("--manual", dest="showManual", action="store_true", default=False,
                            help="Display more usage information about the tool.")
          
          (options, args) = parser.parse_args(argsIn)

          if options.showManual:
              print manual
              return 0

          if len(args) < 5:
              print usage
              raise Exception('Not enough arguments provided!')

    except optparse.OptionError, msg:
        raise Usage(msg)

    cmt.ee_authenticate.initialize()

    # Grab positional arguments
    outputFolder = args[0]
    dateString   = args[1]
    [minLon, minLat, maxLon, maxLat] = [float(x) for x in args[2:6]]
  
    eeBounds = ee.Geometry.Rectangle((minLon, minLat, maxLon, maxLat))
    eeDate   = ee.Date.parse('YYYY-MM-dd', dateString)
    print 'Loaded bounds: ' + str(eeBounds.getInfo())
    print 'Loaded date  : ' + str(eeDate.format().getInfo())
  
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

    # Set up this information which will be written to the output KML file
    floodInfo = {'min_lon': minLon, 'max_lon': maxLon,
                 'min_lat': minLat, 'max_lat': maxLat,
                 'target_date': dateString}

    # Try to load an image from each of the sensors and wrap it in
    #  a SensorObservation object that we can feed into a domain
    print 'Loading sensors...'
    modisSensor     = None
    landsatSensor   = None
    sentinel1Sensor = None
    sensorList = []
    try:
        #print 'Fetching MODIS data...'
        modisImage  = getCloudFreeModis(eeBounds, eeDate, options.searchRangeDays, 
                                        options.maxCloudPercentage, options.minCoverage)
        modisSensor = cmt.domain.SensorObservation()
        #print modisImage.getInfo()
        modisSensor.init_from_image(modisImage, 'modis')
        sensorList.append(modisSensor)
        floodInfo['modis_date'] = cmt.util.miscUtilities.getDateFromImageInfo(modisImage.getInfo())
        floodInfo['modis_id'  ] = modisImage.get('system:index').getInfo()
        print 'Loaded MODIS sensor observation!'
    except Exception as e:
        print 'Unable to load a MODIS image in this date range!'
        print str(e)
    try:
        #print 'Fetching Landsat data...'
        landsatImage  = getCloudFreeLandsat(eeBounds, eeDate, options.searchRangeDays, 
                                            options.maxCloudPercentage, options.minCoverage)
        landsatName   = cmt.util.landsat_functions.get_landsat_name(landsatImage)
        landsatSensor = cmt.domain.SensorObservation()
        landsatSensor.init_from_image(landsatImage, landsatName)
        sensorList.append(landsatSensor)
        floodInfo['landsat_date'] = cmt.util.miscUtilities.getDateFromImageInfo(landsatImage.getInfo())
        floodInfo['landsat_id'  ] = landsatImage.get('system:index').getInfo()
        print 'Loaded Landsat sensor observation!'
    except Exception as e:
        print 'Unable to load a Landsat image in this date range!'
        print str(e)
    try:
        #print 'Fetching Sentinel1 data...'
        sentinel1Image  = getNearestSentinel1(eeBounds, eeDate, options.searchRangeDays, options.minCoverage)
        sentinel1Sensor = cmt.domain.SensorObservation()
        sentinel1Sensor.init_from_image(sentinel1Image, 'sentinel1')
        sensorList.append(sentinel1Sensor)
        floodInfo['sentinel1_date'] = cmt.util.miscUtilities.getDateFromImageInfo(sentinel1Image.getInfo())       
        floodInfo['sentinel1_id'  ] = sentinel1Image.get('system:index').getInfo()
        print 'Loaded Sentinel1 sensor observation!'
    except Exception as e:
        print 'Unable to load a Sentinel1 image in this date range!'
        print str(e)

    if not sensorList:
        print 'Unable to find any sensor data for this date/location!'
        return -1

    # Add DEM data
    # - TODO: Should this be a function?
    demSensor = cmt.domain.SensorObservation()
    if cmt.util.miscUtilities.regionIsInUnitedStates(eeBounds):
        demName = 'ned13.xml'
        floodInfo['dem_used'] = 'NED13'
    else:
        demName = 'srtm90.xml'
        floodInfo['dem_used'] = 'SRTM90'
    xmlPath = os.path.join(cmt.domain.SENSOR_SOURCE_DIR, demName)
    demSensor.init_from_xml(xmlPath)
    sensorList.append(demSensor)
    
    # Add some extra information about the sensor data used
    floodInfo = addEarthEngineImageIds(floodInfo)

    domainName = 'domain_' + dateString
    domain = cmt.domain.Domain()
    domain.load_sensor_observations(domainName, [minLon, minLat, maxLon, maxLat], sensorList)

    print 'Successfully loaded the domain!'

    # This resolution is currently hard coded to strike a balance between the image appearance
    #  and the odds of successfully processing the region.
    outputResolution = 100#getBestResolution(domain)
    outputVisParams  = {'min': 0, 'max': 1} # Binary image data
    print 'Best output resolution = ' + str(outputResolution)

    if options.saveInputs:
        inputPathModis     = os.path.join(outputFolder, 'input_modis.tif'    )
        inputPathLandsat   = os.path.join(outputFolder, 'input_landsat.tif'  )
        inputPathSentinel1 = os.path.join(outputFolder, 'input_sentinel1.tif')
    
        if modisSensor:
            (rgbImage, visParams, name, show) = domain.modis.visualize()
            print visParams
            cmt.util.miscUtilities.safeEeImageDownload(rgbImage, eeBounds, 250, inputPathModis, visParams)
        if landsatSensor:
            (rgbImage, visParams, name, show) = domain.get_landsat().visualize()
            cmt.util.miscUtilities.safeEeImageDownload(rgbImage, eeBounds, 90, inputPathLandsat, visParams)
        if sentinel1Sensor:
            (rgbImage, visParams, name, show) = domain.sentinel1.visualize()
            cmt.util.miscUtilities.safeEeImageDownload(rgbImage, eeBounds, 90, inputPathSentinel1, visParams)        

    print 'Running flood detection!'
    (result, cloudCover) = detect_flood(domain)
    #print result.getInfo()

    resultPath = os.path.join(outputFolder, 'flood_detect_result.tif')
    cmt.util.miscUtilities.safeEeImageDownload(result, eeBounds, outputResolution, resultPath, outputVisParams)
    
    #print '/nclouds...'
    #print cloudCover.getInfo()
    cloudPath = os.path.join(outputFolder, 'cloudCover.tif')
    cmt.util.miscUtilities.safeEeImageDownload(cloudCover, eeBounds, outputResolution, cloudPath, outputVisParams)

    # Perform an erosion followed by a dilation to clean up small specks of detection
    circle     = ee.Kernel.circle(radius=1);
    result     = result.focal_min(    kernel=circle, iterations=2).focal_max(kernel=circle, iterations=2);
    cloudCover = cloudCover.focal_min(kernel=circle, iterations=2).focal_max(kernel=circle, iterations=2);

    # Vectorize the binary result image
    print 'Extracting flood features...'
    resultFeatureInfo = getBinaryFeatures(result,     eeBounds, outputResolution)
    print '\nExtracting cloud features...'
    cloudFeatureInfo  = getBinaryFeatures(cloudCover, eeBounds, outputResolution)
       
    print 'Converting coordinates to KML'
    kmlPath = os.path.join(outputFolder, 'floodCoords.kml')
    coordListsToKml(resultFeatureInfo, cloudFeatureInfo, kmlPath, floodInfo)
        
    return 0


# Call main() when run from command line
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


