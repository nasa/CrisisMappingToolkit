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
import simplekml

import cmt.domain
import cmt.modis.flood_algorithms
import cmt.radar.flood_algorithms
import cmt.util.landsat_functions

from cmt.util.imageRetrievalFunctions import getCloudFreeModis, getCloudFreeLandsat, getNearestSentinel1

'''
Command line flood detection tool
'''

#  --------------------------------------------------------------
# Configuration

# TODO: Load from a config file

#ALGORITHMS = [DIFFERENCE, EVI, XIAO]
ALGORITHMS = []



# --------------------------------------------------------------
# Functions

#def getOutputFolder(domain):
#    '''Get a folder to save output results to'''
#    dateString = getDate(domain)
#    folder     = os.path.join(OUTPUT_FOLDER, dateString)
#    return folder
        

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
    
    if domain.has_sensor('modis'):
        print 'Running ADABOOST flood detection...'
        return cmt.modis.flood_algorithms.detect_flood(domain, cmt.modis.flood_algorithms.ADABOOST)[1]

    try:
        domain.get_landsat()
        print 'Running LANDSAT-only flood detection...'
        return cmt.util.landsat_functions.detect_water(domain.get_landsat().image)
    except LookupError:
        pass
        
    try:
        domain.get_radar()
        print 'Running RADAR-only flood detection...'
        #return cmt.radar.flood_algorithms.detect_flood(domain, cmt.radar.flood_algorithms.MATGEN)[1]
        return domain.get_radar().vv.lt(-15) # A dumb debug function!
    except LookupError:
        pass    

    raise Exception('No data available to detect a flood!')

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


# TODO: EE implementation works ok, just use that.
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

def coordListsToKml(coordLists, typeList, kmlPath):
    '''Converts a local coordinate list to KML'''
    
    MIN_REGION_SIZE = 0.000001
    
    # Initialize kml document
    kml = simplekml.Kml()
    kml.document.name = 'ASP CMT flood detections - DATE'
    kml.hint = 'target=earth'

    height = 2000

    for (coords, isWater) in zip(coordLists, typeList):
        
        #print 'Num pruned coords: ' + str(len(prunedCoords))
        #if len(prunedCoords) < 3: # Skip invalid polygons
        #    continue
 
        # Separate the outer and inner coordinate lists that were passed in
        outerBounds = clean_coordinates(coords[0], height)
        innerBounds = []
        for i in range(1,len(coords)):
            innerBounds.append(clean_coordinates(coords[i], height))
  
        # Make a polygon with the provided borders    
        poly = kml.newpolygon(outerboundaryis=outerBounds,
                              innerboundaryis=innerBounds)

        poly.altitudemode = simplekml.AltitudeMode.relativetoground
        poly.extrude=1
        if isWater:
            poly.style.linestyle.color = simplekml.Color.blue
            #poly.style.polystyle.color = '37F0E614' # Translucent teal
            poly.style.polystyle.color = 'FFF0E614' # Solid teal
            poly.style.polystyle.fill  = 1
    
    
        #height += 100
    
    # Save kml document
    kml.save(kmlPath)


# --------------------------------------------------------------
def main(argsIn):

    logger = logging.getLogger()

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
          
          (options, args) = parser.parse_args(argsIn)

          if len(args) < 5:
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

    # TODO: Make sure our algorithms can use these input formats!

    # Try to load an image from each of the sensors and wrap it in
    #  a SensorObservation object that we can feed into a domain
  
    print 'Loading sensors...'
    modisSensor     = None
    landsatSensor   = None
    sentinel1Sensor = None
    sensorList = []
    try:
        #print 'Fetching MODIS data...'
        modisImage  = getCloudFreeModis(eeBounds, eeDate, options.searchRangeDays, options.maxCloudPercentage)
        modisSensor = cmt.domain.SensorObservation()
        modisSensor.init_from_image(modisImage, 'modis')
        sensorList.append(modisSensor)
        #print modisSensor   
        #print 'Loaded MODIS sensor observation!'
    except Exception as e:
        print 'Unable to load a MODIS image in this date range!'
        print str(e)
    try:
        #print 'Fetching Landsat data...'
        landsatImage  = getCloudFreeLandsat(eeBounds, eeDate, options.searchRangeDays, options.maxCloudPercentage)
        landsatName   = cmt.util.landsat_functions.get_landsat_name(landsatImage)
        landsatSensor = cmt.domain.SensorObservation()
        landsatSensor.init_from_image(landsatImage, landsatName)
        sensorList.append(landsatSensor)
        #print landsatSensor
        #print 'Loaded Landsat sensor observation!'
    except Exception as e:
        print 'Unable to load a Landsat image in this date range!'
        print str(e)
    try:
        #print 'Fetching Sentinel1 data...'
        sentinel1Image  = getNearestSentinel1(eeBounds, eeDate, options.searchRangeDays)
        sentinel1Sensor = cmt.domain.SensorObservation()
        sentinel1Sensor.init_from_image(sentinel1Image, 'sentinel1')
        sensorList.append(sentinel1Sensor)
        #print 'Loaded Sentinel1 sensor observation!'
    except Exception as e:
        print 'Unable to load a Sentinel1 image in this date range!'
        print str(e)

    if not sensorList:
        print 'Unable to find any sensor data for this date/location!'
        return -1


    domainName = 'domain_' + dateString
    domain = cmt.domain.Domain()
    domain.load_sensor_observations(domainName, [minLon, minLat, maxLon, maxLat], sensorList)

    print 'Successfully loaded the domain!'
    print str(domain)
    # TODO: Check the domain loading!

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


    # TODO: Adaboost train command or separate tool?
    #
    # import cmt.modis.adaboost
    # cmt.modis.adaboost.adaboost_learn()         # Adaboost training
    # #cmt.modis.adaboost.adaboost_dem_learn(None) # Adaboost DEM stats collection
    # raise Exception('DEBUG')

    print 'Running flood detection!'
    result = detect_flood(domain)

    resultPath = os.path.join(outputFolder, 'flood_detect_result.tif')
    cmt.util.miscUtilities.safeEeImageDownload(result, eeBounds, 150, resultPath, outputVisParams)

    # Perform an erosion followed by a dilation to clean up small specks of detection
    print 'Filtering result...'
    circle   = ee.Kernel.circle(radius=1);
    filtered = result.focal_min(kernel=circle, iterations=2).focal_max(kernel=circle, iterations=2);

    # Vectorize the binary result image
    print 'Converting to vectors...'
    featureCollection = filtered.reduceToVectors(geometry=eeBounds, scale=outputResolution, eightConnected=False);
    print '\n---Retrieving info---\n'

    # If too many features were detected, keep the N largest ones.
    #MAX_POLYGONS = 100
    MAX_POLYGONS = 1000
    MIN_POLYGONS = 2 # If less than this, must be bad!
    #MIN_AREA = 500000
    MIN_AREA = 0
    
    numFeatures = featureCollection.size().getInfo()
    print 'Detected ' +str(numFeatures)+ ' polygons.'
    if numFeatures < MIN_POLYGONS:
        return 0

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
        return 0

    #if numFeatures > MAX_POLYGONS:
    #    print 'Sorting by area'
    #    featureCollection = featuresWithSize.sort('area', False)    

    # Get a list of the features with simplified geometry
    def simplifyFeature(feature):
        MAX_ERROR = 100
        return ee.Feature(feature).simplify(MAX_ERROR)
        
    fList = featureCollection.toList(MAX_POLYGONS) # Limit total number of polygons
    fList = fList.map(simplifyFeature)
    numFeatures = fList.size().getInfo()

    print 'Grabbing polygon ifo...'
    allFeatureInfo = fList.getInfo()
    print '...done'
    
    coordLists = []
    typeList   = []
    
    print 'Restricted to ' +str(numFeatures)+ ' polygons.'
    #for i in range(0,numFeatures):
    for featureInfo in allFeatureInfo:
       
        #print featureInfo['geometry']['type']
        #print featureInfo['properties']['area']
        
        if featureInfo['geometry']['type'] == 'MultiPolygon':
        
            # TODO: Handle these better?
        
            #print featureInfo
            # These are multi level, add each one as a type of feature
            multiList = featureInfo['geometry']['coordinates']
            print 'MULTIPOLYGON: ' + str(len(multiList))
            for a in multiList:
                print len(a)
                for b in a: 
                    print '-- ' + str(len(b))
                    typeList.append(featureInfo['properties']['label'])
                    coordLists.append(b)
                    
            #print multiList
            #raise Exception('DEBUG')

            ## Extract all the coordinates from the multipolygon
            ##  (which may be multiple levels deep) and flatten them
            ##  into a single long list. 
            #multiList = featureInfo['geometry']['coordinates']
            #s = str(multiList).replace('[','').replace(']','')
            #parts = s.split(',')
            #coordList = []
            #for j in range(0,len(parts),2):
            #    coordList.append((float(parts[j]),float(parts[j+1])))

        else:
            # TODO
            theseCoords = featureInfo['geometry']['coordinates']
            
            typeList.append(featureInfo['properties']['label'])
            coordLists.append(theseCoords)

    # Quit now if we did not find anything
    if numFeatures == 0:
        return 0

        
    print 'Converting coordinates to KML'
    kmlPath = os.path.join(outputFolder, 'floodCoords.kml')
    coordListsToKml(coordLists, typeList, kmlPath)
        
        
    return 0
    
    # Code attempting to use EE to do the polygon work...
    
    featureInfo = features.getInfo()
    try:
        print dir(feature)
    except:
        pass
    print 'Found ' + str(len(featureInfo)) + ' features.'
    for feature in featureInfo:
        print '- Found ' + str(len(feature)) + ' subfeatures.'
        print dir(feature)

    raise Exception('DEBUG')

    # For each of the algorithms
    for a in range(len(ALGORITHMS)):
        # Run the algorithm on the data and get the results
        try:
            (alg, result) = cmt.modis.flood_algorithms.detect_flood(domain, ALGORITHMS[a])
            if result is None:
                logger.warn('Did not get result for algorithm ' + alg)
                continue
            
            thisFileName = alg + '-result.tif'
            resultPath   = os.path.join(resultFolder, thisFileName)
            
            # Record the result to disk
            cmt.util.miscUtilities.downloadEeImage(result, eeBounds, outputResolution, resultPath, outputVisParams)
                    
        except Exception, e:
            print('Caught exception running algorithm: ' + get_algorithm_name(ALGORITHMS[a]) + '\n' +
                  str(e) + '\n')


# Call main() when run from command line
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


