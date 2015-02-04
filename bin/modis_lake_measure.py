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
    Run MODIS based flood detection algorithms on many lakes at a single time
    and log the results compared with the permanent water mask.
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
import time
import os
import ee
import cmt.util.processManyLakes
from cmt.util.processManyLakes import LakeDataLoggerBase
import cmt.modis.flood_algorithms
import cmt.util.evaluation
from cmt.mapclient_qt import downloadEeImage


class LoggingClass(LakeDataLoggerBase):
    '''Log MODIS flood detection results for a lake compared with the permanent water mask'''
    
    def __init__(self, logDirectory, ee_lake):
        '''Open and prep the output file'''
        # Base class init function
        LakeDataLoggerBase.__init__(self, logDirectory, ee_lake)
        
        # Open the file
        filePrefix      = LakeDataLoggerBase.computeLakePrefix(self)
        self.logFolder  = filePrefix + os.path.sep
        if not os.path.exists(self.logFolder): # Create folder if it does not exist
            os.mkdir(self.logFolder)
        logPath         = os.path.join(self.logFolder, 'MODIS_log.txt')
        existingFile    = os.path.exists(logPath) # Check if the file already exists
        print 'DEBUG: Opening log file ' + logPath
        self.fileHandle = open(logPath, 'a+') # Append mode
        
        # Write the header if the file is new
        if not existingFile:
            self.fileHandle.write('date, satellite, algorithm, precision, recall\n')

    def __del__(self):
        '''Close the file on destruction'''
        if self.fileHandle:
            self.fileHandle.close()

    def saveImage(self, classifiedImage, ee_bounds, imageName, waterMask, modisImage):
        '''Records a diagnostic image to the log directory'''
        
        # Currently we are not using the modis image
        # Red channel is detected, blue channel is water mask, green is constant zero.
        mergedImage = classifiedImage.addBands(ee.Image(0)).addBands(waterMask)
        vis_params = {'min': 0, 'max': 1} # Binary image data
        
        DISPLAY_RESOLUTION = 30 # Display at a higher resolution to make things prettier
        imagePath          = os.path.join(self.logFolder, imageName + '.tif')
        return downloadEeImage(mergedImage, ee_bounds, DISPLAY_RESOLUTION, imagePath, 'dummyName', vis_params)

    def findRecordByDate(self, date):
        '''Searches for a record with a particular date and returns it'''
        
        self.fileHandle.seek(0)    # Return to beginning of file
        self.fileHandle.readline() # Skip header line
        while True:
            # Read the next line
            line = self.fileHandle.readline()
            
            # If we hit the end of the file return nothing
            if not line:
                return None
            
            # Return this line if we found the date
            parts = line.split(',')
            if date in parts[0]:
                return line
        
        raise Exception('Should never get here!')
    
    def addDataRecord(self, dataRecord, ee_image=None, ee_bounds=None, resolution=None):
        '''Adds a new record to the log'''
        
        # Add the fixed elements
        s = dataRecord['date']+', '+dataRecord['satellite']
        
        # Add all the algorithm results
        for k in dataRecord:
            v = dataRecord[k]
            if k in ['date', 'satellite']: # Don't double-write these
                continue
            s += ', '+k+', '+str(v[0])+', '+str(v[1])
        
        self.fileHandle.write(s) # The string is automatically written to the end of the file
        
        return True


# TODO: Move to a common file!!!!!!!!
def isRegionInUnitedStates(region):
    '''Returns true if the current region is inside the US.'''
    
    # Extract the geographic boundary of the US.
    nationList = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw')
    nation     = ee.Feature(nationList.filter(ee.Filter.eq('Country', 'United States')).first())
    nationGeo  = ee.Geometry(nation.geometry())
    # Check if the input region is entirely within the US
    result     = nationGeo.contains(region)
    return (str(result.getInfo()) == 'True')

# TODO: Move to a common file!
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

class Object(object):
    '''Helper class to let us add attributes to empty objects'''
    pass

# TODO: Construct an actual domain object!
class FakeDomain(Object):
    '''Class to assist in faking a Domain class instance.'''
    
    def add_dem(self, bounds):
        '''Loads the correct DEM'''
        # Get a DEM
        if isRegionInUnitedStates(bounds):
            self.ned13            = Object()
            self.ned13.image      = ee.Image('ned_13') # US only 10m DEM
            self.ned13.band_names = ['elevation']
            self.ned13.band_resolutions = {'elevation': 10}
        else:
            self.srtm90            = Object()
            self.srtm90.image      = ee.Image('CGIAR/SRTM90_V4') # The default 90m global DEM
            self.srtm90.band_names = ['elevation']
            self.srtm90.band_resolutions = {'elevation': 90}
    
    def get_dem(self):
        '''Returns a DEM image object if one is loaded'''
        try: # Find out which DEM is loaded
            dem = self.ned13
        except:
            try:
                dem = self.srtm90
            except:
                raise Exception('Domain is missing DEM!')
        return dem


# TODO: Do something a little smarter here!
def compute_simple_binary_threshold(valueImage, classification, bounds):
    '''Computes a threshold for a value given examples in a classified binary image'''
    
    # Seperate the values by the binary classification
    valueInFalse = valueImage.mask(classification.Not())
    valueInTrue  = valueImage.mask(classification)
    meanFalse    = valueInFalse.reduceRegion(ee.Reducer.mean(),   bounds) # Set up EE math
    meanTrue     = valueInTrue.reduceRegion( ee.Reducer.mean(),   bounds)
    stdFalse     = valueInFalse.reduceRegion(ee.Reducer.stdDev(), bounds)
    stdTrue      = valueInTrue.reduceRegion( ee.Reducer.stdDev(), bounds)
    meanFalse    = meanFalse.getInfo()['sur_refl_b02'] # Extract the calculated value
    meanTrue     = meanTrue.getInfo()[ 'sur_refl_b02']
    stdFalse     = stdFalse.getInfo()[ 'sur_refl_b02']
    stdTrue      = stdTrue.getInfo()[  'sur_refl_b02']
    
    # Just pick a point between the means based on the ratio of standard deviations
    meanDiff  = meanTrue - meanFalse
    stdRatio  = stdFalse / (stdTrue + stdFalse)
    threshold = meanFalse + meanDiff*stdRatio

    #print 'meanFalse = ' + str(meanFalse)    
    #print 'meanTrue  = ' + str(meanTrue)
    #print 'stdFalse  = ' + str(stdFalse)
    #print 'stdTrue   = ' + str(stdTrue)
    #print 'meanDiff  = ' + str(meanDiff)
    #print 'stdRatio  = ' + str(stdRatio)
    #print 'threshold = ' + str(threshold)
    
    return threshold

def compute_algorithm_parameters(training_domain):
    '''Compute algorithm parameters from a classified training image'''
    
    # Unfortunately we need to recreate a bunch of algorithm code here
    b         = cmt.modis.flood_algorithms.compute_modis_indices(training_domain)
    bounds    = training_domain.bounds
    waterMask = training_domain.ground_truth
    
    modisDiff    = b['b2'].subtract(b['b1'])
    dartmouthVal = b['b2'].add(500).divide(b['b1'].add(2500))
    demHeight    = training_domain.get_dem().image
    
    # These values are computed by comparing the algorithm output in land/water regions
    # - To be passed in to the threshold function a consistent band name is needed
    algorithm_params = dict()
    #print 'MODIS DIFF THRESHOLD'
    algorithm_params['modis_diff_threshold'  ] = compute_simple_binary_threshold(modisDiff,    waterMask, bounds)
    #print 'DARTMOUTH'
    algorithm_params['dartmouth_threshold'   ] = compute_simple_binary_threshold(dartmouthVal, waterMask, bounds)
    #print 'DEM'
    algorithm_params['dem_threshold'         ] = compute_simple_binary_threshold(demHeight.select(['elevation'], ['sur_refl_b02']), waterMask, bounds)
    
    # These would be tougher to compute so we just use some general purpose values
    algorithm_params['modis_mask_threshold'  ] =  4.5
    algorithm_params['modis_change_threshold'] = -3.0

    #print 'Computed the following algorithm parameters: '
    #print algorithm_params
    #print '8888888888888888888888888888888888888888888888888888888888888888888888888888'
    
    
    return algorithm_params


def processing_function(bounds, image, image_date, logger):
    '''Detect water using multiple MODIS algorithms and compare to the permanent water mask'''

    MAX_CLOUD_PERCENTAGE = 0.05

    # First check the input image for clouds.  If there are too many just raise an exception.
    cloudPercentage = cmt.modis.flood_algorithms.getCloudPercentage(image, bounds)
    if cloudPercentage > MAX_CLOUD_PERCENTAGE:
        raise Exception('Input image has too many cloud pixels!')

    # Get the permanent water mask
    # - We change the band name to make this work with the evaluation function call further down
    waterMask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'], ['b1'])
    
    # Needed to change EE formats for later function calls
    rectBounds = unComputeRectangle(bounds.bounds()) 
    
    #print '=========================================='
    #print image.getInfo()
    #for b in image.getInfo()['bands']:
    #    print b['id']
    #    print '--------------------------------------------------------------------'
    
    # Put together a fake domain object and compute the standard 
    fakeDomain       = FakeDomain()
    fakeDomain.modis = Object()
    fakeDomain.modis.sur_refl_b01 = image.select('sur_refl_b01')
    fakeDomain.modis.sur_refl_b02 = image.select('sur_refl_b02')
    fakeDomain.modis.sur_refl_b03 = image.select('sur_refl_b03')
    fakeDomain.modis.sur_refl_b06 = image.select('sur_refl_b06')
    fakeDomain.modis.image        = image
    fakeDomain.ground_truth       = waterMask
    fakeDomain.bounds             = bounds
    fakeDomain.add_dem(bounds)
        
    # Also need to set up a bunch of training information
    
    # First we pick a training image.  We just use the same lake one year in the past.
    eeDate        = ee.Date(image_date)
    trainingStart = eeDate.advance(-1.0, 'year')
    trainingEnd   = eeDate.advance(10.0, 'day') 
    # Fetch a MODIS image for training
    print 'Retrieving training data...'
    modisTrainingCollection = cmt.util.processManyLakes.get_image_collection_modis(bounds, trainingStart, trainingEnd)
    modisTrainingList       = modisTrainingCollection.toList(100)
    modisTrainingInfo       = modisTrainingList.getInfo()
    # Find the first image with a low cloud percentage
    trainingImage = None
    for i in range(len(modisTrainingInfo)):
        thisImage       = ee.Image(modisTrainingList.get(i))
        cloudPercentage = cmt.modis.flood_algorithms.getCloudPercentage(thisImage, bounds)
        if cloudPercentage < MAX_CLOUD_PERCENTAGE:
            trainingImage = thisImage
            break
    if not trainingImage:
        raise Exception('Could not find a training image for date ' + str(image_date))
    
    # Pack the training image into a training domain.
    trainingDomain                    = FakeDomain()
    trainingDomain.modis              = Object()
    trainingDomain.modis.sur_refl_b01 = trainingImage.select('sur_refl_b01')
    trainingDomain.modis.sur_refl_b02 = trainingImage.select('sur_refl_b02')
    trainingDomain.modis.sur_refl_b03 = trainingImage.select('sur_refl_b03')
    trainingDomain.modis.sur_refl_b06 = trainingImage.select('sur_refl_b06')
    trainingDomain.modis.image        = trainingImage
    trainingDomain.ground_truth       = waterMask
    trainingDomain.bounds             = bounds
    trainingDomain.add_dem(bounds)
    fakeDomain.training_domain        = trainingDomain
    
    # Finally, compute a set of algorithm parameters for this image
    # - We use the training image and the water mask to estimate good values where possible.
    fakeDomain.algorithm_params = compute_algorithm_parameters(trainingDomain)


    #vis_params = {'min': 1000, 'max': 3000} # Binary image data
    #downloadEeImage(trainingDomain.get_dem().image, rectBounds, 30, '/home/smcmich1/data/Floods/lakeStudy/Mono_Lake/demTest.tif', 'dem', vis_params)
    #raise Exception('debug')
        
    
    # TODO: Fetch and insert other required information as needed!
    
    # Define a list of all the algorithms we want to test
    algorithmList = [(cmt.modis.flood_algorithms.DEM_THRESHOLD      , 'DEM Threshold'),
                     (cmt.modis.flood_algorithms.EVI                , 'EVI'),
                     (cmt.modis.flood_algorithms.XIAO               , 'XIAO'),
                     (cmt.modis.flood_algorithms.DIFFERENCE         , 'Difference'),
                     (cmt.modis.flood_algorithms.CART               , 'CART'),
                     (cmt.modis.flood_algorithms.SVM                , 'SVM'),
                     (cmt.modis.flood_algorithms.RANDOM_FORESTS     , 'Random Forests'),
                     ##(cmt.modis.flood_algorithms.DNNS               , 'DNNS'),
                     ###(cmt.modis.flood_algorithms.DNNS_REVISED       , 'DNNS Revised'),
                     ##(cmt.modis.flood_algorithms.DNNS_DEM           , 'DNNS with DEM'),
                     #(cmt.modis.flood_algorithms.DIFFERENCE_HISTORY , 'Difference with History'),
                     (cmt.modis.flood_algorithms.DARTMOUTH          , 'Dartmouth'),
                     (cmt.modis.flood_algorithms.MARTINIS_TREE      , 'Martinis Tree') ]

   
    # Loop through each algorithm
    waterResults = dict()
    for a in algorithmList:
        
        print 'Running algorithm ' + a[1]
        
        # Call function to generate the detected water map
        detectedWater = cmt.modis.flood_algorithms.detect_flood(fakeDomain, a[0])[1]
       
        print 'Evaluating detection results...'
        
        
        
        # Compare the detection result to the water mask
        isFractional = False # Currently not using fractional evaluation, but maybe we should for DNSS-DEM
        (precision, recall) = cmt.util.evaluation.evaluate_approach(detectedWater, waterMask, rectBounds, isFractional)
        
        #print 'Evaluation results:'
        #print str(precision) + ' ' + str(recall)
        
        # Store the results for this algorithm
        waterResults[a[1]] = (precision, recall)
        
        # Save image of results so we can look at them later
        
        
        
        imageName = 'alg_' + a[1].replace(' ', '_')
        logger.saveImage(detectedWater, rectBounds, imageName, waterMask, image)
    
    # Return the results for each algorithm
    waterResults['satellite'] = 'MODIS'
    return waterResults



#======================================================================================================
def main():

    # TODO: Command line arguments must specify one day long dates!

    # Call main argument handling function from the supporting file
    return cmt.util.processManyLakes.main(processing_function, LoggingClass, cmt.util.processManyLakes.get_image_collection_modis)



if __name__ == "__main__":
    sys.exit(main())


