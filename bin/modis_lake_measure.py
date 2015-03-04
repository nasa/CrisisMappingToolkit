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
import csv
import ee
import numpy
import traceback
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
        
        # Get the file path
        filePrefix      = LakeDataLoggerBase.computeLakePrefix(self)
        self.logFolder  = filePrefix + os.path.sep
        self.logPath    = os.path.join(self.logFolder, 'MODIS_log.csv')
        
        # Read in any existing data from the file
        self.entryList = LoggingClass.readAllEntries(self.logPath)

    def __del__(self):
        '''On destruction write out the file to disk'''
        if self.entryList:
            self.writeAllEntries()

    def saveResultsImage(self, classifiedImage, ee_bounds, imageName, cloudMask, waterMask, resolution=30):
        '''Records a diagnostic image to the log directory'''
        
        if not os.path.exists(self.logFolder): # Create folder if it does not exist
            os.mkdir(self.logFolder)
        
        # Currently we are not using the modis image
        # Red channel is detected, blue channel is water mask, green is constant zero.
        mergedImage = classifiedImage.addBands(ee.Image(0)).addBands(waterMask)
        mergedImage = mergedImage.Or(cloudMask) #TODO: Make sure this is working!
        vis_params = {'min': 0, 'max': 1} # Binary image data
        
        imagePath          = os.path.join(self.logFolder, imageName + '.tif')
        return downloadEeImage(mergedImage, ee_bounds, resolution, imagePath, vis_params)
        #return downloadEeImage(cloudRgb, ee_bounds, resolution, imagePath, vis_params)

    def saveModisImage(self, modisImage, ee_bounds, imageName):
        '''Record the input MODIS image to the log directory'''
        
        if not os.path.exists(self.logFolder): # Create folder if it does not exist
            os.mkdir(self.logFolder)
        
        imagePath  = os.path.join(self.logFolder, imageName)
        vis_params = {'min': 0, 'max': 8000, 'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06']}
        if not os.path.exists(imagePath): # Don't overwrite this image
            return downloadEeImage(modisImage, ee_bounds, 250, imagePath, vis_params)


    def findRecordByDate(self, date):
        '''Searches for a record with a particular date and returns it'''
        try:
            return self.entryList[date]
        except:
            return None
    
    def addDataRecord(self, dataRecord):
        '''Adds a new record to the log'''
        key = dataRecord['date']
        self.entryList[key] = dataRecord
        
    
    @staticmethod    
    def dictToLine(dataRecord):
        '''Converts an input data record dictionary to a line of text'''
        # Add the fixed elements
        s = dataRecord['date']+', '+dataRecord['satellite']
        
        # Add all the algorithm results
        for k in dataRecord:
            if k in ['date', 'satellite']: # Don't double-write these
                continue
            v = dataRecord[k]
            if v == False: # Log invalid data
                s += ', '+k+', NA, NA, NA'
            else: # Log valid data: Algorithm, precision, recall, eval_resolution
                s += ', '+k+', '+str(v[0])+', '+str(v[1])+', '+str(v[2])
        return (s + '\n')
    
    @staticmethod
    def lineToDict(line):
        '''Extract the information from a single line in the log file in to a dictionary object'''

        MAX_ALGS               = 13
        NUM_HEADER_VALS        = 2 # Date, MODIS 
        ELEMENTS_PER_ALGORITHM = 4 # (alg name, precision, recall, eval_res)
        thisDict = dict()
        
        parts    = line.split(',')
        numAlgs  = (len(parts) - NUM_HEADER_VALS) / ELEMENTS_PER_ALGORITHM # Date, MODIS, (alg name, precision, recall, eval_res)...
        if numAlgs > MAX_ALGS: # Error checking
            print line
            raise Exception('Error: Too many algorithms found!')
        thisDict['date'     ] = parts[0]
        thisDict['satellite'] = parts[1]
        for i in range(numAlgs): # Loop through each algorithm
            startIndex = i*ELEMENTS_PER_ALGORITHM + NUM_HEADER_VALS
            algName    = parts[startIndex].strip()
            if (parts[startIndex+1].strip() == 'NA'): # Check if this was logged as a failure
                thisDict[algName] = False
            else: # Get the successful log results
                precision  = float(parts[startIndex+1])
                recall     = float(parts[startIndex+2])
                evalRes    = float(parts[startIndex+3])
                thisDict[algName] = (precision, recall, evalRes) # Store the pair of results for the algorithm
        return thisDict
    
    @staticmethod
    def readAllEntries(logPath):
        '''Reads the entire contents of the log file into a list of dictionaries'''
        
        # Return an empty dict if the file does not exist
        if not os.path.exists(logPath):
            return dict()

        outputDict = dict()
        fileHandle = open(logPath, 'r')
        line = fileHandle.readline() # Skip header line
        while True:
            # Read the next line
            line = fileHandle.readline()
            
            # If we hit the end of the file return the dictionary
            if not line:
                return outputDict
            
            # Put all the parts of the line into a dictionary
            thisDict = LoggingClass.lineToDict(line)
            
            # Put this dict into an output dictionary
            key = thisDict['date']
            outputDict[key] = thisDict

        raise Exception('Should never get here!')
    
    def writeAllEntries(self):
        '''Dump all the added records to a file on disk'''
                
        if not os.path.exists(self.logFolder): # Create folder if it does not exist
            os.mkdir(self.logFolder)
            
        # Open the file for writing, clobbering any existing file
        fileHandle = open(self.logPath, 'w')
        # Write the header
        fileHandle.write('date, satellite, algorithm, precision, recall, evaluation_resolution\n')
        # Write all the data
        for key in self.entryList:
            line = self.dictToLine(self.entryList[key])
            fileHandle.write(line)
        fileHandle.close()

        return True




# TODO: Move to a common file!!!!!!!!
def isRegionInUnitedStates(region):
    '''Returns true if the current region is inside the US.'''
    
    # Extract the geographic boundary of the US.
    nationList = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw')
    nation     = ee.Feature(nationList.filter(ee.Filter.eq('Country', 'United States')).first())
    nationGeo  = ee.Geometry(nation.geometry())
    # Check if the input region is entirely within the US
    result     = nationGeo.contains(region, 1)
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


#from cmt.mapclient_qt import centerMap, addToMap
#centerMap(-119, 38, 11)

# Constants used to describe how to treat an algorithm result
KEEP               = 0 # Existing results will be preserved.  Recompute if no entry for data.
RECOMPUTE          = 1 # Set an algorithm to this to force recomputation of all results!
RECOMPUTE_IF_FALSE = 2 # Recompute results if we don't have valid results

def getAlgorithmList():
    '''Return the list of available algorithms'''

    # Code, name, recompute_all_results?
    algorithmList = [(cmt.modis.flood_algorithms.DEM_THRESHOLD      , 'DEM Threshold',  KEEP),
                     (cmt.modis.flood_algorithms.EVI                , 'EVI',            KEEP),
                     (cmt.modis.flood_algorithms.XIAO               , 'XIAO',           KEEP),
                     (cmt.modis.flood_algorithms.DIFFERENCE         , 'Difference',     KEEP),
                     (cmt.modis.flood_algorithms.CART               , 'CART',           KEEP),
                     (cmt.modis.flood_algorithms.SVM                , 'SVM',            KEEP),
                     (cmt.modis.flood_algorithms.RANDOM_FORESTS     , 'Random Forests', KEEP ),
                     (cmt.modis.flood_algorithms.DNNS               , 'DNNS',           KEEP),
                     #(cmt.modis.flood_algorithms.DNNS_REVISED       , 'DNNS Revised',  KEEP),
                     (cmt.modis.flood_algorithms.DNNS_DEM           , 'DNNS with DEM',  KEEP),
                     #(cmt.modis.flood_algorithms.DIFFERENCE_HISTORY , 'Difference with History', KEEP), # TODO: May need auto-thresholds!
                     (cmt.modis.flood_algorithms.DARTMOUTH          , 'Dartmouth',      KEEP),
                     (cmt.modis.flood_algorithms.MARTINIS_TREE      , 'Martinis Tree',  KEEP) ]

    return algorithmList

def needToComputeAlgorithm(currentResults, algInfo):
    '''Return true if we should compute this algorithm'''
    algName = algInfo[1]
    return ( (algInfo[2] == RECOMPUTE) or (algName not in currentResults) or
             ((algInfo[2] == RECOMPUTE_IF_FALSE) and (currentResults[algName] == False)) )

def processing_function(bounds, image, image_date, logger):
    '''Detect water using multiple MODIS algorithms and compare to the permanent water mask'''

    # Define a list of all the algorithms we want to test
    algorithmList = getAlgorithmList()

    waterResults = dict() # This is where results will be stored

    # First check if we have already processed this data
    existingResults = logger.findRecordByDate(image_date)
    if existingResults:
        # Go ahead and load the existing results into the output dictionary
        waterResults = existingResults
        
        # Check if we already have all the results we need
        needToRedo = False
        for a in algorithmList:
            # Check conditions to recompute this algorithm result
            if needToComputeAlgorithm(waterResults, a):
                needToRedo = True
                break
            
        if not needToRedo: # If we have everything we need, just return it.
            print 'Nothing new to compute'
            return waterResults
    
    # If we made it to here then we need to run at least one algorithm.
    
    MAX_CLOUD_PERCENTAGE = 0.02

    # Needed to change EE formats for later function calls
    eeDate     = ee.Date(image_date)
    rectBounds = unComputeRectangle(bounds.bounds()) 

    # First check the input image for clouds.  If there are too many just raise an exception.
    cloudPercentage = cmt.modis.flood_algorithms.getCloudPercentage(image, rectBounds)
    if cloudPercentage > MAX_CLOUD_PERCENTAGE:
        raise Exception('Input image has too many cloud pixels!')
    
    # Get the cloud mask and apply it to the input image
    cloudMask   = cmt.modis.flood_algorithms.getModisBadPixelMask(image)
    maskedImage = image.mask(cloudMask.Not()) # TODO: Verify this is having an effect!

    # Save the input image
    imageName = 'input_modis_' + str(image_date)
    logger.saveModisImage(image, rectBounds, imageName)
    

    # Get the permanent water mask
    # - We change the band name to make this work with the evaluation function call further down
    waterMask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'], ['b1'])
    
    # Put together a fake domain object and compute the standard 
    fakeDomain       = FakeDomain()
    fakeDomain.modis = Object()
    fakeDomain.modis.sur_refl_b01 = maskedImage.select('sur_refl_b01')
    fakeDomain.modis.sur_refl_b02 = maskedImage.select('sur_refl_b02')
    fakeDomain.modis.sur_refl_b03 = maskedImage.select('sur_refl_b03')
    fakeDomain.modis.sur_refl_b06 = maskedImage.select('sur_refl_b06')
    fakeDomain.modis.image        = maskedImage
    fakeDomain.modis.get_date     = lambda: eeDate # Fake function that just returns this date
    fakeDomain.ground_truth       = waterMask
    fakeDomain.bounds             = bounds
    fakeDomain.add_dem(bounds)
    
    #addToMap(maskedImage, {'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06'],
    #                      'min': 0, 'max': 3000}, 'MODIS data', True)    
    #addToMap(waterMask, {'min': 0, 'max': 1}, 'Water Mask', False)
    #addToMap(fakeDomain.get_dem(), {'min': 1900, 'max': 2400}, 'DEM', False)
        
    # Also need to set up a bunch of training information
    
    # First we pick a training image.  We just use the same lake one year in the past.
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
        cloudPercentage = cmt.modis.flood_algorithms.getCloudPercentage(thisImage, rectBounds)
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
    fakeDomain.algorithm_params = cmt.modis.flood_algorithms.compute_algorithm_parameters(trainingDomain)


    # Loop through each algorithm
    for a in algorithmList:
        algName = a[1]
        
        # Skip this iteration if we don't need to recompute this algorithm
        if not needToComputeAlgorithm(waterResults, a):
            continue
    
        try:
            print 'Running algorithm ' + algName
            # Call function to generate the detected water map
            detectedWater = cmt.modis.flood_algorithms.detect_flood(fakeDomain, a[0])[1]
            #addToMap(detectedWater, {'min': 0, 'max': 1}, a[1], False)
    
            # TODO: Log this in the future!
            # Try to estimate how accurate the flood detection is without having access to the
            #   ground truth data.
            noTruthEval = cmt.util.evaluation.evaluate_result_quality(detectedWater, rectBounds)
            print 'Eval without truth = ' + str(noTruthEval)

    
            # Save image of results so we can look at them later
            # - Try at a high resolution and if that fails try a lower resolution
            imageName = 'alg_' + algName.replace(' ', '_') +'_'+ str(image_date)
            FULL_DEBUG_IMAGE_RESOLUTION    = 250  # Pixel resolution in meters
            REDUCED_DEBUG_IMAGE_RESOLUTION = 1000
            try: # High res output
                logger.saveResultsImage(detectedWater, rectBounds, imageName, cloudMask, waterMask, FULL_DEBUG_IMAGE_RESOLUTION)
            except:
                print 'Retrying download at lower resolution.'
                try: # Low res output
                    logger.saveResultsImage(detectedWater, rectBounds, imageName, cloudMask, waterMask, REDUCED_DEBUG_IMAGE_RESOLUTION)
                except Exception,e:
                    print 'Saving results image failed with exception --> ' + str(e)
    
        
            print 'Evaluating detection results...'
    
            # Compare the detection result to the water mask
            isFractional = False # Currently not using fractional evaluation, but maybe we should for DNSS-DEM
            (precision, recall, evalRes) = cmt.util.evaluation.evaluate_approach(detectedWater, waterMask, rectBounds, isFractional)
            
            # Store the results for this algorithm
            print 'Evaluation results: ' + str(precision) + ' ' + str(recall) +' at resolution ' + str(evalRes)
            waterResults[algName] = (precision, recall, evalRes, noTruthEval)
                
        except Exception,e: # Handly any failure thet prevents us from obtaining results
            traceback.print_exc(file=sys.stdout)
            print 'Processing results failed with exception --> ' + str(e)
            waterResults[algName] = False # Mark this as a failure

    
    # Return the results for each algorithm
    waterResults['satellite'] = 'MODIS'
    return waterResults


def compileLakeResults(resultsFolder):
    '''Compiles a single csv file comparing algorithm results across lakes'''
    
    # Ignore lakes which did not have this many good days for
    MIN_GOOD_DATES = 1
    
    # Get a list of the algorithms to read
    algorithmList = getAlgorithmList()
        
    # Create the output file
    outputPath   = os.path.join(resultsFolder, 'compiledLogs.csv')
    outputHandle = open(outputPath, 'w')
    print 'Writing composite log file: ' + outputPath
    
    # Write a header line
    headerLine = 'lake_name'
    for a in algorithmList:
        headerLine += ', '+ str(a[1]) +'_precision, '+ str(a[1]) +'_recall, '+ str(a[1]) +'_eval_res'
    outputHandle.write(headerLine + '\n')
        
    # Define local helper function
    def prListStats(prList):
        '''Compute the mean and std of a list of precision/recall value pairs'''
        pList = []
        rList = []
        eList = []
        for i in prList: # Sum the values
            pList.append(i[0]) # Precision
            rList.append(i[1]) # Recall
            eList.append(i[2]) # EvalRes
        return (numpy.mean(pList), numpy.mean(rList), numpy.mean(eList),
                numpy.std(pList),  numpy.std(rList),  numpy.std(eList))
    
    # Loop through the directories
    algStats = dict()
    for d in os.listdir(resultsFolder):

        thisFolder = os.path.join(resultsFolder, d)
        print thisFolder
        if not (os.path.isdir(thisFolder)): # Skip non-folders
            continue
    
        # Skip the directory if it does not contain MODIS_log.csv
        logPath = os.path.join(thisFolder, 'MODIS_log.csv')
        if not os.path.exists(logPath):
            continue
        print 'Reading log file ' + logPath
        
        # Read in the contents of the log file
        dateResultsDict = LoggingClass.readAllEntries(logPath)
        
        # For each algorithm...
        statsDict = dict()
        for a in algorithmList:
            alg = a[1] # Name of the current algorithm

            # Compute the mean precision and recall across all dates for this lake
            prList = []
            for key in dateResultsDict:
                dateResult = dateResultsDict[key]
                try:
                    # Get all the values for this algorithm
                    precision, recall, evalRes = dateResult[alg] 
                    prList.append( (precision, recall, evalRes) )
                except: # This should handle all cases where we don't have data
                    print 'WARNING: Missing results for algorithm ' + alg + ' for lake ' + d
                    
            # Only record something if we got at least one result from the algorithm
            if len(prList) >= MIN_GOOD_DATES:
                # Call local helper function to get the mean precision and recall values
                statsDict[alg] = prListStats(prList)
                
                # Add the means for this algorithm to a list spanning all lakes
                if alg in algStats: # Add to existing list
                    algStats[alg].append(statsDict[alg])
                else: # Start a new list
                    algStats[alg] = [statsDict[alg]]
        
        
        # Build the next line of the output file
        thisLine = d # The line starts with the lake name
        for a in algorithmList:
            alg = a[1]
            try:
                # Add precision, recall, and evaluation resolution
                thisLine += ', '+ str(statsDict[alg][0]) +', '+ str(statsDict[alg][1]) +', '+ str(statsDict[alg][2]) 
            except:
                thisLine += ', NA, NA, NA' # Flag the results as no data!
                print 'WARNING: Missing results for algorithm ' + alg + ' for lake ' + d
        outputHandle.write(thisLine + '\n')
               
    # Add a final summary line containing the means for each algorithm across lakes
    meanSummaries = 'Mean'
    stdSummaries  = 'Standard Deviation'
    for a in algorithmList:
        algName = a[1]
        if algName in algStats: # Extract results
            (pMean, rMean, eMean, pStd, rStd, eStd) = prListStats(algStats[a[1]])
        else: # No results for this data
            (pMean, rMean, eMean, pStd, rStd, eStd) = ('NA', 'NA', 'NA', 'NA', 'NA', 'NA')
        meanSummaries += (', '+ str(pMean) +', '+ str(rMean) +', '+ str(eMean))
        stdSummaries  += (', '+ str(pStd ) +', '+ str(rStd ) +', '+ str(eStd ))
    
    outputHandle.write('\n') # Skip a line
    outputHandle.write(meanSummaries + '\n')
    outputHandle.write(stdSummaries  + '\n')
    outputHandle.write(headerLine) # For convenience reprint the header line at the bottom
    outputHandle.close() # All finished!

    print 'Finished writing log file'
    return 0



#======================================================================================================
def main():

    # Check for the compile logs input flag and if found just compile the logs
    try:
        pos = sys.argv.index('--compile-logs')
    except: # Otherwise call the main argument handling function from the supporting file
        return cmt.util.processManyLakes.main(processing_function, LoggingClass, cmt.util.processManyLakes.get_image_collection_modis)
        
    # Compile flag found, just compile the logs.
    return compileLakeResults(sys.argv[pos+1])



if __name__ == "__main__":
    sys.exit(main())


