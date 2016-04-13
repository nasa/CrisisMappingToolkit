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

import cmt.domain
import cmt.util.processManyLakes
import cmt.modis.flood_algorithms
import cmt.util.evaluation
import cmt.util.miscUtilities
from   cmt.util.processManyLakes import LakeDataLoggerBase
import cmt.util.imageRetrievalFunctions



class LoggingClass(LakeDataLoggerBase):
    '''Log MODIS flood detection results for a lake compared with the permanent water mask'''

    def __init__(self, logDirectory, ee_lake, lake_name):
        '''Open and prep the output file'''
        # Base class init function
        LakeDataLoggerBase.__init__(self, logDirectory, ee_lake, lake_name)

        # Get the file path
        filePrefix = LakeDataLoggerBase.computeLakePrefix(self)
        self.logFolder = filePrefix + os.path.sep
        self.logPath = os.path.join(self.logFolder, 'MODIS_log.csv')

        # Read in any existing data from the file
        self.entryList = LoggingClass.readAllEntries(self.logPath)

    def __del__(self):
        '''On destruction write out the file to disk'''
        if self.entryList:
            self.writeAllEntries()

    def getLakeDirectory(self):
        '''The folder where the log is written'''
        return self.logFolder

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
        return cmt.util.miscUtilities.downloadEeImage(mergedImage, ee_bounds, resolution, imagePath, vis_params)
        #return cmt.util.miscUtilities.downloadEeImage(cloudRgb, ee_bounds, resolution, imagePath, vis_params)

    def saveModisImage(self, modisImage, ee_bounds, imageName):
        '''Record the input MODIS image to the log directory'''

        if not os.path.exists(self.logFolder): # Create folder if it does not exist
            os.mkdir(self.logFolder)

        imagePath  = os.path.join(self.logFolder, imageName)
        vis_params = {'min': 0, 'max': 8000, 'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06']}
        if not os.path.exists(imagePath): # Don't overwrite this image
            return cmt.util.miscUtilities.downloadEeImage(modisImage, ee_bounds, 250, imagePath, vis_params)


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
            if k in ['date', 'satellite']:  # Don't double-write these
                continue
            v = dataRecord[k]
            if v is False:  # Log invalid data
                s += ', '+k+', NA, NA, NA'
            else:  # Log valid data: Algorithm, precision, recall, eval_resolution
                s += ', '+k+', '+str(v[0])+', '+str(v[1])+', '+str(v[2])
        return (s + '\n')

    @staticmethod
    def lineToDict(line):
        '''Extract the information from a single line in the log file in to a dictionary object'''

        MAX_ALGS = 15  # Used for sanity check
        NUM_HEADER_VALS = 2  # Date, MODIS
        ELEMENTS_PER_ALGORITHM = 4  # (alg name, precision, recall, eval_res)
        thisDict = dict()

        parts = line.split(',')
        numAlgs = (len(parts) - NUM_HEADER_VALS) / ELEMENTS_PER_ALGORITHM  # Date, MODIS, (alg name, precision, recall, eval_res)...
        if numAlgs > MAX_ALGS:  # Error checking
            print line
            raise Exception('Error: Too many algorithms found!')
        thisDict['date'] = parts[0]
        thisDict['satellite'] = parts[1]
        for i in range(numAlgs):  # Loop through each algorithm
            startIndex = i*ELEMENTS_PER_ALGORITHM + NUM_HEADER_VALS
            algName = parts[startIndex].strip()
            if (parts[startIndex+1].strip() == 'NA'):  # Check if this was logged as a failure
                thisDict[algName] = False
            else:  # Get the successful log results
                precision = float(parts[startIndex+1])
                recall = float(parts[startIndex+2])
                evalRes = float(parts[startIndex+3])
                thisDict[algName] = (precision, recall, evalRes)  # Store the pair of results for the algorithm
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

        if not os.path.exists(self.logFolder):  # Create folder if it does not exist
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


# from cmt.mapclient_qt import centerMap, addToMap
# centerMap(-119, 38, 11)

# Constants used to describe how to treat an algorithm result
KEEP  = 0  # Existing results will be preserved.  Recompute if no entry for data.
RECOMPUTE = 1  # Set an algorithm to this to force recomputation of all results!
RECOMPUTE_IF_FALSE = 2  # Recompute results if we don't have valid results


def getAlgorithmList():
    '''Return the list of available algorithms'''

    # Code, name, recompute_all_results?
    algorithmList = [  # (cmt.modis.flood_algorithms.DEM_THRESHOLD      , 'DEM Threshold',  KEEP),
                     (cmt.modis.flood_algorithms.EVI, 'EVI', KEEP),
                     (cmt.modis.flood_algorithms.XIAO, 'XIAO', KEEP),
                     (cmt.modis.flood_algorithms.DIFF_LEARNED, 'Difference', KEEP),
                     (cmt.modis.flood_algorithms.CART, 'CART', KEEP),
                     (cmt.modis.flood_algorithms.SVM, 'SVM', KEEP),
                     (cmt.modis.flood_algorithms.RANDOM_FORESTS, 'Random Forests', KEEP),
                     # (cmt.modis.flood_algorithms.DNNS, 'DNNS',           KEEP),
                     # (cmt.modis.flood_algorithms.DNNS_REVISED       , 'DNNS Revised',   KEEP),
                     # (cmt.modis.flood_algorithms.DNNS_DEM           , 'DNNS with DEM',  KEEP),
                     (cmt.modis.flood_algorithms.DNNS_DIFF, 'DNNS Diff', KEEP),
                     (cmt.modis.flood_algorithms.DNNS_DIFF_DEM, 'DNNS Diff DEM', KEEP),
                     # (cmt.modis.flood_algorithms.DIFFERENCE_HISTORY , 'Difference with History', KEEP),
                     (cmt.modis.flood_algorithms.DART_LEARNED, 'Dartmouth', KEEP),
                     (cmt.modis.flood_algorithms.MARTINIS_TREE, 'Martinis Tree', KEEP),
                     (cmt.modis.flood_algorithms.MODNDWI_LEARNED, 'Mod NDWI', KEEP),
                     (cmt.modis.flood_algorithms.FAI_LEARNED, 'Floating Algae Index',  KEEP),
                     (cmt.modis.flood_algorithms.ADABOOST, 'Adaboost', KEEP),
                     (cmt.modis.flood_algorithms.ADABOOST_DEM, 'Adaboost DEM', KEEP)
                     ]

    return algorithmList

def needToComputeAlgorithm(currentResults, algInfo):
    '''Return true if we should compute this algorithm'''
    algName = algInfo[1]
    return ((algInfo[2] == RECOMPUTE) or (algName not in currentResults) or
             ((algInfo[2] == RECOMPUTE_IF_FALSE) and (currentResults[algName] is False)) )


def processing_function(bounds, image, image_date, logger):
    '''Detect water using multiple MODIS algorithms and compare to the permanent water mask'''

    # Define a list of all the algorithms we want to test
    algorithmList = getAlgorithmList()

    waterResults = dict()  # This is where results will be stored

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

    MAX_CLOUD_PERCENTAGE = 0.05

    # Needed to change EE formats for later function calls
    eeDate = ee.Date(image_date)
    rectBounds = cmt.util.miscUtilities.unComputeRectangle(bounds.bounds())

    # First check the input image for clouds.  If there are too many just raise an exception.
    cloudPercentage = cmt.modis.modis_utilities.getCloudPercentage(image, rectBounds)
    if cloudPercentage > MAX_CLOUD_PERCENTAGE:
        cmt.util.processManyLakes.addLakeToBadList(logger.getLakeName(), logger.getBaseDirectory(), image_date)
        raise Exception('Input image has too many cloud pixels!')

    # Get the cloud mask and apply it to the input image
    cloudMask = cmt.modis.modis_utilities.getModisBadPixelMask(image)
    maskedImage = image.mask(cloudMask.Not()) # TODO: Verify this is having an effect!

    # Check if the data is all zero
    onCount = maskedImage.select('sur_refl_b01').reduceRegion(ee.Reducer.sum(), bounds, 4000).getInfo()['sur_refl_b01']
    print 'onCount = ' + str(onCount)
    if onCount < 10:
        cmt.util.processManyLakes.addLakeToBadList(logger.getLakeName(), logger.getBaseDirectory(),
                                                   image_date)
        raise Exception('Masked image is blank!')

    # Save the input image
    imageName = 'input_modis_' + str(image_date)
    logger.saveModisImage(image, rectBounds, imageName)

    # Get the permanent water mask
    # - We change the band name to make this work with the evaluation function call further down
    waterMask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'], ['b1'])

    # Pick a training image without clouds.  We just use the same lake one year in the past.
    dateOneYearPrior = eeDate.advance(-1.0, 'year')
    trainingImage    = imageRetrievalFunctions.getCloudFreeModis(dateOneYearPrior, MAX_CLOUD_PERCENTAGE)

    # Generate a pair of train/test domain files for this lake
    training_date = cmt.util.processManyLakes.get_image_date(trainingImage.getInfo())
    testDomainPath, trainDomainPath = cmt.util.miscUtilities.writeDomainFilePair(logger.getLakeName(), bounds,
                                          ee.Date(image_date), ee.Date(training_date), logger.getLakeDirectory())

    # Load the domains using the standard domain class
    fakeDomain     = cmt.domain.Domain(testDomainPath)
    trainingDomain = cmt.domain.Domain(trainDomainPath)

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
            # addToMap(detectedWater, {'min': 0, 'max': 1}, a[1], False)

            # Save image of results so we can look at them later
            # - Try at a high resolution and if that fails try a lower resolution
            imageName = 'alg_' + algName.replace(' ', '_') + '_' + str(image_date)
            FULL_DEBUG_IMAGE_RESOLUTION  = 250  # Pixel resolution in meters
            REDUCED_DEBUG_IMAGE_RESOLUTION = 1000
            try:  # High res output
                logger.saveResultsImage(detectedWater, rectBounds, imageName, cloudMask, waterMask, FULL_DEBUG_IMAGE_RESOLUTION)
            except:
                print 'Retrying download at lower resolution.'
                try:  # Low res output
                    logger.saveResultsImage(detectedWater, rectBounds, imageName, cloudMask, waterMask, REDUCED_DEBUG_IMAGE_RESOLUTION)
                except Exception,e:
                    print 'Saving results image failed with exception --> ' + str(e)

            print 'Evaluating detection results...'

            # Compare the detection result to the water mask
            isFractional = False  # Currently not using fractional evaluation, but maybe we should for DNSS-DEM
            (precision, recall, evalRes, noTruthEval) = cmt.util.evaluation.evaluate_approach(detectedWater, waterMask, rectBounds, isFractional)

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
    outputPath = os.path.join(resultsFolder, 'compiledLogs.csv')
    outputHandle = open(outputPath, 'w')
    print 'Writing composite log file: ' + outputPath

    # Write a header line
    headerLine = 'lake_name'
    for a in algorithmList:
        headerLine += ', ' + str(a[1]) + '_precision, ' + str(a[1]) + '_recall, ' + str(a[1]) + '_eval_res'
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
        return cmt.util.processManyLakes.main(processing_function, LoggingClass,
                                              cmt.util.processManyLakes.get_image_collection_modis)

    # Compile flag found, just compile the logs.
    try:
        dataFolder = sys.argv[pos+1]
    except:
        print 'The data folder must follow "--compile-logs"'
        return 0
    return compileLakeResults(dataFolder)



if __name__ == "__main__":
    sys.exit(main())
