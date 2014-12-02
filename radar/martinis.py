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

import os
import ee
import math
import numpy
import scipy
import scipy.special
import scipy.optimize

import histogram
import matplotlib
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from util.mapclient_qt import addToMap

#------------------------------------------------------------------------
# - sar_martinis radar algorithm (find threshold by histogram splits on selected subregions)

# Algorithm from paper:
#    "Towards operational near real-time flood detection using a split-based
#    automatic thresholding procedure on high resolution TerraSAR-X data"
#    by S. Martinis, A. Twele, and S. Voigt, Nat. Hazards Earth Syst. Sci., 9, 303-314, 2009


# This algorithm seems extremely sensitive to multiple threshold and
#  scale parameters.  So far it has not worked well on any data set.


def getBoundingBox(bounds):
    '''Returns (minLon, minLat, maxLon, maxLat) from domain bounds'''
    
    coordList = bounds['coordinates'][0]
    minLat =  999
    minLon =  999999
    maxLat = -999
    maxLon = -999999
    for c in coordList:
        if c[0] < minLon:
            minLon = c[0]
        if c[0] > maxLon:
            maxLon = c[0]
        if c[1] < minLat:
            minLat = c[1]
        if c[1] > maxLat:
            maxLat = c[1]
    return (minLon, minLat, maxLon, maxLat)

# TODO: Need to accept a size in meters!
def divideUpBounds(bounds, boxCount):
    '''Divides up a single boundary into an NxN grid where N is equal to the boxSize'''
    
    (minLon, minLat, maxLon, maxLat) = getBoundingBox(bounds)
    
    boxSizeX = (maxLon - minLon) / boxCount
    boxSizeY = (maxLat - minLat) / boxCount
    y = minLat
    boxList = []
    for r in range(0,boxCount):
        y = y + boxSizeY
        x = minLon
        for c in range(0,boxCount):
            x = x + boxSizeX
            boxBounds = ee.Geometry.Rectangle(x, y, x+boxSizeX, y+boxSizeY)
            #print boxBounds
            boxList.append(boxBounds)
            
    return boxList
    
  
def getBoundsCenter(bounds):
    '''Returns the center point of a boundary'''
    
    coordList = bounds['coordinates'][0]
    meanLat  = 0
    meanLon  = 0
    for c in coordList:
        meanLat = meanLat + c[1]
        meanLon = meanLon + c[0]
    meanLat = meanLat / len(coordList)
    meanLon = meanLon / len(coordList)
    return (meanLat, meanLon)



def sar_martinis(domain):
    '''Compute a global threshold via histogram splitting on selected subregions'''
    
    RED_PALETTE   = '000000, FF0000'
    BLUE_PALETTE  = '000000, 0000FF'
    TEAL_PALETTE  = '000000, 00FFFF'
    LBLUE_PALETTE = '000000, ADD8E6'
    GREEN_PALETTE = '000000, 00FF00'
    GRAY_PALETTE  = '000000, FFFFFF'
    
    radarImage = domain.image

    # Many papers reccomend a median type filter to remove speckle noise.
    
    # 1: Divide up the image into a grid of tiles, X
    
    # Divide up the region into a grid of subregions
    BOX_COUNT = 10 # TODO: What unit is this?
    boxList  = divideUpBounds(domain.bounds, BOX_COUNT)
    
    # Extract the center point from each box
    centersList = map(getBoundsCenter, boxList)
    #print centersList
    
    # SENTINEL = 12m/pixel
    metersPerPixel = 12.0
    
    KERNEL_SIZE = 25 # <-- TODO!!! The kernel needs to be the size of a box
    avgKernel   = ee.Kernel.square(KERNEL_SIZE, 'pixels', True); # <-- EE fails if this is in meters!
    
    # Select the radar layer we want to work in
    #channelName = 'hv'
    channelName = 'vh'
    grayLayer = radarImage.select([channelName])
    #addToMap(grayLayer, {'min': 3000, 'max': 70000,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'grayLayer',   False)
    addToMap(grayLayer, {'min': 0, 'max': 700,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'grayLayer',   False)
    
    
    
    # Compute the global mean, then make a constant image out of it.
    globalMean = grayLayer.reduceRegion(ee.Reducer.mean(), domain.bounds, 24)
    globalMeanImage = ee.Image.constant(globalMean.getInfo()[channelName])
    
    print 'global mean = ' + str(globalMean.getInfo())
    
    
    # Compute mean and standard deviation across the entire image
    meanImage          = grayLayer.convolve(avgKernel)
    graysSquared       = grayLayer.pow(ee.Image(2))
    meansSquared       = meanImage.pow(ee.Image(2))
    meanOfSquaredImage = graysSquared.convolve(avgKernel)
    meansDiff          = meanOfSquaredImage.subtract(meansSquared)
    stdImage           = meansDiff.sqrt()
    
    # Debug plots
    #addToMap(meanImage, {'min': 3000, 'max': 70000,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'Mean',   False)
    #addToMap(stdImage,  {'min': 3000, 'max': 200000, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'StdDev', False)
    #addToMap(meanImage, {'min': 0, 'max': 700,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'Mean',   False)
    #addToMap(stdImage,  {'min': 0, 'max': 60, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'StdDev', False)
    
    reprojectDist = 5#metersPerPixel
    
    # Compute these two statistics across the entire image
    CV = meanImage.divide(stdImage).reproject("EPSG:4326", None, reprojectDist)
    R  = meanImage.divide(globalMeanImage).reproject("EPSG:4326", None, reprojectDist)
    
    # TODO: Another paper reccomends replacing CV with CR = (std / gray value range), min value 0.05
    
    # Debug plots
    #addToMap(CV, {'min': 0, 'max': 3, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'CV', False)
    #addToMap(R,  {'min': 0, 'max': 2, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'R',  False)
    
    
    # 2: Prune to a reduced set of tiles X'
    
    # Parameters which control which sub-regions will have their histograms analyzed
    # - These are strongly influenced by the smoothing kernel size!!!
    #MIN_CV = 0.7
    #MAX_R  = 0.9
    #MIN_R  = 0.4
    MIN_CV = 0.7
    MAX_CV = 3.0
    MAX_R  = 0.9
    MIN_R  = 0.3
    
    # Filter out pixels based on computed statistics
    t1   = CV.gte(MIN_CV)
    t2   = CV.lte(MAX_CV)
    t3   = R.gte(MIN_R)
    t4   = R.lte(MAX_R)
    temp = t1.And(t2).And(t3).And(t4)
    X_prime = temp.reproject("EPSG:4326", None, reprojectDist)
    #addToMap(X_prime.mask(X_prime),  {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': TEAL_PALETTE}, 'X_prime',  False)
   
    
    # 3: Prune again to a final set of tiles X''
    
    # Further pruning happens here but for now we are skipping it and using
    #  everything that got by the filter.  This would speed local computation.
    X_doublePrime = X_prime
    
    
    # 4: For each tile, compute the optimal threshold
    
    # Assemble all local gray values at each point ?
    localPixelLists = grayLayer.neighborhoodToBands(avgKernel)
        
    maskWrapper = ee.ImageCollection([X_doublePrime]);
    collection  = ee.ImageCollection([localPixelLists]);
    
    # Extract the point data at from each sub-region!
    
    SAMPLING_SCALE = 30 # meters?
    localThresholdList = []
    usedPointList      = []
    rejectedPointList  = []
    for loc in centersList:
        thisLoc = ee.Geometry.Point(loc[1], loc[0])
    
        # If the mask for this location is invalid, skip this location
        maskValue = maskWrapper.getRegion(thisLoc, SAMPLING_SCALE);
        maskValue = maskValue.getInfo()[1][4] # TODO: Not the best way to grab the value!
        if not maskValue:
            rejectedPointList.append(thisLoc)
            continue
    
        # Otherwise pull down all the pixel values surrounding this center point
        # TODO: Can EE handle making a histogram around this region or do we need to do this ourselves?
        #pointData = localPixelLists.reduceRegion(thisRegion, ee.Reducer.histogram(), SAMPLING_SCALE);
        #print pointData.getInfo()
        #__show_histogram1(pixelVals, domain.instrument)
        
        pointData = collection.getRegion(thisLoc, SAMPLING_SCALE)
        pixelVals = pointData.getInfo()[1][4:] # TODO: Not the best way to grab the value!
        
        # Compute a histogram from the pixels (TODO: Do this with EE!)
        NUM_BINS = 256
        hist, binEdges = numpy.histogram(pixelVals, NUM_BINS)
        binCenters = numpy.divide(numpy.add(binEdges[:NUM_BINS], binEdges[1:]), 2.0)
        
        # Compute a split on the histogram
        splitVal = histogram.splitHistogramKittlerIllingworth(hist, binCenters)
        print "Computed local threshold = " + str(splitVal)
        localThresholdList.append(splitVal)
        usedPointList.append(thisLoc)
        
        #plt.bar(binCenters, histogram)
        #plt.show()
       
    numUsedPoints   = len(usedPointList)
    numUnusedPoints = len(rejectedPointList)

    if (numUsedPoints > 0):
        usedPointListEE = ee.FeatureCollection(ee.Feature(usedPointList[0]))
        for i in range(1,numUsedPoints):
            temp = ee.FeatureCollection(ee.Feature(usedPointList[i]))
            usedPointListEE = usedPointListEE.merge(temp)       

        usedPointsDraw = usedPointListEE.draw('00FF00', 8)
        #addToMap(usedPointsDraw, {}, 'Used PTs', False)
        
    if (numUnusedPoints > 0):
        unusedPointListEE = ee.FeatureCollection(ee.Feature(rejectedPointList[0]))
        for i in range(1,numUnusedPoints):
            temp = ee.FeatureCollection(ee.Feature(rejectedPointList[i]))
            unusedPointListEE = unusedPointListEE.merge(temp)       

        unusedPointsDraw = usedPointListEE.draw('FF0000', 8)
        #addToMap(unusedPointsDraw, {}, 'Unused PTs', False)
    
    
    # 5: Use the individual thresholds to compute a global threshold 
    
    computedThreshold = numpy.mean(localThresholdList) # Nothing fancy going on here!
    
    #print 'Computed global threshold = ' + str(computedThreshold)
    
    finalWaterClass = grayLayer.lte(computedThreshold)
    
    # Rename the channel to what the evaluation function requires
    finalWaterClass = finalWaterClass.select(['vh'], ['b1'])
    
    return finalWaterClass
    
    
