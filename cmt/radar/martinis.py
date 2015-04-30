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
from cmt.mapclient_qt import addToMap

#------------------------------------------------------------------------
''' sar_martinis radar algorithm (find threshold by histogram splits on selected subregions)

 Algorithm from paper:
    "Towards operational near real-time flood detection using a split-based
    automatic thresholding procedure on high resolution TerraSAR-X data"
    by S. Martinis, A. Twele, and S. Voigt, Nat. Hazards Earth Syst. Sci., 9, 303-314, 2009


 This algorithm seems extremely sensitive to multiple threshold and
  scale parameters.  So far it has not worked well on any data set.
'''


RED_PALETTE   = '000000, FF0000'
BLUE_PALETTE  = '000000, 0000FF'
TEAL_PALETTE  = '000000, 00FFFF'
LBLUE_PALETTE = '000000, ADD8E6'
GREEN_PALETTE = '000000, 00FF00'
GRAY_PALETTE  = '000000, FFFFFF'

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

def divideUpBounds(bounds, boxSizeMeters, maxBoxesPerSide):
    '''Divides up a single boundary into a grid based on a grid size in meters'''

    # Get the four corners of the box and side widths in meters
    (minLon, minLat, maxLon, maxLat) = getBoundingBox(bounds)
    bottomLeft  = ee.Geometry.Point(minLon, minLat)
    topLeft     = ee.Geometry.Point(minLon, maxLat)
    bottomRight = ee.Geometry.Point(maxLon, minLat)
    topRight    = ee.Geometry.Point(maxLon, maxLat)
    
    height = float(bottomLeft.distance(topLeft).getInfo())
    width  = float(bottomLeft.distance(bottomRight).getInfo())
    
    # Determine the number of boxes
    numBoxesX = int(math.ceil(width  / boxSizeMeters))
    numBoxesY = int(math.ceil(height / boxSizeMeters))
    if numBoxesX > maxBoxesPerSide:
        numBoxesX = maxBoxesPerSide
    if numBoxesY > maxBoxesPerSide:
        numBoxesY = maxBoxesPerSide
    boxSizeMeters = ((width/numBoxesX) + (height/numBoxesY)) / 2
    print 'Using ' + str(numBoxesX*numBoxesY) + ' boxes of size ' + str(boxSizeMeters)
    
    # Now compute the box boundaries in degrees
    boxWidthLon  = (maxLon - minLon) / numBoxesX
    boxHeightLat = (maxLat - minLat) / numBoxesY 
    y = minLat
    boxList = []
    for r in range(0,numBoxesY):
        y = y + boxHeightLat
        x = minLon
        for c in range(0,numBoxesX):
            x = x + boxWidthLon
            boxBounds = ee.Geometry.Rectangle(x, y, x+boxWidthLon, y+boxHeightLat)
            #print boxBounds
            boxList.append(boxBounds)

    return boxList, boxSizeMeters
    
  
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


#
#def __show_histogram(histogram, binCenters):
#    '''Create a plot of a histogram'''
#    plt.bar(binCenters, histogram)
#
#    plt.show()


#def __show_histogram(histogram, params=None):
#    '''Create a plot of a histogram'''
#    #values = histogram['histogram']
#    #start  = histogram['bucketMin']
#    #width  = histogram['bucketWidth']
#    ind    = numpy.arange(start=start, stop=start + width * len(values), step=width)[:-1]
#    plt.bar(ind, height=values[:-1], width=width, color='b')
#    #if params != None:
#    #    m = domains.MINIMUM_VALUES[instrument]
#    #    if instrument == domains.UAVSAR:
#    #        m = math.log10(m)
#    #    mid = int((params[0] - start) / width)
#    #    cumulative = sum(values[:mid]) + values[mid] / 2
#    #    scale = cumulative / __cdf(params, m, params[0])
#    #    plt.bar(ind, map(lambda x : scale * (__cdf(params, m, x + width / 2) - __cdf(params, m, x - width 
#

def applyCutlerLinearLogScale(grayImage, roi):
    '''Translates the input SAR image into a hybrid linear-log scale as described in
        "Robust automated thresholding of SAR imagery for open-water detection"
        by Patrick J Cutler and Frederick W Koehler'''
    
    TOP_SECTION_PERCENTILE = 99
    TOP_SECTION_START      = 221
    topRange = 256 - TOP_SECTION_START
    
    # Compute a histogram of the entire area
    # - Do this at a lower resolution to reduce computation time
    
    PERCENTILE_SCALE = 50 # Resolution in meters to compute the percentile at
    percentiles = grayImage.reduceRegion(ee.Reducer.percentile([0, TOP_SECTION_PERCENTILE, 100], ['min', 'split', 'max']),
                                         roi, PERCENTILE_SCALE).getInfo()
    
    # Extracting the results is annoying because EE prepends the channel name
    minVal   = next(val for key, val in percentiles.items() if 'min'   in key)
    splitVal = next(val for key, val in percentiles.items() if 'split' in key)
    maxVal   = next(val for key, val in percentiles.items() if 'max'   in key)
    
    lowRange    = splitVal - minVal
    
    logMin   = math.log10(splitVal)
    logMax   = math.log10(maxVal)
    logRange = logMax - logMin
    
    #addToMap(grayImage.select(['vh']), {}, 'vh',   False)
    # Intensities from 0  to  98th percent are mapped to   0 - 220 on a linear scale
    # Intensities from 99 to 100th percent are mapped to 221 - 255 on a log scale
    lowMask       = grayImage.lt(splitVal )
    highMask      = grayImage.gte(splitVal)
    #addToMap(lowMask,  {'min': 0, 'max': 1,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'low range',   False)
    #addToMap(highMask, {'min': 0, 'max': 1,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'high range',   False)
    linearPortion = grayImage.subtract(minVal).divide(lowRange).multiply(TOP_SECTION_START-1).multiply(lowMask )#.uint8()
    logPortion    = grayImage.log10().subtract(logMin).divide(logRange).multiply(topRange).add(TOP_SECTION_START).multiply(highMask)
    #addToMap(linearPortion, {'min': 0, 'max': 255,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'linear',   False)
    #addToMap(logPortion,    {'min': 0, 'max': 255,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'log',   False)
    scaledImage   = linearPortion.add(logPortion)
    
    return scaledImage

def sar_martinis_cr(domain):
    '''Just calls sar_martinis with the CR option instead of the default CV option'''
    return sar_martinis(domain, True)

def sar_martinis(domain, cr_method=False):
    '''Compute a global threshold via histogram splitting on selected subregions'''
    
    sensor     = domain.get_radar()
    radarImage = sensor.image

    # Many papers reccomend a median type filter to remove speckle noise.
    
    # 1: Divide up the image into a grid of tiles, X
    
    # Divide up the region into a grid of subregions
    MAX_BOXES_PER_SIDE = 12 # Cap the number of boxes at 144
    DESIRED_BOX_SIZE_METERS = 3000
    boxList, boxSizeMeters  = divideUpBounds(domain.bounds, DESIRED_BOX_SIZE_METERS, MAX_BOXES_PER_SIDE)
    
    # Extract the center point from each box
    centersList = map(getBoundsCenter, boxList)
    
    # SENTINEL = 12m/pixel
    KERNEL_SIZE = 13 # Each box will be covered by a 13x13 pixel kernel
    metersPerPixel = boxSizeMeters / KERNEL_SIZE
    print 'Using metersPerPixel: ' + str(metersPerPixel)
    
    avgKernel   = ee.Kernel.square(KERNEL_SIZE, 'pixels', True); # <-- EE fails if this is in meters!

    # Select the radar layer we want to work in
    if 'water_detect_radar_channel' in domain.algorithm_params:
        channelName = domain.algorithm_params['water_detect_radar_channel']
    else: # Just use the first radar channel
        channelName = sensor.band_names[0]   
    
    # Rescale the input data so the statistics are not dominated by very bright pixels
    GRAY_MAX  = 255
    grayLayer = applyCutlerLinearLogScale(radarImage.select([channelName]), domain.bounds)
    #addToMap(grayLayer, {'min': 0, 'max': GRAY_MAX,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'grayLayer',   False)
    
    
    # Compute the global mean, then make a constant image out of it.
    globalMean      = grayLayer.reduceRegion(ee.Reducer.mean(), domain.bounds, metersPerPixel)
    globalMeanImage = ee.Image.constant(globalMean.getInfo()[channelName])
    
    print 'global mean = ' + str(globalMean.getInfo()[channelName])
    
    
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
    #addToMap(meanImage, {'min': 0, 'max': GRAY_MAX, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'Mean',   False)
    #addToMap(stdImage,  {'min': 0, 'max': 40,       'opacity': 1.0, 'palette': GRAY_PALETTE}, 'StdDev', False)
        
    # Compute these two statistics across the entire image
    CV = meanImage.divide(stdImage).reproject(       "EPSG:4326", None, metersPerPixel)
    R  = meanImage.divide(globalMeanImage).reproject("EPSG:4326", None, metersPerPixel)
    
    
    # 2: Prune to a reduced set of tiles X'
    
    # Parameters which control which sub-regions will have their histograms analyzed
    # - These are strongly influenced by the smoothing kernel size!!!
    MIN_CV = 0.7
    MAX_CV = 1.3
    MAX_R  = 1.1
    MIN_R  = 0.5
    
    # Debug plots
    addToMap(CV, {'min': 0, 'max': 4.0, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'CV', False)
    addToMap(R,  {'min': 0, 'max': 4.0, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'R',  False)
    
    if cr_method:
        MIN_CR = 0.10
    
        # sar_griefeneder reccomends replacing CV with CR = (std / gray value range), min value 0.05
        imageMin  = grayLayer.reduceRegion(ee.Reducer.min(), domain.bounds, metersPerPixel).getInfo()[channelName]
        imageMax  = grayLayer.reduceRegion(ee.Reducer.max(), domain.bounds, metersPerPixel).getInfo()[channelName]
        grayRange = imageMax - imageMin
        CR = stdImage.divide(grayRange)
    
        #addToMap(CR, {'min': 0, 'max': 0.3, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'CR', False)
    
    
    # Filter out pixels based on computed statistics
    t1 = CV.gte(MIN_CV)
    t2 = CV.lte(MAX_CV)
    t3 = R.gte(MIN_R)
    t4 = R.lte(MAX_R)
    if cr_method:
        temp = CR.gte(MIN_CR).And(t3).And(t4)
    else:
        temp = t1.And(t2).And(t3).And(t4)
    X_prime = temp.reproject("EPSG:4326", None, metersPerPixel)
    addToMap(X_prime.mask(X_prime),  {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': TEAL_PALETTE}, 'X_prime',  False)
       
    # 3: Prune again to a final set of tiles X''
    
    # Further pruning happens here but for now we are skipping it and using
    #  everything that got by the filter.  This would speed local computation.
    # - This is equivalent to using a large number for N'' in the original paper
    #    (which does not suggest a value for N'')
    X_doublePrime = X_prime
    
    
    # 4: For each tile, compute the optimal threshold
    
    # Assemble all local gray values at each point ?
    localPixelLists = grayLayer.neighborhoodToBands(avgKernel)
        
    maskWrapper = ee.ImageCollection([X_doublePrime]);
    collection  = ee.ImageCollection([localPixelLists]);
    
    # Extract the point data at from each sub-region!
    
    localThresholdList = []
    usedPointList      = []
    rejectedPointList  = []
    for loc in centersList:
        
        try:
            thisLoc = ee.Geometry.Point(loc[1], loc[0])
        
            # If the mask for this location is invalid, skip this location
            maskValue = maskWrapper.getRegion(thisLoc, metersPerPixel);
            maskValue = maskValue.getInfo()[1][4] # TODO: Not the best way to grab the value!
            if not maskValue:
                rejectedPointList.append(thisLoc)
                continue
        
            # Otherwise pull down all the pixel values surrounding this center point
            
            pointData = collection.getRegion(thisLoc, metersPerPixel)
            pixelVals = pointData.getInfo()[1][4:] # TODO: Not the best way to grab the value!
    
            # TODO: Can EE handle making a histogram around this region or do we need to do this ourselves?
            #pointData = localPixelLists.reduceRegion(thisRegion, ee.Reducer.histogram(), SAMPLING_SCALE);
            #print pointData.getInfo()
            #print pixelVals
            #__show_histogram(pixelVals)
            #plt.bar(range(len(pixelVals)), pixelVals)
            
            # Compute a histogram from the pixels (TODO: Do this with EE!)
            NUM_BINS = 256
            hist, binEdges = numpy.histogram(pixelVals, NUM_BINS)
            binCenters = numpy.divide(numpy.add(binEdges[:NUM_BINS], binEdges[1:]), 2.0)
            
            # Compute a split on the histogram
            splitVal = histogram.splitHistogramKittlerIllingworth(hist, binCenters)
            print "Computed local threshold = " + str(splitVal)
            localThresholdList.append(splitVal)
            usedPointList.append(thisLoc)
            
            #plt.bar(binCenters, hist)
            #plt.show()
        except Exception,e:
            print 'Failed to compute a location:'
            print str(e)
       
    numUsedPoints   = len(usedPointList)
    numUnusedPoints = len(rejectedPointList)

    if (numUsedPoints > 0):
        usedPointListEE = ee.FeatureCollection(ee.Feature(usedPointList[0]))
        for i in range(1,numUsedPoints):
            temp = ee.FeatureCollection(ee.Feature(usedPointList[i]))
            usedPointListEE = usedPointListEE.merge(temp)       

        usedPointsDraw = usedPointListEE.draw('00FF00', 8)
        addToMap(usedPointsDraw, {}, 'Used PTs', False)
        
    if (numUnusedPoints > 0):
        unusedPointListEE = ee.FeatureCollection(ee.Feature(rejectedPointList[0]))
        for i in range(1,numUnusedPoints):
            temp = ee.FeatureCollection(ee.Feature(rejectedPointList[i]))
            unusedPointListEE = unusedPointListEE.merge(temp)       

        unusedPointsDraw = unusedPointListEE.draw('FF0000', 8)
        addToMap(unusedPointsDraw, {}, 'Unused PTs', False)
    
    
    # 5: Use the individual thresholds to compute a global threshold 
    
    computedThreshold = numpy.median(localThresholdList) # Nothing fancy going on here!
    
    print 'Computed global threshold = ' + str(computedThreshold)
    
    finalWaterClass = grayLayer.lte(computedThreshold)

    #addToMap(finalWaterClass.mask(finalWaterClass),  {'min': 0, 'max': 1, 'opacity': 0.6, 'palette': RED_PALETTE}, 'mirtinis class',  False)
    
    # Rename the channel to what the evaluation function requires
    finalWaterClass = finalWaterClass.select([channelName], ['b1'])
    
    return finalWaterClass
    
    
