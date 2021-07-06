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
    print('Using ' + str(numBoxesX*numBoxesY) + ' boxes of size ' + str(boxSizeMeters))
    
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

    # Many papers recommend a median type filter to remove speckle noise.

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
    print('Using metersPerPixel: ' + str(metersPerPixel))
    
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
    
    print('global mean = ' + str(globalMean.getInfo()[channelName]))
    
    
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
            print("Computed local threshold = " + str(splitVal))
            localThresholdList.append(splitVal)
            usedPointList.append(thisLoc)
            
            #plt.bar(binCenters, hist)
            #plt.show()
        except Exception as e:
            print('Failed to compute a location:')
            print(str(e))
       
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
    
    print('Computed global threshold = ' + str(computedThreshold))
    
    finalWaterClass = grayLayer.lte(computedThreshold)

    #addToMap(finalWaterClass.mask(finalWaterClass),  {'min': 0, 'max': 1, 'opacity': 0.6, 'palette': RED_PALETTE}, 'martinis class',  False)
    
    # Rename the channel to what the evaluation function requires
    finalWaterClass = finalWaterClass.select([channelName], ['b1'])
    
    return finalWaterClass
    
#=======================================================================
# An updated algorithm from:
#
# "A fully automated TerraSAR-X based flood service"
#


def fuzzMemZ(x, a, b):
    '''Standard Z shaped fuzzy math membership function between a and b'''
    
    c = (a+b)/2.0
    
    #if x < a:
    #    return 1.0
    #if x < c:
    #    return 1.0 - 2*((x-a)/(b-a))**2.0
    #if x < b:
    #    return 2*((x-b)/(b-a))**2.0
    #return 0.0

    case1 = x.expression('1.0 - 2.0*((b(0)-a)/(b-a))**2.0', {'a':a, 'b':b})
    case2 = x.expression('      2.0*((b(0)-b)/(b-a))**2.0', {'a':a, 'b':b})
    output = ee.Image(0.0).where(x.lt(ee.Image(b)), case2).where(x.lt(ee.Image(c)), case1).where(x.lt(ee.Image(a)), ee.Image(1.0))
    return output


def fuzzMemS(x, a, b):
    '''Standard S shaped fuzzy math membership function between a and b'''

    c = (a+b)/2.0

    #if x < a:
    #    return 0.0
    #if x < c:
    #    return 2*((x-a)/(b-a))**2.0
    #if x < b:
    #    return 1.0 - 2*((x-b)/(b-a))**2.0
    #return 1.0


    case1 = x.expression('      2.0*((b(0)-a)/(b-a))**2.0', {'a':a, 'b':b})
    case2 = x.expression('1.0 - 2.0*((b(0)-b)/(b-a))**2.0', {'a':a, 'b':b})
    output = ee.Image(1.0).where(x.lt(ee.Image(b)), case2).where(x.lt(ee.Image(c)), case1).where(x.lt(ee.Image(a)), ee.Image(0.0))
    return output

def rescaleNumber(num, currMin, currMax, newMin, newMax):
    '''Changes the scaling of a number from one range to a new one.'''
    currRange = currMax - currMin
    newRange  = newMax - newMin
    scaled    = (num - currMin) / currRange
    output    = scaled*newRange + newMin
    return output

def sar_martinis2(domain):
    '''Main function of algorithm from:
          Martinis, Sandro, Jens Kersten, and Andre Twele. 
          "A fully automated TerraSAR-X based flood service." 
          ISPRS Journal of Photogrammetry and Remote Sensing 104 (2015): 203-212.
'''

    # Set up the grid sizes we will use
    # TODO: Compute these based on the region size and input resolution!
    BASE_RES = 40; # Input meters per pixel
    S2_DS    = 32; # Size of the smaller grid S-
    S1_DS    = 64; # Size of the larger grid S+
    S1_WIDTH_METERS = BASE_RES*S1_DS;

    sensor = domain.get_radar()

    # Select the radar layer we want to work in
    if 'water_detect_radar_channel' in domain.algorithm_params:
        channelName = domain.algorithm_params['water_detect_radar_channel']
    else: # Just use the first radar channel
        channelName = sensor.band_names[0]   

    # Get the channel and specify higher quality image resampling method    
    rawImage = sensor.image.select(channelName)

    # EE does most of the same preprocessing as the paper but we still need to
    #  duplicate the 0 to 400 scale they used.
    PROC_MIN_VAL =   0.0
    PROC_MAX_VAL = 400.0
    minmax = rawImage.reduceRegion(ee.Reducer.minMax(), domain.bounds, scale=BASE_RES).getInfo();
    minVal = [value for key, value in minmax.items() if 'min' in key.lower()][0]
    maxVal = [value for key, value in minmax.items() if 'max' in key.lower()][0]
    radarImage = rawImage.unitScale(minVal, maxVal).multiply(PROC_MAX_VAL)
    # Also implement the median filter used in the paper
    radarImage = radarImage.focal_median(kernelType='square')

    # - Because we can only call reduceResolution on 64x64 tiles,
    #  downsample the input images to get the correct size.
    gray     = radarImage.reproject(radarImage.projection(), scale=BASE_RES);
    grayProj = gray.projection();

    # Compute mean at S- level, and standard deviation at S+ level.
    s2Mean   = gray.reduceResolution(  ee.Reducer.mean(),   True).reproject(grayProj.scale(S2_DS, S2_DS));
    s1StdDev = s2Mean.reduceResolution(ee.Reducer.stdDev(), True).reproject(grayProj.scale(S1_DS, S1_DS));

    #addToMap(gray,     {'min': PROC_MIN_VAL, 'max': PROC_MAX_VAL, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'gray',   False)
    #addToMap(s2Mean,   {'min': PROC_MIN_VAL, 'max': PROC_MAX_VAL, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 's2Mean', False)
    #addToMap(s1StdDev, {'min':   PROC_MIN_VAL, 'max': PROC_MAX_VAL, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 's1StdDev',  False)

    # Pick the highest STD grid locations
    p = s1StdDev.reduceRegion(ee.Reducer.percentile([95]), domain.bounds);
    if not p:
        raise Exception('Failed to find high STD tiles!')
    thresh = ee.Number(p.getInfo().values()[0]);
    print('Computed 95% std threshold: ' + str(thresh.getInfo()))
    kept = s1StdDev.gt(ee.Image(thresh));
    #addToMap(kept, {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': GREEN_PALETTE}, 'top_std_dev',  False)

    # Add lonlat bands to the tile STD values and get a nice list of kept
    #  tile STD values with the center coordinate of the tile.
    augStdDev  = s1StdDev.addBands(ee.Image.pixelLonLat()).mask(kept);
    stdDevInfo = augStdDev.reduceRegion(ee.Reducer.toList(3), domain.bounds);
    stdDevList = ee.List(stdDevInfo.get('list')); # A necessary bit of casting

    print('Selected ' + str(len(stdDevList.getInfo())) + ' tiles to compute thresholds from.')


    # Define a function to get a S+ tile bounding box from the tile center in stdDevList.
    def getTileBoundingBox(p):
        pixel  = ee.List(p);
        center = ee.Geometry.Point([pixel.get(1), pixel.get(2)]);
        buff   = center.buffer(S1_WIDTH_METERS/2-1);
        box    = buff.bounds();
        return ee.Feature(box);

    # Get the bounding box of each chosen grid location as a Geometry type
    # using the function defined above.
    # - Unfortunately EE will only let us get an approximation.
    boxes = stdDevList.map(getTileBoundingBox);

    # Obtain a histogram for each of the bounding boxes
    features = ee.FeatureCollection(boxes);
    hists    = gray.reduceRegions(features, ee.Reducer.histogram())


    # This is the point where we had to leave Earth Engine behind, hopefully it is not too slow.

    # At each selected grid location, compute a threshold
    tileThresholds = []
    #print('\n\n======= HistProc ========\n')
    #histProc = hists.map(histCheck)
    histData  = hists.getInfo()['features']
    mergeHist = []
    for feature in histData:
        # TODO: Improve/test the splitter?
        hist     = feature['properties']['histogram']
        splitVal = histogram.splitHistogramKittlerIllingworth(hist['histogram'], hist['bucketMeans'])
        
        # Display these values in the original unscaled units.
        splitValDb = rescaleNumber(splitVal, PROC_MIN_VAL, PROC_MAX_VAL, minVal, maxVal)        
        #print 'Computed split value (DB): ' + str(splitValDb)
        tileThresholds.append(splitVal)
    #print('\n\n----------------------\n')

    #print 'Range: ' + str((PROC_MIN_VAL, PROC_MAX_VAL, minVal, maxVal))
    #print tileThresholds

    # If the standard deviation of the local thresholds in DB are greater than this,
    #  the result is probably bad (number from the paper)
    MAX_STD_DB = 5.0

    # The maximum allowed value, from the paper.
    MAX_THRESHOLD_DB = 10.0

    # TODO: Some method do discard outliers
    threshMean = numpy.mean(tileThresholds)
    threshStd  = numpy.std(tileThresholds, ddof=1)
    
    # Recompute these values in the original DB units.
    tileThresholdsDb = [rescaleNumber(x, PROC_MIN_VAL, PROC_MAX_VAL, minVal, maxVal) for x in tileThresholds]
    threshMeanDb = numpy.mean(tileThresholdsDb)
    threshStdDb  = numpy.std(tileThresholdsDb, ddof=1)

    #print 'Mean of tile thresholds: ' + str(threshMean)
    #print 'STD  of tile thresholds: ' + str(threshStd)
    print('Mean of tile thresholds (DB): ' + str(threshMeanDb))
    print('STD  of tile thresholds (DB): ' + str(threshStdDb))

    # TODO: Use an alternate method of computing the threshold like they do in the paper!
    if threshStdDb > MAX_STD_DB:
        raise Exception('Failed to compute a good threshold! STD = ' + str(threshStdDb))

    if threshMeanDb > MAX_THRESHOLD_DB:
        threshMean = rescaleNumber(MAX_THRESHOLD_DB, minVal, maxVal, PROC_MIN_VAL, PROC_MAX_VAL)
        print('Saturating the computed threshold at 10 DB!')
    
    initialThresh = threshMean

    rawWater = radarImage.lte(initialThresh)
    #addToMap(rawWater, {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': GRAY_PALETTE }, 'rawWater',  False)

    # Get information needed for fuzzy logic results filtering

    meanRawValue = radarImage.mask(rawWater).reduceRegion(ee.Reducer.mean(), domain.bounds, scale=BASE_RES).getInfo().values()[0]

    # Compute the number of pixels in each blob, up to the maximum we care about (1000m*m)
    maxBlobSize = 1000/BASE_RES
    blobSizes   = rawWater.mask(rawWater).connectedPixelCount(maxBlobSize)
    #addToMap(blobSizes, {'min':   0, 'max': maxBlobSize, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'blobs',  False)

    # Get DEM information
    dem = domain.get_dem().image
    dem = ee.Image(0).where(dem, dem).select(['constant'], ['elevation']) # Fill in DEM holes with zero elevation
    slopeImage = ee.Terrain.slope(dem)
    #addToMap(slopeImage, {'min':   0, 'max': 30, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'slope',  False)
    
    # Compute mean and std of the elevations of the pixels we marked as water
    waterHeights = dem.mask(rawWater)
    meanWaterHeight = waterHeights.reduceRegion(ee.Reducer.mean(),   domain.bounds, scale=BASE_RES).getInfo()['elevation']
    stdWaterHeight  = waterHeights.reduceRegion(ee.Reducer.stdDev(), domain.bounds, scale=BASE_RES).getInfo()['elevation']
    
    # Compute fuzzy classifications on four categories:
    
    # SAR
    sarFuzz = fuzzMemZ(radarImage, meanRawValue,  initialThresh)
    
    # Elevation
    # TODO: The max value seems a little strange.
    heightFuzz = fuzzMemZ(dem, meanWaterHeight, meanWaterHeight + stdWaterHeight*(stdWaterHeight + 3.5))
    
    # Slope
    slopeFuzz = fuzzMemZ(slopeImage, 0, 15) # Values in degrees
    
    # Body size
    minBlobSize = 250/BASE_RES
    blobFuzz = fuzzMemS(blobSizes, minBlobSize, maxBlobSize).mask(blobSizes)
    
    #addToMap(sarFuzz,    {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': RED_PALETTE  }, 'SAR    fuzz',  False)
    #addToMap(heightFuzz, {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': TEAL_PALETTE }, 'Height fuzz',  False)
    #addToMap(slopeFuzz,  {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': LBLUE_PALETTE}, 'Slope  fuzz',  False)
    #addToMap(blobFuzz,   {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': BLUE_PALETTE }, 'Blob   fuzz',  False)
    
    # Combine fuzzy classification into a single fuzzy classification
    zeroes    = sarFuzz.Not().Or(heightFuzz.Not().Or(slopeFuzz.Not().Or(blobFuzz.Not())))
    meanFuzz  = sarFuzz.add(heightFuzz).add(slopeFuzz).add(blobFuzz).divide(ee.Image(4.0))
    finalFuzz = meanFuzz.And(zeroes.Not())
    
    #addToMap(zeroes,    {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': GRAY_PALETTE }, 'zeroes',  False)
    #addToMap(meanFuzz,  {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': GRAY_PALETTE }, 'mean fuzz',  False)
    #addToMap(finalFuzz, {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': GRAY_PALETTE }, 'final fuzz',  False)
    
    # Apply fixed threshold to get updated flooded pixels
    defuzz = finalFuzz.gt(ee.Image(0.6))
    #addToMap(defuzz, {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': BLUE_PALETTE }, 'defuzz',  False)
    
    # Expand water classification outwards using a lower fuzzy threshold.
    # - This step is a little tough for EE so we approximate using a dilation step.
    dilatedWater = defuzz.focal_max(radius=1000, units='meters')
    finalWater   = dilatedWater.And(finalFuzz.gt(ee.Image(0.45)))
    
    #addToMap(dilatedWater, {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': GRAY_PALETTE }, 'dilatedWater',  False)
    
    # Rename the channel to what the evaluation function requires
    return finalWater.select(['constant'], ['b1'])







