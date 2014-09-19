import matgen
import learning

# From Towards an automated SAR-based flood monitoring system:
# Lessons learned from two case studies by Matgen, Hostache et. al.
MATGEN         = 1
RANDOM_FORESTS = 2
DECISION_TREE  = 3
SVM            = 4
MARTINIS       = 5

import math
import numpy
import scipy
import scipy.special
import scipy.optimize

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from util.mapclient_qt import centerMap, addToMap


def __cdf(params, offset, x):
    '''Generate cumulative distribution function?'''
    mode  = params[0]
    k     = params[1]
    theta = (mode - offset) / (k - 1)
    return scipy.special.gammainc(k, (x - offset) / theta)

def __show_histogram(histogram, instrument, params=None):
    '''Create a plot of a histogram'''
    values = histogram['histogram']
    start  = histogram['bucketMin']
    width  = histogram['bucketWidth']
    ind    = numpy.arange(start=start, stop=start + width * len(values), step=width)[:-1]
    plt.bar(ind, height=values[:-1], width=width, color='b')
    if params != None:
        m = domains.MINIMUM_VALUES[instrument]
        if instrument == domains.UAVSAR:
            m = math.log10(m)
        mid = int((params[0] - start) / width)
        cumulative = sum(values[:mid]) + values[mid] / 2
        scale = cumulative / __cdf(params, m, params[0])
        plt.bar(ind, map(lambda x : scale * (__cdf(params, m, x + width / 2) - __cdf(params, m, x - width / 2)), ind), width=width, color='r', alpha=0.5)

def __gamma_function_errors(p, mode, instrument, start, width, values):
    ''''''
    k = p[0]
    if k <= 1.0:
        return [float('inf')] * len(values)
    m = domains.MINIMUM_VALUES[instrument]
    if instrument == domains.UAVSAR:
        m = math.log10(m)
    error = 0.0
    last_cdf = 0.0
    errors = numpy.zeros(len(values))
    mid = int((mode - start) / width)
    cumulative = sum(values[:mid]) + values[mid] / 2
    scale = cumulative / __cdf((mode, k), m, mode)
    for i in range(len(values)):
        if start + i * width - m > mode:
            break
        cdf = scale * __cdf((mode, k), m, start + i * width)
        errors[i] = (cdf - last_cdf) - values[i]
        last_cdf = cdf
    return errors

__WATER_MODE_RANGES = {
        domains.UAVSAR    : {'vv' : (3.0, 4.1), 'hv' : (3.0, 3.6), 'hh' : (3.0, 3.6)},
        domains.SENTINEL1 : {'vv' : (0,   90),  'vh' : (0,   90)}
}

__HISTOGRAM_CLAMP_MAX = {
        domains.UAVSAR:     None,
        domains.SENTINEL1 : 600
}

__NUM_BUCKETS = {
        domains.UAVSAR:    128,
        domains.SENTINEL1: 512
}

def __find_threshold_histogram(histogram, instrument, channel):
    ''''''
    start  = histogram[channel]['bucketMin']
    width  = histogram[channel]['bucketWidth']
    values = histogram[channel]['histogram']

    # find the mode
    i = int((__WATER_MODE_RANGES[instrument][channel][0] - start) / width)
    biggest_bin = i
    while i < len(values) and start + i * width <= __WATER_MODE_RANGES[instrument][channel][1]:
        if values[i] > values[biggest_bin]:
            biggest_bin = i
        i += 1
    mode = start + biggest_bin * width

    # find the other parameters of the distribution
    (value, result) = scipy.optimize.leastsq(__gamma_function_errors, [10], factor=1.0, args=(mode, instrument, start, width, values))
    params = (mode, value[0])

    # choose the best threshold where we can no longer discriminate
    m = domains.MINIMUM_VALUES[instrument]
    if instrument == domains.UAVSAR:
        m = math.log10(m)
    mid = int((mode - start) / width)
    cumulative = sum(values[:mid]) + values[mid] / 2
    scale = cumulative / __cdf(params, m, params[0])
    i = mid
    while i < len(values):
        cumulative += values[i]
        diff = cumulative - scale * __cdf(params, m, start + i * width)
        if diff > 0.01:
            break
        i += 1
    threshold = start + i * width

    return (threshold, params)

def threshold(domain):
    hist_image = domain.image
    if domain.instrument == domains.UAVSAR:
        hist_image = hist_image.log10()
    if __HISTOGRAM_CLAMP_MAX[domain.instrument]:
        hist_image = hist_image.clamp(0, __HISTOGRAM_CLAMP_MAX[domain.instrument])
    histogram = hist_image.reduceRegion(ee.Reducer.histogram(__NUM_BUCKETS[domain.instrument], None, None), domain.bounds, 30, None, None, True).getInfo()
    
    thresholds = {}
    results = []
    #plt.figure(1)
    for c in range(len(domain.channels)):
        ch = domain.channels[c]
        # ignore first bucket, too many... for UAVSAR in particular
        histogram[ch]['bucketMin'] += histogram[ch]['bucketWidth']
        histogram[ch]['histogram'] =  histogram[ch]['histogram'][1:]
        total = sum(histogram[ch]['histogram'])
        histogram[ch]['histogram'] = map(lambda x : x / total, histogram[ch]['histogram'])
        (threshold, params) = __find_threshold_histogram(histogram, domain.instrument, ch)
        if domain.instrument == domains.UAVSAR:
            threshold = 10 ** threshold
        thresholds[ch] = threshold

        channel_result = domain.image.select([ch], [ch]).lte(threshold)
        results.append(channel_result)
        
        #plt.subplot(100 * len(domain.channels) + 10 + c + 1)
        #__show_histogram(histogram[domain.channels[c]], domain.instrument, params)

    result_image = results[0]
    for c in range(1, len(results)):
        result_image = result_image.addBands(results[c], [domain.channels[c]])
    addToMap(result_image, {'min': 0, 'max': 1}, 'Color Image', True)
    
    result_image = results[0].select([domain.channels[0]], ['b1'])
    for c in range(1, len(results)):
        result_image = result_image.And(results[c])
    #plt.show()
    return result_image



#------------------------------------------------------------------------
# - sar_martinis radar algorithm (find threshold by histogram splits on selected subregions)

# Algorithm from paper:
#    Towards operational near real-time flood detection using a split-based
#    automatic thresholding procedure on high resolution TerraSAR-X data

def getBoundingBox(bounds):
    '''Returns (minLon, minLat, maxLon, maxLat) from domain bounds'''
    
    coordList = bounds['coordinates'][0]
    minLat = 999
    minLon = 999999
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
    meanLon = 0
    for c in coordList:
        meanLat = meanLat + c[1]
        meanLon = meanLon + c[0]
    meanLat = meanLat / len(coordList)
    meanLon = meanLon / len(coordList)
    return (meanLat, meanLon)


#
#def __show_histogram2(histogram, binCenters):
#    '''Create a plot of a histogram'''
#    plt.bar(binCenters, histogram)
#
#    plt.show()




def __computeJT(relHist, binVals, T):
    '''As part of the Kittler/Illingworth method, compute J(T) for a given T'''

    FAIL_VAL = 999999

    # Split the hostogram at the threshold T.
    histogram1 = relHist[0:T]
    histogram2 = relHist[T+1:]

    # Compute the number of pixels in the two classes.
    P1 = sum(histogram1)
    P2 = sum(histogram2)

    #print P1
    #print P2

    # Only continue if both classes contain at least one pixel.
    if (P1 <= 0) or (P2 <= 0):
        return FAIL_VAL

    # Compute the standard deviations of the classes.
    weightedBins1 = numpy.multiply(histogram1, binVals[0:T])
    weightedBins2 = numpy.multiply(histogram2, binVals[T+1:])
    mean1         = sum(weightedBins1) / P1;
    mean2         = sum(weightedBins2) / P2;
    diffs1        = numpy.subtract(binVals[0:T],  mean1)
    diffs2        = numpy.subtract(binVals[T+1:], mean2)
    diffsSq1      = [d*d for d in diffs1]
    diffsSq2      = [d*d for d in diffs2]
    weightedBins1 = numpy.multiply(histogram1, diffsSq1)
    weightedBins2 = numpy.multiply(histogram2, diffsSq2)
    sigma1        = math.sqrt(sum(weightedBins1) / P1)
    sigma2        = math.sqrt(sum(weightedBins2) / P2)

    #print mean1
    #print diffs1
    #print diffsSq1
    #print weightedBins1
    #print sigma1

    # Make sure both classes contain at least two intensity values.
    if (sigma1 <= 0) or (sigma2 <= 0):
        return FAIL_VAL

    # Compute J(T).
    J = 1 + 2*(P1*math.log(sigma1) + P2*math.log(sigma2)) - 2*(P1*math.log(P1) + P2*math.log(P2))
    
    #print str(mean1) + ' -> ' + str(mean2) + ' -> ' + str(sigma1) + ' -> ' + str(sigma2)
    #print str(T) + ' -> ' + str(binVals[T]) + ' -> ' + str(J)
    
    return J

def splitHistogramKittlerIllingworth(histogram, binVals):
    '''Tries to compute an optimal histogram threshold using the Kittler/Illingworth method'''

    # Normalize the histogram (each bin is now a percentage)
    histSum = float(sum(histogram))
    relativeHist = numpy.divide(histogram,histSum)

    #print histogram
    #print binVals
    #print relativeHist

    # Try out every bin value in the histogram and pick the best one
    #  - For more resolution, use more bins!
    #  - Could write up a smarter solver for this.
    numBins = len(binVals)
    J = []
    for i in range(0, numBins):
        J.append(__computeJT(relativeHist, binVals, i))
        
    minJ      = J[1]
    threshold = binVals[0]
    for i in range(1, numBins):
        if J[i] < minJ:
            minJ      = J[i]
            threshold = (binVals[i] + binVals[i-1])/2 # Threshold is below current bin value
    
    return threshold

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
    
    #//var minLon = ee.List(bounds.coordinates().get(0)).get(0).getInfo()[0]
    #//var minLat = ee.List(bounds.coordinates().get(0)).get(0).getInfo()[1]
    #//var maxLon = ee.List(bounds.coordinates().get(0)).get(2).getInfo()[0]
    #//var maxLat = ee.List(bounds.coordinates().get(0)).get(2).getInfo()[1]
    #print domain.bounds
    
    # Divide up the region into a grid of subregions
    BOX_COUNT = 10 # TODO: What unit is this?
    boxList  = divideUpBounds(domain.bounds, BOX_COUNT)
    
    # Extract the center point from each box
    centersList = map(getBoundsCenter, boxList)
    #print centersList
    
    KERNEL_SIZE = 20 # <-- TODO!!! The kernel needs to be the size of a box
    avgKernel = ee.Kernel.square(KERNEL_SIZE, 'pixels', True);
    
    # Select the radar layer we want to work in
    channelName = 'hv'
    grayLayer = radarImage.select([channelName])
    addToMap(grayLayer, {'min': 3000, 'max': 70000,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'grayLayer',   False)
    
    
    
    # Compute the global mean, then make a constant image out of it.
    globalMean = grayLayer.reduceRegion(ee.Reducer.mean(), domain.bounds, 30)
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
    addToMap(meanImage, {'min': 3000, 'max': 70000,  'opacity': 1.0, 'palette': GRAY_PALETTE}, 'Mean',   False)
    addToMap(stdImage,  {'min': 3000, 'max': 200000, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'StdDev', False)
    
    # Compute these two statistics across the entire image
    CV = meanImage.divide(stdImage)
    R  = meanImage.divide(globalMeanImage)
    
    # Debug plots
    addToMap(CV, {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'CV', False)
    addToMap(R,  {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'R',  False)
    
    
    # 2: Prune to a reduced set of tiles X'
    
    # Parameters which control which sub-regions will have their histograms analyzed
    # - These are strongly influenced by the smoothing kernel size!!!
    MIN_CV = 0.7
    MAX_R  = 0.9
    MIN_R  = 0.4
    
    # Filter out pixels based on computed statistics
    X_prime = CV.gte(MIN_CV).And(R.gte(MIN_R)).And(R.lte(MAX_R))
    addToMap(X_prime.mask(X_prime),  {'min': 0, 'max': 1, 'opacity': 1.0, 'palette': GRAY_PALETTE}, 'X_prime',  False)
   
    
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
        histogram, binEdges = numpy.histogram(pixelVals, NUM_BINS)
        binCenters = numpy.divide(numpy.add(binEdges[:NUM_BINS], binEdges[1:]), 2.0)
        
        # Compute a split on the histogram
        splitVal = splitHistogramKittlerIllingworth(histogram, binCenters)
        print "Thresh = " + str(splitVal)
        localThresholdList.append(splitVal)
        usedPointList.append(thisLoc)
        
        #plt.bar(binCenters, histogram)
        #plt.show()
        
        #break
        
        #TODO: Make histogram out of this data
        #
        #
        #TODO: Extract the threshold from this histogram!
        #        Can we use this function?  __find_threshold_histogram(histogram, instrument, channel)
        #
        #localThresholdList.append(thisThreshold)
    
    #print localThresholdList
    
    numUsedPoints   = len(usedPointList)
    numUnusedPoints = len(rejectedPointList)
    #print numUsedPoints
    #print numUnusedPoints
    if (numUsedPoints > 0):
        usedPointListEE = ee.FeatureCollection(ee.Feature(usedPointList[0]))
        for i in range(1,numUsedPoints):
            temp = ee.FeatureCollection(ee.Feature(usedPointList[i]))
            usedPointListEE = usedPointListEE.merge(temp)       

        addToMap(usedPointListEE, {'opacity': 1.0, 'palette': GREEN_PALETTE}, 'Used PTs', False)
    if (numUnusedPoints > 0):
        unusedPointListEE = ee.FeatureCollection(ee.Feature(rejectedPointList[0]))
        for i in range(1,numUnusedPoints):
            temp = ee.FeatureCollection(ee.Feature(rejectedPointList[i]))
            unusedPointListEE = unusedPointListEE.merge(temp)       

        addToMap(unusedPointListEE, {'opacity': 1.0, 'palette': RED_PALETTE}, 'Unused PTs', False)
    
    #return X_doublePrime
    
    # 5: Use the individual thresholds to compute a global threshold 
    
    computedThreshold = numpy.mean(localThresholdList) # Nothing fancy going on here!
    
    print 'Computed threshold = ' + str(computedThreshold)
    
    finalWaterClass = grayLayer.lte(computedThreshold)

    print 'FINISHED WITH CALCULATION'
    
    return finalWaterClass
    
    




#------------------------------------------------------------------------



# For each algorithm specify the name, function, and color.
__ALGORITHMS = {
	MATGEN : ('Matgen Threshold', matgen.threshold, '00FFFF'),
	RANDOM_FORESTS : ('Random Forests', learning.random_forests, 'FFFF00'),
	DECISION_TREE  : ('Decision Tree', learning.decision_tree, 'FF00FF'),
	SVM : ('SVM', learning.svm, '00AAFF')
    MARTINIS  : ('Martinis',  sar_martinis, 'FF00FF')
}

# These functions just redirect the call to the correct algorithm

def detect_flood(image, algorithm):
    try:
        approach = __ALGORITHMS[algorithm]
    except:
        return None
    return approach[1](image)

def get_algorithm_name(algorithm):
    try:
        return __ALGORITHMS[algorithm][0]
    except:
        return None

def get_algorithm_color(algorithm):
    try:
        return __ALGORITHMS[algorithm][2]
    except:
        return None

