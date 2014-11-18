import ee
from domains import *

from util.mapclient_qt import addToMap

EVI                = 1
XIAO               = 2
DIFFERENCE         = 3
CART               = 4
SVM                = 5
RANDOM_FORESTS     = 6
DNNS               = 7
DNNS_DEM           = 8
DIFFERENCE_HISTORY = 9
DARTMOUTH          = 10
DNNS_REVISED       = 11

def __compute_indices(domain):
    '''Compute several common interpretations of the MODIS bands'''
    
    band1 = domain.high_res_modis.select(['sur_refl_b01']) # pRED
    band2 = domain.high_res_modis.select(['sur_refl_b02']) # pNIR

    # Other bands must be used at lower resolution
    band3 = domain.low_res_modis.select(['sur_refl_b03']) # pBLUE
    band6 = domain.low_res_modis.select(['sur_refl_b06']) # pSWIR

    NDVI = (band2.subtract(band1)).divide(band2.add(band1));
    # Normalized difference water index
    NDWI = (band1.subtract(band6)).divide(band1.add(band6));
    # Enhanced vegetation index
    EVI = band2.subtract(band1).multiply(2.5).divide( band2.add(band1.multiply(6)).subtract(band3.multiply(7.5)).add(1));
    # Land surface water index
    LSWI = (band2.subtract(band6)).divide(band2.add(band6));

    return {'b1': band1, 'b2': band2, 'b3': band3, 'b6': band6,
            'NDVI': NDVI, 'NDWI': NDWI, 'EVI': EVI, 'LSWI': LSWI}

def evi(domain, b):
    no_clouds = b['b3'].lte(2100).select(['sur_refl_b03'], ['b1'])
    criteria1 = b['EVI'].lte(0.3).And(b['LSWI'].subtract(b['EVI']).gte(0.05)).select(['sur_refl_b02'], ['b1'])
    criteria2 = b['EVI'].lte(0.05).And(b['LSWI'].lte(0.0)).select(['sur_refl_b02'], ['b1'])
    return no_clouds.And(criteria1.Or(criteria2))

def xiao(domain, b):
    return b['LSWI'].subtract(b['NDVI']).gte(0.05).Or(b['LSWI'].subtract(b['EVI']).gte(0.05)).select(['sur_refl_b02'], ['b1']);

MODIS_DIFF_THRESHOLDS = {
        BORDER         : 1200,
        BORDER_JUNE    : 1550,
        ARKANSAS_CITY  : 1200,
        KASHMORE       : 350,
        KASHMORE_NORTH : 350,
        NEW_ORLEANS    : 1200,
        SLIDELL        : 1200,
        BAY_AREA       : 650,
        BERKELEY       : 650,
        NIGER          : 1200}

def modis_diff(domain, b, threshold=None):
    '''Compute (b2-b1) < threshold'''
    if threshold == None: # If no threshold value passed in, load it based on the data set.
        threshold = MODIS_DIFF_THRESHOLDS[domain.id]
    return b['b2'].subtract(b['b1']).lte(threshold).select(['sur_refl_b02'], ['b1']) # Rename sur_refl_b02 to b1

def __create_learning_image(domain, b):
    diff  = b['b2'].subtract(b['b1'])
    ratio = b['b2'].divide(b['b1'])
    return b['b1'].addBands(b['b2']).addBands(diff).addBands(ratio).addBands(b['NDVI']).addBands(b['NDWI'])

def earth_engine_classifier(domain, b, classifier_name):
    training_domain = retrieve_domain(TRAINING_DOMAINS[domain.id])
    training_image = __create_learning_image(training_domain, __compute_indices(training_domain))
    classifier = ee.apply("TrainClassifier", {
            'image': training_image,
            'subsampling' : 0.5,
            'training_image' : training_domain.ground_truth,
            'training_band': "b1",
            'training_region' : training_domain.bounds,
            'max_classification': 2,
            'classifier_name': classifier_name
        })
    classified = ee.call("ClassifyImage", __create_learning_image(domain, b), classifier).select(['classification'], ['b1']); 
    return classified;

def cart(domain, b):
    return earth_engine_classifier(domain, b, 'Cart')

def svm(domain, b):
    return earth_engine_classifier(domain, b, 'Pegasos')

def random_forests(domain, b):
    return earth_engine_classifier(domain, b, 'RifleSerialClassifier')

def dnns(domain, b):
    '''Dynamic Nearest Neighbor Search'''
    
    # This algorithm does not actually work that well.  The algorithm really needs a very large search region for
    #  each pixel but this slows Earth Engine to an unstable crawl.  With a small search radius this algorithm
    #  performs very similarly to a simple threshold algorithm, but with many more parameters.
    
    # Parameters
    KERNEL_SIZE = 10 # The original paper used a 100x100 pixel box = 25,000 meters!
    PURELAND_THRESHOLD = 0.5
    
    # Set up two square kernels of the same size
    # - These kernels define the search range for nearby pure water and land pixels
    kernel            = ee.Kernel.square(KERNEL_SIZE, 'pixels', False)
    kernel_normalized = ee.Kernel.square(KERNEL_SIZE, 'pixels', True)
    
    # Compute b1/b6 and b2/b6
    composite_image = b['b1'].addBands(b['b2']).addBands(b['b6'])
    #ratio1 = b['b1'].divide(b['b6'])
    #ratio2 = b['b2'].divide(b['b6'])
    
    # Compute (b2 - b1) < threshold, a simple water detection algorithm.  Treat the result as "pure water" pixels.
    PURE_WATER_THRESHOLD_RATIO = 1.0
    pureWaterThreshold = MODIS_DIFF_THRESHOLDS[domain.id] * PURE_WATER_THRESHOLD_RATIO
    purewater = modis_diff(domain, b, pureWaterThreshold)
    
    # Compute the mean value of pure water pixels across the entire region, then store in a constant value image.
    WATER_AVERAGE_SCALE_METERS = 30 # This value seems to have no effect on the results
    averagewater      = purewater.mask(purewater).multiply(composite_image).reduceRegion(ee.Reducer.mean(), domain.bounds, WATER_AVERAGE_SCALE_METERS)
    averagewaterimage = ee.Image([averagewater.getInfo()['sur_refl_b01'], averagewater.getInfo()['sur_refl_b02'], averagewater.getInfo()['sur_refl_b06']])
    
    # For each pixel, compute the number of nearby pure water pixels
    purewatercount = purewater.convolve(kernel)
    # Get mean of nearby pure water (b1,b2,b6) values for each pixel with enough pure water nearby
    MIN_PUREWATER_NEARBY = 100
    purewaterref = purewater.multiply(composite_image).convolve(kernel).multiply(purewatercount.gte(MIN_PUREWATER_NEARBY)).divide(purewatercount)
    # For pixels that did not have enough pure water nearby, just use the global average water value
    purewaterref = purewaterref.add(averagewaterimage.multiply(purewaterref.Not()))
    # Computed an intermediate fraction = min(b1/b6, b2/b6), sort of a water measure.
    fraction = purewaterref.select('sur_refl_b01').divide(b['b6']).min(purewaterref.select('sur_refl_b02').divide(b['b6']))
    # fraction = fraction.add(purewater.multiply(ee.Image(1.0).subtract(fraction))); // purewater fraction is always 1
       
    pureland      = fraction.lte(PURELAND_THRESHOLD) # Classify pixels as pure land
    purelandcount = pureland.convolve(kernel)        # Get nearby pure land count for each pixel
    average       = b['b6'].convolve(kernel_normalized)  # Get nearby mean value of b6
    averageland   = pureland.multiply(b['b6']).convolve(kernel).divide(purelandcount) # Get mean nearby LAND value of b6
    averageland   = averageland.add(average.multiply(averageland.Not())) # For pixels that did not have any pure land nearby, use mean b6
    
    # Compute the water fraction: (land[b6] - b6) / (land[b6] - water[b6])
    # - Ultimately, relying solely on band 6 for the final classification may not be a good idea!
    water_fraction = (averageland.subtract(b['b6'])).divide(averageland.subtract(purewaterref.select('sur_refl_b06'))).clamp(0, 1)
       
    # Set pure water to 1, pure land to 0
    water_fraction = water_fraction.subtract(pureland.multiply(water_fraction))
    water_fraction = water_fraction.add(purewater.multiply(ee.Image(1.0).subtract(water_fraction)))
    
    #addToMap(fraction,       {'min': 0, 'max': 1},   'fraction', False)
    #addToMap(purewater,      {'min': 0, 'max': 1},   'pure water', False)
    #addToMap(pureland,       {'min': 0, 'max': 1},   'pure land', False)
    #addToMap(purewatercount, {'min': 0, 'max': 300}, 'pure water count', False)
    #addToMap(purelandcount,  {'min': 0, 'max': 300}, 'pure land count', False)
    #addToMap(water_fraction, {'min': 0, 'max': 5},   'water_fractionDNNS', False)
    #addToMap(purewaterref,   {'min': 0, 'max': 3000, 'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06']}, 'purewaterref', False)
    #addToMap(averageland,    {'min': 0, 'max': 3000, 'bands': ['sur_refl_b01']}, 'averageland', False)
    
    return water_fraction.select(['sur_refl_b01'], ['b1']) # Rename sur_refl_b02 to b1

def dnns_dem(domain, b):
    '''Enhance the DNNS result with high resolution DEM information'''
    
    MODIS_PIXEL_SIZE_METERS = 250
    
    # Call the DNNS function to get the starting point
    water_fraction = dnns(domain, b)

   
    ## Treating the DEM values contained in the MODIS pixel as a histogram, find the N'th percentile
    ##  where N is the water fraction computed by DNNS.  That should be the height of the flood water.
    #modisPixelKernel = ee.Kernel.square(MODIS_PIXEL_SIZE_METERS, 'meters', False)
    #domain.dem.mask(water_fraction).reduceNeighborhood(ee.Reducer.percentile(), modisPixelKernel)
    # --> We would like to compute a percentile here, but this would require a different reducer input for each pixel!
    

    # Get min and max DEM height within each water containing pixel
    # - If a DEM pixel contains any water then the water level must be at least that high.
    dem_min = domain.dem.mask(water_fraction).focal_min(MODIS_PIXEL_SIZE_METERS, 'square', 'meters')
    dem_max = domain.dem.mask(water_fraction).focal_max(MODIS_PIXEL_SIZE_METERS, 'square', 'meters')
    
    # Approximation, linearize each tile's fraction point
    # - The water percentage is used as a percentage between the two elevations
    water_high = dem_min.add(dem_max.subtract(dem_min).multiply(water_fraction))
    water_high = water_high.multiply(water_fraction.eq(1.0)) # Don't include full pixels, they don't give us clues to their height.   
    
    # Problem: Averaging process spreads water way out to pixels where it was not detected!!
    #          Reducing the averaging is a simple way to deal with this and probably does not hurt results at all
    
    #dilate_kernel = ee.Kernel.circle(250, 'meters', False)
    #allowed_water_mask = water_fraction.gt(0.0)#.convolve(dilate_kernel)
    
    # Smooth out the water elevations with a broad kernel; nearby pixels probably have the same elevation!
    water_dem_kernel = ee.Kernel.circle(500, 'meters', False)
    num_nearby_water_pixels = water_high.gt(0.0).convolve(water_dem_kernel)
    average_high = water_high.convolve(water_dem_kernel).divide(num_nearby_water_pixels)
    #addToMap(water_fraction, {}, 'Water Fraction', false);
    #addToMap(average_high, {min:25, max:40}, 'Water Level', false);
    #addToMap(dem.subtract(average_high), {min : -0, max : 10}, 'Water Difference', false);
    #addToMap(dem.lte(average_high).and(domain.groundTruth.not()));
    
    #addToMap(allowed_water_mask, {'min': 0, 'max': 1}, 'allowed_water', False);
    #addToMap(water_high, {'min': 0, 'max': 100}, 'water_high', False);
    #addToMap(average_high, {'min': 0, 'max': 100}, 'average_high', False);
    #addToMap(domain.dem, {'min': 0, 'max': 100}, 'DEM', False);
    
    # Classify DEM pixels as flooded based on being under the local water elevation or being completely flooded.
    #return domain.dem.lte(average_high).Or(water_fraction.eq(1.0)).select(['elevation'], ['b1'])
    dem_water = domain.dem.lte(average_high).mask(water_fraction) # Mask prevents pixels with 0% water from being labeled as water
    return dem_water.Or(water_fraction.eq(1.0)).select(['elevation'], ['b1'])

HISTORY_THRESHOLDS = {
        BORDER         : (3.5,     -3.5),
        BORDER_JUNE    : (6.5,     -3.5),
        ARKANSAS_CITY  : (6.5,     -3.5),
        KASHMORE       : (4.5,     -3.0),
        KASHMORE_NORTH : (4.5,     -3.0),
        NEW_ORLEANS    : (4.5,     -3.0),
        SLIDELL        : (4.5,     -3.0),
        BAY_AREA       : (3.5,     -2.0),
        BERKELEY       : (3.5,     -2.0)
    }

def history_diff(domain, b):
    '''Leverage historical data and the permanent water mask to improve the threshold method'''
    
    # Load pre-selected constants for this domain
    (dev_thresh, change_thresh) = HISTORY_THRESHOLDS[domain.id]
    
    # Retrieve all the MODIS images for the region in the last several years
    NUM_YEARS_BACK         = 5
    NUM_DAYS_COMPARE_RANGE = 40.0 # Compare this many days before/after the target day in previous years
    YEAR_RANGE_PERCENTAGE  = NUM_DAYS_COMPARE_RANGE / 365.0
    #print 'YEAR_RANGE_PERCENTAGE = ' + str(YEAR_RANGE_PERCENTAGE)
    #print 'Start: ' + str(domain.date.advance(-1 - YEAR_RANGE_PERCENTAGE, 'year'))
    #print 'End:   ' + str(domain.date.advance(-1 + YEAR_RANGE_PERCENTAGE, 'year'))
    
    history = ee.ImageCollection('MOD09GQ').filterDate(domain.date.advance(-1 - YEAR_RANGE_PERCENTAGE, 'year'), domain.date.advance(-1 + YEAR_RANGE_PERCENTAGE, 'year')).filterBounds(domain.bounds);
    #print history.getInfo()
    for i in range(1, NUM_YEARS_BACK-1):
        yearMin = -(i+1) - YEAR_RANGE_PERCENTAGE
        yearMax = -(i+1) + YEAR_RANGE_PERCENTAGE
        history.merge(ee.ImageCollection('MOD09GQ').filterDate(domain.date.advance(yearMin, 'year'), domain.date.advance(yearMax, 'year')).filterBounds(domain.bounds));
    
    # Simple function implements the b2 - b1 difference method
    flood_diff_function = (lambda x : x.select(['sur_refl_b02']).subtract(x.select(['sur_refl_b01'])))
    
    # Apply difference function to all images in history, then compute mean and standard deviation of difference scores.
    historyDiff   = history.map(flood_diff_function)
    historyMean   = historyDiff.mean()
    historyStdDev = historyDiff.reduce(ee.Reducer.stdDev())
    
    #addToMap(historyMean,   {'min' : 0, 'max' : 4000}, 'History mean',   False)
    #addToMap(historyStdDev, {'min' : 0, 'max' : 2000}, 'History stdDev', False)
    
    # Compute flood diff on current image and compare to historical mean/STD.
    floodDiff   = flood_diff_function(domain.high_res_modis)
    diffOfDiffs = floodDiff.subtract(historyMean)
    ddDivDev    = diffOfDiffs.divide(historyStdDev)
    changeFlood = ddDivDev.lt(change_thresh)  # Mark all pixels which are enough STD's away from the mean.
    
    #addToMap(floodDiff,   {'min' : 0, 'max' : 4000}, 'floodDiff',   False)
    #addToMap(diffOfDiffs, {'min' : -2000, 'max' : 2000}, 'diffOfDiffs', False)
    #addToMap(ddDivDev,    {'min' : -10,   'max' : 10}, 'ddDivDev',    False)
    #addToMap(changeFlood,    {'min' : 0,   'max' : 1}, 'changeFlood',    False)
    #addToMap(domain.water_mask,    {'min' : 0,   'max' : 1}, 'Permanent water mask',    False)
    
    
    #plainDiff       = highResImage.subtract(historyMean) # ??
    
    # TODO: This isn't working at all!
    
    # Compute the difference statistics inside permanent water mask pixels
    MODIS_RESOLUTION = 250 # Meters
    diffInWaterMask  = floodDiff.multiply(domain.water_mask)
    maskedMean       = diffInWaterMask.reduceRegion(ee.Reducer.mean(),   domain.bounds, MODIS_RESOLUTION)
    maskedStdDev     = diffInWaterMask.reduceRegion(ee.Reducer.stdDev(), domain.bounds, MODIS_RESOLUTION)
    
    #print 'Water mean = ' + str(maskedMean.getInfo())
    #print 'Water STD  = ' + str(maskedStdDev.getInfo())
    
    # Use the water mask statistics to compute a difference threshold, then find all pixels below the threshold.
    waterThreshold  = maskedMean.getInfo()['sur_refl_b02'] + dev_thresh*(maskedStdDev.getInfo()['sur_refl_b02']);
    waterPixels     = modis_diff(domain, b, waterThreshold)
    
    #addToMap(waterPixels,    {'min' : 0,   'max' : 1}, 'waterPixels',    False)
    
    # Combine water pixels from the historical and water mask methods.
    return (waterPixels.Or(changeFlood));


DARTMOUTH_THRESHOLDS = {
        BORDER         : 0.75,
        BORDER_JUNE    : 0.75,
        ARKANSAS_CITY  : 0.75,
        KASHMORE       : 0.65,
        KASHMORE_NORTH : 0.65,
        NEW_ORLEANS    : 0.75,
        SLIDELL        : 0.75,
        BAY_AREA       : 0.55,
        BERKELEY       : 0.55
    }

def dartmouth(domain, b):
    A = 500
    B = 2500
    return b['b2'].add(A).divide(b['b1'].add(B)).lte(DARTMOUTH_THRESHOLDS[domain.id]).select(['sur_refl_b02'], ['b1'])



def dnns_revised(domain, b):
    '''Dynamic Nearest Neighbor Search with revisions to improve performance on our test data'''
    
    # One issue with this algorithm is that its large search range slows down even Earth Engine!
    # - With a tiny kernel size, everything is relative to the region average which seems to work just as well.
    
    # Parameters
    KERNEL_SIZE = 1 # The original paper used a 100x100 pixel box = 25,000 meters!
    PURELAND_THRESHOLD = 3500 # TODO: Vary by domain?
    PURE_WATER_THRESHOLD_RATIO = 0.2
    
    # Set up two square kernels of the same size
    # - These kernels define the search range for nearby pure water and land pixels
    kernel            = ee.Kernel.square(KERNEL_SIZE, 'meters', False)
    kernel_normalized = ee.Kernel.square(KERNEL_SIZE, 'meters', True)
    
    composite_image = b['b1'].addBands(b['b2']).addBands(b['b6'])
    
    # Compute (b2 - b1) < threshold, a simple water detection algorithm.  Treat the result as "pure water" pixels.
    pureWaterThreshold = MODIS_DIFF_THRESHOLDS[domain.id] * PURE_WATER_THRESHOLD_RATIO
    pureWater = modis_diff(domain, b, pureWaterThreshold)
    
    # Compute the mean value of pure water pixels across the entire region, then store in a constant value image.
    AVERAGE_SCALE_METERS = 30 # This value seems to have no effect on the results
    averageWater      = pureWater.mask(pureWater).multiply(composite_image).reduceRegion(ee.Reducer.mean(), domain.bounds, AVERAGE_SCALE_METERS)
    averageWaterImage = ee.Image([averageWater.getInfo()['sur_refl_b01'], averageWater.getInfo()['sur_refl_b02'], averageWater.getInfo()['sur_refl_b06']])
    
    # For each pixel, compute the number of nearby pure water pixels
    pureWaterCount = pureWater.convolve(kernel)
    # Get mean of nearby pure water (b1,b2,b6) values for each pixel with enough pure water nearby
    MIN_PURE_NEARBY = 10
    averageWaterLocal = pureWater.multiply(composite_image).convolve(kernel).multiply(pureWaterCount.gte(MIN_PURE_NEARBY)).divide(pureWaterCount)
    # For pixels that did not have enough pure water nearby, just use the global average water value
    averageWaterLocal = averageWaterLocal.add(averageWaterImage.multiply(averageWaterLocal.Not()))
    
    # Use simple diff method to select pure land pixels
    #LAND_THRESHOLD   = 2000 # TODO: Move to domain selector
    pureLand         = b['b2'].subtract(b['b1']).gte(PURELAND_THRESHOLD).select(['sur_refl_b02'], ['b1']) # Rename sur_refl_b02 to b1
    averageLand      = pureLand.mask(pureLand).multiply(composite_image).reduceRegion(ee.Reducer.mean(), domain.bounds, AVERAGE_SCALE_METERS)
    averageLandImage = ee.Image([averageLand.getInfo()['sur_refl_b01'], averageLand.getInfo()['sur_refl_b02'], averageLand.getInfo()['sur_refl_b06']])
    pureLandCount    = pureLand.convolve(kernel)        # Get nearby pure land count for each pixel
    averageLandLocal = pureLand.multiply(composite_image).convolve(kernel).multiply(pureLandCount.gte(MIN_PURE_NEARBY)).divide(pureLandCount)
    averageLandLocal = averageLandLocal.add(averageLandImage.multiply(averageLandLocal.Not())) # For pixels that did not have any pure land nearby, use mean
    
    # Compute the water fraction: (land - b) / (land - water)
    landDiff  = averageLandLocal.subtract(composite_image)
    waterDiff = averageWaterLocal.subtract(composite_image)
    typeDiff  = averageLandLocal.subtract(averageWaterLocal)
    #water_vector   = (averageLandLocal.subtract(b)).divide(averageLandLocal.subtract(averageWaterLocal))
    landDist  = landDiff.expression("b('sur_refl_b01')*b('sur_refl_b01') + b('sur_refl_b02') *b('sur_refl_b02') + b('sur_refl_b06')*b('sur_refl_b06')").sqrt();
    waterDist = waterDiff.expression("b('sur_refl_b01')*b('sur_refl_b01') + b('sur_refl_b02') *b('sur_refl_b02') + b('sur_refl_b06')*b('sur_refl_b06')").sqrt();
    typeDist  = typeDiff.expression("b('sur_refl_b01')*b('sur_refl_b01') + b('sur_refl_b02') *b('sur_refl_b02') + b('sur_refl_b06')*b('sur_refl_b06')").sqrt();
       
    #waterOff = landDist.divide(waterDist.add(landDist)) 
    waterOff = landDist.divide(typeDist) # TODO: Improve this math, maybe full matrix treatment?

    # Set pure water to 1, pure land to 0
    waterOff = waterOff.subtract(pureLand.multiply(waterOff))
    waterOff = waterOff.add(pureWater.multiply(ee.Image(1.0).subtract(waterOff)))
    
    # TODO: Better way of filtering out low fraction pixels.
    waterOff = waterOff.multiply(waterOff)
    waterOff = waterOff.gt(0.6)
    
    #addToMap(fraction,       {'min': 0, 'max': 1},   'fraction', False)
    addToMap(pureWater,      {'min': 0, 'max': 1},   'pure water', False)
    addToMap(pureLand,       {'min': 0, 'max': 1},   'pure land', False)
    addToMap(pureWaterCount, {'min': 0, 'max': 100}, 'pure water count', False)
    addToMap(pureLandCount,  {'min': 0, 'max': 100}, 'pure land count', False)
    addToMap(averageWaterImage,  {'min': 0, 'max': 3000, 'bands': ['constant', 'constant_1', 'constant_2']}, 'average water', False)
    addToMap(averageLandImage,   {'min': 0, 'max': 3000, 'bands': ['constant', 'constant_1', 'constant_2']}, 'average land',  False)
    addToMap(averageWaterLocal,  {'min': 0, 'max': 3000, 'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06']}, 'local water ref', False)
    addToMap(averageLandLocal,   {'min': 0, 'max': 3000, 'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06']}, 'local land ref',  False)
    
    
    return waterOff.select(['sur_refl_b01'], ['b1']) # Rename sur_refl_b02 to b1


# End of algorithm definitions
#=======================================================================================================
#=======================================================================================================






__ALGORITHMS = {
        # Algorithm,    Display name,   Function name,    Display by default,    Display color
        EVI                : ('EVI',                     evi,            False, 'FF00FF'),
        XIAO               : ('XIAO',                    xiao,           False, 'FFFF00'),
        DIFFERENCE         : ('Difference',              modis_diff,     False, '00FFFF'),
        CART               : ('CART',                    cart,           False, 'CC6600'),
        SVM                : ('SVM',                     svm,            False, 'FFAA33'),
        RANDOM_FORESTS     : ('Random Forests',          random_forests, False, 'CC33FF'),
        DNNS               : ('DNNS',                    dnns,           False, '0000FF'),
        DNNS_REVISED       : ('DNNS Revised',            dnns_revised,   False, '00FF00'),
        DNNS_DEM           : ('DNNS with DEM',           dnns_dem,       False, '9900FF'),
        DIFFERENCE_HISTORY : ('Difference with History', history_diff,   False, '0099FF'),
        DARTMOUTH          : ('Dartmouth',               dartmouth,      False, '33CCFF')
}

def detect_flood(domain, algorithm):
    try:
        approach = __ALGORITHMS[algorithm]
    except:
        return None
    return (approach[0], approach[1](domain, __compute_indices(domain)))

def get_algorithm_name(algorithm):
    try:
        return __ALGORITHMS[algorithm][0]
    except:
        return None

def get_algorithm_color(algorithm):
    try:
        return __ALGORITHMS[algorithm][3]
    except:
        return None

def is_algorithm_fractional(algorithm):
    try:
        return __ALGORITHMS[algorithm][2]
    except:
        return None

