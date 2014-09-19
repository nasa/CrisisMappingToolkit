import ee
from domains import *

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

def __compute_indices(domain):
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
    if threshold == None:
        threshold = MODIS_DIFF_THRESHOLDS[domain.id]
    return b['b2'].subtract(b['b1']).lte(threshold).select(['sur_refl_b02'], ['b1'])

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
    KERNEL_SIZE = 10
    PURELAND_THRESHOLD = 0.5
    kernel = ee.Kernel.square(KERNEL_SIZE, 'pixels', False)
    kernel_normalized = ee.Kernel.square(KERNEL_SIZE, 'pixels', True)
    composite_image = b['b1'].addBands(b['b2']).addBands(b['b6'])
    ratio1 = b['b1'].divide(b['b6'])
    ratio2 = b['b2'].divide(b['b6'])
    purewater = modis_diff(domain, b)
    averagewater = purewater.mask(purewater).multiply(composite_image).reduceRegion(ee.Reducer.mean(), domain.bounds, 30)
    averagewaterimage = ee.Image([averagewater.getInfo()['sur_refl_b01'], averagewater.getInfo()['sur_refl_b02'], averagewater.getInfo()['sur_refl_b06']])
    
    # pure water channel computation
    purewatercount = purewater.convolve(kernel)
    purewaterref = purewater.multiply(composite_image).convolve(kernel).multiply(purewatercount.gte(100)).divide(purewatercount)
    purewaterref = purewaterref.add(averagewaterimage.multiply(purewaterref.Not()))
    fraction = purewaterref.select('sur_refl_b01').divide(b['b6']).min(purewaterref.select('sur_refl_b02').divide(b['b6']))
    # fraction = fraction.add(purewater.multiply(ee.Image(1.0).subtract(fraction))); // purewater fraction is always 1
    
    pureland = fraction.lte(PURELAND_THRESHOLD)
    purelandcount = pureland.convolve(kernel)
    averageland = pureland.multiply(b['b6']).convolve(kernel).divide(purelandcount)
    average = b['b6'].convolve(kernel_normalized)
    averageland = averageland.add(average.multiply(averageland.Not()))
    water_fraction = (averageland.subtract(b['b6'])).divide(averageland.subtract(purewaterref.select('sur_refl_b06'))).clamp(0, 1)
    # set pure water to 1, pure land to 0
    water_fraction = water_fraction.subtract(pureland.multiply(water_fraction))
    water_fraction = water_fraction.add(purewater.multiply(ee.Image(1.0).subtract(water_fraction)))
    #addToMap(fraction)
    #addToMap(purewater, {}, 'Pure Water', false)
    #addToMap(pureland, {}, 'Pure Land', false)
    #addToMap(averageland, {}, 'Average Land', false)
    #addToMap(water_fraction, {}, 'Water Fraction', false)
    #addToMap(purewaterref, {}, 'Pure Water Reflection', false)
    return water_fraction.select(['sur_refl_b01'], ['b1'])

def dnns_dem(domain, b):
    water_fraction = dnns(domain, b)

    dem_min = domain.dem.mask(water_fraction).focal_min(250, 'square', 'meters')
    dem_max = domain.dem.mask(water_fraction).focal_max(250, 'square', 'meters')
    # approximation, linearize each tile's fraction point
    water_high = dem_min.add(dem_max.subtract(dem_min).multiply(water_fraction))
    water_high = water_high.multiply(water_fraction.eq(1.0)) # don't include full pixels
    water_dem_kernel = ee.Kernel.circle(5000, 'meters', False)
    # TODO: find percentile median
    average_high = water_high.convolve(water_dem_kernel).divide(water_high.gt(0.0).convolve(water_dem_kernel))
    #addToMap(water_fraction, {}, 'Water Fraction', false);
    #addToMap(average_high, {min:25, max:40}, 'Water Level', false);
    #addToMap(dem.subtract(average_high), {min : -0, max : 10}, 'Water Difference', false);
    #addToMap(dem.lte(average_high).and(domain.groundTruth.not()));
    return domain.dem.lte(average_high).Or(water_fraction.eq(1.0)).select(['elevation'], ['b1'])

HISTORY_THRESHOLDS = {
        BORDER         : (6.5,     -3.5),
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
    (dev_thresh, change_thresh) = HISTORY_THRESHOLDS[domain.id]
    history = ee.ImageCollection('MOD09GQ').filterDate(domain.date.advance(-1 - 0.03, 'year'), domain.date.advance(-1 + 0.03, 'year')).filterBounds(domain.bounds);
    for i in range(1, 4):
        history.merge(ee.ImageCollection('MOD09GQ').filterDate(domain.date.advance(-(i+1) - 0.03, 'year'), domain.date.advance(-(i+1) + 0.03, 'year')).filterBounds(domain.bounds));
    flood_diff = (lambda x : x.select(['sur_refl_b02']).subtract(x.select(['sur_refl_b01'])))
    historyDiff = history.map(flood_diff)
    historyMean   = historyDiff.mean()
    historyStdDev = historyDiff.reduce(ee.Reducer.stdDev())
    floodDiff  = flood_diff(domain.high_res_modis)
    diffOfDiffs = floodDiff.subtract(historyMean)
    plainDiff   = highResImage.subtract(historyMean)
    ddDivDev = diffOfDiffs.divide(historyStdDev)
    changeFlood = ddDivDev.lt(change_thresh)
    diffInWaterMask = floodDiff.multiply(domain.water_mask)
    maskedMean   = diffInWaterMask.reduceRegion(ee.Reducer.mean(), domain.bounds, 250)
    maskedStdDev = diffInWaterMask.reduceRegion(ee.Reducer.stdDev(), domain.bounds, 250)
    waterThreshold = maskedMean.getInfo().sur_refl_b02 + dev_thresh * maskedStdDev.getInfo().sur_refl_b02;
    waterPixels = modis_diff(domain, b, waterThreshold)
    return waterPixels.Or(changeFlood).select(['sur_refl_b02'], ['b1']);

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

__ALGORITHMS = {
        EVI :  ('EVI',  evi, False, 'FF00FF'),
        XIAO : ('XIAO', xiao, False, 'FFFF00'),
        DIFFERENCE : ('Difference', modis_diff, False, '00FFFF'),
        CART : ('CART', cart, False, 'CC6600'),
        SVM : ('SVM', svm, False, 'FFAA33'),
        RANDOM_FORESTS : ('Random Forests', random_forests, False,'CC33FF'),
        DNNS : ('DNNS', dnns, False, '0000FF'),
        DNNS_DEM : ('DNNS with DEM', dnns_dem, False, '9900FF'),
        DIFFERENCE_HISTORY : ('Difference with History', history_diff, False, '0099FF'),
        DARTMOUTH : ('Dartmouth', dartmouth, False, '33CCFF')
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

