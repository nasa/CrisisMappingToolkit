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

import ee
import math
#from domains import *

from cmt.mapclient_qt import addToMap

'''
Contains implementations of multiple MODIS-based flood detection algorithms.
'''

# Each algorithm name has an integer assigned to it.
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
DEM_THRESHOLD      = 12
MARTINIS_TREE      = 13
SKYBOX_ASSIST      = 14
DNNS_DIFF          = 15
DNNS_DIFF_DEM      = 16
DIFF_LEARNED       = 17
DART_LEARNED       = 18
FAI                = 19
FAI_LEARNED        = 20
EXPERIMENTAL       = 21

def compute_modis_indices(domain):
    '''Compute several common interpretations of the MODIS bands'''
    
    band1 = domain.modis.sur_refl_b01 # pRED
    band2 = domain.modis.sur_refl_b02 # pNIR

    # Other bands must be used at lower resolution
    band3 = domain.modis.sur_refl_b03 # pBLUE
    band5 = domain.modis.sur_refl_b05
    band6 = domain.modis.sur_refl_b06 # pSWIR

    NDVI = (band2.subtract(band1)).divide(band2.add(band1));
    # Normalized difference water index
    NDWI = (band1.subtract(band6)).divide(band1.add(band6));
    # Enhanced vegetation index
    EVI = band2.subtract(band1).multiply(2.5).divide( band2.add(band1.multiply(6)).subtract(band3.multiply(7.5)).add(1));
    # Land surface water index
    LSWI = (band2.subtract(band6)).divide(band2.add(band6));
    # Convenience measure
    DVEL = EVI.subtract(LSWI)

    return {'b1': band1, 'b2': band2, 'b3': band3, 'b5' : band5, 'b6': band6,
            'NDVI': NDVI, 'NDWI': NDWI, 'EVI': EVI, 'LSWI': LSWI, 'DVEL': DVEL,
            'pRED': band1, 'pNIR': band2, 'pBLUE': band3, 'pSWIR': band6}



def getQABits(image, start, end, newName):
    '''Extract bits from positions "start" to "end" in the image'''
    # Create a bit mask of the bits we need
    pattern = 0
    for i in range(start,end):
       pattern += 2**i
    # Extract the bits, shift them over, and rename the channel.
    temp = ee.Image(pattern)
    return image.select([0], [newName]).bitwise_and(temp).rightShift(start)

def getModisBadPixelMask(lowResModis):
    '''Retrieves the 1km MODIS bad pixel mask (identifies clouds)'''

    # Select the QA band
    qaBand = lowResModis.select('state_1km').uint16()
   
    # Get the cloud_state bits and find cloudy areas.
    cloudBits = getQABits(qaBand, 0, 1, 'cloud_state')
    cloud = cloudBits.eq(1).Or(cloudBits.eq(2))

    return cloud # The second part of this, the land water flag, does not work well at all.
    
    ## Get the land_water_flag bits.
    #landWaterFlag = getQABits(qaBand, 3, 5, 'land_water_flag')
    #
    ## Create a mask that filters out deep ocean and cloudy areas.
    #mask = landWaterFlag.neq(7).And(cloud.Not())
    #return mask

def getCloudPercentage(lowResModis, region):
    '''Returns the percentage of a region flagged as clouds by the MODIS metadata'''

    MODIS_CLOUD_RESOLUTION = 1000 # Clouds are flagged at this resolution

    # Divide the number of cloud pixels by the total number of pixels
    oneMask    = ee.Image(1.0) 
    cloudMask  = getModisBadPixelMask(lowResModis)
    areaCount  = oneMask.reduceRegion(  ee.Reducer.sum(), region, MODIS_CLOUD_RESOLUTION)
    cloudCount = cloudMask.reduceRegion(ee.Reducer.sum(), region, MODIS_CLOUD_RESOLUTION)
    percentage = cloudCount.getInfo()['cloud_state'] / areaCount.getInfo()['constant']
    print 'Detected cloud percentage: ' + str(percentage)
    return percentage



# if mixed_thresholds is true, we find the thresholds that contain 0.05 land and 0.95 water
def compute_binary_threshold(valueImage, classification, bounds, mixed_thresholds=False):
    '''Computes a threshold for a value given examples in a classified binary image'''
    
    # Build histograms of the true and false labeled values
    valueInFalse   = valueImage.mask(classification.Not())
    valueInTrue    = valueImage.mask(classification)
    NUM_BINS       = 128
    SCALE          = 250 # In meters
    histogramFalse = valueInFalse.reduceRegion(ee.Reducer.histogram(NUM_BINS, None, None), bounds, SCALE).getInfo()['b1']
    histogramTrue  = valueInTrue.reduceRegion( ee.Reducer.histogram(NUM_BINS, None, None), bounds, SCALE).getInfo()['b1']
    
    # Get total number of pixels in each histogram
    false_total = sum(histogramFalse['histogram'])
    true_total  = sum(histogramTrue[ 'histogram'])
    
    # WARNING: This method assumes that the false histogram is composed of greater numbers than the true histogram!!
    #        : This happens to be the case for the three algorithms we are currently using this for.
    
    false_index = 0
    false_sum   = false_total
    true_sum    = 0.0
    threshold_index = None
    lower_mixed_index = None
    upper_mixed_index = None
    for i in range(len(histogramTrue['histogram'])): # Iterate through the bins of the true histogram
        # Add the number of pixels in the current true bin
        true_sum += histogramTrue['histogram'][i]
        
        # Set x equal to the max end of the current bin
        x = histogramTrue['bucketMin'] + (i+1)*histogramTrue['bucketWidth']
        
        # Determine the bin of the false histogram that x falls in
        # - Also update the number of 
        while ( (false_index < len(histogramFalse['histogram'])) and
                (histogramFalse['bucketMin'] + false_index*histogramFalse['bucketWidth'] < x) ):
            false_sum   -= histogramFalse['histogram'][false_index] # Remove the pixels from the current false bin
            false_index += 1 # Move to the next bin of the false histogram
    
        percent_true_under_thresh = true_sum/true_total
        percent_false_over_thresh = false_sum/false_total
            
        if mixed_thresholds:
            if (false_total - false_sum) / float(true_sum) <= 0.05:
                lower_mixed_index = i
            if upper_mixed_index == None and (true_total - true_sum) / float(false_sum) <= 0.05:
                upper_mixed_index = i
        else:
            if threshold_index == None and (percent_false_over_thresh < percent_true_under_thresh) and (percent_true_under_thresh > 0.5):
                break

    
    if mixed_thresholds:
        lower = histogramTrue['bucketMin'] + lower_mixed_index * histogramTrue['bucketWidth'] + histogramTrue['bucketWidth']/2
        upper = histogramTrue['bucketMin'] + upper_mixed_index * histogramTrue['bucketWidth'] + histogramTrue['bucketWidth']/2
        if lower > upper:
            temp = lower
            lower = upper
            upper = temp
        print 'Thresholds (%g, %g) found.' % (lower, upper)
        return (lower, upper)
    else:
        # Put threshold in the center of the current true histogram bin/bucket
        threshold = histogramTrue['bucketMin'] + i*histogramTrue['bucketWidth'] + histogramTrue['bucketWidth']/2
        print 'Threshold %g Found. %g%% of water pixels and %g%% of land pixels separated.' % \
            (threshold, true_sum / true_total * 100.0, false_sum / false_total * 100.0)
        return threshold

def dem_threshold(domain, b):
    '''Just use a height threshold on the DEM!'''

    heightLevel = float(domain.algorithm_params['dem_threshold'])
    dem         = domain.get_dem().image
    return dem.lt(heightLevel).select(['elevation'], ['b1'])

def evi(domain, b):
    '''Simple EVI based classifier'''
    #no_clouds = b['b3'].lte(2100).select(['sur_refl_b03'], ['b1'])
    criteria1 = b['EVI'].lte(0.3).And(b['LSWI'].subtract(b['EVI']).gte(0.05)).select(['sur_refl_b02'], ['b1'])
    criteria2 = b['EVI'].lte(0.05).And(b['LSWI'].lte(0.0)).select(['sur_refl_b02'], ['b1'])
    #return no_clouds.And(criteria1.Or(criteria2))
    return criteria1.Or(criteria2)

def xiao(domain, b):
    '''Method from paper: Xiao, Boles, Frolking, et. al. Mapping paddy rice agriculture in South and Southeast Asia using
                          multi-temporal MODIS images, Remote Sensing of Environment, 2006.
                          
        This method implements a very simple decision tree from several standard MODIS data products.
        The default constants were tuned for (wet) rice paddy detection.
    '''
    return b['LSWI'].subtract(b['NDVI']).gte(0.05).Or(b['LSWI'].subtract(b['EVI']).gte(0.05)).select(['sur_refl_b02'], ['b1']);

def get_permanent_water_mask():
    return ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'], ['b1'])

def get_diff(b):
    return b['b2'].subtract(b['b1']).select(['sur_refl_b02'], ['b1'])

def diff_learned(domain, b):
    '''modis_diff but with the threshold calculation included (training image required)'''
    if domain.unflooded_domain == None:
        print 'No unflooded training domain provided.'
        return None
    unflooded_b = compute_modis_indices(domain.unflooded_domain)
    water_mask = get_permanent_water_mask()
    
    threshold = compute_binary_threshold(get_diff(unflooded_b), water_mask, domain.bounds)
    return modis_diff(domain, b, threshold)

def modis_diff(domain, b, threshold=None):
    '''Compute (b2-b1) < threshold, a simple water detection index.
    
       This method may be all that is needed in cases where the threshold can be hand tuned.
    '''
    if threshold == None: # If no threshold value passed in, load it based on the data set.
        threshold = float(domain.algorithm_params['modis_diff_threshold'])
    return get_diff(b).lte(threshold)

def get_fai(b):
    return b['b2'].subtract(b['b1'].add(b['b5'].subtract(b['b1']).multiply((859.0 - 645) / (1240 - 645)))).select(['sur_refl_b02'], ['b1'])

def fai_learned(domain, b):
    if domain.unflooded_domain == None:
        print 'No unflooded training domain provided.'
        return None
    unflooded_b = compute_modis_indices(domain.unflooded_domain)
    water_mask = get_permanent_water_mask()
    
    threshold = compute_binary_threshold(get_fai(unflooded_b), water_mask, domain.bounds)
    return fai(domain, b, threshold)

def fai(domain, b, threshold=None):
    ''' Floating Algae Index. Method from paper: Feng, Hu, Chen, Cai, Tian, Gan,
    Assessment of inundation changes of Poyang Lake using MODIS observations
    between 2000 and 2010. Remote Sensing of Environment, 2012.
    '''
    if threshold == None:
        threshold = float(domain.algorithm_params['fai_threshold'])
    return get_fai(b).lte(threshold)

def _create_learning_image(domain, b):
    '''Set up features for the classifier to be trained on: [b2, b2/b1, b2/b1, NDVI, NDWI]'''
    diff        = b['b2'].subtract(b['b1'])
    ratio       = b['b2'].divide(b['b1'])
    modisBands  = b['b1'].addBands(b['b2']).addBands(diff).addBands(ratio).addBands(b['NDVI']).addBands(b['NDWI'])
    outputBands = modisBands
    
    # Try to add a DEM
    try:
        dem = domain.get_dem().image
        outputBands.addBands(dem)
        #outputBands = dem
    except AttributeError:
        pass # Suppress error if there is no DEM data
    
    # Try to add Skybox RGB info (NIR is handled seperately because not all Skybox images have it)
    # - Use all the base bands plus a grayscale texture measure
    try:
        try: # The Skybox data can be in one of two names
            skyboxSensor = domain.skybox
        except:
            skyboxSensor = domain.skybox_nir
            
        rgbBands    = skyboxSensor.Red.addBands(skyboxSensor.Green).addBands(skyboxSensor.Blue)
        grayBand    = rgbBands.select('Red').add(rgbBands.select('Green')).add(rgbBands.select('Blue')).divide(ee.Image(3.0)).uint16()
        edges       = grayBand.convolve(ee.Kernel.laplacian8(normalize=True)).abs()
        texture     = edges.convolve(ee.Kernel.square(3, 'pixels')).select(['Red'], ['Texture'])
        texture2Raw = grayBand.glcmTexture()
        bandList    = texture2Raw.getInfo()['bands']
        bandName    = [x['id'] for x in bandList if 'idm' in x['id']]
        texture2    = texture2Raw.select(bandName).convolve(ee.Kernel.square(5, 'pixels'))
        #skyboxBands = rgbBands.addBands(texture).addBands(texture2)
        skyboxBands = rgbBands.addBands(texture2)
        outputBands = outputBands.addBands(skyboxBands)
        #outputBands = skyboxBands
        
        #addToMap(grayBand, {'min': 0, 'max': 1200}, 'grayBand')       
        #addToMap(edges, {'min': 0, 'max': 250}, 'edges')
        #addToMap(texture, {'min': 0, 'max': 250}, 'texture')
        #addToMap(texture2, {'min': 0, 'max': 1}, 'texture2')
        
    except AttributeError:
        pass # Suppress error if there is no Skybox data
    
    # Try to add Skybox Near IR band
    try:
        outputBands = outputBands.addBands(domain.skybox_nir.NIR)       
        #addToMap(domain.skybox.NIR, {'min': 0, 'max': 1200}, 'Near IR')       
    except AttributeError:
        pass # Suppress error if there is no Skybox NIR data
    
    return outputBands

def earth_engine_classifier(domain, b, classifier_name, extra_args={}):
    '''Apply EE classifier tool using a ground truth image.'''
    
    # Training requires a training image plus either ground truth or training features.
    
    if not domain.training_domain:
        raise Exception('Cannot run classifier algorithm without a training domain!')
    training_domain = domain.training_domain    
    training_image  = _create_learning_image(training_domain, compute_modis_indices(training_domain))
    if training_domain.training_features:
        #print 'USING FEATURES'
        #print training_domain.training_features.getInfo()
        args = {
                'training_features' : training_domain.training_features,
                'training_property' : 'classification',
                #'crs'               : 'EPSG:32736',
                #'crs_transform'     : [0.8,0,733605.2,0,-0.8,8117589.2]
                "crs": "EPSG:4326", # TODO: What to use here???
                "crs_transform": [8.9831528411952135e-05, 0, -180, 0, -8.9831528411952135e-05, 90],
               }
    elif training_domain.ground_truth:
        args = {
                'training_image'    : training_domain.ground_truth,
                'training_band'     : "b1",
                'training_region'   : training_domain.bounds
               }
    else:
        raise Exception('Cannot run classifier algorithm without a training features or a ground truth!')
    common_args = {
                   'image'             : training_image,
                   'subsampling'       : 0.2, # TODO: Reduce this on failure?
                   'max_classification': 2,
                   'classifier_mode' : 'classification',
                   'classifier_name'   : classifier_name
                  }
    args.update(common_args)
    args.update(extra_args)
    classifier = ee.apply("TrainClassifier", args)  # Call the EE classifier
    classified = _create_learning_image(domain, b).classify(classifier).select(['classification'], ['b1'])
    
    
    # For high resolution Skybox images, apply an additional filter step to clean up speckles.
    try:
        try: # The Skybox data can be in one of two names
            skyboxSensor = domain.skybox
        except:
            skyboxSensor = domain.skybox_nir
        classified = classified.focal_min(13, 'circle', 'meters').focal_max(13, 'circle', 'meters')
    except:
        pass
    
    return classified;

def cart(domain, b):
    '''Classify using CART (Classification And Regression Tree)'''
    return earth_engine_classifier(domain, b, 'Cart')

def svm(domain, b):
    '''Classify using Pegasos classifier'''
    return earth_engine_classifier(domain, b, 'Pegasos')

def random_forests(domain, b):
    '''Classify using RifleSerialClassifier (Random Forests)'''
    return earth_engine_classifier(domain, b, 'RifleSerialClassifier')

def dnns_diff(domain, b):
    '''The DNNS algorithm but faster because it approximates the initial CART classification with
        a simple difference based method.'''
    return dnns(domain, b, True)

def dnns(domain, b, use_modis_diff=False):
    '''Dynamic Nearest Neighbor Search adapted from the paper:
        "Li, Sun, Yu, et. al. "A new short-wave infrared (SWIR) method for
        quantitative water fraction derivation and evaluation with EOS/MODIS
        and Landsat/TM data." IEEE Transactions on Geoscience and Remote Sensing, 2013."
        
        The core idea of this algorithm is to compute local estimates of a "pure water"
        and "pure land" pixel and compute each pixel's water percentage as a mixed
        composition of those two pure spectral types.
    '''
    
    # This algorithm has some differences from the original paper implementation.
    #  The most signficant of these is that it does not make use of land/water/partial
    #  preclassifications like the original paper does.  The search range is also
    #  much smaller in order to make the algorithm run faster in Earth Engine.
    # - Running this with a tiny kernel (effectively treating the entire region
    #    as part of the kernel) might get the best results!

    # Parameters
    KERNEL_SIZE = 40 # The original paper used a 100x100 pixel box = 25,000 meters!
    
    AVERAGE_SCALE_METERS = 250 # This scale is used to compute averages over the entire region
    
    # Set up two square kernels of the same size
    # - These kernels define the search range for nearby pure water and land pixels
    kernel            = ee.Kernel.square(KERNEL_SIZE, 'pixels', False)
    kernel_normalized = ee.Kernel.square(KERNEL_SIZE, 'pixels', True)
    
    # Compute b1/b6 and b2/b6
    composite_image = b['b1'].addBands(b['b2']).addBands(b['b6'])
    
    # Use CART classifier to divide pixels up into water, land, and mixed.
    # - Mixed pixels are just low probability water/land pixels.
    if use_modis_diff:
        unflooded_b = compute_modis_indices(domain.unflooded_domain)
        water_mask = get_permanent_water_mask()
        thresholds = compute_binary_threshold(get_diff(unflooded_b), water_mask, domain.bounds, True)

        pureWater = modis_diff(domain, b, thresholds[0])
        pureLand = modis_diff(domain, b, thresholds[1]).Not()
        mixed = pureWater.Or(pureLand).Not()
    else:
        classes   = earth_engine_classifier(domain, b, 'Pegasos', {'classifier_mode' : 'probability'})
        pureWater = classes.gte(0.95)
        pureLand  = classes.lte(0.05)
        #addToMap(classes, {'min': -1, 'max': 1}, 'CLASSES')
        #raise Exception('DEBUG')
        mixed     = pureWater.Not().And(pureLand.Not())
    averageWater      = pureWater.mask(pureWater).multiply(composite_image).reduceRegion(ee.Reducer.mean(), domain.bounds, AVERAGE_SCALE_METERS)
    averageWaterImage = ee.Image([averageWater.getInfo()['sur_refl_b01'], averageWater.getInfo()['sur_refl_b02'], averageWater.getInfo()['sur_refl_b06']])
    
    # For each pixel, compute the number of nearby pure water pixels
    pureWaterCount = pureWater.convolve(kernel)
    # Get mean of nearby pure water (b1,b2,b6) values for each pixel with enough pure water nearby
    MIN_PUREWATER_NEARBY = 1
    pureWaterRef = pureWater.multiply(composite_image).convolve(kernel).multiply(pureWaterCount.gte(MIN_PUREWATER_NEARBY)).divide(pureWaterCount)
    # For pixels that did not have enough pure water nearby, just use the global average water value
    pureWaterRef = pureWaterRef.add(averageWaterImage.multiply(pureWaterRef.Not()))

   
    # Compute a backup, global pure land value to use when pixels have none nearby.
    averagePureLand      = pureLand.mask(pureLand).multiply(composite_image).reduceRegion(ee.Reducer.mean(), domain.bounds, AVERAGE_SCALE_METERS)
    #averagePureLand      = composite_image.mask(pureLand).reduceRegion(ee.Reducer.mean(), domain.bounds, AVERAGE_SCALE_METERS)
    
    averagePureLandImage = ee.Image([averagePureLand.getInfo()['sur_refl_b01'], averagePureLand.getInfo()['sur_refl_b02'], averagePureLand.getInfo()['sur_refl_b06']])
    
    # Implement equations 10 and 11 from the paper --> It takes many lines of code to compute the local land pixels!
    oneOverSix   = b['b1'].divide(b['b6'])
    twoOverSix   = b['b2'].divide(b['b6'])
    eqTenLeft    = oneOverSix.subtract( pureWaterRef.select('sur_refl_b01').divide(b['b6']) )
    eqElevenLeft = twoOverSix.subtract( pureWaterRef.select('sur_refl_b02').divide(b['b6']) )
    
    # For each pixel, grab all the ratios from nearby pixels
    nearbyPixelsOneOverSix = oneOverSix.neighborhoodToBands(kernel) # Each of these images has one band per nearby pixel
    nearbyPixelsTwoOverSix = twoOverSix.neighborhoodToBands(kernel)
    nearbyPixelsOne        = b['b1'].neighborhoodToBands(kernel)
    nearbyPixelsTwo        = b['b2'].neighborhoodToBands(kernel)
    nearbyPixelsSix        = b['b6'].neighborhoodToBands(kernel)
    
    # Find which nearby pixels meet the EQ 10 and 11 criteria
    eqTenMatches        = ( nearbyPixelsOneOverSix.gt(eqTenLeft   ) ).And( nearbyPixelsOneOverSix.lt(oneOverSix) )
    eqElevenMatches     = ( nearbyPixelsTwoOverSix.gt(eqElevenLeft) ).And( nearbyPixelsTwoOverSix.lt(twoOverSix) )
    nearbyLandPixels    = eqTenMatches.And(eqElevenMatches)
    
    # Find the average of the nearby matching pixels
    numNearbyLandPixels = nearbyLandPixels.reduce(ee.Reducer.sum())
    meanNearbyBandOne   = nearbyPixelsOne.multiply(nearbyLandPixels).reduce(ee.Reducer.sum()).divide(numNearbyLandPixels)
    meanNearbyBandTwo   = nearbyPixelsTwo.multiply(nearbyLandPixels).reduce(ee.Reducer.sum()).divide(numNearbyLandPixels)
    meanNearbyBandSix   = nearbyPixelsSix.multiply(nearbyLandPixels).reduce(ee.Reducer.sum()).divide(numNearbyLandPixels)

    # Pack the results into a three channel image for the whole region
    # - Use the global pure land calculation to fill in if there are no nearby equation matching pixels
    MIN_PURE_NEARBY = 1
    meanPureLand = meanNearbyBandOne.addBands(meanNearbyBandTwo).addBands(meanNearbyBandSix)
    meanPureLand = meanPureLand.multiply(numNearbyLandPixels.gte(MIN_PURE_NEARBY)).add( averagePureLandImage.multiply(numNearbyLandPixels.lt(MIN_PURE_NEARBY)) )

    # Compute the water fraction: (land[b6] - b6) / (land[b6] - water[b6])
    # - Ultimately, relying solely on band 6 for the final classification may not be a good idea!
    meanPureLandSix = meanPureLand.select('sum_2')
    water_fraction = (meanPureLandSix.subtract(b['b6'])).divide(meanPureLandSix.subtract(pureWaterRef.select('sur_refl_b06'))).clamp(0, 1)
       
    # Set pure water to 1, pure land to 0
    water_fraction = water_fraction.add(pureWater).subtract(pureLand).clamp(0, 1)
    
    #addToMap(fraction, {'min': 0, 'max': 1},   'fraction', False)
    #addToMap(pureWater,      {'min': 0, 'max': 1},   'pure water', False)
    #addToMap(pureLand,       {'min': 0, 'max': 1},   'pure land', False)
    #addToMap(mixed,          {'min': 0, 'max': 1},   'mixed', False)
    #addToMap(pureWaterCount, {'min': 0, 'max': 300}, 'pure water count', False)
    #addToMap(water_fraction, {'min': 0, 'max': 5},   'water_fractionDNNS', False)
    #addToMap(pureWaterRef,   {'min': 0, 'max': 3000, 'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06']}, 'pureWaterRef', False)

    return water_fraction.select(['sum_2'], ['b1']) # Rename sum_2 to b1

def dnns_diff_dem(domain, b):
    '''The DNNS DEM algorithm but faster because it approximates the initial CART classification with
        a simple difference based method.'''
    return dnns_dem(domain, b, True)

def dnns_dem(domain, b, use_modis_diff=False):
    '''Enhance the DNNS result with high resolution DEM information, adapted from the paper:
        "Li, Sun, Goldberg, and Stefanidis. "Derivation of 30-m-resolution
        water maps from TERRA/MODIS and SRTM." Remote Sensing of Environment, 2013."
        
        This enhancement to the DNNS_DEM algorithm combines a higher resolution DEM with
        input pixels that have a percent flooded value.  Based on the percentage and DEM
        values, the algorithm classifies each higher resolution DEM pixel as either
        flooded or dry.
        '''
    
    MODIS_PIXEL_SIZE_METERS = 250
    
    # Call the DNNS function to get the starting point
    water_fraction = dnns(domain, b, use_modis_diff)

   
    ## Treating the DEM values contained in the MODIS pixel as a histogram, find the N'th percentile
    ##  where N is the water fraction computed by DNNS.  That should be the height of the flood water.
    #modisPixelKernel = ee.Kernel.square(MODIS_PIXEL_SIZE_METERS, 'meters', False)
    #dem.mask(water_fraction).reduceNeighborhood(ee.Reducer.percentile(), modisPixelKernel)
    # --> We would like to compute a percentile here, but this would require a different reducer input for each pixel!
    
    # Get whichever DEM is loaded in the domain
    dem = domain.get_dem().image
    
    # Get min and max DEM height within each water containing pixel
    # - If a DEM pixel contains any water then the water level must be at least that high.    
    dem_min = dem.mask(water_fraction).focal_min(MODIS_PIXEL_SIZE_METERS, 'square', 'meters')
    dem_max = dem.mask(water_fraction).focal_max(MODIS_PIXEL_SIZE_METERS, 'square', 'meters')
    
    # Approximation, linearize each tile's fraction point
    # - The water percentage is used as a percentage between the two elevations
    # - Don't include full or empty pixels, they don't give us clues to their height.
    water_high = dem_min.add(dem_max.subtract(dem_min).multiply(water_fraction))
    water_high = water_high.multiply(water_fraction.lt(1.0)).multiply(water_fraction.gt(0.0)) 
    
    # Problem: Averaging process spreads water way out to pixels where it was not detected!!
    #          Reducing the averaging is a simple way to deal with this and probably does not hurt results at all
    
    #dilate_kernel = ee.Kernel.circle(250, 'meters', False)
    #allowed_water_mask = water_fraction.gt(0.0)#.convolve(dilate_kernel)
    
    ## Smooth out the water elevations with a broad kernel; nearby pixels probably have the same elevation!
    water_dem_kernel        = ee.Kernel.circle(5000, 'meters', False)
    num_nearby_water_pixels = water_high.gt(0.0).convolve(water_dem_kernel)
    average_high            = water_high.convolve(water_dem_kernel).divide(num_nearby_water_pixels)
    
    # Alternate smoothing method using available tool EE
    water_present   = water_fraction.gt(0.0)#.mask(water_high)
    #connected_water = ee.Algorithms.ConnectedComponentLabeler(water_present, water_dem_kernel, 256) # Perform blob labeling
    #addToMap(water_present,   {'min': 0, 'max':   1}, 'water present', False);
    #addToMap(connected_water, {'min': 0, 'max': 256}, 'labeled blobs', False);
    
    #addToMap(water_fraction, {'min': 0, 'max':   1}, 'Water Fraction', False);
    #addToMap(average_high, {min:25, max:40}, 'Water Level', false);
    #addToMap(dem.subtract(average_high), {min : -0, max : 10}, 'Water Difference', false);
    #addToMap(dem.lte(average_high).and(domain.groundTruth.not()));
    
    #addToMap(allowed_water_mask, {'min': 0, 'max': 1}, 'allowed_water', False);
    #addToMap(water_high, {'min': 0, 'max': 100}, 'water_high', False);
    #addToMap(average_high, {'min': 0, 'max': 100}, 'average_high', False);
    #addToMap(dem, {'min': 0, 'max': 100}, 'DEM', False);
    
    # Classify DEM pixels as flooded based on being under the local water elevation or being completely flooded.
    #return dem.lte(average_high).Or(water_fraction.eq(1.0)).select(['elevation'], ['b1'])
    dem_water = dem.lte(average_high).mask(water_fraction) # Mask prevents pixels with 0% water from being labeled as water
    return dem_water.Or(water_fraction.eq(1.0)).select(['elevation'], ['b1'])


def history_diff(domain, b):
    '''Wrapper function for passing domain data into history_diff_core'''
    
    # Load pre-selected constants for this domain   
    dev_thresh    = float(domain.algorithm_params['modis_mask_threshold' ])
    change_thresh = float(domain.algorithm_params['modis_change_threshold'])
    
    # Call the core algorithm with all the parameters it needs from the domain
    return history_diff_core(domain.modis, domain.modis.get_date(), dev_thresh, change_thresh, domain.bounds)
    
def history_diff_core(high_res_modis, date, dev_thresh, change_thresh, bounds):
    '''Leverage historical data and the permanent water mask to improve the threshold method.
    
       This method computes statistics from the permanent water mask to set a good b2-b1
       water detection threshold.  It also adds pixels which have a b2-b1 level significantly
       lower than the historical seasonal average.
    '''

    # Retrieve all the MODIS images for the region in the last several years
    NUM_YEARS_BACK         = 5
    NUM_DAYS_COMPARE_RANGE = 40.0 # Compare this many days before/after the target day in previous years
    YEAR_RANGE_PERCENTAGE  = NUM_DAYS_COMPARE_RANGE / 365.0
    #print 'YEAR_RANGE_PERCENTAGE = ' + str(YEAR_RANGE_PERCENTAGE)
    #print 'Start: ' + str(domain.date.advance(-1 - YEAR_RANGE_PERCENTAGE, 'year'))
    #print 'End:   ' + str(domain.date.advance(-1 + YEAR_RANGE_PERCENTAGE, 'year'))

    # Get both the high and low res MODIS data    
    historyHigh = ee.ImageCollection('MOD09GQ').filterDate(date.advance(-1 - YEAR_RANGE_PERCENTAGE, 'year'), date.advance(-1 + YEAR_RANGE_PERCENTAGE, 'year')).filterBounds(bounds);
    historyLow  = ee.ImageCollection('MOD09GA').filterDate(date.advance(-1 - YEAR_RANGE_PERCENTAGE, 'year'), date.advance(-1 + YEAR_RANGE_PERCENTAGE, 'year')).filterBounds(bounds);
    for i in range(1, NUM_YEARS_BACK-1):
        yearMin = -(i+1) - YEAR_RANGE_PERCENTAGE
        yearMax = -(i+1) + YEAR_RANGE_PERCENTAGE
        historyHigh.merge(ee.ImageCollection('MOD09GQ').filterDate(date.advance(yearMin, 'year'), date.advance(yearMax, 'year')).filterBounds(bounds));
        historyLow.merge( ee.ImageCollection('MOD09GA').filterDate(date.advance(yearMin, 'year'), date.advance(yearMax, 'year')).filterBounds(bounds));
        # TODO: Add a filter here to remove cloud-filled images
    
    # Simple function implements the b2 - b1 difference method
    # - Using two methods like this is a hack to get a call from modis_lake_measure.py to work!
    flood_diff_function1 = lambda x : x.select(['sur_refl_b02']).subtract(x.select(['sur_refl_b01']))
    flood_diff_function2 = lambda x : x.sur_refl_b02.subtract(x.sur_refl_b01)
    
    # Apply difference function to all images in history, then compute mean and standard deviation of difference scores.
    historyDiff   = historyHigh.map(flood_diff_function1)
    historyMean   = historyDiff.mean()
    historyStdDev = historyDiff.reduce(ee.Reducer.stdDev())
    
    # Display the mean image for bands 1/2/6
    history3  = historyHigh.mean().select(['sur_refl_b01', 'sur_refl_b02']).addBands(historyLow.mean().select(['sur_refl_b06']))
    #addToMap(history3, {'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06'], 'min' : 0, 'max': 3000, 'opacity' : 1.0}, 'MODIS_HIST', False)
    
    #addToMap(historyMean,   {'min' : 0, 'max' : 4000}, 'History mean',   False)
    #addToMap(historyStdDev, {'min' : 0, 'max' : 2000}, 'History stdDev', False)
    
    # Compute flood diff on current image and compare to historical mean/STD.
    floodDiff   = flood_diff_function2(high_res_modis)   
    diffOfDiffs = floodDiff.subtract(historyMean)
    ddDivDev    = diffOfDiffs.divide(historyStdDev)
    changeFlood = ddDivDev.lt(change_thresh)  # Mark all pixels which are enough STD's away from the mean.
    
    #addToMap(floodDiff,   {'min' : 0, 'max' : 4000}, 'floodDiff',   False)
    #addToMap(diffOfDiffs, {'min' : -2000, 'max' : 2000}, 'diffOfDiffs', False)
    #addToMap(ddDivDev,    {'min' : -10,   'max' : 10}, 'ddDivDev',    False)
    #addToMap(changeFlood,    {'min' : 0,   'max' : 1}, 'changeFlood',    False)
    #addToMap(domain.water_mask,    {'min' : 0,   'max' : 1}, 'Permanent water mask',    False)
    
    # Compute the difference statistics inside permanent water mask pixels
    MODIS_RESOLUTION = 250 # Meters
    water_mask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'])
    diffInWaterMask  = floodDiff.multiply(water_mask)
    maskedMean       = diffInWaterMask.reduceRegion(ee.Reducer.mean(),   bounds, MODIS_RESOLUTION)
    maskedStdDev     = diffInWaterMask.reduceRegion(ee.Reducer.stdDev(), bounds, MODIS_RESOLUTION)
    
    #print 'Water mean = ' + str(maskedMean.getInfo())
    #print 'Water STD  = ' + str(maskedStdDev.getInfo())
    
    # Use the water mask statistics to compute a difference threshold, then find all pixels below the threshold.
    waterThreshold  = maskedMean.getInfo()['sur_refl_b02'] + dev_thresh*(maskedStdDev.getInfo()['sur_refl_b02']);
    #print 'Water threshold == ' + str(waterThreshold)
    waterPixels     = flood_diff_function2(high_res_modis).lte(waterThreshold)
    #waterPixels     = modis_diff(domain, b, waterThreshold)
    
    #addToMap(waterPixels,    {'min' : 0,   'max' : 1}, 'waterPixels',    False)
    #
    ## Is it worth it to use band 6?
    #B6_DIFF_AMT = 500
    #bHigh1 = (b['b6'].subtract(B6_DIFF_AMT).gt(b['b1']))
    #bHigh2 = (b['b6'].subtract(B6_DIFF_AMT).gt(b['b2']))
    #bHigh = bHigh1.And(bHigh2).reproject('EPSG:4326', scale=MODIS_RESOLUTION)
    #addToMap(bHigh.mask(bHigh),    {'min' : 0,   'max' : 1}, 'B High',    False)
    
    # Combine water pixels from the historical and water mask methods.
    return (waterPixels.Or(changeFlood)).select(['sur_refl_b02'], ['b1'])#.And(bHigh.eq(0));



def get_dartmouth(b):
    A = 500
    B = 2500
    return b['b2'].add(A).divide(b['b1'].add(B)).select(['sur_refl_b02'], ['b1'])

def dart_learned(domain, b):
    '''The dartmouth method but with threshold calculation included (training image required)'''
    if domain.unflooded_domain == None:
        print 'No unflooded training domain provided.'
        return None
    unflooded_b = compute_modis_indices(domain.unflooded_domain)
    water_mask = get_permanent_water_mask()
    threshold = compute_binary_threshold(get_dartmouth(unflooded_b), water_mask, domain.bounds)
    return dartmouth(domain, b, threshold)

def dartmouth(domain, b, threshold=None):
    '''A flood detection method from the Dartmouth Flood Observatory.
    
        This method is a refinement of the simple b2-b1 detection method.
    '''
    if threshold == None:
        threshold = float(domain.algorithm_params['dartmouth_threshold'])
    return get_dartmouth(b).lte(threshold)


# This algorithm is not too different from the corrected DNNS algorithm.
def dnns_revised(domain, b):
    '''Dynamic Nearest Neighbor Search with revisions to improve performance on our test data'''
    
    # One issue with this algorithm is that its large search range slows down even Earth Engine!
    # - With a tiny kernel size, everything is relative to the region average which seems to work pretty well.
    # Another problem is that we don't have a good way of identifying 'Definately land' pixels like we do for water.
    
    # Parameters
    KERNEL_SIZE = 1 # The original paper used a 100x100 pixel box = 25,000 meters!
    PURELAND_THRESHOLD = 3500 # TODO: Vary by domain?
    PURE_WATER_THRESHOLD_RATIO = 0.1
    
    # Set up two square kernels of the same size
    # - These kernels define the search range for nearby pure water and land pixels
    kernel            = ee.Kernel.square(KERNEL_SIZE, 'pixels', False)
    kernel_normalized = ee.Kernel.square(KERNEL_SIZE, 'pixels', True)
    
    composite_image = b['b1'].addBands(b['b2']).addBands(b['b6'])
    
    # Compute (b2 - b1) < threshold, a simple water detection algorithm.  Treat the result as "pure water" pixels.
    pureWaterThreshold = float(domain.algorithm_params['modis_diff_threshold']) * PURE_WATER_THRESHOLD_RATIO
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
    pureLand             = b['b2'].subtract(b['b1']).gte(PURELAND_THRESHOLD).select(['sur_refl_b02'], ['b1']) # Rename sur_refl_b02 to b1
    averagePureLand      = pureLand.mask(pureLand).multiply(composite_image).reduceRegion(ee.Reducer.mean(), domain.bounds, AVERAGE_SCALE_METERS)
    averagePureLandImage = ee.Image([averagePureLand.getInfo()['sur_refl_b01'], averagePureLand.getInfo()['sur_refl_b02'], averagePureLand.getInfo()['sur_refl_b06']])
    pureLandCount        = pureLand.convolve(kernel)        # Get nearby pure land count for each pixel
    averagePureLandLocal = pureLand.multiply(composite_image).convolve(kernel).multiply(pureLandCount.gte(MIN_PURE_NEARBY)).divide(pureLandCount)
    averagePureLandLocal = averagePureLandLocal.add(averagePureLandImage.multiply(averagePureLandLocal.Not())) # For pixels that did not have any pure land nearby, use mean


    # Implement equations 10 and 11 from the paper
    oneOverSix   = b['b1'].divide(b['b6'])
    twoOverSix   = b['b2'].divide(b['b6'])
    eqTenLeft    = oneOverSix.subtract( averageWaterLocal.select('sur_refl_b01').divide(b['b6']) )
    eqElevenLeft = twoOverSix.subtract( averageWaterLocal.select('sur_refl_b02').divide(b['b6']) )
    
    # For each pixel, grab all the ratios from nearby pixels
    nearbyPixelsOneOverSix = oneOverSix.neighborhoodToBands(kernel) # Each of these images has one band per nearby pixel
    nearbyPixelsTwoOverSix = twoOverSix.neighborhoodToBands(kernel)
    nearbyPixelsOne        = b['b1'].neighborhoodToBands(kernel)
    nearbyPixelsTwo        = b['b2'].neighborhoodToBands(kernel)
    nearbyPixelsSix        = b['b6'].neighborhoodToBands(kernel)
    
    # Find which nearby pixels meet the EQ 10 and 11 criteria
    eqTenMatches        = ( nearbyPixelsOneOverSix.gt(eqTenLeft   ) ).And( nearbyPixelsOneOverSix.lt(oneOverSix) )
    eqElevenMatches     = ( nearbyPixelsTwoOverSix.gt(eqElevenLeft) ).And( nearbyPixelsTwoOverSix.lt(twoOverSix) )
    nearbyLandPixels    = eqTenMatches.And(eqElevenMatches)
    
    # Find the average of the nearby matching pixels
    numNearbyLandPixels = nearbyLandPixels.reduce(ee.Reducer.sum())
    meanNearbyBandOne   = nearbyPixelsOne.multiply(nearbyLandPixels).reduce(ee.Reducer.sum()).divide(numNearbyLandPixels)
    meanNearbyBandTwo   = nearbyPixelsTwo.multiply(nearbyLandPixels).reduce(ee.Reducer.sum()).divide(numNearbyLandPixels)
    meanNearbyBandSix   = nearbyPixelsSix.multiply(nearbyLandPixels).reduce(ee.Reducer.sum()).divide(numNearbyLandPixels)


    # Pack the results into a three channel image for the whole region
    meanNearbyLand = meanNearbyBandOne.addBands(meanNearbyBandTwo).addBands(meanNearbyBandSix)
    meanNearbyLand = meanNearbyLand.multiply(numNearbyLandPixels.gte(MIN_PURE_NEARBY)).add( averagePureLandImage.multiply(numNearbyLandPixels.lt(MIN_PURE_NEARBY)) )

    addToMap(numNearbyLandPixels,  {'min': 0, 'max': 400, }, 'numNearbyLandPixels', False)
    addToMap(meanNearbyLand,       {'min': 0, 'max': 3000, 'bands': ['sum', 'sum_1', 'sum_2']}, 'meanNearbyLand', False)

    
    # Compute the water fraction: (land - b) / (land - water)
    landDiff  = averagePureLandLocal.subtract(composite_image)
    waterDiff = averageWaterLocal.subtract(composite_image)
    typeDiff  = averagePureLandLocal.subtract(averageWaterLocal)
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
    
    
    #addToMap(numNearbyLandPixels,  {'min': 0, 'max': 400, }, 'numNearbyLandPixels', False)
    #addToMap(meanNearbyLand,       {'min': 0, 'max': 3000, 'bands': ['sum', 'sum_1', 'sum_2']}, 'meanNearbyLand', False)
    addToMap(averageWaterImage,    {'min': 0, 'max': 3000, 'bands': ['constant', 'constant_1', 'constant_2']}, 'average water', False)
    addToMap(averagePureLandImage, {'min': 0, 'max': 3000, 'bands': ['constant', 'constant_1', 'constant_2']}, 'average pure land',  False)
    addToMap(averageWaterLocal,    {'min': 0, 'max': 3000, 'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06']}, 'local water ref', False)
    addToMap(averagePureLandLocal, {'min': 0, 'max': 3000, 'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06']}, 'local pure land ref',  False)
    
    
    return waterOff.select(['sur_refl_b01'], ['b1']) # Rename sur_refl_b02 to b1



def compute_dem_slope_degrees(dem, resolution):
    '''Computes a slope in degrees for each pixel of the DEM'''
    
    deriv = dem.derivative()
    dZdX    = deriv.select(['elevation_x']).divide(resolution)
    dZdY    = deriv.select(['elevation_y']).divide(resolution)
    slope = dZdX.multiply(dZdX).add(dZdY.multiply(dZdY)).sqrt().reproject("EPSG:4269", None, resolution); 
    RAD2DEG = 180 / 3.14159
    slopeAngle = slope.atan().multiply(RAD2DEG);
    return slopeAngle


def martinis_tree(domain, b):
    '''Based on Figure 3 from "A Multi-Scale Flood Monitoring System Based on Fully
        Automatic MODIS and TerraSAR-X Processing Chains" by 
        Sandro Martinis, Andre Twele, Christian Strobl, Jens Kersten and Enrico Stein
        
       Some steps had to be approximated such as the cloud filtering.  The main pixel
       classification steps are implemented as accurately as can be determined from 
       the figure.
'''
    # Note: Nodes with the same name are numbered top to bottom, left to right in order
    #       to distinguish which one a variable represents.

    # ---- Apply the initial classification at the top of the figure ----
    # Indices already computed in the 'b' object
    clouds        = b['pBLUE'].gte(0.27)
    temp1         = b['EVI'].lte(0.3 ).And(b['DVEL'].lte(0.05))
    temp2         = b['EVI'].lte(0.05).And(b['LSWI'].lte(0.00))
    nonFlood1     = b['EVI'].gt(0.3).And(temp1.Not())
    waterRelated1 = temp1.Or(temp2)
    nonFlood2     = temp1.And(temp2.Not())
    
    #addToMap(b['EVI'],  {'min': 0, 'max': 1}, 'EVI',   False)
    #addToMap(b['LSWI'], {'min': 0, 'max': 1}, 'LSWI',  False)
    #addToMap(b['DVEL'], {'min': 0, 'max': 1}, 'DVEL',  False)
    #addToMap(waterRelated1, {'min': 0, 'max': 1}, 'waterRelated1',  False)
    
    # ---- Apply DEM filtering ----
    
    # Retrieve the dem compute slopes
    demSensor       = domain.get_dem()
    dem             = demSensor.image
    demSlopeDegrees = compute_dem_slope_degrees(demSensor.image, demSensor.band_resolutions[demSensor.band_names[0]])
    #addToMap(demSlopeDegrees, {'min': 0, 'max': 90}, 'DEM slope',  False)
    
    # Filter regions of high slope
    highSlope = demSlopeDegrees.gt(10).Or( demSlopeDegrees.gt(8).And(dem.gt(2000)) )
    
    waterRelated2 = waterRelated1.And(highSlope.Not())
    nonFlood3     = waterRelated1.And(highSlope).Or(nonFlood2)
    
    mixture1 = waterRelated2.And(b['EVI'].gt(0.1))
    flood1   = waterRelated2.And(b['EVI'].lte(0.1))
    
    #addToMap(waterRelated2, {'min': 0, 'max': 1}, 'waterRelated2',  False)
    
    # ---- Approximate region growing ----
    
    # Earth Engine can't do a real region grow so approximate one with a big dilation
    REGION_GROW_SIZE = 35 # Pixels
    expansionKernel  = ee.Kernel.circle(REGION_GROW_SIZE, 'pixels', False)
    
    # Region growing 1
    temp1 = b['EVI'].lte(0.31).And(b['DVEL'].lte(0.07))
    temp2 = b['EVI'].lte(0.06).And(b['LSWI'].lte(0.01))
    relaxedConditions      = temp1.Or(temp2)
    potentialExpansionArea = waterRelated2.convolve(expansionKernel).And(nonFlood3)
    waterRelated3          = potentialExpansionArea.And(relaxedConditions)
    
    mixture3 = waterRelated3.And(b['EVI'].gt(0.1))
    flood3   = waterRelated3.And(b['EVI'].lte(0.1))
    
    # Region growing 2
    potentialExpansionArea = flood1.convolve(expansionKernel).And(mixture1)
    mixture2 = potentialExpansionArea.And(b['LSWI'].lt(0.08))
    flood2   = potentialExpansionArea.And(b['LSWI'].gte(0.08))
    
    #addToMap(nonFlood1, {'min': 0, 'max': 1}, 'nonFlood1',  False)
    #addToMap(nonFlood2, {'min': 0, 'max': 1}, 'nonFlood2',  False)
    #addToMap(nonFlood3, {'min': 0, 'max': 1}, 'nonFlood3',  False)
    #
    #addToMap(mixture1, {'min': 0, 'max': 1}, 'mixture1',  False)
    #addToMap(mixture2, {'min': 0, 'max': 1}, 'mixture2',  False)
    #addToMap(mixture3, {'min': 0, 'max': 1}, 'mixture3',  False)
    #addToMap(waterRelated3, {'min': 0, 'max': 1}, 'waterRelated3',  False)
    
    # ---- Apply water mask ----
    
    waterMask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'])

    # Not sure how exactly the paper does this but we don't care about anything in the permanent water mask!
    recedingWater = nonFlood3.And(waterMask)
    
    mergedFlood   = flood1.Or(flood2).Or(flood3)
    standingWater = mergedFlood.And(waterMask)
    flood4        = mergedFlood.And(waterMask.Not())

    # ---- Cloud handling? ----

    # In place of the complicated cloud geometery, just remove pixels
    #  that are flagged as bad in the input MODIS imagery.
    badModisPixels = getModisBadPixelMask(domain.modis.image)

    flood5 = flood4.And(badModisPixels.Not())

    # ---- Time series handling ----
    # --> This step is not included

    fullMixture = mixture2.Or(mixture3)
    #addToMap(fullMixture, {'min': 0, 'max': 1}, 'fullMixture',  False)

    # Generate the binary flood detection output we are interested in
    outputFlood = flood5.Or(standingWater)

    return outputFlood.select(['sur_refl_b02'], ['b1']) # Rename sur_refl_b02 to b1

def _create_extended_learning_image(domain, b):
    #a = get_diff(b).select(['b1'], ['b1'])
    a = b['b1'].select(['sur_refl_b01'], ['b1'])
    a = a.addBands(b['b2'].select(['sur_refl_b02'], ['b2']))
    a = a.addBands(b['b2'].divide(b['b1']).select(['sur_refl_b02'], ['ratio']))
    a = a.addBands(b['NDVI'].select(['sur_refl_b02'], ['NDVI']))
    a = a.addBands(b['NDWI'].select(['sur_refl_b01'], ['NDWI']))
    a = a.addBands(get_diff(b).select(['b1'], ['diff']))
    a = a.addBands(get_fai(b).select(['b1'], ['fai']))
    a = a.addBands(get_dartmouth(b).select(['b1'], ['dartmouth']))
    a = a.addBands(b['LSWI'].subtract(b['NDVI']).subtract(0.05).select(['sur_refl_b02'], ['LSWIminusNDVI']))
    a = a.addBands(b['LSWI'].subtract(b['EVI']).subtract(0.05).select(['sur_refl_b02'], ['LSWIminusEVI']))
    a = a.addBands(b['EVI'].subtract(0.3).select(['sur_refl_b02'], ['EVI']))
    a = a.addBands(b['LSWI'].select(['sur_refl_b02'], ['LSWI']))
    return a

# binary search to find best threshold
def __find_optimal_threshold(domains, images, truths, band_name, weights, splits):
    choices = []
    for i in range(len(splits) - 1):
        choices.append((splits[i] + splits[i+1]) / 2)
    best = None
    best_value = None
    for i in range(len(choices)):
        c = choices[i]
        flood_and_threshold_sum = sum([weights[i].mask(images[i].select(band_name).lte(c)).reduceRegion(ee.Reducer.sum(), domains[i].bounds, 250).getInfo()['constant'] for i in range(len(domains))])
        ts = [truths[i].multiply(weights[i]).divide(flood_and_threshold_sum).mask(images[i].select(band_name).lte(c)) for i in range(len(domains))]
        entropies1 = [-ts[i].multiply(ts[i].log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, 250).getInfo()['b1'] for i in range(len(domains))]# H(Y | X <= c)

        ts = [truths[i].multiply(weights[i]).divide(1 - flood_and_threshold_sum).mask(images[i].select(band_name).gt(c)) for i in range(len(domains))]
        entropies2 = [-ts[i].multiply(ts[i].log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, 250).getInfo()['b1'] for i in range(len(domains))]# H(Y | X > c)
        
        entropy1 = sum(entropies1)
        entropy2 = sum(entropies2)
        gain = (entropy1 * flood_and_threshold_sum + entropy2 * (1 - flood_and_threshold_sum))
        print c, gain, flood_and_threshold_sum, entropy1, entropy2
        if best == None or gain < best_value:
            best = i
            best_value = gain
    return (choices[best], best + 1, best_value)

def apply_classifier(image, band, threshold):
    return image.select(band).lte(threshold).multiply(2).subtract(1)

def adaboost(domain, b, classifier = None):
    if classifier == None:
        classifier = [(u'dartmouth', 0.6746916226821668, 0.9545783139872039), (u'b1', 817.4631578947368, -0.23442294410851128), (u'ratio', 3.4957167876866304, -0.20613698036794326), (u'LSWIminusNDVI', -0.18319560006184613, -0.20191291743554216), (u'EVI', 1.2912227247420454, -0.11175138956289551), (u'dartmouth', 0.7919185558963437, 0.09587432900090082), (u'diff', 971.2916666666666, -0.10554565939141827), (u'LSWIminusEVI', -0.06318265061389294, -0.09533981402236558), (u'LSWI', 0.18809171182282547, 0.07057035145131643), (u'LSWI', 0.29177473609507737, -0.10405606800405826), (u'b1', 639.7947368421053, 0.04609306857534169), (u'b1', 550.9605263157895, 0.08536825945329486), (u'LSWI', 0.23993322395895142, 0.0686895188858698), (u'LSWIminusNDVI', -0.2895140352747048, -0.05149197092271741), (u'b2', 1713.761111111111, -0.05044107229585143), (u'b2', 2147.325, 0.08569272886858223), (u'dartmouth', 0.7333050892892552, -0.0658128496826074), (u'LSWI', 0.21401246789088846, 0.047469648515471446), (u'LSWIminusNDVI', -0.34267325288113415, -0.0402902367049306), (u'ratio', 2.382344524011886, -0.03795571511345347), (u'fai', 611.5782694327731, 0.03742837135530962), (u'ratio', 2.9390306558492583, -0.03454143179044789), (u'fai', 925.5945969012605, 0.05054908824123665), (u'diff', 1336.7708333333333, -0.042539270450854885), (u'LSWIminusNDVI', -0.3160936440779195, -0.03518287810525178)]
    test_image = _create_extended_learning_image(domain, b)
    total = ee.Image(0).select(['constant'], ['b1'])
    for c in classifier:
      total = total.add(test_image.select(c[0]).lte(c[1]).multiply(2).subtract(1).multiply(c[2]))
    return total.gte(0.0)

def __compute_threshold_ranges(training_domains, training_images, water_masks, bands):
    band_splits = dict()
    for band_name in bands:
      split = None
      print band_name
      for i in range(len(training_domains)):
        ret = training_images[i].select(band_name).mask(water_masks[i]).reduceRegion(ee.Reducer.percentile([20, 80], ['s', 'b']), training_domains[i].bounds, 250).getInfo()
        s = [ret[band_name + '_s'], ret[band_name + '_b']]
        if split == None:
            split = s
        else:
            split[0] = min(split[0], s[0])
            split[1] = max(split[1], s[1])
      band_splits[band_name] = [split[0], split[1], split[1] + (split[1] - split[0])]
    return band_splits

def adaboost_learn(domain, b):
    training_domains = [domain.unflooded_domain]
    water_masks = [get_permanent_water_mask()]
    transformed_masks = [water_mask.multiply(2).subtract(1) for water_mask in water_masks]
    training_images = [_create_extended_learning_image(d, compute_modis_indices(d)) for d in training_domains]
    bands = training_images[0].bandNames().getInfo()
    print 'Computing threshold ranges.'
    band_splits = __compute_threshold_ranges(training_domains, training_images, water_masks, bands)
    counts = [training_images[i].select('b1').reduceRegion(ee.Reducer.count(), training_domains[i].bounds, 250).getInfo()['b1'] for i in range(len(training_domains))]
    count = sum(counts)

    weights = [ee.Image(1.0 / count) for i in range(len(training_domains))]
    full_classifier = []
    # initialize for pre-existing partially trained classifier
    for (c, t, alpha) in full_classifier:
      band_splits[c].append(t)
      band_splits[c] = sorted(band_splits[c])
      total = 0
      for i in range(len(training_domains)):
        weights[i] = weights[i].multiply(apply_classifier(training_images[i], c, t).multiply(transformed_masks[i]).multiply(-alpha).exp())
        total += weights[i].reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, 250).getInfo()['constant']
      for i in range(len(training_domains)):
        weights[i] = weights[i].divide(total)
    
    test_image = _create_extended_learning_image(domain, b)
    while len(full_classifier) < 20:
      best = None
      for band_name in bands:
        (threshold, ind, value) = __find_optimal_threshold(training_domains, training_images, water_masks, band_name, weights, band_splits[band_name])
        errors = [weights[i].multiply(training_images[i].select(band_name).lte(threshold).neq(water_masks[i])).reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, 250).getInfo()['constant'] for i in range(len(training_domains))]
        error = sum(errors)
        print '%s found threshold %g with entropy %g error %g' % (band_name, threshold, value, error)
        if best == None or abs(0.5 - error) > abs(0.5 - best[0]): # classifiers that are always wrong are also good with negative alpha
          best = (error, band_name, threshold, ind)
      band_splits[best[1]].insert(best[3], best[2])
      print 'Using %s < %g. Error %g.' % (best[1], best[2], best[0])
      alpha = 0.5 * math.log((1 - best[0]) / best[0])
      classifier = (best[1], best[2], alpha)
      full_classifier.append(classifier)
      weights = [weights[i].multiply(apply_classifier(training_images[i], classifier[0], classifier[1]).multiply(transformed_masks[i]).multiply(-alpha).exp()) for i in range(len(training_domains))]
      totals = [weights[i].reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, 250).getInfo()['constant'] for i in range(len(training_domains))]
      total = sum(totals)
      weights = [w.divide(total) for w in weights]
      print full_classifier
      addToMap(adaboost(domain, b, full_classifier))

def experimental(domain, b):
    return adaboost_learn(domain, b)
    #args = {
    #        'training_image'    : water_mask,
    #        'training_band'     : "b1",
    #        'training_region'   : training_domain.bounds,
    #        'image'             : training_image,
    #        'subsampling'       : 0.2, # TODO: Reduce this on failure?
    #        'max_classification': 2,
    #        'classifier_name'   : 'Cart'
    #       }
    #classifier = ee.apply("TrainClassifier", args)  # Call the EE classifier
    #classified = _create_extended_learning_image(domain, b).classify(classifier).select(['classification'], ['b1']); 
    return classified;


def skyboxAssist(domain, b):
    ''' Combine MODIS and RGBN to detect flood pixels.
    '''
    
    raise Exception('Algorithm is not ready yet!')
    
    #lowModisThresh = 300
    #
    ## Simple function implements the b2 - b1 difference method
    ## - Using two methods like this is a hack to get a call from modis_lake_measure.py to work!
    ##flood_diff_function1 = lambda x : x.select(['sur_refl_b02']).subtract(x.select(['sur_refl_b01']))
    #flood_diff_function2 = lambda x : x.sur_refl_b02.subtract(x.sur_refl_b01)
    #
    #
    ## Compute the difference statistics inside permanent water mask pixels
    #MODIS_RESOLUTION = 250 # Meters
    #water_mask       = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'])
    #floodDiff        = flood_diff_function2(domain.modis)
    #diffInWaterMask  = floodDiff.multiply(water_mask)
    #maskedMean       = diffInWaterMask.reduceRegion(ee.Reducer.mean(),   domain.bounds, MODIS_RESOLUTION)
    #maskedStdDev     = diffInWaterMask.reduceRegion(ee.Reducer.stdDev(), domain.bounds, MODIS_RESOLUTION)
    #
    #print 'Water mean = ' + str(maskedMean.getInfo())
    #print 'Water STD  = ' + str(maskedStdDev.getInfo())
    #
    ## TODO: Get this!
    #water_thresh = -8.0
    #land_thresh  =  10.0
    #
    ## Use the water mask statistics to compute a difference threshold, then find all pixels below the threshold.
    #pureWaterThreshold  = maskedMean.getInfo()['sur_refl_b02'] + water_thresh*(maskedStdDev.getInfo()['sur_refl_b02']);
    #print 'Pure water threshold == ' + str(pureWaterThreshold)
    #pureWaterPixels     = flood_diff_function2(domain.modis).lte(pureWaterThreshold)
    #
    #
    ##pureLandThreshold  = maskedMean.getInfo()['sur_refl_b02'] + land_thresh*(maskedStdDev.getInfo()['sur_refl_b02']);
    #pureLandThreshold = pureWaterThreshold + 3000
    #print 'Pure land threshold == ' + str(pureLandThreshold)
    #pureLandPixels     = flood_diff_function2(domain.modis).gt(pureLandThreshold)
    #
    #
    #addToMap(pureWaterPixels.mask(pureWaterPixels), {'min': 0, 'max': 1}, 'Pure Water',  False)
    #addToMap(pureLandPixels.mask(pureLandPixels),   {'min': 0, 'max': 1}, 'Pure Land',   False)
    #
    #
    ##TODO: Train on these pixels!
    #
    #
#    
#    def _create_learning_image(domain, b):
#    '''Set up features for the classifier to be trained on: [b2, b2/b1, b2/b1, NDVI, NDWI]'''
#    diff  = b['b2'].subtract(b['b1'])
#    ratio = b['b2'].divide(b['b1'])
#    return b['b1'].addBands(b['b2']).addBands(diff).addBands(ratio).addBands(b['NDVI']).addBands(b['NDWI'])
#
#def earth_engine_classifier(domain, b, classifier_name, extra_args={}):
#    '''Apply EE classifier tool using a ground truth image.'''
#    training_domain = domain.training_domain
#    training_image  = _create_learning_image(training_domain, compute_modis_indices(training_domain))
#    args = {
#            'image'             : training_image,
#            'subsampling'       : 0.5,
#            'training_image'    : training_domain.ground_truth,
#            'training_band'     : "b1",
#            'training_region'   : training_domain.bounds,
#            'max_classification': 2,
#            'classifier_name'   : classifier_name
#           }
#    args.update(extra_args)
#    classifier = ee.apply("TrainClassifier", args)  # Call the EE classifier
#    classified = ee.call("ClassifyImage", _create_learning_image(domain, b), classifier).select(['classification'], ['b1']); 
#    return classified;
#
#def cart(domain, b):
#    '''Classify using CART (Classification And Regression Tree)'''
#    return earth_engine_classifier(domain, b, 'Cart')
#    
    
    
    #return b['b2'].subtract(b['b1']).lte(500).select(['sur_refl_b02'], ['b1']) # Rename sur_refl_b02 to b1



# End of algorithm definitions
#=======================================================================================================
#=======================================================================================================





# Set up some information for each algorithm, used by the functions below.
__ALGORITHMS = {
        # Algorithm,    Display name,   Function name,    Fractional result?,    Display color
        EVI                : ('EVI',                     evi,            False, 'FF00FF'),
        XIAO               : ('XIAO',                    xiao,           False, 'FFFF00'),
        DIFFERENCE         : ('Difference',              modis_diff,     False, '00FFFF'),
        DIFF_LEARNED       : ('Diff. Learned',           diff_learned,   False, '00FFFF'),
        DARTMOUTH          : ('Dartmouth',               dartmouth,      False, '33CCFF'),
        DART_LEARNED       : ('Dartmouth Learned',       dart_learned,   False, '33CCFF'),
        FAI                : ('Floating Algae',          fai,            False, '3399FF'),
        FAI_LEARNED        : ('Floating Algae Learned',  fai_learned,    False, '3399FF'),
        CART               : ('CART',                    cart,           False, 'CC6600'),
        SVM                : ('SVM',                     svm,            False, 'FFAA33'),
        RANDOM_FORESTS     : ('Random Forests',          random_forests, False, 'CC33FF'),
        DNNS               : ('DNNS',                    dnns,           True,  '0000FF'),
        DNNS_DIFF          : ('DNNS Diff.',              dnns_diff,      True,  '0000FF'),
        DNNS_REVISED       : ('DNNS Revised',            dnns_revised,   False, '00FF00'),
        DNNS_DEM           : ('DNNS with DEM',           dnns_dem,       False, '9900FF'),
        DNNS_DIFF_DEM      : ('DNNS Diff with DEM',      dnns_diff_dem,  False, '9900FF'),
        DIFFERENCE_HISTORY : ('Difference with History', history_diff,   False, '0099FF'),
        DEM_THRESHOLD      : ('DEM Threshold',           dem_threshold,  False, 'FFCC33'),
        MARTINIS_TREE      : ('Martinis Tree',           martinis_tree,  False, 'CC0066'),
        SKYBOX_ASSIST      : ('Skybox Assist',           skyboxAssist,   False, '00CC66'),
        EXPERIMENTAL       : ('Experimental',            experimental,   False, '00FFFF')
}


def detect_flood(domain, algorithm):
    '''Run flood detection with a named algorithm in a given domain.'''
    try:
        approach = __ALGORITHMS[algorithm]
    except:
        return None
    return (approach[0], approach[1](domain, compute_modis_indices(domain)))

def get_algorithm_name(algorithm):
    '''Return the text name of an algorithm.'''
    try:
        return __ALGORITHMS[algorithm][0]
    except:
        return None

def get_algorithm_color(algorithm):
    '''Return the color assigned to an algorithm.'''
    try:
        return __ALGORITHMS[algorithm][3]
    except:
        return None

def is_algorithm_fractional(algorithm):
    '''Return True if the algorithm has a fractional output.'''
    try:
        return __ALGORITHMS[algorithm][2]
    except:
        return None

