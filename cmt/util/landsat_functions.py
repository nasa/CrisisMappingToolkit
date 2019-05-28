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
from cmt.util.miscUtilities import safe_get_info


def get_landsat_name(image):
    '''Returns the name of the landsat sensor that produced the image.'''
    text = image.get('METADATA_FILE_NAME').getInfo()
    if 'LE5' in text:
        return 'landsat5'
    if 'LE7' in text:
        return 'landsat7'
    if 'LC8' in text:
        return 'landsat8'
    print(text)
    raise Exception('Unrecognized Landsat image!')

def rename_landsat_bands(collection, collectionName):
    '''Selects and renames the landsat bands we are interested in.  
       Works with any Landsat satellite.'''

    # The list of bands we are interested in
    # - temp = temperature.  Landsat8 splits this into two bands, we use the first of these.
    # - The panchromatic band might be nice but it is only on Landsat7/8
    LANDSAT_BANDS_OF_INTEREST = ee.List(['blue', 'green', 'red', 'nir', 'swir1', 'temp', 'swir2'])

    # The indices where these bands are found in the Landsat satellites
    LANDSAT_BAND_INDICES = {'L8': ee.List([1, 2, 3, 4, 5, 9, 6]),
                            'L7': ee.List([0, 1, 2, 3, 4, 5, 7]),
                            'L5': ee.List([0, 1, 2, 3, 4, 5, 6])}

    landsat_index = 'L8';
    if '5' in collectionName:
        landsat_index = 'L5'
    if '7' in collectionName:
        landsat_index = 'L7'

    return collection.select(LANDSAT_BAND_INDICES[landsat_index], LANDSAT_BANDS_OF_INTEREST)


def compute_fai(image):
    '''Compute the Floating Algae Index'''
    fai_band = image.expression('(b("nir") + (b("red") + (b("swir1") - b("red"))*(170/990)))')
    return fai_band

    #visparams = {'bands': ['fai'],
    #             "min": [0.02],
    #             "max": [.3]
    #             }

def compute_ndti(image):
    '''Compute NDTI (Turbidity) index'''
    ndti = image.normalizedDifference(['red', 'green']).select([0], ['ndti'])
    return ndti

    #visparams = {'bands': ['ndti'],
    #             "min": [-0.35],
    #             "max": [0]
    #             }


def expression_and_rescale(img, exp, thresholds):
    '''Apply an expression and rescale the result'''
    rng = thresholds[1] - thresholds[0]
    return img.expression(exp, {'img': img}).subtract(thresholds[0]).divide(rng)

def detect_clouds(img):
    '''Compute several indicators of cloudiness and take the minimum of them.'''
    
    score = ee.Image(1.0)
    # Clouds are reasonably bright in the blue band.
    score = score.min(expression_and_rescale(img, 'img.blue', [0.1, 0.3]))

    # Clouds are reasonably bright in all visible bands.
    score = score.min(expression_and_rescale(img, 'img.red + img.green + img.blue', [0.2, 0.8]))

    # Clouds are reasonably bright in all infrared bands.
    score = score.min(expression_and_rescale(img, 'img.nir + img.swir1 + img.swir2', [0.3, 0.8]))

    # Clouds are reasonably cool in temperature.
    score = score.min(expression_and_rescale(img, 'img.temp', [300, 290]))

    # However, clouds are not snow.
    ndsi  = img.normalizedDifference(['green', 'swir1'])
    score = score.min(expression_and_rescale(ndsi, 'img', [0.8, 0.6]))

    CLOUD_THRESHOLD = 0.35
    return score.gt(CLOUD_THRESHOLD)

def getCloudPercentage(image, region):
    '''Estimates the cloud cover percentage in a Landsat image'''
    
    # The function will attempt the calculation in these ranges
    # - Native Landsat resolution is 30
    MIN_RESOLUTION = 60
    MAX_RESOLUTION = 1000
    
    resolution = MIN_RESOLUTION
    while True:
        try:
            oneMask     = ee.Image(1.0)
            cloudScore  = detect_clouds(image)
            areaCount   = oneMask.reduceRegion(  ee.Reducer.sum(),  region, resolution)
            cloudCount  = cloudScore.reduceRegion(ee.Reducer.sum(), region, resolution)
            percentage  = safe_get_info(cloudCount)['constant'] / safe_get_info(areaCount)['constant']
            return percentage
        except Exception as e:
            # Keep trying with lower resolution until we succeed
            resolution = 2*resolution
            if resolution > MAX_RESOLUTION:
                raise e

def compute_water_threshold(sun_angle):
    '''Function to scale water detection sensitivity based on sun angle.'''
    waterThresh = ((.6 / 54) * (62 - sun_angle)) + .05
    return waterThresh

def detect_water(image):
    
    shadowSumBands = ee.List(['nir','swir1','swir2'])# Bands for shadow masking
    # Compute several indicators of water and take the minimum of them.
    score = ee.Image(1.0)

    # Set up some params
    darkBands = ['green','red','nir','swir2','swir1']# ,'nir','swir1','swir2']
    brightBand = 'blue'

    # Water tends to be dark
    shadowSum = image.select(shadowSumBands).reduce(ee.Reducer.sum())
    shadowSum = expression_and_rescale(shadowSum,'img',[0.35,0.2]).clamp(0,1)
    score     = score.min(shadowSum)

    # It also tends to be relatively bright in the blue band
    mean  = image.select(darkBands).reduce(ee.Reducer.mean())
    std   = image.select(darkBands).reduce(ee.Reducer.stdDev())
    z     = (image.select([brightBand]).subtract(std)).divide(mean)
    z     = expression_and_rescale(z,'img',[0,1]).clamp(0,1)
    score = score.min(z)

    # Water is at or above freezing

    score = score.min(expression_and_rescale(image, 'img.temp', [273, 275]))

    # Water is nigh in ndsi (aka mndwi)
    ndsi = image.normalizedDifference(['green', 'swir1'])
    ndsi = expression_and_rescale(ndsi, 'img', [0.3, 0.8])

    # Go ahead and restrict the score to this range.
    score = score.min(ndsi).clamp(0,1)

    # Select water pixels from the raw score

    try:    
        sunElevation = image.get('SUN_ELEVATION').getInfo()
    except: # Default guess if sun elevation is not provided
        sunElevation = 45
    waterThresh = compute_water_threshold(sunElevation)
    
    clouds = detect_clouds(image)
    # TODO: Detect snow also

    water = score.gt(waterThresh).And(clouds.Not())
    
    return water


