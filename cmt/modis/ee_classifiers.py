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

from cmt.mapclient_qt import addToMap
from cmt.util.miscUtilities import safe_get_info
from cmt.modis.simple_modis_algorithms import *
from modis_utilities import *

'''
Contains algorithms and tools for using the built-in Earth Engine classifiers.
'''


#==============================================================


def _create_learning_image(domain, b):
    '''Set up features for the classifier to be trained on'''

    outputBands = _get_modis_learning_bands(domain, b) # Get the standard set of MODIS learning bands
    #outputBands = _get_extensive_modis_learning_bands(domain, b) # Get the standard set of MODIS learning bands
    
    
    # Try to add a DEM
    try:
        dem = domain.get_dem().image
        outputBands.addBands(dem)
        #outputBands = dem
    except AttributeError:
        pass # Suppress error if there is no DEM data
    
    # Try to add Skybox RGB info (NIR is handled separately because not all Skybox images have it)
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
        bandList    = safe_get_info(texture2Raw)['bands']
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


def _get_modis_learning_bands(domain, b):
    '''Set up features for the classifier to be trained on: [b2, b2/b1, b2/b1, NDVI, NDWI]'''
    diff        = b['b2'].subtract(b['b1'])
    ratio       = b['b2'].divide(b['b1'])
    modisBands  = b['b1'].addBands(b['b2']).addBands(diff).addBands(ratio).addBands(b['NDVI']).addBands(b['NDWI'])
    return modisBands


def _get_extensive_modis_learning_bands(domain, b):
    '''Like _get_modis_learning_bands but adding a lot of simple classifiers'''
    
    #a = get_diff(b).select(['b1'], ['b1'])
    a = b['b1'].select(['sur_refl_b01'],                                                 ['b1'           ])
    a = a.addBands(b['b2'].select(['sur_refl_b02'],                                      ['b2'           ]))
    a = a.addBands(b['b2'].divide(b['b1']).select(['sur_refl_b02'],                      ['ratio'        ]))
    a = a.addBands(b['LSWI'].subtract(b['NDVI']).subtract(0.05).select(['sur_refl_b02'], ['LSWIminusNDVI']))
    a = a.addBands(b['LSWI'].subtract(b['EVI']).subtract(0.05).select(['sur_refl_b02'],  ['LSWIminusEVI' ]))
    a = a.addBands(b['EVI'].subtract(0.3).select(['sur_refl_b02'],                       ['EVI'          ]))
    a = a.addBands(b['LSWI'].select(['sur_refl_b02'],                                    ['LSWI'         ]))
    a = a.addBands(b['NDVI'].select(['sur_refl_b02'],                                    ['NDVI'         ]))
    a = a.addBands(b['NDWI'].select(['sur_refl_b01'],                                    ['NDWI'         ]))
    a = a.addBands(get_diff(b).select(['b1'],                                            ['diff'         ]))
    a = a.addBands(get_fai(b).select(['b1'],                                             ['fai'          ]))
    a = a.addBands(get_dartmouth(b).select(['b1'],                                       ['dartmouth'    ]))
    a = a.addBands(get_mod_ndwi(b).select(['b1'],                                        ['MNDWI'        ]))
    return a

def earth_engine_classifier(domain, b, classifier_name, extra_args={}):
    '''Apply EE classifier tool using a ground truth image.'''
    
    # Training requires a training image plus either ground truth or training features.
    training_domain = None
    #if domain.training_domain:
    training_domain = domain.training_domain
    #elif domain.unflooded_domain:
    #training_domain = domain.unflooded_domain
    if not training_domain:
        raise Exception('Cannot run classifier algorithm without a training domain!')

    training_image  = _create_learning_image(training_domain, compute_modis_indices(training_domain))
    if training_domain.training_features:
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
    else: # Use the permanent water mask
        args = {
                'training_image'    : get_permanent_water_mask(),
                'training_band'     : "b1",
                'training_region'   : training_domain.bounds
               }
    common_args = {
                   'subsampling'       : 0.2, # TODO: Reduce this on failure?
                   'max_classification': 2,
                   'classifier_mode' : 'classification',
                   'classifier_name'   : classifier_name
                  }
    args.update(common_args)
    args.update(extra_args)
    classifier = training_image.trainClassifier(**args)  # Call the EE classifier
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

