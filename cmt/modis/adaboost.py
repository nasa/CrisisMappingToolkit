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

import learned_adaboost

import ee
import math

from cmt.domain import Domain
from cmt.modis.simple_modis_algorithms import *
from cmt.mapclient_qt import addToMap
from cmt.util.miscUtilities import safe_get_info
import cmt.modis.modis_utilities

"""
   Contains functions needed to implement an Adaboost algorithm using several of the
   simple MODIS classifiers.
"""


def _create_adaboost_learning_image(domain, b):
    '''Like _create_learning_image but using a lot of simple classifiers to feed into Adaboost'''
    
    # A large set of MODIS band configurations, each is assigned a unique band name for reference.
    a = b['b1'].select(['sur_refl_b01'],                                                 ['b1'           ])
    a = a.addBands(b['b2'].select(['sur_refl_b02'],                                      ['b2'           ]))
    #a = a.addBands(b['b3'].select(['sur_refl_b03'],                                      ['b3'           ]))
    #a = a.addBands(b['b4'].select(['sur_refl_b04'],                                      ['b4'           ]))
    #a = a.addBands(b['b5'].select(['sur_refl_b05'],                                      ['b5'           ]))
    #a = a.addBands(b['b6'].select(['sur_refl_b06'],                                      ['b6'           ]))
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
    
    # If available, try adding Landsat data
    try:
        landsat_sensor = domain.get_landsat()
        added = ['blue', 'green', 'red', 'nir', 'swir1', 'temp', 'swir2']
        a = a.addBands(landsat_sensor.image.select(added))
        print('Added Landsat to Adaboost!')
    except: # No Landsat data is present
        pass
    
    # If available, try adding radar data
    try:
        # Add all of the bands from the radar sensor
        # - All of the input training images need to have the same bands available!
        radar_sensor = domain.get_radar()
        a = a.addBands(radar_sensor.image) 
        print('Added Radar to Adaboost!')
    except: # No radar data is present
        pass

    # If available, try adding Skybox data
    try:
        try: # The Skybox data can be in one of two names
            skybox_sensor = domain.skybox
        except:
            skybox_sensor = domain.skybox_nir
        
        # Add all Skybox bands
        a = a.addBands(skybox_sensor.image)

        # Add an additional texture band
        rgbBands    = skybox_sensor.Red.addBands(skybox_sensor.Green).addBands(skybox_sensor.Blue)
        grayBand    = rgbBands.select('Red').add(rgbBands.select('Green')).add(rgbBands.select('Blue')).divide(ee.Image(3.0)).uint16()
        textureRaw  = grayBand.glcmTexture()
        bandList    = safe_get_info(textureRaw)['bands']
        bandName    = [x['id'] for x in bandList if 'idm' in x['id']]
        texture     = textureRaw.select(bandName).convolve(ee.Kernel.square(5, 'pixels'))
        a = a.addBands(texture)
        
    except: # No Skybox data is present
        pass
    
    return a


def _find_adaboost_optimal_threshold(domains, images, truths, band_name, weights, splits):
    '''Binary search to find best threshold for this band'''
    
    EVAL_RESOLUTION = 250
    choices = []
    for i in range(len(splits) - 1):
        choices.append((splits[i] + splits[i+1]) / 2)
        
    domain_range = range(len(domains))
    best         = None
    best_value   = None
    for k in range(len(choices)):
        # Pick a threshold and count how many pixels fall under it across all the input images
        c = choices[k]
        errors = [safe_get_info(weights[i].multiply(images[i].select(band_name).lte(c).neq(truths[i])).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION, 'EPSG:4326'))['constant']
                       for i in range(len(images))]
        error  = sum(errors)
        #threshold_sums = [safe_get_info(weights[i].mask(images[i].select(band_name).lte(c)).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in domain_range]
        #flood_and_threshold_sum = sum(threshold_sums)
        #
        ##ts         = [truths[i].multiply(weights[i]).divide(flood_and_threshold_sum).mask(images[i].select(band_name).lte(c))              for i in domain_range]
        ##entropies1 = [-safe_get_info(ts[i].multiply(ts[i].log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'] for i in domain_range]# H(Y | X <= c)
        ##ts         = [truths[i].multiply(weights[i]).divide(1 - flood_and_threshold_sum).mask(images[i].select(band_name).gt(c))           for i in domain_range]
        ##entropies2 = [-safe_get_info(ts[i].multiply(ts[i].log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'] for i in domain_range]# H(Y | X > c)
        #
        ## Compute the sums of two entropy measures across all images
        #entropies1 = entropies2 = []
        #for i in domain_range:
        #    band_image     = images[i].select(band_name)
        #    weighted_truth = truths[i].multiply(weights[i])
        #    ts1            = weighted_truth.divide(    flood_and_threshold_sum).mask(band_image.lte(c)) # <= threshold
        #    ts2            = weighted_truth.divide(1 - flood_and_threshold_sum).mask(band_image.gt( c)) # >  threshold
        #    entropies1.append(-safe_get_info(ts1.multiply(ts1.log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'])# H(Y | X <= c)
        #    entropies2.append(-safe_get_info(ts2.multiply(ts2.log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'])# H(Y | X > c)
        #entropy1 = sum(entropies1)
        #entropy2 = sum(entropies2)
        #
        ## Compute the gain for this threshold choice
        #gain = (entropy1 * (    flood_and_threshold_sum)+
        #        entropy2 * (1 - flood_and_threshold_sum))
        #print 'c = %f, error = %f' % (c, error)
        if (best == None) or abs(0.5 - error) > abs(0.5 - best_value): # Record the maximum gain
            best       = k
            best_value = error
    
    # TODO: What is causing this inaccuracy?
    if best_value > 0.99:
        best_value = 0.99
    
    # ??
    return (choices[best], best + 1, best_value)

def apply_classifier(image, band, threshold):
    '''Apply LTE threshold and convert to -1 / 1 (Adaboost requires this)'''
    return image.select(band).lte(threshold).multiply(2).subtract(1)

def get_adaboost_sum(domain, b, classifier = None):
    if classifier == None:
        classifier = learned_adaboost.modis_classifiers['default']
        
    test_image = _create_adaboost_learning_image(domain, b)
    total = ee.Image(0).select(['constant'], ['b1'])
    for c in classifier:
      total = total.add(test_image.select(c[0]).lte(c[1]).multiply(2).subtract(1).multiply(c[2]))
    return total

def adaboost(domain, b, classifier = None):
    '''Run Adaboost classifier'''
    total = get_adaboost_sum(domain, b, classifier)
    return total.gte(-1.0) # Just threshold the results at zero (equal chance of flood / not flood)


def adaboost_radar(domain):
    '''Run Adaboost classifier trained for a radar data set'''

    classifier = learned_adaboost.radar_classifiers['malawi']
    
    b = modis_utilities.compute_modis_indices(domain)
    total = get_adaboost_sum(domain, b, classifier)
    return total.gte(-1.0) # Just threshold the results at zero (equal chance of flood / not flood)

def adaboost_dem(domain, b, classifier = None):
    
    # Get raw adaboost output
    total = get_adaboost_sum(domain, b, classifier)
    #addToMap(total, {'min': -10, 'max': 10}, 'raw ADA', False)
    
    # Convert this range of values into a zero to one probability scale
    #MIN_SUM = -3.5 # These numbers are a pretty good probability conversion, but it turns out
    #MAX_SUM =  1.0 #  that probability does not make a good input to apply_dem().
    
    MIN_SUM =-1.0 # These numbers are tuned to get better results
    MAX_SUM = 0.0
    
    val_range = MAX_SUM - MIN_SUM
    
    fraction = total.subtract(ee.Image(MIN_SUM)).divide(ee.Image(val_range)).clamp(0.0, 1.0)
    return modis_utilities.apply_dem(domain, fraction)

def __compute_threshold_ranges(training_domains, training_images, water_masks, bands):
    '''For each band, find lowest and highest fixed percentiles among the training domains.'''
    LOW_PERCENTILE  = 20
    HIGH_PERCENTILE = 100
    EVAL_RESOLUTION = 250
    
    band_splits = dict()
    for band_name in bands: # Loop through each band (weak classifier input)
        split = None
        print('Computing threshold ranges for: ' + band_name)
      
        mean = 0
        for i in range(len(training_domains)): # Loop through all input domains
            # Compute the low and high percentiles for the data in the training image
            masked_input_band = training_images[i].select(band_name).mask(water_masks[i])
            ret = safe_get_info(masked_input_band.reduceRegion(ee.Reducer.percentile([LOW_PERCENTILE, HIGH_PERCENTILE], ['s', 'b']), training_domains[i].bounds, EVAL_RESOLUTION))
            s   = [ret[band_name + '_s'], ret[band_name + '_b']] # Extract the two output values
            mean += modis_utilities.compute_binary_threshold(training_images[i].select([band_name], ['b1']), water_masks[i], training_domains[i].bounds)
            
            if split == None: # True for the first training domain
                split = s
            else: # Track the minimum and maximum percentiles for this band
                split[0] = min(split[0], s[0])
                split[1] = max(split[1], s[1])
        mean = mean / len(training_domains)
            
        # For this band: bound by lowest percentile and maximum percentile, start by evaluating mean
        band_splits[band_name] = [split[0], split[0] + (mean - split[0]) / 2, mean + (split[1] - mean) / 2, split[1]]
    return band_splits

def adaboost_learn(ignored=None, ignored2=None):
    '''Train Adaboost classifier'''
    
    EVAL_RESOLUTION = 250

    # Learn this many weak classifiers
    NUM_CLASSIFIERS_TO_TRAIN = 50

    # Load inputs for this domain and preprocess
    # - Kashmore does not have a good unflooded comparison location so it is left out of the training.
    #all_problems      = ['kashmore_2010_8.xml', 'mississippi_2011_5.xml', 'mississippi_2011_6.xml', 'new_orleans_2005_9.xml', 'sf_bay_area_2011_4.xml']
    #all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
    #training_domains  = [domain.unflooded_domain for domain in all_domains[:-1]] + [all_domains[-1]] # SF is unflooded
    
    # This is a cleaned up set where all the permanent water masks are known to be decent.
    all_problems      = ['unflooded_mississippi_2010.xml', 'unflooded_new_orleans_2004.xml', 'sf_bay_area_2011_4.xml', 'unflooded_bosnia_2013.xml']
    #all_problems.extend(['arkansas_city_2011_5.xml', 'baghlan_south_2014_6.xml', 
    #                     'bosnia_west_2014_5.xml', 'kashmore_north_2010_8.xml', 'slidell_2005_9.xml'])
    #all_problems      = ['unflooded_mississippi_2010_5.xml']
    all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
    
    # Try testing on radar domains
    #all_problems   = ['rome.xml',]
    #                  #'malawi_2015_1.xml',]
    #                  #'mississippi.xml']
    #all_domains    = [Domain('config/domains/sentinel1/rome.xml'),]
    #                  #Domain('config/domains/sentinel1/malawi_2015_1.xml'),
    #                  #Domain('config/domains/uavsar/mississippi.xml')]
    #training_domains  = [domain.training_domain for domain in all_domains]
    
    ## Try testing on Skybox images
    #all_problems     = [#'gloucester_2014_10.xml',] # TODO: Need dense training data for the other images!!!
    #                    #'new_bedford_2014_10.xml',
    #                    #'sumatra_2014_10.xml',]
    #                    'malawi_2015.xml',] 
    #all_domains      = [Domain('config/domains/skybox/' + d) for d in all_problems]
    #training_domains = [domain.training_domain for domain in all_domains]
    
    ## Add a bunch of lakes to the training data
    #lake_problems = ['Amistad_Reservoir/Amistad_Reservoir_2014-07-01_train.xml',
    #                 'Cascade_Reservoir/Cascade_Reservoir_2014-09-01_train.xml',
    #                 'Edmund/Edmund_2014-07-01_train.xml',
    #                 'Hulun/Hulun_2014-07-01_train.xml',
    #                 'Keeley/Keeley_2014-06-01_train.xml',
    #                 'Lake_Mead/Lake_Mead_2014-09-01_train.xml',
    #                 'Miguel_Aleman/Miguel_Aleman_2014-08-01_train.xml',
    #                 'Oneida_Lake/Oneida_Lake_2014-06-01_train.xml',
    #                 'Quesnel/Quesnel_2014-08-01_train.xml',
    #                 'Shuswap/Shuswap_2014-08-01_train.xml',
    #                 'Trikhonis/Trikhonis_2014-07-01_train.xml',
    #                 'Pickwick_Lake/Pickwick_Lake_2014-07-01_train.xml',
    #                 'Rogoaguado/Rogoaguado_2014-08-01_train.xml',
    #                 'Zapatosa/Zapatosa_2014-09-01_train.xml']
    #lake_domains  = [Domain('/home/smcmich1/data/Floods/lakeStudy/' + d) for d in lake_problems]
    #all_problems += lake_problems
    #all_domains  += lake_domains
    
    #all_problems      = ['unflooded_mississippi_2010.xml']
    #all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]

    #all_problems      = ['sf_bay_area_2011_4.xml']
    #all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
    #
    #all_problems      = ['unflooded_bosnia_2013.xml']
    #all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
    #
    #all_problems      = ['unflooded_new_orleans_2004.xml']
    #all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
    
    training_domains  = all_domains
    
    water_masks = [modis_utilities.get_permanent_water_mask() for d in training_domains]
    for i in range(len(all_domains)):
        if all_domains[i].ground_truth != None:
            water_masks[i] = all_domains[i].ground_truth
    #water_masks = [d.ground_truth for d in training_domains] # Manual mask   
    
    training_images = [_create_adaboost_learning_image(d, modis_utilities.compute_modis_indices(d)) for d in training_domains]
    
    # add pixels in flood permanent water masks to training
    #training_domains.extend(all_domains)
    #water_masks.extend([get_permanent_water_mask() for d in all_domains])
    #training_images.append([_create_adaboost_learning_image(domain, compute_modis_indices(domain)).mask(get_permanent_water_mask()) for domain in all_domains])
    
    transformed_masks = [water_mask.multiply(2).subtract(1) for water_mask in water_masks]

    bands             = safe_get_info(training_images[0].bandNames())
    print('Computing threshold ranges.')
    band_splits = __compute_threshold_ranges(training_domains, training_images, water_masks, bands)
    counts = [safe_get_info(training_images[i].select('b1').reduceRegion(ee.Reducer.count(), training_domains[i].bounds, 250))['b1'] for i in range(len(training_images))]
    count = sum(counts)
    weights = [ee.Image(1.0 / count) for i in training_images] # Each input pixel in the training images has an equal weight
    
    # Initialize for pre-existing partially trained classifier
    full_classifier = []
    for (c, t, alpha) in full_classifier:
        band_splits[c].append(t)
        band_splits[c] = sorted(band_splits[c])
        total = 0
        for i in range(len(training_images)):
            weights[i] = weights[i].multiply(apply_classifier(training_images[i], c, t).multiply(transformed_masks[i]).multiply(-alpha).exp())
            total += safe_get_info(weights[i].reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant']
        for i in range(len(training_images)):
            weights[i] = weights[i].divide(total)
    
    ## Apply weak classifiers to the input test image
    #test_image = _create_adaboost_learning_image(domain, b)
    
    
    while len(full_classifier) < NUM_CLASSIFIERS_TO_TRAIN:
        best = None
        for band_name in bands: # For each weak classifier
            # Find the best threshold that we can choose
            (threshold, ind, error) = _find_adaboost_optimal_threshold(training_domains, training_images, water_masks, band_name, weights, band_splits[band_name])
            
            # Compute the sum of weighted classification errors across all of the training domains using this threshold
            #errors = [safe_get_info(weights[i].multiply(training_images[i].select(band_name).lte(threshold).neq(water_masks[i])).reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in range(len(training_images))]
            #error  = sum(errors)
            print('%s found threshold %g with error %g' % (band_name, threshold, error))
            
            # Record the band/threshold combination with the highest abs(error)
            if (best == None) or (abs(0.5 - error) > abs(0.5 - best[0])): # Classifiers that are always wrong are also good with negative alpha
                best = (error, band_name, threshold, ind)
        
        # add an additional split point to search between for thresholds
        band_splits[best[1]].insert(best[3], best[2])
      
        print('---> Using %s < %g. Error %g.' % (best[1], best[2], best[0]))
        alpha      = 0.5 * math.log((1 - best[0]) / best[0])
        classifier = (best[1], best[2], alpha)
        full_classifier.append(classifier)
        print('---> Now have %d out of %d classifiers.' % (len(full_classifier), NUM_CLASSIFIERS_TO_TRAIN))
        
        # update the weights
        weights = [weights[i].multiply(apply_classifier(training_images[i], classifier[0], classifier[1]).multiply(transformed_masks[i]).multiply(-alpha).exp()) for i in range(len(training_images))]
        totals  = [safe_get_info(weights[i].reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in range(len(training_images))]
        total   = sum(totals)
        weights = [w.divide(total) for w in weights]
        print(full_classifier)

#
#import modis_utilities
#import pickle
#def adaboost_dem_learn(classifier = None):
#    '''Train Adaboost classifier'''
#    
#    EVAL_RESOLUTION = 250
#
#    # Load inputs for this domain and preprocess
#    #all_problems      = ['kashmore_2010_8.xml', 'mississippi_2011_5.xml', 'mississippi_2011_6.xml', 'new_orleans_2005_9.xml', 'sf_bay_area_2011_4.xml']
#    all_problems      = ['mississippi_2011_6.xml', 'new_orleans_2005_9.xml', 'sf_bay_area_2011_4.xml']
#    all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
#    training_domains  = [domain.unflooded_domain for domain in all_domains[:-1]] + [all_domains[-1]] # SF is unflooded
#    water_masks       = [modis_utilities.get_permanent_water_mask() for d in training_domains]
#    
#    THRESHOLD_INTERVAL =  0.5
#    MIN_THRESHOLD      = -5.0
#    MAX_THRESHOLD      =  2.0
#    
#    print 'Computing thresholds'
#    
#    results = []
#        
#    # Loop through each of the raw result images
#    for (truth_image, train_domain, name) in zip(water_masks, training_domains, all_problems):
#
#        truth_image = truth_image.mask(ee.Image(1))
#
#        # Apply the Adaboost computation to each training image and get the raw results
#        b = modis_utilities.compute_modis_indices(train_domain)
#        sum_image = get_adaboost_sum(train_domain, b, classifier)
#        #addToMap(sum_image, {'min': -10, 'max': 10}, 'raw ADA', False)
#        #addToMap(truth_image, {'min': 0, 'max': 1}, 'truth', False)
#        print '================================'
#        print name
#
#        #pickle.dump( truth_image, open( "truth.pickle", "wb" ) )
#        
#        results_list = []
#        
#        # For each threshold level above zero, how likely is the pixel to be actually flooded?
#        curr_threshold = MIN_THRESHOLD
#        percentage = 0
#        #while percentage < TARGET_PERCENTAGE:
#        while curr_threshold <= MAX_THRESHOLD:
#            
#            curr_results = sum_image.gte(curr_threshold)
#            
#            curr_results = curr_results.mask(ee.Image(1))
#            
#            #addToMap(curr_results, {'min': 0, 'max': 1}, str(curr_threshold), False)
#            
#            #addToMap(curr_results.multiply(truth_image), {'min': 0, 'max': 1}, 'mult', False)
#            
#            sum_correct  = safe_get_info(curr_results.multiply(truth_image).reduceRegion(ee.Reducer.sum(), train_domain.bounds, EVAL_RESOLUTION, 'EPSG:4326'))['b1']
#            sum_total    = safe_get_info(curr_results.reduceRegion(ee.Reducer.sum(), train_domain.bounds, EVAL_RESOLUTION, 'EPSG:4326'))['b1']
#            #print sum_correct
#            if sum_total > 0:
#                percentage   = sum_correct / sum_total
#                print str(curr_threshold) +': '+ str(sum_total) + ' --> '+ str(percentage)
#                results_list.append(percentage)
#                #pickle.dump( curr_results, open( "detect.pickle", "wb" ) )
#            else: # Time to break out of the loop
#                results.append(results_list)
#                break
#            curr_threshold += THRESHOLD_INTERVAL
#        else:
#            results.append(results_list)
#        
#    logFile = open('adaboostProbabilityLog.txt', 'w')
#    logFile.write('Threshold, Miss_5, Miss_6, NO, SF\n')
#    for r in range(15):
#        logFile.write(str(r*THRESHOLD_INTERVAL + MIN_THRESHOLD))
#        for i in range(3):
#            logFile.write(str(results[i][r]) + ', ')
#        logFile.write(str(results[3][r]) + '\n')
#    logFile.close()
#        
#        
#        #raise Exception('DEBUG')
#
##        # For each threshold level below zero, how likely is the pixel to be actually dry?
        



