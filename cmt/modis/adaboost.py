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
from cmt.util.evaluation import safe_get_info


"""
   Contains functions needed to implement an Adaboost algorithm using several of the
   simple MODIS classifiers.
"""


def _create_adaboost_learning_image(domain, b):
    '''Like _create_learning_image but using a lot of simple classifiers to feed into Adaboost'''
    
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


def _find_adaboost_optimal_threshold(domains, images, truths, band_name, weights, splits):
    '''Binary search to find best threshold for this band'''
    
    EVAL_RESOLUTION = 250
    choices = []
    for i in range(len(splits) - 1):
        choices.append((splits[i] + splits[i+1]) / 2)
        
    domain_range = range(len(domains))
    best         = None
    best_value   = None
    for j in range(len(choices)):
        # Pick a threshold and count how many pixels fall under it across all the input images
        c = choices[i]
        threshold_sums = [safe_get_info(weights[i].mask(images[i].select(band_name).lte(c)).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in domain_range]
        flood_and_threshold_sum = sum(threshold_sums)
        
        #ts         = [truths[i].multiply(weights[i]).divide(flood_and_threshold_sum).mask(images[i].select(band_name).lte(c))              for i in domain_range]
        #entropies1 = [-safe_get_info(ts[i].multiply(ts[i].log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'] for i in domain_range]# H(Y | X <= c)
        #ts         = [truths[i].multiply(weights[i]).divide(1 - flood_and_threshold_sum).mask(images[i].select(band_name).gt(c))           for i in domain_range]
        #entropies2 = [-safe_get_info(ts[i].multiply(ts[i].log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'] for i in domain_range]# H(Y | X > c)
        
        # Compute the sums of two entropy measures across all images
        entropies1 = entropies2 = []
        for i in domain_range:
            band_image     = images[i].select(band_name)
            weighted_truth = truths[i].multiply(weights[i])
            ts1            = weighted_truth.divide(    flood_and_threshold_sum).mask(band_image.lte(c)) # <= threshold
            ts2            = weighted_truth.divide(1 - flood_and_threshold_sum).mask(band_image.gt( c)) # >  threshold
            entropies1.append(-safe_get_info(ts1.multiply(ts1.log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'])# H(Y | X <= c)
            entropies2.append(-safe_get_info(ts2.multiply(ts2.log()).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['b1'])# H(Y | X > c)
        entropy1 = sum(entropies1)
        entropy2 = sum(entropies2)
        
        # Compute the gain for this threshold choice
        gain = (entropy1 * (    flood_and_threshold_sum)+
                entropy2 * (1 - flood_and_threshold_sum))
        print c, gain, flood_and_threshold_sum, entropy1, entropy2
        
        if (best == None) or (gain < best_value): # Record the maximum gain
            best       = j
            best_value = gain
    
    # ??
    return (choices[best], best + 1, best_value)

def apply_classifier(image, band, threshold):
    '''Apply LTE threshold and convert to -1 / 1 (Adaboost requires this)'''
    return image.select(band).lte(threshold).multiply(2).subtract(1)

def adaboost(domain, b, classifier = None):
    '''Run Adaboost classifier'''
    
    # These are a set of known good computed values:  (Algorithm, Detection threshold, Weight)
    # - These were trained on non-flooded MODIS images.
    if classifier == None:
        classifier = [(u'dartmouth',        0.6746916226821668,   0.9545783139872039),
                      (u'b1',             817.4631578947368,     -0.23442294410851128),
                      (u'ratio',            3.4957167876866304,  -0.20613698036794326),
                      (u'LSWIminusNDVI',   -0.18319560006184613, -0.20191291743554216),
                      (u'EVI',              1.2912227247420454,  -0.11175138956289551),
                      (u'dartmouth',        0.7919185558963437,   0.09587432900090082),
                      (u'diff',           971.2916666666666,     -0.10554565939141827),
                      (u'LSWIminusEVI',    -0.06318265061389294, -0.09533981402236558),
                      (u'LSWI',             0.18809171182282547,  0.07057035145131643),
                      (u'LSWI',             0.29177473609507737, -0.10405606800405826),
                      (u'b1',             639.7947368421053,      0.04609306857534169),
                      (u'b1',             550.9605263157895,      0.08536825945329486),
                      (u'LSWI',             0.23993322395895142,  0.0686895188858698),
                      (u'LSWIminusNDVI',   -0.2895140352747048,  -0.05149197092271741),
                      (u'b2',            1713.761111111111, -     0.05044107229585143),
                      (u'b2',            2147.325,                0.08569272886858223),
                      (u'dartmouth',        0.7333050892892552,  -0.0658128496826074),
                      (u'LSWI',             0.21401246789088846,  0.047469648515471446),
                      (u'LSWIminusNDVI',   -0.34267325288113415, -0.0402902367049306),
                      (u'ratio',            2.382344524011886,   -0.03795571511345347),
                      (u'fai',            611.5782694327731,      0.03742837135530962),
                      (u'ratio',            2.9390306558492583,  -0.03454143179044789),
                      (u'fai',            925.5945969012605,      0.05054908824123665),
                      (u'diff',          1336.7708333333333,     -0.042539270450854885),
                      (u'LSWIminusNDVI',   -0.3160936440779195,  -0.03518287810525178)]
        
    test_image = _create_adaboost_learning_image(domain, b)
    total      = ee.Image(0).select(['constant'], ['b1'])
    for c in classifier:
        total = total.add(test_image.select(c[0]).lte(c[1]).multiply(2).subtract(1).multiply(c[2]))
    return total.gte(0.0)

def _compute_threshold_ranges(training_domains, training_images, water_masks, bands):
    '''For each band, find lowest and highest fixed percentiles among the training domains.'''
    LOW_PERCENTILE  = 20
    HIGH_PERCENTILE = 80
    EVAL_RESOLUTION = 250
    
    band_splits = dict()
    for band_name in bands: # Loop through each band (weak classifier input)
        split = None
        print 'Computing threshold ranges for: ' + band_name
      
        for i in range(len(training_domains)): # Loop through all input domains
            # Compute the low and high percentiles for the data in the training image
            masked_input_band = training_images[i].select(band_name).mask(water_masks[i])
            ret = safe_get_info(masked_input_band.reduceRegion(ee.Reducer.percentile([LOW_PERCENTILE, HIGH_PERCENTILE], ['s', 'b']), training_domains[i].bounds, EVAL_RESOLUTION))
            s   = [ret[band_name + '_s'], ret[band_name + '_b']] # Extract the two output values
            
            if split == None: # True for the first training domain
                split = s
            else: # Track the minimum and maximum percentiles for this band
                split[0] = min(split[0], s[0])
                split[1] = max(split[1], s[1])
            
        # For this band: [lowest 20th percentile, highest 80th percentile, 80th + the diff between them]
        band_splits[band_name] = [split[0],  split[1],  split[1] + (split[1] - split[0])] 
    return band_splits

def adaboost_learn(domain, b):
    '''Train Adaboost classifier'''
    
    EVAL_RESOLUTION = 250
    
    # Load inputs for this domain and preprocess
    print 'Preprocessing input data...'
    training_domains  = [domain.unflooded_domain] # The same location at an earlier date with normal water levels.
    water_masks       = [get_permanent_water_mask()]
    transformed_masks = [water_mask.multiply(2).subtract(1) for water_mask in water_masks] # Convert from 0/1 to +/-1
    training_images   = [_create_adaboost_learning_image(d, compute_modis_indices(d)) for d in training_domains]
    bands             = safe_get_info(training_images[0].bandNames()) # Get list of all available input algorithms
    training_domains_range = range(len(training_domains)) # Store this for convenience
    
    print 'Computing threshold ranges.' # Low and high percentile information for each band
    band_splits = _compute_threshold_ranges(training_domains, training_images, water_masks, bands)
    
    # For each training image compute the number of non-null pixels
    counts  = [safe_get_info(training_images[i].select('b1').reduceRegion(ee.Reducer.count(), training_domains[i].bounds, EVAL_RESOLUTION))['b1'] for i in training_domains_range]
    count   = sum(counts) # Sum of non-null pixels over all training images
    weights = [ee.Image(1.0 / count) for i in training_domains_range] # Each input pixel in the training images has an equal weight
    
    # ????
    # Initialize for pre-existing partially trained classifier
    full_classifier = []
    for (c, t, alpha) in full_classifier:
        band_splits[c].append(t)
        band_splits[c] = sorted(band_splits[c])
        total = 0
        for i in training_domains_range:
            weights[i] = weights[i].multiply(apply_classifier(training_images[i], c, t).multiply(transformed_masks[i]).multiply(-alpha).exp())
            total += safe_get_info(weights[i].reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant']
        for i in training_domains_range:
            weights[i] = weights[i].divide(total)
    
    # Apply weak classifiers to the input test image
    test_image = _create_adaboost_learning_image(domain, b)
    
    # ????
    while len(full_classifier) < 100:
        best = None
        for band_name in bands: # For each weak classifier
            # Find the best threshold that we can choose
            (threshold, ind, value) = _find_adaboost_optimal_threshold(training_domains, training_images, water_masks, band_name, weights, band_splits[band_name])
            
            # Compute the sum of weighted classification errors across all of the training domains using this threshold
            errors = [safe_get_info(weights[i].multiply(training_images[i].select(band_name).lte(threshold).neq(water_masks[i])).reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in training_domains_range]
            error  = sum(errors)
            print '%s found threshold %g with entropy %g error %g' % (band_name, threshold, value, error)
            
            # Record the band/threshold combination with the highest abs(error)
            if (best == None) or (abs(0.5 - error) > abs(0.5 - best[0])): # Classifiers that are always wrong are also good with negative alpha
                best = (error, band_name, threshold, ind)
        
        # ??  
        band_splits[best[1]].insert(best[3], best[2])
      
        print 'Using %s < %g. Error %g.' % (best[1], best[2], best[0])
        alpha      = 0.5 * math.log((1 - best[0]) / best[0])
        classifier = (best[1], best[2], alpha)
        full_classifier.append(classifier)
        
        # ???
        weights = [weights[i].multiply(apply_classifier(training_images[i], classifier[0], classifier[1]).multiply(transformed_masks[i]).multiply(-alpha).exp()) for i in training_domains_range]
        totals  = [safe_get_info(weights[i].reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in training_domains_range]
        total   = sum(totals)
        weights = [w.divide(total) for w in weights]
        print full_classifier

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
    #classified = _create_adaboost_learning_image(domain, b).classify(classifier).select(['classification'], ['b1']); 
    return classified;
