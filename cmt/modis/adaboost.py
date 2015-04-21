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
    for k in range(len(choices)):
        # Pick a threshold and count how many pixels fall under it across all the input images
        c = choices[k]
        errors = [safe_get_info(weights[i].multiply(images[i].select(band_name).lte(c).neq(truths[i])).reduceRegion(ee.Reducer.sum(), domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in range(len(images))]
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
        print c, error
        if (best == None) or abs(0.5 - error) > abs(0.5 - best_value): # Record the maximum gain
            best       = k
            best_value = error
    
    # ??
    return (choices[best], best + 1, best_value)

def apply_classifier(image, band, threshold):
    '''Apply LTE threshold and convert to -1 / 1 (Adaboost requires this)'''
    return image.select(band).lte(threshold).multiply(2).subtract(1)

def get_adaboost_sum(domain, b, classifier = None):
    if classifier == None:
        # These are a set of known good computed values:  (Algorithm, Detection threshold, Weight)
        # learned from everything
        classifier = [(u'dartmouth', 0.30887438055782945, 1.4558371112080295), (u'b2', 2020.1975382568198, 0.9880130793929531), (u'MNDWI', 0.3677501330908955, 0.5140443440746121), (u'b2', 1430.1463073852296, 0.15367606716883875), (u'b1', 1108.5241042345276, 0.13193086117959033), (u'dartmouth', 0.7819758531686796, -0.13210548296374583), (u'dartmouth', 0.604427824270283, 0.12627962195951867), (u'b2', 1725.1719228210247, -0.07293616881105353), (u'b2', 1872.6847305389224, -0.09329031467870501), (u'b2', 1577.659115103127, 0.1182474134065663), (u'b2', 1946.441134397871, -0.13595282841411163), (u'b2', 2610.24876912841, 0.10010381165310277), (u'b2', 1983.3193363273454, -0.0934455057392682), (u'b2', 1503.9027112441784, 0.13483194249576771), (u'b2', 2001.7584372920826, -0.10099203054937314), (u'b2', 2905.2743845642053, 0.1135686859467779), (u'dartmouth', 0.5156538098210846, 0.07527677772747364), (u'b2', 2010.9779877744513, -0.09535260187161688), (u'b2', 1798.9283266799735, 0.07889358547222977), (u'dartmouth', 0.36787708796485785, -0.07370319016383906), (u'MNDWI', -0.6422574132273133, 0.06922934793487515), (u'dartmouth', 0.33837573426134365, -0.10266747186797487), (u'dartmouth', 0.4712668025964854, 0.09612545197834421), (u'dartmouth', 0.3236250574095866, -0.10754218805531587), (u'MNDWI', -0.48248013602276113, 0.111365639029263), (u'dartmouth', 0.316249718983708, -0.10620217821842894), (u'dartmouth', 0.4490732989841858, 0.09743861137429623), (u'dartmouth', 0.31256204977076874, -0.08121162639185005), (u'MNDWI', -0.5623687746250372, 0.10344420165347998), (u'dartmouth', 0.3107182151642991, -0.08899821447581886), (u'LSWI', -0.29661326544921773, 0.08652882218688322), (u'dartmouth', 0.3097962978610643, -0.07503568257204306), (u'MNDWI', 0.022523637136343283, 0.08765150582301148), (u'b2', 2015.5877630156356, -0.06978548014829108), (u'b2', 3052.7871922821028, 0.08567389991115743), (u'LSWI', -0.19275063787434812, 0.08357667312445341), (u'dartmouth', 0.3093353392094469, -0.08053950648462435), (u'LSWI', -0.14081932408691333, 0.07186342090261867), (u'dartmouth', 0.30910485988363817, -0.05720223719278896), (u'MNDWI', 0.19513688511361937, 0.07282637257701345), (u'NDWI', -0.361068160450533, 0.06565995208358431), (u'NDWI', -0.2074005503754442, -0.0522715989389411), (u'b1', 775.4361563517915, 0.05066415016422507), (u'b2', 2017.8926506362277, -0.0596357907686033), (u'b2', 1762.050124750499, 0.06600172638129476), (u'b2', 2019.0450944465238, -0.05498763067596745), (u'b1', 941.9801302931596, 0.06500771792028737), (u'dartmouth', 0.24987167315080105, 0.06409775979747406), (u'b2', 2979.0307884231543, 0.06178896578945445), (u'dartmouth', 0.22037031944728686, 0.04708770942378687), (u'dartmouth', 0.30898962022073384, -0.06357932266591948), (u'EVI', -0.13991172174597732, 0.061167901067941045), (u'dartmouth', 0.30893200038928165, -0.047538992866687814), (u'dartmouth', 0.23512099629904396, 0.055800430467148325), (u'dartmouth', 0.3089031904735555, -0.04993911823852714), (u'dartmouth', 0.22774565787316542, 0.045917043382747345), (u'b1', 232.32231270358304, -0.04624672841408699), (u'LSWIminusEVI', -1.3902019910129537, 0.044122210356250594), (u'fai', 914.8719936250361, 0.04696283008449494), (u'b2', 2019.6213163516718, -0.051114386132496435), (u'b2', 2315.2231536926147, 0.048898662215419296), (u'fai', 1434.706585047812, -0.05352547959475242), (u'diff', -544.4250000000001, -0.04459039609050114), (u'dartmouth', 0.39737844166837205, 0.045452678171318414), (u'dartmouth', 0.3088887855156925, -0.03891014191130265), (u'dartmouth', 0.22405798866022614, 0.042128457713671935), (u'diff', -777.2958333333333, -0.03902784979889064), (u'dartmouth', 0.2222141540537565, 0.03788131334473313), (u'dartmouth', 0.30888158303676094, -0.037208213701295255), (u'dartmouth', 0.3531264111131007, 0.0375648736301961), (u'dartmouth', 0.3088779817972952, -0.03427856593613819), (u'LSWI', -0.16678498098063071, 0.03430983541990538), (u'fai', -425.5957838307736, -0.03348006551810443), (u'NDWI', -0.13056674533789978, -0.03552899660957818), (u'b2', 2019.3332053990978, -0.0344936369203531), (u'b2', 1835.806528609448, 0.03856210900250611), (u'b2', 1467.0245093147041, -0.0345449746977328), (u'fai', 395.0374022022602, 0.031130251540884356), (u'fai', 654.9546979136481, 0.04214466417320743), (u'b2', 1448.5854083499669, -0.05667775728680656), (u'fai', 135.12010649087222, 0.03948338203848539), (u'dartmouth', 0.493460306208785, -0.045802615250103394), (u'fai', 784.9133457693422, 0.03128133499873274), (u'fai', 1174.7892893364242, -0.04413487095880613), (u'b2', 3015.9089903526283, 0.04133685218791008), (u'fai', 1304.7479371921181, -0.04107557606064173), (u'b2', 2462.7359614105126, 0.03777625735990945), (u'fai', 1369.727261119965, -0.03524600268462714), (u'b2', 2997.4698893878913, 0.03864830537283341), (u'dartmouth', 0.22313607135699132, 0.0348041704038284), (u'fai', -575.9950811359025, -0.036345846940478974), (u'fai', 1402.2169230838886, -0.03481517966048645), (u'fai', 719.9340218414952, 0.032833655233338276), (u'b2', 2019.1891499228109, -0.03272953788499046), (u'b2', 2388.9795575515636, 0.03713369823962704), (u'b2', 2019.1171221846673, -0.027949075715791222), (u'b2', 1743.611023785762, 0.03310357200312585), (u'LSWIminusNDVI', -0.3990346417915731, 0.029045726328998267), (u'NDWI', -0.16898364785667197, -0.025735337614573982), (u'dartmouth', 0.3088761811775623, -0.02973898070330325)]

        
    test_image = _create_adaboost_learning_image(domain, b)
    total = ee.Image(0).select(['constant'], ['b1'])
    for c in classifier:
      total = total.add(test_image.select(c[0]).lte(c[1]).multiply(2).subtract(1).multiply(c[2]))
    return total

def adaboost(domain, b, classifier = None):
    '''Run Adaboost classifier'''
    total = get_adaboost_sum(domain, b, classifier)
    return total.gte(-1.0) # Just threshold the results at zero (equal chance of flood / not flood)

def adaboost_dem(domain, b, classifier = None):
    
    # Get raw adaboost output
    total = get_adaboost_sum(domain, b, classifier)
    addToMap(total, {'min': -10, 'max': 10}, 'raw ADA', False)
    
    # Convert this range of values into a zero to one probability scale
    # - These bounds represent where the probability plateaus.  Thes plateaus are
    #    usually not at 0% or 100% !!
    MIN_SUM = -6.0
    MAX_SUM =  2.0
    val_range = MAX_SUM - MIN_SUM
    
    fraction = total.subtract(ee.Image(MIN_SUM)).divide(ee.Image(val_range)).clamp(0.0, 1.0)
    addToMap(fraction, {'min': 0, 'max': 1}, 'fraction', False)
    return cmt.modis.modis_utilities.apply_dem(domain, fraction)

def __compute_threshold_ranges(training_domains, training_images, water_masks, bands):
    '''For each band, find lowest and highest fixed percentiles among the training domains.'''
    LOW_PERCENTILE  = 20
    HIGH_PERCENTILE = 100
    EVAL_RESOLUTION = 250
    
    band_splits = dict()
    for band_name in bands: # Loop through each band (weak classifier input)
        split = None
        print 'Computing threshold ranges for: ' + band_name
      
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

def adaboost_learn(domain, b):
    '''Train Adaboost classifier'''
    
    EVAL_RESOLUTION = 250

    # Load inputs for this domain and preprocess
    #all_problems      = ['kashmore_2010_8.xml', 'mississippi_2011_5.xml', 'mississippi_2011_6.xml', 'new_orleans_2005_9.xml', 'sf_bay_area_2011_4.xml']
    #all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
    #training_domains  = [domain.unflooded_domain for domain in all_domains[:-1]] + [all_domains[-1]] # SF is unflooded
    all_problems      = ['unflooded_mississippi_2010.xml', 'unflooded_new_orleans_2004.xml', 'sf_bay_area_2011_4.xml', 'unflooded_baghlan_2013.xml', 'unflooded_bosnia_2013.xml']
    all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
    training_domains  = all_domains
    water_masks       = [modis_utilities.get_permanent_water_mask() for d in training_domains]
    training_images   = [_create_adaboost_learning_image(d, modis_utilities.compute_modis_indices(d)) for d in training_domains]
    
    # add pixels in flood permanent water masks to training
    #training_domains.extend(all_domains)
    #water_masks.extend([get_permanent_water_mask() for d in all_domains])
    #training_images.append([_create_adaboost_learning_image(domain, compute_modis_indices(domain)).mask(get_permanent_water_mask()) for domain in all_domains])
    
    transformed_masks = [water_mask.multiply(2).subtract(1) for water_mask in water_masks]

    bands             = safe_get_info(training_images[0].bandNames())
    print 'Computing threshold ranges.'
    band_splits = __compute_threshold_ranges(training_domains, training_images, water_masks, bands)
    counts = [safe_get_info(training_images[i].select('diff').reduceRegion(ee.Reducer.count(), training_domains[i].bounds, 250))['diff'] for i in range(len(training_images))]
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
    
    # Apply weak classifiers to the input test image
    test_image = _create_adaboost_learning_image(domain, b)
    
    # learn 100 weak classifiers
    while len(full_classifier) < 1000:
        best = None
        for band_name in bands: # For each weak classifier
            # Find the best threshold that we can choose
            (threshold, ind, error) = _find_adaboost_optimal_threshold(training_domains, training_images, water_masks, band_name, weights, band_splits[band_name])
            
            # Compute the sum of weighted classification errors across all of the training domains using this threshold
            #errors = [safe_get_info(weights[i].multiply(training_images[i].select(band_name).lte(threshold).neq(water_masks[i])).reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in range(len(training_images))]
            #error  = sum(errors)
            print '%s found threshold %g with error %g' % (band_name, threshold, error)
            
            # Record the band/threshold combination with the highest abs(error)
            if (best == None) or (abs(0.5 - error) > abs(0.5 - best[0])): # Classifiers that are always wrong are also good with negative alpha
                best = (error, band_name, threshold, ind)
        
        # add an additional split point to search between for thresholds
        band_splits[best[1]].insert(best[3], best[2])
      
        print 'Using %s < %g. Error %g.' % (best[1], best[2], best[0])
        alpha      = 0.5 * math.log((1 - best[0]) / best[0])
        classifier = (best[1], best[2], alpha)
        full_classifier.append(classifier)
        
        # update the weights
        weights = [weights[i].multiply(apply_classifier(training_images[i], classifier[0], classifier[1]).multiply(transformed_masks[i]).multiply(-alpha).exp()) for i in range(len(training_images))]
        totals  = [safe_get_info(weights[i].reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in range(len(training_images))]
        total   = sum(totals)
        weights = [w.divide(total) for w in weights]
        print full_classifier


# The results from this don't look great!
#
#import modis_utilities
#def adaboost_dem_learn(classifier = None):
#    '''Train Adaboost classifier'''
#    
#    EVAL_RESOLUTION = 250
#
#    # Load inputs for this domain and preprocess
#    all_problems      = ['kashmore_2010_8.xml', 'mississippi_2011_5.xml', 'mississippi_2011_6.xml', 'new_orleans_2005_9.xml', 'sf_bay_area_2011_4.xml']
#    all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
#    training_domains  = [domain.unflooded_domain for domain in all_domains[:-1]] + [all_domains[-1]] # SF is unflooded
#    water_masks       = [get_permanent_water_mask() for d in training_domains]
#    
#    THRESHOLD_INTERVAL = 1
#    TARGET_PERCENTAGE  = 0.95
#    
#    print 'Computing thresholds'
#    
#    # Loop through each of the raw result images
#    for (truth_image, train_domain, name) in zip(water_masks, training_domains, all_problems):
#
#        # Apply the Adaboost computation to each training image and get the raw results
#        b = modis_utilities.compute_modis_indices(train_domain)
#        sum_image = get_adaboost_sum(train_domain, b, classifier)
#        #addToMap(sum_image, {'min': -10, 'max': 10}, 'raw ADA', False)
#        print '================================'
#        print name
#
#        # For each threshold level above zero, how likely is the pixel to be actually flooded?
#        curr_threshold = -5.0
#        percentage = 0
#        #while percentage < TARGET_PERCENTAGE:
#        while curr_threshold < 5.0:
#            
#            curr_results = sum_image.gte(curr_threshold)
#            #addToMap(curr_results, {'min': 0, 'max': 1}, str(curr_threshold), False)
#            #addToMap(truth_image, {'min': 0, 'max': 1}, 'truth', False)
#            sum_correct  = safe_get_info(curr_results.multiply(truth_image).reduceRegion(ee.Reducer.sum(), train_domain.bounds, EVAL_RESOLUTION))['b1']
#            sum_total    = safe_get_info(curr_results.reduceRegion(ee.Reducer.sum(), train_domain.bounds, EVAL_RESOLUTION))['b1']
#            #print sum_correct
#            if sum_total > 0:
#                percentage   = sum_correct / sum_total
#                print str(curr_threshold) +': '+ str(sum_total) + ' --> '+ str(percentage)
#            else:
#                break
#            curr_threshold += THRESHOLD_INTERVAL
#        
#    raise Exception('DEBUG')
#
#        # For each threshold level below zero, how likely is the pixel to be actually dry?
        



