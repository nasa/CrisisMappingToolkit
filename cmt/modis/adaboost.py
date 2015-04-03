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
            best       = k
            best_value = gain
    
    # ??
    return (choices[best], best + 1, best_value)

def apply_classifier(image, band, threshold):
    '''Apply LTE threshold and convert to -1 / 1 (Adaboost requires this)'''
    return image.select(band).lte(threshold).multiply(2).subtract(1)

def get_adaboost_sum(domain, b, classifier = None):
    if classifier == None:
        # These are a set of known good computed values:  (Algorithm, Detection threshold, Weight)
        # learned from everything
        classifier = [(u'b2', 1066.9529712504814, 1.5025586686710706), (u'NDWI', 0.14661938301755412, -0.21891567708553822), (u'dartmouth', 0.48798681823081197, -0.15726997982017618), (u'dartmouth', 0.6365457444743317, 0.18436960110357703), (u'LSWIminusNDVI', 0.3981981030948878, -0.10116535428832296), (u'fai', 355.7817695891917, -0.11241883192214887), (u'dartmouth', 0.7108252075960915, 0.16267637123701892), (u'diff', 528.9578763019633, -0.08056940174311174), (u'NDWI', -0.2608919987915783, -0.0662560864223818), (u'diff', 945.4263065720343, -0.06468547496541238), (u'LSWI', 0.10099215524728983, 0.06198258456041972), (u'LSWI', 0.4036574931704132, -0.13121098919819557), (u'NDVI', -0.11873600974959503, -0.06877321671018986), (u'dartmouth', 0.6736854760352116, 0.058740830970174365), (u'diff', 737.1920914369988, -0.07784405443757562), (u'fai', 637.5040900767088, 0.06383077739328656), (u'LSWI', 0.2523248242088515, -0.06159092845229366), (u'diff', 841.3091990045166, -0.03543296624866381), (u'NDWI', -0.0571363078870121, -0.033363758883119425), (u'NDWI', -0.1590141533392952, -0.04351253722452526), (u'LSWIminusNDVI', -0.021934005871228984, 0.05405714553564335), (u'LSWIminusNDVI', -0.23200006035428739, -0.05945459980438702), (u'diff', 893.3677527882754, -0.04401238934808345), (u'LSWIminusNDVI', -0.33703308759581657, -0.03270875405530488), (u'LSWIminusEVI', -0.06154292961108998, 0.03531804439403144), (u'LSWIminusEVI', -0.6658933123079813, 0.049495070534741545), (u'LSWIminusNDVI', -0.284516573975052, -0.0652646748372963), (u'LSWI', 0.17665848972807066, 0.035535330348720445), (u'LSWIminusEVI', -0.9680685036564269, 0.03385062752160848), (u'dartmouth', 0.6551156102547716, 0.0255425888326403), (u'diff', 867.338475896396, -0.029608603189018888), (u'dartmouth', 0.6644005431449915, 0.031453944391964694), (u'b2', 1597.1343803620828, -0.032483706321846446), (u'b2', 1862.2250849178836, 0.11634887737020584), (u'diff', 880.3531143423356, -0.0759983094592842), (u'LSWIminusNDVI', -0.2582583171646697, -0.03107758927177279), (u'LSWI', 0.13882532248768026, 0.028397075633745206), (u'fai', 496.64292983295024, -0.033912739368973946), (u'EVI', -0.050627832167768394, 0.01948538465509801), (u'MNDWI', -0.314992942152472, -0.028318432983029183), (u'LSWIminusEVI', -0.8169809079822041, 0.017475858734323002), (u'EVI', 0.393374399578999, 0.02621185146352496), (u'LSWIminusEVI', -0.7414371101450927, 0.024903163093461297), (u'LSWIminusNDVI', -0.27138744556986083, -0.026585292504625754), (u'MNDWI', 0.008416570589500877, -0.020089052435031476), (u'MNDWI', 0.17012132696048732, 0.09013852992730866), (u'MNDWI', 0.2509737051459805, 0.07860221464785291), (u'NDWI', -0.20995307606543676, 0.12420114244297033), (u'MNDWI', 0.2105475160532339, 0.058700913757496295), (u'diff', 873.8457951193658, -0.06536366152935989), (u'MNDWI', 0.2307606105996072, 0.05774578894612022), (u'LSWIminusNDVI', -0.3107748307854343, 0.06324847449109565), (u'MNDWI', 0.24086715787279384, 0.05222308145337588), (u'NDVI', 0.2321600980908248, -0.044636312533643016), (u'b1', 840.9896718836895, -0.04386969870931187), (u'b1', 431.64304441090013, 0.10652512181225056), (u'MNDWI', 0.24592043150938717, 0.04629035855680991)]
        # learned from everything and floods permanent water mask
        #classifier = [(u'b2', 1066.9529712504814, 1.5025586686710706), (u'NDWI', 0.14661938301755412, -0.21891567708553822), (u'dartmouth', 0.48798681823081197, -0.15726997982017618), (u'dartmouth', 0.6365457444743317, 0.18436960110357703), (u'LSWIminusNDVI', 0.3981981030948878, -0.10116535428832296), (u'fai', 355.7817695891917, -0.11241883192214887), (u'dartmouth', 0.7108252075960915, 0.16267637123701892), (u'diff', 528.9578763019633, -0.08056940174311174), (u'NDWI', -0.2608919987915783, -0.0662560864223818), (u'diff', 945.4263065720343, -0.06468547496541238), (u'LSWI', 0.10099215524728983, 0.06198258456041972), (u'LSWI', 0.4036574931704132, -0.13121098919819557), (u'NDVI', -0.11873600974959503, -0.06877321671018986), (u'dartmouth', 0.6736854760352116, 0.058740830970174365), (u'diff', 737.1920914369988, -0.07784405443757562), (u'fai', 637.5040900767088, 0.06383077739328656), (u'LSWI', 0.2523248242088515, -0.06159092845229366), (u'diff', 841.3091990045166, -0.03543296624866381), (u'NDWI', -0.0571363078870121, -0.033363758883119425), (u'NDWI', -0.1590141533392952, -0.04351253722452526), (u'LSWIminusNDVI', -0.021934005871228984, 0.05405714553564335), (u'LSWIminusNDVI', -0.23200006035428739, -0.05945459980438702), (u'diff', 893.3677527882754, -0.04401238934808345), (u'LSWIminusNDVI', -0.33703308759581657, -0.03270875405530488), (u'LSWIminusEVI', -0.06154292961108998, 0.03531804439403144), (u'LSWIminusEVI', -0.6658933123079813, 0.049495070534741545), (u'LSWIminusNDVI', -0.284516573975052, -0.0652646748372963), (u'LSWI', 0.17665848972807066, 0.035535330348720445), (u'LSWIminusEVI', -0.9680685036564269, 0.03385062752160848), (u'dartmouth', 0.6551156102547716, 0.0255425888326403), (u'diff', 867.338475896396, -0.029608603189018888), (u'dartmouth', 0.6644005431449915, 0.031453944391964694), (u'b2', 1597.1343803620828, -0.032483706321846446), (u'b2', 1862.2250849178836, 0.11634887737020584), (u'diff', 880.3531143423356, -0.0759983094592842), (u'LSWIminusNDVI', -0.2582583171646697, -0.03107758927177279), (u'LSWI', 0.13882532248768026, 0.028397075633745206), (u'fai', 496.64292983295024, -0.033912739368973946), (u'EVI', -0.050627832167768394, 0.01948538465509801), (u'MNDWI', -0.314992942152472, -0.028318432983029183), (u'LSWIminusEVI', -0.8169809079822041, 0.017475858734323002), (u'EVI', 0.393374399578999, 0.02621185146352496), (u'LSWIminusEVI', -0.7414371101450927, 0.024903163093461297), (u'LSWIminusNDVI', -0.27138744556986083, -0.026585292504625754), (u'MNDWI', 0.008416570589500877, -0.020089052435031476), (u'MNDWI', 0.17012132696048732, 0.09013852992730866), (u'MNDWI', 0.2509737051459805, 0.07860221464785291), (u'NDWI', -0.20995307606543676, 0.12420114244297033), (u'MNDWI', 0.2105475160532339, 0.058700913757496295), (u'diff', 873.8457951193658, -0.06536366152935989), (u'MNDWI', 0.2307606105996072, 0.05774578894612022), (u'LSWIminusNDVI', -0.3107748307854343, 0.06324847449109565), (u'MNDWI', 0.24086715787279384, 0.05222308145337588), (u'NDVI', 0.2321600980908248, -0.044636312533643016), (u'b1', 840.9896718836895, -0.04386969870931187), (u'b1', 431.64304441090013, 0.10652512181225056), (u'MNDWI', 0.24592043150938717, 0.04629035855680991), (u'LSWI', 0.15774190610787547, 0.036843417853100205), (u'dartmouth', 0.6690430095901015, -0.03212513821387442), (u'NDVI', 0.40760815201103473, 0.02804021082285224), (u'NDWI', -0.23542253742850755, 0.03186450807020461), (u'MNDWI', 0.24339379469109051, 0.02862992753172424), (u'NDWI', -0.24815726811004293, 0.02759848052686244), (u'MNDWI', 0.24844706832768382, 0.028767258391409676), (u'NDWI', -0.2545246334508106, 0.023920554168868943), (u'MNDWI', 0.24971038673683216, 0.019530411666440373), (u'diff', 854.3238374504563, -0.017420419212339854), (u'fai', 567.0735099548295, 0.017319660531085516), (u'dartmouth', 0.6667217763675466, -0.01514266152502469), (u'EVI', 0.6153755154523827, 0.015811343455312793), (u'NDWI', -0.2577083161211945, 0.016367445780215994), (u'EVI', 0.5043749575156908, 0.021397992947186542), (u'b1', 636.3163581472949, 0.017679309609453388), (u'b1', 533.9797012790975, 0.052480869491753616), (u'LSWIminusNDVI', -0.2779520097724564, -0.025442696952484856), (u'b2', 1729.6797326399833, -0.02851944225149679), (u'b2', 1663.407056501033, -0.031126464288873154), (u'EVI', 0.4488746785473449, 0.02559032221555967), (u'b1', 585.1480297131961, 0.02206908624237135), (u'diff', 789.2506452207576, -0.016514173533494117), (u'fai', 531.8582198938899, 0.023477946621113858), (u'b1', 610.7321939302456, -0.012825485385474514), (u'diff', 815.2799221126371, -0.01149904087913293), (u'MNDWI', 0.249078727532258, 0.01515097935650566), (u'NDWI', -0.2593001574563864, 0.01083047626473297), (u'LSWIminusNDVI', -0.28123429187375415, -0.012898006552737352), (u'NDWI', -0.2600960781239824, 0.011987178213307102), (u'LSWIminusNDVI', -0.2828754329244031, -0.010878187527895457), (u'NDWI', -0.26049403845778035, 0.010250229825634711), (u'LSWIminusNDVI', -0.2820548623990786, -0.009697140264622033), (u'LSWIminusEVI', -0.703665211226537, 0.00944101306441367), (u'NDVI', 0.3198841250509298, 0.009414806179635152), (u'NDVI', 0.36374613853098225, 0.013660042816894051), (u'NDWI', -0.2606930186246793, 0.01871187931420404), (u'NDVI', 0.3856771452710085, 0.02480484723062163), (u'ratio', 1.9036367764380129, -0.025236576927683663), (u'ratio', 2.8194639395525054, 0.02624098822658558), (u'LSWIminusEVI', -0.6847792617672592, 0.0235627702128661), (u'ratio', 2.361550357995259, 0.017108797919490024), (u'diff', 828.2945605585769, -0.01756108404460174)]
    test_image = _create_extended_learning_image(domain, b)
    total = ee.Image(0).select(['constant'], ['b1'])
    for c in classifier:
      total = total.add(test_image.select(c[0]).lte(c[1]).multiply(2).subtract(1).multiply(c[2]))
    return total

def adaboost(domain, b, classifier = None):
    '''Run Adaboost classifier'''
    total = get_adaboost_sum(domain, b, classifier)
    return total.gte(0.0)

def adaboost_dem(domain, b, classifier = None):
    total = get_adaboost_sum(domain, b, classifier)
    fraction = total.add(ee.Image(2.0)).divide(ee.Image(1.0)).min(1.0).max(0.0)
    return apply_dem(domain, fraction)

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
    all_problems      = ['kashmore_2010_8.xml', 'mississippi_2011_5.xml', 'mississippi_2011_6.xml', 'new_orleans_2005_9.xml', 'sf_bay_area_2011_4.xml']
    all_domains       = [Domain('config/domains/modis/' + d) for d in all_problems]
    training_domains  = [domain.unflooded_domain for domain in all_domains[:-1]] + [all_domains[-1]]
    water_masks       = [get_permanent_water_mask() for d in training_domains]
    # water_masks.extend([get_permanent_water_mask() for d in training_domains])
    transformed_masks = [water_mask.multiply(2).subtract(1) for water_mask in water_masks]
    training_images   = [_create_extended_learning_image(d, compute_modis_indices(d)) for d in training_domains]
    # add pixels in flood permanent water masks to training
    # training_images.extend([_create_extended_learning_image(d, compute_modis_indices(d)).mask(get_permanent_water_mask()) for d in all_domains])
    bands             = safe_get_info(training_images[0].bandNames())
    print 'Computing threshold ranges.'
    band_splits = __compute_threshold_ranges(training_domains, training_images, water_masks, bands)
    counts = [safe_get_info(training_images[i].select('b1').reduceRegion(ee.Reducer.count(), training_domains[i].bounds, 250))['b1'] for i in range(len(training_domains))]
    count = sum(counts)
    weights = [ee.Image(1.0 / count) for i in training_domains_range] # Each input pixel in the training images has an equal weight
    
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
    
    # learn 100 weak classifiers
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
        
        # add an additional split point to search between for thresholds
        band_splits[best[1]].insert(best[3], best[2])
      
        print 'Using %s < %g. Error %g.' % (best[1], best[2], best[0])
        alpha      = 0.5 * math.log((1 - best[0]) / best[0])
        classifier = (best[1], best[2], alpha)
        full_classifier.append(classifier)
        
        # update the weights
        weights = [weights[i].multiply(apply_classifier(training_images[i], classifier[0], classifier[1]).multiply(transformed_masks[i]).multiply(-alpha).exp()) for i in training_domains_range]
        totals  = [safe_get_info(weights[i].reduceRegion(ee.Reducer.sum(), training_domains[i].bounds, EVAL_RESOLUTION))['constant'] for i in training_domains_range]
        total   = sum(totals)
        weights = [w.divide(total) for w in weights]
        print full_classifier

