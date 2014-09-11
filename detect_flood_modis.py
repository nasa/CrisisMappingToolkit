import logging
logging.basicConfig(level=logging.ERROR)
import util.ee_authenticate
util.ee_authenticate.initialize()

import os
import ee
import functools

import modis.domains
from modis.flood_algorithms import *

from util.mapclient_qt import centerMap, addToMap
from util.evaluation import evaluate_approach

DOMAIN = modis.domains.BORDER
ALGORITHMS = [DARTMOUTH, DIFFERENCE, DNNS, DNNS_DEM]

def evaluation_function(pair, alg):
	precision, recall = pair
	print '%s: (%4g, %4g)' % (get_algorithm_name(alg), precision, recall)

d = modis.domains.retrieve_domain(DOMAIN)

centerMap(d.center[0], d.center[1], 11)
addToMap(d.landsat, {'bands': ['B3', 'B2', 'B1'], 'gain': d.landsat_gain}, 'Landsat RGB')
addToMap(d.low_res_modis, {'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06'], 'min' : 0, 'max': 3000, 'opacity' : 1.0}, 'MODIS', False)

for a in range(len(ALGORITHMS)):
	(alg, result) = detect_flood(d, ALGORITHMS[a])
	color = get_algorithm_color(ALGORITHMS[a])
	addToMap(result.mask(result), {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': '000000, ' + color}, alg, False)
	evaluate_approach(functools.partial(evaluation_function, alg=ALGORITHMS[a]), result, d.ground_truth, d.bounds, is_algorithm_fractional(ALGORITHMS[a]))

#addToMap(domain.groundTruth.mask(domain.groundTruth), {'min': 0, 'max' : 1, 'opacity' : 0.2}, 'Ground Truth', false);
#addToMap(domain.dem, {min:25, max:50}, 'DEM', false);

