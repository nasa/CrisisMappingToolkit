import logging
logging.basicConfig(level=logging.ERROR)
import util.ee_authenticate
util.ee_authenticate.initialize()
import matplotlib
matplotlib.use('tkagg')

import os
import ee
import functools

import radar.domains
from radar.flood_algorithms import *

from util.mapclient_qt import centerMap, addToMap
from util.evaluation import evaluate_approach

DOMAIN = radar.domains.UAVSAR_MISSISSIPPI_FLOODED
#DOMAIN = radar.domains.UAVSAR_ARKANSAS_CITY
#DOMAIN = radar.domains.UAVSAR_MISSISSIPPI_UNFLOODED
#DOMAIN = radar.domains.UAVSAR_NAPO_RIVER
#DOMAIN = radar.domains.SENTINEL1_ROME
#DOMAIN = radar.domains.SENTINEL1_LANCIANO
ALGORITHMS = [MATGEN]
#ALGORITHMS = [MATGEN, DECISION_TREE, RANDOM_FORESTS, SVM]

def evaluation_function(pair, alg):
	precision, recall = pair
	print '%s: (%4g, %4g)' % (get_algorithm_name(alg), precision, recall)

im = radar.domains.get_radar_image(DOMAIN)

centerMap(im.center[0], im.center[1], 11)
apply(addToMap, im.visualize())
ground_truth = radar.domains.get_ground_truth(im)
if ground_truth != None:
	addToMap(ground_truth, {}, 'Ground Truth', False)

for a in range(len(ALGORITHMS)):
	alg = ALGORITHMS[a]
	result = detect_flood(im, alg)
	color = get_algorithm_color(alg)
	addToMap(result.mask(result), {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': '000000, ' + color}, get_algorithm_name(alg), False)
	#print result.getDownloadUrl(params={'name' : 'Result',  'region':im.bounds.toGeoJSONString(), 'scale' : 12})
	#print im.image.visualize(['vv', 'vh'], 1.0).getDownloadUrl(params={'name' : 'Radar', 'region':im.bounds.toGeoJSONString(), 'scale' : 12})
	if ground_truth != None:
		evaluate_approach(functools.partial(evaluation_function, alg=alg), result, ground_truth, im.bounds)

#addToMap(domain.groundTruth.mask(domain.groundTruth), {'min': 0, 'max' : 1, 'opacity' : 0.2}, 'Ground Truth', false);
#addToMap(domain.dem, {min:25, max:50}, 'DEM', false);

