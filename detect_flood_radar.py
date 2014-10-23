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

# --------------------------------------------------------------
# Configuration

# Specify the data set to use - see /radar/domains.py
DOMAIN = radar.domains.UAVSAR_MISSISSIPPI_FLOODED
#DOMAIN = radar.domains.UAVSAR_ARKANSAS_CITY
#DOMAIN = radar.domains.UAVSAR_MISSISSIPPI_UNFLOODED
#DOMAIN = radar.domains.UAVSAR_NAPO_RIVER
#DOMAIN = radar.domains.SENTINEL1_ROME
#DOMAIN = radar.domains.SENTINEL1_LANCIANO
ALGORITHMS = [MATGEN]
#ALGORITHMS = [MATGEN, DECISION_TREE, RANDOM_FORESTS, SVM]
#ALGORITHMS = [ACTIVE_CONTOUR]

# --------------------------------------------------------------
# Functions

def evaluation_function(pair, alg):
    '''Pretty print an algorithm and its statistics'''
    precision, recall = pair
    print '%s: (%4g, %4g)' % (get_algorithm_name(alg), precision, recall)

# --------------------------------------------------------------
# main()

# Fetch data set information
im = radar.domains.get_radar_image(DOMAIN)

# Display the Landsat, MODIS, and ground truth data for the data set
centerMap(im.center[0], im.center[1], 11)
apply(addToMap, im.visualize())
ground_truth = radar.domains.get_ground_truth(im)
if ground_truth != None:
    addToMap(ground_truth, {}, 'Ground Truth', False)

#print im.image.getDownloadUrl({'name' : 'sar', 'region':ee.Geometry.Rectangle(-91.23, 32.88, -91.02, 33.166).toGeoJSONString(), 'scale': 6.174})

# For each of the algorithms
for a in range(len(ALGORITHMS)):
    # Run the algorithm on the data and get the results
    alg    = ALGORITHMS[a]
    result = detect_flood(im, alg)
    
    # Get a color pre-associated with the algorithm, then draw it on the map
    color  = get_algorithm_color(alg)
    addToMap(result.mask(result), {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': '000000, ' + color}, get_algorithm_name(alg), False)
    
    # Compare the algorithm output to the ground truth and print the results
    if ground_truth != None:
        evaluate_approach(functools.partial(evaluation_function, alg=alg), result, ground_truth, im.bounds)

#addToMap(domain.groundTruth.mask(domain.groundTruth), {'min': 0, 'max' : 1, 'opacity' : 0.2}, 'Ground Truth', false);
#addToMap(domain.dem, {min:25, max:50}, 'DEM', false);




