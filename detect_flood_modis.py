import logging
logging.basicConfig(level=logging.ERROR)
import util.ee_authenticate
util.ee_authenticate.initialize()

import os
import ee
from util.mapclient_qt import centerMap, addToMap

import modis.domains
import modis.flood_algorithms

DOMAIN = modis.domains.BORDER
ALGORITHMS = [modis.flood_algorithms.DARTMOUTH, modis.flood_algorithms.DIFFERENCE, modis.flood_algorithms.DNNS, modis.flood_algorithms.DNNS_DEM]

d = modis.domains.retrieve_domain(DOMAIN)

center = d.bounds.centroid().getInfo()['coordinates']
centerMap(center[0], center[1], 11)
addToMap(d.landsat, {'bands': ['B3', 'B2', 'B1'], 'gain': d.landsat_gain}, 'Landsat RGB')
addToMap(d.low_res_modis, {'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06'], 'min' : 0, 'max': 3000, 'opacity' : 1.0}, 'MODIS', False)

for a in range(len(ALGORITHMS)):
	(alg, result) = modis.flood_algorithms.detect_flood(d, ALGORITHMS[a])
	color = modis.flood_algorithms.get_algorithm_color(ALGORITHMS[a])
	addToMap(result.mask(result), {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': '000000, ' + color}, alg, False);

#addToMap(domain.groundTruth.mask(domain.groundTruth), {'min': 0, 'max' : 1, 'opacity' : 0.2}, 'Ground Truth', false);
#addToMap(domain.dem, {min:25, max:50}, 'DEM', false);

