import logging
logging.basicConfig(level=logging.ERROR)
import ee_authenticate
ee_authenticate.initialize()

import os
import ee
from mapclient_qt import centerMap, addToMap

import domain
import flood_algorithms

d = domain.retrieve_domain(domain.BORDER)

center = d.bounds.centroid().getInfo()['coordinates']
centerMap(center[0], center[1], 11)
addToMap(d.landsat, {'bands': ['B3', 'B2', 'B1'], 'gain': d.landsat_gain}, 'Landsat RGB')
addToMap(d.low_res_modis, {'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06'], 'min' : 0, 'max': 3000, 'opacity' : 1.0}, 'MODIS', False)

(alg, result) = flood_algorithms.detect_flood(d, flood_algorithms.DARTMOUTH)
addToMap(result.mask(result), {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': '000000, FF00FF'}, alg, True);

#addToMap(domain.groundTruth.mask(domain.groundTruth), {'min': 0, 'max' : 1, 'opacity' : 0.2}, 'Ground Truth', false);
#addToMap(domain.dem, {min:25, max:50}, 'DEM', false);

