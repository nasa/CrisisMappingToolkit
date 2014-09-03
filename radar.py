import ee_authenticate
ee_authenticate.initialize()

from pprint import pprint
import os
import ee
from ee.mapclient import centerMap, addToMap

RADARSAT = 1
TERRASAR = 2
UAVSAR   = 3

def get_radar_image(index):
	if index == RADARSAT:
		im_hh = ee.Image('18108519531116889794-06793893466375912303')
		im_hv = ee.Image('18108519531116889794-13933004153574033452')
		bounds = ee.geometry.Geometry.Rectangle(-123.60, 48.95, -122.75, 49.55)
		im = im_hh.select(['b1'], ['hh'])
		im = im.addBands(im_hv.select(['b1'], ['hv'])).addBands(0)
	elif index == TERRASAR:
		im_hh = ee.Image('18108519531116889794-04996796288385000359')
		bounds = ee.geometry.Geometry.Rectangle(-79.64, 8.96, -79.55, 9.015)
		return (im_hh.select(['b1'], ['hh']), bounds)
	return (im, bounds)

(im, bounds) = get_radar_image(TERRASAR)

center = bounds.centroid().getInfo()['coordinates']
centerMap(center[0], center[1], 11)
#addToMap(im, {'bands' : ['constant', 'hh', 'hv'], 'min' : 0, 'max' : 3000}, 'HH')
addToMap(im, {'min' : 0, 'max' : 300}, 'HH')
#addToMap(d.low_res_modis, {'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06'], 'min' : 0, 'max': 3000, 'opacity' : 1.0}, 'MODIS', False)

#(alg, result) = flood_algorithms.detect_flood(d, flood_algorithms.DIFFERENCE)
#addToMap(result.mask(result), {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': '000000, 00FFFF'}, alg, True);

#addToMap(domain.groundTruth.mask(domain.groundTruth), {'min': 0, 'max' : 1, 'opacity' : 0.2}, 'Ground Truth', false);
#addToMap(domain.dem, {min:25, max:50}, 'DEM', false);


