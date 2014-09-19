import ee

from util.mapclient_qt import centerMap, addToMap

import domains
from histogram import RadarHistogram

# From Towards an automated SAR-based flood monitoring system:
# Lessons learned from two case studies by Matgen, Hostache et. al.
THRESHOLD    = 1

def grow_regions(domain, thresholded, thresholds):
	REGION_GROWTH_RANGE = 20
	neighborhood_kernel = ee.Kernel.square(1, 'pixels', False)
	loose_thresholded = domain.image.select([domain.channels[0]]).lte(thresholds[0])
	for i in range(1, len(domain.channels)):
		loose_thresholded = loose_thresholded.And(domain.image.select([domain.channels[i]]).lte(thresholds[i]))
	addToMap(loose_thresholded, {'min': 0, 'max': 1}, 'Loose', False)
	for i in range(REGION_GROWTH_RANGE):
		thresholded = thresholded.convolve(neighborhood_kernel).And(loose_thresholded)
	return thresholded

def threshold(domain):
	hist = RadarHistogram(domain)
	
	thresholds = hist.find_thresholds()
	
	results = []
	for c in range(len(thresholds)):
		ch = domain.channels[c]
		results.append(domain.image.select([ch], [ch]).lte(thresholds[c]))

	result_image = results[0]
	for c in range(1, len(results)):
		result_image = result_image.addBands(results[c], [domain.channels[c]])
	addToMap(result_image, {'min': 0, 'max': 1}, 'Color Image', False)
	
	result_image = results[0].select([domain.channels[0]], ['b1'])
	for c in range(1, len(results)):
		result_image = result_image.And(results[c])
	
	growing_thresholds = hist.find_loose_thresholds()
	result_image = grow_regions(domain, result_image, growing_thresholds)
	
	hist.show_histogram()

	return result_image

__ALGORITHMS = {
	THRESHOLD : ('Threshold', threshold, '00FFFF')
}

def detect_flood(image, algorithm):
	try:
		approach = __ALGORITHMS[algorithm]
	except:
		return None
	return approach[1](image)

def get_algorithm_name(algorithm):
	try:
		return __ALGORITHMS[algorithm][0]
	except:
		return None

def get_algorithm_color(algorithm):
	try:
		return __ALGORITHMS[algorithm][2]
	except:
		return None

