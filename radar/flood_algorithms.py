import ee
import domains

import math
import numpy
import scipy
import scipy.special
import scipy.optimize

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from util.mapclient_qt import centerMap, addToMap

THRESHOLD    = 1

def __cdf(params, offset, x):
	mode = params[0]
	k = params[1]
	theta = (mode - offset) / (k - 1)
	return scipy.special.gammainc(k, (x - offset) / theta)

def __show_histogram(histogram, instrument, params=None):
	values = histogram['histogram']
	start = histogram['bucketMin']
	width = histogram['bucketWidth']
	ind = numpy.arange(start=start, stop=start + width * len(values), step=width)[:-1]
	plt.bar(ind, height=values[:-1], width=width, color='b')
	if params != None:
		m = domains.MINIMUM_VALUES[instrument]
		if instrument == domains.UAVSAR:
			m = math.log10(m)
		mid = int((params[0] - start) / width)
		cumulative = sum(values[:mid]) + values[mid] / 2
		scale = cumulative / __cdf(params, m, params[0])
		plt.bar(ind, map(lambda x : scale * (__cdf(params, m, x + width / 2) - __cdf(params, m, x - width / 2)), ind), width=width, color='r', alpha=0.5)

def __gamma_function_errors(p, mode, instrument, start, width, values):
	k = p[0]
	if k <= 1.0:
		return [float('inf')] * len(values)
	m = domains.MINIMUM_VALUES[instrument]
	if instrument == domains.UAVSAR:
		m = math.log10(m)
	error = 0.0
	last_cdf = 0.0
	errors = numpy.zeros(len(values))
	mid = int((mode - start) / width)
	cumulative = sum(values[:mid]) + values[mid] / 2
	scale = cumulative / __cdf((mode, k), m, mode)
	for i in range(len(values)):
		if start + i * width - m > mode:
			break
		cdf = scale * __cdf((mode, k), m, start + i * width)
		errors[i] = (cdf - last_cdf) - values[i]
		last_cdf = cdf
	return errors

__WATER_MODE_RANGES = {
		domains.UAVSAR : {'vv' : (3.0, 4.1), 'hv' : (3.0, 3.6), 'hh' : (3.0, 3.6)},
		domains.SENTINEL1 : {'vv' : (0, 90), 'vh' : (0, 90)}
}

__HISTOGRAM_CLAMP_MAX = {
		domains.UAVSAR: None,
		domains.SENTINEL1 : 600
}

__NUM_BUCKETS = {
		domains.UAVSAR: 128,
		domains.SENTINEL1: 512
}

def __find_threshold_histogram(histogram, instrument, channel):
	start = histogram[channel]['bucketMin']
	width = histogram[channel]['bucketWidth']
	values = histogram[channel]['histogram']

	# find the mode
	i = int((__WATER_MODE_RANGES[instrument][channel][0] - start) / width)
	biggest_bin = i
	while i < len(values) and start + i * width <= __WATER_MODE_RANGES[instrument][channel][1]:
		if values[i] > values[biggest_bin]:
			biggest_bin = i
		i += 1
	mode = start + biggest_bin * width

	# find the other parameters of the distribution
	(value, result) = scipy.optimize.leastsq(__gamma_function_errors, [10], factor=1.0, args=(mode, instrument, start, width, values))
	params = (mode, value[0])

	# choose the best threshold where we can no longer discriminate
	m = domains.MINIMUM_VALUES[instrument]
	if instrument == domains.UAVSAR:
		m = math.log10(m)
	mid = int((mode - start) / width)
	cumulative = sum(values[:mid]) + values[mid] / 2
	scale = cumulative / __cdf(params, m, params[0])
	i = mid
	while i < len(values):
		cumulative += values[i]
		diff = cumulative - scale * __cdf(params, m, start + i * width)
		if diff > 0.01:
			break
		i += 1
	threshold = start + i * width

	return (threshold, params)

def threshold(domain):
	hist_image = domain.image
	if domain.instrument == domains.UAVSAR:
		hist_image = hist_image.log10()
	if __HISTOGRAM_CLAMP_MAX[domain.instrument]:
		hist_image = hist_image.clamp(0, __HISTOGRAM_CLAMP_MAX[domain.instrument])
	histogram = hist_image.reduceRegion(ee.Reducer.histogram(__NUM_BUCKETS[domain.instrument], None, None), domain.bounds, 30, None, None, True).getInfo()
	
	thresholds = {}
	results = []
	#plt.figure(1)
	for c in range(len(domain.channels)):
		ch = domain.channels[c]
		# ignore first bucket, too many... for UAVSAR in particular
		histogram[ch]['bucketMin'] += histogram[ch]['bucketWidth']
		histogram[ch]['histogram'] =  histogram[ch]['histogram'][1:]
		total = sum(histogram[ch]['histogram'])
		histogram[ch]['histogram'] = map(lambda x : x / total, histogram[ch]['histogram'])
		(threshold, params) = __find_threshold_histogram(histogram, domain.instrument, ch)
		if domain.instrument == domains.UAVSAR:
			threshold = 10 ** threshold
		thresholds[ch] = threshold

		channel_result = domain.image.select([ch], [ch]).lte(threshold)
		results.append(channel_result)
		
		#plt.subplot(100 * len(domain.channels) + 10 + c + 1)
		#__show_histogram(histogram[domain.channels[c]], domain.instrument, params)

	result_image = results[0]
	for c in range(1, len(results)):
		result_image = result_image.addBands(results[c], [domain.channels[c]])
	addToMap(result_image, {'min': 0, 'max': 1}, 'Color Image', True)
	
	result_image = results[0].select([domain.channels[0]], ['b1'])
	for c in range(1, len(results)):
		result_image = result_image.And(results[c])
	#plt.show()
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

