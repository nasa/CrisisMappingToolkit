import ee
import domains

import math
import numpy
import scipy
import scipy.special
import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt

class RadarHistogram(object):
	__HISTOGRAM_CLAMP_MAX = {
		domains.SENTINEL1 : 600
	}

	__NUM_BUCKETS = {
		domains.UAVSAR: 128,
		domains.UAVSAR_LOG: 128,
		domains.SENTINEL1: 512
	}
	
	__WATER_MODE_RANGES = {
		domains.UAVSAR : {'vv' : (3.0, 4.1), 'hv' : (3.0, 3.6), 'hh' : (3.0, 3.6)},
		domains.UAVSAR_LOG : {'vv' : (0, 50), 'hv' : (0, 100), 'hh' : (0, 100)},
		domains.SENTINEL1 : {'vv' : (0, 90), 'vh' : (0, 50)}
	}

	BACKSCATTER_MODEL_GAMMA    = 1
	BACKSCATTER_MODEL_GAUSSIAN = 2
	BACKSCATTER_MODEL_DIP      = 3
	BACKSCATTER_MODEL_PEAK     = 4

	def __init__(self, domain, backscatter_model = None):
		self.domain = domain
		if backscatter_model == None:
			if domain.instrument == domains.UAVSAR:
				backscatter_model = RadarHistogram.BACKSCATTER_MODEL_PEAK
			elif domain.instrument == domains.UAVSAR_LOG:
				backscatter_model = RadarHistogram.BACKSCATTER_MODEL_DIP
			else:
				backscatter_model = RadarHistogram.BACKSCATTER_MODEL_GAMMA
		self.backscatter_model = backscatter_model
		
		self.hist_image = self.__preprocess_image(domain.image)

		self.histograms = self.__compute_histogram(self.hist_image)
		self.__find_thresholds()
	
	def __preprocess_image(self, image):
		if not self.domain.log_scale:
			image = image.log10()
		if RadarHistogram.__HISTOGRAM_CLAMP_MAX.has_key(self.domain.instrument):
			image = image.clamp(0, RadarHistogram.__HISTOGRAM_CLAMP_MAX[self.domain.instrument])
		return image
	
	def __compute_histogram(self, image):
		histogram = image.reduceRegion(ee.Reducer.histogram(RadarHistogram.__NUM_BUCKETS[self.domain.instrument], None, None), self.domain.bounds, 30, None, None, True).getInfo()
		h = []
	
		for c in range(len(self.domain.channels)):
			ch = self.domain.channels[c]
			# ignore first bucket, too many... for UAVSAR in particular
			#histogram[ch]['bucketMin'] += histogram[ch]['bucketWidth']
			#histogram[ch]['histogram'] =  histogram[ch]['histogram'][1:]
			
			# normalize
			total = sum(histogram[ch]['histogram'])
			histogram[ch]['histogram'] = map(lambda x : x / total, histogram[ch]['histogram'])
			h.append((histogram[ch]['bucketMin'], histogram[ch]['bucketWidth'], histogram[ch]['histogram']))
		return h

	def __cdf(self, params, x):
		mode = params[0]
		k = params[1]
		offset = params[2]
		if self.backscatter_model == RadarHistogram.BACKSCATTER_MODEL_GAUSSIAN:
			return 0.5 * (1 + scipy.special.erf((x - mode) / (k * math.sqrt(2))))
		theta = (mode - offset) / (k - 1)
		return scipy.special.gammainc(k, (x - offset) / theta)

	# find x where __cdf(params, offset, x) = percentile
	def __cdf_percentile(self, params, percentile):
		mode = params[0]
		k = params[1]
		offset = params[2]
		if self.backscatter_model == RadarHistogram.BACKSCATTER_MODEL_GAUSSIAN:
			return scipy.special.erfinv(percentile / 0.5 - 1) * k * math.sqrt(2) + mode
		theta = (mode - offset) / (k - 1)
		v = scipy.special.gammaincinv(k, percentile) * theta + offset
		return v

	def __show_histogram(self, channel):
		start  = self.histograms[channel][0]
		width  = self.histograms[channel][1]
		values = self.histograms[channel][2]
		
		ind = numpy.arange(start=start, stop=start + width * len(values), step=width)[:-1]
		plt.bar(ind, height=values[:-1], width=width, color='b')
		plt.ylabel(self.domain.channels[channel])
		
		(threshold, params) = self.__find_threshold_histogram(channel)
		if params != None:
			mid = int((params[0] - start) / width)
			cumulative = sum(values[:mid]) + values[mid] / 2
			scale = cumulative / self.__cdf(params, params[0])
			plt.bar(ind, map(lambda x : scale * (self.__cdf(params, x + width) - self.__cdf(params, x)), ind), width=width, color='r', alpha=0.5)
		plt.plot((threshold, threshold), (0, 0.02), 'g--')

	def __gamma_function_errors(self, p, mode, fit_end, offset, channel):
		start  = self.histograms[channel][0]
		width  = self.histograms[channel][1]
		values = self.histograms[channel][2]
		instrument = self.domain.instrument
		k = p[0]
		if (self.backscatter_model == RadarHistogram.BACKSCATTER_MODEL_GAMMA and k <= 1.0):
			return [float('inf')] * len(values)
		if (self.backscatter_model == RadarHistogram.BACKSCATTER_MODEL_GAUSSIAN and k <= 0.0):
			return [float('inf')] * len(values)
		error = 0.0
		last_cdf = 0.0
		errors = numpy.zeros(len(values))
		mid = int((mode - start) / width)
		cumulative = sum(values[:mid]) + values[mid] / 2
		scale = cumulative / self.__cdf((mode, k, offset), mode)
		for i in range(len(values)):
			if start + i * width - offset > fit_end:
				break
			cdf = scale * self.__cdf((mode, k, offset), start + i * width)
			errors[i] = (cdf - last_cdf) - values[i]
			last_cdf = cdf
		return errors

	def __find_threshold_histogram(self, channel):
		start  = self.histograms[channel][0]
		width  = self.histograms[channel][1]
		values = self.histograms[channel][2]
	
		# find the mode
		(minv, maxv) = RadarHistogram.__WATER_MODE_RANGES[self.domain.instrument][self.domain.channels[channel]]
		i = int((minv - start) / width)
		biggest_bin = i
		while i < len(values) and start + i * width <= maxv:
			if values[i] > values[biggest_bin]:
				biggest_bin = i
			i += 1
		mode = start + biggest_bin * width
		
		if self.backscatter_model == RadarHistogram.BACKSCATTER_MODEL_PEAK:
			return (mode, None)

		# find the local minimum after the mode
		i = biggest_bin + 1
		while i < len(values)-3 and start + i * width <= maxv:
			if values[i] < values[i+1] and values[i] < values[i+2] and values[i] < values[i+3]:
				break
			i += 1
		local_min = start + i * width

		if self.backscatter_model == RadarHistogram.BACKSCATTER_MODEL_DIP:
			return (local_min, None)
	
		m = domains.MINIMUM_VALUES[self.domain.instrument]
		if not self.domain.log_scale:
			m = math.log10(m)
		# find the other parameters of the distribution
		(value, result) = scipy.optimize.leastsq(self.__gamma_function_errors, [10], factor=1.0, args=(mode, mode, m, channel))
		params = (mode, value[0], m)
	
		# choose the best threshold where we can no longer discriminate
		mid = int((mode - start) / width)
		cumulative = sum(values[:mid]) + values[mid] / 2
		scale = cumulative / self.__cdf(params, params[0])
		i = mid
		while i < len(values):
			cumulative += values[i]
			diff = cumulative - scale * self.__cdf(params, start + i * width)
			if diff > 0.01:
				break
			i += 1
		threshold = start + i * width
	
		return (threshold, params)

	def __find_thresholds(self):
		self.thresholds = []
		self.distributions = []
		for c in range(len(self.domain.channels)):
			(threshold, params) = self.__find_threshold_histogram(c)
			if not self.domain.log_scale:
				threshold = 10 ** threshold
			self.thresholds.append(threshold)
			self.distributions.append(params)
	
	def get_thresholds(self):
		return self.thresholds
	
	def find_loose_thresholds(self, percentile=0.99):
		results = []
		for c in range(len(self.domain.channels)):
			(threshold, params) = (self.thresholds[c], self.distributions[c])
			if params != None:
				t = self.__cdf_percentile(params, percentile)
				if not self.domain.log_scale:
					t = 10 ** t
				results.append(t)
			else:
				# if finding dip or peak, find next local min / max as threshold
				if not self.domain.log_scale:
					threshold = math.log10(threshold)
				start = self.histograms[c][0]
				width = self.histograms[c][1]
				values = self.histograms[c][2]
				i = int((threshold - start) / width) + 1
				if self.backscatter_model == RadarHistogram.BACKSCATTER_MODEL_DIP:
					while i < len(values) - 3:
						if values[i] > values[i+1] and values[i] > values[i+2] and values[i] > values[i+3]:
							break
						i += 1
				else:
					# find the local minimum after the mode
					while i < len(values) - 3:
						if values[i] < values[i+1] and values[i] < values[i+2] and values[i] < values[i+3]:
							break
						i += 1
				t = start + i * width
				if not self.domain.log_scale:
					t = 10 ** t
				results.append(t)
		return results

	def show_histogram(self):
		plt.figure(1)
		for c in range(len(self.domain.channels)):
			ch = self.domain.channels[c]
			plt.subplot(100 * len(self.domain.channels) + 10 + c + 1)
			self.__show_histogram(c)
		plt.show()

