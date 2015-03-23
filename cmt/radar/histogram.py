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
import numpy
import scipy
import scipy.special
import scipy.optimize

import matplotlib
import matplotlib.pyplot as plt

'''
This file contains tools for histogram based detection of water in radar images.
'''


class RadarHistogram(object):
    '''Detect water using a single split in the radar histogram.  Adapted from the paper:
        "Matgen, Hostache, Schumann, et. al. 'Towards an automated SAR-based flood monitoring system:
         Lessons learned from two case studies.' Physics and Chemistry of the Earth, 2011."'''
    BACKSCATTER_MODEL_GAMMA    = 1
    BACKSCATTER_MODEL_GAUSSIAN = 2
    BACKSCATTER_MODEL_DIP      = 3
    BACKSCATTER_MODEL_PEAK     = 4

    def __init__(self, domain, sensor, backscatter_model = None):
        self.domain = domain
        self.sensor = sensor
        self.backscatter_model = []
        for i in range(len(sensor.band_names)):

            try:
                if backscatter_model != None:
                    self.backscatter_model.append(backscatter_model)
                else:
                    if sensor.water_distributions[sensor.band_names[i]]['model'] == 'peak':
                        self.backscatter_model.append(RadarHistogram.BACKSCATTER_MODEL_PEAK)
                    elif sensor.water_distributions[sensor.band_names[i]]['model'] == 'dip':
                        self.backscatter_model.append(RadarHistogram.BACKSCATTER_MODEL_DIP)
                    elif sensor.water_distributions[sensor.band_names[i]]['model'] == 'lambda':
                        self.backscatter_model.append(RadarHistogram.BACKSCATTER_MODEL_GAMMA)
                    else:
                        self.backscatter_model.append(RadarHistogram.BACKSCATTER_MODEL_GAUSSIAN)
            except KeyError: # Skip bands without a distribution model
                pass
        
        self.hist_image = self.__preprocess_image(sensor)

        self.histograms = self.__compute_histogram(self.hist_image)
        self.__find_thresholds()
    
    def __preprocess_image(self, sensor):
        image = sensor.image
        if self.sensor.log_scale:
            image = image.log10()
        # clamp bands to specified water range
        for b in sensor.band_names:
            if 'range' in sensor.water_distributions[b]:
                other_bands = list(sensor.band_names)
                other_bands.remove(b)
                r     = sensor.water_distributions[b]['range']
                image = image.select([b], [b]).clamp(r[0], r[1]).addBands(image.select(other_bands, other_bands))
        return image
    
    def __compute_histogram(self, image):
        # buckets must be same for all bands
        buckets = 128 if 'buckets' not in self.sensor.water_distributions[self.sensor.band_names[0]] else self.sensor.water_distributions[self.sensor.band_names[0]]['buckets']
        histogram = image.reduceRegion(ee.Reducer.histogram(buckets, None, None), self.domain.bounds, 30, None, None, True).getInfo()
        h = []
    
        for c in range(len(self.sensor.band_names)):
            ch = self.sensor.band_names[c]
            # ignore first bucket, too many... for UAVSAR in particular
            #histogram[ch]['bucketMin'] += histogram[ch]['bucketWidth']
            #histogram[ch]['histogram'] =  histogram[ch]['histogram'][1:]
            
            # normalize
            total = sum(histogram[ch]['histogram'])
            histogram[ch]['histogram'] = map(lambda x : x / total, histogram[ch]['histogram'])
            h.append((histogram[ch]['bucketMin'], histogram[ch]['bucketWidth'], histogram[ch]['histogram']))
        return h

    def __cdf(self, params, x, backscatter_model):
        mode = params[0]
        k = params[1]
        offset = params[2]
        if backscatter_model == RadarHistogram.BACKSCATTER_MODEL_GAUSSIAN:
            return 0.5 * (1 + scipy.special.erf((x - mode) / (k * math.sqrt(2))))
        theta = (mode - offset) / (k - 1)
        return scipy.special.gammainc(k, (x - offset) / theta)

    # find x where __cdf(params, offset, x) = percentile
    def __cdf_percentile(self, params, percentile, backscatter_model):
        mode = params[0]
        k = params[1]
        offset = params[2]
        if backscatter_model == RadarHistogram.BACKSCATTER_MODEL_GAUSSIAN:
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
        plt.ylabel('pdf')
        plt.xlabel(self.sensor.band_names[channel] + ' Value')

        bm = self.backscatter_model[channel]
        
        (threshold, params) = self.__find_threshold_histogram(channel)
        if params != None:
            mid = int((params[0] - start) / width)
            cumulative = sum(values[:mid]) + values[mid] / 2
            scale = cumulative / self.__cdf(params, params[0], bm)
            plt.bar(ind, map(lambda x : scale * (self.__cdf(params, x + width, bm) - self.__cdf(params, x, bm)), ind), width=width, color='r', alpha=0.5)
        plt.plot((threshold, threshold), (0, 0.02), 'g--')

    def __gamma_function_errors(self, p, mode, fit_end, offset, channel):
        start  = self.histograms[channel][0]
        width  = self.histograms[channel][1]
        values = self.histograms[channel][2]
        k = p[0]
        bm = self.backscatter_model[channel]
        if (bm == RadarHistogram.BACKSCATTER_MODEL_GAMMA and k <= 1.0):
            return [float('inf')] * len(values)
        if (bm == RadarHistogram.BACKSCATTER_MODEL_GAUSSIAN and k <= 0.0):
            return [float('inf')] * len(values)
        error = 0.0
        last_cdf = 0.0
        errors = numpy.zeros(len(values))
        mid = int((mode - start) / width)
        cumulative = sum(values[:mid]) + values[mid] / 2
        scale = cumulative / self.__cdf((mode, k, offset), mode, bm)
        for i in range(len(values)):
            if start + i * width - offset > fit_end:
                break
            cdf = scale * self.__cdf((mode, k, offset), start + i * width, bm)
            errors[i] = (cdf - last_cdf) - values[i]
            last_cdf = cdf
        return errors

    def __find_threshold_histogram(self, channel):
        start  = self.histograms[channel][0]
        width  = self.histograms[channel][1]
        values = self.histograms[channel][2]
    
        # find the mode
        temp = self.sensor.water_distributions[self.sensor.band_names[channel]]['mode']
        (minv, maxv) = (temp['min'], temp['max'])
        i = int((minv - start) / width)
        biggest_bin = i
        while i < len(values) and start + i * width <= maxv:
            if values[i] > values[biggest_bin]:
                biggest_bin = i
            i += 1
        mode = start + biggest_bin * width
        
        if self.backscatter_model[channel] == RadarHistogram.BACKSCATTER_MODEL_PEAK:
            return (mode, None)

        # find the local minimum after the mode
        i = biggest_bin + 1
        while i < len(values)-3 and start + i * width <= maxv:
            if values[i] < values[i+1] and values[i] < values[i+2] and values[i] < values[i+3]:
                break
            i += 1
        local_min = start + i * width

        if self.backscatter_model[channel] == RadarHistogram.BACKSCATTER_MODEL_DIP:
            return (local_min, None)
    
        m = self.sensor.minimum_value
        if self.sensor.log_scale:
            m = math.log10(m)
        # find the other parameters of the distribution
        (value, result) = scipy.optimize.leastsq(self.__gamma_function_errors, [10], factor=1.0, args=(mode, mode, m, channel))
        params = (mode, value[0], m)
    
        # choose the best threshold where we can no longer discriminate
        mid = int((mode - start) / width)
        cumulative = sum(values[:mid]) + values[mid] / 2
        scale = cumulative / self.__cdf(params, params[0], self.backscatter_model[channel])
        i = mid
        while i < len(values):
            cumulative += values[i]
            diff = cumulative - scale * self.__cdf(params, start + i * width, self.backscatter_model[channel])
            if diff > 0.01:
                break
            i += 1
        threshold = start + i * width
    
        return (threshold, params)

    def __find_thresholds(self):
        self.thresholds = []
        self.distributions = []
        for c in range(len(self.sensor.band_names)):
            (threshold, params) = self.__find_threshold_histogram(c)
            if self.sensor.log_scale:
                threshold = 10 ** threshold
            self.thresholds.append(threshold)
            self.distributions.append(params)
    
    def get_thresholds(self):
        return self.thresholds
    
    def find_loose_thresholds(self, percentile=0.99):
        results = []
        for c in range(len(self.sensor.band_names)):
            (threshold, params) = (self.thresholds[c], self.distributions[c])
            if params != None:
                t = self.__cdf_percentile(params, percentile, self.backscatter_model[c])
                if self.sensor.log_scale:
                    t = 10 ** t
                results.append(t)
            else:
                # if finding dip or peak, find next local min / max as threshold
                if self.sensor.log_scale:
                    threshold = math.log10(threshold)
                start = self.histograms[c][0]
                width = self.histograms[c][1]
                values = self.histograms[c][2]
                i = int((threshold - start) / width) + 1
                if self.backscatter_model[c] == RadarHistogram.BACKSCATTER_MODEL_DIP:
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
                if self.sensor.log_scale:
                    t = 10 ** t
                results.append(t)
        return results

    def show_histogram(self):
        plt.figure(1)
        for c in range(len(self.sensor.band_names)):
            ch = self.sensor.band_names[c]
            plt.subplot(100 * len(self.sensor.band_names) + 10 + c + 1)
            self.__show_histogram(c)
        plt.show()


#=========================================================================================
# This is another method which has not been integrated into the class structure above.
# - If you need to use it for something, go for it!

def __computeJT(relHist, binVals, T):
    '''As part of the Kittler/Illingworth method, compute J(T) for a given T'''

    FAIL_VAL = 999999

    # Split the hostogram at the threshold T.
    histogram1 = relHist[0:T]
    histogram2 = relHist[T+1:]

    # Compute the number of pixels in the two classes.
    P1 = sum(histogram1)
    P2 = sum(histogram2)

    # Only continue if both classes contain at least one pixel.
    if (P1 <= 0) or (P2 <= 0):
        return FAIL_VAL

    # Compute the standard deviations of the classes.
    weightedBins1 = numpy.multiply(histogram1, binVals[0:T])
    weightedBins2 = numpy.multiply(histogram2, binVals[T+1:])
    mean1         = sum(weightedBins1) / P1;
    mean2         = sum(weightedBins2) / P2;
    diffs1        = numpy.subtract(binVals[0:T],  mean1)
    diffs2        = numpy.subtract(binVals[T+1:], mean2)
    diffsSq1      = [d*d for d in diffs1]
    diffsSq2      = [d*d for d in diffs2]
    weightedBins1 = numpy.multiply(histogram1, diffsSq1)
    weightedBins2 = numpy.multiply(histogram2, diffsSq2)
    sigma1        = math.sqrt(sum(weightedBins1) / P1)
    sigma2        = math.sqrt(sum(weightedBins2) / P2)

    # Make sure both classes contain at least two intensity values.
    if (sigma1 <= 0) or (sigma2 <= 0):
        return FAIL_VAL

    # Compute J(T).
    J = 1 + 2*(P1*math.log(sigma1) + P2*math.log(sigma2)) - 2*(P1*math.log(P1) + P2*math.log(P2))
    
    return J

def splitHistogramKittlerIllingworth(histogram, binVals):
    '''Tries to compute an optimal histogram threshold using the Kittler/Illingworth method'''

    # Normalize the histogram (each bin is now a percentage)
    histSum      = float(sum(histogram))
    relativeHist = numpy.divide(histogram,histSum)


    # Try out every bin value in the histogram and pick the best one
    #  - For more resolution, use more bins!
    #  - Could write up a smarter solver for this.
    numBins = len(binVals)
    J = []
    for i in range(0, numBins):
        J.append(__computeJT(relativeHist, binVals, i))
        
    minJ      = J[1]
    threshold = binVals[0]
    for i in range(1, numBins):
        if J[i] < minJ:
            minJ      = J[i]
            threshold = (binVals[i] + binVals[i-1])/2 # Threshold is below current bin value
    
    return threshold


