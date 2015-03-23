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
from cmt.mapclient_qt import centerMap, addToMap

from histogram import RadarHistogram



def grow_regions(sensor, thresholded, thresholds):
    '''???????'''
    REGION_GROWTH_RANGE = 20
    neighborhood_kernel = ee.Kernel.square(1, 'pixels', False)
    loose_thresholded   = sensor.image.select([sensor.band_names[0]]).lte(thresholds[0])
    for i in range(1, len(sensor.band_names)):
        loose_thresholded = loose_thresholded.And(sensor.image.select([sensor.band_names[i]]).lte(thresholds[i]))
    addToMap(loose_thresholded, {'min': 0, 'max': 1}, 'Loose', False)
    for i in range(REGION_GROWTH_RANGE):
        thresholded = thresholded.convolve(neighborhood_kernel).And(loose_thresholded)
    return thresholded

def threshold(domain, historical_domain=None):
    '''An implementation of the paper:
           Matgen, Hostache, Schumann, et. al. "Towards an automated SAR-based flood monitoring system:
           Lessons learned from two case studies." Physics and Chemistry of the Earth, 2011.
           
        TODO: ????
    '''    
    
    sensor = domain.get_radar() # Get the first radar sensor available
    hist   = RadarHistogram(domain, sensor)
    
    thresholds = hist.get_thresholds() # Get thresholds for each band
    
    # For each band, get pixels less than a threshold
    results = []
    for c in range(len(thresholds)):
        ch = sensor.band_names[c]
        results.append(sensor.image.select([ch], [ch]).lte(thresholds[c]))

    #?????
    result_image = results[0]
    for c in range(1, len(results)):
        result_image = result_image.addBands(results[c], [sensor.band_names[c]])
    addToMap(result_image, {'min': 0, 'max': 1}, 'Color Image', False)

    # TODO: Compare water pixels to expected distribution
    #       take difference of two image, remove pixels that aren't below
    #       original non region-growing threshold and don't change by at
    #       least fixed amount
    
    #?????
    result_image = results[0].select([sensor.band_names[0]], ['b1'])
    for c in range(1, len(results)):
        result_image = result_image.And(results[c])
    
    #????
    growing_thresholds = hist.find_loose_thresholds()
    result_image       = grow_regions(sensor, result_image, growing_thresholds)
    
    #hist.show_histogram() # This is useful for debugging

    return result_image

