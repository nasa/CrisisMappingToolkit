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
import threading
import functools

class WaitForResult(threading.Thread):
    '''Starts up a thread to run a pair of functions in series'''

    def __init__(self, function, finished_function = None):
        threading.Thread.__init__(self)
        self.function = function # Main function -> Run this!
        self.finished_function = finished_function # Run this after the main function is finished
        self.setDaemon(True) # Don't hold up the program on this thread
        self.start()
    def run(self):
        self.finished_function(self.function())

def __evaluate_approach(result, ground_truth, region, fractional=False):
    '''Compare result to ground truth in region and compute precision and recall'''

    if fractional:  # Apply a MODIS pixel sized smoothing kernel ground truth
        ground_truth = ground_truth.convolve(ee.Kernel.square(250, 'meters', True))
    
    
    # wrong  = ground_truth.multiply(-1).add(1).min(result);
    # missed = ground_truth.subtract(result).max(0.0);
    correct     = ground_truth.min(result);
    correct_sum = ee.data.getValue({'image': correct.stats(     30000, region, 'EPSG:4326').serialize(), 'fields': 'b1'})['properties']['b1']['values']['sum']
    result_sum  = ee.data.getValue({'image': result.stats(      30000, region, 'EPSG:4326').serialize(), 'fields': 'b1'})['properties']['b1']['values']['sum']
    truth_sum   = ee.data.getValue({'image': ground_truth.stats(30000, region, 'EPSG:4326').serialize(), 'fields': 'b1'})['properties']['b1']['values']['sum']
    # correct_sum = correct.reduceRegion(ee.Reducer.sum(), region, 30).getInfo()['b1']
    # result_sum  = result.reduceRegion(ee.Reducer.sum(), region, 30).getInfo()['b1']
    # truth_sum = ground_truth.reduceRegion(ee.Reducer.sum(), region, 30).getInfo()['b1']
    precision   = 1.0 if (result_sum == 0.0) else (correct_sum / result_sum)
    recall      = 1.0 if (truth_sum  == 0.0) else (correct_sum / truth_sum)
    return (precision, recall)


def evaluate_approach(evaluation_function, result, ground_truth, region, fractional=False):
    '''Computes precision and recall of the given result/ground truth pair, then passes the result to the input function'''
    WaitForResult(functools.partial(__evaluate_approach, result=result, ground_truth=ground_truth,
        region=region, fractional=fractional), evaluation_function)

