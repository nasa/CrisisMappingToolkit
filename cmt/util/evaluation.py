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
import time
import cmt.util.miscUtilities
#import cmt.mapclient_qt


def countNumBlobs(classifiedImage, region, maxBlobSize, evalResolution=500): # In pixels?
    '''Count the number of unconnected blobs in an image'''
    
    # Count up the number of islands smaller than a certain size
    antiResult = classifiedImage.Not()
    onBlobs   = classifiedImage.connectedComponents(ee.Kernel.square(3), maxBlobSize).select('b1')
    offBlobs  = antiResult.connectedComponents(     ee.Kernel.square(3), maxBlobSize).select('b1')
    vectorsOn  = onBlobs.reduceToVectors(scale=evalResolution, geometry=region,
                                         geometryType='centroid', bestEffort=True)
    vectorsOff = offBlobs.reduceToVectors(scale=evalResolution, geometry=region,
                                         geometryType='centroid', bestEffort=True)
    numOnBlobs  = len(vectorsOn.getInfo()['features'])
    numOffBlobs = len(vectorsOff.getInfo()['features'])
    return (numOnBlobs, numOffBlobs)
    

def evaluate_result_quality(resultIn, region):
    '''Try to appraise the quality of a result without access to ground truth data!'''
    
    
    EVAL_RESOLUTION = 500
    
    waterMask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'], ['b1'])
    
    # Check percentage of region classified as true
    result = resultIn.round().uint8() # Eliminate fractional inputs
    fillCount         = result.reduceRegion(ee.Reducer.mean(), region, EVAL_RESOLUTION)
    percentClassified = fillCount.getInfo()['b1']

    #print 'percentClassified = ' + str(percentClassified)

    # Too much or too little fill generally indicates a bad match
    MAX_FILL_PERCENT = 0.95
    MIN_FILL_PERCENT = 0.05    
    if (percentClassified < MIN_FILL_PERCENT) or (percentClassified > MAX_FILL_PERCENT):
        return 0.0

    # Make sure enough of the water mask has been filled in
    MIN_PERCENT_MASK_FILL = 0.60
    filledWaterMask      = waterMask.And(result)
    filledWaterCount     = filledWaterMask.reduceRegion(ee.Reducer.sum(), region, EVAL_RESOLUTION).getInfo()['b1']
    waterMaskCount       = waterMask.reduceRegion(ee.Reducer.sum(), region, EVAL_RESOLUTION).getInfo()['b1']
    if waterMaskCount == 0: # Can't do much without the water mask!
        return 1.0 # Give it the benefit of the doubt.
    waterMaskPercentFill = filledWaterCount / waterMaskCount
    #print 'Water mask percent fill = ' + str(waterMaskPercentFill)
    if waterMaskPercentFill < MIN_PERCENT_MASK_FILL:
        return 0.0
   
    # Count up the number of islands smaller than a certain size
    MAX_SPECK_SIZE = 150 # In pixels?   
    (waterSpecks, landSpecks) = countNumBlobs(result, region, MAX_SPECK_SIZE, EVAL_RESOLUTION)
    #print 'Found ' + str(waterSpecks) + ' water specks'
   
    # Count up the number of islands in the water mask -> Only need to do this once!
    (waterMaskSpecks, landMaskSpecks) = countNumBlobs(waterMask, region, MAX_SPECK_SIZE, EVAL_RESOLUTION)
    #print 'Found ' + str(waterMaskSpecks) + ' water mask specks'

    # Floods tend to reduce the number of isolated water bodies, not increase them.
    MAX_RATIO = 10
    waterSpeckRatio = waterSpecks / waterMaskSpecks
    landSpeckRatio  = landSpecks  / landMaskSpecks
    #print 'waterSpeckRatio = ' + str(waterSpeckRatio)
    #print 'landSpeckRatio  = ' + str(landSpeckRatio)
    if (waterSpeckRatio > MAX_RATIO) or (landSpeckRatio > MAX_RATIO):
        return 0
    
    # At this point all of the pass/fail checks have passed.
    # Compute a final percentage by assesing some penalties
    score = 1.0
    penalty = min(max(1.0 - waterMaskPercentFill, 0), 0.4)
    score  -= penalty
    penalty = min(max(waterSpeckRatio - 1.0, 0)/10.0, 0.3)
    score  -= penalty
    penalty = min(max(landSpeckRatio - 1.0, 0)/10.0, 0.3)
    score  -= penalty
    
    return score


def evaluate_approach(result, ground_truth, region, fractional=False):
    '''Compare result to ground truth in region and compute precision and recall'''
    ground_truth = ground_truth.mask(ground_truth.mask().And(result.mask()))

    # TODO: Fix this!
    if fractional:  # Apply a MODIS pixel sized smoothing kernel ground truth
        ground_truth = ground_truth.convolve(ee.Kernel.square(250, 'meters', True))
    
    # Correct detections mean water detected in the same location.
    # - This does not include correct non-detections!
    correct = ground_truth.min(result)
    
    # Keep reducing the evaluation resolution until Earth Engine finishes without timing out
    MIN_EVAL_POINTS =  5000
    eval_points     = 60000
    while True:
        try:
            # This probably works now
            #correct_sum = correct.reduceRegion(     ee.Reducer.sum(), region, eval_res, 'EPSG:4326' ).getInfo()['b1'] # Correct detections
            #result_sum  = result.reduceRegion(      ee.Reducer.sum(), region, eval_res, 'EPSG:4326' ).getInfo()['b1'] # Total detections
            #truth_sum   = ground_truth.reduceRegion(ee.Reducer.sum(), region, eval_res, 'EPSG:4326' ).getInfo()['b1'] # Total water
    
            # Evaluate the results at a large number of random sample points
            correct_sum = ee.data.getValue({'image': correct.stats(     eval_points, region, 'EPSG:4326').serialize(), 'fields': 'b1'})['properties']['b1']['values']['sum']
            result_sum  = ee.data.getValue({'image': result.stats(      eval_points, region, 'EPSG:4326').serialize(), 'fields': 'b1'})['properties']['b1']['values']['sum']
            truth_sum   = ee.data.getValue({'image': ground_truth.stats(eval_points, region, 'EPSG:4326').serialize(), 'fields': 'b1'})['properties']['b1']['values']['sum']
            
            break # Quit the loop if the calculations were successful
        except Exception as e: # On failure coursen the resolution and try again
            print(str(e))
            eval_points /= 2
            if eval_points < MIN_EVAL_POINTS:
                raise Exception('Unable to evaluate results at resolution ' + str(eval_points*2))

    # Compute ratios, avoiding divide by zero.
    precision   = 1.0 if (result_sum == 0.0) else (correct_sum / result_sum)
    recall      = 1.0 if (truth_sum  == 0.0) else (correct_sum / truth_sum)
    
    if (precision > 1.0) or (recall > 1.0):
        print('EVALUATION_ERROR')
        print('correct_sum = ' + str(correct_sum))
        print('result_sum  = ' + str(result_sum))
        print('truth_sum   = ' + str(truth_sum))
        #cmt.mapclient_qt.addToMap(correct, {}, 'CORRECT')
 
    ## A test of our result evaluation that does not depend on the ground truth!
    #no_truth_result = evaluate_result_quality(result, region)
    no_truth_result = 0 # For now skip calculating this to reduce the computation time
    
    return (precision, recall, eval_points, no_truth_result)


def evaluate_approach_thread(evaluation_function, result, ground_truth, region, fractional=False):
    '''Computes precision and recall of the given result/ground truth pair, then passes the result to the input function'''
    cmt.util.miscUtilities.waitForEeResult(functools.partial(evaluate_approach, result=result, ground_truth=ground_truth,
                                           region=region, fractional=fractional), evaluation_function)

