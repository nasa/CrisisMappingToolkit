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

from cmt.mapclient_qt import addToMap
from cmt.util.miscUtilities import safe_get_info, get_permanent_water_mask
from modis_utilities import *

'''
Contains algorithms that do not go in any of the other files!
'''


#=====================================================================================


def martinis_tree(domain, b):
    '''Based on Figure 3 from "A Multi-Scale Flood Monitoring System Based on Fully
        Automatic MODIS and TerraSAR-X Processing Chains" by 
        Sandro Martinis, Andre Twele, Christian Strobl, Jens Kersten and Enrico Stein
        
       Some steps had to be approximated such as the cloud filtering.  The main pixel
       classification steps are implemented as accurately as can be determined from 
       the figure.
'''
    # Note: Nodes with the same name are numbered top to bottom, left to right in order
    #       to distinguish which one a variable represents.

    # ---- Apply the initial classification at the top of the figure ----
    # Indices already computed in the 'b' object
    clouds        = b['pBLUE'].gte(0.27)
    temp1         = b['EVI'].lte(0.3 ).And(b['DVEL'].lte(0.05))
    temp2         = b['EVI'].lte(0.05).And(b['LSWI'].lte(0.00))
    nonFlood1     = b['EVI'].gt(0.3).And(temp1.Not())
    waterRelated1 = temp1.Or(temp2)
    nonFlood2     = temp1.And(temp2.Not())
    
    # ---- Apply DEM filtering ----
    
    # Retrieve the dem compute slopes
    demSensor       = domain.get_dem()
    dem             = demSensor.image
    demSlopeDegrees = compute_dem_slope_degrees(demSensor.image, demSensor.band_resolutions[demSensor.band_names[0]])
    #addToMap(demSlopeDegrees, {'min': 0, 'max': 90}, 'DEM slope',  False)
    
    # Filter regions of high slope
    highSlope = demSlopeDegrees.gt(10).Or( demSlopeDegrees.gt(8).And(dem.gt(2000)) )
    
    waterRelated2 = waterRelated1.And(highSlope.Not())
    nonFlood3     = waterRelated1.And(highSlope).Or(nonFlood2)
    
    mixture1 = waterRelated2.And(b['EVI'].gt(0.1))
    flood1   = waterRelated2.And(b['EVI'].lte(0.1))
    
    # ---- Approximate region growing ----
    
    # Earth Engine can't do a real region grow so approximate one with a big dilation
    REGION_GROW_SIZE = 35 # Pixels
    expansionKernel  = ee.Kernel.circle(REGION_GROW_SIZE, 'pixels', False)
    
    # Region growing 1
    temp1 = b['EVI'].lte(0.31).And(b['DVEL'].lte(0.07))
    temp2 = b['EVI'].lte(0.06).And(b['LSWI'].lte(0.01))
    relaxedConditions      = temp1.Or(temp2)
    potentialExpansionArea = waterRelated2.convolve(expansionKernel).And(nonFlood3)
    waterRelated3          = potentialExpansionArea.And(relaxedConditions)
    
    mixture3 = waterRelated3.And(b['EVI'].gt(0.1))
    flood3   = waterRelated3.And(b['EVI'].lte(0.1))
    
    # Region growing 2
    potentialExpansionArea = flood1.convolve(expansionKernel).And(mixture1)
    mixture2 = potentialExpansionArea.And(b['LSWI'].lt(0.08))
    flood2   = potentialExpansionArea.And(b['LSWI'].gte(0.08))
    
    # ---- Apply water mask ----
    
    waterMask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'])

    # Not sure how exactly the paper does this but we don't care about anything in the permanent water mask!
    recedingWater = nonFlood3.And(waterMask)
    
    mergedFlood   = flood1.Or(flood2).Or(flood3)
    standingWater = mergedFlood.And(waterMask)
    flood4        = mergedFlood.And(waterMask.Not())

    # ---- Cloud handling? ----

    # In place of the complicated cloud geometery, just remove pixels
    #  that are flagged as bad in the input MODIS imagery.
    badModisPixels = getModisBadPixelMask(domain.modis.image)

    flood5 = flood4.And(badModisPixels.Not())

    # ---- Time series handling ----
    # --> This step is not included

    fullMixture = mixture2.Or(mixture3)

    # Generate the binary flood detection output we are interested in
    outputFlood = flood5.Or(standingWater)

    return outputFlood.select(['sur_refl_b02'], ['b1']) # Rename sur_refl_b02 to b1


    
#==============================================================
# Experimental classifiers

def history_diff(domain, b):
    '''Wrapper function for passing domain data into history_diff_core'''
    
    # Load pre-selected constants for this domain   
    dev_thresh    = float(domain.algorithm_params['modis_mask_threshold' ])
    change_thresh = float(domain.algorithm_params['modis_change_threshold'])
    
    # Call the core algorithm with all the parameters it needs from the domain
    return history_diff_core(domain.modis, domain.modis.get_date(), dev_thresh, change_thresh, domain.bounds)

    
def history_diff_core(high_res_modis, date, dev_thresh, change_thresh, bounds):
    '''Leverage historical data and the permanent water mask to improve the threshold method.
    
       This method computes statistics from the permanent water mask to set a good b2-b1
       water detection threshold.  It also adds pixels which have a b2-b1 level significantly
       lower than the historical seasonal average.
    '''

    # Retrieve all the MODIS images for the region in the last several years
    NUM_YEARS_BACK         = 5
    NUM_DAYS_COMPARE_RANGE = 40.0 # Compare this many days before/after the target day in previous years
    YEAR_RANGE_PERCENTAGE  = NUM_DAYS_COMPARE_RANGE / 365.0
    #print 'YEAR_RANGE_PERCENTAGE = ' + str(YEAR_RANGE_PERCENTAGE)
    #print 'Start: ' + str(domain.date.advance(-1 - YEAR_RANGE_PERCENTAGE, 'year'))
    #print 'End:   ' + str(domain.date.advance(-1 + YEAR_RANGE_PERCENTAGE, 'year'))

    # Get both the high and low res MODIS data    
    historyHigh = ee.ImageCollection('MOD09GQ').filterDate(date.advance(-1 - YEAR_RANGE_PERCENTAGE, 'year'), date.advance(-1 + YEAR_RANGE_PERCENTAGE, 'year')).filterBounds(bounds);
    historyLow  = ee.ImageCollection('MOD09GA').filterDate(date.advance(-1 - YEAR_RANGE_PERCENTAGE, 'year'), date.advance(-1 + YEAR_RANGE_PERCENTAGE, 'year')).filterBounds(bounds);
    for i in range(1, NUM_YEARS_BACK-1):
        yearMin = -(i+1) - YEAR_RANGE_PERCENTAGE
        yearMax = -(i+1) + YEAR_RANGE_PERCENTAGE
        historyHigh.merge(ee.ImageCollection('MOD09GQ').filterDate(date.advance(yearMin, 'year'), date.advance(yearMax, 'year')).filterBounds(bounds));
        historyLow.merge( ee.ImageCollection('MOD09GA').filterDate(date.advance(yearMin, 'year'), date.advance(yearMax, 'year')).filterBounds(bounds));
        # TODO: Add a filter here to remove cloud-filled images
    
    # Simple function implements the b2 - b1 difference method
    # - This needs to work with domains and with the input from production_gui
    def flood_diff_function(image):
        try:
            return image.select(['sur_refl_b02']).subtract(image.select(['sur_refl_b01']))
        except:
            return image.sur_refl_b02.subtract(image.sur_refl_b01)
    
    # Apply difference function to all images in history, then compute mean and standard deviation of difference scores.
    historyDiff   = historyHigh.map(flood_diff_function)
    historyMean   = historyDiff.mean()
    historyStdDev = historyDiff.reduce(ee.Reducer.stdDev())
    
    # Display the mean image for bands 1/2/6
    history3  = historyHigh.mean().select(['sur_refl_b01', 'sur_refl_b02']).addBands(historyLow.mean().select(['sur_refl_b06']))
    #addToMap(history3, {'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06'], 'min' : 0, 'max': 3000, 'opacity' : 1.0}, 'MODIS_HIST', False)
    
    #addToMap(historyMean,   {'min' : 0, 'max' : 4000}, 'History mean',   False)
    #addToMap(historyStdDev, {'min' : 0, 'max' : 2000}, 'History stdDev', False)
    
    # Compute flood diff on current image and compare to historical mean/STD.
    floodDiff   = flood_diff_function(high_res_modis)   
    diffOfDiffs = floodDiff.subtract(historyMean)
    ddDivDev    = diffOfDiffs.divide(historyStdDev)
    changeFlood = ddDivDev.lt(change_thresh)  # Mark all pixels which are enough STD's away from the mean.
    
    #addToMap(floodDiff,   {'min' : 0, 'max' : 4000}, 'floodDiff',   False)
    #addToMap(diffOfDiffs, {'min' : -2000, 'max' : 2000}, 'diffOfDiffs', False)
    #addToMap(ddDivDev,    {'min' : -10,   'max' : 10}, 'ddDivDev',    False)
    #addToMap(changeFlood,    {'min' : 0,   'max' : 1}, 'changeFlood',    False)
    #addToMap(domain.water_mask,    {'min' : 0,   'max' : 1}, 'Permanent water mask',    False)
    
    # Compute the difference statistics inside permanent water mask pixels
    MODIS_RESOLUTION = 250 # Meters
    water_mask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'])
    diffInWaterMask  = floodDiff.multiply(water_mask)
    maskedMean       = diffInWaterMask.reduceRegion(ee.Reducer.mean(),   bounds, MODIS_RESOLUTION)
    maskedStdDev     = diffInWaterMask.reduceRegion(ee.Reducer.stdDev(), bounds, MODIS_RESOLUTION)
    
    # Use the water mask statistics to compute a difference threshold, then find all pixels below the threshold.
    waterThreshold  = safe_get_info(maskedMean)['sur_refl_b02'] + dev_thresh*(safe_get_info(maskedStdDev)['sur_refl_b02']);
    #print 'Water threshold == ' + str(waterThreshold)
    waterPixels     = flood_diff_function(high_res_modis).lte(waterThreshold)
    #waterPixels     = modis_diff(domain, b, waterThreshold)
    
    #addToMap(waterPixels,    {'min' : 0,   'max' : 1}, 'waterPixels',    False)
    #
    ## Is it worth it to use band 6?
    #B6_DIFF_AMT = 500
    #bHigh1 = (b['b6'].subtract(B6_DIFF_AMT).gt(b['b1']))
    #bHigh2 = (b['b6'].subtract(B6_DIFF_AMT).gt(b['b2']))
    #bHigh = bHigh1.And(bHigh2).reproject('EPSG:4326', scale=MODIS_RESOLUTION)
    #addToMap(bHigh.mask(bHigh),    {'min' : 0,   'max' : 1}, 'B High',    False)
    
    # Combine water pixels from the historical and water mask methods.
    return (waterPixels.Or(changeFlood)).select(['sur_refl_b02'], ['b1'])#.And(bHigh.eq(0));


