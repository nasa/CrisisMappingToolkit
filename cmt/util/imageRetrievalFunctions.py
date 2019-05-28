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
import os
import functools
import cmt.modis.modis_utilities
import cmt.util.landsat_functions
import miscUtilities


#=================================================================================
# A set of functions to fetch an image collection given a date range

def get_image_collection_landsat(bounds, start_date, end_date, collectionName='LT5_L1T'):
    '''Retrieve Landsat imagery for the selected location and dates.'''

    ee_bounds  = bounds
    ee_points  = ee.List(bounds.bounds().coordinates().get(0))
    points     = ee_points.getInfo()
    points     = map(functools.partial(apply, ee.Geometry.Point), points)
#    collection = ee.ImageCollection(collectionName).filterDate(start_date, end_date) \
#                                    .filterBounds(points[0]).filterBounds(points[1]) \
#                                    .filterBounds(points[2]).filterBounds(points[3])
    collection = ee.ImageCollection(collectionName).filterDate(start_date, end_date) \
                                    .filterBounds(bounds.centroid())
                                    
    # Select and rename the bands we want
    temp = cmt.util.landsat_functions.rename_landsat_bands(collection, collectionName)
    return temp.sort('system:time_start')
    



def get_image_collection_modis(region, start_date, end_date):
    '''Retrieve MODIS imagery for the selected location and dates.
       This merges the high-res and low-res MODIS bands into one collection
       and cleans up the band names slightly.'''

    ee_points    = ee.List(region.bounds().coordinates().get(0))
    points       = ee_points.getInfo()
    points       = map(functools.partial(apply, ee.Geometry.Point), points)
    highResModisTerra = ee.ImageCollection('MOD09GQ').filterDate(start_date, end_date) \
                                 .filterBounds(points[0]).filterBounds(points[1]) \
                                 .filterBounds(points[2]).filterBounds(points[3])
    lowResModisTerra  = ee.ImageCollection('MOD09GA').filterDate(start_date, end_date) \
                                 .filterBounds(points[0]).filterBounds(points[1]) \
                                 .filterBounds(points[2]).filterBounds(points[3])
    highResModisAqua = ee.ImageCollection('MYD09GQ').filterDate(start_date, end_date) \
                                 .filterBounds(points[0]).filterBounds(points[1]) \
                                 .filterBounds(points[2]).filterBounds(points[3])
    lowResModisAqua  = ee.ImageCollection('MYD09GA').filterDate(start_date, end_date) \
                                 .filterBounds(points[0]).filterBounds(points[1]) \
                                 .filterBounds(points[2]).filterBounds(points[3])

    # This set of code is needed to merge the low and high res MODIS bands
    def merge_bands(element):
        # A function to merge the bands together.
        # After a join, results are in 'primary' and 'secondary' properties.       
        return ee.Image.cat(element.get('primary'), element.get('secondary'))
        
    def merge_and_clean(lowRes, highRes):
        '''Call merge_bands and clean up the band names'''
        join          = ee.Join.inner()
        f             = ee.Filter.equals('system:time_start', None, 'system:time_start')
        modisJoined   = ee.ImageCollection(join.apply(lowRes, highRes, f));
        roughJoined   = modisJoined.map(merge_bands);
        # Clean up the joined band names
        band_names_in = ['num_observations_1km','state_1km','SensorZenith','SensorAzimuth','Range','SolarZenith','SolarAzimuth','gflags','orbit_pnt',
                         'num_observations_500m','sur_refl_b03','sur_refl_b04','sur_refl_b05','sur_refl_b06','sur_refl_b07',
                         'QC_500m','obscov_500m','iobs_res','q_scan','num_observations', 'sur_refl_b01_1','sur_refl_b02_1','QC_250m','obscov']
        band_names_out = ['num_observations_1km','state_1km','SensorZenith','SensorAzimuth','Range','SolarZenith','SolarAzimuth','gflags','orbit_pnt',
                          'num_observations_500m','sur_refl_b03','sur_refl_b04','sur_refl_b05','sur_refl_b06','sur_refl_b07',
                          'QC_500m','obscov_500m','iobs_res','q_scan','num_observations_250m', 'sur_refl_b01','sur_refl_b02','QC_250m','obscov']
        collection    = roughJoined.select(band_names_in, band_names_out)
        return collection
        
    modisAqua  = merge_and_clean(lowResModisAqua,  highResModisAqua)
    modisTerra = merge_and_clean(lowResModisTerra, highResModisTerra)
    collection = modisAqua.merge(modisTerra)
        
    return collection.sort('system:time_start')


def get_image_collection_sentinel1(bounds, start_date, end_date, min_images=1):
    '''Retrieve Sentinel1 imagery for the selected location and dates.
       This function tries different Sentinel1 settings in order of 
       decreasing quality until it finds at least min_images using
       a combination of settings.'''

    # Currently we try to load both VV and VH polarization (the most common).
    # This leaves us with two variables that we don't want to mix up.
    # - TODO: Can we mix the different resolutions?
    resolutionList = [10.0, 25.0, 40.0]
    angleList      = ['ASCENDING', 'DESCENDING']

    ee_bounds  = bounds
    ee_points  = ee.List(bounds.bounds().coordinates().get(0))
    points     = ee_points.getInfo()
    points     = map(functools.partial(apply, ee.Geometry.Point), points)
    print('Searching for S1 centroid: ' + str(bounds.centroid().getInfo()))
    
    # Loop through the variable combinations
    for resolution in resolutionList:
        print('Searching S1 resolution: ' + str(resolution))
        for angle in angleList:
            print('Searching S1 angle: ' + str(angle))
            collection = ee.ImageCollection('COPERNICUS/S1_GRD').filterDate(start_date, end_date) \
                          .filterBounds(bounds.centroid()) \
                          .filter(ee.Filter.eq('resolution_meters',    resolution)) \
                          .filter(ee.Filter.eq('orbitProperties_pass', angle))
#                                            .filterBounds(points[0]).filterBounds(points[1]) \
#                                            .filterBounds(points[2]).filterBounds(points[3]) \

            # Accept the result if we got as many images as the user requested.
            numFound = collection.size().getInfo()
            print('Found ' + str(numFound) + ' images')
            if numFound >= min_images:
                # Switch band names to lower case to be consistent with domain notation               
                temp = cmt.util.miscUtilities.safeRename(collection, ['VV', 'VH'], ['vv', 'vh'])
                return temp.sort('system:time_start')

                                            
    return collection # Failed to find anything!


#=================================================================================
# A set of functions to find a cloud free image near a date


def getIndicesSortedByNearestDate(imageList, targetDate):
    '''Sort an EE image list by distance from a target date.
       Got a weird error trying to return a list, so returning indices instead.'''

    numFound = imageList.length().getInfo()

    # Compute the time dists from 
    targetTimeMs = targetDate.millis().getInfo()
    dists = []
    for i in range(0,numFound):
        thisTime = ee.Image(imageList.get(i)).get('system:time_start').getInfo()
        diff = abs(targetTimeMs - thisTime)
        dists.append( (i, diff) )

    # Sort the indices based on the time diffs
    def getKey(item):  # Fetches the time from a pair
        return item[1]
    sortedVals = sorted(dists, key=getKey)

    sortedIndexList = []
    for (index, diff) in sortedVals:
        sortedIndexList.append(index)
    
    return sortedIndexList

def getCloudFreeModis(bounds, targetDate, maxRangeDays=10, maxCloudPercentage=0.05,
                      minCoverage=0.8, searchMethod='nearest'):
    '''Search for the closest cloud-free MODIS image near the target date.
       The result preference is determined by searchMethod and can be  set to:
       nearest, increasing, or decreasing'''
    
    # Get the date range to search
    if searchMethod == 'nearest':
        dateStart = targetDate.advance(-1*maxRangeDays, 'day')
        dateEnd   = targetDate.advance(   maxRangeDays, 'day')
    else:
        dateStart = targetDate
        dateEnd   = targetDate.advance(   maxRangeDays, 'day')
    
    # Get a list of candidate images
    imageCollection = get_image_collection_modis(bounds, dateStart, dateEnd)
    imageList       = imageCollection.toList(100)
    imageInfo       = imageList.getInfo()
    numFound        = len(imageInfo)
    
    #print 'Modis dates:'
    #print dateStart.format().getInfo()
    #print dateEnd.format().getInfo()
    print('Found ' + str(numFound) + ' candidate MODIS images.')   

    # Find the first image that meets the requirements
    if searchMethod == 'nearest':
        # The images start sorted by time, change to sort by distance from target.
        searchIndices = getIndicesSortedByNearestDate(imageList, targetDate)
    elif searchMethod == 'increasing':
        searchIndices = range(0,numFound)
    else:
        searchIndices = range(numFound-1, -1, -1)
    #print searchIndices
    for i in searchIndices:
        COVERAGE_RES = 250
        thisImage       = ee.Image(imageList.get(i)).resample('bicubic')
        percentCoverage = thisImage.mask().reduceRegion(ee.Reducer.mean(), bounds, COVERAGE_RES).getInfo().values()[0]
        print('percentCoverage = ' + str(percentCoverage))
        if percentCoverage < minCoverage: # MODIS has high coverage, but there are gaps.
            continue
        cloudPercentage = cmt.modis.modis_utilities.getCloudPercentage(thisImage, bounds)
        print('Detected MODIS cloud percentage: ' + str(cloudPercentage))
        if cloudPercentage < maxCloudPercentage:
            return thisImage

    raise Exception('Could not find a nearby cloud-free MODIS image for date ' + str(targetDate.getInfo()))


def getCloudFreeLandsat(bounds, targetDate, maxRangeDays=10, maxCloudPercentage=0.05,
                        minCoverage=0.8, searchMethod='nearest'):
    '''Search for the closest cloud-free Landsat image near the target date.
       The result preference is determined by searchMethod and can be  set to:
       nearest, increasing, or decreasing'''

    # Get the date range to search
    if searchMethod == 'nearest':
        dateStart = targetDate.advance(-1*maxRangeDays, 'day')
        dateEnd   = targetDate.advance(   maxRangeDays, 'day')
    else:
        dateStart = targetDate
        dateEnd   = targetDate.advance(   maxRangeDays, 'day')

    # Try each of the satellites in decreasing order of quality
    collectionNames = ['LC8_L1T_TOA', 'LT5_L1T_TOA', 'LE7_L1T_TOA']
    for name in collectionNames:

        # Get candidate images for this sensor
        imageCollection = get_image_collection_landsat(bounds, dateStart, dateEnd, name)
        imageList       = imageCollection.toList(100)
        imageInfo       = imageList.getInfo()
        
        # Find the first image that meets the requirements
        numFound = len(imageInfo)
        if searchMethod == 'nearest':
            # The images start sorted by time, change to sort by distance from target.
            searchIndices = getIndicesSortedByNearestDate(imageList, targetDate)
        elif searchMethod == 'increasing':
            searchIndices = range(0,numFound)
        else:
            searchIndices = range(numFound-1, -1, -1)
            
        for i in searchIndices:
            COVERAGE_RES = 60
            thisImage       = ee.Image(imageList.get(i)).resample('bicubic')
            percentCoverage = thisImage.mask().reduceRegion(ee.Reducer.mean(), bounds, COVERAGE_RES).getInfo().values()[0]
            if percentCoverage < minCoverage:
                continue
            cloudPercentage = cmt.util.landsat_functions.getCloudPercentage(thisImage, bounds)
            print('Detected Landsat cloud percentage: ' + str(cloudPercentage))
            if cloudPercentage < maxCloudPercentage:
                return thisImage
        # If we got here this satellite did not produce a good image, try the next satellite.

    raise Exception('Could not find a nearby cloud-free Landsat image for date ' + str(targetDate.getInfo()))


def getNearestSentinel1(bounds, targetDate, maxRangeDays=10, minCoverage=0.8, searchMethod='nearest'):
    '''Search for the closest Sentinel1 image near the target date.
       Sentinel1 images are radar and see through clouds!
       The result preference is determined by searchMethod and can be  set to:
       nearest, increasing, or decreasing'''

    # TODO: Some options to balance nearness and resolution preferences?

    # Get the date range to search
    if searchMethod == 'nearest':
        dateStart = targetDate.advance(-1*maxRangeDays, 'day')
        dateEnd   = targetDate.advance(   maxRangeDays, 'day')
    else:
        dateStart = targetDate
        dateEnd   = targetDate.advance(   maxRangeDays, 'day')
    
    # Get a list of candidate images
    imageCollection = get_image_collection_sentinel1(bounds, dateStart, dateEnd)
    imageList       = imageCollection.toList(100)
    imageInfo       = imageList.getInfo()
    
    if len(imageInfo) == 0:
        raise Exception('Could not find a nearby Sentinel1 image for date ' + str(targetDate.getInfo()))
  
    # Find the first image that meets the requirements
    numFound = len(imageInfo)
    if searchMethod == 'nearest':
        # The images start sorted by time, change to sort by distance from target.
        searchIndices = getIndicesSortedByNearestDate(imageList, targetDate)
    elif searchMethod == 'increasing':
        searchIndices = range(0,numFound)
    else:
        searchIndices = range(numFound-1, -1, -1)

    for i in searchIndices:
        COVERAGE_RES = 30
        thisImage       = ee.Image(imageList.get(i)).resample('bicubic')
        percentCoverage = thisImage.mask().reduceRegion(ee.Reducer.mean(), bounds, COVERAGE_RES).getInfo().values()[0]
        print('S1 Coverage = ' + str(percentCoverage))
        if percentCoverage >= minCoverage: # MODIS has high coverage, but there are gaps.
            return thisImage

    raise Exception('Could not find a nearby Sentinel image with coverage for date ' + str(targetDate.getInfo()))



