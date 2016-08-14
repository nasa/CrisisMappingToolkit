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

from cmt.util.landsat_functions import *

# Inappropriate bounding box
def containsLake(image, bounds):
    image = ee.Image(image) 
    coords = image.geometry()
    TF = coords.contains(bounds)
    TF = TF.getInfo()
    return TF

# Appropriate bounding box
def containsLakev2(image,bounds):
    imageinfo = image.getInfo()
    coords = imageinfo['properties']['system:footprint']['coordinates']
    polygon = ee.Geometry.Polygon(coords)
    TF = polygon.contains(bounds) 
    TF = TF.getInfo()
    return TF

# Using masking, works for image collections, not just a singular image
def containsLakev3(image,bounds):
    MIN_RESOLUTION = 60
    MAX_RESOLUTION = 1000
    resolution = MAX_RESOLUTION
    oneMask = ee.Image(1.0)
    reducedcollection = image.reduce(ee.Reducer.allNonZero())
    areaCount = reducedcollection.reduceRegion(ee.Reducer.sum(), bounds, resolution)
    areaCount2 = oneMask.reduceRegion(ee.Reducer.sum(), bounds, resolution)
    areaCount = areaCount.getInfo()
    areaCount2 = areaCount2.getInfo()
    print areaCount, areaCount2
    coveredpixels = float(areaCount['all'])
    totalpixels = float(areaCount2['constant'])
    cover = coveredpixels/totalpixels
    return cover


def getCloudPercentagev2(image, region):
    '''Estimates the cloud cover percentage in a Landsat image

    IMPROVED: First masks for shape of the landsat imagery, to base cloud cover percentages based on ONLY
    the area that the lake is actually in the image
    '''
    # The function will attempt the calculation in these ranges
    # - Native Landsat resolution is 30
    MIN_RESOLUTION = 60
    MAX_RESOLUTION = 1000
    resolution = MIN_RESOLUTION
    cloudScore = detect_clouds(image)
    reducedimage = image.reduce(ee.Reducer.allNonZero())
    areaCount = reducedimage.reduceRegion(ee.Reducer.sum(), region, resolution)
    cloudCount = cloudScore.reduceRegion(ee.Reducer.sum(), region, resolution)
    #mine = cloudCount.getInfo()['constant']
    #percentage = cloudCount.getInfo()['constant'] / areaCount.getInfo()['all']
    return cloudCount, areaCount
