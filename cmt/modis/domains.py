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
import ee.geometry

from pprint import pprint

BORDER           = 1
BORDER_JUNE      = 2
ARKANSAS_CITY    = 3
KASHMORE         = 4
KASHMORE_NORTH   = 5
NEW_ORLEANS      = 6
SLIDELL          = 7
BAY_AREA         = 8
BERKELEY         = 9
NIGER            = 10

# (name, bounding box, landsat date range, modis date range,
# ground truth, dem (default SRTM), landsat type (default L5_L1T), landsat gain (default [1.9, 1.8, 1.0])
__ALL_DOMAINS = [None, \
    ( 'Mississippi Border', (-91.23, 32.88, -91.02, 33.166), ('2011-05-08', '2011-05-11'), ('2011-05-08', '2011-05-11'), \
            '18108519531116889794-00161330875310406093', 'ned_13', None, None),
    ( 'Mississippi June',   (-91.23, 32.88, -91.02, 33.166), ('2011-06-10', '2011-06-12'), ('2011-06-12', '2011-06-13'), \
            '18108519531116889794-12921502713420913455', 'ned_13', None, None),
    ( 'Arkansas City',      (-91.32, 33.56, -91.03, 33.7),   ('2011-05-08', '2011-05-11'), ('2011-05-08', '2011-05-11'), \
            '18108519531116889794-09052745394509652502', 'ned_13', None, None),
    ( 'Kashmore',           (69.5, 28.25, 70.1, 28.65),      ('2010-08-12', '2010-08-13'), ('2010-08-13', '2010-08-14'), \
            '18108519531116889794-14495543923251622067', None, None, [1.5, 1.6, 1.0]),
    ( 'Kashmore North',  (70.105, 28.67, 70.185, 28.97),     ('2010-08-12', '2010-08-13'), ('2010-08-13', '2010-08-14'), \
            '18108519531116889794-12421761926155747447', None, [1.5, 1.6, 1.0]),
    ( 'New Orleans',        (-90.3, 29.90,-89.76, 30.07),    ('2005-09-07', '2005-09-08'), ('2005-09-07', '2005-09-08'), \
            '18108519531116889794-17234028967417318364', 'ned_13', None, None),
    ( 'Slidell',          (-89.88, 30.18,-89.75, 30.32),     ('2005-09-07', '2005-09-08'), ('2005-09-07', '2005-09-08'), \
            '18108519531116889794-15363260755447510668', 'ned_13', [1.5, 1.5, 1.0]),
    ( 'SF Bay Area',        (-122.55, 37.40, -121.7, 37.67), ('2011-04-27', '2011-04-28'), ('2011-04-27', '2011-04-28'), \
            '18108519531116889794-12847023993615557481', 'ned_13', None, [1.8, 1.5, 1.0]),
    ( 'Berkeley',        (-122.54, 37.85, -122.126, 37.926), ('2011-04-27', '2011-04-28'), ('2011-04-27', '2011-04-28'), \
            '18108519531116889794-13496919088645259843', 'ned_13', None, [1.8, 1.5, 1.0]),
    ( 'Niger',        (6.6, 7.75, 7.1, 8.15),                  ('2012-09-15', '2012-11-08'), ('2012-10-20', '2012-10-27'), \
            '18108519531116889794-13496919088645259843', None, 'L7_L1T', None)
]

TRAINING_DOMAINS = {
        BORDER         : ARKANSAS_CITY,
        BORDER_JUNE    : ARKANSAS_CITY,
        ARKANSAS_CITY  : BORDER,
        KASHMORE       : KASHMORE_NORTH,
        KASHMORE_NORTH : KASHMORE,
        NEW_ORLEANS    : SLIDELL,
        SLIDELL        : NEW_ORLEANS,
        BAY_AREA       : BERKELEY,
        BERKELEY       : BAY_AREA}

class FloodDomain(object):
    def __init__(self, id, name, bounds, date, high_res_image, low_res_image, landsat, ground_truth, \
            dem, landsat_type = 'L5_L1T', landsat_gain = [1.9, 1.8, 1.0], water_mask = None, center=None):
        self.id             = id
        self.name           = name
        self.bounds         = bounds
        self.date           = date
        self.high_res_modis = high_res_image
        self.low_res_modis  = low_res_image
        self.landsat        = landsat
        self.ground_truth   = ground_truth
        self.dem            = dem
        self.water_mask     = water_mask
        self.landsat_type   = landsat_type
        self.landsat_gain   = landsat_gain
        self.center         = center

        if self.water_mask == None:
            self.water_mask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'])

def retrieve_domain(index):
    if index <= 0 or index >= len(__ALL_DOMAINS):
        return None
    tup           = __ALL_DOMAINS[index]
    name          = tup[0]
    bounds        = apply(ee.geometry.Geometry.Rectangle, tup[1])
    center        = ((tup[1][0] + tup[1][2]) / 2, (tup[1][1] + tup[1][3]) / 2)
    landsat_dates = map(ee.Date, tup[2])
    modis_dates   = map(ee.Date, tup[3])
    ground_truth  = ee.Image(tup[4]).clamp(0, 1)
    dem           = ee.Image('CGIAR/SRTM90_V4' if tup[5] == None else tup[5])
    landsat_type  = ('L5_L1T' if tup[6] == None else tup[6])
    landsat_gains = ([1.9, 1.8, 1.0] if tup[7] == None else tup[7])
  
    landsat = ee.ImageCollection(landsat_type).filterDate(landsat_dates[0], landsat_dates[1]).filterBounds(bounds).limit(1).mean();
    if landsat_type == 'L7_L1T':
        landsat = landsat.select(['10', '20', '30', '40', '50', '62', '70', '80'], ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'])
    else:
        landsat = landsat.select(['10', '20', '30', '40', '50', '60', '70'],       ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'])
    #pprint(ee.ImageCollection(landsat_type).filterDate(landsat_dates[0], landsat_dates[1]).filterBounds(bounds).getInfo())
    high_res_modis = ee.ImageCollection('MOD09GQ').filterBounds(bounds).filterDate(modis_dates[0], modis_dates[1]).limit(1).mean();
    low_res_modis  = ee.ImageCollection('MOD09GA').filterBounds(bounds).filterDate(modis_dates[0], modis_dates[1]).limit(1).mean();
    #print(ee.ImageCollection(landsat).filterDate(landsatFloodDateStart, landsatFloodDateEnd).filterBounds(polygon).getInfo());
    return FloodDomain(index, name, bounds, modis_dates[0], high_res_modis, low_res_modis, landsat, ground_truth, \
            dem, landsat_type, landsat_gains, None, center)

