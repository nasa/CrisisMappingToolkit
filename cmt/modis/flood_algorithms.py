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

from adaboost import *
from dnns import *
from ee_classifiers import *
from misc_algorithms import *
from modis_utilities import *
from simple_modis_algorithms import *
import cmt.radar.active_contour


'''
Contains implementations of multiple MODIS-based flood detection algorithms.
'''

# Each algorithm name has an integer assigned to it.
EVI                = 1
XIAO               = 2
DIFFERENCE         = 3
CART               = 4
SVM                = 5
RANDOM_FORESTS     = 6
DNNS               = 7
DNNS_DEM           = 8
DIFFERENCE_HISTORY = 9
DARTMOUTH          = 10
DNNS_REVISED       = 11
DEM_THRESHOLD      = 12
MARTINIS_TREE      = 13
DNNS_DIFF          = 14
DNNS_DIFF_DEM      = 15
DIFF_LEARNED       = 16
DART_LEARNED       = 17
FAI                = 18
FAI_LEARNED        = 19
MODNDWI            = 20
MODNDWI_LEARNED    = 21
ADABOOST           = 22
ADABOOST_LEARNED   = 23
ADABOOST_DEM       = 24
ACTIVE_CONTOUR     = 25



# Set up some information for each algorithm, used by the functions below.
_ALGORITHMS = {
        # Algorithm,    Display name,   Function name,    Fractional result?,    Display color
        EVI                : ('EVI',                     evi,            False, 'FF00FF'),
        XIAO               : ('XIAO',                    xiao,           False, 'FFFF00'),
        DIFFERENCE         : ('Difference',              modis_diff,     False, '00FFFF'),
        DIFF_LEARNED       : ('Diff. Learned',           diff_learned,   False, '00FFFF'),
        DARTMOUTH          : ('Dartmouth',               dartmouth,      False, '33CCFF'),
        DART_LEARNED       : ('Dartmouth Learned',       dart_learned,   False, '33CCFF'),
        FAI                : ('Floating Algae',          fai,            False, '3399FF'),
        FAI_LEARNED        : ('Floating Algae Learned',  fai_learned,    False, '3399FF'),
        MODNDWI            : ('Mod. NDWI',               mod_ndwi,       False, '00FFFF'),
        MODNDWI_LEARNED    : ('Mod. NDWI Learned',      mod_ndwi_learned,False, '00FFFF'),
        CART               : ('CART',                    cart,           False, 'CC6600'),
        SVM                : ('SVM',                     svm,            False, 'FFAA33'),
        RANDOM_FORESTS     : ('Random Forests',          random_forests, False, 'CC33FF'),
        DNNS               : ('DNNS',                    dnns,           True,  '0000FF'),
        DNNS_DIFF          : ('DNNS Diff.',              dnns_diff,      True,  '0000FF'),
        DNNS_REVISED       : ('DNNS Revised',            dnns_revised,   False, '00FF00'),
        DNNS_DEM           : ('DNNS with DEM',           dnns_dem,       False, '9900FF'),
        DNNS_DIFF_DEM      : ('DNNS Diff with DEM',      dnns_diff_dem,  False, '9900FF'),
        DIFFERENCE_HISTORY : ('Difference with History', history_diff,   False, '0099FF'),
        DEM_THRESHOLD      : ('DEM Threshold',           dem_threshold,  False, 'FFCC33'),
        MARTINIS_TREE      : ('Martinis Tree',           martinis_tree,  False, 'CC0066'),
        ADABOOST           : ('Adaboost',                adaboost,       False, '9933FF'),
        ADABOOST_LEARNED   : ('Adaboost Learned',        adaboost_learn, False, 'FF3399'),
        ADABOOST_DEM       : ('Adaboost DEM',            adaboost_dem,   False, '6600CC'),
        ACTIVE_CONTOUR     : ('Active Countour',
                              cmt.radar.active_contour.active_countour_skybox,  False, '0066CC'),
}


def detect_flood(domain, algorithm):
    '''Run flood detection with a named algorithm in a given domain.'''
    try:
        approach = _ALGORITHMS[algorithm]
    except:
        return None
    return (approach[0], approach[1](domain, compute_modis_indices(domain)))

def get_algorithm_name(algorithm):
    '''Return the text name of an algorithm.'''
    try:
        return _ALGORITHMS[algorithm][0]
    except:
        return None

def get_algorithm_color(algorithm):
    '''Return the color assigned to an algorithm.'''
    try:
        return _ALGORITHMS[algorithm][3]
    except:
        return None

def is_algorithm_fractional(algorithm):
    '''Return True if the algorithm has a fractional output.'''
    try:
        return _ALGORITHMS[algorithm][2]
    except:
        return None

