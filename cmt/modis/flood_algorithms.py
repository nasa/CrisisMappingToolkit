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

import adaboost
import dnns
import ee_classifiers
import misc_algorithms
import modis_utilities
import simple_modis_algorithms


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
#SKYBOX_ASSIST      = 14 # Currently unused
DNNS_DIFF          = 15
DNNS_DIFF_DEM      = 16
DIFF_LEARNED       = 17
DART_LEARNED       = 18
FAI                = 19
FAI_LEARNED        = 20
EXPERIMENTAL       = 21
MODNDWI            = 22
MODNDWI_LEARNED    = 23



# Set up some information for each algorithm, used by the functions below.
_ALGORITHMS = {
        # Algorithm,    Display name,   Function name,    Fractional result?,    Display color
        EVI                : ('EVI',                     simple_modis_algorithms.evi,             False, 'FF00FF'),
        XIAO               : ('XIAO',                    simple_modis_algorithms.xiao,            False, 'FFFF00'),
        DIFFERENCE         : ('Difference',              simple_modis_algorithms.modis_diff,      False, '00FFFF'),
        DIFF_LEARNED       : ('Diff. Learned',           simple_modis_algorithms.diff_learned,    False, '00FFFF'),
        DARTMOUTH          : ('Dartmouth',               simple_modis_algorithms.dartmouth,       False, '33CCFF'),
        DART_LEARNED       : ('Dartmouth Learned',       simple_modis_algorithms.dart_learned,    False, '33CCFF'),
        FAI                : ('Floating Algae',          simple_modis_algorithms.fai,             False, '3399FF'),
        FAI_LEARNED        : ('Floating Algae Learned',  simple_modis_algorithms.fai_learned,     False, '3399FF'),
        MODNDWI            : ('Mod. NDWI',               simple_modis_algorithms.mod_ndwi,        False, '00FFFF'),
        MODNDWI_LEARNED    : ('Mod. NDWI Learned',       simple_modis_algorithms.mod_ndwi_learned,False, '00FFFF'),
        CART               : ('CART',                    ee_classifiers.cart,                     False, 'CC6600'),
        SVM                : ('SVM',                     ee_classifiers.svm,                      False, 'FFAA33'),
        RANDOM_FORESTS     : ('Random Forests',          ee_classifiers.random_forests,           False, 'CC33FF'),
        DNNS               : ('DNNS',                    dnns.dnns,                               True,  '0000FF'),
        DNNS_DIFF          : ('DNNS Diff.',              dnns.dnns_diff,                          True,  '0000FF'),
        DNNS_REVISED       : ('DNNS Revised',            dnns.dnns_revised,                       False, '00FF00'),
        DNNS_DEM           : ('DNNS with DEM',           dnns.dnns_dem,                           False, '9900FF'),
        DNNS_DIFF_DEM      : ('DNNS Diff with DEM',      dnns.dnns_diff_dem,                      False, '9900FF'),
        DIFFERENCE_HISTORY : ('Difference with History', misc_algorithms.history_diff,            False, '0099FF'),
        DEM_THRESHOLD      : ('DEM Threshold',           simple_modis_algorithms.dem_threshold,   False, 'FFCC33'),
        MARTINIS_TREE      : ('Martinis Tree',           misc_algorithms.martinis_tree,           False, 'CC0066'),
#        SKYBOX_ASSIST      : ('Skybox Assist',           skyboxAssist,    False, '00CC66'),
        EXPERIMENTAL       : ('Experimental',            adaboost.experimental,                   False, '00FFFF')
}


def detect_flood(domain, algorithm):
    '''Run flood detection with a named algorithm in a given domain.'''
    try:
        approach = _ALGORITHMS[algorithm]
    except:
        return None
    return (approach[0], approach[1](domain, modis_utilities.compute_modis_indices(domain)))

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

