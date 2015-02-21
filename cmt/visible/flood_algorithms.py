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

from cmt.mapclient_qt import addToMap

'''
Contains implementations of primarily RGB based detection algorithms
'''

# Each algorithm name has an integer assigned to it.
SKYBOX = 1


def skybox(domain):
    '''TODO: Detect floods in this image!'''
    
    outputFlood = ee.Image(1)
    
    return outputFlood#.select(['sur_refl_b02'], ['b1']) # Rename sur_refl_b02 to b1


# End of algorithm definitions
#=======================================================================================================
#=======================================================================================================





# Set up some information for each algorithm, used by the functions below.
__ALGORITHMS = {
        # Algorithm,    Display name,   Function name,    Fractional result?,    Display color
        SKYBOX                : ('Skybox',                     skybox,            False, 'FF00FF')
}


def detect_flood(domain, algorithm):
    '''Run flood detection with a named algorithm in a given domain.'''
    try:
        approach = __ALGORITHMS[algorithm]
    except:
        return None
    return approach[1](domain)

def get_algorithm_name(algorithm):
    '''Return the text name of an algorithm.'''
    try:
        return __ALGORITHMS[algorithm][0]
    except:
        return None

def get_algorithm_color(algorithm):
    '''Return the color assigned to an algorithm.'''
    try:
        return __ALGORITHMS[algorithm][3]
    except:
        return None

def is_algorithm_fractional(algorithm):
    '''Return True if the algorithm has a fractional output.'''
    try:
        return __ALGORITHMS[algorithm][2]
    except:
        return None

