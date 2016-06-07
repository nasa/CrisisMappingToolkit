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

import matgen
import learning
import martinis
import active_contour
import cmt.modis.adaboost

'''
This file contains a summary and selectors for all the supported radar algorithms.
'''


# Assign each algorithm an index
MATGEN         = 1
RANDOM_FORESTS = 2
DECISION_TREE  = 3
SVM            = 4
MARTINIS_CV    = 5
MARTINIS_CR    = 6
ACTIVE_CONTOUR = 7
ADABOOST       = 8
MARTINIS_2     = 9

# For each algorithm specify the name, function, and color.
__ALGORITHMS = {
    MATGEN         : ('Matgen Threshold', matgen.threshold,              '00FFFF'),
    RANDOM_FORESTS : ('Random Forests',   learning.random_forests,       'FFFF00'),
    DECISION_TREE  : ('Decision Tree',    learning.decision_tree,        'FF00FF'),
    SVM            : ('SVM',              learning.svm,                  '00AAFF'),
    MARTINIS_CV    : ('Martinis CV',      martinis.sar_martinis,         'AAFF00'),
    MARTINIS_CR    : ('Martinis CR',      martinis.sar_martinis_cr,      'AA00FF'),
    MARTINIS_2     : ('Martinis 2',       martinis.sar_martinis2,        'AA0000'),
    ACTIVE_CONTOUR : ('Active Contour',   active_contour.active_contour, 'FF00AA'),
    ADABOOST       : ('Adaboost',         cmt.modis.adaboost.adaboost_radar, '00FFFF')
}

# These functions just redirect the call to the correct algorithm

def detect_flood(image, algorithm):
    '''Run the chosen algorithm on the given image.'''
    try:
        approach = __ALGORITHMS[algorithm]
    except:
        return None
    return (approach[0], approach[1](image))

def get_algorithm_name(algorithm):
    '''Return the text name of the algorithm.'''
    try:
        return __ALGORITHMS[algorithm][0]
    except:
        return None

def get_algorithm_color(algorithm):
    '''Return the color assigned to an algorithm.'''
    try:
        return __ALGORITHMS[algorithm][2]
    except:
        return None

