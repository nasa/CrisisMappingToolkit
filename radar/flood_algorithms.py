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

# From Towards an automated SAR-based flood monitoring system:
# Lessons learned from two case studies by Matgen, Hostache et. al.
MATGEN         = 1
RANDOM_FORESTS = 2
DECISION_TREE  = 3
SVM            = 4
MARTINIS       = 5
ACTIVE_CONTOUR = 6

# For each algorithm specify the name, function, and color.
__ALGORITHMS = {
    MATGEN : ('Matgen Threshold', matgen.threshold, '00FFFF'),
    RANDOM_FORESTS : ('Random Forests', learning.random_forests, 'FFFF00'),
    DECISION_TREE  : ('Decision Tree', learning.decision_tree, 'FF00FF'),
    SVM : ('SVM', learning.svm, '00AAFF'),
    MARTINIS  : ('Martinis',  martinis.sar_martinis, 'FF00FF'),
    ACTIVE_CONTOUR : ('Active Contour',  active_contour.active_contour, 'FF00FF')
}

# These functions just redirect the call to the correct algorithm

def detect_flood(image, algorithm):
    try:
        approach = __ALGORITHMS[algorithm]
    except:
        return None
    return approach[1](image)

def get_algorithm_name(algorithm):
    try:
        return __ALGORITHMS[algorithm][0]
    except:
        return None

def get_algorithm_color(algorithm):
    try:
        return __ALGORITHMS[algorithm][2]
    except:
        return None

