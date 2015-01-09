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

import logging
logging.basicConfig(level=logging.ERROR)
try:
    import cmt.ee_authenticate
except:
    import sys
    import os.path
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    import cmt.ee_authenticate
import matplotlib
#matplotlib.use('tkagg')

import sys
import os
import ee
import functools

#import modis.domains
import cmt.domain
from cmt.modis.flood_algorithms import *

from cmt.mapclient_qt    import centerMap, addToMap
from cmt.util.evaluation import evaluate_approach

'''
Tool for testing MODIS based flood detection algorithms using a simple GUI.
'''


# --------------------------------------------------------------
# Configuration

# Specify the data set to use - see /modis/domains.py
#DOMAIN = modis.domains.BORDER

# Specify each algorithm to be concurrently run on the data set - see /modis/flood_algorithms.py
ALGORITHMS = [DARTMOUTH, DIFFERENCE, DNNS, DNNS_DEM]


# --------------------------------------------------------------
# Functions

def evaluation_function(pair, alg):
    '''Pretty print an algorithm and its statistics'''
    precision, recall = pair
    print '%s: (%4g, %4g)' % (get_algorithm_name(alg), precision, recall)


# TODO: This could live elsewhere
def visualizeDomain(domain, show=True):
    '''Draw all the sensors and ground truth from a domain'''
    centerMap(domain.center[0], domain.center[1], 11)
    for s in domain.sensor_list:
        apply(addToMap, s.visualize(show=show))
    if domain.ground_truth != None:
        addToMap(domain.ground_truth, {}, 'Ground Truth', False)

# --------------------------------------------------------------
# main()

# TODO: Unify these files based on the selected domain!

# Get the domain XML file from the command line arguments
if len(sys.argv) < 2:
    print 'Usage: detect_flood_modis.py domain.xml'
    sys.exit(0)

cmt.ee_authenticate.initialize()

# Fetch data set information
domain = cmt.domain.Domain(sys.argv[1])

# Display the Landsat and MODIS data for the data set
visualizeDomain(domain)
#centerMap(d.center[0], d.center[1], 11)
#addToMap(d.landsat,       {'bands': ['B3', 'B2', 'B1'], 'gain': d.landsat_gain}, 'Landsat RGB')
#addToMap(d.low_res_modis, {'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06'], 'min' : 0, 'max': 3000, 'opacity' : 1.0}, 'MODIS', False)
#apply(addToMap, d.visualize())
#addToMap(d.ground_truth.mask(d.ground_truth), {'min': 0, 'max': 1, 'opacity': 0.5}, 'Ground Truth', False)


# For each of the algorithms
for a in range(len(ALGORITHMS)):
    # Run the algorithm on the data and get the results
    (alg, result) = detect_flood(domain, ALGORITHMS[a])

    # Get a color pre-associated with the algorithm, then draw it on the map
    color = get_algorithm_color(ALGORITHMS[a])
    addToMap(result.mask(result), {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': '000000, ' + color}, alg, False)

    # Compare the algorithm output to the ground truth and print the results
    if domain.ground_truth:
        evaluate_approach(functools.partial(evaluation_function, alg=ALGORITHMS[a]), result, domain.ground_truth, domain.bounds, is_algorithm_fractional(ALGORITHMS[a]))

#addToMap(domain.dem, {min:25, max:50}, 'DEM', false);










