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
#matplotlib.use('tkagg') # Needed to display a histogram

import os
import sys
import ee
import functools

import cmt.domain
from cmt.visible.flood_algorithms import *

from cmt.mapclient_qt    import centerMap, addToMap
import cmt.util.evaluation

'''
Tool for testing RGB based flood detection algorithms using a simple GUI.
'''


# --------------------------------------------------------------
# Configuration

ALGORITHMS = [SKYBOX]


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

# Get the domain XML file from the command line arguments
if len(sys.argv) < 2:
    print 'Usage: detect_flood_radar.py domain.xml'
    sys.exit(0)

cmt.ee_authenticate.initialize()

# Fetch data set information
domain = cmt.domain.Domain(sys.argv[1])

# Display radar and ground truth 
visualizeDomain(domain)

#print im.image.getDownloadUrl({'name' : 'sar', 'region':ee.Geometry.Rectangle(-91.23, 32.88, -91.02, 33.166).toGeoJSONString(), 'scale': 6.174})

# For each of the algorithms
for a in range(len(ALGORITHMS)):
    # Run the algorithm on the data and get the results
    alg    = ALGORITHMS[a]
    result = detect_flood(domain, alg)
    
    # Get a color pre-associated with the algorithm, then draw it on the map
    color  = get_algorithm_color(alg)
    addToMap(result.mask(result), {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': '000000, ' + color}, get_algorithm_name(alg), False)
    
    # Compare the algorithm output to the ground truth and print the results
    if domain.ground_truth != None:
        cmt.util.evaluation.evaluate_approach_thread(functools.partial(evaluation_function, alg=alg), result, domain.ground_truth, domain.bounds)

#addToMap(domain.groundTruth.mask(domain.groundTruth), {'min': 0, 'max' : 1, 'opacity' : 0.2}, 'Ground Truth', false);
#addToMap(domain.dem, {min:25, max:50}, 'DEM', false);

