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
# matplotlib.use('tkagg') # Needed to display a histogram

import os
import sys
import ee
import functools
import threading

import cmt.domain
import cmt.util.evaluation
import cmt.util.gui_util

from cmt.radar.flood_algorithms import *
from cmt.mapclient_qt import centerMap, addToMap

'''
Tool for testing radar based flood detection algorithms using a simple GUI.
'''


# --------------------------------------------------------------
# Configuration

# ALGORITHMS = [DECISION_TREE, RANDOM_FORESTS, SVM, MATGEN, MARTINIS_CV, MARTINIS_CR, ACTIVE_CONTOUR]
#ALGORITHMS = [SVM, RANDOM_FORESTS, DECISION_TREE]
#ALGORITHMS = [ADABOOST, ADABOOST_DEM]
# ALGORITHMS = [MARTINIS_CV]#, MARTINIS_CR]
ALGORITHMS = [MARTINIS_2]

# --------------------------------------------------------------
# Functions


def evaluation_function(pair, alg):
    '''Pretty print an algorithm and its statistics'''
    (precision, recall, evalRes, noTruth) = pair
    print '%s: (%4g, %4g, %4g)' % (get_algorithm_name(alg), precision, recall, noTruth)

# --------------------------------------------------------------
def main():
    # Get the domain XML file from the command line arguments
    if len(sys.argv) < 2:
        print 'Usage: detect_flood_radar.py domain.xml'
        sys.exit(0)
    
    cmt.ee_authenticate.initialize()
    
    # Fetch data set information
    domain = cmt.domain.Domain()
    domain.load_xml(sys.argv[1])
    
    # Display radar and ground truth
    cmt.util.gui_util.visualizeDomain(domain)
    
    waterMask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'], ['b1'])
    addToMap(waterMask.mask(waterMask), {'min': 0, 'max': 1}, 'Permanent Water Mask', False)
    
    # print im.image.getDownloadUrl({'name' : 'sar', 'region':ee.Geometry.Rectangle(-91.23, 32.88, -91.02, 33.166).toGeoJSONString(), 'scale': 6.174})
    
    # For each of the algorithms
    for a in range(len(ALGORITHMS)):
        #try:
        # Run the algorithm on the data and get the results
        (alg, result) = detect_flood(domain, ALGORITHMS[a])

        # Needed for certain images which did not mask properly from maps engine
        # result = result.mask(domain.get_radar().image.reduce(ee.Reducer.allNonZero()))

        # Get a color pre-associated with the algorithm, then draw it on the map
        color = get_algorithm_color(ALGORITHMS[a])
        addToMap(result.mask(result), {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': '000000, ' + color},
                 alg, False)

        # Compare the algorithm output to the ground truth and print the results
        if domain.ground_truth is not None:
            cmt.util.evaluation.evaluate_approach_thread(functools.partial(
                evaluation_function, alg=ALGORITHMS[a]), result, domain.ground_truth, domain.bounds)
        #except Exception, e:
        #    print('Caught exception running algorithm: ' + get_algorithm_name(ALGORITHMS[a]) + '\n' +
        #          str(e) + '\n')

# This code needs to be outside the main function for OSX!
t = threading.Thread(target=main)
t.start()

cmt.mapclient_qt.run()

