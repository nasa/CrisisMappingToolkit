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
# matplotlib.use('tkagg')

import sys
import os
import ee
import functools
import threading

import cmt.domain
from cmt.modis.flood_algorithms import *

from cmt.mapclient_qt import centerMap, addToMap
import cmt.util.evaluation
import cmt.util.gui_util

'''
Tool for testing MODIS based flood detection algorithms using a simple GUI.
'''

#  --------------------------------------------------------------
# Configuration

# Specify each algorithm to be concurrently run on the data set - see /modis/flood_algorithms.py
# ALGORITHMS = [DARTMOUTH, DART_LEARNED, DIFFERENCE, DIFF_LEARNED, FAI, FAI_LEARNED, EVI, XIAO, SVM, RANDOM_FORESTS, CART, DNNS, DNNS_DEM]
# ALGORITHMS = [DART_LEARNED, DIFF_LEARNED, FAI_LEARNED, MODNDWI_LEARNED, EVI, XIAO, MARTINIS_TREE, SVM, RANDOM_FORESTS, CART, DNNS, DNNS_DEM, ADABOOST, ADABOOST_DEM]
# ALGORITHMS = [DNNS, DNNS_DEM]
# ALGORITHMS = [SVM, RANDOM_FORESTS, CART]
# ALGORITHMS = [ADABOOST, ADABOOST_DEM]
# ALGORITHMS = [ACTIVE_CONTOUR]

#ALGORITHMS = [DART_LEARNED, EVI, XIAO, MARTINIS_TREE, CART, ADABOOST, ADABOOST_DEM]
ALGORITHMS = [DIFFERENCE, EVI, XIAO, ADABOOST]
#ALGORITHMS = []

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
      print 'Usage: detect_flood_modis.py domain.xml'
      sys.exit(0)
  
  cmt.ee_authenticate.initialize()
  
  # Fetch data set information
  domain = cmt.domain.Domain()
  domain.load_xml(sys.argv[1])
  
  # Display the Landsat and MODIS data for the data set
  cmt.util.gui_util.visualizeDomain(domain)
  
  #
  # import cmt.modis.adaboost
  # cmt.modis.adaboost.adaboost_learn()         # Adaboost training
  # #cmt.modis.adaboost.adaboost_dem_learn(None) # Adaboost DEM stats collection
  # raise Exception('DEBUG')
  
  waterMask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'], ['b1'])
  addToMap(waterMask.mask(waterMask), {'min': 0, 'max': 1}, 'Permanent Water Mask', False)
  
  # For each of the algorithms
  for a in range(len(ALGORITHMS)):
      # Run the algorithm on the data and get the results
      try:
          (alg, result) = detect_flood(domain, ALGORITHMS[a])
          if result is None:
              continue
  
          # These lines are needed for certain data sets which EE is not properly masking!!!
          # result = result.mask(domain.skybox_nir.image.reduce(ee.Reducer.allNonZero()))
          # result = result.mask(domain.skybox.image.reduce(ee.Reducer.allNonZero()))
  
          # Get a color pre-associated with the algorithm, then draw it on the map
          color = get_algorithm_color(ALGORITHMS[a])
          addToMap(result.mask(result), {'min': 0, 'max': 1, 'opacity': 0.5, 'palette': '000000, ' + color},
                   alg, False)
  
          # Compare the algorithm output to the ground truth and print the results
          if domain.ground_truth:
              cmt.util.evaluation.evaluate_approach_thread(functools.partial(
                  evaluation_function, alg=ALGORITHMS[a]), result, domain.ground_truth, domain.bounds,
                  is_algorithm_fractional(ALGORITHMS[a]))
      except Exception, e:
          print('Caught exception running algorithm: ' + get_algorithm_name(ALGORITHMS[a]) + '\n' +
                str(e) + '\n')

# This code needs to be outside the main function for OSX!
t = threading.Thread(target=main)
t.start()

cmt.mapclient_qt.run()
