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

import cmt.domain
import cmt.mapclient_qt
import cmt.util.gui_util

'''
GUI related utilities too small for their own file
'''



def visualizeDomain(domain, show=True):
    '''Draw all the sensors and ground truth from a domain'''
    cmt.mapclient_qt.centerMap(domain.center[0], domain.center[1], 11)
    for s in domain.sensor_list:
        apply(cmt.mapclient_qt.addToMap, s.visualize(show=show))
    if domain.ground_truth != None:
        cmt.mapclient_qt.addToMap(domain.ground_truth.mask(domain.ground_truth), {}, 'Ground Truth', False)