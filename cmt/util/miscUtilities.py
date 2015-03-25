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
import threading
import functools
import time
#import cmt.mapclient_qt

def safe_get_info(ee_object, max_num_attempts=None):
    '''Keep trying to call getInfo() on an Earth Engine object until it succeeds.'''
    num_attempts = 0
    while True:
        try:
            return ee_object.getInfo()
        except Exception as e:
            print 'Earth Engine Error: %s. Waiting 10s and then retrying.' % (e)
            time.sleep(10)
            num_attempts += 1
        if max_num_attempts and (num_attempts >= max_num_attempts):
            raise Exception('safe_get_info failed to succeed after ' +str(num_attempts)+ ' attempts!')

def get_permanent_water_mask():
    '''Returns the global permanent water mask'''
    return ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'], ['b1'])
