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
import signal

from os.path import expanduser


__MY_ACCOUNT_FILE = expanduser('~/.local/google_service_account.txt')
#__MY_PRIVATE_KEY_FILE = expanduser('~/.local/google_service_api_private_key.p12')
__MY_PRIVATE_KEY_FILE = expanduser('~/.local/google_service_api_private_key.pem')

def initialize(account=None, key_file=None):
    '''Initialize the Earth Engine object, using your authentication credentials.'''
    try:
        ee.Initialize()
    except:
        # in the past, EE keys had to be installed manually. We keep this old method for
        # backwards compatibility
        if account == None:
            f = open(__MY_ACCOUNT_FILE, 'r')
            account = f.readline().strip()
        if key_file == None:
            key_file = __MY_PRIVATE_KEY_FILE
        ee.Initialize(ee.ServiceAccountCredentials(account, key_file))

