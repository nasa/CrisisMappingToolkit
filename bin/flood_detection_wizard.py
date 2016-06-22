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

'''
    Setup script for the "production" flood detection tool.

    Instructions for using this tool:

    1 - Run this script.  The GUI should appear.
    2 - Select the date of the flood you are interested in using
        the button with the date in the top left corner.
    3 - Pan and zoom to the region of interest.
    4 - Press the 'Set Processing Region' button to set the
        area where image statistics are calculated.
    5 - Press the 'Load Images' button to search for Landsat,
        MODIS, DEM, and cloud mask images for that region/date.
    6 - Press either 'Detect Flood' button to run the flood detection
        algorithm on the MODIS image.
    7 - You can adjust the two slider bars at the bottom to control
        the sensitivity of the flood detection algorithm:
        - Change Detection Threshold = Decrease this number to find more flood pixels.
        - Water Mask Threshold = Increase this number to find more flood pixels.
    8 - If you want to look at a new area, press the 'Clear Map' button and go back to
        step 2.
'''

import logging
logging.basicConfig(level=logging.ERROR)

try:
    import cmt.ee_authenticate
except:
    import sys
    import os.path
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    import cmt.ee_authenticate
cmt.ee_authenticate.initialize()

# The GUI type for the project must be set like this!
import cmt.mapclient_qt
import cmt.util.production_gui
cmt.mapclient_qt.gui_type = cmt.util.production_gui.ProductionGui

# --------------------------------------------------------------
# Configuration

# --------------------------------------------------------------
# main()

# util/production_gui.py does all the work!
cmt.mapclient_qt.run()
