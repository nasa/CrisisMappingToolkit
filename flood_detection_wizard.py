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


import util.ee_authenticate
util.ee_authenticate.initialize()



# The GUI type for the project must be set like this!
import util.mapclient_qt
import util.production_gui
util.mapclient_qt.gui_type = util.production_gui.ProductionGui

# --------------------------------------------------------------
# Configuration



# --------------------------------------------------------------
# main()

# util/production_gui.py does all the work!
util.mapclient_qt.addEmptyGui()











