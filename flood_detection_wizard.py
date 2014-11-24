'''
    Setup script for the "production" flood detection tool.
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











