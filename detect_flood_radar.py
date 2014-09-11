import logging
logging.basicConfig(level=logging.ERROR)
import util.ee_authenticate
util.ee_authenticate.initialize()

from pprint import pprint
import os
import ee
from util.mapclient_qt import centerMap, addToMap

import radar.domains

r = radar.domains.get_radar_image(radar.domains.UAVSAR, radar.domains.UAVSAR_MISSISSIPPI_FLOODED)
center = r.bounds.centroid().getInfo()['coordinates']
centerMap(center[0], center[1], 11)
apply(addToMap, r.visualize())

