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

"""A simple map GUI.

Implements a tiled map using QT. Displays map tiles using
whatever projection the tiles are in and only knows about tile coordinates,
(as opposed to geospatial coordinates.) This assumes that the tile-space is
organized as a power-of-two pyramid, with the origin in the upper left corner.
This currently has several spots that are hard-coded for 256x256 tiles, even
though TileManager tries to track this.

Supports mouse-based pan and zoom as well as tile upsampling while waiting
for new tiles to load.  The map to display is specified by a TileManager, and
added to the GUI on creation or manually using addOverlay()
    gui = GuiWrapper(MakeTileManager(mapid))

Tiles are referenced using a key of (level, x, y) throughout.

Several of the functions are named to match the Google Maps Javascript API,
and therefore violate style guidelines.

Based on the TK map interface from Google Earth Engine.

Terminology guide:
 - overlay = One of the things that can be displayed on the map.
             There is one of these for each "addToMap()" call.
 - layer   = Short for Layer Number, used for indexing a list of overlays.

This file contains the core GUI implementation.  Customized GUI instances are
located in seperate files.
"""


import collections
import cStringIO
import functools
import math
import random
import Queue
import sys
import time
import threading
import urllib2
import json
import ee
import os
import zipfile
import cPickle as pickle

# check if the Python imaging libraries used by the mapclient module are installed
try:
    from PIL import ImageQt                      # pylint: disable=g-import-not-at-top
    from PIL import Image, ImageChops            # pylint: disable=g-import-not-at-top
except ImportError:
    print """
        ERROR: A Python library (PILLOW) used by the CMT mapclient_qt module
        was not found. Information on PILLOW can be found at:
        https://pillow.readthedocs.org/
        """
    raise

try:
    import PyQt4                         # pylint: disable=g-import-not-at-top
    from PyQt4 import QtCore, QtGui
except ImportError:
    print """
        ERROR: A Python library (PyQt4) used by the CMT mapclient_qt
        module was not found.
        """
    raise

import cmt.util.miscUtilities

# The default URL to fetch tiles from.  We could pull this from the EE library,
# however this doesn't have any other dependencies on that yet, so let's not.
BASE_URL = 'https://earthengine.googleapis.com'

# Default directory to save images to
DEFAULT_SAVE_DIR = os.path.abspath(__file__)

# This is a URL pattern for creating an overlay from the google maps base map.
# The z, x and y arguments at the end correspond to level, x, y here.
DEFAULT_MAP_URL_PATTERN = ('http://mt1.google.com/vt/lyrs=m@176000000&hl=en&'
                                                     'src=app&z=%d&x=%d&y=%d')


# Tiles downloaded from Google Maps are cached here between 
LOCAL_MAP_CACHE_PATH = '/home/smcmich1/repo/earthEngine/gm_tile_cache.dat'


# Text to display in "About" buttons for legal purposes
ABOUT_TEXT = '''Crisis Mapping Toolkit (CMT) v1

A tool for assisting in crisis measurement and detection using Google's Earth Engine.


Copyright * 2014, United States Government as represented by the Administrator of the National Aeronautics and Space Administration. All rights reserved.

The Crisis Mapping Toolkit (CMT) v1 framework is licensed under the Apache License, Version 2.0 (the "License"); you may not use this application except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.'''



#================================================================================
# Classes that implement the GUI




class MapViewOverlay(object):
    '''Structure that stores all information about a single overlay in a MapViewWidget'''
    def __init__(self, tileManager, eeobject, name, show=True, vis_params=dict()):#, opacity=1.0):
        self.tileManager = tileManager # A TileManager instance for this overlay
        self.eeobject    = eeobject    # Earth Engine function object which computes the overlay.
        self.name        = name        # Name of the overlay.
        self.show        = show        # True/False if the overlay is currently being displayed.
        self.vis_params  = vis_params  # EE-style visualization parameters string.
        self.opacity     = 1.0         # Current opacity level for display - starts at 1.0

    def __str__(self):
        s = 'MapViewOverlay object: ' + self.name

# The map will display a stack of these when you right click on it.
class MapViewOverlayInfoWidget(QtGui.QWidget):
    '''Displays information for one layer at one location in a small horizontal bar.  Easy to stack vertically.
       Includes an opacity control and an on/off toggle checkbox.'''
    def __init__(self, parent, layer, x, y):
        super(MapViewOverlayInfoWidget, self).__init__()
        self.parent = parent # The parent is a MapViewWidget object
        self.layer  = layer  # The index of the layer in question
        self.x      = x      # Click location
        self.y      = y
        overlay = self.parent.overlays[self.layer] # This is a MapViewOverlay object
        
        # Constants that define the field size
        NAME_WIDTH   = 130
        ITEM_HEIGHT  = 10
        INFO_WIDTH   = 450
        SLIDER_WIDTH = 100
        OPACITY_MAX  = 100
        
        # Set up the visibility checkbox
        self.check_box = QtGui.QCheckBox(self) 
        self.check_box.setChecked(overlay.show)
        self.check_box.stateChanged.connect(self.toggle_visible)

        # Set up the opacity slider
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setRange(0, OPACITY_MAX) # 0 to 100 percent
        self.slider.setValue(int(overlay.opacity * OPACITY_MAX))
        self.slider.setTickInterval(25) # Add five tick marks
        self.slider.setMinimumSize(SLIDER_WIDTH, ITEM_HEIGHT)
        self.slider.valueChanged.connect(self.set_transparency) # Whenever the slider is moved, call set_transparency

        # Add the overlay name
        self.name = QtGui.QLabel(overlay.name, self)
        self.name.setMinimumSize(NAME_WIDTH, ITEM_HEIGHT)
        
        # Add the pixel value
        self.value = QtGui.QLabel('...', self) # Display this until the real value is ready
        self.value.setMinimumSize(INFO_WIDTH, ITEM_HEIGHT)

        def get_pixel():
            '''Helper function to retrieve the value of a single pixel in a single layer.'''
            try:
                return self.parent.getPixel(layer, x, y).getInfo()
            except: # features throw ee exception, ignore
                return None

        self.pixel_loader = cmt.util.miscUtilities.waitForEeResult(get_pixel, self.set_pixel_value)

        # Set up all the components in a horizontal box layout
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.check_box)
        hbox.addWidget(self.name)
        hbox.addWidget(self.slider)
        hbox.addWidget(self.value)

        self.setLayout(hbox) # Call QT function derived from parent QWidget class
    
    def set_pixel_value(self, value):
        '''Generate the text description for the pixel we clicked on'''
        # Handle values with not enough data
        if value == None: 
            self.value.setText('')
            return
        if len(value) <= 1: 
            self.value.setText('')
            return

        headers = value[0] # Extract the two parts of 'value'
        data    = value[1]       
        names   = headers[4:] # Skip id, lon, lat, time
        values  = data[4:] # Skip id, lon, lat, time
        
        # Get the object which contains information about the bands to display
        vis_params = self.parent.overlays[self.layer].vis_params
        
        text = ''
        for i in range(len(names)):
            # If bands were defined for this layer, only display the names of the selected bands.
            if vis_params and ('bands' in vis_params):
                if not (names[i] in vis_params['bands']): # WARNING: This parsing could be more robust!
                    continue
            
            if len(text) > 0: # Add comma after first entry
                text += ', '
            text += str(names[i]) + ': ' + str(values[i]) # Just keep appending strings
        self.value.setText(text)
    
    
    def toggle_visible(self):
        self.parent.overlays[self.layer].show = not self.parent.overlays[self.layer].show
        self.parent.reload()
    
    def set_transparency(self, value): # This is called whenever the slider bar is changed
        '''Set the layer transparency with the input value''' 
        self.parent.overlays[self.layer].opacity = value / 100.0
        self.parent.reload()

    def hideEvent(self, event):
        self.parent.setFocus()






class MapViewWidget(QtGui.QWidget):
    """A simple discrete zoom level map viewer.
        This class handles user input, coordinate conversion, and image painting.
        It requests tiles from the TileManager class when it needs them."""

    # Signals are defined here which other widgets can listen in on
    mapClickedSignal = QtCore.pyqtSignal(int, int) # x and y click coordinates.
    
    def __init__(self, inputTileManager=None):
        super(MapViewWidget, self).__init__()
        
        # for adding new layers to map
        self.executing_threads = []
        self.thread_lock = threading.Lock()

        self.tiles    = {}    # The cached stack of images at each grid cell.
        self.qttiles  = {}    # The cached PhotoImage at each grid cell.
        self.qttiles_lock = threading.RLock()
        self.level    = 2        # Starting zoom level
        self.origin_x = None     # The map origin x offset at the current level.
        self.origin_y = None     # The map origin y offset at the current level.
        self.anchor_x = None     # Drag anchor.
        self.anchor_y = None     # Drag anchor.

        # Map origin offsets; start at the center of the map.
        self.origin_x = (-(2 ** self.level) * 128) + self.width() / 2
        self.origin_y = (-(2 ** self.level) * 128) + self.height() / 2

        if not inputTileManager:
            # Default to a google maps basemap
            self.inputTileManager = TileManager(DEFAULT_MAP_URL_PATTERN)
        else:
            self.inputTileManager = inputTileManager

        # The array of overlays are displayed as last on top.
        self.overlays = [MapViewOverlay(self.inputTileManager, None, 'Google Maps')]
        #print 'Added base overlay!'

    
    def paintEvent(self, event):
        '''Rasterize each of the tiles on to the output image display'''
        painter = QtGui.QPainter()
        with self.qttiles_lock:
            painter.begin(self)
            for key in self.qttiles.keys():
                if key[0] != self.level:
                    continue
                image = self.qttiles[key]
                xpos  = key[1] * image.width()  + self.origin_x
                ypos  = key[2] * image.height() + self.origin_y
                painter.drawImage(QtCore.QPoint(xpos, ypos), image)
            painter.end()

    def addOverlay(self, inputTileManager, eeobject, name, show, vis_params):   # pylint: disable=g-bad-name
        """Add an overlay to the map."""
        self.overlays.append(MapViewOverlay(inputTileManager, eeobject, name, show, vis_params))
        #print 'Added overlay: ' + name
        self.LoadTiles()

    def GetViewport(self):
        """Return the visible portion of the map as [xlo, ylo, xhi, yhi] in weird Google coordinates."""
        width, height = self.width(), self.height()
        return [-self.origin_x,         -self.origin_y,
                -self.origin_x + width, -self.origin_y + height]

    def GetMapBoundingBox(self):
        """Return the bounding box of the current view as [minLon, minLat, maxLon, maxLat]"""
        # Just get the coordinates of the pixel corners of the map image
        topLeftLonLat  = self.pixelCoordToLonLat(0, 0)
        botRightLonLat = self.pixelCoordToLonLat(self.width(), self.height())
        return [topLeftLonLat[0], botRightLonLat[1], botRightLonLat[0], topLeftLonLat[1]]

    def LoadTiles(self):
        """Refresh the entire map."""
        #print 'Refreshing the map...'
        
        # Start with the overlay on top.
        for i, overlay in reversed(list(enumerate(self.overlays))):
            if not overlay.show:
                continue
            
            #print 'Refreshing layer = ' + str(i)
            tile_list = overlay.tileManager.CalcTiles(self.level, self.GetViewport())
            for key in tile_list:
                callback = functools.partial(self.AddTile, key=key, overlay=self.overlays[i], layer=i)
                overlay.tileManager.getTile(key, callback)

    def Flush(self):
        """Empty out all the image fetching queues."""
        for overlay in self.overlays:
            overlay.tileManager.Flush()

    def CompositeTiles(self, key):
        """Composite together all the tiles in this cell into a single image."""
        composite = None
        
        numLayers   = len(self.tiles[key])
        numOverlays = len(self.overlays)
        #if numLayers > numOverlays:
        #    print 'numLayers   = ' + str(numLayers)
        #    print 'numOverlays = ' + str(numOverlays)
        
        for layer in sorted(self.tiles[key]):
            image = self.tiles[key][layer]
            if not composite:
                composite = image.copy() # Create output image buffer
            else:
                #composite = Image.blend(composite, image, self.overlays[layer].opacity)#composite.paste(image, (0, 0), image)
                #if layer >= len(self.overlays):
                #    print 'Error coming!'
                #    print key
                try:
                    composite.paste(image, (0, 0), 
                                    ImageChops.multiply(image.split()[3], 
                                                        ImageChops.constant(image, int(self.overlays[layer].opacity * 255))))
                except: # TODO: Why do we get errors here after deleting overlays?
                    pass
                    #print 'CompositeTiles Exception caught!'
                    #print image.split()
                    #print layer
                    #print self.overlays
                    #print '========================'
        return composite

    def AddTile(self, image, key, overlay, layer):
        """Add a tile to the map.

        This keeps track of the tiles for each overlay in each grid cell.
        As new tiles come in, all the tiles in a grid cell are composited together
        into a new tile and any old tile for that spot is replaced.

        Args:
            image: The image tile to display.
            key: A tuple containing the key of the image (level, x, y)
            overlay: The overlay this tile belongs to (MapViewOverlay object).
            layer: The layer number this overlay corresponds to.    Only used
                    for caching purposes.
        """

        # This function is called from multiple threads, and
        # could use some synchronization, but it seems to work.
        if self.level == key[0] and overlay.show:   # Don't add late tiles from another level.
            self.tiles[key] = self.tiles.get(key, {})
            self.tiles[key][layer] = image

            newtile = self.CompositeTiles(key) # Combine all images into a single tile image
            newtile = ImageQt.ImageQt(newtile)
            with self.qttiles_lock:
                self.qttiles[key] = newtile
            self.update()

    def Zoom(self, event, direction):
        """Zoom the map.

        Args:
            event: The event that caused this zoom request.
            direction: The direction to zoom.   +1 for higher zoom, -1 for lower.
        """
        if self.level + direction >= 0:
            # Discard everything cached in the MapClient, and flush the fetch queues.
            self.Flush()
            self.tiles = {}
            with self.qttiles_lock:
                self.qttiles = {}

            if direction > 0:
                self.origin_x = self.origin_x * 2 - event.x()
                self.origin_y = self.origin_y * 2 - event.y()
            else:
                self.origin_x = (self.origin_x + event.x()) / 2
                self.origin_y = (self.origin_y + event.y()) / 2

            self.level += direction
            self.LoadTiles()
            
            # Notes on level/zoom:
            #  : pixels_per_lon_degree = (mercator_range / 360.0) * (2**level)
            #  : Each level of zoom doubles pixels_per_degree


    def wheelEvent(self, event):
        self.Zoom(event, 1 if event.delta() > 0 else -1)
        event.accept()

    def reload(self):
        self.Flush()
        self.tiles = {}
        with self.qttiles_lock:
            self.qttiles = {}
        self.LoadTiles()

    def __showAboutText(self):
        '''Pop up a little text box to display legal information'''
        QtGui.QMessageBox.about(self, 'about', ABOUT_TEXT)
    
    def __saveCurrentView(self):
        '''Saves the current map view to disk as a GeoTIFF'''
        
        # Get the handle of the currently active overlay
        # - This is what we will save to disk
        overlayToSave = None
        for o in self.overlays:
            if o.show:
                overlayToSave = o
        assert(overlayToSave != None) # Should at least be the google base map!
        
        current_view_bbox = self.GetMapBoundingBox()
        
        metersPerPixel = self.getApproxMetersPerPixel()
        scale = metersPerPixel
        
        # Pop open a window to get a file name from the user
        file_path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save image to', DEFAULT_SAVE_DIR))
        
        ## This will be used as a file name so it must be legal
        #saveName = overlayToSave.name.replace(' ', '_').replace('/', '-')
        
        #print overlayToSave.eeobject.getInfo()
        cmt.util.miscUtilities.downloadEeImage(overlayToSave.eeobject, current_view_bbox, scale, file_path, overlayToSave.vis_params)

    def contextMenuEvent(self, event):
    
        menu = QtGui.QMenu(self)

        TOP_BUTTON_HEIGHT  = 20
        TINY_BUTTON_WIDTH  = 50
        LARGE_BUTTON_WIDTH = 150

        # Set up text showing the location which was right-clicked
        (lon, lat) = self.pixelCoordToLonLat(event.x(), event.y()) # The event returns pixel coordinates
        location_widget = QtGui.QWidgetAction(menu)
        location_widget.setDefaultWidget(QtGui.QLabel("  Location: (%g, %g)" % (lon, lat)))
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(QtGui.QLabel("  Location: (%g, %g)" % (lon, lat)))

        # Add a "save image" button
        saveButton = QtGui.QPushButton('Save Current View', self)
        saveButton.setMinimumSize(LARGE_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        saveButton.setMaximumSize(LARGE_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        saveButton.clicked[bool].connect(self.__saveCurrentView)
        hbox.addWidget(saveButton)

        # Make a tiny "About" box for legal information
        aboutButton = QtGui.QPushButton('About', self)
        aboutButton.setMinimumSize(TINY_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        aboutButton.setMaximumSize(TINY_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        aboutButton.clicked[bool].connect(self.__showAboutText)
        hbox.addWidget(aboutButton)
        
        # Add the location and button to the pop up menu
        mainWidget = QtGui.QWidget()
        mainWidget.setLayout(hbox)
        location_widget.setDefaultWidget(mainWidget)
        menu.addAction(location_widget)

        # Add a toggle for each layer and put it in the right click menu
        for i in range(1, len(self.overlays)):
            action = QtGui.QWidgetAction(menu)
            item   = MapViewOverlayInfoWidget(self, i, event.x(), event.y())
            action.setDefaultWidget(item)
            menu.addAction(action)
            
        # Now pop up the new window!
        menu.popup(QtGui.QCursor.pos())
    
    def getPixel(self, layer, x, y):
        collection = ee.ImageCollection([self.overlays[layer].eeobject])
        # note: scale likely not correct
        (lon, lat) = self.pixelCoordToLonLat(x, y)
        point_extracted = collection.getRegion(ee.Geometry.Point(lon, lat), 1)

        return point_extracted
    
    def mousePressEvent(self, event):
        """Records the anchor location and sets drag handler."""
        
        self.mapClickedSignal.emit(event.x(), event.y()) # Send out clicked signal
        
        if event.button() == QtCore.Qt.LeftButton: # Now handle locally
            self.anchor_x = event.x()
            self.anchor_y = event.y()
            event.accept()
            return
        event.ignore()
        return

    def mouseMoveEvent(self, event):
        """Updates the map position and anchor position."""
        if self.anchor_x == None:
            event.ignore()
            return
        dx = event.x() - self.anchor_x
        dy = event.y() - self.anchor_y
        if dx or dy:
            self.origin_x += dx
            self.origin_y += dy
            self.anchor_x = event.x()
            self.anchor_y = event.y()
            self.update()
            event.accept()
            return
        event.ignore()

    def mouseReleaseEvent(self, event):
        """Unbind drag handler and redraw."""
        if event.button() == QtCore.Qt.LeftButton:
            self.anchor_x = None
            self.anchor_y = None
            self.LoadTiles()
            event.accept()
            return
        event.ignore()
        return

    def resizeEvent(self, event):
        """Handle resize events."""
        self.LoadTiles()
    
    def getApproxMetersPerPixel(self):
        '''Returns the approximate meters per pixel at the current location/zoom'''
        # The actual value differs in the X and Y direction and across the image
        
        mercator_range = 256.0
        scale = 2 ** self.level
        pixels_per_degree = (mercator_range / 360.0) * scale
        
        # Get the lat/lon of the center pixel
        width, height = self.width(), self.height()
        lon,   lat    = self.pixelCoordToLonLat(width/2, height/2)
        
        # Formula to compute the length of a degree at this latitude
        m1 = 111132.92
        m2 = -559.82
        m3 = 1.175
        m4 = -0.0023
        p1 = 111412.84
        p2 = -93.5
        p3 = 0.118
        lat_len_meters  = m1 + (m2 * math.cos(2 * lat)) + (m3 * math.cos(4 * lat)) + (m4 * math.cos(6 * lat))
        long_len_meters = (p1 * math.cos(lat)) + (p2 * math.cos(3 * lat)) + (p3 * math.cos(5 * lat))

        # Just take the average of the vertical and horizontal size
        meters_per_degree =  (lat_len_meters + long_len_meters) / 2
        # Convert to pixel units
        meters_per_pixel  = meters_per_degree / pixels_per_degree
        return meters_per_pixel
    
    
    def pixelCoordToLonLat(self, column, row):
        '''Return the longitude and latitude of a pixel in the map'''
        mercator_range = 256.0
        scale = 2 ** self.level
        origin_x = (mercator_range / 2.0) * scale
        origin_y = (mercator_range / 2.0) * scale
        pixels_per_lon_degree = (mercator_range / 360.0) * scale
        pixels_per_lon_radian = (mercator_range / (2 * math.pi)) * scale
        lng        = (column - self.origin_x - origin_x) /  pixels_per_lon_degree
        latRadians = (row    - self.origin_y - origin_y) / -pixels_per_lon_radian
        lat = (2 * math.atan(math.exp(latRadians)) - math.pi / 2) / (math.pi / 180.0)
        return (lng, lat)

    def lonLatToPixelCoord(self, lon, lat):
        '''Return the pixel coordinate in the map for a given longitude and latitude'''
        # From maps/api/javascript/geometry/mercator_projection.js
        mercator_range = 256.0
        scale = 2 ** self.level
        origin_x = (mercator_range / 2.0) * scale
        origin_y = (mercator_range / 2.0) * scale
        pixels_per_lon_degree = (mercator_range / 360.0) * scale
        pixels_per_lon_radian = (mercator_range / (2 * math.pi)) * scale

        column = origin_x + (lon * pixels_per_lon_degree)
        siny   = math.sin(lat * math.pi / 180.0)
        # Prevent sin() overflow.
        e = 1 - 1e-15
        if siny > e:
            siny = e
        elif siny < -e:
            siny = -e
        row = origin_y + (0.5 * math.log((1 + siny) / (1 - siny)) *
                                        -pixels_per_lon_radian)
        return (column, row)

    def CenterMap(self, lon, lat, opt_zoom=None):
        """Center the map at the given lon, lat and zoom level."""
        self.Flush()
        self.tiles = {}
        with self.qttiles_lock:
            self.qttiles = {}
        width, height = self.width(), self.height()
        if opt_zoom is not None:
            self.level = opt_zoom

        (column, row) = self.lonLatToPixelCoord(lon, lat)

        self.origin_x = -column + width  / 2
        self.origin_y = -row    + height / 2
        self.LoadTiles()

    def addToMap(self, eeobject, vis_params=None, name="", show=True):
        '''Ads an EE object to the map'''
        
        # Flatten any lists to comma separated strings - needed for eeobject.getMapId() call below!
        if vis_params:
            vis_params = dict(vis_params)
            for key in vis_params.keys():
                item = vis_params.get(key)
                if (isinstance(item, collections.Iterable) and (not isinstance(item, basestring))):
                     vis_params[key] = ','.join([str(x) for x in item])

        def execute_thread(waiting_threads):
            # get thread before starting
            with self.thread_lock:
                pass
            result = eeobject.getMapId(vis_params)
            for t in waiting_threads:
                t.join()
            with self.thread_lock:
                self.executing_threads.pop(0)
            return result

        with self.thread_lock:
            self.executing_threads.append(cmt.util.miscUtilities.waitForEeResult(functools.partial(execute_thread, list(self.executing_threads)),
                        lambda a : self.addOverlay(MakeTileManager(a), eeobject, name, show, vis_params)))


    def removeFromMap(self, eeobject):
        '''Removes an overlay from the map by matching its EE object'''
        self.Flush()
        for i in range(len(self.overlays)):
            if self.overlays[i].eeobject == eeobject:
                #print 'Removing overlay: ' + self.overlays[i].name
                del self.overlays[i]
                break
        self.LoadTiles()
        return


class TileManager(object):
    """Retrieves tiles from EE, resizes them, and manages the tile cache.
       Each overlay on the map requires its own TileManager instance."""

    TILE_WIDTH  = 256
    TILE_HEIGHT = 256
    MAX_CACHE   = 1000   # The maximum number of tiles to cache.
    _images   = {}       # The tile cache, keyed by (url, level, x, y).  Static class variable.
    _lru_keys = []       # Keys to the cached tiles, for cache ejection.

    def __init__(self, url):
        """Initialize the TileManager."""
        self.url = url
        NUM_WORKERS = 10
        self.delay = False
        # Google's map tile server thinks we are automating queries and blocks us, so we forcibly slow down
        if self.url == DEFAULT_MAP_URL_PATTERN:
            print 'Throttling tile download'
            NUM_WORKERS = 1
            self.delay = True
        # Make 10 workers, each an instance of the TileFetcher helper class.
        self.queue    = Queue.Queue()
        self.fetchers = [TileManager.TileFetcher(self) for unused_x in range(NUM_WORKERS)]
        self.constant = None

    def getTile(self, key, callback):       # pylint: disable=g-bad-name
        """Get the requested tile.

        If the requested tile is already cached, it's returned (sent to the
        callback) directly. If it's not cached, a check is made to see if
        a lower-res version is cached, and if so that's interpolated up, before
        a request for the actual tile is made.

        Args:
            key: The key of the tile to fetch.
            callback: The callback to call when the tile is available.  The callback
                    may be called more than once if a low-res version is available.
        """
        result = self.GetCachedTile(key)
        if result:
            callback(result) # Already have the tile, execute callback
        else:
            # Interpolate what we have and put the key on the fetch queue.
            # - The callback will get called once now and once when we get the tile
            self.queue.put((key, callback))
            self.Interpolate(key, callback)

    def Flush(self):
        """Empty the tile queue."""
        while not self.queue.empty():
            self.queue.get_nowait()

    def CalcTiles(self, level, bbox):
        """Calculate which tiles to load based on the visible viewport.

        Args:
            level: The level at which to calculate the required tiles.
            bbox: The viewport coordinates as a tuple (xlo, ylo, xhi, yhi])

        Returns:
            The list of tile keys to fill the given viewport.
        """
        tile_list = []
        for y in xrange(int(bbox[1] / TileManager.TILE_HEIGHT),
                                        int(bbox[3] / TileManager.TILE_HEIGHT + 1)):
            for x in xrange(int(bbox[0] / TileManager.TILE_WIDTH),
                                            int(bbox[2] / TileManager.TILE_WIDTH + 1)):
                tile_list.append((level, x, y))
        return tile_list

    def Interpolate(self, key, callback):
        """Upsample a lower res tile if one is available.

        Args:
            key: The tile key to upsample.
            callback: The callback to call when the tile is ready.
        """
        level, x, y = key
        delta  = 1
        result = None
        while level - delta > 0 and result is None:
            prevkey = (level - delta, x / 2, y / 2)
            result = self.GetCachedTile(prevkey)
            if not result:
                (_, x, y) = prevkey
                delta += 1

        if result:
            px = (key[1] % 2 ** delta) * TileManager.TILE_WIDTH / 2 ** delta
            py = (key[2] % 2 ** delta) * TileManager.TILE_HEIGHT / 2 ** delta
            image = (result.crop([px, py,
                                                        px + TileManager.TILE_WIDTH  / 2 ** delta,
                                                        py + TileManager.TILE_HEIGHT / 2 ** delta])
                             .resize((TileManager.TILE_WIDTH, TileManager.TILE_HEIGHT)))
            callback(image)

    def PutCacheTile(self, key, image):
        """Insert a new tile in the cache and eject old ones if it's too big."""
        cache_key = (self.url,) + key            # Generate key
        TileManager._images[cache_key] = image   # Store image in cache
        TileManager._lru_keys.append(cache_key)  # Record the key in insertion order
        
        # When the cache gets too big, clear the oldest tile.
        while len(TileManager._lru_keys) > TileManager.MAX_CACHE:
            remove_key = TileManager._lru_keys.pop(0) # The first entry is the oldest
            try:
                TileManager._images.pop(remove_key)
            except KeyError:
                # Just in case someone removed this before we did, don't die on cache clear!
                pass

    def GetCachedTile(self, key):
        """Returns the specified tile if it's in the cache."""
        cache_key = (self.url,) + key
        return TileManager._images.get(cache_key, None)

    def SaveCacheToDisk(self, path):
        '''Record all tile cache information to a file on disk'''
        def makePickleImage(image):
            return {'pixels': image.tostring(),
                        'size'  : image.size,
                        'mode'  : image.mode}
        # Prepare the images for pickle one at a time (the in-memory format is incompatible)
        pickle_images = []
        matched_keys  = []
        for key in TileManager._lru_keys:
            if not (key in TileManager._images):
                print 'Warning: Key not found in _images: ' + str(key)
                continue
            pickle_images.append(makePickleImage(TileManager._images[key]))
            matched_keys.append(key)
            
        with open(path, 'wb') as f:
            pickle.dump( (pickle_images, matched_keys), f)
        print 'Saved '+str(len(pickle_images))+' tiles from cache to path: ' + path
        
    def LoadCacheFromDisk(self, path):
        '''Read a cache file from disk'''
        
        def readPickleImage(pImage):
          return Image.fromstring(pImage['mode'], pImage['size'], pImage['pixels'])
        
        # Load the pickle formatted data
        with open(path, 'rb') as f:
            (pickle_images, TileManager._lru_keys) = pickle.load(f)
        # Unpack images one at a time
        TileManager._images = {}
        for (pImage, key) in zip(pickle_images, TileManager._lru_keys):
           TileManager._images[key] = readPickleImage(pImage)
        print 'Loaded '+str(len(TileManager._lru_keys))+' tiles to cache from path: ' + path

    class TileFetcher(threading.Thread):
        """A threaded URL fetcher used to retrieve tiles."""

        def __init__(self, parentTileMananger):
            threading.Thread.__init__(self)
            self.manager = parentTileMananger
            self.setDaemon(True)
            self.start()

        def run(self):
            """Pull URLs off the TileManager's queue and call the callback when done."""

            MAX_403_ERRORS = 10
            errorCount403 = 0
            while True:
                (key, callback) = self.manager.queue.get()
                # Google tile manager thinks we are automating queries and blocks us, so slow down
                if self.manager.delay and not self.manager.GetCachedTile(key):
                    delayTime = 0.05 + (random.random() * 0.2)
                    time.sleep(delayTime)
                # Check one more time that we don't have this yet.
                if not self.manager.GetCachedTile(key):
                
                    if errorCount403 > MAX_403_ERRORS:
                        continue
                
                    (level, x, y) = key
                    if x >= 0 and y >= 0 and x <= 2 ** level-1 and y <= 2 ** level-1:
                        url = self.manager.url % key
                        try:
                            data = urllib2.urlopen(url).read()
                        except urllib2.HTTPError as e:
                            print >> sys.stderr, e
                            print e
                            if 'HTTP Error 403' in e:
                                errorCount403 += 1
                                if errorCount403 > MAX_403_ERRORS:
                                    print 'Maximum HTTP Error 403 count exceeded, tile fetching disabled.'
                        else:
                            # PhotoImage can't handle alpha on LA images.
                            # - The convert command forces the image to be loaded into memory.
                            image = Image.open(cStringIO.StringIO(data)).convert('RGBA')
                            callback(image)
                            self.manager.PutCacheTile(key, image)


def MakeTileManager(mapid, baseurl=BASE_URL):
    """Create a TileManager from a mapid."""
    # The url is generated in a particular manner from the map ID.
    url = (baseurl + '/map/' + mapid['mapid'] + '/%d/%d/%d?token=' + mapid['token'])
    return TileManager(url)


class QtGuiWrapper(object):
    '''This class is created as a singleton and wraps the QT GUI.
        It offers a few interface functions for manipulating the map.
        
        The class is initalized with the TYPE of GUI class it will wrap.'''
        
    def __init__(self, guiClass):
        '''Initialize the class with the type of QT GUI to run'''
        self.guiClass = guiClass # Record the class type
        self.gui      = None     # The GUI is not initialized yet
        self.ready    = False

    def run(self):
        app        = QtGui.QApplication(sys.argv) # Do required QT init
        self.gui   = self.guiClass()              # Instantiate a GUI class object
        self.ready = True                         # Now we are ready to rock
        sys.exit(app.exec_())
    
    
    def __getattr__(self, attr):
        '''Forward any function call to the GUI class we instantiated'''
        while not self.ready:
            time.sleep(0.01) # Don't try anything until we are ready!
        try:
            return getattr(self.gui, attr) # Forward the call to the GUI class instance
        except:
            raise AttributeError(attr) # This happens if the GUI class does not support the call


#=================================================================================
# A Generic GUI implementation

class GenericMapGui(QtGui.QMainWindow):
    '''This sets up the main viewing window in QT, fills it up with a MapViewWidget,
       and then forwards all function calls to it.'''
    
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        
        self.tileManager = TileManager(DEFAULT_MAP_URL_PATTERN)
        if os.path.exists(LOCAL_MAP_CACHE_PATH):
            self.tileManager.LoadCacheFromDisk(LOCAL_MAP_CACHE_PATH)
        #except:
        #    print 'Unable to load cache information from ' + LOCAL_MAP_CACHE_PATH
        
        self.mapWidget = MapViewWidget(self.tileManager)


        # Set up all the components in a vertical layout
        vbox = QtGui.QVBoxLayout()
        
        # Add the main map widget
        vbox.addWidget(self.mapWidget)

        # QMainWindow requires that its layout be set in this manner
        mainWidget = QtGui.QWidget()
        mainWidget.setLayout(vbox)
        self.setCentralWidget(mainWidget)

        # This is the initial window size, but the user can resize it.
        self.setGeometry(100, 100, 720, 720) 
        self.setWindowTitle('EE Map View')
        self.show()

    def closeEvent(self,event):
        '''Dump the cache to disk'''
        #try:
        print 'Attempting to save tile cache...'
        self.tileManager.SaveCacheToDisk(LOCAL_MAP_CACHE_PATH)
        #except:
        #    print 'Unable to load cache information from ' + LOCAL_MAP_CACHE_PATH

    def keyPressEvent(self, event):
        """Handle keypress events."""
        if event.key() == QtCore.Qt.Key_Q:
            QtGui.QApplication.quit()

    def __getattr__(self, attr):
        '''Forward any unknown function call to MapViewWidget() widget we created'''
        try:
            return getattr(self.mapWidget, attr) # Forward the call to the MapViewWidget class
        except:
            raise AttributeError(attr) # This happens if the MapViewWidget class does not support the call



#=================================================================================
# Global objects and functions for interacting with the GUI
# - These are common operations and every GUI needs to support them.
# - These interfaces match an old deprecated version of the Earth Engine interface.

# A global GuiWrapper instance for addToMap convenience.
map_instance = None

# This is the type of GUI the functions below will create.
# - This defaults to the generic GUI, but it can be overwritten in the importing file.
gui_type = GenericMapGui


def addEmptyGui():
    '''Brings up the GUI without adding any new data to it'''
    # This just requires map_instance to be constructed
    global map_instance
    if not map_instance:
        map_instance = QtGuiWrapper(gui_type)

def run():
    ''' Runs the GUI thread (blocking). '''
    addEmptyGui()
    map_instance.run()

def addToMap(eeobject, vis_params=None, name="", show=True):
    """Adds a layer to the default map instance.

    Args:
            eeobject: The object to add to the map.
            vis_params: A dictionary of visualization parameters.   See
                        ee.data.getMapId().
            *unused_args: Unused arguments, left for compatibility with the JS API.

    This call exists to be an equivalent to the playground addToMap() call.
    It uses a global MapInstance to hang on to "the map".   If the MapInstance
    isn't initialized, this creates a new one.
    """
    addEmptyGui()
    map_instance.addToMap(eeobject, vis_params, name, show)

def removeFromMap(eeobject):
    """Removes a layer to the default map instance.

    Args:
            eeobject: The object to add to the map.
            
    This call uses a global MapInstance to hang on to "the map".   If the MapInstance
    isn't initialized, this creates a new one.
    """
    addEmptyGui()
    map_instance.removeFromMap(eeobject)


def centerMap(lng, lat, zoom):  # pylint: disable=g-bad-name
    """Center the default map instance at the given lat, lon and zoom values."""
    addEmptyGui()
    map_instance.CenterMap(lng, lat, zoom)

