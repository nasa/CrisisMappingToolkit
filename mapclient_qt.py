"""A slippy map GUI.

Implements a tiled slippy map using Tk canvas. Displays map tiles using
whatever projection the tiles are in and only knows about tile coordinates,
(as opposed to geospatial coordinates.)	This assumes that the tile-space is
organized as a power-of-two pyramid, with the origin in the upper left corner.
This currently has several spots that are hard-coded for 256x256 tiles, even
though MapOverlay tries to track this.

Supports mouse-based pan and zoom as well as tile upsampling while waiting
for new tiles to load.	The map to display is specified by a MapOverlay, and
added to the GUI on creation or manually using addOverlay()
	gui = MapClient(MakeOverlay(mapid))

Tiles are referenced using a key of (level, x, y) throughout.

Several of the functions are named to match the Google Maps Javascript API,
and therefore violate style guidelines.
"""



# TODO(user):
# 1) Add a zoom bar.
# 2) When the move() is happening inside the Drag function, it'd be
#		a good idea to use a semaphore to keep new tiles from being added
#		and subsequently moved.

import collections
import cStringIO
import functools
import math
import Queue
import sys
import time
import threading
import urllib2

import ee

# check if the Python imaging libraries used by the mapclient module are
# installed
try:
	from PIL import ImageQt						 # pylint: disable=g-import-not-at-top
	from PIL import Image, ImageChops							 # pylint: disable=g-import-not-at-top
except ImportError:
	print """
		ERROR: A Python library (PIL) used by the Earth Engine API mapclient module
		was not found. Information on PIL can be found at:
		http://pypi.python.org/pypi/PIL
		"""
	raise

try:
	import PyQt4						 # pylint: disable=g-import-not-at-top
	from PyQt4 import QtCore, QtGui
except ImportError:
	print """
		ERROR: A Python library (PyQt4) used by the Earth Engine API mapclient
		module was not found.
		"""
	raise

# The default URL to fetch tiles from.	We could pull this from the EE library,
# however this doesn't have any other dependencies on that yet, so let's not.
BASE_URL = 'https://earthengine.googleapis.com'

# This is a URL pattern for creating an overlay from the google maps base map.
# The z, x and y arguments at the end correspond to level, x, y here.
DEFAULT_MAP_URL_PATTERN = ('http://mt1.google.com/vt/lyrs=m@176000000&hl=en&'
													 'src=app&z=%d&x=%d&y=%d')

class WaitForEEResult(threading.Thread):
	def __init__(self, eeobject, function):
		threading.Thread.__init__(self)
		self.eeobject = eeobject
		self.function = function
		self.setDaemon(True)
		self.start()
	def run(self):
		self.function(self.eeobject.getInfo())

class MapGui(QtGui.QMainWindow):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self, parent)
		self.mapwidget = MapView()

		self.setCentralWidget(self.mapwidget)
		
		self.setGeometry(100, 100, 720, 720)
		self.setWindowTitle('EE Map View')
		self.show()

	def CenterMap(self, lon, lat, opt_zoom=None):
		self.mapwidget.CenterMap(lon, lat, opt_zoom)
	
	def addOverlay(self, overlay, eeobject, name, show):
		self.mapwidget.addOverlay(overlay, eeobject, name, show)
	
	def keyPressEvent(self, event):
		"""Handle keypress events."""
		if event.key() == QtCore.Qt.Key_Q:
			QtGui.QApplication.quit()

class MapViewOverlay(object):
	def __init__(self, overlay, eeobject, name, show=True, opacity=1.0):
		self.overlay = overlay
		self.eeobject = eeobject
		self.name = name
		self.show = show
		self.opacity = opacity

class MapOverlayMenuWidget(QtGui.QWidget):
	def __init__(self, parent, layer, x, y):
		super(MapOverlayMenuWidget, self).__init__()
		self.parent = parent
		self.layer = layer
		self.x = x
		self.y = y
		overlay = self.parent.overlays[layer]
		
		self.check_box = QtGui.QCheckBox(self)
		self.check_box.setChecked(overlay.show)
		self.check_box.stateChanged.connect(self.toggle_visible)

		self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
		self.slider.setRange(0, 100)
		self.slider.setValue(int(overlay.opacity * 100))
		self.slider.setTickInterval(25)
		self.slider.setMinimumSize(100, 10)
		self.slider.setMaximumSize(100, 50)
		self.slider.valueChanged.connect(self.set_transparency)

		self.name = QtGui.QLabel(overlay.name, self)
		self.name.setMinimumSize(100, 10)
		
		self.value = QtGui.QLabel('...', self)
		self.value.setMinimumSize(200, 10)

		self.pixel_loader = WaitForEEResult(self.parent.getPixel(layer, x, y), self.set_pixel_value)

		hbox = QtGui.QHBoxLayout()
		hbox.addWidget(self.check_box)
		hbox.addWidget(self.name)
		hbox.addWidget(self.slider)
		hbox.addWidget(self.value)

		self.setLayout(hbox)
	
	def set_pixel_value(self, value):
		names = value[0][4:]
		values = value[1][4:]
		text = ''
		for i in range(len(names)):
			if i != 0:
				text += ', '
			text += str(names[i]) + ': ' + str(values[i])
		self.value.setText(text)
	
	def toggle_visible(self):
		self.parent.overlays[self.layer].show = not self.parent.overlays[self.layer].show
		self.parent.reload()
	
	def set_transparency(self, value):
		self.parent.overlays[self.layer].opacity = value / 100.0
		self.parent.reload()

	def hideEvent(self, event):
		self.parent.setFocus()

class MapView(QtGui.QWidget):
	"""A simple discrete zoom level map viewer."""

	def __init__(self, opt_overlay=None):
		super(MapView, self).__init__()
		
		self.tiles = {}		 # The cached stack of images at each grid cell.
		self.qttiles = {}	 # The cached PhotoImage at each grid cell.
		self.qttiles_lock = threading.RLock()
		self.level = 2			# Starting zoom level
		self.origin_x = None	 # The map origin x offset at the current level.
		self.origin_y = None	 # The map origin y offset at the current level.
		self.anchor_x = None	 # Drag anchor.
		self.anchor_y = None	 # Drag anchor.

		# Map origin offsets; start at the center of the map.
		self.origin_x = (-(2 ** self.level) * 128) + self.width() / 2
		self.origin_y = (-(2 ** self.level) * 128) + self.height() / 2

		if not opt_overlay:
			# Default to a google maps basemap
			opt_overlay = MapOverlay(DEFAULT_MAP_URL_PATTERN)

		# The array of overlays are displayed as last on top.
		self.overlays = [MapViewOverlay(opt_overlay, None, 'Google Maps')]
	
	def paintEvent(self, event):
		painter = QtGui.QPainter()
		with self.qttiles_lock:
			painter.begin(self)
			for key in self.qttiles.keys():
				if key[0] != self.level:
					continue
				image = self.qttiles[key]
				xpos = key[1] * image.width() + self.origin_x
				ypos = key[2] * image.height() + self.origin_y
				painter.drawImage(QtCore.QPoint(xpos, ypos), image)
			painter.end()

	def addOverlay(self, overlay, eeobject, name, show):						 # pylint: disable=g-bad-name
		"""Add an overlay to the map."""
		self.overlays.append(MapViewOverlay(overlay, eeobject, name, show))
		self.LoadTiles()

	def GetViewport(self):
		"""Return the visible portion of the map as [xlo, ylo, xhi, yhi]."""
		width, height = self.width(), self.height()
		return [-self.origin_x, -self.origin_y,
						-self.origin_x + width, -self.origin_y + height]

	def LoadTiles(self):
		"""Refresh the entire map."""
		# Start with the overlay on top.
		for i, overlay in reversed(list(enumerate(self.overlays))):
			if not overlay.show:
				continue
			tile_list = overlay.overlay.CalcTiles(self.level, self.GetViewport())
			for key in tile_list:
				overlay.overlay.getTile(key, functools.partial(
						self.AddTile, key=key, overlay=self.overlays[i], layer=i))

	def Flush(self):
		"""Empty out all the image fetching queues."""
		for overlay in self.overlays:
			overlay.overlay.Flush()

	def CompositeTiles(self, key):
		"""Composite together all the tiles in this cell into a single image."""
		composite = None
		for layer in sorted(self.tiles[key]):
			image = self.tiles[key][layer]
			if not composite:
				composite = image.copy()
			else:
				#composite = Image.blend(composite, image, self.overlays[layer].opacity)#composite.paste(image, (0, 0), image)
				composite.paste(image, (0, 0), ImageChops.multiply(image.split()[3], ImageChops.constant(image, int(self.overlays[layer].opacity * 255))))
		return composite

	def AddTile(self, image, key, overlay, layer):
		"""Add a tile to the map.

		This keeps track of the tiles for each overlay in each grid cell.
		As new tiles come in, all the tiles in a grid cell are composited together
		into a new tile and any old tile for that spot is replaced.

		Args:
			image: The image tile to display.
			key: A tuple containing the key of the image (level, x, y)
			overlay: The overlay this tile belongs to.
			layer: The layer number this overlay corresponds to.	Only used
					for caching purposes.
		"""
		# TODO(user): This function is called from multiple threads, and
		# could use some synchronization, but it seems to work.
		if self.level == key[0]:			# Don't add late tiles from another level.
			self.tiles[key] = self.tiles.get(key, {})
			self.tiles[key][layer] = image

			newtile = self.CompositeTiles(key)
			newtile = ImageQt.ImageQt(newtile)
			with self.qttiles_lock:
				self.qttiles[key] = newtile
			self.update()

	def Zoom(self, event, direction):
		"""Zoom the map.

		Args:
			event: The event that caused this zoom request.
			direction: The direction to zoom.	+1 for higher zoom, -1 for lower.
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
	
	def wheelEvent(self, event):
		self.Zoom(event, 1 if event.delta() > 0 else -1)
		event.accept()

	def reload(self):
		self.Flush()
		self.tiles = {}
		with self.qttiles_lock:
			self.qttiles = {}
		self.LoadTiles()

	def contextMenuEvent(self, event):
		menu = QtGui.QMenu(self)
		for i in range(1, len(self.overlays)):
			action = QtGui.QWidgetAction(menu)
			item = MapOverlayMenuWidget(self, i, event.x(), event.y())
			action.setDefaultWidget(item)
			menu.addAction(action)
		menu.popup(QtGui.QCursor.pos())
	
	def getPixel(self, layer, x, y):
		collection = ee.ImageCollection([self.overlays[layer].eeobject])
		# note: scale likely not correct
		(lon, lat) = self.XYToLonLat(x, y)
		point_extracted = collection.getRegion(ee.Geometry.Point(lon, lat), 1)

		return point_extracted
	
	def mousePressEvent(self, event):
		"""Records the anchor location and sets drag handler."""
		if event.button() == QtCore.Qt.LeftButton:
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
	
	def XYToLonLat(self, x, y):
		mercator_range = 256.0
		scale = 2 ** self.level
		origin_x = (mercator_range / 2.0) * scale
		origin_y = (mercator_range / 2.0) * scale
		pixels_per_lon_degree = (mercator_range / 360.0) * scale
		pixels_per_lon_radian = (mercator_range / (2 * math.pi)) * scale
		lng = (x - self.origin_x - origin_x) / pixels_per_lon_degree
		latRadians = (y - self.origin_y - origin_y) / -pixels_per_lon_radian
		lat = (2 * math.atan(math.exp(latRadians)) - math.pi / 2) / (math.pi / 180.0)
		return (lng, lat)

	def lonLatToXY(self, lon, lat):
		# From maps/api/javascript/geometry/mercator_projection.js
		mercator_range = 256.0
		scale = 2 ** self.level
		origin_x = (mercator_range / 2.0) * scale
		origin_y = (mercator_range / 2.0) * scale
		pixels_per_lon_degree = (mercator_range / 360.0) * scale
		pixels_per_lon_radian = (mercator_range / (2 * math.pi)) * scale

		x = origin_x + (lon * pixels_per_lon_degree)
		siny = math.sin(lat * math.pi / 180.0)
		# Prevent sin() overflow.
		e = 1 - 1e-15
		if siny > e:
			siny = e
		elif siny < -e:
			siny = -e
		y = origin_y + (0.5 * math.log((1 + siny) / (1 - siny)) *
										-pixels_per_lon_radian)
		return (x, y)

	def CenterMap(self, lon, lat, opt_zoom=None):
		"""Center the map at the given lon, lat and zoom level."""
		self.Flush()
		self.tiles = {}
		with self.qttiles_lock:
			self.qttiles = {}
		width, height = self.width(), self.height()
		if opt_zoom is not None:
			self.level = opt_zoom

		(x, y) = self.lonLatToXY(lon, lat)

		self.origin_x = -x + width / 2
		self.origin_y = -y + height / 2
		self.LoadTiles()


class MapOverlay(object):
	"""A class representing a map overlay."""

	TILE_WIDTH = 256
	TILE_HEIGHT = 256
	MAX_CACHE = 1000					# The maximum number of tiles to cache.
	_images = {}							 # The tile cache, keyed by (url, level, x, y).
	_lru_keys = []						 # Keys to the cached tiles, for cache ejection.

	def __init__(self, url):
		"""Initialize the MapOverlay."""
		self.url = url
		# Make 10 workers.
		self.queue = Queue.Queue()
		self.fetchers = [MapOverlay.TileFetcher(self) for unused_x in range(10)]
		self.constant = None

	def getTile(self, key, callback):		# pylint: disable=g-bad-name
		"""Get the requested tile.

		If the requested tile is already cached, it's returned (sent to the
		callback) directly.	If it's not cached, a check is made to see if
		a lower-res version is cached, and if so that's interpolated up, before
		a request for the actual tile is made.

		Args:
			key: The key of the tile to fetch.
			callback: The callback to call when the tile is available.	The callback
					may be called more than once if a low-res version is available.
		"""
		result = self.GetCachedTile(key)
		if result:
			callback(result)
		else:
			# Interpolate what we have and put the key on the fetch queue.
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
		for y in xrange(int(bbox[1] / MapOverlay.TILE_HEIGHT),
										int(bbox[3] / MapOverlay.TILE_HEIGHT + 1)):
			for x in xrange(int(bbox[0] / MapOverlay.TILE_WIDTH),
											int(bbox[2] / MapOverlay.TILE_WIDTH + 1)):
				tile_list.append((level, x, y))
		return tile_list

	def Interpolate(self, key, callback):
		"""Upsample a lower res tile if one is available.

		Args:
			key: The tile key to upsample.
			callback: The callback to call when the tile is ready.
		"""
		level, x, y = key
		delta = 1
		result = None
		while level - delta > 0 and result is None:
			prevkey = (level - delta, x / 2, y / 2)
			result = self.GetCachedTile(prevkey)
			if not result:
				(_, x, y) = prevkey
				delta += 1

		if result:
			px = (key[1] % 2 ** delta) * MapOverlay.TILE_WIDTH / 2 ** delta
			py = (key[2] % 2 ** delta) * MapOverlay.TILE_HEIGHT / 2 ** delta
			image = (result.crop([px, py,
														px + MapOverlay.TILE_WIDTH / 2 ** delta,
														py + MapOverlay.TILE_HEIGHT / 2 ** delta])
							 .resize((MapOverlay.TILE_WIDTH, MapOverlay.TILE_HEIGHT)))
			callback(image)

	def PutCacheTile(self, key, image):
		"""Insert a new tile in the cache and eject old ones if it's too big."""
		cache_key = (self.url,) + key
		MapOverlay._images[cache_key] = image
		MapOverlay._lru_keys.append(cache_key)
		while len(MapOverlay._lru_keys) > MapOverlay.MAX_CACHE:
			remove_key = MapOverlay._lru_keys.pop(0)
			try:
				MapOverlay._images.pop(remove_key)
			except KeyError:
				# Just in case someone removed this before we did.
				pass

	def GetCachedTile(self, key):
		"""Returns the specified tile if it's in the cache."""
		cache_key = (self.url,) + key
		return MapOverlay._images.get(cache_key, None)

	class TileFetcher(threading.Thread):
		"""A threaded URL fetcher."""

		def __init__(self, overlay):
			threading.Thread.__init__(self)
			self.overlay = overlay
			self.setDaemon(True)
			self.start()

		def run(self):
			"""Pull URLs off the ovelay's queue and call the callback when done."""
			while True:
				(key, callback) = self.overlay.queue.get()
				# Check one more time that we don't have this yet.
				if not self.overlay.GetCachedTile(key):
					(level, x, y) = key
					if x >= 0 and y >= 0 and x <= 2 ** level-1 and y <= 2 ** level-1:
						url = self.overlay.url % key
						try:
							data = urllib2.urlopen(url).read()
						except urllib2.HTTPError as e:
							print >> sys.stderr, e
						else:
							# PhotoImage can't handle alpha on LA images.
							image = Image.open(cStringIO.StringIO(data)).convert('RGBA')
							callback(image)
							self.overlay.PutCacheTile(key, image)


def MakeOverlay(mapid, baseurl=BASE_URL):
	"""Create an overlay from a mapid."""
	url = (baseurl + '/map/' + mapid['mapid'] + '/%d/%d/%d?token=' +
				 mapid['token'])
	return MapOverlay(url)

class MapClient(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.ready = False
		self.start()

	def run(self):
		app = QtGui.QApplication(sys.argv)
		self.gui = MapGui()
		self.gui.show()
		self.ready = True
		sys.exit(app.exec_())
	
	def CenterMap(self, lon, lat, opt_zoom=None):
		while not self.ready:
			time.sleep(0.01)
		self.gui.CenterMap(lon, lat, opt_zoom)
	
	def addOverlay(self, overlay, eeobject, name, show):
		while not self.ready:
			time.sleep(0.01)
		self.gui.addOverlay(overlay, eeobject, name, show)

#
# A global MapClient instance for addToMap convenience.
#
map_instance = None


# pylint: disable=g-bad-name
def addToMap(eeobject, vis_params=None, name="", show=True):
	"""Adds a layer to the default map instance.

	Args:
			eeobject: the object to add to the map.
			vis_params: a dictionary of visualization parameters.	See
					ee.data.getMapId().
			*unused_args: unused arguments, left for compatibility with the JS API.

	This call exists to be an equivalent to the playground addToMap() call.
	It uses a global MapInstance to hang on to "the map".	If the MapInstance
	isn't initializd, this creates a new one.
	"""
	# Flatten any lists to comma separated strings.
	if vis_params:
		vis_params = dict(vis_params)
		for key in vis_params.keys():
			item = vis_params.get(key)
			if (isinstance(item, collections.Iterable) and
					not isinstance(item, basestring)):
				vis_params[key] = ','.join([str(x) for x in item])

	overlay = MakeOverlay(eeobject.getMapId(vis_params))

	global map_instance
	if not map_instance:
		map_instance = MapClient()
	map_instance.addOverlay(overlay, eeobject, name, show)


def centerMap(lng, lat, zoom):	# pylint: disable=g-bad-name
	"""Center the default map instance at the given lat, lon and zoom values."""
	global map_instance
	if not map_instance:
		map_instance = MapClient()

	map_instance.CenterMap(lng, lat, zoom)

