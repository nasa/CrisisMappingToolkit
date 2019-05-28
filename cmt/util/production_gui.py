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

"""
A GUI to assist in quickly finding good parameters for well tested flood detection
algorithms.  It is built using tools from the core GUI file.

It consists of one large map with an inspector tool that appears when it is right clicked.
"""


import functools
import sys
import os
import json
import ee
import cmt.domain
import miscUtilities
import cmt.modis.modis_utilities
from cmt.util.imageRetrievalFunctions import *


import landsat_functions


try:
    import PyQt4                         # pylint: disable=g-import-not-at-top
    from PyQt4 import QtCore, QtGui
except ImportError:
    print("""
        ERROR: A Python library (PyQt4) used by the Earth Engine API mapclient
        module was not found.
        """)
    raise

from cmt.mapclient_qt import MapViewWidget, TileManager, ABOUT_TEXT, DEFAULT_MAP_URL_PATTERN, LOCAL_MAP_CACHE_PATH


import cmt.modis.flood_algorithms


#-----------------------------------------------------------------------------------------------

class DatePickerWidget(QtGui.QWidget):
    '''Simple calendar widget to select a date'''
    def __init__(self, callback, start_date):
        super(DatePickerWidget, self).__init__()
        
        self.datePicker = QtGui.QCalendarWidget(self)
        if start_date != None:
            self.datePicker.setSelectedDate(start_date)
        self.datePicker.clicked.connect(callback)

        # Set up all the components in a box layout
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.datePicker)

        self.setLayout(hbox) # Call QT function derived from parent QWidget class


class FeatureTrainerWindow(QtGui.QWidget):
    '''Window to control the selection of class regions in the main image'''
    def __init__(self, mapWidgetHandle, parent=None):
        super(FeatureTrainerWindow, self).__init__()
        
        # Set up connections to the main map widget
        self.mapWidgetHandle = mapWidgetHandle
        self.mapWidgetHandle.mapClickedSignal.connect(self._handleMapClick)
        
        # Start with no features
        self.classDict     = dict() # Class polygon storage
        self.selectedClass = None   # The name of the currently selected class
        self.polygonOnMap  = None   # The polygon currently drawn on the map
        self.lastFilePath  = ''
    
        # Set up the controls
        self.classNameLine     = QtGui.QLineEdit("Class name", self)
        self.addClassButton    = QtGui.QPushButton('Add New Class',   self)
        self.deleteClassButton = QtGui.QPushButton('Delete Class',    self)
        self.saveButton        = QtGui.QPushButton('Save Class File', self)
        self.loadButton        = QtGui.QPushButton('Load Class File', self)
        self.undoButton        = QtGui.QPushButton('Undo Last Point', self)
        self.deselectButton    = QtGui.QPushButton('Deselect Class',  self)
        self.classListBox   = QtGui.QListWidget()
        
        self.addClassButton.clicked.connect(self._addClass)
        self.deleteClassButton.clicked.connect(self._deleteClass)
        self.saveButton.clicked.connect(self._saveToFile)
        self.loadButton.clicked.connect(self._loadFromFile)
        self.undoButton.clicked.connect(self._undoPoint)
        self.deselectButton.clicked.connect(self._deselectList)
        self.classListBox.itemClicked.connect(self._setCurrentClass)
        
        # Lay everything out
        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.loadButton)
        vbox.addWidget(self.saveButton)
        vbox.addWidget(self.classListBox)
        vbox.addWidget(self.deselectButton)
        vbox.addWidget(self.classNameLine)
        vbox.addWidget(self.addClassButton)
        vbox.addWidget(self.deleteClassButton)
        vbox.addWidget(self.undoButton)
        vbox.addStretch(1)
        
        self.setLayout(vbox)
        
    def __del__(self):
        '''Clean up drawings on exit'''
        self.clearDrawings()

    def _clearDrawings(self):
        '''Clear anything this window has drawn on the map'''
        if self.polygonOnMap:
            self.mapWidgetHandle.removeFromMap(self.polygonOnMap)
            self.polygonOnMap = None

    def _updateMap(self):
        '''Updates the map with the current class'''

        self._clearDrawings() # Remove all existing drawings
        
        if not self.selectedClass:
            return

        # Convert from current class to an EE feature
        coordList = self.classDict[self.selectedClass]
        if len(coordList) == 0:
            return # Don't try to draw an empty list
        if len(coordList) < 3:
            eeRing = ee.Geometry.LineString(coordList)
        else:
            eeRing = ee.Geometry.LinearRing(coordList)

        # Add the feature to the map
        fc = ee.FeatureCollection(ee.Feature(eeRing))
        polyImage = ee.Image().byte().paint(fc, 0, 4)
        self.polygonOnMap = polyImage
        self.mapWidgetHandle.addToMap(polyImage, {}, self.selectedClass, True)

    def _setCurrentClass(self, current):
        '''Handle when a class name is clicked in the list'''
        self.selectedClass = str(current.text())
        self._updateMap()
        
    def _deselectList(self):
        '''Deselect all items in the class list'''
        for i in range(self.classListBox.count()):
            item = self.classListBox.item(i)
            self.classListBox.setItemSelected(item, False)
        self.selectedClass = None
        self._updateMap()
    
    def _repopulateList(self):
        '''Reset the list after loading a file'''
        self.classListBox.clear()
        for k in self.classDict:
            self.classListBox.addItem(k)
        
        
    def _addClass(self):
        '''Add a new class to the list'''
        className = str(self.classNameLine.text())
        if className in self.classDict: # Ensure that the class name is unique
            print('A class with this name already exists!')
            return
        self.classDict[className] = []
        self.classListBox.addItem(className)

    def _deleteClass(self):
        '''Remove class from the list'''
        if not self.selectedClass:
            print('No class selected!')
        className = self.selectedClass
        for item in self.classListBox.selectedItems():
            self.classListBox.takeItem(self.classListBox.row(item))
        self.classDict.pop(className, None) # Remove from our list
        self._deselectList()
    
    
    def _saveToFile(self):
        '''Write the class list to a JSON formatted text file'''
        path = str(QtGui.QFileDialog.getSaveFileName(self, 'Save File', '', '*.json'))
        #self.lastFilePath = path
        print(str(self.classDict))
        with open(path, 'w') as f:
            json.dump(self.classDict, f)
        print('Saved file ' + path)
    
    def _loadFromFile(self):
        '''Load a class list from a JSON formatted text file'''
        path = str(QtGui.QFileDialog.getOpenFileName(self, 'Open File', '', '*.json'))
        #self.lastFilePath = path
        with open(path, 'r') as f:
            self.classDict = json.load(f)
        print('Loaded file ' + path)
        print(self.classDict)
        self._repopulateList()
        self._updateMap()
        
    def _handleMapClick(self, x, y):
        '''When the map is clicked, add the point to the currently selected class'''
        if self.selectedClass:
            newCoord = self.mapWidgetHandle.pixelCoordToLonLat(x, y)
            self.classDict[self.selectedClass].append(newCoord)
            self._updateMap()
    
    def _undoPoint(self):
        '''Remove the last point of the currently selected class'''
        if self.selectedClass:
            self.classDict[self.selectedClass].pop()
            self._updateMap()
            

class FloodDetectParams:
    '''Stores the parameters used by the flood detection algorithm'''    
    def __init__(self):
        '''Construct with default parameters'''
        self.changeDetectThreshold = -3.0
        self.waterMaskThreshold    =  3.0
        self.statisticsRegion      = None # TODO: How to set this?
    
    def toString(self):
        print('Change threshold  = ' + str(self.changeDetectThreshold))
        print('Mask threshold    = ' + str(self.waterMaskThreshold))
        print('Statistics region = ' + str(self.statisticsRegion.getInfo()['coordinates']))


class ProductionGui(QtGui.QMainWindow):
    '''This sets up the main viewing window in QT, fills it up with a MapView,
       and then forwards all function calls to it.'''
    
    def __init__(self, parent=None):
        # First set up the flood detection stuff
        self.detectParams = FloodDetectParams()
        self.qtDate         = None # Date of the flood to analyze.
        self.floodDate      = None # Date of the flood to analyze.
        self.modisCloudMask = None # Cloud mask from 500m MODIS
        self.modisPrior     = None # First cloud free  MODIS image < date.
        self.modisPost      = None # First cloud light MODIS image >= the date.
        self.landsatPrior   = None # First cloud free  Landsat image < date.
        self.landsatPost    = None # First cloud light Landsat image >= the date.
        self.sentinel1Prior = None # First Sentinel1 image < date.
        self.sentinel1Post  = None # First Sentinel1 image >= date.
        self.demImage       = None # DEM image
        self.permWaterMask  = None # The permanent water mask, never changes.
        self.guestImage     = None # A manually loaded image
        self.eeFunction     = None # Flood detection results TODO: Rename variable!
        self.landsatType    = None # One of the types at the top of the file
        self.classWindow    = None # Handle for class training window
        
        
        # Init a tile manager and load it with cache from disk
        self.tileManager = TileManager(DEFAULT_MAP_URL_PATTERN)
        if os.path.exists(LOCAL_MAP_CACHE_PATH):
            self.tileManager.LoadCacheFromDisk(LOCAL_MAP_CACHE_PATH)
        #except:
        #    print 'Unable to load cache information from ' + LOCAL_MAP_CACHE_PATH

        # Now set up all the GUI stuff
        QtGui.QWidget.__init__(self, parent)
        self.mapWidget = MapViewWidget(self.tileManager)
        
        # Set up all the components in a vertical layout
        vbox = QtGui.QVBoxLayout()
        
        # Add a horizontal row of widgets at the top
        topHorizontalBox = QtGui.QHBoxLayout()
        
        TOP_BUTTON_HEIGHT = 30
        TOP_LARGE_BUTTON_WIDTH  = 150
        TOP_SMALL_BUTTON_WIDTH  = 100
        
        # Add a date selector to the top row of widgets
        DEFAULT_START_DATE = ee.Date.fromYMD(2015, 11, 15)
        self.floodDate = DEFAULT_START_DATE
        dateString     = '2015/11/15' # TODO: Generate from the default start date
        self.dateButton = QtGui.QPushButton(dateString, self)
        self.dateButton.setMinimumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.dateButton.setMaximumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.dateButton.clicked[bool].connect(self._showCalendar)
        topHorizontalBox.addWidget(self.dateButton)
        
        # Add a "Set Region" button to the top row of widgets
        self.regionButton = QtGui.QPushButton('Set Processing Region', self)
        self.regionButton.setMinimumSize(TOP_LARGE_BUTTON_WIDTH, TOP_BUTTON_HEIGHT) 
        self.regionButton.setMaximumSize(TOP_LARGE_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.regionButton.clicked[bool].connect(self._setRegionToView)
        topHorizontalBox.addWidget(self.regionButton)

        # Add a "Load Images" button to the top row of widgets
        self.loadImagesButton = QtGui.QPushButton('Load Images', self)
        self.loadImagesButton.setMinimumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.loadImagesButton.setMaximumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.loadImagesButton.clicked[bool].connect(self._loadImageData)
        topHorizontalBox.addWidget(self.loadImagesButton)

        # Add a "Detect Flood" button to the top row of widgets
        self.loadFloodButton1 = QtGui.QPushButton('Detect Flood', self)
        self.loadFloodButton1.setMinimumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.loadFloodButton1.setMaximumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.loadFloodButton1.clicked[bool].connect(self._loadFloodDetect)
        topHorizontalBox.addWidget(self.loadFloodButton1)

        # Add a "Load ME Image" button to the top row of widgets
        self.loadMeImageButton = QtGui.QPushButton('Load ME Image', self)
        self.loadMeImageButton.setMinimumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.loadMeImageButton.setMaximumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.loadMeImageButton.clicked[bool].connect(self._loadMapsEngineImage)
        topHorizontalBox.addWidget(self.loadMeImageButton)

        # Add a "Open Class Trainer" button to the top row of widgets
        self.openTrainerButton = QtGui.QPushButton('Open Class Trainer', self)
        self.openTrainerButton.setMinimumSize(TOP_LARGE_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.openTrainerButton.setMaximumSize(TOP_LARGE_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.openTrainerButton.clicked[bool].connect(self._openClassTrainer)
        topHorizontalBox.addWidget(self.openTrainerButton)

        # Add a "Clear All" button to the top row of widgets
        self.clearButton = QtGui.QPushButton('Clear Map', self)
        self.clearButton.setMinimumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.clearButton.setMaximumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.clearButton.clicked[bool].connect(self._unloadCurrentImages)
        topHorizontalBox.addWidget(self.clearButton)

        # Add the row of widgets on the top of the GUI
        vbox.addLayout(topHorizontalBox)
        # Add the main map widget
        vbox.addWidget(self.mapWidget)
                
        # Set up a horizontal box below the map
        bottomHorizontalBox = QtGui.QHBoxLayout()
        # On the left side is a vertical box for the parameter controls
        paramControlBoxV    = QtGui.QVBoxLayout()
        
        # First set up some sliders to adjust thresholds
        # - Currently we have two thresholds
        sliderParams = ['Change Detection Threshold', 'Water Mask Threshold']
        paramMin     = [-10, -10]
        paramMax     = [ 10,  10]
        defaultVal   = [-3, 3]
        
        # Build each of the parameter sliders
        self.sliderList = []
        for name, minVal, maxVal, default in zip(sliderParams, paramMin, paramMax, defaultVal):
            # Stick the horizontal box on the bottom of the main vertical box
            paramControlBoxV = self._addParamSlider(name, maxVal, minVal, default, paramControlBoxV)
        bottomHorizontalBox.addLayout(paramControlBoxV) # Add sliders to bottom horizontal box

        # Add a "Detect Flood" button to the right of the parameter controls
        # - This is identical to the button above the map.
        self.loadFloodButton2 = QtGui.QPushButton('Detect Flood', self)
        self.loadFloodButton2.setMinimumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.loadFloodButton2.setMaximumSize(TOP_SMALL_BUTTON_WIDTH, TOP_BUTTON_HEIGHT)
        self.loadFloodButton2.clicked[bool].connect(self._loadFloodDetect)
        bottomHorizontalBox.addWidget(self.loadFloodButton2)

        # Add all the stuff at the bottom to the main layout
        vbox.addLayout(bottomHorizontalBox)

        # QMainWindow requires that its layout be set in this manner
        mainWidget = QtGui.QWidget()
        mainWidget.setLayout(vbox)
        self.setCentralWidget(mainWidget)
        
        # This is the initial window size, but the user can resize it.
        self.setGeometry(100, 100, 720, 720) 
        self.setWindowTitle('EE Flood Detector Tool')
        self.show()

    def closeEvent(self, event):
        '''Cleanup all other windows and dump cache to disk'''
        if self.classWindow:
            self.classWindow.close()

        #try:
        print('Attempting to save tile cache...')
        self.tileManager.SaveCacheToDisk(LOCAL_MAP_CACHE_PATH)
        #except:
        #    print 'Unable to load cache information from ' + LOCAL_MAP_CACHE_PATH


    def _addParamSlider(self, name, maxVal, minVal, defaultVal, container):
        '''Adds a single parameter slider to the passed in container.'''
        # All parameter sliders are handled by the _handleParamChange function
    
        NAME_WIDTH    = 250
        SLIDER_HEIGHT = 20
        SLIDER_WIDTH  = 400
        NUM_TICKS     = 4
        
        # Set up this value slider
        slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        slider.setRange(minVal, maxVal) 
        slider.setValue(defaultVal)
        valRange = maxVal - minVal
        slider.setTickInterval(valRange/NUM_TICKS) # Add five tick marks
        slider.setMinimumSize(SLIDER_WIDTH, SLIDER_HEIGHT)
        slider.setMaximumSize(SLIDER_WIDTH, SLIDER_HEIGHT)
        # Use 'partial' to send the param name to the callback function
        callbackFunction = functools.partial(self._handleParamChange, parameterName=name) 
        slider.valueChanged.connect(callbackFunction) # Whenever the slider is moved, trigger callback function
        self.sliderList.append(slider) # TODO: Do we need this?
    
        # Make box with the name
        nameBox = QtGui.QLabel(name, self)
        nameBox.setMinimumSize(NAME_WIDTH, SLIDER_HEIGHT)
        nameBox.setMaximumSize(NAME_WIDTH, SLIDER_HEIGHT)
        
        # Put the name to the left of the slider
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(nameBox)
        hbox.addWidget(slider)
        
        # Stick the horizontal box on the bottom of the main vertical box
        container.addLayout(hbox)
        return container
    
    
    def _openClassTrainer(self):
        '''Open a new window to define classifier training regions'''
        # Create the class trainer window and connect it to the map widget
        
        if not self.classWindow:
            self.classWindow = FeatureTrainerWindow(self.mapWidget)
            self.classWindow.show()
        else:
            self.classWindow.show()
        
        

    def _unloadCurrentImages(self):
        '''Just unload all the current images. Low level function'''
        if self.modisPrior:
            self.mapWidget.removeFromMap(self.modisPrior)
            self.modisPrior = None
        if self.modisPost:
            self.mapWidget.removeFromMap(self.modisPost)
            self.modisPost = None
        if self.modisCloudMask:
            self.mapWidget.removeFromMap(self.modisCloudMask)
            self.modisCloudMask = None
        if self.landsatPrior:
            self.mapWidget.removeFromMap(self.landsatPrior)
            self.landsatPrior = None
        if self.landsatPost:
            self.mapWidget.removeFromMap(self.landsatPost)
            self.landsatPost = None
        if self.sentinel1Prior:
            self.mapWidget.removeFromMap(self.sentinel1Prior)
            self.sentinel1Prior = None
        if self.sentinel1Post:
            self.mapWidget.removeFromMap(self.sentinel1Post)
            self.sentinel1Post = None
        if self.demImage:
            self.mapWidget.removeFromMap(self.demImage)
            self.demImage = None
        if self.guestImage:
            self.mapWidget.removeFromMap(self.guestImage)
            self.guestImage = None
        if self.eeFunction:
            self.mapWidget.removeFromMap(self.eeFunction)
            self.eeFunction = None

    class GuestImageDialog(QtGui.QDialog):
        '''Popup window for the user to fill in information about an image loaded in Maps Engine'''
        def __init__(self, parent = None):
            super(ProductionGui.GuestImageDialog, self).__init__(parent)
            
            # Get the list of available sensors
            self.radioNames = miscUtilities.getDefinedSensorNames()
            if not self.radioNames:
                raise Exception('Could not load any sensors!')
            # Set up sensor selection buttons
            self.radioButtons = []
            for name in self.radioNames:
                self.radioButtons.append(QtGui.QRadioButton(name))
            self.radioButtons[0].setChecked(True)
            
            # Set up the other controls
            self.lineIn       = QtGui.QLineEdit("Asset ID")
            self.okButton     = QtGui.QPushButton('Ok',     self)
            self.cancelButton = QtGui.QPushButton('Cancel', self)            
            self.okButton.clicked.connect(self.accept)
            self.cancelButton.clicked.connect(self.reject)
            
            # Lay everything out
            vbox = QtGui.QVBoxLayout(self)
            vbox.addWidget(self.lineIn)
            for button in self.radioButtons:
               vbox.addWidget(button)
            vbox.addWidget(self.okButton)
            vbox.addWidget(self.cancelButton)
            vbox.addStretch(1)
        
        def getValues(self):
            '''Return the selected values'''
            for button in self.radioButtons: # Should always be exactly one button checked!
                if button.isChecked():
                    return (str(button.text()), str(self.lineIn.text()))
            raise Exception('GUI buttons are broken!')
        
    def _loadMapsEngineImage(self):
        '''Loads in an image stored in Google Maps Engine'''

        raise Exception('TODO: Update for the death of Maps Engine!')
    
        # Pop up a dialog and get the input values back
        dialog = self.GuestImageDialog(self)
        result = dialog.exec_()
        sensorName, eeID = dialog.getValues()
        
        if not result: # Do nothing if the cancel button was pressed
            return
        
        # Extract the asset ID from the Earth Engine ID
        # - The Earth Engine ID looks like this: GME/images/18108519531116889794-15007110928626476628
        pt = eeID.rfind('/')
        if not pt:
            print('Invalid Earth Engine ID entered!')
        assetID = eeID[pt+1:]
        
        # Clear any existing guest image from the map
        if self.guestImage:
            self.mapWidget.removeFromMap(self.guestImage)
            self.guestImage = None

        # Retrieve the sensor information for this image from the XML file
        sensorXmlPath = os.path.join(SENSOR_FILE_DIR, sensorName + ".xml")
        print('Reading file: ' + sensorXmlPath)
        sensorInfo = cmt.domain.SensorObservation(xml_source=sensorXmlPath, manual_ee_ID=assetID)
        
        # Add the new guest image to the map
        self.guestImage, vis_params, im_name, show = sensorInfo.visualize()
        self.mapWidget.addToMap(self.guestImage, vis_params, sensorName, True)
        
        



    def _displayCurrentImages(self):
        '''Add all the current images to the map. Low level function'''
        
        MODIS_RANGE  = [0, 3000]
        DEM_RANGE    = [0, 1000]
        
        # TODO: Come up with a method for setting the intensity bounds!
        landsatVisParams = {'bands': ['red', 'green', 'blue'], 'min': 0, 'max': 0.75}
            
        sentinel1VisParams = {'bands': ['vv'], 'min': -30, 'max': 5}
        
        modisVisParams = {'bands': ['sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06'],
                          'min': MODIS_RANGE[0], 'max': MODIS_RANGE[1]}

        if self.landsatPrior:
            self.mapWidget.addToMap(self.landsatPrior, landsatVisParams, 'LANDSAT Pre-Flood', False)
        if self.landsatPost:
            self.mapWidget.addToMap(self.landsatPost, landsatVisParams, 'LANDSAT Post-Flood', True)
           
        if self.sentinel1Prior:
            self.mapWidget.addToMap(self.sentinel1Prior, sentinel1VisParams, 'Sentinel-1 Pre-Flood', False)
        if self.sentinel1Post:
            self.mapWidget.addToMap(self.sentinel1Post, sentinel1VisParams, 'Sentinel-1 Post-Flood', False)
         
        if self.modisPrior:
            self.mapWidget.addToMap(self.modisPrior, modisVisParams, 'MODIS Pre-Flood', False)
        if self.modisPost:
            self.mapWidget.addToMap(self.modisPost, modisVisParams, 'MODIS Post-Flood', False)
            
        # This one works a little differently since it never changes
        # - We just add this once and never remove it.
        if not self.permWaterMask:
            self.permWaterMask = cmt.modis.modis_utilities.get_permanent_water_mask()
            self.mapWidget.addToMap(self.permWaterMask.mask(self.permWaterMask),
                                    {'min': 0 , 'max': 1, 'palette': '000000, 0000FF'}, 'Permanent Water Mask', False)
            
        if self.modisCloudMask:
            vis_params = {'min': 0, 'max': 1, 'palette': '000000, FF0000'}
            self.mapWidget.addToMap(self.modisCloudMask, vis_params, '1km Bad MODIS pixels', False)

        if self.demImage:
            vis_params = {'min': DEM_RANGE[0], 'max': DEM_RANGE[1]}
            self.mapWidget.addToMap(self.demImage, vis_params, 'Digital Elevation Map', False)

    def _loadImageData(self):
        '''Updates the MODIS and LANDSAT images for the current date'''
        
        # Check that we have all the information we need
        bounds = self.detectParams.statisticsRegion
        if (not self.floodDate) or (not bounds):
            print("Can't load any images until the date and bounds are set!")
            return
 
        # Unload all the current images, including any flood detection results.
        self._unloadCurrentImages()

        # Check if we are inside the US.
        # - Some higher res data is only available within the US.
        boundsInsideTheUS = miscUtilities.regionIsInUnitedStates(self.detectParams.statisticsRegion)

        # Set up the search range of dates for each image type
        PRIOR_SEARCH_RANGE_DAYS = 20 # Not picky about the pre-flooding image
        POST_SEARCH_RANGE_DAYS  = 20  # After too many days the flood will have receded.
        priorStartDate = self.floodDate.advance(-1*PRIOR_SEARCH_RANGE_DAYS, 'day') # Prior stops before the date
        postStartDate  = self.floodDate # Post starts at the date

        # Load before and after Landsat data
        # - We can afford to be pickier about clouds in the prior image than in the post image.
        try:
            self.landsatPrior = getCloudFreeLandsat(bounds, priorStartDate, PRIOR_SEARCH_RANGE_DAYS, 
                                                    maxCloudPercentage=0.05, searchMethod='decreasing')
            priorLsDate = cmt.util.miscUtilities.getDateFromLandsatInfo(self.landsatPrior.getInfo())
            print('Found prior Landsat date: ' + priorLsDate)
        except Exception as e:
            print('Failed to find prior Landsat image!')
            print(str(e))
            print((sys.exc_info()[0]))
            self.landsatPrior = None
        try:
            self.landsatPost  = getCloudFreeLandsat(bounds, postStartDate,  POST_SEARCH_RANGE_DAYS,  
                                                    maxCloudPercentage=0.25, searchMethod='increasing')
            postLsDate = cmt.util.miscUtilities.getDateFromLandsatInfo(self.landsatPost.getInfo())
            print('Found post Landsat date: ' +postLsDate)
        except Exception as e:
            print('Failed to find post Landsat image!') 
            print(str(e))
            print((sys.exc_info()[0]))
            self.landsatPost = None


        # Load before and after Sentinel-1 data
        try:
            self.sentinel1Prior = getNearestSentinel1(bounds, priorStartDate, PRIOR_SEARCH_RANGE_DAYS, 
                                                     searchMethod='decreasing')
            priorS1Date = cmt.util.miscUtilities.getDateFromSentinel1Info(self.sentinel1Prior.getInfo())
            print('Found prior Sentinel-1 date: ' + priorS1Date)
        except Exception as e:
            print('Failed to find prior Sentinel-1 image!')
            print(str(e))
            print((sys.exc_info()[0]))
            self.sentinel1Prior = None
        try:
            self.sentinel1Post  = getNearestSentinel1(bounds, postStartDate,  POST_SEARCH_RANGE_DAYS,  
                                                     searchMethod='increasing')
            postS1Date = cmt.util.miscUtilities.getDateFromSentinel1Info(self.sentinel1Post.getInfo())
            print('Found post Sentinel-1 date: ' + postS1Date)
        except Exception as e:
            print('Failed to find post Sentinel-1 image!') 
            print(str(e))
            print((sys.exc_info()[0]))
            self.sentinel1Post = None
            
        
        # Load before and after MODIS data
        # - We can afford to be pickier about clouds in the prior image than in the post image.
        try:
            self.modisPrior = getCloudFreeModis(bounds, priorStartDate, PRIOR_SEARCH_RANGE_DAYS, 
                                                maxCloudPercentage=0.05, searchMethod='decreasing')
            priorModisDate = cmt.util.miscUtilities.getDateFromModisInfo(self.modisPrior.getInfo())
            print('Found prior MODIS date: ' + priorModisDate)
        except Exception as e:
            print('Failed to find prior MODIS image!')
            print(str(e))
            print((sys.exc_info()[0]))
            self.modisPrior = None
        try:
            self.modisPost  = getCloudFreeModis(bounds, postStartDate,  POST_SEARCH_RANGE_DAYS,  
                                                maxCloudPercentage=0.25, searchMethod='increasing')
            postModisDate = cmt.util.miscUtilities.getDateFromModisInfo(self.modisPost.getInfo())
            print('Found post MODIS date: ' + postModisDate)
        except Exception as e:
            print('Failed to find post MODIS image!')
            print(str(e))
            print((sys.exc_info()[0]))
            self.modisPost = None

        if self.modisPost:
            # Extract the MODIS cloud mask
            # - We only use the cloud mask from the POST image
            self.modisCloudMask = cmt.modis.modis_utilities.getModisBadPixelMask(self.modisPost)
            self.modisCloudMask = self.modisCloudMask.mask(self.modisCloudMask)

        # Load a DEM
        demName = 'CGIAR/SRTM90_V4' # The default 90m global DEM
        if (boundsInsideTheUS):
            demName = 'ned_13' # The US 10m DEM
        self.demImage = ee.Image(demName)
        
        # Now add all the images to the map!
        self._displayCurrentImages()


    def _loadFloodDetect(self):
        '''Creates the Earth Engine flood detection function and adds it to the map'''
        
        # Check prerequisites
        if (not self.modisPost) or (not self.floodDate) or (not self.detectParams.statisticsRegion):
            print("Can't detect floods without image data and flood date!")
            return
        
        # Remove the last EE function from the map
        if self.eeFunction:
            self.mapWidget.removeFromMap(self.eeFunction)
        
        print('Starting flood detection with the following parameters:')
        print('--> Water mask threshold       = ' + str(self.detectParams.waterMaskThreshold))
        print('--> Change detection threshold = ' + str(self.detectParams.changeDetectThreshold))
        
        # Generate a new EE function
        self.eeFunction = cmt.modis.misc_algorithms.history_diff_core(self.modisPost,
                                        self.floodDate, self.detectParams.waterMaskThreshold,
                                        self.detectParams.changeDetectThreshold, self.detectParams.statisticsRegion)
        self.eeFunction = self.eeFunction.mask(self.eeFunction)
        
        # Add the new EE function to the map
        # TODO: Set display parameters with widgets
        OPACITY = 0.5
        COLOR   = '00FFFF'
        self.mapWidget.addToMap(self.eeFunction, {'min': 0, 'max': 1, 'opacity': OPACITY, 'palette': COLOR}, 'Flood Detection Results', True)

    def _handleParamChange(self, value, parameterName='DEBUG'):
        '''Reload an EE algorithm when one of its parameters is set in the GUI'''
        if parameterName == 'Change Detection Threshold':
            self.detectParams.changeDetectThreshold = value
            return
        if parameterName == 'Water Mask Threshold':
            self.detectParams.waterMaskThreshold = value
            return
        print('WARNING: Parameter ' + parameterName + ' is set to: ' + str(value))
        
    def _setDate(self, date):
        '''Sets the current date'''
        self.qtDate = date
        self.floodDate = ee.Date.fromYMD(date.year(), date.month(), date.day()) # Load into an EE object
        self.dateButton.setText(date.toString('yyyy/MM/dd')) # Format for humans to read
        
    def _setRegionToView(self):
        '''Sets the processing region to the current viewable area'''
        # Extract the current viewing bounds as [minLon, minLat, maxLon, maxLat]
        lonLatBounds = self.mapWidget.GetMapBoundingBox() # TODO: This function does not work!!!!
        print('Setting region to: ' + str(lonLatBounds))
        self.detectParams.statisticsRegion = apply(ee.geometry.Geometry.Rectangle, lonLatBounds)

    def _showCalendar(self):
        '''Pop up a little calendar window so the user can select a date'''
        menu   = QtGui.QMenu(self)
        action = QtGui.QWidgetAction(menu)
        item   = DatePickerWidget(self._setDate, self.qtDate) # Pass in callback function
        action.setDefaultWidget(item)
        menu.addAction(action)
        menu.popup(QtGui.QCursor.pos())

    def _showAboutText(self):
        '''Pop up a little text box to display legal information'''
        QtGui.QMessageBox.about(self, 'about', ABOUT_TEXT)

    def keyPressEvent(self, event):
        """Handle keypress events."""
        if event.key() == QtCore.Qt.Key_Q:
            QtGui.QApplication.quit()

    def __getattr__(self, attr):
        '''Forward any undefined function call to the main map widget'''
        try:
            return getattr(self.mapWidget, attr) # Forward the call to the MapViewWidget class
        except:
            print(str(attr))
            raise AttributeError(attr) # This happens if the MapViewWidget class does not support the call




































