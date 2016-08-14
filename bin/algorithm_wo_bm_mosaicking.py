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


# -----------------------------------------------------------------------------
# algorithm_wo_bm_mosaicking.py Code Summary 
#
# This code is not running or up to date. This is a code that was started
# to visual images that were both cloudless over lakes and near each other 
# in date and mosaic them together. This is a time consuming and arduous 
# process, but if it can be perfect and sped up, larger lakes can be
# run on the model. 
#
# -----------------------------------------------------------------------------


from sys import argv
import ee
import os 
ee.Initialize()
import sys 
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from cmt.mapclient_qt import centerMap, addToMap
import cmt.mapclient_qt
from lake_measure import *
import threading
from cmt.util.imageRetrievalFunctions import *
import cmt
from cmt.lake_functions import * 

script, lake, start_date, end_date = argv

# Determine which mask is asked for 
mask = 'llama' # Given mask
if mask == 'permanent':
	permbool = True
	llamabool = False
elif mask == 'llama':
	llamabool = True 
	permbool = False

start_date = ee.Date(start_date)
end_date   = ee.Date(end_date)
daterange = 2000.
results_dir = 'results'
fai = False
ndti = False

# Band Options:  ['blue', 'green', 'red', 'nir', 'swir1', 'temp', 'swir2']
# Import specific lake according to what was specified
if lake is not None:
	mylake = ee.FeatureCollection('ft:1igNpJRGtsq2RtJuieuV0DwMg5b7nU8ZHGgLbC7iq', "geometry").filterMetadata(u'system:index', u'equals', lake).toList(1000000)


def main():
	ee_lake = ee.Feature(mylake.get(0))
	actualbounds = ee_lake.geometry() # The actual bounds of the lake
	ee_bounds = ee_lake.geometry().buffer(1000) # Adding a small buffer
	ee_extrabounds = ee_lake.geometry().buffer(50000) # Adding a big buffer for the image to display in that circular area

	# Selects from all images in Landsat 5 & 8 
	collection = get_image_collection(ee_bounds, start_date, end_date)
	collection = collection.sort('system:time_start')
	imageList = collection.toList(100)
	imageInfo = imageList.getInfo()
	numFound = len(imageInfo)
	searchIndices = range(0,numFound)

	# Filtering for cloud cover
	maxCloudPercentage = 0.05  

	num_cloudy = collection.toList(100).length().getInfo()
	info_cloudy = collection.getInfo()
	imageList = collection.toList(100)


	mylist = ee.List([])
	fitandclear = ee.List([])
	unfit = ee.List([])

	for num in range(0,num_cloudy):
		thisImage = ee.Image(imageList.get(num))
		percentage = getCloudPercentage(thisImage,ee_bounds)
		print 'Detected Landsat cloud percentage: ' + str(percentage)
		TF = containsLakev2(thisImage,ee_bounds)
		if percentage < 0.05 and TF == True:
			fitandclear = fitandclear.add(thisImage)
		elif percentage < 0.05 and TF == False:
			unfit = unfit.add(thisImage)





	# BELOW IS MY ATTEMPTS AT TRYING TO MAP THE FUNCTIONS. it doesn't work 
	# FIRST AND FOREMOST because it can't access the server while it's being called
	# ei, you can't use getInfo() because it can't access it's servers, so =(

	'''
	#firstimage = ee.Image(collection.first())
	#cloudPercentage = FindClouds(firstimage)
	#print cloudPercentage
	def squrt(number):
		output = number**0.5
		return output 

	def getPixelSize(image,bounds):
		resolution = 60
		reducedimage = image.reduce(ee.Reducer.allNonZero())
		areaCount = reducedimage.reduceRegion(ee.Reducer.sum(), ee_bounds, resolution)
		return areaCount


	#myimage = ee.Image(collection.first())
	#mypercent = getCloudPercentage(myimage,ee_bounds)
	#print mypercent 

	def CloudsAndFit(image):
		#areaCount = getPixelSize(image,ee_bounds)
		cloudCount,areaCount = getCloudPercentagev2(image,ee_bounds) # From cmt.util.landsat_function
		#print 'Detected Landsat cloud percentage: ' + str(cloudPercentage)
		#image = image.set({'Cloudiness': areaCount})
		#imageinfo = image.getInfo()
		percentage = cloudCount.getInfo()['constant'] / areaCount.getInfo()['all']

		return image 

		
		cloudMax = 0.05
		print 'Detected Landsat cloud percentage: ' + str(cloudPercentage)
		if cloudPercentage > 0.05:
			image = image.set({'Cloudiness':'Pass'})
			TF = containsLakev2(image,ee_bounds)
			if TF == True:
				image = image.set({'Within_Image':'Pass'})
			elif TF == False:
				image = image.set({'Within_Image':'Fail'})
		else:
			image = image.set({'Cloudiness':'Fail'})
			image = image.set({'Within_Image':'NA'})
	
		

	definedcollection = collection.map(CloudsAndFit)
	#fitandclear = definedcollection.filter(ee.Filter.eq('Within_Image','Pass')) # Filters out BOTH those who aren't in image & those who are too cloudy
	#unfit = definedcollection.filter(ee.Filter.eq('Within_Image','Fail')) # Takes those that aren't in the image completely & removes those who are too cloudy

	print 'done'

	'''


	num_unfit = unfit.length().getInfo()
	listoflists = []

	for num in range(0,num_unfit):
		currentImage = ee.Image(unfit.get(num))
		currentdate = (float(currentImage.getInfo()['properties']['system:time_start']))/100000000.
		nearbylist = ee.List([])
		for i in range(0,num_unfit):
			nearImage = ee.Image(unfit.get(i))
			nearDate = (float(nearImage.getInfo()['properties']['system:time_start']))/100000000.
			dateDifference = abs(nearDate-currentdate)
			if dateDifference < 10:
				nearbylist = nearbylist.add(nearImage)
			nearbyIC = ee.ImageCollection(nearbylist)
			cover = containsLakev3(nearbyIC,ee_bounds)
			if cover >= 0.95:
				listoflists = listoflists.append(nearbylist)

	print len(listoflists)

	currentIC = listoflists[0]


	

	# Determine the Dates of the Images & The Satellite They're From 
	num_features = currentIC.toList(100).length().getInfo()
	info_on_IC = currentIC.getInfo()

	#firstimage = ee.Image(qualityCollection.first())
	#addToMap(firstimage,{'bands':['blue', 'green', 'red'],'gain':'480, 420, 330'}, 'firstimage')
	#TF = containsLakev2(firstimage,actualbounds)
	#print TF

	
	# If there are enough features, gets the names of all the features 
	if num_features>0:

		mymosiac = currentIC.mosaic();
		#addToMap(mymosiac,{'bands':['blue', 'green', 'red'],'gain':'480, 420, 330'}, 'spatial mosaic')
		addToMap(mymosiac,{'bands':['blue', 'green', 'red'],'max':0.4}, 'spatial mosaic')

		clippedmosiac = mymosiac.clip(ee_bounds)
		
		permWaterMask = cmt.modis.modis_utilities.get_permanent_water_mask()
		addToMap(permWaterMask.mask(permWaterMask),
			{'min': 0 , 'max': 1, 'paleitte': '000000, 0000FF'}, 'Permanent Water Mask', permbool)
		print 'added colorssss'

		waterarea = detect_water(clippedmosiac)
		addToMap(waterarea.mask(waterarea),
			{'min': 0 , 'max': 1, 'palette': '000000, 0000FF'}, 'Llama Water Mask', llamabool)

		for num in range(0,num_features):
			dates = info_on_IC['features'][num]['properties']['DATE_ACQUIRED']
			satellite = info_on_IC['features'][num]['properties']['SPACECRAFT_ID']
			print num+1, dates, satellite
		


	elif num_features == 0:
		print 'Not enough cloudless images; please expand time range.'
	

	center = ee_bounds.centroid().getInfo()['coordinates']
	centerMap(center[0], center[1], 11)
	print 'centering done'
	
	



t = threading.Thread(target=main)
t.start()
cmt.mapclient_qt.run()

