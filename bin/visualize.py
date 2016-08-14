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
# visualize.py Code Summary
#
# The arguments using argparse are presented below, or accessed using
# python visualize.py -h. You can input a specific lake and dates for that
# lake to run. You can set it to write the lakes to a file, or you can have
# the program visualize the imagery with the LLAMA mask.
#
# -----------------------------------------------------------------------------

import logging
logging.basicConfig(level=logging.ERROR)
import functools
import urllib
import glob
from sys import argv
import ee
import os 
ee.Initialize()
import sys 
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from cmt.mapclient_qt import centerMap, addToMap
import cmt.mapclient_qt
from cmt.lake_helper.lake_measure import *
import threading
from cmt.util.imageRetrievalFunctions import *
import cmt
from cmt.util.lake_functions import * 
import argparse


#! /usr/local/bin/python
'''
import signal

def signal_handler(signal, frame):
	print('You pressed Ctrl+C!')
	Lake_Level_Cancel()
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

'''

# -----------------------------------------------------------------------
# Use argparse to interpret inputs in the command line in a more user-
# friendly fashion/Other inputs
#
# Outputs when -h or --help is used: 
#
#  -h, --help            show this help message and exit
#  --name NAME, -n NAME  Enter name of lake (option A)
#  --id ID, -i ID        Enter lake ID (option B)
#  --start_date START_DATE, -s START_DATE
#                        Enter start date of period to assess
#  --end_date END_DATE, -e END_DATE
#                        Enter end date of period to assess
#  --visualize, -v       Use command if you want to view LANDSAT imagery
#  --write, -w           Use command to scan through all imagery and write to
#                        file
# -----------------------------------------------------------------------



def mapImage(image, bounds):
	addToMap(image, {'bands':['blue', 'green', 'red'], 'max':0.4}, 'LANDSAT Image')
	clippedImage = image.clip(bounds)
	waterarea = detect_water(clippedImage)
	llamabool = True
	thisImageInfo = image.getInfo()
	date = thisImageInfo['properties']['DATE_ACQUIRED']
	print 'Image Date:', date
	satellite = thisImageInfo['properties']['SPACECRAFT_ID']
	print 'Satellite:', satellite

	center = bounds.centroid().getInfo()['coordinates']
	centerMap(center[0], center[1], 11)


#def main():


# ----------- Manipulting Input Arguments ---------------------------
# Initialize
lakeid = None 
lakename = None 


# Set up all of the arguments 
parser = argparse.ArgumentParser()
parser.add_argument("--name", "-n", help="Enter name of lake (option A)")
parser.add_argument("--id", "-i", help="Enter lake ID (option B)")
parser.add_argument("--start_date", "-s", help="Enter start date of period to assess")
parser.add_argument("--end_date", "-e", help="Enter end date of period to assess")
parser.add_argument("--visualize", "-v", help="Use command if you want to view LANDSAT imagery", \
	action="store_true")
parser.add_argument("--write", "-w", help="Use command to scan through all imagery and write to file ", \
	action="store_true")
args = parser.parse_args()


if args.id is not None:
	lakeid = args.id 
elif args.name is not None:
	lakename = args.name
elif args.id and args.name is not None:
   print "Please input either lake name or lake id using --name (-n) or --id (-i)"

date = args.start_date
enddate = args.end_date
start_date = ee.Date(date)
end_date   = ee.Date(enddate)


results_dir = 'results' # Set the directory where you want the files to go 


# If you have the lake id, and not the lake name, get the lake name & vice versa
if lakeid is not None:
	mylake = acquireLake(lakeid)
elif lakename is not None:
	mylake = acquireLake(lakename)


mylakeinfo = mylake.getInfo()
mylakeinfo = mylakeinfo[0]


if lakename is None:
	try:
		lakename = mylakeinfo['LAKE_NAME']
	except: 
		lakename = 'No_Name'

if lakeid is None:
	lakeid = mylakeinfo['id']
# --------------------------------------------------------------------




# ------------- Manipulating image collections -----------------------
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
# --------------------------------------------------------------------




# ----------- Writing to File Aspect ---------------------------------
# Check if the argument for write is on, start threads with either
# the lake ID or the lake name, depending on which is given 
if args.write is True:
	if lakeid is not None:
		Lake_Level_Run([lakeid], date, enddate,'results',fai=False, ndti=False, \
			update_function = None, complete_function = None)
	elif lakename is not None:
		Lake_Level_Run([lakeid], date, enddate,'results',fai=False, ndti=False, \
			update_function = None, complete_function = None)
# --------------------------------------------------------------------
	


# ----------- Visualization Aspect -----------------------------------
# Check if visualization is on and cycle through the 
if args.visualize is True:
	cmt.mapclient_qt.run() # Open mapping GUI

	for num in range(0,num_cloudy):
		thisImage = ee.Image(imageList.get(num))
		percentage = getCloudPercentage(thisImage,ee_bounds)
		print 'Detected Landsat cloud percentage: ' + str(percentage)
		TF = True
		#TF = containsLakev2(thisImage,ee_bounds)
		if percentage < 0.05 and TF == True:
			break
	mapImage(thisImage,ee_bounds)
# --------------------------------------------------------------------



