#!/usr/bin/python

import os
import sys

import zipfile
import tempfile

from xml.etree import ElementTree

if len(sys.argv) < 3:
	print 'Usage: kmz_to_geotiff.py input.kmz output.tiff'
	sys.exit(0)

try:
	z = zipfile.ZipFile(sys.argv[1], 'r')
except:
	print >> sys.stderr, 'Could not open file ' + sys.argv[1] + '.'
	sys.exit(1)

namespace = 'http://earth.google.com/kml/2.1'
ElementTree.register_namespace('', namespace)

try:
	tree = ElementTree.parse(z.open('overlay.kml', 'r'))
except:
	print >> sys.stderr, 'Could not open overlay.kml in kmz file.'
	sys.exit(1)

try:
	links = tree.getroot().find('{%s}Document' % (namespace)).findall('{%s}NetworkLink' % namespace)
except:
	print >> sys.stderr, 'Kmz file did not have expected structure.'
	raise
	sys.exit(1)

tempdir = tempfile.mkdtemp()

input_images = []
input_image_string = ''
failure = False

for link in links:
	latlon = link.find('{%s}Region' % namespace).find('{%s}LatLonAltBox' % namespace)
	name = link.find('{%s}Link' % namespace).find('{%s}href' % namespace).text[:-4]
	north = latlon.find('{%s}north' % namespace).text
	south = latlon.find('{%s}south' % namespace).text
	east = latlon.find('{%s}east' % namespace).text
	west = latlon.find('{%s}west' % namespace).text
	output_name = tempdir + os.sep + name + '.tiff'
	z.extract('images/%s.png' % (name), tempdir)
	imagefile = tempdir + os.sep + ('images/%s.png' % (name))
	ret = os.system('gdal_translate -q -a_srs "+proj=longlat +datum=WGS84 +no_defs" -a_ullr %s %s %s %s %s %s' % (west, north, east, south, imagefile, output_name))
	os.remove(imagefile)
	if ret != 0:
		print >> sys.stderr, 'Failed to convert tile %s to geotiff.' % (name)
		failure = True
		break
	input_images.append(output_name)
	input_image_string += ' ' + output_name
	print 'Converting ' + output_name + '...'

print 'Merging tiles...'

if not failure:
	ret = os.system('gdal_merge.py -init "0 0 0 0" -o %s %s' % (sys.argv[2], input_image_string))
	if ret != 0:
		print >> sys.stderr, 'Merge failed.'
		failure = True
	else:
		print 'Merge successful. Output to ' + sys.argv[2] + '.'

for i in input_images:
	os.remove(i)
os.rmdir(tempdir + os.sep + 'images')
os.rmdir(tempdir)

if failure:
	sys.exit(1)

