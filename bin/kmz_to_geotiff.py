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

import os
import subprocess
import sys

import zipfile
import tempfile

from xml.etree import ElementTree

'''
Converts an UAVSAR image to geotiff format.
'''

# How to call this script
USAGE_STRING = 'kmz_to_geotiff.py input.kmz output.tiff'


# --- Start of script ---

if len(sys.argv) < 3:
    print 'Usage: ' + USAGE_STRING
    sys.exit(0)

single_mosaic = False

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

failure = False

input_images = []
if single_mosaic:
    input_image_string = ''
else:
    nonblank_images = []
    prev_lats = (None, None)

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

    try:
        out = subprocess.check_output('identify -verbose %s | grep "standard deviation"' % (imagefile),
                                      shell=True)
        blank = False
        if float(out.split('\n')[0].strip().split()[2]) == 0.0:
            blank = True
    except:
        print >> sys.stderr, "Failed to determine if image %s was blank." % (imagefile)
        failure = True
        break

    print 'Converting ' + output_name + '...'
    if not blank:
        ret = os.system('gdal_translate -q -b 1 -b 2 -b 3 -mask 4 -co "PHOTOMETRIC=RGB" -a_srs "+proj=longlat +datum=WGS84 +no_defs" -a_ullr %s %s %s %s --config GDAL_TIFF_INTERNAL_MASK YES %s %s' % (west, north, east, south, imagefile, output_name))
    os.remove(imagefile)
    if blank:
        continue
    if ret != 0:
        print >> sys.stderr, 'Failed to convert tile %s to geotiff.' % (name)
        failure = True
        break
    input_images.append(output_name)
    if single_mosaic:
        input_image_string += ' ' + output_name
    else:
        if prev_lats[0] == north and prev_lats[1] == south:
            nonblank_images[-1].append(output_name)
        else:
            nonblank_images.append([output_name])
            prev_lats = (north, south)

if not failure:
    print 'Merging tiles...'
    if single_mosaic:
        ret = os.system('gdal_merge.py -init "0 0 0 0" -o %s %s' % (sys.argv[2], input_image_string))
        if ret != 0:
            print >> sys.stderr, 'Merge failed.'
            failure = True
        else:
            print 'Merge successful. Output to ' + sys.argv[2] + '.'
    else:
        print nonblank_images
        cur_line = 0
        while cur_line < len(nonblank_images):
            cur_images = ''
            ROWS = 1
            for i in range(ROWS):
                if cur_line + i < len(nonblank_images):
                    for a in nonblank_images[cur_line+i]:
                        cur_images += a + ' '
            cur_line = cur_line + ROWS
            ret = os.system('gdal_merge.py -o %s_%d.tiff %s' % (sys.argv[2], cur_line / ROWS, cur_images))
            if ret != 0:
                print >> sys.stderr, 'Merge failed.'
                failure = True
            else:
                print 'Merge successful. Output to ' + sys.argv[2] + '_' + str(cur_line / ROWS) + '.tiff.'


for i in input_images:
    os.remove(i)
os.rmdir(tempdir + os.sep + 'images')
os.rmdir(tempdir)

if failure:
    sys.exit(1)
