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


import os, sys, optparse
from bs4 import BeautifulSoup
import urllib2
import datetime
import shutil
import subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import detect_flood_cmd


manual='''---=== daily_detector.py ===---
This is a dedicated tool intended to process available image data from
currently active flood regions and record the results.  It relies on
the GDACS website to obtain flood alerts and Earth Engine to fetch and
process the flood data.  The results can be uploaded to a server where they
can be accessed with a Google App Engine tool.

This tool uses "detect_flood_cmd.py" to process each date/location
found on the GDACS website.  See the documentation for that tool for
information about how flood detection is performed.

Currently this tool is hard-coded to upload files to a location
on our server but this could be easily modified.  The "app_engine"
folder in our contains the Google App Engine tool which is 
designed to display the results created by this tool.
'''



STORAGE_URL = 'http://byss.arc.nasa.gov/smcmich1/cmt_detections/'

# Currently they are stored in our web share on byss
SERVER_STORAGE_PATH = '/byss/docroot/smcmich1/cmt_detections/'

def archiveResult(kmlPath, dateString):
    '''Archives completed flood detection results.
       Call updateArchiveFolder once all individual results are archived.'''

    dateFolder = os.path.join(SERVER_STORAGE_PATH, dateString)
    os.system('mkdir '+dateFolder)

    # Write the output file
    filename = os.path.split(kmlPath)[1]
    destinationPath = os.path.join(dateFolder, filename)
    cmd = 'cp ' +kmlPath+ ' '+destinationPath
    os.system(cmd)

    # Update the HTML indices
    # - TODO: Include this tool!
    cmd = 'python $HOME/makeHtmlIndex.py ' + dateFolder
    print cmd
    os.system(cmd)


def updateArchiveFolder():
    '''Does the final updates needed at the top level storage folder.'''
    cmd = 'python $HOME/makeHtmlIndex.py ' + SERVER_STORAGE_PATH
    print cmd
    os.system(cmd)

    generateCompositeKmlFile(SERVER_STORAGE_PATH)


def generateCompositeKmlFile(topFolder):
    '''Create a single KML file in the top folder that includes
       network links to all of the other folders.'''
       
    # Get output path and make sure it is clear
    outputPath = os.path.join(topFolder, 'composite.kml')
    if os.path.exists(outputPath):
        os.remove(outputPath)
    
    # Start writing the file  
    f = open(outputPath, 'w')
    f.write('''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">
<Document>''')

    # Add all the subfolders    
    topLevelItems = os.listdir(topFolder)
    for item in topLevelItems:
        # Skip everything except for the date sub folders
        dateFolder = os.path.join(topFolder, item)
        if not os.path.isdir(dateFolder):
            continue

        # Get the files in the date folder            
        dateString = item
        dateFiles = os.listdir(dateFolder)
        for d in dateFiles:
            # Skip everything except for kml files
            if '.kml' not in d:
                continue
                
            # Add a link to this file to the output kml file
            parts  = d.split('_')
            place  = parts[1]
            name   = dateString +'_'+ place
            string = '''<NetworkLink>
	<name>'''+name+'''</name>
	<Link><href>'''+dateString+'/'+d+'''</href></Link>
</NetworkLink>'''
            f.write(string)
    
    # Clean up
    f.write('''</Document>
</kml>''')
    f.close()
    return outputPath
       

# TODO: We are not currently using these functions!

def fetchArchivedDateList():
    '''Fetches the list of dates we have archived data for.'''
    
    dateList = []
    parsedIndexPage = BeautifulSoup(urllib2.urlopen((STORAGE_URL)).read(), 'html.parser')
    
    print parsedIndexPage
    
    for line in parsedIndexPage.findAll('a'):
        print line
        dateList.append(line.string)
        
    return dateList

def getKmlUrlsForDate(dateString):
    '''Fetches all the kml files archived for a given date'''

    kmlList = []
    subUrl = STORAGE_URL + dateString
    
    parsedSubPage = BeautifulSoup(urllib2.urlopen((subUrl)).read(), 'html.parser')
      
    print parsedSubPage
      
    for line in parsedSubPage.findAll('a'):
        kmlList.append(os.path.join(subUrl, line.string))

    return kmlList


def monthToNum(shortMonth):
    '''Convert abbreviated month name to integer'''
    return{
            'Jan' : 1,
            'Feb' : 2,
            'Mar' : 3,
            'Apr' : 4,
            'May' : 5,
            'Jun' : 6,
            'Jul' : 7,
            'Aug' : 8,
            'Sep' : 9, 
            'Oct' : 10,
            'Nov' : 11,
            'Dec' : 12
    }[shortMonth]


def grabGdacsResults(startDate, endDate):
    '''Get a list of flood locations from GDACS.
       Dates are in the format YYYY-MM-DD'''

    BASE_URL = "http://www.gdacs.org/rss.gdacs.aspx?profile=ARCHIVE&alertlevel=&country=&eventtype=FL"
       
    startDateString = ('%d-%02d-%02d' % (startDate.year, startDate.month, startDate.day)) 
    endDateString   = ('%d-%02d-%02d' % (endDate.year,   endDate.month,   endDate.day  )) 
    dateString = '&from='+startDateString+'&to='+endDateString
    print dateString
    #url = BASE_URL + dateString
   
    # GDACS turned off their nice flood interface, so we are restricted to the last seven days!
    url = 'http://www.gdacs.org/xml/rss_fl_7d.xml'
    
    print 'Looking up GDACS url: ' + url
    rawPage    = urllib2.urlopen((url)).read()
    print 'Done fetching page'
    parsedPage = BeautifulSoup(rawPage, 'html.parser')    
    print 'Done parsing page'

    centers = []
    for line in parsedPage.findAll('georss:point'):
        coord = line.string.split()
        coord.reverse()
        centers.append([float(x) for x in coord]) # Grabs lat, lon
        
    labels  = []
    for line in parsedPage.findAll('gdacs:country'):
        labels.append(line.string)

    toDates  = []
    for line in parsedPage.findAll('gdacs:todate'):
        parts = line.string.split() 
        year  = int(parts[3])
        month = monthToNum(parts[2])
        day   = int(parts[1])
        date  = datetime.datetime(year, month, day)
        toDates.append(date)

    if not centers: # Quit early if no data found
        return ([], [])

    if len(labels) != len(centers):
        raise Exception('RSS parsing code failed to properly extract names!')

    # Filter out dates older than specified
    kept = []
    for x in zip(centers, labels, toDates):
        if (x[2] >= startDate) and (x[2] <=endDate):
            kept.append((x[0],x[1]))
    if not kept: # Quit early if no data found
        return ([], [])
    centers, labels = zip(*kept)
    print centers, labels

    # Ensure that none of the labels are duplicates
    for i in range(0,len(labels)):
        label = labels[i]
        count = labels.count(label)
        if count == 1: # No duplicates
            continue
            
        # Append a number to the end of matching labels
        index = 0
        for j in range(i,len(labels)):
            if labels[j] == label:
                labels[j] += str(index)
                index += 1

    labels = [x.replace(' ', '_') for x in labels] # Don't allow spaces

    return (centers, labels)

def getSearchRegions(date, daySpan, regionSize):
    '''Gets the search regions for today.
       Each region is (minLon, minLat, maxLon, maxLat)'''

    timeDelta = datetime.timedelta(days=daySpan)
    startDate = date - timeDelta
   
    (centers, labels) = grabGdacsResults(startDate, date)

    regions = []
    for coord in centers:
        region = (coord[0]-regionSize, coord[1]-regionSize,
                  coord[0]+regionSize, coord[1]+regionSize)
        regions.append(region)
    
    return (regions, labels)




# --------------------------------------------------------------
def main(argsIn):

    #logger = logging.getLogger() TODO: Switch to using a logger!

    try:
          usage = "usage: daily_detector.py [--help]\n  "
          parser = optparse.OptionParser(usage=usage)

          parser.add_option("--archive-results", dest="archiveResults", action="store_true", default=False,
                            help="Archive results so they can be found by the web API.")
          parser.add_option("--manual", dest="showManual", action="store_true", default=False,
                            help="Display more usage information about the tool.")
          (options, args) = parser.parse_args(argsIn)

          if options.showManual:
              print manual
              return 0

    except optparse.OptionError, msg:
        raise Usage(msg)

    print '---=== Starting daily flood detection process ===---'

    date = datetime.datetime.now()
    dateString = ('%d-%02d-%02d' % (date.year, date.month, date.day))

    # Store outputs here before they are archived
    BASE_OUTPUT_FOLDER = '/home/smcmich1/data/Floods/auto_detect'

    # Look at flood alerts this many days old
    DAY_SPAN = 7

    # Search this far around the center point in degrees
    # - If it is too large Earth Engine will time out during processing!
    REGION_SIZE = 0.5

    # How many days around each flood alert to look for images
    MAX_SEARCH_DAYS      = '7'
    MAX_CLOUD_PERCENTAGE = '0.50'
    RECORD_INPUTS        = False # Save the processing inputs?
        
    # Get a list of search regions for today
    (searchRegions, labels) = getSearchRegions(date, DAY_SPAN, REGION_SIZE)
                      
    print 'Detected ' + str(len(searchRegions)) + ' candidate flood regions.'

    dateFolder = os.path.join(BASE_OUTPUT_FOLDER, dateString)
    if not os.path.exists(dateFolder):
        os.mkdir(dateFolder)

    for (region, label) in zip(searchRegions, labels):
        print '---------------------------------------------'
        
        #if label in ['Sudan']: #DEBUG
        #    continue
        
        centerPoint = ( (region[0] + region[2])/2.0, 
                        (region[1] + region[3])/2.0 )
        
        print 'Detecting floods in '+label+': ' + str(region)
        
        outputFolder = os.path.join(dateFolder, label)
        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
        #try:
        
        # Run this command as a subprocess so we can capture all the output for a log file
        cmd = ['python', os.path.join(os.path.dirname(os.path.realpath(__file__)),'detect_flood_cmd.py'), 
               '--search-days', MAX_SEARCH_DAYS, 
               '--max-cloud-percentage', MAX_CLOUD_PERCENTAGE]
              # '--',
        if RECORD_INPUTS:
            cmd.append('--save-inputs')
        cmd += ['--', outputFolder, dateString, 
               str(region[0]), str(region[1]), 
               str(region[2]), str(region[3])]
        print ' '.join(cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        textOutput, err = p.communicate()
        print textOutput
       
        # Log the program output to disk
        logPath = os.path.join(outputFolder, 'log.txt')
        with open(logPath, 'w') as f:
            f.write(str(cmd))
            f.write('\n============================================\n')
            f.write(textOutput)
        
                        
        # Check if we successfully generated a kml output file
        kmlPath = os.path.join(outputFolder, 'floodCoords.kml')
        if not os.path.exists(kmlPath):
            #raise Exception('DEBUG')
            continue
        
        # Read the sensors we used from the file and add them to the title
        floodInfo = None
        with open(kmlPath) as f:
            for line in f:
                if '<description>' in line:
                    floodInfo = detect_flood_cmd.parseKmlDescription(line)
                    break
        if not floodInfo:
            raise Exception('Failed to load flood information!')

        pairs = [('modis', 'M'), ('landsat', 'L'), ('sentinel-1', 'S')]
        sensorCode = '' # Will be something like "MLS"
        # Look for fields in floodInfo indicating the presence of each sensor.
        for pair in pairs:
            for s in floodInfo.keys():
                if pair[0] in s:
                    sensorCode += pair[1]
                    break
        
        # Insert the center into the kml file name    
        newKmlName = (('results_%s_%s_%05f_%05f.kml') % (label, sensorCode, centerPoint[0], centerPoint[1]))
        newKmlPath = os.path.join(outputFolder, newKmlName)
        shutil.move(kmlPath, newKmlPath)

        if options.archiveResults:
           archiveResult(newKmlPath, dateString)
        
        #raise Exception('DEBUG')
        
        #except Exception as e:
        #    print 'Failure!'
        #    print str(e)
        #    print sys.exc_info()[0]
        #    pass

    if options.archiveResults:
        print 'Finalizing archive folder...'
        updateArchiveFolder()
    print 'Done!'

# Call main() when run from command line
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))








    
