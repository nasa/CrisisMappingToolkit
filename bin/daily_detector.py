
import os, sys
from bs4 import BeautifulSoup
import urllib2
import datetime
import shutil

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import detect_flood_cmd

'''
Script to process multiple regions across the globe to detect floods.
- Intended to be run daily with a cron task.
'''


STORAGE_URL = 'http://byss.arc.nasa.gov/smcmich1/cmt_detections/'

def archiveResult(kmlPath, dateString):
    '''Archives completed flood detection results'''

    # Currently they are stored in our web share on byss
    DEST_BASE = '/byss/docroot/smcmich1/cmt_detections/'

    dateFolder = os.path.join(DEST_BASE, dateString)
    cmd = 'mkdir '+dateFolder
    print cmd
    os.system(cmd)

    filename = os.path.split(kmlPath)[1]
    destinationPath = os.path.join(dateFolder, filename)
    cmd = 'cp ' +kmlPath+ ' '+destinationPath
    print cmd
    os.system(cmd)

    # Update the HTML indices
    cmd = 'python ~/makeHtmlIndex.py ' + DEST_BASE
    print cmd
    os.system(cmd)
    cmd = 'python ~/makeHtmlIndex.py ' + dateFolder
    print cmd
    os.system(cmd)


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


def grabGdacsResults(startDate, endDate):
    '''Get a list of flood locations from GDACS.
       Dates are in the format YYYY-MM-DD'''

    BASE_URL = "http://www.gdacs.org/rss.gdacs.aspx?profile=ARCHIVE&alertlevel=&country=&eventtype=FL"
    dateString = '&from='+startDate+'&to='+endDate
    
    url = BASE_URL + dateString
    print url
    parsedPage = BeautifulSoup(urllib2.urlopen((url)).read(), 'html.parser')    

    centers = []
    for line in parsedPage.findAll('georss:point'):
        coord = line.string.split()
        coord.reverse()
        centers.append([float(x) for x in coord]) # Grabs lat, lon
        
    labels  = []
    for line in parsedPage.findAll('gdacs:country'):
        labels.append(line.string)

    # TODO: Set up logging
    if len(labels) != len(centers):
        print 'RSS parsing code failed to properly extract names!'
        labels = ['' for i in centers]

    # Unsure that none of the labels are duplicates
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

def getSearchRegions(date):
    '''Gets the search regions for today.
       Each region is (minLon, minLat, maxLon, maxLat)'''

    # Look at flood alerts this many days old
    DAY_SPAN = 10

    # Search this far around the center point in degrees
    REGION_SIZE = 0.3

    timeDelta = datetime.timedelta(days=DAY_SPAN)
    startDate = date - timeDelta
    startDateString = ('%d-%02d-%02d' % (startDate.year, startDate.month, startDate.day)) 
    endDateString   = ('%d-%02d-%02d' % (date.year, date.month, date.day)) 
    
    (centers, labels) = grabGdacsResults(startDateString, endDateString)

    regions = []
    for coord in centers:
        region = (coord[0]-REGION_SIZE, coord[1]-REGION_SIZE,
                  coord[0]+REGION_SIZE, coord[1]+REGION_SIZE)
        regions.append(region)
    
    return (regions, labels)



if __name__ == "__main__":

    print '---=== Starting daily flood detection process ===---'

    date = datetime.datetime.now()
    dateString = ('%d-%02d-%02d' % (date.year, date.month, date.day))
        
    # Get a list of search regions for today
    (searchRegions, labels) = getSearchRegions(date)
    #searchRegions =  [(8.868, 41.291000000000004, 9.668000000000001, 42.091), 
    #                  (-11.954, 33.422000000000004, -11.154, 34.222), 
    #                  (8.766, 45.448, 9.566, 46.248), 
    #                  (-31.543, -58.623999999999995, -30.743000000000002, -57.824), 
    #                  (32.654, 71.37899999999999, 33.454, 72.179)]
                      
    print 'Detected ' + str(len(searchRegions)) + ' candidate flood regions.'

    # TODO: Launch multiple processes to detect these floods

    BASE_OUTPUT_FOLDER = '/home/smcmich1/data/Floods/auto_detect'

    dateFolder = os.path.join(BASE_OUTPUT_FOLDER, dateString)
    if not os.path.exists(dateFolder):
        os.mkdir(dateFolder)

    #searchRegions = [searchRegions[2]]
    #labels = ['debug']
    

    for (region, label) in zip(searchRegions, labels):
        print '---------------------------------------------'
        
        #if label in ['Rwanda', 'China', 'Dominican_Republic']: #DEBUG
        #    continue
        
        print 'Detecting floods in '+label+': ' + str(region)
        
        outputFolder = os.path.join(dateFolder, label)
        #try:
        detect_flood_cmd.main(['--search-days', '10', 
                               '--max-cloud-percentage', '0.20', 
                               #'--save-inputs', 
                               '--',
                               outputFolder, dateString, 
                               str(region[0]), str(region[1]), 
                               str(region[2]), str(region[3])])
        centerPoint = ( (region[0] + region[2])/2.0, 
                        (region[1] + region[3])/2.0 )
                        
        # Check if we successfully generated a kml output file
        kmlPath = os.path.join(outputFolder, 'floodCoords.kml')
        if not os.path.exists(kmlPath):
            continue
        
        # Read the sensors we used from the file and add them to the title
        with open(kmlPath) as f:
            for line in f:
                if '<description>' in line:
                    start = line.find('>')
                    end   = line.rfind('<')
                    s     = line[start+1:end]
                    sensorList = s.split()
                    break

        pairs = [('modis', 'M'), ('landsat', 'L'), ('sentinel-1', 'S')]
        sensorCode = '' # Will be something like "MLS"
        for pair in pairs:
            for s in sensorList:
                if pair[0] in s:
                    sensorCode += pair[1]
                    break
        
        # Insert the center into the kml file name    
        newKmlName = (('results_%s_%s_%05f_%05f.kml') % (label, sensorCode, centerPoint[0], centerPoint[1]))
        newKmlPath = os.path.join(outputFolder, newKmlName)
        shutil.move(kmlPath, newKmlPath)

        #raise Exception('DEBUG')

        archiveResult(newKmlPath, dateString)
        
        #raise Exception('DEBUG')
        
        #except Exception as e:
        #    print 'Failure!'
        #    print str(e)
        #    print sys.exc_info()[0]
        #    pass









    
