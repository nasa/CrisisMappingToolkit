
import os, sys, optparse
from bs4 import BeautifulSoup
import urllib2
import datetime
import shutil
import subprocess

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

    if len(labels) != len(centers):
        raise Exception('RSS parsing code failed to properly extract names!')

    # Filter out dates older than specified
    kept = []
    for x in zip(centers, labels, toDates):
        if (x[2] >= startDate) and (x[2] <=endDate):
            kept.append((x[0],x[1]))
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

def getSearchRegions(date):
    '''Gets the search regions for today.
       Each region is (minLon, minLat, maxLon, maxLat)'''

    # Look at flood alerts this many days old
    DAY_SPAN = 10

    # Search this far around the center point in degrees
    REGION_SIZE = 0.5

    timeDelta = datetime.timedelta(days=DAY_SPAN)
    startDate = date - timeDelta
   
    (centers, labels) = grabGdacsResults(startDate, date)

    regions = []
    for coord in centers:
        region = (coord[0]-REGION_SIZE, coord[1]-REGION_SIZE,
                  coord[0]+REGION_SIZE, coord[1]+REGION_SIZE)
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
          
          (options, args) = parser.parse_args(argsIn)

    except optparse.OptionError, msg:
        raise Usage(msg)

    print '---=== Starting daily flood detection process ===---'

    date = datetime.datetime.now()
    #date = datetime.date(2016, 6, 1) # DEBUG date!
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
        
        if label in ['Sudan']: #DEBUG
            continue
        
        centerPoint = ( (region[0] + region[2])/2.0, 
                        (region[1] + region[3])/2.0 )
        
        print 'Detecting floods in '+label+': ' + str(region)
        
        outputFolder = os.path.join(dateFolder, label)
        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
        #try:
        
        # Run this command as a subprocess so we can capture all the output for a log file
        cmd = ['python', os.path.join(os.path.dirname(os.path.realpath(__file__)),'detect_flood_cmd.py'), 
               '--search-days', '2', 
               '--max-cloud-percentage', '0.20', 
               #'--save-inputs', 
               '--',
               outputFolder, dateString, 
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
        # Look for fields in floodInfo indicating the precense of each sensor.
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


# Call main() when run from command line
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))








    
