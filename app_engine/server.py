# Libraries built in to Google Apps Engine
import sys
import os
import cgi
import webapp2
import urllib2
import ee
import config
import json

# Libraries that we need to provide ourselves in the libs folder

rootdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(rootdir, 'libs'))

from bs4 import BeautifulSoup




# TODO: Move this!
STORAGE_URL = 'http://byss.arc.nasa.gov/smcmich1/cmt_detections/'    

feed_url = STORAGE_URL + 'daily_flood_detect_feed.kml'

# Go ahead and load the HTML files for later use.

with open('index.html', 'r') as f:
    PAGE_HTML = f.read()
with open('map.html', 'r') as f:
    MAP_HTML = f.read()



def renderHtml(html, pairList):
    '''Simple alternative to html template rendering software'''

    for pair in pairList:
        html = html.replace(pair[0], pair[1])
    return html



def fetchDateList(datesOnly=False):
    '''Fetches the list of available dates'''
    
    dateList = []
    parsedIndexPage = BeautifulSoup(urllib2.urlopen(STORAGE_URL).read(), 'html.parser')
    
    for line in parsedIndexPage.findAll('a'):
        dateString = line.string
        
        if datesOnly:
            dateList.append(dateString)
            continue

        # Else look through each page so we can make date__location pairs.
        subUrl = STORAGE_URL + dateString
        
        try:
            parsedSubPage = BeautifulSoup(urllib2.urlopen((subUrl)).read(), 'html.parser')
            
            for line in parsedSubPage.findAll('a'):
                kmlName = line.string
                info = extractInfoFromKmlUrl(kmlName)
                
                # Store combined date/location string.
                displayString = dateString +'__'+ info['location']
                dateList.append(displayString)
        except:
            pass # Ignore pages that we fail to parse

    return dateList

def getKmlUrlsForKey(key):
    '''Fetches all the kml files from a given date.
       If the dateString includes a location, only fetch the matching URL.
       Otherwise return all URLs for that date.'''

    # The key contains the date and optionally the location
    if '__' in key:
        parts = key.split('__')
        dateString = parts[0]
        location   = parts[1]
    else:
        dateString = key
        location   = None

    kmlList = []
    subUrl = STORAGE_URL + dateString
    parsedSubPage = BeautifulSoup(urllib2.urlopen((subUrl)).read(), 'html.parser')
      
    for line in parsedSubPage.findAll('a'):
        kmlName = line.string
        fullUrl = os.path.join(subUrl, kmlName)
        
        # If this location matches a provided location, just return this URL.
        if location and (location in kmlName):
            return [fullUrl]
        else:
            kmlList.append(fullUrl)

    return kmlList


def extractInfoFromKmlUrl(url):
    '''Extract the information encoded in the KML filename into a dictionary.'''
    
    # Format is: 'STUFF/results_location_SENSORS_%05f_%05f.kml'

    # Get just the kml name    
    rslash = url.rfind('/')
    if rslash:
        filename = url[rslash+1:]
    else:
        filename = url

    # Split up the kml name
    parts     = filename.split('_')
    parts[-1] = parts[-1].replace('.kml','')

    location = parts[1]
    if len(parts) == 5:
        sensors = parts[2]
    else:
        sensors = ''
    lon = float(parts[-2])
    lat = float(parts[-1])
    
    # Pack the results into a dictionary
    return {'location':location, 'sensors':sensors, 'lon':lon, 'lat':lat}

def fetchKmlDescription(url):
    '''Read the description field from the kml file'''
    
    ## If any of these fields are not found, replace with a placeholder.
    #EXPECTED_FIELDS = ['modis_id', 'landsat_id', 'sentinel1_id']
    #EMPTY_FIELD_TAG = 'None'

    # Try to read in the description string    
    kmlText = urllib2.urlopen(url).read()
    parsedFile = BeautifulSoup(kmlText)
    for line in parsedFile.findAll('description'):
        text = line.string
    
    # Parse the JSON data if it exists
    try:
        info = json.loads(text)
    except:
        info = dict()
    
    ## Fill in missing fields
    #for f in EXPECTED_FIELDS:
    #    if not (f in info):
    #        info[f] = EMPTY_FIELD_TAG
            
    return info

def expandSensorsList(sensors):
    '''Expand the abbreviated sensor list to full sensor names'''
    
    string = ''
    pairs = [('Modis', 'M'), ('Landsat', 'L'), ('Sentinel-1', 'S')]
    for pair in pairs:
        if pair[1] in sensors:
            string += (' ' + pair[0])
    if not string:
        string = 'Error: Sensor list "'+sensors+'" not parsed!'
    return string


def getLayerInfo(kmlInfo):
    '''Given the parsed KML description object, set up EE layer info'''
    
    # The information is already in an easy to use format
    # TODO: Refine the display parameters?
    
    layers = []
    if 'modis_image_id_A' in kmlInfo:
        modisA = ee.Image(kmlInfo['modis_image_id_A'])
        modisQ = ee.Image(kmlInfo['modis_image_id_Q'])
        modis = modisQ.addBands(modisA, ['sur_refl_b06'])
        modis_visualization = modis.getMapId({
            'min': 0,
            'max': 3000,
            'bands': 'sur_refl_b01, sur_refl_b02, sur_refl_b06'
        })
        layers.append({
            'mapid': modis_visualization['mapid'],
            'label': 'modis',
            'token': modis_visualization['token']
        })
    if 'landsat_image_id' in kmlInfo:
        landsat = ee.Image(kmlInfo['landsat_image_id'])
        # Pick the correct bands for this satellite
        bands = 'B3, B2, B1'
        if 'LC8' in kmlInfo['landsat_image_id']:
            bands = 'B4, B3, B2'
        
        landsat_visualization = landsat.getMapId({
            'min': 0,
            'max': 0.75,
            'bands': bands
        })
        layers.append({
            'mapid': landsat_visualization['mapid'],
            'label': 'landsat',
            'token': landsat_visualization['token']
        })
    if 'sentinel1_image_id' in kmlInfo:
        sentinel1 = ee.Image(kmlInfo['sentinel1_image_id'])
        sentinel1_visualization = sentinel1.getMapId({
            'min': -30,
            'max': 5,
            'bands': sentinel1.bandNames().getInfo()[0]
        })
        layers.append({
            'mapid': sentinel1_visualization['mapid'],
            'label': 'sentinel1',
            'token': sentinel1_visualization['token']
        })
    return layers


class GetMapData(webapp2.RequestHandler):
    """Retrieves EE data on request."""

    def get(self):

        ee.Initialize(config.EE_CREDENTIALS)
        
        layers = [] # We will fill this up with EE layer information

        # Use the MCD12 land-cover as training data.
        modis_landcover = ee.Image('MCD12Q1/MCD12Q1_005_2001_01_01').select('Land_Cover_Type_1')

        # A palette to use for visualizing landcover images.
        modis_landcover_palette = ','.join([
            'aec3d4',  # water
            '152106', '225129', '369b47', '30eb5b', '387242',  # forest
            '6a2325', 'c3aa69', 'b76031', 'd9903d', '91af40',  # shrub, grass and
                                                               # savanah
            '111149',  # wetlands
            '8dc33b',  # croplands
            'cc0013',  # urban
            '6ca80d',  # crop mosaic
            'd7cdcc',  # snow and ice
            'f7e084',  # barren
            '6f6f6f'   # tundra
        ])

        # A set of visualization parameters using the landcover palette.
        modis_landcover_visualization_options = {
            'palette': modis_landcover_palette,
            'min': 0,
            'max': 17,
            'format': 'png'
        }

        # Add the MODIS landcover image.
        modis_landcover_visualization = modis_landcover.getMapId(modis_landcover_visualization_options)
        layers.append({
            'mapid': modis_landcover_visualization['mapid'],
            'label': 'MODIS landcover',
            'token': modis_landcover_visualization['token']
        })

        # Add the Landsat composite, visualizing just the [30, 20, 10] bands.
        landsat_composite = ee.Image('L7_TOA_1YEAR_2000')
        landsat_composite_visualization = landsat_composite.getMapId({
            'min': 0,
            'max': 100,
            'bands': ','.join(['30', '20', '10'])
        })
        layers.append({
            'mapid': landsat_composite_visualization['mapid'],
            'label': 'Landsat composite',
            'token': landsat_composite_visualization['token']
        })

        text = json.dumps(layers)
        print text
        self.response.out.write(text)



class MainPage(webapp2.RequestHandler):
    '''The splash page that the user sees when they access the site'''

    def get(self):

        # Grab all dates where data is available
        self._dateList = fetchDateList()

        # Build the list of date options
        optionText = ''
        for dateString in self._dateList:
            optionText += '<option>'+dateString.replace('_',' ')+'</option>'
        
        # Insert the option section, leave the output section empty.
        self._htmlText = renderHtml(PAGE_HTML, [('[OPTION_SECTION]', optionText), 
                                                ('[OUTPUT_SECTION]', ''),
                                                ('[FEED_URL]', feed_url)])
        
        # Write the output    
        self.response.write(self._htmlText)


MAP_MODIS_RADIO_SNIPPET     = '<input type="radio" name="image" value="modis"          > MODIS<br>'
MAP_LANDSAT_RADIO_SNIPPET   = '<input type="radio" name="image" value="landsat"        > Landsat<br>'
MAP_SENTINEL1_RADIO_SNIPPET = '<input type="radio" name="image" value="sentinel1"      > Sentinel-1<br>'

class MapPage(webapp2.RequestHandler):
    '''Similar to the main page, but with a map displayed.'''

    def post(self):

        # Init demo ee image
        ee.Initialize(config.EE_CREDENTIALS)
        #mapid = ee.Image('srtm90_v4').getMapId({'min': 0, 'max': 1000})

        # Grab all dates where data is available
        self._dateList = fetchDateList()
        
        # Build the list of date options
        optionText = ''
        for dateString in self._dateList:
            optionText += '<option>'+dateString.replace('_',' ')+'</option>'

        # Insert the options section
        self._htmlText = renderHtml(PAGE_HTML, [('[OPTION_SECTION]', optionText),
                          ('[API_KEY]', 'AIzaSyAlcB6oaJeUdTz3I97cL47tFLIQfSu4j58'),
                          ('[FEED_URL]', feed_url)])

        # Fetch user selection    
        dateLocString = self.request.get('date_select', 'default_date!')

        # This should only return one URL, provided that the location is included in dateLocString
        try:
            kmlUrls = getKmlUrlsForKey(dateLocString.replace(' ', '__'))
        except:
            kmlUrls = None
        
        if not kmlUrls:
            #newText = 'No KML files were found for this date!'
            newText = dateLocString 
        else:
            # Prepare the map HTML with the data we found
            kmlUrl       = kmlUrls[0]
            kmlUrlInfo   = extractInfoFromKmlUrl(kmlUrl) # TODO: Clean this up!
            detailedInfo = fetchKmlDescription(kmlUrl) # TODO: Get all info from here!
            layerInfo    = getLayerInfo(detailedInfo)
            sensorList   = expandSensorsList(kmlUrlInfo['sensors'])
            
            (modisRadioText, landsatRadioText, sentinel1RadioText) = ('', '', '')
            if 'Modis' in sensorList:
                modisRadioText     = MAP_MODIS_RADIO_SNIPPET
            if 'Landsat' in sensorList:
                landsatRadioText   = MAP_LANDSAT_RADIO_SNIPPET
            if 'Sentinel-1' in sensorList:
                sentinel1RadioText = MAP_SENTINEL1_RADIO_SNIPPET
            
            detailedInfo['layers'] = layerInfo
            #raise Exception(json.dumps(detailedInfo))
            newText = renderHtml(MAP_HTML, [#('[EE_MAPID]',    mapid['mapid']),
                                            #('[EE_TOKEN]',    mapid['token']),
                                            ('[API_KEY]', 'AIzaSyAlcB6oaJeUdTz3I97cL47tFLIQfSu4j58'),
                                            ('[MAP_TITLE]',   dateLocString),
                                            ('[KML_URL]',     kmlUrl), 
                                            #('[MODIS_ID]',    detailedInfo['modis_id']),
                                            ('[RADIO_SECTION_MODIS]',     modisRadioText),
                                            ('[RADIO_SECTION_LANDSAT]',   landsatRadioText),
                                            ('[RADIO_SECTION_SENTINEL1]', sentinel1RadioText),
                                            ('[MAP_JSON_TEXT]', json.dumps(detailedInfo)),
                                            ('[SENSOR_LIST]', sensorList), 
                                            ('[LAT]',         str(kmlUrlInfo['lat'])), 
                                            ('[LON]',         str(kmlUrlInfo['lon']))
                                           ])

        #newText = 'You selected: <pre>'+ cgi.escape(date) +'</pre>'
        #newText = MAP_HTML
        
        # Fill in the output section
        text = renderHtml(self._htmlText, [('[OUTPUT_SECTION]', newText)])
        
        # Write the output
        self.response.write(text)

app = webapp2.WSGIApplication([
    ('/',         MainPage),
    ('/selected', MapPage),
    ('/getmapdata', GetMapData)
], debug=True)



