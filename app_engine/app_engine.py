# Libraries built in to Google Apps Engine
import sys
import os
import cgi
import webapp2
import urllib2

# Libraries that we need to provide ourselves in the libs folder

rootdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(rootdir, 'libs'))

from bs4 import BeautifulSoup




# TODO: Move this!
STORAGE_URL = 'http://byss.arc.nasa.gov/smcmich1/cmt_detections/'    

feed_url = STORAGE_URL + 'daily_flood_detect_feed.kml'

# This is the HTML used to draw each of the web pages.
# - Normally this would be use with web template rendering software 
#   like Django or Jinja but we are keeping the complexity down.
PAGE_HTML = """\
<html>

  <head>
    <style>
      #map {
        width: 500px;
        height: 400px;
      }
    </style>
  </head>

  <body>
  
    [OUTPUT_SECTION]
  
    <form action="/selected" method="post">
      <select name="date_select">
        [OPTION_SECTION]
      </select>
      <div><input type="submit" value="Fetch Map"></div>
    </form>
    <a href='"""+feed_url+"""'>Daily feed kml file</a>
    
  </body>
</html>
"""

# DEBUG url
# center: {lat: 41.876, lng: -87.624},
# url: 'http://googlemaps.github.io/js-v2-samples/ggeoxml/cta.kml',

# DEBUG url from us
# center: {lat: 28.6, lng: 70.0},
#           url: 'http://byss.arc.nasa.gov/smcmich1/cmt_detections/2010-08-13detections-2010-08-13-kashmore.kml',


MAP_HTML = """\
    <div id="map"></div>

    <table width="60" style="border: 1px solid #000" rules="all">
    <tr>

    <td>
    <strong>[MAP_TITLE]</strong>
    <br>
    Sensors: [SENSOR_LIST]
    </td>

    <td>
    <table width="60" style="border: 1px solid #000" rules="all">
    <tr><td>Water</td><td>Clouds</td></tr>
    <tr height="50" ><td bgcolor="00ffff"></td><td bgcolor="ffffff"></td></tr>
    </table>
    <a href='[KML_URL]'>Download this kml file</a>
    </td>

    </tr>

    <script>
      function initMap() {
        var mapDiv = document.getElementById('map');
        var map = new google.maps.Map(mapDiv, {
          center: {lat: [LAT], lng: [LON]},
          zoom: 11
        });
      var ctaLayer = new google.maps.KmlLayer({
          url: '[KML_URL]',
          map: map
        });
      }
      
    </script>
    <script src="https://maps.googleapis.com/maps/api/js?callback=initMap"
        async defer></script>
    <br><br>
"""

def renderHtml(html, pairList):
    '''Simple alternative to html template rendering software'''

    for pair in pairList:
        html = html.replace(pair[0], pair[1])
    return html



def fetchDateList(datesOnly=False):
    '''Fetches the list of available dates'''
    
    dateList = []
    parsedIndexPage = BeautifulSoup(urllib2.urlopen((STORAGE_URL)).read(), 'html.parser')
    
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
                                                ('[OUTPUT_SECTION]', '')])
        
        # Write the output    
        self.response.write(self._htmlText)


class MapPage(webapp2.RequestHandler):
    '''Similar to the main page, but with a map displayed.'''

    def post(self):

        # Grab all dates where data is available
        self._dateList = fetchDateList()
        
        # Build the list of date options
        optionText = ''
        for dateString in self._dateList:
            optionText += '<option>'+dateString.replace('_',' ')+'</option>'

        # Insert the options section
        self._htmlText = renderHtml(PAGE_HTML, [('[OPTION_SECTION]', optionText)])

        # Fetch user selection    
        dateLocString = self.request.get('date_select', 'default_date!')

        ## This should only return one URL, provided that the location is included in dateLocString
        try:
            kmlUrls = getKmlUrlsForKey(dateLocString.replace(' ', '__'))
        except:
            kmlUrls = None
        
        if not kmlUrls:
            #newText = 'No KML files were found for this date!'
            newText = dateLocString 
        else:
            # Prepare the map HTML with the data we found
            kmlUrl     = kmlUrls[0]
            info       = extractInfoFromKmlUrl(kmlUrl)
            sensorList = expandSensorsList(info['sensors'])
            newText = renderHtml(MAP_HTML, [('[MAP_TITLE]',   dateLocString),
                                            ('[KML_URL]',     kmlUrl), 
                                            ('[SENSOR_LIST]', sensorList), 
                                            ('[LAT]',         str(info['lat'])), 
                                            ('[LON]',         str(info['lon']))])

        #newText = 'You selected: <pre>'+ cgi.escape(date) +'</pre>'
        #newText = MAP_HTML
        
        # Fill in the output section
        text = renderHtml(self._htmlText, [('[OUTPUT_SECTION]', newText)])
        
        # Write the output
        self.response.write(text)

app = webapp2.WSGIApplication([
    ('/',         MainPage),
    ('/selected', MapPage),
], debug=True)



