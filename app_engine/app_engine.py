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
"""



# TODO: Store the dates in a better manner!
def fetchDateList():
    '''Fetches the list of available dates'''
    
    dateList = []
    parsedIndexPage = BeautifulSoup(urllib2.urlopen((STORAGE_URL)).read(), 'html.parser')
    
    for line in parsedIndexPage.findAll('a'):
        dateList.append(line.string)
        
    return dateList

def getKmlUrlsForDate(dateString):
    '''Fetches all the kml files from a given date'''

    kmlList = []
    subUrl = STORAGE_URL + dateString
    parsedSubPage = BeautifulSoup(urllib2.urlopen((subUrl)).read(), 'html.parser')
      
    for line in parsedSubPage.findAll('a'):
        kmlList.append(os.path.join(subUrl, line.string))

    return kmlList

def extractCenterFromKmlUrl(filename):
    '''Extract the lon and lat values encoded in the KML filename'''
    
    # Format is: 'STUFF/results_%05f_%05f.kml'
    
    end = filename.rfind('.kml')
    b2  = filename.rfind('_', end)
    b1  = filename.rfind('_', b2)
    
    lon = float(filename[b1+1:b2 ])
    lat = float(filename[b2+1:end])
    
    return (lon, lat)

class MainPage(webapp2.RequestHandler):
    '''The splash page that the user sees when they access the site'''

    def get(self):

        # Grab all dates where data is available
        self._dateList = fetchDateList()

        # Build the list of date options
        optionText = ''
        for dateString in self._dateList:
            optionText += '<option>'+dateString+'</option>'
            
        self._htmlText = PAGE_HTML.replace('[OPTION_SECTION]', optionText).replace('[OUTPUT_SECTION]', '')
        
        # Write the output    
        self.response.write(self._htmlText)


class MapPage(webapp2.RequestHandler):

    def post(self):

        # Grab all dates where data is available
        self._dateList = fetchDateList()
        
        
        # TODO: Only do this once!
        # Build the list of date options
        optionText = ''
        for dateString in self._dateList:
            optionText += '<option>'+dateString+'</option>'
            
        self._htmlText = PAGE_HTML.replace('[OPTION_SECTION]', optionText)


        # Fetch user selection    
        dateString = self.request.get('date_select', 'default_date!')

        kmlUrls = getKmlUrlsForDate(dateString)
        
        if not kmlUrls:
            newText = 'No KML files were found for this date!'
        else:
            kmlUrl = kmlUrls[0]
            (lat, lon) = extractCenterFromKmlUrl(kmlUrl)
            newText = MAP_HTML.replace('[KML_URL]', kmlUrl).replace('[LAT]',lat).replace('[LON]',lon)

        #newText = 'You selected: <pre>'+ cgi.escape(date) +'</pre>'
        #newText = MAP_HTML
        text = self._htmlText.replace('[OUTPUT_SECTION]', newText)
        self.response.write(text)

app = webapp2.WSGIApplication([
    ('/',         MainPage),
    ('/selected', MapPage),
], debug=True)



