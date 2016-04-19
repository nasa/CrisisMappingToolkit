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
        <option>Milk</option>
        <option>Coffee</option>
        <option>Tea</option>
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
          center: {lat: 28.6, lng: 70.0},
          zoom: 11
        });
      var ctaLayer = new google.maps.KmlLayer({
          url: 'http://byss.arc.nasa.gov/smcmich1/cmt_detections/2010-08-13/detections-2010-08-13-kashmore.kml',
          map: map
        });
      }
      
    </script>
    <script src="https://maps.googleapis.com/maps/api/js?callback=initMap"
        async defer></script>
"""


class MainPage(webapp2.RequestHandler):
    '''The splash page that the user sees when they access the site'''

    def get(self):
        text = PAGE_HTML.replace('[OUTPUT_SECTION]', '')
        self.response.write(text)
        
    # TODO: Store the dates in a better manner!
    def fetchDateList():
        '''Fetches the list of available dates'''
        
        dateList = []
        parsedIndexPage = BeautifulSoup(urllib2.urlopen((STORAGE_URL)).read(), 'html.parser')
        
        for line in parsedIndexPage.findAll('a'):
            dateList.append(line.string)

    def getKmlsUrlsForDate(dateString):
        '''Fetches all the kml files from a given date'''

        kmlList = []
        subUrl = STORAGE_URL + dateString
        parsedSubPage = BeautifulSoup(urllib2.urlopen((subUrl)).read(), 'html.parser')
          
        for line in parsedSubPage.findAll('a'):
            kmlList.append(os.path.join(subUrl, line.string))


class MapPage(webapp2.RequestHandler):
    def post(self):
        date = self.request.get('date_select', 'default_date!')

        #newText = 'You selected: <pre>'+ cgi.escape(date) +'</pre>'
        newText = MAP_HTML
        text = PAGE_HTML.replace('[OUTPUT_SECTION]', newText)
        self.response.write(text)

app = webapp2.WSGIApplication([
    ('/',         MainPage),
    ('/selected', MapPage),
], debug=True)



