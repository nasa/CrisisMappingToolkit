

// Basic options for the Google Map.
var mapOptions = {
  center: new google.maps.LatLng(0.0, 0.0),
  zoom: 8,
  streetViewControl: false
};

// Create the base Google Map, set up a drawing manager and listen for updates
var map = new google.maps.Map(document.getElementById('map'), mapOptions);

// Initialize the Google Map and add our custom layer overlay.
var kmlLayer;
var addKml = function(kmlUrl) {

  kmlLayer = new google.maps.KmlLayer({
                   url: kmlUrl,
                   map: map
                 });
};

var imageNames = []

var loadMapImages2 = function(modisId) {
  // At the start, load up the image data for the map.
  $.getJSON(           // Fetch JSON data from the server
      '/getmapdata',   // Request it from this URL
      {},              // Data passed to the server handler
      function(data) { // Call this function with the data we get back
        // Clear out any old layers.
        map.overlayMapTypes.clear();
        //$('#layers').empty();

        data.forEach(function(layer, i) {
          // Configuration for the image map type. The Google Maps API calls
          // getTileUrl when it tries to display a map tile. Our method will
          // provide a valid URL to an Earth Engine map tile based on the mapid and token.
          var eeMapOptions = {
            getTileUrl: buildGetTileUrl(layer.mapid, layer.token),
            tileSize: new google.maps.Size(256, 256)
          };

          // Create the map type.
          var mapType = new google.maps.ImageMapType(eeMapOptions);

          // Add the EE layer to the map.
          map.overlayMapTypes.push(mapType);
          // Record the name for checkbox interaction later
          imageNames.push(layer.label);
          // Default all layers to hidden.
          map.overlayMapTypes.getAt(i).setOpacity(0);
          
        }); // end response function call
      }); // end getJSON function call
};



var loadMapImages = function(jsonText) {

  var data = JSON.parse(jsonText)
  layers = data['layers']

  // Clear out any old layers.
  map.overlayMapTypes.clear();

  layers.forEach(function(layer, i) {
    // Configuration for the image map type. The Google Maps API calls
    // getTileUrl when it tries to display a map tile. Our method will
    // provide a valid URL to an Earth Engine map tile based on the mapid and token.
    var eeMapOptions = {
      getTileUrl: buildGetTileUrl(layer['mapid'], layer['token']),
      tileSize: new google.maps.Size(256, 256)
    };

    // Create the map type.
    var mapType = new google.maps.ImageMapType(eeMapOptions);

    // Add the EE layer to the map.
    map.overlayMapTypes.push(mapType);
    // Record the name for checkbox interaction later
    imageNames.push(layer['label']);
    // Default all layers to hidden.
    map.overlayMapTypes.getAt(i).setOpacity(0);
    
  }); // end response function call
};


// Function to handle when the radio selector is changed
$('input:radio[name="image"]').change(
  function(){
    // Get the selected name, then turn on the map overlay with the same name.
    // - This requires that the names in HTML and python be synchronized!
    if ($(this).is(':checked')) {
      var checkedName = this.value
      
      map.overlayMapTypes.forEach(function(overlay, i) {
        if (imageNames[i] == checkedName) {
          overlay.setOpacity(100);
        } else {
          overlay.setOpacity(0);
        }
      }); // end loop through map layers
    } // end if
});


// Function to handle when the kml checkbox is changed
$('input:checkbox[name="kmlButton"]').change(
  function(){
    // Get the selected name, then turn on the map overlay with the same name.
    // - This requires that the names in HTML and python be synchronized!
    if ($(this).is(':checked')) {
      kmlLayer.setMap(map);
    } else {
      kmlLayer.setMap(null);
    }
});



// Returns a function that builds a valid tile URL to Earth Engine based on
// the mapid and token.
function buildGetTileUrl(mapid, token) {
  return function(tile, zoom) {
    var baseUrl = 'https://earthengine.googleapis.com/map';
    var url = [baseUrl, mapid, zoom, tile.x, tile.y].join('/');
    url += '?token=' + token;
    return url;
  };
}; // end change?


    
