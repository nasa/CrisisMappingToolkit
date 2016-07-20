// Initialize the Google Map and add our custom layer overlay.
var initialize = function(mapId, token) {
  var eeMapOptions = {
    getTileUrl: function(tile, zoom) {
      var baseUrl = 'https://earthengine.googleapis.com/map';
      var url = [baseUrl, mapId, zoom, tile.x, tile.y].join('/');
      url += '?token=' + token;
      return url;
    },
    tileSize: new google.maps.Size(256, 256)
  };

  // Create the map type.
  var mapType = new google.maps.ImageMapType(eeMapOptions);

  var myLatLng = new google.maps.LatLng(-34.397, 150.644);
  var mapOptions = {
    center: myLatLng,
    zoom: 8,
    maxZoom: 10,
    streetViewControl: false
  };

  // Create the base Google Map.
  var map = new google.maps.Map(
      document.getElementById('map'), mapOptions);



  // Add the EE layer to the map.
  map.overlayMapTypes.push(mapType);
  
  //var ctaLayer = new google.maps.KmlLayer({
  //                 url: '[KML_URL]',
  //                 map: map
  //               });
};

    
