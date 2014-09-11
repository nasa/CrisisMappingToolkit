import ee

# radar data sources
RADARSAT  = 1
TERRASAR  = 2
UAVSAR    = 3
SENTINEL1 = 4

VISUALIZATION_MAXIMUMS = {
	RADARSAT  : 5000,
	TERRASAR  : 1000,
	UAVSAR    : 65000,
	SENTINEL1 : 1200 
}

# image ids
UAVSAR_MISSISSIPPI_FLOODED   = 1
UAVSAR_MISSISSIPPI_UNFLOODED = 2

class RadarImage(object):
	def __init__(self, instrument, image, bounds, ground_truth = None):
		self.instrument = instrument
		self.image = image
		self.bounds = bounds
		self.ground_truth = ground_truth
	
	def visualize(self, params = {}, name = 'Radar', show=True):
		all_bands = self.image.bandNames()
		bands = all_bands.getInfo()
		image = self.image
		if len(bands) == 2:
			image = image.addBands(0)
			bands.insert(0, 'constant')
		new_params = {'bands' : bands, 'min' : 0, 'max' : VISUALIZATION_MAXIMUMS[self.instrument]}
		new_params.update(params)
		return (image, new_params, name, show)

def get_radar_image(instrument, id=None):
	if instrument == RADARSAT:
		im_hh = ee.Image('18108519531116889794-06793893466375912303')
		im_hv = ee.Image('18108519531116889794-13933004153574033452')
		bounds = ee.geometry.Geometry.Rectangle(-123.60, 48.95, -122.75, 49.55)
		im = im_hh.select(['b1'], ['hh'])
		im = im.addBands(im_hv.select(['b1'], ['hv']))
	elif instrument == TERRASAR:
		im_hh = ee.Image('18108519531116889794-04996796288385000359')
		bounds = ee.geometry.Geometry.Rectangle(-79.64, 8.96, -79.55, 9.015)
		im = im_hh.select(['b1'], ['hh'])
	elif instrument == UAVSAR:
		if id == UAVSAR_MISSISSIPPI_UNFLOODED:
			im = ee.Image('18108519531116889794-16648596607414356603')
		else:
			im = ee.Image('18108519531116889794-12113912950916481117')
		bounds = ee.geometry.Geometry.Rectangle(-91.23, 32.88, -91.02, 33.166)
		im = im.select(['b1', 'b2', 'b3'], ['hh', 'hv', 'vv']).mask(im.select(['b4'], ['a']))
	elif instrument == SENTINEL1:
		im_vv = ee.Image('18108519531116889794-15063535589376921925')
		im_vv = im_vv.mask(im_vv)
		im_vh = ee.Image('18108519531116889794-10203767773364605611')
		im_vh = im_vv.mask(im_vh)
		bounds = ee.geometry.Geometry.Rectangle(12.0, 41.5, 14.0, 42.5)
		im = im_vv.select(['b1'], ['vv']).addBands(im_vh.select(['b1'], ['vh']))
	return RadarImage(instrument, im, bounds)

