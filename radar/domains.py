import ee

# radar data sources
RADARSAT   = 1
TERRASAR   = 2
UAVSAR     = 3
UAVSAR_LOG = 4
SENTINEL1  = 5

MAXIMUM_VALUES = {
    RADARSAT   : 5000,
    TERRASAR   : 1000,
    UAVSAR     : 65000,
    UAVSAR_LOG : 255,
    SENTINEL1  : 1200 
}

MINIMUM_VALUES = {
    UAVSAR : 257,
    UAVSAR_LOG : 0,
    SENTINEL1 : 0
}

# image ids
UAVSAR_MISSISSIPPI_FLOODED   = 1
UAVSAR_MISSISSIPPI_UNFLOODED = 2
UAVSAR_ARKANSAS_CITY         = 3
UAVSAR_NAPO_RIVER            = 4
SENTINEL1_ROME               = 5
SENTINEL1_LANCIANO           = 6

__RADAR_DOMAIN_INSTRUMENTS = {
    UAVSAR_MISSISSIPPI_FLOODED   : UAVSAR,
    UAVSAR_MISSISSIPPI_UNFLOODED : UAVSAR_LOG,
    UAVSAR_ARKANSAS_CITY         : UAVSAR,
    UAVSAR_NAPO_RIVER            : UAVSAR,
    SENTINEL1_ROME               : SENTINEL1,
    SENTINEL1_LANCIANO           : SENTINEL1
}

HISTORICAL_DATA = {
    UAVSAR_MISSISSIPPI_FLOODED : UAVSAR_MISSISSIPPI_UNFLOODED
}

TRAINING_DATA = {
    UAVSAR_MISSISSIPPI_FLOODED : UAVSAR_ARKANSAS_CITY,
    SENTINEL1_ROME             : SENTINEL1_LANCIANO
}

class RadarDomain(object):
    def __init__(self, instrument, id, image, bounds, ground_truth = None):
        self.instrument = instrument
        self.id = id
        self.image = image.clamp(MINIMUM_VALUES[instrument], MAXIMUM_VALUES[instrument])
        if instrument == UAVSAR or instrument == UAVSAR_LOG:
            self.vv = image.select(['vv'], ['b1'])
            self.hv = image.select(['hv'], ['b1'])
            self.hh = image.select(['hh'], ['b1'])
            self.vh = None
            self.channels = ['vv', 'hv', 'hh']
        elif instrument == SENTINEL1:
            self.vv = image.select(['vv'], ['b1'])
            self.vh = image.select(['vh'], ['b1'])
            self.channels = ['vv', 'vh']
        else:
            self.channels = []
        self.log_scale = (instrument != UAVSAR)
        self.bbox = bounds
        self.bounds = apply(ee.geometry.Geometry.Rectangle, bounds)
        self.center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        self.ground_truth = ground_truth
        self.water_mask = ee.Image("MODIS/MOD44W/MOD44W_005_2000_02_24").select(['water_mask'])
    
    def visualize(self, params = {}, name = 'Radar', show=True):
        all_bands = self.image.bandNames()
        bands = all_bands.getInfo()
        image = self.image
        if len(bands) == 2:
            image = image.addBands(0)
            bands.insert(0, 'constant')
        new_params = {'bands' : bands, 'min' : 0, 'max' : MAXIMUM_VALUES[self.instrument]}
        new_params.update(params)
        return (image, new_params, name, show)

def get_radar_image(id):
    instrument = __RADAR_DOMAIN_INSTRUMENTS[id]
    if instrument == RADARSAT:
        im_hh = ee.Image('18108519531116889794-06793893466375912303')
        im_hv = ee.Image('18108519531116889794-13933004153574033452')
        bounds = (-123.60, 48.95, -122.75, 49.55)
        im = im_hh.select(['b1'], ['hh'])
        im = im.addBands(im_hv.select(['b1'], ['hv']))
    elif instrument == TERRASAR:
        im_hh = ee.Image('18108519531116889794-04996796288385000359')
        bounds = (-79.64, 8.96, -79.55, 9.015)
        im = im_hh.select(['b1'], ['hh'])
    elif instrument == UAVSAR or instrument == UAVSAR_LOG:
        if id == UAVSAR_MISSISSIPPI_UNFLOODED:
            im = ee.Image('18108519531116889794-16648596607414356603')
            im = im.mask(im.select(['b4']))
            bounds = (-91.23, 32.88, -91.02, 33.166)
        elif id == UAVSAR_MISSISSIPPI_FLOODED:
            im = ee.Image('18108519531116889794-12113912950916481117')
            im = im.mask(im.select(['b4']))
            bounds = (-91.23, 32.88, -91.02, 33.166)
        elif id == UAVSAR_ARKANSAS_CITY:
            im = ee.Image('18108519531116889794-12113912950916481117')
            im = im.mask(im.select(['b4']))
            bounds = (-91.32, 33.56, -91.03, 33.7)
        elif id == UAVSAR_NAPO_RIVER:
            ims = ee.ImageCollection('18108519531116889794-08950626664510535970')
            im = ims.mosaic()
            im = im.mask(im)#.eq(ee.Image.constant([257, 257, 257])))
            bounds = (-76.38, -0.27, -74.41, -1.6)
        else:
            return None
        im = im.select(['b1', 'b2', 'b3'], ['hh', 'hv', 'vv'])
    elif instrument == SENTINEL1:
        im_vv = ee.Image('18108519531116889794-15063535589376921925')
        im_vv = im_vv.mask(im_vv)
        im_vh = ee.Image('18108519531116889794-10203767773364605611')
        im_vh = im_vh.mask(im_vh)
        if id == SENTINEL1_ROME:
            bounds = (12.2, 41.72, 12.72, 41.83)
        elif id == SENTINEL1_LANCIANO:
            bounds = (14.15, 41.9, 14.4, 42.5)
        im = im_vv.select(['b1'], ['vv']).addBands(im_vh.select(['b1'], ['vh']))
    return RadarDomain(instrument, id, im, bounds)

def get_ground_truth(domain):
    if domain.id == UAVSAR_MISSISSIPPI_FLOODED:
        #im = ee.Image('18108519531116889794-12921502713420913455')
        im = ee.Image('18108519531116889794-03516536627963450262')
        return im.select(['b1']).clamp(0, 1).mask(domain.image.select(['hh']))
    elif domain.id == UAVSAR_ARKANSAS_CITY:
        im = ee.Image('18108519531116889794-09052745394509652502')
        return im.select(['b1']).clamp(0, 1).mask(domain.image.select(['hh']))
    elif domain.id == SENTINEL1_ROME:
        im = ee.Image('18108519531116889794-01677512005004143246')
        return im.select(['b1']).clamp(0, 1)
    elif domain.id == SENTINEL1_LANCIANO:
        im = ee.Image('18108519531116889794-10734434487272614892')
        return im.select(['b1']).clamp(0, 1)
    return None


