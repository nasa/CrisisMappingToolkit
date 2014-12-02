import ee
from util.mapclient_qt import centerMap, addToMap

import domains
from histogram import RadarHistogram

def grow_regions(domain, thresholded, thresholds):
    REGION_GROWTH_RANGE = 20
    neighborhood_kernel = ee.Kernel.square(1, 'pixels', False)
    loose_thresholded = domain.image.select([domain.bands[0]]).lte(thresholds[0])
    for i in range(1, len(domain.bands)):
        loose_thresholded = loose_thresholded.And(domain.image.select([domain.bands[i]]).lte(thresholds[i]))
    addToMap(loose_thresholded, {'min': 0, 'max': 1}, 'Loose', False)
    for i in range(REGION_GROWTH_RANGE):
        thresholded = thresholded.convolve(neighborhood_kernel).And(loose_thresholded)
    return thresholded

def threshold(domain, historical_domain=None):
    hist = RadarHistogram(domain)
    
    thresholds = hist.get_thresholds()
    
    results = []
    for c in range(len(thresholds)):
        ch = domain.bands[c]
        results.append(domain.image.select([ch], [ch]).lte(thresholds[c]))

    result_image = results[0]
    for c in range(1, len(results)):
        result_image = result_image.addBands(results[c], [domain.bands[c]])
    addToMap(result_image, {'min': 0, 'max': 1}, 'Color Image', False)

    # TODO: compare water pixels to expected distribution
    # take difference of two image, remove pixels that aren't below original non region-growing threshold and don't change by at least fixed amount
    
    result_image = results[0].select([domain.bands[0]], ['b1'])
    for c in range(1, len(results)):
        result_image = result_image.And(results[c])
    
    growing_thresholds = hist.find_loose_thresholds()
    result_image = grow_regions(domain, result_image, growing_thresholds)
    
    hist.show_histogram()

    return result_image

