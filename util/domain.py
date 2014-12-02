# -----------------------------------------------------------------------------
# Copyright * 2014, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.
#
# The Crisis Mapping Toolkit (CMT) v1 platform is licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# -----------------------------------------------------------------------------

import os
import xml.etree.ElementTree as ET

import ee

DATA_SOURCE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
        ".." + os.path.sep + "config" + os.path.sep + "sensors")

class Domain(object):
    def __init__(self, xml_file):
        self.name = 'Unnamed'
        self.image = None
        self.truth = None
        self.sensor = None
        self.bands = []
        self.log_scale = False
        self.water = dict()
        self.ground_truth = None
        # also data member for each band

        self.__mask_source = None
        self.__band_sources = dict()
        self.__load_xml(xml_file)

    def __load_source(self, source_element):
        d = dict()
        source_band = source_element.find('source')
        if source_band != None:
            mosaic = source_band.get('mosaic')
            if mosaic != None:
                if mosaic.lower() == 'true':
                    d['mosaic'] = True
                elif mosaic.lower() == 'false':
                    d['mosaic'] = False
                else:
                    raise Exception('Unexpected value of mosaic, %s.' % (source_band.get('mosaic')))
            name = source_band.find('name')
            if name != None:
                d['source'] = name.text
            band_eeid = source_band.find('eeid')
            if band_eeid != None:
                d['eeid'] = band_eeid.text
        return d

    def __load_distribution(self, root):
        d = dict()
        model = root.find('model')
        if model != None:
            d['model'] = model.text
        mode = root.find('mode')
        if mode != None:
            d['mode'] = dict()
            if mode.find('range') != None:
                (d['mode']['min'], d['mode']['max']) = self.__load_range(mode.find('range'))
        r = root.find('range')
        if r != None:
            d['range'] = self.__load_range(r)
        b = root.find('buckets')
        if b != None:
            try:
                d['buckets'] = int(b.text)
            except:
                raise Exception('Buckets in distribution must be integer.')
        return d

    def __load_bands(self, root_element):
        default_water = dict()
        for d in root_element.findall('distribution'):
            if d.get('name').lower() == 'water':
                default_water = self.__load_distribution(d)

        # read bands
        bands = root_element.find('bands')
        if bands != None:
            default_source = self.__load_source(bands)
            for b in self.bands:
                self.__band_sources[b].update(default_source)
            if self.__mask_source != None:
                self.__mask_source.update(default_source)

            for b in bands.findall('band'):
                try:
                    name = b.find('name').text
                except:
                    raise Exception('Unnamed band.')
                if name not in self.bands:
                    self.bands.append(name)
                if name not in self.__band_sources:
                    self.__band_sources[name] = dict()
                self.__band_sources[name].update(self.__load_source(b))

                if name not in self.water:
                    self.water[name] = dict()
                self.water[name].update(default_water)
                for d in b.findall('distribution'):
                    if d.get('name').lower() == 'water':
                        self.water[name].update(self.__load_distribution(d))
            # read mask
            mask = bands.find('mask')
            if mask != None:
                if self.__mask_source == None:
                    self.__mask_source = dict()
                if mask.get('self') == 'true':
                    self.__mask_source['self'] = True
                source = mask.find('source')
                if source != None:
                    self.__mask_source.update(self.__load_source(mask))

    def __load_image(self):
        # load the bands, combine into image
        for i in range(len(self.bands)):
            source = self.__band_sources[self.bands[i]]
            if 'source' not in source or 'eeid' not in source or source['eeid'] == None:
                raise Exception('Incomplete band specification.')
            if 'mosaic' in source:
                ims = ee.ImageCollection(source['eeid'])
                im = ims.mosaic()
            else:
                im = ee.Image(source['eeid'])
            band = im.select([source['source']], [self.bands[i]])
            if self.image == None:
                self.image = band
            else:
                self.image = self.image.addBands(band)
            # set band as member variable
            self.__dict__[self.bands[i]] = band
        if self.__mask_source != None:
            if 'self' in self.__mask_source and self.__mask_source['self']:
                self.image = self.image.mask(self.image)
            else:
                self.image = self.image.mask(ee.Image(self.__mask_source['eeid']).select([self.__mask_source['source']], ['b1']))
        if self.minimum_value == None or  self.maximum_value == None:
            raise Exception('Minimum and maximum value not specified.')
        self.image = self.image.clamp(self.minimum_value, self.maximum_value)

    def __load_bbox(self, root):
        b = None
        if root != None:
            try:
                bl = root.find('bottomleft')
                tr = root.find('topright')
                b = tuple(map(lambda x: float(x.text),
                            [bl.find('lon'), bl.find('lat'), tr.find('lon'), tr.find('lat')]))
            except:
                raise Exception("Failed to load bounding box for domain.")
        return b

    def __load_range(self, tag):
        a = None
        b = None
        if tag != None:
            try:
                a = tag.find('minimum').text
                b = tag.find('maximum').text
                try:
                    a = int(a)
                    b = int(b)
                except:
                    a = float(a)
                    b = float(b)
            except:
                raise Exception('Failed to load range tag.')
        return (a, b)

    def __load_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        domain = False
        if root.tag != "domain" and root.tag != "sensor":
            raise Exception("XML file not a domain or sensor.")

        # load default values for given source
        data_source = None
        try:
            data_source = root.find('sensor').text.lower()
            self.sensor = data_source
        except:
            pass
        if data_source != None:
            self.__load_xml(DATA_SOURCE_DIR + os.path.sep + data_source + ".xml")

        # get name of domain
        try:
            self.name = root.find('name').text
        except:
            raise Exception('Domain has no name.')

        self.bbox = self.__load_bbox(root.find('bbox'))
        (a, b) = self.__load_range(root.find('range'))
        if a != None:
            self.minimum_value = a
        if b != None:
            self.maximum_value = b
        
        scale = root.find('scaling')
        if scale != None:
            self.log_scale = scale.get('type') == 'log10'
        
        self.__load_bands(root)
        if root.tag == "domain":
            self.__load_image()
            self.bounds = apply(ee.geometry.Geometry.Rectangle, self.bbox)
            self.center = ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
            truth = root.find('truth')
            if truth != None:
                self.ground_truth = ee.Image(truth.text).select(['b1']).clamp(0, 1).mask(self.image.select([self.bands[0]]))
    
    def visualize(self, params = {}, name = None, show=True):
        if name == None:
            name = self.name
        bands = self.bands
        image = self.image
        if len(bands) == 2:
            image = self.image.addBands(0)
            bands = ['constant'] + bands
        new_params = {'bands' : bands, 'min' : self.minimum_value, 'max' : self.maximum_value}
        new_params.update(params)
        return (image, new_params, name, show)

