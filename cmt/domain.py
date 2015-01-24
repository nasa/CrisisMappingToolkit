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

# Default search path for domain xml files: [root]/config/domains/[sensor_name]/
DOMAIN_SOURCE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
        ".." + os.path.sep + "config" + os.path.sep + "domains")

# Default search path for sensors description xml files: [root]/config/sensors
SENSOR_SOURCE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
        ".." + os.path.sep + "config" + os.path.sep + "sensors")

class SensorObservation(object):
    '''A class for accessing a sensor's observation at one time.'''

    def __init__(self, xml_root, eeBounds, isDomainFile=False):
        '''Initialize the object from XML data and the desired bounding box'''
        
        # Public class members
        self.sensor_name   = 'Unnamed' # Name of the sensor!
        self.image         = None      # EE image object containing the selected sensor bands
        self.band_names    = []        # The name assigned to each band
        self.log_scale     = False     # True if the sensor uses a log 10 scale
        self.minimum_value = None      # Min and max sensor values (shared across bands)
        self.maximum_value = None
        self.band_resolutions    = dict() # Specified resolution of each band in meters
        self.water_distributions = dict() # Info about water characteristics in each band

        # You can also access each band as a member variable, e.g. self.hv
        #   gives access to the band named 'hv'

        # Private class members
        self.__display_bands = None
        self.__display_gains = None
        self.__mask_info     = None
        self.__band_sources  = dict() # Where to get each band from

        # Parse the xml file to fill in the class variables
        self.__load_xml(xml_root, isDomainFile)
        
        # Set up the EE image object using the band information
        self.__load_image(eeBounds)


    def get_date(self):
        '''Returns the start date for the image if one was provided, None otherwise.'''
        if ('start_date' in self.__band_sources[0]):
            return self.__band_sources[0]['start_date']
        else:
            return None

    def __loadPieceOfSourceInfo(self, source_band, info_name, dictionary):
        '''Helper function - Look for and load source info about a band'''
        result = source_band.find(info_name)
        if result != None:
            dictionary[info_name] = result.text

    def __load_source(self, source_element):
        '''load a data source for a band or mask, represented by the <source> tag.'''
        # A source is stored like this: {'mosaic', 'source', 'eeid'}
        d = dict()
        source_band = source_element.find('source')
        if source_band == None:
            return d # Source not specified, leave the dictionary empty!
            
        # if it's a mosaic, combine the images in an EE ImageCollection
        mosaic = source_band.get('mosaic')
        if mosaic != None:
            if mosaic.lower() == 'true':
                d['mosaic'] = True
            elif mosaic.lower() == 'false':
                d['mosaic'] = False
            else:
                raise Exception('Unexpected value of mosaic, %s.' % (source_band.get('mosaic')))
            
        # The name of the band in the source data, maybe not what we will call it in the output image.
        name = source_band.find('name')
        if name != None:
            # the name of the band in the original image
            d['source'] = name.text

        # Load more information about the band source
        self.__loadPieceOfSourceInfo(source_band, 'eeid',       d) # The id of the image to load, if a single image.
        self.__loadPieceOfSourceInfo(source_band, 'collection', d) # The ImageCollection name of the data, if any.
        self.__loadPieceOfSourceInfo(source_band, 'start_date', d)    # Start and end dates used to filter an ImageCollection.
        self.__loadPieceOfSourceInfo(source_band, 'end_date',   d)

        return d

    def __load_distribution(self, root):
        '''load a probability distribution into a python dictionary, which may, for
            example, represent the expected distribution of water pixels'''
        d     = dict()
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
            
        #print 'Created water distribution: '
        #print d
            
        return d

    def __load_bands(self, root_element):
        '''Read the band specification and load it into __band_sources and __mask_source.
            Does not load the bands'''
        # Look for default water distribution info at the top level
        default_water = dict()
        for d in root_element.findall('distribution'):
            if d.get('name').lower() == 'water':
                default_water = self.__load_distribution(d)

        # Read bands, represented by <band> tag
        bands = root_element.find('bands')
        if bands == None:
            return # Nothing to do if no bands tag!
        
        # Look for display bands at the top band level
        display_bands = bands.find('display_bands')
        if display_bands != None:
            display_band_list = display_bands.text.replace(' ','').split(',') # The band names are comma seperated
            if len(display_band_list) > 3:
                raise Exception('Cannot have more than three display bands!')
            self.__display_bands = display_band_list

        # Looks for display band gains at the top level
        display_gains = bands.find('display_gains')
        if display_gains != None:
            display_gain_list = display_gains.text.split(',') # The band names are comma seperated
            if len(display_gain_list) > 3:
                raise Exception('Cannot have more than three display band gains!')
            self.__display_gains = display_gain_list


        # shared information (e.g., all bands have same eeid) is loaded directly in <bands>
        #print 'Reading default source...'
        default_source = self.__load_source(bands) # Located in <bands><source>
        # If any bands are already loaded (meaning we are in the domain file), apply this source info to them.
        for b in self.band_names:  
            self.__band_sources[b].update(default_source)
        if self.__mask_info != None:
            self.__mask_info.update(default_source)
        resolution = bands.find('resolution')
        if resolution != None: # <bands><resolution>
            default_resolution = float(resolution.text)
        else:
            default_resolution = 10 # Default resolution is 10 meters if not specified!


        # load individual <band> tags
        for b in bands.findall('band'):
            try:
                name = b.find('name').text
            except:
                raise Exception('Unnamed band.')
            #print 'Getting info for band: ' + name
            if name not in self.band_names: # Only append each band name once
                self.band_names.append(name)
            if name not in self.__band_sources: # Only append each band source once
                self.__band_sources[name] = dict()
            self.__band_sources[name].update(default_source) # Start with the default source information
            self.__band_sources[name].update(self.__load_source(b)) # Band source information is stored like: {'mosaic', 'source', 'eeid'}

            #print 'Source for this band = '
            #print str(self.__band_sources[name])

            # Look for water distribution information in this band
            if name not in self.water_distributions:
                self.water_distributions[name] = dict()
            self.water_distributions[name].update(default_water)
            for d in b.findall('distribution'):
                if d.get('name').lower() == 'water':
                    self.water_distributions[name].update(self.__load_distribution(d))
                    
            # Load resolution for this band
            resolution = b.find('resolution')
            if resolution != None:
                self.band_resolutions[name] = float(resolution.text)
            else:
                self.band_resolutions[name] = default_resolution
            #print 'For band name ' + name + ' found resolution = ' + str(self.band_resolutions[name])
                
                    
        # read mask, in <mask> tag
        mask = bands.find('mask')
        if mask != None:
            if self.__mask_info == None:
                self.__mask_info = dict()
            if mask.get('self') == 'true': # Self mask means that zero-valued pixels in the source will be masked out.
                self.__mask_info['self'] = True
            else: # Otherwise there must be an external source
                self.__mask_info['self'] = False
                source = mask.find('source')
                if source == None: # Read in source information about the mask
                    raise Exception('Mask specified with no source!')
                else:
                    self.__mask_info.update(self.__load_source(mask))




    def __load_image(self, eeBounds):
        '''given band specifications in __band_sources and __mask_source, load them into self.image'''
        # This is setting up an EE object, not actually downloading any data from the web.
        
        # Load the bands, combine into image
        for i in range(len(self.band_names)):
            thisBandName = self.band_names[i]
            source       = self.__band_sources[thisBandName]
            #print '======================================='
            #print 'Loading band: ' + thisBandName
            #print source
            if 'mosaic' in source:
                ims = ee.ImageCollection(source['eeid'])
                im  = ims.mosaic()
            elif 'eeid' in source:
                im = ee.Image(source['eeid'])
            elif ('collection' in source) and ('start_date' in source) and ('end_date' in source):
                # Select a single image from an Earth Engine image collection
                im = ee.ImageCollection(source['collection']).filterBounds(eeBounds).filterDate(source['start_date'], source['end_date']).mean();
            else: # Not enough information was provided!
                raise Exception('Incomplete source information for band: ' + thisBandName)
                
            band = im.select([source['source']], [thisBandName])
            #print band.getInfo()
            if self.image == None:
                self.image = band
            else:
                self.image = self.image.addBands(band)
            # set band as member variable, e.g., self.__dict__['hv'] is equivalent to self.hv
            self.__dict__[thisBandName] = band
            
        #print '---------------------------'
        #print self.image.getInfo()
        #    
        # Apply mask once all the bands are loaded
        if self.__mask_info != None:
            if 'self' in self.__mask_info and self.__mask_info['self']:
                self.image = self.image.mask(self.image) # Apply self-mask
            elif 'eeid' in self.__mask_info: # Apply a mask from an external source
                self.image = self.image.mask(ee.Image(self.__mask_info['eeid']).select([self.__mask_info['source']], ['b1']))
            else:
                raise Exception('Not enough mask information specified!')

        # Apply minimum and maximum value to all bands if specified
        #if self.minimum_value == None or  self.maximum_value == None:
        #    raise Exception('Minimum and maximum value not specified.')
        if (self.minimum_value != None) and (self.maximum_value != None):
            self.image = self.image.clamp(self.minimum_value, self.maximum_value)

        #print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        #print self.image.getInfo()
        #print '\n\n\n'

    def __load_range(self, tag):
        '''read a <range> tag'''
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

    def __load_xml(self, xml_root, isDomainFile=False):
        '''Parse an xml document representing a domain or a sensor'''

        if (xml_root.tag != "sensor"):
            raise Exception("Sensor XML file required!")

        # Read the sensor name
        name = xml_root.find('name')
        if name == None:
            raise Exception('Sensor name not found!')
        self.sensor_name = name.text.lower()

        if isDomainFile: # Look for the matching sensor xml file and load it first
            # Make sure the file exists
            sensor_xml_path = os.path.join(SENSOR_SOURCE_DIR, self.sensor_name + ".xml")
            if not os.path.exists(sensor_xml_path):
                raise Exception('Could not find sensor file: ' + sensor_xml_path)
            # Load the XML and recursively call this function to parse it
            #print 'Reading file: ' + sensor_xml_path
            tree       = ET.parse(sensor_xml_path)
            other_root = tree.getroot() # This will be a sensor node
            self.__load_xml(other_root, False)
            #print 'Finished loading sensor file -----------------------'

        # Search for the min and max values of the sensor
        (a, b) = self.__load_range(xml_root.find('range'))
        if a != None:
            self.minimum_value = a
        if b != None:
            self.maximum_value = b
        
        # If scaling tag with type log10 present, take the log of the image
        scale = xml_root.find('scaling')
        if scale != None:
            self.log_scale = (scale.get('type') == 'log10')
        
        # Read data about all the bands
        self.__load_bands(xml_root)

    
    def visualize(self, params = {}, name = None, show=True):
        '''Return all the parameters needed for the "addToMap()" function for a human-readable image of this sensor'''
        
        if name == None: # The display name
            name = self.sensor_name
            
        image      = self.image
        band_names = self.band_names

        if self.__display_bands != None: # The user has specified display bands
            if len(self.__display_bands) != 3:
                raise Exception('')
            
            b0         = self.image.select(self.__display_bands[0])
            b1         = self.image.select(self.__display_bands[1])
            b2         = self.image.select(self.__display_bands[2])
            image      = b0.addBands(b1).addBands(b2)
            band_names = self.__display_bands
            
        else: # Automatically decide the display bands
            
            if len(band_names) == 2:  # If two bands, add a constant zero band to fake a "B" channel
                image      = self.image.addBands(0)
                band_names = ['constant'] + band_names
            if (len(band_names) > 3): # If more than three bands, just use the first three.
                image      = self.image.select(band_names[0]).addBands(self.image.select(band_names[1])).addBands(self.image.select(band_names[2]))
                band_names = self.band_names[0:2]
        
        if (self.minimum_value != None) and (self.maximum_value != None):
            new_params = {'bands' : band_names, 'min' : self.minimum_value, 'max' : self.maximum_value}
        else: # No min and max set
            new_params = {'bands' : band_names}
        new_params.update(params)
        
        #print '-------------------------------------------------'
            
        if (not 'gain' in params) and (self.__display_gains != None): # If gains were not specified, use ours!
            new_params['gain'] = self.__display_gains
       
        return (image, new_params, name, show)




#=========================================================




class Domain(object):
    '''A class representing a problem domain. Loads sensor and location
        information from an xml file. Default information may be specified in a
        file specific to a sensor type, which can be overridden.'''
    def __init__(self, xml_file, is_training=False):
        
        self.name             = 'Unnamed' # The name assigned to the domain.
        self.bbox             = None      # Bounding box of the domain.
        self.bounds           = None      # Copy of self.bbox in Earth Engine format
        self.center           = None      # Center of the bounding box.
        self.ground_truth     = None      # Ground truth image.
        self.training_domain  = None      # Another domain used only for training.
        self.algorithm_params = {}        # Dictionary of algorithm parameters
        self.sensor_list      = []        # Contains a SensorObservation object for each related sensor.
        

        # You can also access each sensor as a member variable, e.g. self.uavsar
        #   gives access to the sensor named 'uavsar'

        self.__load_xml(xml_file, is_training) # Call function to initialize from the XML file


    def get_dem(self):
        '''Returns a DEM image object if one is loaded'''
        
        # Use ned13 DEM if available, otherwise use the global srtm90 DEM.
        hasNedDem = False
        for s in self.sensor_list:
            if s.sensor_name.lower() == 'ned13':
                hasNedDem = True
                break
        if hasNedDem:
            return self.ned13
        else:
            return self.srtm90

    

    def __load_bbox(self, root):
        '''read a bbox, <bbox>'''
        b = None
        if root != None:
            try:
                bl = root.find('bottomleft')
                tr = root.find('topright')
                b  = tuple(map(lambda x: float(x.text),
                            [bl.find('lon'), bl.find('lat'), tr.find('lon'), tr.find('lat')]))
            except:
                raise Exception("Failed to load bounding box for domain.")
        if (b[0] > b[2]) or (b[1] > b[3]): # Check that min and max values are properly ordered
            raise Exception("Illegal bounding box values!")
        return b

    def __load_xml(self, xml_file, is_training=False):
        '''load an xml file representing a domain or a sensor'''
        #print 'Reading file: ' + xml_file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        if root.tag != "domain":
            raise Exception("Domain XML file required!")

        # get name of domain
        try:
            self.name = root.find('name').text
        except:
            raise Exception('Domain has no name.')

        # Load the bounding box that contains the domain
        self.bbox   = self.__load_bbox(root.find('bbox'))
        self.bounds = apply(ee.geometry.Geometry.Rectangle, self.bbox)
        self.center = ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
        
        if not is_training:
            # Try to load the training domain
            training_domain = root.find('training_domain')
            if training_domain != None:
                #sensor_domain_folder     = os.path.join(DOMAIN_SOURCE_DIR,
                sensor_domain_folder = os.path.dirname(xml_file) # Look in the same directory as the primary xml file
                training_file_xml_path = os.path.join(sensor_domain_folder, training_domain.text + '.xml')
                if not os.path.exists(training_file_xml_path):
                    raise Exception('Training file not found: ' + training_file_xml_path)
                self.training_domain = Domain(training_file_xml_path, True)
                
            # Load any algorithm params
            algorithm_params = root.find('algorithm_params')
            if algorithm_params != None:
                for child in algorithm_params:
                    self.algorithm_params[child.tag] = float(child.text)
                
        
        # Make sure the <sensors> tag is present
        sensors = root.find('sensors')
        if sensors == None:
            raise Exception('Must have at least one sensor for the domain!')
        
        # Load each <sensor> tag seperately
        for sensor_node in sensors.findall('sensor'):
            newSensor = SensorObservation(sensor_node, self.bounds, True) # Send the sensor node of the domain file for parsing
            self.sensor_list.append(newSensor)                             # Store the new sensor object
            
            # Set sensor as member variable, e.g., self.__dict__['uavsar'] is equivalent to self.uavsar
            self.__dict__[newSensor.sensor_name] = newSensor

        # Load a ground truth image if one was specified
        # - These are always binary and loaded from an asset ID in Maps Engine.
        # TODO: The ground truth should be masked to a sensor image in the eval function!
        truth_ee_id = root.find('truth')
        if truth_ee_id != None:
            self.ground_truth = ee.Image(truth_ee_id.text).select(['b1']).clamp(0, 1).mask(self.sensor_list[0].image.select([self.sensor_list[0].band_names[0]]))


