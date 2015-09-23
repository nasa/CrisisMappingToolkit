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


import logging
logging.basicConfig(level=logging.ERROR)
try:
    import cmt.ee_authenticate
except:
    import sys
    import os.path
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    import cmt.ee_authenticate
cmt.ee_authenticate.initialize()

import time

import cmt.domain
from cmt.radar.active_contour import *

import sys
import PIL
from PIL import ImageQt
import numpy
from PyQt4 import QtGui, QtCore
app = QtGui.QApplication(sys.argv)


# from PIL import Image, ImageChops
# import matplotlib.pyplot as plt
# plt.imread('/home/smcmich1/fileTest.tif')
# raise Exception('DEBUG')


THIS_FILE_FOLDER = os.path.dirname(os.path.realpath(__file__))


# DOMAIN SELECTION IS HERE!
# domain = cmt.domain.Domain(os.path.join(THIS_FILE_FOLDER, '..') + '/config/domains/uavsar/mississippi.xml')
# domain = cmt.domain.Domain(os.path.join(THIS_FILE_FOLDER, '..') + '/config/domains/sentinel1/malawi_2015_1.xml')
# domain = cmt.domain.Domain(os.path.join(THIS_FILE_FOLDER, '..') + '/config/domains/sentinel1/rome_small.xml')
# domain = cmt.domain.Domain(os.path.join(THIS_FILE_FOLDER, '..') + '/config/domains/skybox/malawi_2015.xml')
# domain = cmt.domain.Domain(os.path.join(THIS_FILE_FOLDER, '..') + '/config/domains/skybox/gloucester_2014_10.xml')
# domain = cmt.domain.Domain(os.path.join(THIS_FILE_FOLDER, '..') + '/config/domains/skybox/sumatra_2014_10.xml')
domain = cmt.domain.Domain(os.path.join(THIS_FILE_FOLDER, '..') + '/config/domains/skybox/new_bedford_2014_10.xml')

# result = active_contour(domain) # Run this to compute the final results!


def active_contour_step(local_image, snake, step):
    '''Perform another step of the active contour algorithm'''
    if snake.done:
        return True

    t = time.time()
    if step % 10 == 0: # Do extra work every tenth iteration
        snake.respace_nodes()
        snake.shift_nodes() # shift before fixing geometry since reversal of orientation possible
        snake.fix_geometry()
    else:
        snake.shift_nodes()
    print time.time() - t

    return False

class ActiveContourWindow(QtGui.QWidget):
    '''Dedicated class for drawing the progress of the active contour algorithm'''
    def __init__(self, domain):
        super(ActiveContourWindow, self).__init__()
        self.setGeometry(300, 300, 650, 650)
        self.setWindowTitle('Active Contour')
        self.domain = domain

        """
        Fetch image and compute statistics
        sensor         = domain.get_radar()
        detect_channel = domain.algorithm_params['water_detect_radar_channel']
        ee_image       = sensor.image.select([detect_channel]).toUint16()
        if sensor.log_scale:
            statisics_image = ee_image.log10()
        else:
            statisics_image = ee_image
        (band_names, band_statistics) = compute_band_statistics(statisics_image, domain.ground_truth, domain.bounds)

        (self.local_image, self.snake) = initialize_active_contour(domain, ee_image, band_statistics, sensor.log_scale)

         Retrieve the local image bands and merge them into a fake RGB image
        channels = [self.local_image.get_image(detect_channel), self.local_image.get_image(detect_channel), self.local_image.get_image(detect_channel)]
        channel_images = [PIL.Image.fromarray(numpy.uint8(c*255/1200)) for c in channels]  Convert from 16 bit to 8 bit
        self.display_image = PIL.Image.merge('RGB', channel_images)
        self.step = 1
        self.show()
        """
        """
        Initialize the contour with the selected sensor band
        sensor_name = 'uavsar'
        sensor      = getattr(domain, sensor_name)
        ee_image    = sensor.image.select(['hh'])

        # TODO: Make sure the name and statistics line up inside the class!
        # Compute statistics for each band -> Log10 needs to be applied here!
        (band_names, band_statistics) = compute_band_statistics(statisics_image, domain.ground_truth, domain.bounds)

        (self.local_image, self.snake) = initialize_active_contour(domain, ee_image, band_statistics, sensor.log_scale)

        # Retrieve the local image bands and merge them into a fake RGB image
        #channels = [self.local_image.get_image('hh'), self.local_image.get_image('hv'), self.local_image.get_image('vv')]
        channels = [self.local_image.get_image('hh'), self.local_image.get_image('hh'), self.local_image.get_image('hh')]
        channel_images = [PIL.Image.fromarray(numpy.uint8(c >> 8)) for c in channels] # Convert from 16 bit to 8 bit
        self.display_image = PIL.Image.merge('RGB', channel_images)
        self.step = 1
        self.show()
        """

        SKYBOX_SCALE = 1200 / 256
        train_domain = domain.training_domain  # For skybox data there is probably no earlier image to train off of
        try:  # The Skybox data can be in one of two names
            sensor = domain.skybox
            trainSensor = train_domain.skybox
        except:
            sensor = domain.skybox_nir
            trainSensor = train_domain.skybox_nir
        ee_image = sensor.image.toUint16()  # For Skybox, these are almost certainly the same image.
        ee_image_train = trainSensor.image.toUint16()

        if train_domain.training_features:  # Train using features
            (band_names, band_statistics) = compute_band_statistics_features(ee_image_train, train_domain.training_features)
        else: # Train using training truth
            (band_names, band_statistics) = compute_band_statistics(ee_image_train, train_domain.ground_truth,
                                                                    train_domain.bounds)
        (self.local_image, self.snake) = initialize_active_contour(domain, ee_image, band_statistics, False)

        # Retrieve the local image bands and merge them into a fake RGB image
        channels = [self.local_image.get_image('Red'), self.local_image.get_image('Green'), self.local_image.get_image('Blue')]
        channel_images = [PIL.Image.fromarray(numpy.uint8(c / SKYBOX_SCALE)) for c in channels] # Convert from Skybox range to 8 bit
        self.display_image = PIL.Image.merge('RGB', channel_images)
        self.step = 1
        self.show()

    def paintEvent(self, event):
        imageqt = ImageQt.ImageQt(self.display_image)
        p = QtGui.QPainter()
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True);
        scale = self.height() / float(imageqt.height() + 10)
        p.scale(scale, scale)
        p.translate((self.width() / 2 / scale - imageqt.width()  / 2),
                    (self.height() / 2 / scale - imageqt.height() / 2))
        p.fillRect(0, 0, imageqt.width(), imageqt.height(), QtGui.QColor(0, 0, 0))
        p.drawImage(0, 0, imageqt)
        NODE_RADIUS = 4
        # draw nodes
        for loop in self.snake.loops:
            for i in range(len(loop.nodes)):
                p.setPen(QtGui.QColor(255, 0, 0))
                p.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
                p.drawEllipse(loop.nodes[i][1] - NODE_RADIUS / 2.0,
                        loop.nodes[i][0] - NODE_RADIUS / 2.0, NODE_RADIUS, NODE_RADIUS)
        # draw lines between nodes
        for loop in self.snake.loops:
            for i in range(len(loop.nodes)):
                if len(loop.nodes) > 1:
                    n = i+1
                    if n == len(loop.nodes):
                        n = 0
                    p.setPen(QtGui.QColor(0, 255, 0))
                    p.drawLine(loop.nodes[i][1], loop.nodes[i][0], loop.nodes[n][1], loop.nodes[n][0])
        p.end()

    def keyPressEvent(self, event):
        '''Update the algorithm on space, quit on "q"'''
        if event.key() == QtCore.Qt.Key_Space:
            active_contour_step(self.local_image, self.snake, self.step)
            self.repaint()
            self.step += 1
        if event.key() == QtCore.Qt.Key_Q:
            QtGui.QApplication.quit()

ex = ActiveContourWindow(domain)
sys.exit(app.exec_())
