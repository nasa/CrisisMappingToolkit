import logging
logging.basicConfig(level=logging.ERROR)
import util.ee_authenticate
util.ee_authenticate.initialize()

import os
import ee
import functools

import radar.domains
from radar.active_contour import *

import sys
import PIL
import numpy
from PyQt4 import QtGui
app = QtGui.QApplication(sys.argv)

# Specify the data set to use - see /radar/domains.py
DOMAIN = radar.domains.UAVSAR_MISSISSIPPI_FLOODED

domain = radar.domains.get_radar_image(DOMAIN)
#result = active_contour(domain)

def active_contour_step(local_image, nodes, step):
    if step % 10 == 0:
        respace_nodes(nodes)
    shift_nodes(local_image, nodes)

class ActiveContourWindow(QtGui.QWidget):
    def __init__(self, domain):
        super(ActiveContourWindow, self).__init__()
        self.setGeometry(300, 300, 650, 650)
        self.setWindowTitle('Active Contour')
        self.domain = domain
        (self.local_image, self.nodes) = initialize_active_contour(domain)
        channels = [self.local_image.get_image('hh'), self.local_image.get_image('hv'), self.local_image.get_image('vv')]
        channel_images = [PIL.Image.fromarray(numpy.uint8(c >> 8)) for c in channels]
        self.display_image = PIL.Image.merge('RGB', channel_images)
        self.step = 0
        self.show()

    def paintEvent(self, event):
        imageqt = ImageQt.ImageQt(self.display_image)
        p = QtGui.QPainter()
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True);
        scale = self.height() / float(imageqt.height() + 10)
        p.scale(scale, scale)
        p.translate((self.width() / 2 / scale - imageqt.width() / 2),
                    (self.height() / 2 / scale - imageqt.height() / 2))
        p.fillRect(0, 0, imageqt.width(), imageqt.height(), QtGui.QColor(0, 0, 0))
        p.drawImage(0, 0, imageqt)
        NODE_RADIUS = 4
        # draw nodes
        for loop in self.nodes:
            for i in range(len(loop)):
                p.setPen(QtGui.QColor(255, 0, 0))
                p.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
                p.drawEllipse(loop[i][1] - NODE_RADIUS / 2.0,
                        loop[i][0] - NODE_RADIUS / 2.0, NODE_RADIUS, NODE_RADIUS)
        # draw lines between nodes
        for loop in self.nodes:
            for i in range(len(loop)):
                if len(loop) > 1:
                    n = i+1
                    if n == len(loop):
                        n = 0
                    p.setPen(QtGui.QColor(0, 255, 0))
                    p.drawLine(loop[i][1], loop[i][0], loop[n][1], loop[n][0])
        p.end()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            active_contour_step(self.local_image, self.nodes, self.step)
            self.repaint()
            print 'Step'
            self.step += 1
        if event.key() == QtCore.Qt.Key_Q:
            QtGui.QApplication.quit()

ex = ActiveContourWindow(domain)
sys.exit(app.exec_())

