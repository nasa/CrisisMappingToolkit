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

domain = cmt.domain.Domain(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..') + '/config/domains/uavsar/mississippi.xml')
#result = active_contour(domain)

def active_contour_step(local_image, snake, step):
    if snake.done:
        return
    t = time.time()
    if step % 10 == 0:
        snake.respace_nodes()
        snake.shift_nodes() # shift before fixing geometry since reversal of orientation possible
        snake.fix_geometry()
    else:
        snake.shift_nodes()
    print time.time() - t

class ActiveContourWindow(QtGui.QWidget):
    def __init__(self, domain):
        super(ActiveContourWindow, self).__init__()
        self.setGeometry(300, 300, 650, 650)
        self.setWindowTitle('Active Contour')
        self.domain = domain
        (self.local_image, self.snake) = initialize_active_contour(domain)
        channels = [self.local_image.get_image('hh'), self.local_image.get_image('hv'), self.local_image.get_image('vv')]
        channel_images = [PIL.Image.fromarray(numpy.uint8(c >> 8)) for c in channels]
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
        p.translate((self.width() / 2 / scale - imageqt.width() / 2),
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
        if event.key() == QtCore.Qt.Key_Space:
            active_contour_step(self.local_image, self.snake, self.step)
            self.repaint()
            self.step += 1
        if event.key() == QtCore.Qt.Key_Q:
            QtGui.QApplication.quit()

ex = ActiveContourWindow(domain)
sys.exit(app.exec_())

