import domains

from util.mapclient_qt import centerMap, addToMap
from util.local_ee_image import LocalEEImage

import ee
import math
import PIL
from PIL import ImageQt
from PyQt4 import QtCore, QtGui

MIN_NODE_SEPARATION =  5
MAX_NODE_SEPARATION = 15
SEED_REGION_BORDER  =  5
EXPECTED_WATER_MEAN = 750
EXPECTED_WATER_STD_DEV = 500
ALLOWED_DEVIATIONS = 2.5

def save_image(filename, tile):
    imageqt = ImageQt.ImageQt(tile)
    pixmap = QtGui.QPixmap(imageqt.width(), imageqt.height())
    painter = QtGui.QPainter()
    painter.begin(pixmap)
    painter.fillRect(0, 0, imageqt.width(), imageqt.height(), QtGui.QColor(0, 0, 0))
    painter.drawImage(0, 0, imageqt)
    painter.end()
    pixmap.save(filename, "PNG")

def __inside_line(a, b, x):
    v = (b[0] - a[0]) * (x[1] - a[1]) - (b[1] - a[1]) * (x[0] - a[0])
    return v >= 0

def get_goodness(data, n1, n2, n3):
    mean = 0
    mean_2 = 0
    n = 0
    x_min = max(0,             min(n1[0], n2[0], n3[0]) - SEED_REGION_BORDER)
    x_max = min(data.shape[0], max(n1[0], n2[0], n3[0]) + SEED_REGION_BORDER)
    y_min = max(0,             min(n1[1], n2[1], n3[1]) - SEED_REGION_BORDER)
    y_max = min(data.shape[1], max(n1[1], n2[1], n3[1]) + SEED_REGION_BORDER)
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            if not __inside_line(n1, n2, (x, y)):
                continue
            if not __inside_line(n2, n3, (x, y)):
                continue
            val = data[x, y]
            mean += val
            mean_2 += val ** 2
            n += 1
    if n == 0:
        return 0
    mean /= float(n)
    mean_2 /= float(n)
    var = mean_2 - mean ** 2
    V = EXPECTED_WATER_STD_DEV ** 2
    g_u = 1.0 - (n * (mean - EXPECTED_WATER_MEAN) ** 2) / (V * (ALLOWED_DEVIATIONS ** 2))
    P = 1.01 + 0.258 * n
    sigma = var * (1 - 0.509 * math.exp(-0.0744 * n))
    C = 0
    g_v = 0
    #g_v = 1.0 / (ALLOWED_DEVIATIONS ** 2) * (-P * var / sigma +
    #        P * math.log(P * V / sigma) - math.log(V)) + C
    return g_u + g_v

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
def shift_node(data, n1, n2, n3, verbose=False):
    best = n2 
    best_goodness = -float('inf')
    for d in NEIGHBORS:
        n = (n2[0] + d[0], n2[1] + d[1])
        g = get_goodness(data, n1, n, n2)
        if verbose:
            print d, g
        if g > best_goodness:
            best_goodness = g
            best = n
    return best

def shift_nodes(local_image, nodes):
    im = local_image.get_image('hh')

    for loop in nodes:
        for i in range(len(loop)):
            p = i - 1 if i > 0 else len(loop) - 1
            n = i + 1 if i < len(loop) - 1 else 0
            loop[i] = shift_node(im, loop[p], loop[i], loop[n], verbose=i==0)
    return nodes

# insert new nodes if nodes are too far apart
# remove nodes if too close together
def respace_nodes(nodes):
    l = 0
    while l < len(nodes):
        loop = nodes[l]
        # remove loops with two or fewer nodes
        if len(loop) <= 2:
            del nodes[l]
            continue

        # go through nodes in loop
        i = 0
        while i < len(loop):
            n = i + 1 if i < len(loop) - 1 else 0
            dist2 = (loop[i][0] - loop[n][0]) ** 2 + (loop[i][1] - loop[n][1]) ** 2
            # delete node if too close
            if dist2 < MIN_NODE_SEPARATION ** 2:
                del loop[n]
                continue
            # add node if too far
            elif dist2 > MAX_NODE_SEPARATION ** 2:
                mid = ((loop[i][0] + loop[n][0]) / 2, (loop[i][1] + loop[n][1]) / 2)
                loop.insert(i + 1, mid)
                continue
            i += 1
        l += 1
    return nodes

def fix_snake_geometry(nodes):
    return nodes

def initialize_active_contour(domain):
    #local_image = LocalEEImage(domain.image, domain.bbox, 6.174, ['hh', 'hv', 'vv'], 'Radar_' + str(domain.id))
    local_image = LocalEEImage(domain.image, domain.bbox, 100, ['hh', 'hv', 'vv'], 'Radar_' + str(domain.id))
    (w, h) = local_image.size()
    nodes = [[(260, 110), (260, 140), (300, 150), (300, 100)]]
    nodes = respace_nodes(nodes)

    return (local_image, nodes)

def active_contour(domain):
    (local_image, nodes) = initialize_active_contour(domain)
    for i in range(10):
        shift_nodes(local_image, nodes)
    nodes = respace_nodes(nodes)
    nodes = fix_snake_geometry(nodes)

