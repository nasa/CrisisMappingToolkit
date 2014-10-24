import domains

from util.mapclient_qt import centerMap, addToMap
from util.local_ee_image import LocalEEImage

import ee
import math
import PIL
from PIL import ImageQt
from PyQt4 import QtCore, QtGui

MIN_NODE_SEPARATION    =    5
MAX_NODE_SEPARATION    =   15

SEED_REGION_BORDER     =    6

EXPECTED_WATER_MEAN    =  2.7
EXPECTED_WATER_STD_DEV = 0.35
ALLOWED_DEVIATIONS     =  2.5

VARIANCE_C             =  0.0
CURVATURE_GAMMA        =    3
TENSION_LAMBDA         = 0.05

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
    v = (b[0] - a[0]) * (b[1] - x[1]) - (b[1] - a[1]) * (b[0] - x[0])
    return v >= 0

def __line_distance_2(a, b, x):
    v = (x[0] - a[0], x[1] - a[1])
    l = (b[0] - a[0], b[1] - a[1])
    mag = math.sqrt(l[0] * l[0] + l[1] * l[1])
    if mag == 0.0:
        return (a[0] - x[0]) ** 2 + (a[1] - x[1]) ** 2
    l = (l[0] / mag, l[1] / mag)
    n = v[0] * l[0] + v[1] * l[1]
    n = max(0, min(mag, n))
    tx = v[0] - n * l[0]
    ty = v[1] - n * l[1]
    return tx * tx + ty * ty

def __curvature(n1, n2, n3, nn1=None, nn2=None, nn3=None):
    a1 = math.atan2(n1[1] - n2[1], n1[0] - n2[0])
    a2 = math.atan2(n3[1] - n2[1], n3[0] - n2[0])
    if nn1:
        b1 = math.atan2(nn1[1] - n2[1], nn1[0] - n2[0])
        b2 = math.atan2(n3[1] - n2[1], n3[0] - n2[0])
    elif nn2:
        b1 = math.atan2(n1[1] - nn2[1], n1[0] - nn2[0])
        b2 = math.atan2(n3[1] - nn2[1], n3[0] - nn2[0])
    else:
        b1 = math.atan2(n1[1] - n2[1], n1[0] - n2[0])
        b2 = math.atan2(nn3[1] - n2[1], nn3[0] - n2[0])
    change = (b1 - b2) - (a1 - a2)
    while change > math.pi:
        change -= 2 * math.pi
    while change <= -math.pi:
        change += 2 * math.pi

    mid1 = (n1[0] - n2[0], n1[1] - n2[1])
    mid2 = (n3[0] - n2[0], n3[1] - n2[1])
    a = math.sqrt((mid2[0] - mid1[0]) ** 2 + (mid2[1] - mid1[1]) ** 2)
    if a == 0:
        return float('inf')
    return change * change / a

def __tension(n1, n2, n3, n):
    old =  (n2[0] - n1[0]) ** 2 + (n2[1] - n1[1]) ** 2
    old += (n3[0] - n2[0]) ** 2 + (n3[1] - n2[1]) ** 2
    new =  ( n[0] - n1[0]) ** 2 + ( n[1] - n1[1]) ** 2
    new += (n3[0] -  n[0]) ** 2 + (n3[1] -  n[1]) ** 2
    return new - old

def get_goodness(data, bbox, n1, n2, n3):
    mean = 0
    mean_2 = 0
    n = 0
    for x in range(bbox[0], bbox[1]):
        for y in range(bbox[2], bbox[3]):
            #if __line_distance_2(n1, n2, (x, y)) > MAX_LINE_DISTANCE ** 2 and \
            #   __line_distance_2(n2, n3, (x, y)) > MAX_LINE_DISTANCE ** 2:
            #    continue
            acute   = __inside_line(n1, n2, n3)
            # add .5 so we don't get integer effects where a
            # shift of one pixel removes the entire row
            inside1 =  __inside_line(n1, n2, (x + 0.5, y + 0.5))
            inside2 =  __inside_line(n2, n3, (x + 0.5, y + 0.5))
            if acute:
                if not (inside1 and inside2):
                    continue
            else:
                if (not inside1) and (not inside2):
                    continue
            val = math.log10(data[x, y])
            mean += val
            mean_2 += val ** 2
            n += 1
    if n == 0:
        return (0, 0)
    mean /= float(n)
    mean_2 /= float(n)
    var = mean_2 - mean ** 2
    V = EXPECTED_WATER_STD_DEV ** 2
    g_u = 1.0 - ((mean - EXPECTED_WATER_MEAN) ** 2) / (V * (ALLOWED_DEVIATIONS ** 2))
    g_u = max(-1.0, min(1.0, g_u))
    #P = 1.01 + 0.258 * n
    # we have more pixels so use different order function
    P = 1.01 + 0.02 * n
    sigma = V * (1 - 0.509 * math.exp(-0.0744 * n))
    if var == 0.0:
        g_v = 0.0
    else:
        g_v = 1.0 / (ALLOWED_DEVIATIONS ** 2) * (-P * V / sigma +
            P * math.log(P * var / sigma) - math.log(var))
        # for some reason I don't think it's negative in the paper,
        # but it clearly ought to be
        g_v = -g_v + VARIANCE_C
        g_v = max(-1.0, min(1.0, g_v))
    g = (g_u + g_v) / 2.0
    return (n, g)

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
def shift_node(data, loop, i):
    p  = i - 1 if i > 0 else len(loop) - 1
    pp = p - 1 if p > 0 else len(loop) - 1
    n  = i + 1 if i < len(loop) - 1 else 0
    nn = n + 1 if n < len(loop) - 1 else 0
    n1 = loop[p]
    n2 = loop[i]
    n3 = loop[n]
    # use entire lines
    #x_min = max(0,             min(n1[0], n2[0], n3[0]) - SEED_REGION_BORDER)
    #x_max = min(data.shape[0], max(n1[0], n2[0], n3[0]) + SEED_REGION_BORDER)
    #y_min = max(0,             min(n1[1], n2[1], n3[1]) - SEED_REGION_BORDER)
    #y_max = min(data.shape[1], max(n1[1], n2[1], n3[1]) + SEED_REGION_BORDER)
    # use immediate vicinity of node
    x_min = max(0,             n2[0] - SEED_REGION_BORDER)
    x_max = min(data.shape[0], n2[0] + SEED_REGION_BORDER)
    y_min = max(0,             n2[1] - SEED_REGION_BORDER)
    y_max = min(data.shape[1], n2[1] + SEED_REGION_BORDER)
    bbox = (x_min, x_max, y_min, y_max)

    best = n2 
    (original_count, ignored) = get_goodness(data, bbox, n1, n2, n3)
    best_goodness = 0

    for d in NEIGHBORS:
        np = (n2[0] + d[0], n2[1] + d[1])
        if np[0] < 0 or np[0] >= data.shape[0] or np[1] < 0 or np[1] >= data.shape[1]:
            continue
        (count, g) = get_goodness(data, bbox, n1, np, n3)
        curveg = __curvature(n1, n2, n3, nn2=np) + __curvature(loop[pp], n1, n2, nn2=np) + \
                __curvature(n2, n3, loop[nn], nn1=np)
        tensiong = __tension(n1, n2, n3, np)
        fullg = (count - original_count) * g - CURVATURE_GAMMA * curveg - \
                TENSION_LAMBDA * tensiong
        if fullg > best_goodness:
            best_goodness = fullg
            best = np
    return best

def shift_nodes(local_image, nodes):
    im = local_image.get_image('hh')

    for loop in nodes:
        for i in range(len(loop)):
            loop[i] = shift_node(im, loop, i)
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

