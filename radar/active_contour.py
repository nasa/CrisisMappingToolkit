from util.mapclient_qt import centerMap, addToMap
from util.local_ee_image import LocalEEImage

import ee

import math

class Loop(object):
    MIN_NODE_SEPARATION    =    5
    MAX_NODE_SEPARATION    =   15
    
    SEED_REGION_BORDER     =    4
    
    EXPECTED_WATER_MEAN    =  2.7
    EXPECTED_WATER_STD_DEV = 0.35
    ALLOWED_DEVIATIONS     =  2.5

    VARIANCE_C             =  -0.1
    CURVATURE_GAMMA        =    1.5
    TENSION_LAMBDA         = 0.05

    def __init__(self, image_data, nodes):
        self.data = image_data
        # third parameter of node is how long it's been still
        if len(nodes[0]) == 2:
            nodes = map(lambda x: (x[0], x[1], 0), nodes)
        self.nodes = nodes
        self.clockwise = self.__is_clockwise()
        self.done = False
        self.almost_done_count = 0

    
    def __line_segments_intersect(self, a, b, c, d):
        # is counterclockwise order
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

    # count the number of segments (excluding the one starting with the node with ID ignored)
    # which a ray in the direction perp emanating from p crosses
    def __count_intersections(self, p, perp, ignored=None):
        intersections = 0
        for i in range(len(self.nodes)):
            pred = i - 1
            if pred < 0:
                pred = len(self.nodes) - 1
            if pred == ignored: # don't count line segment with origin of ray
                continue
            if self.__line_segments_intersect(p, (p[0] + perp[0] * 1e7, p[1] + perp[1] * 1e7), self.nodes[pred], self.nodes[i]):
                intersections += 1
        return intersections

    def __is_clockwise(self):
        a = self.nodes[0]
        b = self.nodes[1]
        diff = (b[0] - a[0], b[1] - a[1])
        mid = (a[0] + diff[0] / 2, a[1] + diff[1] / 2)
        perp = (diff[1], -diff[0])
        # ray through polygon interior intersects an odd number of segments
        count = self.__count_intersections(mid, perp, 0)
        return count % 2 == 1
    
    def __inside_line(self, a, b, x):
        v = (b[0] - a[0]) * (b[1] - x[1]) - (b[1] - a[1]) * (b[0] - x[0])
        return v >= 0
    
    def __curvature(self, n1, n2, n3, nn1=None, nn2=None, nn3=None):
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
    
    def __tension(self, n1, n2, n3, n):
        old =  (n2[0] - n1[0]) ** 2 + (n2[1] - n1[1]) ** 2
        old += (n3[0] - n2[0]) ** 2 + (n3[1] - n2[1]) ** 2
        new =  ( n[0] - n1[0]) ** 2 + ( n[1] - n1[1]) ** 2
        new += (n3[0] -  n[0]) ** 2 + (n3[1] -  n[1]) ** 2
        return new - old

    def __get_goodness(self, bbox, n1, n2, n3):
        mean = 0
        mean_2 = 0
        n = 0
        for x in range(bbox[0], bbox[1]):
            for y in range(bbox[2], bbox[3]):
                acute   = self.__inside_line(n1, n2, n3)
                # add .5 so we don't get integer effects where a
                # shift of one pixel removes the entire row
                inside1 =  self.__inside_line(n1, n2, (x + 0.5, y + 0.5))
                inside2 =  self.__inside_line(n2, n3, (x + 0.5, y + 0.5))
                if acute:
                    if not (inside1 and inside2):
                        continue
                else:
                    if (not inside1) and (not inside2):
                        continue
                val = math.log10(self.data[x, y])
                mean += val
                mean_2 += val ** 2
                n += 1
        if n == 0:
            return (0, 0)
        mean /= float(n)
        mean_2 /= float(n)
        var = mean_2 - mean ** 2
        if var <= 0.0:
            var = 0.0 # can happen due to precision errors
        V = self.EXPECTED_WATER_STD_DEV ** 2
        g_u = 1.0 - ((mean - self.EXPECTED_WATER_MEAN) ** 2) / (V * (self.ALLOWED_DEVIATIONS ** 2))
        g_u = max(-1.0, min(1.0, g_u))
        #P = 1.01 + 0.258 * n
        # we have more pixels so use different order function
        P = 1.01 + 0.02 * n
        sigma = V * (1 - 0.509 * math.exp(-0.0744 * n))
        if var == 0.0:
            g_v = 0.0
        else:
            g_v = 1.0 / (self.ALLOWED_DEVIATIONS ** 2) * (-P * V / sigma +
                P * math.log(P * var / sigma) - math.log(var))
            # for some reason I don't think it's negative in the paper,
            # but it clearly ought to be
            g_v = -g_v + self.VARIANCE_C
            g_v = max(-1.0, min(1.0, g_v))
        g = (g_u + g_v) / 2.0
        return (n, g)

    NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    def __shift_node(self, i):
        n2 = self.nodes[i]
        if n2[2] > 5:
            return n2
        p  = i - 1 if i > 0 else len(self.nodes) - 1
        pp = p - 1 if p > 0 else len(self.nodes) - 1
        n  = i + 1 if i < len(self.nodes) - 1 else 0
        nn = n + 1 if n < len(self.nodes) - 1 else 0
        n1 = self.nodes[p]
        n3 = self.nodes[n]
        # use immediate vicinity of node
        x_min = max(0,                  n2[0] - self.SEED_REGION_BORDER)
        x_max = min(self.data.shape[0], n2[0] + self.SEED_REGION_BORDER)
        y_min = max(0,                  n2[1] - self.SEED_REGION_BORDER)
        y_max = min(self.data.shape[1], n2[1] + self.SEED_REGION_BORDER)
        bbox = (x_min, x_max, y_min, y_max)
    
        best = n2 
        (original_count, ignored) = self.__get_goodness(bbox, n1, n2, n3)
        best_goodness = 0
    
        for d in self.NEIGHBORS:
            np = (n2[0] + d[0], n2[1] + d[1])
            if np[0] < 0 or np[0] >= self.data.shape[0] or np[1] < 0 or np[1] >= self.data.shape[1]:
                continue
            (count, g) = self.__get_goodness(bbox, n1, np, n3)
            curveg = self.__curvature(n1, n2, n3, nn2=np) + self.__curvature(self.nodes[pp], n1, n2, nn2=np) + \
                    self.__curvature(n2, n3, self.nodes[nn], nn1=np)
            tensiong = self.__tension(n1, n2, n3, np)
            fullg = (count - original_count) * g - self.CURVATURE_GAMMA * curveg - \
                    self.TENSION_LAMBDA * tensiong
            if fullg > best_goodness:
                best_goodness = fullg
                best = np
        if best_goodness == 0:
            return (n2[0], n2[1], n2[2] + 1)
        else:
            return (best[0], best[1], 0)

    def shift_nodes(self):
        if self.done:
            return
        self.moving_count = 0
        for i in range(len(self.nodes)):
            self.nodes[i] = self.__shift_node(i)
            # this node updated, neighboring nodes should too
            if self.nodes[i][2] == 0:
                p = i - 1 if i > 0 else len(self.nodes) - 1
                n = i + 1 if i < len(self.nodes) - 1 else 0
                self.nodes[p] = (self.nodes[p][0], self.nodes[p][1], 0)
                self.nodes[n] = (self.nodes[n][0], self.nodes[n][1], 0)
                self.moving_count += 1
        if self.moving_count <= 4 or float(self.moving_count) / len(self.nodes) < 0.01:
            self.almost_done_count += 1
        else:
            self.almost_done_count = 0
        # just mark it as done after a while of small oscillations
        if self.almost_done_count >= 50:
            self.done = True
    
    # insert new nodes if nodes are too far apart
    # remove nodes if too close together
    def respace_nodes(self):
        if self.done:
            return
        # go through nodes in loop
        i = 0
        while i < len(self.nodes):
            n = i + 1 if i < len(self.nodes) - 1 else 0
            dist2 = (self.nodes[i][0] - self.nodes[n][0]) ** 2 + (self.nodes[i][1] - self.nodes[n][1]) ** 2
            # delete node if too close
            if dist2 < self.MIN_NODE_SEPARATION ** 2:
                del self.nodes[n]
                p = i - 1 if i > 0 else len(self.nodes) - 1
                # update neighbors
                self.nodes[p] = (self.nodes[p][0], self.nodes[p][1], 0)
                n = n if n < len(self.nodes) else 0 # last might have been deleted
                self.nodes[n] = (self.nodes[n][0], self.nodes[n][1], 0)
                continue
            # add node if too far
            elif dist2 > self.MAX_NODE_SEPARATION ** 2:
                mid = ((self.nodes[i][0] + self.nodes[n][0]) / 2, (self.nodes[i][1] + self.nodes[n][1]) / 2, 0)
                # update neighbors
                self.nodes[i] = (self.nodes[i][0], self.nodes[i][1], 0)
                self.nodes[n] = (self.nodes[n][0], self.nodes[n][1], 0)
                self.nodes.insert(i + 1, mid)
                continue
            i += 1

    # includes loop_start but not loop_end
    def __create_loops(self, intersections, loop_start, loop_end):
        i = loop_start
        cur_loop = []
        all_loops = []
        def lind(n):
            return n if n >= loop_start else n + len(self.nodes)
        split_before = False
        # find next intersection
        while True:
            initial_length = len(cur_loop)
            closest = lind(loop_end-1) # don't include new loop where prev = loop_end
            closest_other = None
            for (prev1, prev2) in intersections:
                if lind(prev1) < closest and prev1 >= i:
                    closest = lind(prev1)
                    closest_other = prev2 + 1 if prev2 < lind(loop_end-1) else loop_end
                if lind(prev2) < closest and prev2 >= i and lind(prev1) >= i:# ignore loops in wrong order
                    closest = lind(prev2)
                    closest_other = prev1 + 1 if prev1 < lind(loop_end-1) else loop_end
            closest_next = closest + 1
            # extend loop with passed nodes
            if closest_next >= len(self.nodes):
                closest_next -= len(self.nodes)
                cur_loop.extend(self.nodes[i:])
                closest -= len(self.nodes)
                cur_loop.extend(self.nodes[0:closest_next])
            else:
                cur_loop.extend(self.nodes[i:closest_next])
            if closest_other == None:
                if split_before:
                    cur_loop[0] = (cur_loop[0][0], cur_loop[0][1], 0)
                break
            new_loops = self.__create_loops(intersections, closest_next, closest_other)
            # update neighbors that had changed connectivity
            cur_loop[initial_length] = (cur_loop[initial_length][0], cur_loop[initial_length][1], 0)
            cur_loop[-1] = (cur_loop[-1][0], cur_loop[-1][1], 0)
            split_before = True
            all_loops.extend(new_loops)
            i = closest_other
            if i == loop_end:
                break
        if len(cur_loop) <= 2:
            return all_loops
        return [Loop(self.data, cur_loop)] + all_loops

    def __inside_loop(self, loop):
        start = loop.nodes[0]
        return self.__count_intersections(start, (1, 0)) % 2 == 1

    def __filter_loops(self, loops):
        # find biggest loop with our own orientation
        biggest_length = -1
        biggest_loop = -1
        for i in range(len(loops)):
            if loops[i].clockwise == self.clockwise and len(loops[i].nodes) > biggest_length:
                biggest_length = len(loops[i].nodes)
                biggest_loop = i
        accepted_loops = [loops[biggest_loop]]
        for i in range(len(loops)):
            if i == biggest_loop:
                continue
            inside = loops[biggest_loop].__inside_loop(loops[i])
            same_orientation = (self.clockwise == loops[i].clockwise)
            if inside and same_orientation:
                continue
            if (not inside) and (not same_orientation):
                continue
            accepted_loops.append(loops[i])
        return accepted_loops

    # returns any new loops that split off
    def fix_self_intersections(self):
        if len(self.nodes) <= 3:
            return []
        self_intersections = []
        for i in range(len(self.nodes)):
            cur1 = self.nodes[i]
            prev_i = i - 1 if i > 0 else len(self.nodes) - 1
            prev1 = self.nodes[prev_i]
            for j in range(i+1, len(self.nodes)):
                prev_j = j - 1 if j > 0 else len(self.nodes) - 1
                if j == i or j == prev_i or prev_j == i:
                    continue
                cur2 = self.nodes[j]
                prev2 = self.nodes[prev_j]
                if not self.__line_segments_intersect(prev1, cur1, prev2, cur2):
                    continue
                self_intersections.append((prev_i, prev_j))
        if len(self_intersections) != 0:
            loops = self.__create_loops(self_intersections, 0, 0)
            return self.__filter_loops(loops)
        return [self]


class Snake(object):
    def __init__(self, local_image, initial_nodes):
        self.local_image = local_image
        self.data = local_image.get_image('hh')
        # numpy is slower than tuples
        #initial_nodes = map(lambda x: map(lambda y: np.array(y), x), initial_nodes)
        self.loops = [Loop(self.data, l) for l in initial_nodes]
        self.done = False

    def shift_nodes(self):
        if self.done:
            return
        self.done = True
        for loop in self.loops:
            loop.shift_nodes()
            if not loop.done:
                self.done = False
        if self.done:
            print 'Done!'
    
    # insert new nodes if nodes are too far apart
    # remove nodes if too close together
    def respace_nodes(self):
        for l in self.loops:
            l.respace_nodes()
    
    # remove self loops, merge intersecting loops
    def fix_geometry(self):
        new_loops = []
        for l in self.loops:
            new_loops.extend(l.fix_self_intersections())
        self.loops = new_loops

    # first is features to paint, second is features to unpaint
    def to_ee_feature_collections(self):
        # currently only supports single unfilled region inside filled region
        exterior = []
        interior = []
        for l in self.loops:
            coords = map(lambda x: self.local_image.image_to_global(x[0], x[1]), l.nodes)
            print coords
            f = ee.Feature.Polygon(coords)
            if not l.clockwise:
                interior.append(f)
            else:
                exterior.append(f)
        return (ee.FeatureCollection(exterior), ee.FeatureCollection(interior))

    def to_ee_image(self):
        (exterior, interior) = self.to_ee_feature_collections()
        return ee.Image(0).toByte().select(['constant'], ['b1']).paint(exterior, 1).paint(interior, 0)


def initialize_active_contour(domain):
    #local_image = LocalEEImage(domain.image, domain.bbox, 6.174, ['hh', 'hv', 'vv'], 'Radar_' + str(domain.id))
    local_image = LocalEEImage(domain.image, domain.bbox, 100, ['hh', 'hv', 'vv'], 'Radar_' + str(domain.id))
    (w, h) = local_image.size()
    # s = Snake([[(260, 110), (260, 140), (300, 150), (300, 100)]])
    B = 10
    s = Snake(local_image, [[(B, B), (B, h - B), (w - B, h - B), (w - B, 3 * B)]])
    s.respace_nodes()

    return (local_image, s)

MAX_STEPS = 10000

def active_contour(domain):
    (local_image, snake) = initialize_active_contour(domain)
    for i in range(MAX_STEPS):
        if i % 10 == 0:
            snake.respace_nodes()
            snake.shift_nodes() # shift before fixing geometry since reversal of orientation possible
            snake.fix_geometry()
        else:
            snake.shift_nodes()
        if snake.done:
            break
    return snake.to_ee_image().clip(domain.bounds)

