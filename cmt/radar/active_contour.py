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

from cmt.local_ee_image import LocalEEImage

import ee

import math

# TODO: Move out of the radar directory?

'''
Active Contour (snake) water detector based on the paper:
    "Flood boundary delineation from Synthetic Aperture Radar imagery using a
     statistical active contour model. M. S. Horritt , D. C. Mason & A. J. Luckman"
     
     
     Use a large set of small adaptive contour objects to trace the boundaries
     of a flood.  The small contours can merge with each other to more cleanly
     span larger sections of water.
'''


def compute_band_statistics(ee_image, classified_image, region):
    '''Use training data to compute the band statistics needed by the active contour algorithm.
       - This version uses a single region and a labeled raster image.'''

    EVAL_RESOLUTION    = 30  # Meters
    ALLOWED_DEVIATIONS = 2.5 # For now this is a constant
    
    masked_image = ee_image.mask(classified_image)
    means        = masked_image.reduceRegion(ee.Reducer.mean(),   region, EVAL_RESOLUTION).getInfo() # These result in lists with one entry per band
    stdDevs      = masked_image.reduceRegion(ee.Reducer.stdDev(), region, EVAL_RESOLUTION).getInfo()
    
    # Pack up the results per-band
    band_statistics = []
    band_names      = []
    for km, ks in zip(means, stdDevs):
        band_statistics.append((means[km], stdDevs[ks], ALLOWED_DEVIATIONS))
        #band_statistics.append((2.7, 0.35, ALLOWED_DEVIATIONS))
        band_names.append(km)

    print('Computed band statistics: ')
    for b, s in zip(band_names, band_statistics):
        print(b + ': ' + str(s))
    
    return (band_names, band_statistics)

def compute_band_statistics_features(ee_image, regionList):
    '''Use training data to compute the band statistics needed by the active contour algorithm.
       This version uses multiple labeled regions.'''

    EVAL_RESOLUTION    = 30  # Meters
    ALLOWED_DEVIATIONS = 2.5 # For now this is a constant
    
    masked_image = ee_image.mask(ee_image)
    meanSums = {}
    stdSums  = {}
    listFormat = regionList.toList(100)
    numInputs  = listFormat.size().getInfo()
    for index in range(numInputs):
        region = listFormat.get(index)
        # Skip land regions (water is 1)
        if region.getInfo()['properties']['classification'] != 1:
            continue
        
        # Need to take the border out of EE format and then put it back in!
        regionBorder = ee.Geometry.LinearRing(region.getInfo()['geometry']['coordinates'])
        
        # These result in lists with one entry per band
        means   = masked_image.reduceRegion(ee.Reducer.mean(),   regionBorder, EVAL_RESOLUTION).getInfo()
        stdDevs = masked_image.reduceRegion(ee.Reducer.stdDev(), regionBorder, EVAL_RESOLUTION).getInfo()
        # Accumulate the mean and standard deviation for each band over the regions
        for km, ks in zip(means, stdDevs):
            meanSums[km] = (meanSums[km]+means[km]  ) if (km in meanSums) else (means[km]  )
            stdSums[ks]  = (stdSums[ks] +stdDevs[ks]) if (ks in stdSums ) else (stdDevs[ks])
    
    # We just take the mean of the statistics across the input regions.
    band_statistics = []
    band_names      = []
    for km, ks in zip(meanSums, stdSums):
        band_statistics.append((meanSums[km]/len(meanSums), stdSums[ks]/len(stdSums), ALLOWED_DEVIATIONS))
        band_names.append(km)

    print('Computed band statistics: ')
    for b, s in zip(band_names, band_statistics):
        print(b + ': ' + str(s))
    
    return (band_names, band_statistics)


class Loop(object):
    MIN_NODE_SEPARATION    =    5 # In pixels
    MAX_NODE_SEPARATION    =   15
    
    SEED_REGION_BORDER     =    2

    # Parameters for contour curvature
    VARIANCE_C             = -0.1
    CURVATURE_GAMMA        =  1.5
    TENSION_LAMBDA         = 0.05
    
    ## These three are used in the "goodness" function below
    ## - These values are for a Mississippi UAVSAR image
    #EXPECTED_WATER_MEAN    =  2.7
    #EXPECTED_WATER_STD_DEV = 0.35
    #ALLOWED_DEVIATIONS     =  2.5
    ### - These values are for a Malawi Skybox image
    ##EXPECTED_WATER_MEAN    = 270
    ##EXPECTED_WATER_STD_DEV = 10
    ##ALLOWED_DEVIATIONS     = 2.5

    # band_statistics needs to be in the same order as the image bands and
    #   contains the following for each band: (mean, stdDev, allowed_deviations)

    def __init__(self, image_data, nodes, band_statistics, image_is_log_10=False):
        self.data = image_data
        # third parameter of node is how long it's been still
        if len(nodes[0]) == 2:
            nodes = map(lambda x: (x[0], x[1], 0), nodes)
        self.nodes     = nodes
        self.clockwise = self._is_clockwise()
        self.done      = False
        self.almost_done_count = 0
        self.band_statistics = band_statistics
        self.image_is_log_10 = image_is_log_10

    def get_image_width(self):
        return self.data.size()[0]
    
    def get_image_height(self):
        return self.data.size()[1]
    
    def get_image_pixel(self, x, y):
        '''Get the image value across all bands at this point'''
        if self.image_is_log_10:
            pixel = self.data.get(x, y)
            value = [math.log10(v) for v in pixel]
        else: # Normal image
            value = self.data.get(x, y)
        return value
    
    
    def _line_segments_intersect(self, a, b, c, d):
        # is counterclockwise order
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        return (ccw(a, c, d) != ccw(b, c, d)) and (ccw(a, b, c) != ccw(a, b, d))

    # count the number of segments (excluding the one starting with the node with ID ignored)
    # which a ray in the direction perp emanating from p crosses
    def _count_intersections(self, p, perp, ignored=None):
        intersections = 0
        for i in range(len(self.nodes)):
            pred = i - 1
            if pred < 0:
                pred = len(self.nodes) - 1
            if pred == ignored: # don't count line segment with origin of ray
                continue
            if self._line_segments_intersect(p, (p[0] + perp[0] * 1e7, p[1] + perp[1] * 1e7), self.nodes[pred], self.nodes[i]):
                intersections += 1
        return intersections

    def _is_clockwise(self):
        s = 0
        for i in range(len(self.nodes)):
            n = (i + 1) if (i < len(self.nodes) - 1) else 0 # TODO: There should be a function for this line!
            s += (self.nodes[n][0] - self.nodes[i][0]) * (self.nodes[n][1] + self.nodes[i][1])
        return s >= 0
    
    def _inside_line(self, a, b, x):
        v = (b[0] - a[0]) * (b[1] - x[1]) - (b[1] - a[1]) * (b[0] - x[0])
        return v >= 0
    
    def _curvature(self, n1, n2, n3, nn1=None, nn2=None, nn3=None):
        a1 = math.atan2(n1[1] - n2[1], n1[0] - n2[0])
        a2 = math.atan2(n3[1] - n2[1], n3[0] - n2[0])
        if nn1:
            b1 = math.atan2(nn1[1] - n2[1], nn1[0] - n2[0])
            b2 = math.atan2( n3[1] - n2[1],  n3[0] - n2[0])
        elif nn2:
            b1 = math.atan2(n1[1] - nn2[1], n1[0] - nn2[0])
            b2 = math.atan2(n3[1] - nn2[1], n3[0] - nn2[0])
        else:
            b1 = math.atan2( n1[1] - n2[1],  n1[0] - n2[0])
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
    
    def _tension(self, n1, n2, n3, n):
        old =  (n2[0] - n1[0]) ** 2 + (n2[1] - n1[1]) ** 2
        old += (n3[0] - n2[0]) ** 2 + (n3[1] - n2[1]) ** 2
        new =  ( n[0] - n1[0]) ** 2 + ( n[1] - n1[1]) ** 2
        new += (n3[0] -  n[0]) ** 2 + (n3[1] -  n[1]) ** 2
        return new - old

    # Cost function for how much pixels inside curve (n1, n2, n3) within bbox look like water
    def _get_goodness(self, bbox, n1, n2, n3): # n1,n2,n3 are three points on the contour

        # These are used to compute if points are inside the contour
        acute   = self._inside_line(n1, n2, n3)
        n21diff = (n2[0] - n1[0], n2[1] - n1[1])
        n32diff = (n3[0] - n2[0], n3[1] - n2[1])

        # Initialize computation storage        
        num_bands     = len(self.get_image_pixel(0, 0))
        band_means   = [0]*num_bands
        band_means_2 = [0]*num_bands
        band_counts  = [0]*num_bands
        
        # Go over everything in bbox, check if it is inside the contour,
        #  and find the mean and standard deviation of those locations.
        for x in range(bbox[0], bbox[1]):
            for y in range(bbox[2], bbox[3]):
                # add .5 so we don't get integer effects where a
                # shift of one pixel removes the entire row
                p = (x + 0.5, y + 0.5)
                # doing this still but optimized more
                # inside1 =  self._inside_line(n1, n2, p)
                # inside2 =  self._inside_line(n2, n3, p)
                inside1 = (n21diff[0] * (n2[1] - p[1]) - n21diff[1] * (n2[0] - p[0])) >= 0
                inside2 = (n32diff[0] * (n3[1] - p[1]) - n32diff[1] * (n3[0] - p[0])) >= 0
                if acute:
                    if not (inside1 and inside2):
                        continue
                else:
                    if (not inside1) and (not inside2):
                        continue
                
                # Get the value at this point
                pixel = self.get_image_pixel(x, y)
                
                # Accumulate mean and std data for each band
                for i in range(0,num_bands):
                    val = pixel[i]
                    band_means[i]   += val
                    band_means_2[i] += val ** 2
                    band_counts[i]  += 1
        
        # Perform goodness calculations for each band
        mean_count = 0
        mean_good  = 0
        for i in range(0,num_bands):

            # Extract parameters for this band
            mean   = band_means[i]
            mean_2 = band_means_2[i]
            n      = band_counts[i]

            (expected_water_mean, expected_water_std_dev, allowed_deviations) = self.band_statistics[i]
            #print str(expected_water_mean) +', ' +  str(expected_water_std_dev)
            
            if n == 0: # No points inside the contour
                return (0, 0)
            
            mean   /= float(n)
            mean_2 /= float(n)
            var = mean_2 - mean ** 2
            if var <= 0.0:
                var = 0.0 # can happen due to precision errors
                
            # Computations from the paper -> Compare std and mean to constant expected values
            V   = expected_water_std_dev ** 2
            g_u = 1.0 - ((mean - expected_water_mean) ** 2) / (V * (allowed_deviations ** 2))
            g_u = max(-1.0, min(1.0, g_u))
            #P = 1.01 + 0.258 * n
            # we have more pixels so use different order function
            P     = 1.01 + 0.02 * n
            sigma = V * (1 - 0.509 * math.exp(-0.0744 * n))
            if var == 0.0:
                g_v = 0.0
            else:
                g_v = 1.0 / (allowed_deviations ** 2) * (-P * V / sigma +
                    P * math.log(P * var / sigma) - math.log(var))
                # for some reason I don't think it's negative in the paper,
                # but it clearly ought to be
                g_v = -g_v + self.VARIANCE_C
                g_v = max(-1.0, min(1.0, g_v))
            g = (g_u + g_v) / 2.0
            
            mean_count += n
            mean_good  += g
            
        # Take the mean of all the band's responses.
        mean_count /= num_bands
        mean_good  /= num_bands
        #print str(mean_count) + ', ' + str(mean_good)
        
        return (mean_count, mean_good)

    NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
                 ( 0, -1),          ( 0, 1),
                 ( 1, -1), ( 1, 0), ( 1, 1)]
    # shift a single node to neighboring pixel which reduces cost function the most
    def _shift_node(self, i):
        n2 = self.nodes[i]
        if n2[2] > 5:
            return n2
        p     = (i - 1) if (i > 0)                   else (len(self.nodes) - 1)
        pp    = (p - 1) if (p > 0)                   else (len(self.nodes) - 1)
        n     = (i + 1) if (i < len(self.nodes) - 1) else 0
        nn    = (n + 1) if (n < len(self.nodes) - 1) else 0
        n1    = self.nodes[p]
        n3    = self.nodes[n]
        # use immediate vicinity of node
        x_min = max(0,                       n2[0] - self.SEED_REGION_BORDER)
        x_max = min(self.get_image_width(),  n2[0] + self.SEED_REGION_BORDER)
        y_min = max(0,                       n2[1] - self.SEED_REGION_BORDER)
        y_max = min(self.get_image_height(), n2[1] + self.SEED_REGION_BORDER)
        bbox = (x_min, x_max, y_min, y_max)
    
        best = n2 
        (original_count, ignored) = self._get_goodness(bbox, n1, n2, n3)
        best_goodness = 0
    
        for d in self.NEIGHBORS:
            np = (n2[0] + d[0], n2[1] + d[1])
            # don't move outside image
            if ( (np[0] < 0) or (np[0] >= self.get_image_width() ) or
                 (np[1] < 0) or (np[1] >= self.get_image_height())   ):
                continue
            # find how similar pixels inside curve are to expected water distribution
            (count, g) = self._get_goodness(bbox, n1, np, n3)
            # penalty for curving sharply
            curveg = self._curvature(n1,             n2, n3,             nn2=np) + \
                     self._curvature(self.nodes[pp], n1, n2,             nn2=np) + \
                     self._curvature(n2,             n3, self.nodes[nn], nn1=np)
            # encourage nodes to stay the right distance apart
            tensiong = self._tension(n1, n2, n3, np)
            fullg = (count - original_count) * g - self.CURVATURE_GAMMA * curveg - \
                    self.TENSION_LAMBDA * tensiong
            if fullg > best_goodness:
                best_goodness = fullg
                best          = np
        if best_goodness == 0:
            return (n2[0], n2[1], n2[2] + 1)
        else:
            return (best[0], best[1], 0)

    # shift all nodes in loop to neighboring pixel of lowest cost
    def shift_nodes(self):
        if self.done:
            return
        self.moving_count = 0
        for i in range(len(self.nodes)):
            self.nodes[i] = self._shift_node(i)
            # this node updated, neighboring nodes should too
            if self.nodes[i][2] == 0:
                p = (i - 1) if (i > 0)                   else (len(self.nodes) - 1)
                n = (i + 1) if (i < len(self.nodes) - 1) else 0
                self.nodes[p] = (self.nodes[p][0], self.nodes[p][1], 0)
                self.nodes[n] = (self.nodes[n][0], self.nodes[n][1], 0)
                self.moving_count += 1
        if self.moving_count <= 4 or float(self.moving_count) / len(self.nodes) < 0.05:
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
            n     = (i + 1) if (i < len(self.nodes) - 1) else 0
            dist2 = (self.nodes[i][0] - self.nodes[n][0]) ** 2 + (self.nodes[i][1] - self.nodes[n][1]) ** 2
            # delete node if too close
            if dist2 < self.MIN_NODE_SEPARATION ** 2:
                # update neighbors
                if len(self.nodes) <= 2:
                    break
                self.nodes[i] = (self.nodes[i][0], self.nodes[i][1], 0)
                del self.nodes[n]
                n = n if (n < len(self.nodes)) else 0 # last might have been deleted
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

    # recursively create new loops based on intersections, in loop between
    # loop_start and loop_end
    # includes loop_start but not loop_end
    def _create_loops(self, intersections, loop_start, loop_end):
        assert not (loop_end < loop_start and loop_end != 0)
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
            # find next intersection
            for (prev1, prev2) in intersections:
                if lind(prev1) < closest and prev1 >= i:
                    closest       = lind(prev1)
                    closest_other = (prev2 + 1) if (prev2 < lind(loop_end-1)) else loop_end
                if lind(prev2) < closest and prev2 >= i and prev1 >= i:# ignore loops in wrong order
                    closest       = lind(prev2)
                    closest_other = (prev1 + 1) if (prev1 < lind(loop_end-1)) else loop_end
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
            # recursively create new loops within next intersections
            new_loops = self._create_loops(intersections, closest_next, closest_other)
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
        return [Loop(self.data, cur_loop, self.band_statistics, self.image_is_log_10)] + all_loops

    def _inside_loop(self, loop):
        start = loop.nodes[0]
        return self._count_intersections(start, (1, 0)) % 2 == 1

    # eliminate self loops
    def _filter_loops(self, loops):
        if len(loops) == 0:
            return []
        # find biggest loop with our own orientation
        biggest_length = -1
        biggest_loop   = -1
        for i in range(len(loops)):
            if loops[i].clockwise == self.clockwise and len(loops[i].nodes) > biggest_length:
                biggest_length = len(loops[i].nodes)
                biggest_loop   = i
        accepted_loops = [loops[biggest_loop]]
        for i in range(len(loops)):
            if i == biggest_loop:
                continue
            inside = loops[biggest_loop]._inside_loop(loops[i])
            same_orientation = (self.clockwise == loops[i].clockwise)
            if inside and same_orientation:
                continue
            if (not inside) and (not same_orientation):
                continue
            accepted_loops.append(loops[i])
        return accepted_loops

    # returns any new loops that split off
    def fix_self_intersections(self):
        if self.done:
            return [self]
        # kill self if empty
        if len(self.nodes) <= 3:
            return []
        # kill self if collapsed and switched orientation
        if self._is_clockwise() != self.clockwise:
            return []
        # find self intersections
        self_intersections = []
        for i in range(len(self.nodes)):
            cur1   = self.nodes[i]
            prev_i = (i - 1) if (i > 0) else (len(self.nodes) - 1)
            prev1  = self.nodes[prev_i]
            for j in range(i+1, len(self.nodes)):
                prev_j = (j - 1) if (j > 0) else (len(self.nodes) - 1)
                if (j == i) or (j == prev_i) or (prev_j == i):
                    continue
                cur2  = self.nodes[j]
                prev2 = self.nodes[prev_j]
                if not self._line_segments_intersect(prev1, cur1, prev2, cur2):
                    continue
                self_intersections.append((prev_i, prev_j))
        # recursively divide loops based on self intersections, e.g., merge when
        # surrounding island from two sides and meeting
        if len(self_intersections) != 0:
            loops = self._create_loops(self_intersections, 0, 0)
            return self._filter_loops(loops)
        return [self]

    # merge two loops if they intersect
    def merge(self, other):
        intersection = None
        # find intersections
        for i in range(len(self.nodes)):
            cur1   = self.nodes[i]
            prev_i = (i - 1) if (i > 0) else (len(self.nodes) - 1)
            prev1  = self.nodes[prev_i]
            for j in range(len(other.nodes)):
                prev_j = (j - 1) if (j > 0) else (len(other.nodes) - 1)
                cur2   = other.nodes[j]
                prev2  = other.nodes[prev_j]
                if not self._line_segments_intersect(prev1, cur1, prev2, cur2):
                    continue
                intersection = (i, j)
                break
        if intersection == None:
            return None
        (i, j) = intersection
        loop   = self.nodes[0:i] + other.nodes[j:] + other.nodes[0:j] + self.nodes[i:]
        assert len(loop) == len(self.nodes) + len(other.nodes)
        return Loop(self.data, loop, self.band_statistics, self.image_is_log_10)

# an active contour, which is composed of a number of Loops
class Snake(object):
    def __init__(self, local_image, initial_nodes, band_statistics, image_is_log_10=False):
        self.local_image = local_image
        self.data = local_image # Using all bands
        
        # numpy is slower than tuples
        #initial_nodes = map(lambda x: map(lambda y: np.array(y), x), initial_nodes)
        self.loops = [Loop(self.data, l, band_statistics, image_is_log_10) for l in initial_nodes]
        self.done  = False

    def shift_nodes(self):
        if self.done:
            return
        self.done = True
        for loop in self.loops:
            loop.shift_nodes()
            if not loop.done:
                self.done = False
        if self.done:
            print('Done!')
    
    # insert new nodes if nodes are too far apart
    # remove nodes if too close together
    def respace_nodes(self):
        for l in self.loops:
            l.respace_nodes()
    
    # remove self loops, merge intersecting loops
    def fix_geometry(self):
        # fix self loops
        new_loops = []
        for l in self.loops:
            new_loops.extend(l.fix_self_intersections())
        self.loops = new_loops

        # merge intersecting loops
        i = 0
        while i < len(self.loops):
            j = i + 1
            while j < len(self.loops):
                merged = self.loops[i].merge(self.loops[j])
                if merged == None:
                    j += 1
                    continue
                else:
                    merged = merged.fix_self_intersections()
                    del self.loops[j]
                    self.loops[i:i+1] = merged
                    # faster way to do this?
                    if len(merged) > 1:
                        i -= 1
                        break
            i += 1


    # first is features to paint, second is features to unpaint
    def to_ee_feature_collections(self):
        # currently only supports single unfilled region inside filled region
        exterior = []
        interior = []
        for l in self.loops:
            coords = map(lambda x: self.local_image.image_to_global(x[0], x[1]), l.nodes)
            f      = ee.Feature.Polygon(coords)
            if not l.clockwise:
                interior.append(f)
            else:
                exterior.append(f)
        return (ee.FeatureCollection(exterior), ee.FeatureCollection(interior))

    def to_ee_image(self):
        (exterior, interior) = self.to_ee_feature_collections()
        return ee.Image(0).toByte().select(['constant'], ['b1']).paint(exterior, 1).paint(interior, 0)

def initialize_active_contour(domain, ee_image, band_statistics, image_is_log_10=False):
    '''Initialize a Snake class on an input image'''

    scale_meters = 25 # TODO: Make this a parameter?
    # TODO: Make initial loop sizes a parameter
    
    # Get the image we will be working on
    # - All bands will be used so only pass in the bands you want used!
    band_entries = ee_image.getInfo()['bands']
    band_names   = [b['id'] for b in band_entries]
    #print 'Running active contour on bands: ' + str(band_names)
    local_image = LocalEEImage(ee_image, domain.bbox, scale_meters, band_names, 'ActiveContour_' + str(domain.name))
    (w, h) = local_image.size()
    
    # Initialize the algorithm with a grid of small loops that cover the region of interest
    B              = min(w / scale_meters, h / scale_meters)
    CELL_SIZE      = 20 # This is the initial loop size
    loops = []
    for i in range(B, w - B, CELL_SIZE):
        for j in range(B, h - B, CELL_SIZE):
            nextj = min(j + CELL_SIZE, h - B)
            nexti = min(i + CELL_SIZE, w - B)
            loops.append([(i, j), (i, nextj), (nexti, nextj), (nexti, j)])
    # Finished setting up the loops, load them into the Snake class.
    s = Snake(local_image, loops, band_statistics, image_is_log_10)
    s.respace_nodes()

    return (local_image, s) # Ready to go!

MAX_STEPS = 10000

def active_contour(domain):
    '''Start up an active contour and process it until it finishes'''

    train_domain   = domain.training_domain   # Since this is radar data, an earlier date is probably not available.
    sensor         = domain.get_radar()
    train_sensor   = train_domain.get_radar()
    detect_channel = domain.algorithm_params['water_detect_radar_channel']
    ee_image       = sensor.image.select([detect_channel]).toUint16()
    train_ee_image = train_sensor.image.select([detect_channel]).toUint16()
    if sensor.log_scale:
        statisics_image = train_ee_image.log10()
    else:
        statisics_image = train_ee_image
    (band_names, band_statistics) = compute_band_statistics(statisics_image, train_domain.ground_truth, train_domain.bounds)
    
    (local_image, snake) = initialize_active_contour(domain, ee_image, band_statistics, sensor.log_scale)
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


#==========================================================================================

# Specialized version of this call for Skybox data 
def active_countour_skybox(domain, modis_indices):
    '''Special Active Contour radar function wrapper to work with Skybox images'''
    
    # Currently the modis data is ignored when running this
    
    train_domain = domain.training_domain # For skybox data there is probably no earlier image to train off of
    try: # The Skybox data can be in one of two names
        sensor      = domain.skybox
        trainSensor = train_domain.skybox
    except:
        sensor      = domain.skybox_nir
        trainSensor = train_domain.skybox_nir
    ee_image       = sensor.image.toUint16()  # For Skybox, these are almost certainly the same image.
    ee_image_train = trainSensor.image.toUint16()
    
    if train_domain.training_features: # Train using features
        (band_names, band_statistics)  = compute_band_statistics_features(ee_image_train, train_domain.training_features)
    else: # Train using training truth
        (band_names, band_statistics)  = compute_band_statistics(ee_image_train, train_domain.ground_truth, train_domain.bounds())
    
    (local_image, snake) = initialize_active_contour(domain, ee_image, band_statistics, False)

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
