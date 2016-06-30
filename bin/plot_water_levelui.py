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

import sys
import os
import os.path
import glob
import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy
import csv
from itertools import izip

'''
Draws a graph of the output from "lake_measure.py"

To use, pass in the output file from that tool.
'''


def parse_lake_results(name, startdate, enddate):
    f = open(name, 'r')

    x_axis = []
    y_axis = []
    cloud_axis = []

    startdate_parts = startdate.split('-')
    startdate = datetime.date(int(startdate_parts[0]), int(startdate_parts[1]), int(startdate_parts[2]))

    enddate_parts = enddate.split('-')
    enddate = datetime.date(int(enddate_parts[0]), int(enddate_parts[1]), int(enddate_parts[2]))

    f.readline()
    parts = f.readline().split(',')
    names = parts[0]
    country = parts[1]
    # Dynamic cloud pixel thresholding.
    area = float(parts[2].replace('\n', ''))
    pixel_area = area/.03/.03
    cloud_pix_threshold = pixel_area*.002475
    f.readline()

    for l in f:
        parts = l.split(',')
        date_parts = parts[0].split('-')
        date = datetime.date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
        if date < startdate or date > enddate:
            continue
        satellite = parts[1]
        cloud = int(parts[2])
        water = int(parts[3])
        # take values with low cloud cover
        if cloud < cloud_pix_threshold and water > 0:
            x_axis.append(date)
            y_axis.append(water * 0.03 * 0.03)  # pixels * km^2 / pixel, km^2 / pixel = 0.03 * 0.03 / 1
            cloud_axis.append(cloud * 0.03 * 0.03)

    # Error-catcher for situation where a date range is selected and no good points are available for plotting.
    if len(y_axis) < 3:
        features = False
        dates = False
        water = False
        clouds = False
        return (features, dates, water, clouds)

    # Sorts data so that data points are in order of date then satellite, not vice-versa. Only needed if we want to use
    # Landsat 7 data.
    x_sorter = x_axis
    y_axis = [y_axis for (x_sorter, y_axis) in sorted(zip(x_sorter, y_axis), key = lambda pair: pair[0])]
    x_axis = sorted(x_axis)
    f.close()

    # Remove values that differ from neighbors by large amounts
    NEIGHBOR_RADIUS = 3
    OUTLIER_FACTOR = 0.995#Was 0.98
    remove = []
    for i in range(len(y_axis)):
        start = max(0, i - NEIGHBOR_RADIUS)
        end = min(len(y_axis), i + NEIGHBOR_RADIUS)
        if i > 0:
            neighbors = y_axis[start:i-1]
        else:
            neighbors = []
        if i < len(y_axis) - 1:
            neighbors.extend(y_axis[i+1:end])
        num_neighbors = end - start - 1
        num_outliers = 0
        for v in neighbors:
            if (v < y_axis[i] * OUTLIER_FACTOR) or (v > y_axis[i] / OUTLIER_FACTOR):
                num_outliers += 1
        if (num_neighbors == 0) or (float(num_outliers) / num_neighbors >= 0.5):
            remove.append(i)

    for i in reversed(remove):
        y_axis.pop(i)
        cloud_axis.pop(i)
        x_axis.pop(i)

    results = dict()
    results['name'] = names
    results['country'] = country
    results['area'] = str(area)
    return (results, x_axis, y_axis, cloud_axis)

def plot_results(features, dates, water, clouds, save_directory=None, ground_truth_file=None):
    fig, ax = plt.subplots()
    water_line = ax.plot(dates, water, linestyle='-', color='b', linewidth=1,
                         label='Landsat-Generated Surface Area')
    ax.plot(dates, water, 'gs', ms=3)
    # ax.bar(dates, water, color='b', width=15, linewidth=0)
    # ax.bar(dates, clouds, bottom=water, color='r', width=15, linewidth=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlabel('Time')
    ax.format_xdata = mdates.DateFormatter('%m/%d/%Y')

    if ground_truth_file is not None:
        (ground_truth_dates, ground_truth_levels) = load_ground_truth(ground_truth_file)
        ax2 = ax.twinx()
        ground_truth_line = ax2.plot(ground_truth_dates, ground_truth_levels, linestyle='--', color='r', linewidth=2, label='Measured Elevation')
        ax2.set_ylabel('Lake Elevation (ft)')
        ax2.format_ydata = (lambda x: '%g ft' % (x))
        ax2.set_ylim([6372, 6385.5])

    def onpick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        print 'onpick points:', zip(xdata[ind], ydata[ind])
    fig.canvas.mpl_connect('pick_event', onpick)

    ax.format_ydata = (lambda x: '%g km^2' % (x))
    ax.set_ylabel('Lake Surface Area (km^2)')
    fig.suptitle(features['name'] + ' Surface Area from Landsat')
    lns = water_line# + ground_truth_line
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=4)

    ax.grid(True)
    fig.autofmt_xdate()

    if save_directory is not None:
        fig.savefig(os.path.join(save_directory, features['name'] + '.pdf'))


def load_ground_truth(filename):
    f = open(filename, 'r')
    dates = []
    levels = []
    all_months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                  'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    for line in f:
        parts = line.split()
        month = all_months[parts[0].split('-')[0]]
        year = int(parts[0].split('-')[1])
        if year > 50:
            year = 1900 + year
        else:
            year = 2000 + year
        dates.append(datetime.datetime(year, month, 1))
        levels.append(float(parts[1]))
    return (dates, levels)

# --- Main script ---
def table_water_level(lake, startdate, enddate, result_dir, output_file=None):
    # Grabs lake names from .txt files in the results folder.
    lakes = [i.split('.')[0] for i in glob.glob1(result_dir,'*txt')]

    # Compares lake names found from .txt files with the chosen lake. If a match is found, the parser is run.
    if lake in lakes:
        lake_dir = result_dir + os.sep + lake + '.txt'
        (features, dates, water, clouds) = parse_lake_results(lake_dir, startdate, enddate)

        # Error-catcher for situation where a date range is selected and no good points are available for plotting.
        if water == False:
            print "No good data points found in selected date range. Please try a larger date range and retry."
        # Table creating and saving block:
        else:
            # Error catching for invalid directories.
            if output_file == None:
                output_file = result_dir + '/' + lake + '.csv'
            with open(output_file, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Area (km^2)"])
                writer.writerows(izip(dates, water))

def plot_water_level(lake, startdate, enddate, result_dir):
    # Grabs lake names from .txt files in the results folder.
    lakes = [i.split('.')[0] for i in glob.glob1(result_dir,'*txt')]

    # Compares lake names found from .txt files with the chosen lake. If a match is found, the parser is run.
    if lake in lakes:
        lake = result_dir + os.sep + lake + '.txt'
        (features, dates, water, clouds) = parse_lake_results(lake, startdate, enddate)

        # Error-catcher for situation where a date range is selected and no good points are available for plotting.
        if water == False:
            print "No good data points found in selected date range. Please try a larger date range."
        # plot_results(features, dates, water, clouds, None, 'results/mono_lake_elevation.txt')
        else:
            plot_results(features, dates, water, clouds)
            plt.show()

    # Notifies user if the data file for the selected lake has not been generated yet.
    else:
        print "Specified lake data file not found. Please retrieve data and try again."

