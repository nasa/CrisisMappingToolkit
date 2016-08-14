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
import os.path
import glob
import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy

'''
Draws a graph of the output from "lake_measure.py"

To use, pass in the output file from that tool.
'''


def parse_lake_results(name):
    f = open(name, 'r')

    x_axis = []
    y_axis = []
    cloud_axis = []

    f.readline()
    parts = f.readline().split(',')
    names = parts[0]
    country = parts[1]
    area = parts[2]
    f.readline()
    for l in f:
        parts = l.split(',')
        date_parts = parts[0].split('-')
        date = datetime.date(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]))
        satellite = parts[1]
        cloud = int(parts[2])
        water = int(parts[3])
        # take values with low cloud cover
        if cloud < (1 / 0.03 / 0.03):
            x_axis.append(date)
            y_axis.append(water * 0.03 * 0.03)  # pixels * km^2 / pixel, km^2 / pixel = 0.03 * 0.03 / 1
            cloud_axis.append(cloud * 0.03 * 0.03)

    f.close()

    # remove values that differ from neighbors by large amounts
    NEIGHBOR_RADIUS = 4
    OUTLIER_FACTOR = 0.90
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
    results['area'] = area
    return (results, x_axis, y_axis, cloud_axis)


def plot_results(features, dates, water, clouds, save_directory=None, ground_truth_file=None):
    fig, ax = plt.subplots()
    water_line = ax.plot(dates, water, linestyle='-', color='b', linewidth=1,
                         label='Landsat Surface Area')
    ax.plot(dates, water, 'gs', ms=3)
    # ax.bar(dates, water, color='b', width=15, linewidth=0)
    # ax.bar(dates, clouds, bottom=water, color='r', width=15, linewidth=0)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))
    ax.set_xlabel('Time')
    ax.format_xdata = mdates.DateFormatter('%m/%d/%Y')
    ax.set_xlim([datetime.date(1984, 1, 1), datetime.date(1984, 12, 31)])
    ax.set_ylim([150, 190])

    lns = water_line
    if ground_truth_file is not None:
        (ground_truth_dates, ground_truth_levels) = load_ground_truth(ground_truth_file)
        ax2 = ax.twinx()
        ground_truth_line = ax2.plot(ground_truth_dates, ground_truth_levels, linestyle='--', color='r', linewidth=2, label='Measured Elevation')
        ax2.set_ylabel('Lake Elevation (ft)')
        ax2.format_ydata = (lambda x: '%g ft' % (x))
        ax2.set_ylim([6372.0, 6385.5])
        lns = lns + ground_truth_line
        ax2.set_xlim([datetime.date(1984, 6, 1), datetime.date(2015, 10, 1)])

    ax.format_ydata = (lambda x: '%g km^2' % (x))
    ax.set_ylabel('Lake Surface Area (km^2)')
    fig.suptitle(features['name'] + ' Surface Area from Landsat')
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


if len(sys.argv) > 1:
    (features, dates, water, clouds) = parse_lake_results(sys.argv[1])
    # plot_results(features, dates, water, clouds, None, 'results/mono_lake_elevation.txt')
    if len(sys.argv) > 2:
        plot_results(features, dates, water, clouds, 'results', sys.argv[2])
    else:
        plot_results(features, dates, water, clouds)
    plt.show()
else:
    for fname in glob.iglob(os.path.join('results', '*.txt')):
        try:
            (features, dates, water, clouds) = parse_lake_results(fname)
        except:
            print 'Error parsing %s.' % (fname)
            continue
        if len(dates) > 100:
            plot_results(features, dates, water, clouds, save_directory=os.path.join('results', 'graphs'))
