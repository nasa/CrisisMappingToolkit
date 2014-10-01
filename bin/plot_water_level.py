import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime

import numpy

if len(sys.argv) < 2:
    print >> sys.stderr, 'Usage: plot_water_level.py water_levels.txt'

f = open(sys.argv[1], 'r')

x_axis = []
y_axis = []

# take values with low cloud cover
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
    if cloud < 1000:
        x_axis.append(date)
        y_axis.append(water * 0.03 * 0.03) #pixels * km^2 / pixel, km^2 / pixel = 0.03 * 0.03 / 1

f.close()


# remove values that differ from neighbors by large amounts
NEIGHBOR_RADIUS = 3
OUTLIER_FACTOR = 0.9
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
        if v < y_axis[i] * OUTLIER_FACTOR or v > y_axis[i] / OUTLIER_FACTOR:
            num_outliers += 1
    if float(num_outliers) / num_neighbors >= 0.5:
        remove.append(i)

for i in reversed(remove):
    y_axis.pop(i)
    x_axis.pop(i)

fig, ax = plt.subplots()
ax.plot(x_axis, y_axis, marker='o', linestyle='--')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.format_xdata = mdates.DateFormatter('%m/%d/%Y')
ax.format_ydata = (lambda x : '%g km^2' % (x))
ax.set_xlabel('Time')
ax.set_ylabel('Lake Surface Area (km^2)')
fig.suptitle(names + ' Surface Area from Landsat')

ax.grid(True)
fig.autofmt_xdate()

plt.show()

