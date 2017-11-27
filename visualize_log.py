import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import util
import argparse
import numpy
import random

#
# Code adapted from http://matplotlib.org/examples/showcase/bachelors_degrees_by_gender.html
#

parser = argparse.ArgumentParser()
parser.add_argument("logfiles", help="List of log files", nargs="+")
parser.add_argument("--output", help="Location to output graph to")
args = parser.parse_args()

logfiles = args.logfiles
assert len(logfiles) != 0 and len(logfiles) <= 20

log_data = {}
for logfile in logfiles:
    with open(logfile) as f:
        lines = [line.split(",") for line in f.read().split("\n") if line and line[:4] != "TEST" and line[0] != "#"]
        xs = [float(line[0]) for line in lines]
        losses = [float(line[1]) for line in lines]
        perps = [float(line[2]) for line in lines]
        bpcs = [float(line[3]) for line in lines]
        # log_data[logfile + " loss"] = {"xs":xs, "ys":losses}
        log_data[logfile + " ppl"] = {"xs":xs, "ys":perps}
        # log_data[logfile + " bpc"] = {"xs":xs, "ys":bpcs}

# x_min = min(util.flatten([data["xs"] for data in log_data.values()]))
x_min = 0
x_max = max(util.flatten([data["xs"] for data in log_data.values()]))
x_tick_size = (x_max - x_min)/24.
# x_min -= x_tick_size/2
x_max += x_tick_size/2

y_min = min(util.flatten([data["ys"] for data in log_data.values()]))
y_max = max(util.flatten([data["ys"] for data in log_data.values()]))
y_tick_size = (y_max - y_min)/8.
y_min -= y_tick_size/2
y_max += y_tick_size/2

# These are the colors that will be used in the plot
color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
# random.shuffle(color_sequence)

# You typically want your plot to be ~1.33x wider than tall.
# Common sizes: (10, 7.5) and (12, 9)
fig, ax = plt.subplots(1, 1, figsize=(12, 9))

# Remove the plot frame lines. They are unnecessary here.
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Limit the range of the plot to only where the data is.
# Avoid unnecessary whitespace.
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

def human_format(num, dec=2):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return ('%.'+str(dec)+'f%s') % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

# Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
plt.xticks(numpy.arange(x_min, x_max, x_tick_size),
           [human_format(x, 0) for x in numpy.arange(x_min, x_max, x_tick_size)],
           fontsize=14, rotation=45)
plt.yticks(numpy.arange(y_min, y_max, y_tick_size),
           [human_format(y, 1) for y in numpy.arange(y_min, y_max, y_tick_size)],
           fontsize=14)

# Provide tick lines across the plot to help your viewers trace along
# the axis ticks. Make sure that the lines are light and small so they
# don't obscure the primary data lines.
for y in numpy.arange(y_min, y_max, y_tick_size):
    plt.plot(numpy.arange(x_min, x_max), [y] * len(numpy.arange(x_min, x_max)), '--',
             lw=0.5, color='black', alpha=0.3)

# Remove the tick marks; they are unnecessary with the tick lines we just
# plotted.
plt.tick_params(axis='both', which='both', bottom='on', top='off',
                labelbottom='on', left='off', right='off', labelleft='on')

## Now that the plot is prepared, it's time to actually plot the data!

for n, title in enumerate(log_data):
    # Plot each line separately with its own color.
    line = plt.plot(log_data[title]["xs"],
                    log_data[title]["ys"],
                    lw=2.5,
                    color=color_sequence[n])

    # Add a text label to the right end of every line.
    x_pos = log_data[title]["xs"][0] + 15000
    y_pos = log_data[title]["ys"][0] * .95

    # Again, make sure that all labels are large enough to be easily read
    # by the viewer.
    plt.text(x_pos, y_pos, title, fontsize=14, color=color_sequence[n])

# Make the title big enough so it spans the entire plot, but don't make it
# so big that it requires two lines to show.

# Note that if the title is descriptive enough, it is unnecessary to include
# axis labels; they are self-evident, in this plot's case.
plt.title('Validation Set Perplexity After N Training Iterations\n', fontsize=18, ha='center')

# Finally, save the figure.
# Just change the file extension in this call.
if args.output is not None: plt.savefig(args.output, bbox_inches='tight')
else:                       plt.show()
