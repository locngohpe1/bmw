import numpy as np
import matplotlib.pyplot as plt
import csv

def read_result_file(file_path, field):
    results = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            results.append(row[field])
    return results

x = read_result_file('diagram/results/real_map_bwave.csv', 'test case')

y1 = read_result_file('diagram/results/real_map_bwave.csv', 'number of return')
y2 = read_result_file('diagram/results/real_map_estar.csv', 'number of return')
y3 = read_result_file('diagram/results/real_map_bwzone.csv', 'number of return')

y1 = [int(i) for i in y1]
y2 = [int(i) for i in y2]
y3 = [int(i) for i in y3]

idx = np.arange(len(x))
fig, ax1 = plt.subplots()

# set the width of each bar
bar_width = 0.2

# create the bar chart for path length data
ax1.bar(idx - bar_width, y1, width=bar_width * 0.8, label="BWave", color="#F08080", hatch='xxxx')
ax1.bar(idx, y2, width=bar_width * 0.8, label="ε*+", color="#87CEEB", hatch='\\\\\\\\')
ax1.bar(idx + bar_width, y3, width=bar_width * 0.8, label="B-WZone", color="#98FB98", hatch='....')
ax1.set_xlabel('Map', fontweight='bold')
ax1.set_ylabel('Number of return', fontweight='bold')

ax1.set_xticks(idx)
ax1.set_xticklabels(x, rotation=45)

ax1.legend(loc='best')
fig.subplots_adjust(bottom=0.2)

# ask matplotlib for the plotted objects and their labels
# lines, labels = ax1.get_legend_handles_labels()

fig.set_size_inches(h=5, w=12)
# plt.title("Scenario 5 - Real map environment", fontweight='bold')
plt.tight_layout()

# show the plot
plt.savefig('diagram/img/scenario5_3.png')
plt.show()
