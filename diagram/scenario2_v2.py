import numpy as np
import matplotlib.pyplot as plt
import csv

# data for algorithm 1 - proposed algorithm
coverage_length = []
return_length = []
advance_length = []
number_of_return = []

with open('diagram/results/result_bwave.csv', 'r') as file:
    # create a CSV reader
    reader = csv.DictReader(file)

    # print the data
    for row in reader:
        if row['name'] == 'scenario4\map_1.txt':
            coverage_length.append(round(float(row['coverage length']), 2))
            return_length.append(round(float(row['retreat length']), 2))
            advance_length.append(round(float(row['advance length']), 2))
            number_of_return.append(int(row['number of return']))

# create an array of indices for the bars
# x = [100, 200, 300, 400, 500]
x = ['Test case 4.1', 'Test case 4.2', 'Test case 4.3', 'Test case 4.4', 'Test case 4.5']
idx = np.arange(len(coverage_length))

fig, ax1 = plt.subplots()

# set the width of each bar
bar_width = 0.25

# create the bar chart for path length data
ax1.bar(idx, coverage_length, width=bar_width, label="coverage length", color="#FFB6C1")
ax1.bar(idx, return_length, width=bar_width, label="return length", bottom=coverage_length, color="#87CEFA")
ax1.bar(idx, advance_length, width=bar_width, label="advance length", bottom=[c + r for c, r in zip(coverage_length, return_length)], color="#98FB98")
# ax1.set_xlabel('Energy capacity', fontweight='bold')
ax1.set_ylabel('Total path length (unit)', fontweight='bold')
# ax1.legend(title='Total path length', loc='upper right')
# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=True)

ax2 = ax1.twinx()

ax2.plot(idx, number_of_return, marker='o', markersize=5, label="Number of return", color="#696969")
ax2.set_ylabel('Number of return', fontweight='bold')
ax2.set_yticks([0, 4, 8, 12, 16])

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax2.set_xticks(idx)
ax2.set_xticklabels(x)

ax1.set_ylim(0, 2500)
ax2.set_ylim(0, 20)

ax1.set_yticks([0, 500, 1000, 1500, 2000])
ax1.set_axisbelow(True)
ax1.yaxis.grid(color='gray', linestyle='dashed')

# ax2.legend(title='Time', loc='upper right')
# ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)
ax2.legend(lines + lines2, labels + labels2, loc='best')

fig.set_size_inches(h=4, w=8)
# plt.title("Scenario 2 - Influence of energy capacity", fontweight='bold')

# show the plot
plt.savefig('diagram/img/scenario2_v2.pdf')
plt.show()
