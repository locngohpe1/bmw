import numpy as np
import matplotlib.pyplot as plt
import csv

# data for algorithm 1 - proposed algorithm
total_path_length = []
number_of_return = []

with open('diagram/results/result_bwave.csv', 'r') as file:
    # create a CSV reader
    reader = csv.DictReader(file)

    # print the data
    for row in reader:
        if row['name'] == 'scenario4\map_1.txt':
            total_path_length.append(round(float(row['total length']), 2))
            number_of_return.append(int(row['number of return']))

# create an array of indices for the bars
# x = [100, 200, 300, 400, 500]
x = ['Test case 4.1', 'Test case 4.2', 'Test case 4.3', 'Test case 4.4', 'Test case 4.5']
idx = np.arange(len(total_path_length))

fig, ax1 = plt.subplots()

# set the width of each bar
bar_width = 0.25

# create the bar chart for path length data
ax1.bar(idx, total_path_length, width=bar_width, label="Total path length", color="#F08080")
ax1.set_xlabel('Energy capacity', fontweight='bold')
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
# ax2.legend(title='Time', loc='upper right')
# ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)
ax2.legend(lines + lines2, labels + labels2, loc='best')

fig.set_size_inches(h=4, w=8)
plt.title("Scenario 4 - Influence of energy capacity", fontweight='bold')

# show the plot
plt.savefig('diagram/img/scenario2.pdf')
plt.show()
