import numpy as np
import matplotlib.pyplot as plt
import csv

# data for algorithm 1 - proposed algorithm
total_path_length1 = []
total_path_length2 = []
number_of_return1 = []
number_of_return2 = []

with open('result.csv', 'r') as file:
    # create a CSV reader
    reader = csv.DictReader(file)

    # print the data
    for row in reader:
        if row['name'][0:9] == 'scenario5':
            total_path_length1.append(round(float(row['total length']), 2))
            number_of_return1.append(int(row['number of return']) - 1)

with open('result_e_star.csv', 'r') as file:
    # create a CSV reader
    reader = csv.DictReader(file)

    # print the data
    for row in reader:
        if row['name'][0:9] == 'scenario5':
            total_path_length2.append(round(float(row['total length']), 2))
            number_of_return2.append(int(row['number of return']) - 1)

# create an array of indices for the bars
# x = ['indoor', 'cave', 'storage']
x = ['Test case 5.1', 'Test case 5.2', 'Test case 5.3']

idx = np.arange(len(total_path_length1))

fig, ax1 = plt.subplots()

ax1.plot(idx, number_of_return1, marker='o', markersize=5, label="BWave", color="#696969")
ax1.plot(idx, number_of_return2, marker='o', markersize=5, label="ε*+", color="#FFFF00")

ax1.set_xlabel('Real-life simulated environment', fontweight='bold')
ax1.set_yticks([4, 5, 6, 7, 8, 9])
ax1.set_ylabel('Number of return', fontweight='bold')

ax1.set_xticks(idx)
ax1.set_xticklabels(x)

ax1.legend(loc='best')

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()

# ax2.set_xticks(idx)
# ax2.set_xticklabels(x)
# ax2.legend(title='Time', loc='upper right')
# ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)
# ax2.legend(lines + lines2, labels + labels2, loc='best')

# fig.set_size_inches(h=4, w=8)
plt.title("Scenario 5 - Real-life simulated environment", fontweight='bold')

# show the plot
plt.savefig('draw/img/scenario5.2.png')
plt.show()
