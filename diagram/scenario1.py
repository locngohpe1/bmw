import numpy as np
import matplotlib.pyplot as plt
import csv

coverage_length = []
return_length = []
advance_length = []

with open('diagram/results/result_bwave.csv', 'r') as file:
    # create a CSV reader
    reader = csv.DictReader(file)

    # print the data
    for row in reader:
        if row['name'] == 'scenario3\map_1.txt':
            coverage_length.append(round(float(row['coverage length']), 2))
            return_length.append(round(float(row['retreat length']), 2))
            advance_length.append(round(float(row['advance length']), 2))


# create an array of indices for the bars
idx = np.arange(len(coverage_length))
x = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']

# plt.figure(figsize=(6, 4))

# set the width of each bar
bar_width = 0.25

plt.bar(idx, coverage_length, width=bar_width, label="coverage length", color="#FFB6C1")
plt.bar(idx, return_length, width=bar_width, label="return length", bottom=coverage_length, color="#87CEFA")
plt.bar(idx, advance_length, width=bar_width, label="advance length", bottom=[c + r for c, r in zip(coverage_length, return_length)], color="#98FB98")

plt.xticks(idx, x)

plt.gca().set_axisbelow(True)
plt.gca().yaxis.grid(color='gray', linestyle='dashed')

# add a legend
plt.legend(bbox_to_anchor=(0.5, -0.25), loc='lower center', ncol=3, frameon=True)

# add axis labels and a title
plt.xlabel("Charge station position", fontweight='bold')
plt.ylabel("Total path length (unit)", fontweight='bold')
# plt.title("Scenario 1 - Influence of charging position", fontweight='bold')

# show the plot
plt.tight_layout()
plt.savefig('diagram/img/scenario1.pdf', bbox_inches='tight', pad_inches = 0)
plt.show()
