import matplotlib.pyplot as plt
import numpy as np
import csv

def read_result_file(file_path, field):
    results = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            results.append(row[field])
    return results

x = read_result_file('diagram/results/real_map_bwave.csv', 'test case')

idx = np.arange(len(x))
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 12))
bar_width = 0.2

# Plot the first subplot
y1 = read_result_file('diagram/results/real_map_bwave.csv', 'coverage length')
y2 = read_result_file('diagram/results/real_map_estar.csv', 'coverage length')
y3 = read_result_file('diagram/results/real_map_bwzone.csv', 'coverage length')

y1 = [float(i) for i in y1]
y2 = [float(i) for i in y2]
y3 = [float(i) for i in y3]

axs[0].bar(idx - bar_width, y1, width=bar_width * 0.8, label="BWave", color="#F08080", hatch='xxxx')
axs[0].bar(idx, y2, width=bar_width * 0.8, label="ε*+", color="#87CEEB", hatch='\\\\\\\\')
axs[0].bar(idx + bar_width, y3, width=bar_width * 0.8, label="B-WZone", color="#98FB98", hatch='....')

axs[0].set_ylabel('Total path length (unit)', fontweight='bold', fontsize='16')
axs[0].set_yticks([0, 5000, 10000, 15000, 20000])
axs[0].tick_params(axis='y', labelsize=16)
axs[0].yaxis.set_label_coords(-0.06, 0.5)
axs[0].set_axisbelow(True)
axs[0].yaxis.grid(color='gray', linestyle='dashed')
# axs[0].legend(loc='best')

axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize='16')

axs[0].set_xticks(idx)
axs[0].set_xticklabels([])
# axs[0].set_title("Scenario 5 - Real indoor map dataset", fontweight='bold')

# Plot the second subplot
y1 = read_result_file('diagram/results/real_map_bwave.csv', 'overlap rate')
y2 = read_result_file('diagram/results/real_map_estar.csv', 'overlap rate')
y3 = read_result_file('diagram/results/real_map_bwzone.csv', 'overlap rate')

y1 = [float(i) for i in y1]
y2 = [float(i) for i in y2]
y3 = [float(i) for i in y3]

axs[1].bar(idx - bar_width, y1, width=bar_width * 0.8, label="BWave", color="#F08080", hatch='xxxx')
axs[1].bar(idx, y2, width=bar_width * 0.8, label="ε*+", color="#87CEEB", hatch='\\\\\\\\')
axs[1].bar(idx + bar_width, y3, width=bar_width * 0.8, label="B-WZone", color="#98FB98", hatch='....')

axs[1].set_ylabel('Overlap rate (%)', fontweight='bold', fontsize='16')
axs[1].tick_params(axis='y', labelsize=16)
axs[1].yaxis.set_label_coords(-0.06, 0.5)
axs[1].set_axisbelow(True)
axs[1].yaxis.grid(color='gray', linestyle='dashed')
# axs[1].legend(loc='best')

axs[1].set_xticks(idx)
axs[1].set_xticklabels([])

# Plot the third subplot
y1 = read_result_file('diagram/results/real_map_bwave.csv', 'number of return')
y2 = read_result_file('diagram/results/real_map_estar.csv', 'number of return')
y3 = read_result_file('diagram/results/real_map_bwzone.csv', 'number of return')

y1 = [int(i) for i in y1]
y2 = [int(i) for i in y2]
y3 = [int(i) for i in y3]

axs[2].bar(idx - bar_width, y1, width=bar_width * 0.8, label="BWave", color="#F08080", hatch='xxxx')
axs[2].bar(idx, y2, width=bar_width * 0.8, label="ε*+", color="#87CEEB", hatch='\\\\\\\\')
axs[2].bar(idx + bar_width, y3, width=bar_width * 0.8, label="B-WZone", color="#98FB98", hatch='....')

axs[2].set_ylabel('Number of return', fontweight='bold', fontsize='16')
axs[2].tick_params(axis='y', labelsize=16)
axs[2].yaxis.set_label_coords(-0.06, 0.5)
axs[2].set_axisbelow(True)
axs[2].yaxis.grid(color='gray', linestyle='dashed')

# axs[2].legend(loc='best')

axs[2].set_xticks(idx)
axs[2].set_xticklabels(x, rotation=45, fontsize='16')
axs[2].set_xlabel('Map', fontweight='bold', fontsize='16')

fig.subplots_adjust(bottom=0.2)
plt.tight_layout()
plt.savefig('diagram/img/scenario5.pdf')
plt.show()
