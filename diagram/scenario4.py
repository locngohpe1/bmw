import numpy as np
import matplotlib.pyplot as plt

# data for algorithm 1 - proposed algorithm
y1 = [826.32, 986.07, 1007.17, 975.97]

# data for algorithm 2 - e star plus
y2 = [849.74, 1010.71, 1047.82, 1019.78]

# data for algorithm 3 - B-WZone
y3 = [1004.28, 998.12, 1012.32, 1087.92]

# create an array of indices for the bars
# x = ['rectangle', 'convex', 'concave rectangle', 'concave random']
x = ['Test case 4.1', 'Test case 4.2', 'Test case 4.3', 'Test case 4.4']

idx = np.arange(len(y1))

fig, ax1 = plt.subplots()

# set the width of each bar
bar_width = 0.15

# create the bar chart for path length data
ax1.bar(idx - bar_width, y1, width=bar_width * 0.8, label="BWave", color="#F08080", hatch='xxxx')
ax1.bar(idx, y2, width=bar_width * 0.8, label="ε*+", color="#87CEEB", hatch='\\\\\\\\')
ax1.bar(idx + bar_width, y3, width=bar_width * 0.8, label="B-WZone", color="#98FB98", hatch='....')
# ax1.set_xlabel('Obstacle shape', fontweight='bold', fontsize='13')
ax1.set_ylabel('Total path length (unit)', fontweight='bold', fontsize='12')
# ax1.legend(title='Total path length', loc='upper right')
# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=True)

ax1.set_axisbelow(True)
ax1.yaxis.grid(color='gray', linestyle='dashed')
ax1.tick_params(axis='both', labelsize=12)

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()

ax1.set_xticks(idx)
ax1.set_xticklabels(x)
ax1.legend(lines, labels, bbox_to_anchor=(0.5, -0.25), loc='lower center', ncol=3, frameon=True, fontsize='12')

fig.set_size_inches(h=5, w=8)
# plt.title("Scenario 4 - Influence of obstacle shape", fontweight='bold', fontsize='12')

# show the plot
plt.tight_layout()
plt.savefig('diagram/img/scenario4.pdf', pad_inches = 0)
plt.show()
