import numpy as np
import matplotlib.pyplot as plt

# data for algorithm 1
y1 = [1121.56, 1058.49, 998.43, 854.83, 758.79]

# data for algorithm 2
y2 = [1252.63, 1014.20, 1033.52, 882.79, 819.66]

# data for algorithm 3
y3 = [1163.65, 1226.97, 1071.01, 982.28, 890.42]

# create an array of indices for the bars
# x = ['7.77%', '14.44%', '24.77%', '34.55%', '42.11%']
x = ['Test case 3.1', 'Test case 3.2', 'Test case 3.3', 'Test case 3.4', 'Test case 3.5']

idx = np.arange(len(y1))

fig, ax1 = plt.subplots()

# set the width of each bar
bar_width = 0.15

# create the bar chart for path length data
ax1.bar(idx - bar_width, y1, width=bar_width * 0.8, label="BWave", color="#F08080", hatch='xxxx')
ax1.bar(idx, y2, width=bar_width * 0.8, label="ε*+", color="#87CEEB", hatch='\\\\\\\\')
ax1.bar(idx + bar_width, y3, width=bar_width * 0.8, label="B-WZone", color="#98FB98", hatch='....')
# ax1.set_xlabel('Obstacle density', fontweight='bold')
ax1.set_ylabel('Total path length (unit)', fontweight='bold')
# ax1.legend(title='Total path length', loc='upper right')
# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=True)

ax1.set_axisbelow(True)
ax1.yaxis.grid(color='gray', linestyle='dashed')

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()

ax1.set_xticks(idx)
ax1.set_xticklabels(x)
ax1.legend(lines, labels, loc='upper right')

fig.set_size_inches(h=4, w=8)
plt.title("Scenario 3 - Influence of obstacle density", fontweight='bold')

# show the plot
plt.savefig('diagram/img/scenario3.png')
plt.show()
