import numpy as np
import matplotlib.pyplot as plt

# data for algorithm 1
y1 = [1121.56, 1058.49, 998.43, 854.83, 758.79]
time1 = [3.57, 3.37, 2.97, 2.48, 2.24]

# data for algorithm 2
y2 = [1252.63, 1014.20, 1033.52, 882.79, 819.66]
time2 = [11.30, 8.92, 8.08, 6.98, 6.01]

# create an array of indices for the bars
# x = ['7.77%', '14.44%', '24.77%', '34.55%', '42.11%']
x = ['Test case 1.1', 'Test case 1.2', 'Test case 1.3', 'Test case 1.4', 'Test case 1.5']

idx = np.arange(len(y1))

fig, ax1 = plt.subplots()

# set the width of each bar
bar_width = 0.25

# create the bar chart for path length data
ax1.bar(idx - bar_width/2, y1, width=bar_width, label="(path length) BWave", color="#F08080")
ax1.bar(idx + bar_width/2, y2, width=bar_width, label="(path length) ε*+", color="#87CEEB")
ax1.set_xlabel('Obstacle density', fontweight='bold')
ax1.set_ylabel('Total path length (unit)', fontweight='bold')
# ax1.legend(title='Total path length', loc='upper right')
# ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=True)

ax2 = ax1.twinx()

ax2.plot(idx, time1, marker='o', markersize=5, label="(time) Bwave", color="#FFA500")
ax2.plot(idx, time2, marker='o', markersize=5, label="(time) ε*+", color="#90EE90")
ax2.set_ylabel('Time (s)', fontweight='bold')

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax2.set_xticks(idx)
ax2.set_xticklabels(x)
# ax2.legend(title='Time', loc='upper right')
# ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

fig.set_size_inches(h=4, w=8)
plt.title("Scenario 1 - Influence of obstacle density", fontweight='bold')

# show the plot
plt.savefig('draw/img/scenario1.png')
plt.show()
