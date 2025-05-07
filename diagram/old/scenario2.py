import numpy as np
import matplotlib.pyplot as plt

# data for algorithm 1 - proposed algorithm
y1 = [826.32, 986.07, 1007.17, 975.97]
time1 = [2.58, 2.92, 2.98, 2.89]

# data for algorithm 2 - e star plus
y2 = [849.74, 1010.71, 1047.82, 1019.78]
time2 = [6.67, 7.03, 8.08, 6.57]

# create an array of indices for the bars
# x = ['rectangle', 'convex', 'concave rectangle', 'concave random']
x = ['Test case 2.1', 'Test case 2.2', 'Test case 2.3', 'Test case 2.4']

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

ax2.plot(idx, time1, marker='o', markersize=5, label="(time) BWave", color="#FFA500")
ax2.plot(idx, time2, marker='o', markersize=5, label="(time) ε*+", color="#90EE90")
ax2.set_ylabel('Time (s)', fontweight='bold')

# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax2.set_xticks(idx)
ax2.set_xticklabels(x)
# ax2.legend(title='Time', loc='upper right')
# ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False)
ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.5, -0.27), loc='lower center', ncol=5, frameon=False)
# ax2.legend(lines + lines2, labels + labels2, loc='best')

fig.set_size_inches(h=4.27, w=8)
plt.title("Scenario 2 - Influence of obstacle shape", fontweight='bold')

# show the plot
plt.tight_layout()
plt.savefig('draw/img/scenario2.png')
plt.show()
