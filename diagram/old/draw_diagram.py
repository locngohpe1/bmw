import numpy as np
import matplotlib.pyplot as plt

# data for algorithm 1
y1 = [1121.5634918610417, 1058.4924240491762, 998.4335495461307, 854.8355697996838, 758.7939392393408]
time1 = [3.5706660747528076, 3.3766860961914062, 2.9791994094848633, 2.484827756881714, 2.2401483058929443]

# data for algorithm 2
y2 = [1252.6345596729072, 1014.208152801714, 1033.521861300699, 882.793939239341, 819.66399692443]
time2 = [11.309162616729736, 8.921501874923706, 8.082579612731934, 6.989507436752319, 6.019455194473267]

# create an array of indices for the bars
x = np.arange(len(y1))

# set the width of each bar
bar_width = 0.25

# create the bar chart for path length data
plt.bar(x - bar_width/2, y1, width=bar_width, label="algorithm 1 (path length)", color="#FFA07A")
plt.bar(x + bar_width/2, y2, width=bar_width, label="algorithm 2 (path length)", color="#ADD8E6")

# create a second y-axis for time data
ax2 = plt.gca().twinx()

# create the bar chart for time data
ax2.plot(x, time1, label="algorithm 1 (time)")
ax2.plot(x, time2, label="algorithm 2 (time)")

# add a legend
plt.legend(loc=1)
plt.legend(loc=2)

# add axis labels and a title
plt.xlabel("Obstacle density")
plt.ylabel("Total path length")
plt.title("Performance comparison of two algorithms")

# show the plot
plt.show()
