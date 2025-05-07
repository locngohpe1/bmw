import matplotlib.pyplot as plt

# data for repetition rate
repetition = [0.5, 0.4, 0.3, 0.2, 0.1]

# data for coverage rate
coverage = [0.7, 0.6, 0.5, 0.4, 0.3]

# data for the amount of obstacles
obstacles = [0, 1, 2, 3, 4]

# create the plot
fig, ax = plt.subplots()

# plot the repetition rate as an area chart
ax.fill_between(obstacles, 0, repetition, label="repetition rate", color="#FFA07A")

# plot the coverage rate as an area chart
ax.fill_between(obstacles, 0, coverage, label="coverage rate", color="#ADD8E6")

# add a legend
ax.legend()

# add axis labels and a title
ax.set_xlabel("Amount of obstacles")
ax.set_ylabel("Rate")
ax.set_title("Repetition and coverage rates")

# show the plot
plt.show()
