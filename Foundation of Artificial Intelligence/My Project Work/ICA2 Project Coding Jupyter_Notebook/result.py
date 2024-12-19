import matplotlib.pyplot as plt
import numpy as np

# Data for the scenarios and algorithms
scenarios = ['Scenario 1', 'Scenario 2', 'Scenario 3']
algorithms = ['BFS', 'Greedy_BFS', 'A*']

# Path Lengths and Search Lengths for each algorithm in each scenario
path_lengths = {
    'BFS': [189, 174, 83],
    'Greedy_BFS': [229, 204, 101],
    'A*': [189, 174, 83]
}

search_lengths = {
    'BFS': [5991, 5707, 2253],
    'Greedy_BFS': [401, 331, 152],
    'A*': [4264, 2358, 657]
}

# Set up the bar width and positions for the bars
bar_width = 0.35
index = np.arange(len(scenarios))

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars for Path Length and Search Length
bar1 = ax.bar(index - bar_width/2, [path_lengths[alg][i] for i, alg in enumerate(algorithms)], bar_width, label='Path Length')
bar2 = ax.bar(index + bar_width/2, [search_lengths[alg][i] for i, alg in enumerate(algorithms)], bar_width, label='Search Length')

# Add labels and title
ax.set_xlabel('Scenarios')
ax.set_ylabel('Length')
ax.set_title('Comparison of Search Algorithms for Different Scenarios')
ax.set_xticks(index)
ax.set_xticklabels(scenarios)
ax.legend()

plt.tight_layout()
plt.show()
