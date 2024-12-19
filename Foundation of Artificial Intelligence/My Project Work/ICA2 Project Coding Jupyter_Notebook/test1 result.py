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

# Create the plot with stacked bars
fig, ax = plt.subplots(figsize=(12, 6))

# Stacked bars: Path Length + Search Length for each algorithm
ax.barh(index, [path_lengths['BFS'][i] for i in range(len(scenarios))], bar_width, label='BFS - Path Length', color='skyblue')
ax.barh(index, [search_lengths['BFS'][i] for i in range(len(scenarios))], bar_width, left=[path_lengths['BFS'][i] for i in range(len(scenarios))], label='BFS - Search Length', color='lightcoral')

ax.barh(index + bar_width, [path_lengths['Greedy_BFS'][i] for i in range(len(scenarios))], bar_width, label='Greedy BFS - Path Length', color='lightgreen')
ax.barh(index + bar_width, [search_lengths['Greedy_BFS'][i] for i in range(len(scenarios))], bar_width, left=[path_lengths['Greedy_BFS'][i] for i in range(len(scenarios))], label='Greedy BFS - Search Length', color='yellow')

ax.barh(index + 2 * bar_width, [path_lengths['A*'][i] for i in range(len(scenarios))], bar_width, label='A* - Path Length', color='lightcoral')
ax.barh(index + 2 * bar_width, [search_lengths['A*'][i] for i in range(len(scenarios))], bar_width, left=[path_lengths['A*'][i] for i in range(len(scenarios))], label='A* - Search Length', color='lightblue')

# Add labels and title
ax.set_xlabel('Length', fontsize=12)
ax.set_ylabel('Scenarios', fontsize=12)
ax.set_title('Comparison of Search Algorithms for Different Scenarios', fontsize=14)
ax.set_yticks(index + bar_width)
ax.set_yticklabels(scenarios, fontsize=12)

# Add a legend to describe the colors
ax.legend()

# Display values for each segment in the bars
for i, scenario in enumerate(scenarios):
    for j, algorithm in enumerate(algorithms):
        ax.text(path_lengths[algorithm][i] / 2, index[i] + j * bar_width, path_lengths[algorithm][i], ha='center', fontsize=10, color='black')
        ax.text(path_lengths[algorithm][i] + search_lengths[algorithm][i] / 2, index[i] + j * bar_width, search_lengths[algorithm][i], ha='center', fontsize=10, color='black')

# Tight layout for better spacing
plt.tight_layout()
plt.show()
