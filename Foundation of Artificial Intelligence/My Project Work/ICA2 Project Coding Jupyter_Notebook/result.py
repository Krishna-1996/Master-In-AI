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
bar_width = 0.2
index = np.arange(len(scenarios))

# Create the figure with two subplots: one for Path Length and one for Search Length
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Create grouped bar charts for Path Length and Search Length
# Path Length plot
ax1.bar(index - bar_width, path_lengths['BFS'], bar_width, label='BFS', color='skyblue')
ax1.bar(index, path_lengths['Greedy_BFS'], bar_width, label='Greedy_BFS', color='lightgreen')
ax1.bar(index + bar_width, path_lengths['A*'], bar_width, label='A*', color='lightcoral')
ax1.set_xlabel('Scenarios', fontsize=12)
ax1.set_ylabel('Path Length', fontsize=12)
ax1.set_title('Path Length Comparison', fontsize=14)
ax1.set_xticks(index)
ax1.set_xticklabels(scenarios, fontsize=10)
ax1.legend()

# Search Length plot
ax2.bar(index - bar_width, search_lengths['BFS'], bar_width, label='BFS', color='skyblue')
ax2.bar(index, search_lengths['Greedy_BFS'], bar_width, label='Greedy_BFS', color='lightgreen')
ax2.bar(index + bar_width, search_lengths['A*'], bar_width, label='A*', color='lightcoral')
ax2.set_xlabel('Scenarios', fontsize=12)
ax2.set_ylabel('Search Length', fontsize=12)
ax2.set_title('Search Length Comparison', fontsize=14)
ax2.set_xticks(index)
ax2.set_xticklabels(scenarios, fontsize=10)
ax2.legend()

# Annotate each bar with its value
for i, scenario in enumerate(scenarios):
    for j, algorithm in enumerate(algorithms):
        ax1.text(index[i] - bar_width + j * bar_width, path_lengths[algorithm][i] + 5, path_lengths[algorithm][i], ha='center', fontsize=10)
        ax2.text(index[i] - bar_width + j * bar_width, search_lengths[algorithm][i] + 5, search_lengths[algorithm][i], ha='center', fontsize=10)

# Tight layout for better spacing
plt.tight_layout()
plt.show()
