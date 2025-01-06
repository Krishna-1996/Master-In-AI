import matplotlib.pyplot as plt
import pandas as pd

# Data for Greedy BFS Algorithm
data_greedy_bfs = {
    "Direction": ["N,E,S,W"] * 8,
    "Weight": [
        (10, -10, -10, 10), (-10, 10, 10, -10), (0, 10, 10, 0),
        (0, -10, -10, 0), (0, 0, 0, 0), (10, 10, 10, 10),
        (-10, 0, 0, -10), (10, 0, 0, 10)
    ],
    "Manhattan Path Length": [1837, 1837, 1837, 1837, 1837, 1837, 1837, 1837],
    "Manhattan Search Length": [4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377],
    "Euclidean Path Length": [1841, 1841, 1841, 1841, 1841, 1841, 1841, 1841],
    "Euclidean Search Length": [4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377],
    "Chebyshev Path Length": [1735, 1735, 1735, 1735, 1735, 1735, 1735, 1735],
    "Chebyshev Search Length": [4377, 4377, 4377, 4377, 4377, 4377, 4377, 4377]
}

# Create DataFrame
df_greedy_bfs = pd.DataFrame(data_greedy_bfs)

# Extract values for plotting
weights_greedy = df_greedy_bfs["Weight"].apply(str).tolist()
manhattan_path = df_greedy_bfs["Manhattan Path Length"].tolist()
euclidean_path = df_greedy_bfs["Euclidean Path Length"].tolist()
chebyshev_path = df_greedy_bfs["Chebyshev Path Length"].tolist()

manhattan_search = df_greedy_bfs["Manhattan Search Length"].tolist()
euclidean_search = df_greedy_bfs["Euclidean Search Length"].tolist()
chebyshev_search = df_greedy_bfs["Chebyshev Search Length"].tolist()

# Plot 1: Path Lengths

fig, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(weights_greedy, manhattan_path, label="Manhattan Path Length", color='blue', marker='o', linestyle='-', linewidth=2)
ax1.plot(weights_greedy, euclidean_path, label="Euclidean Path Length", color='green', marker='s', linestyle='--', linewidth=2)
ax1.plot(weights_greedy, chebyshev_path, label="Chebyshev Path Length", color='red', marker='^', linestyle='-.', linewidth=2)

ax1.set_xlabel('Weights')
ax1.set_ylabel('Path Length', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_title('Greedy BFS Algorithm - Path Lengths for Different Heuristics')
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Search Length

fig, ax2 = plt.subplots(figsize=(12, 8))

ax2.plot(weights_greedy, manhattan_search, label="Manhattan Search Length", color='blue', marker='o', linestyle=':', linewidth=2)
ax2.plot(weights_greedy, euclidean_search, label="Euclidean Search Length", color='green', marker='s', linestyle=':', linewidth=2)
ax2.plot(weights_greedy, chebyshev_search, label="Chebyshev Search Length", color='red', marker='^', linestyle=':', linewidth=2)

ax2.set_xlabel('Weights')
ax2.set_ylabel('Search Length', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
ax2.set_title('Greedy BFS Algorithm - Search Lengths for Different Heuristics')
ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()
