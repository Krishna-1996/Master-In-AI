import matplotlib.pyplot as plt
import pandas as pd

# Data for Greedy BFS Algorithm
data_greedybfs = {
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
df_greedybfs = pd.DataFrame(data_greedybfs)

# Extract values for plotting
weights_greedybfs = df_greedybfs["Weight"].apply(str).tolist()
manhattan_path_greedybfs = df_greedybfs["Manhattan Path Length"].tolist()
euclidean_path_greedybfs = df_greedybfs["Euclidean Path Length"].tolist()
chebyshev_path_greedybfs = df_greedybfs["Chebyshev Path Length"].tolist()

manhattan_search_greedybfs = df_greedybfs["Manhattan Search Length"].tolist()
euclidean_search_greedybfs = df_greedybfs["Euclidean Search Length"].tolist()
chebyshev_search_greedybfs = df_greedybfs["Chebyshev Search Length"].tolist()

# --- Plot 1: Path Lengths for Greedy BFS ---
fig, ax1 = plt.subplots(figsize=(12, 8))

# Set the title and labels before plotting the data
ax1.set_title('Greedy BFS Algorithm - Path Lengths for Different Heuristics')
ax1.set_xlabel('Weights')
ax1.set_ylabel('Path Length', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# First plot the data points (markers) to avoid lines covering the points
ax1.scatter(weights_greedybfs, manhattan_path_greedybfs, label="Manhattan Path Length", color='blue', marker='o', s=100)
ax1.scatter(weights_greedybfs, euclidean_path_greedybfs, label="Euclidean Path Length", color='green', marker='s', s=100)
ax1.scatter(weights_greedybfs, chebyshev_path_greedybfs, label="Chebyshev Path Length", color='red', marker='^', s=100)

# Now plot the lines (after the data points)
ax1.plot(weights_greedybfs, manhattan_path_greedybfs, color='blue', linestyle='-', linewidth=2, alpha=0.5)
ax1.plot(weights_greedybfs, euclidean_path_greedybfs, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax1.plot(weights_greedybfs, chebyshev_path_greedybfs, color='red', linestyle='-.', linewidth=2, alpha=0.5)

ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Search Lengths for Greedy BFS ---
fig, ax2 = plt.subplots(figsize=(12, 8))

# Set the title and labels before plotting the data
ax2.set_title('Greedy BFS Algorithm - Search Lengths for Different Heuristics')
ax2.set_xlabel('Weights')
ax2.set_ylabel('Search Length', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# First plot the data points (markers) to avoid lines covering the points
ax2.scatter(weights_greedybfs, manhattan_search_greedybfs, label="Manhattan Search Length", color='blue', marker='o', s=100)
ax2.scatter(weights_greedybfs, euclidean_search_greedybfs, label="Euclidean Search Length", color='green', marker='s', s=100)
ax2.scatter(weights_greedybfs, chebyshev_search_greedybfs, label="Chebyshev Search Length", color='red', marker='^', s=100)

# Now plot the lines (after the data points)
ax2.plot(weights_greedybfs, manhattan_search_greedybfs, color='blue', linestyle=':', linewidth=2, alpha=0.5)
ax2.plot(weights_greedybfs, euclidean_search_greedybfs, color='green', linestyle=':', linewidth=2, alpha=0.5)
ax2.plot(weights_greedybfs, chebyshev_search_greedybfs, color='red', linestyle=':', linewidth=2, alpha=0.5)

ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()
