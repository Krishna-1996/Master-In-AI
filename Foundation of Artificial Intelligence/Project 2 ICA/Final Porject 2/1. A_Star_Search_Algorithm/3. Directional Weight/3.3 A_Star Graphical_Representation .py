import matplotlib.pyplot as plt
import pandas as pd

# Data for A* Search Algorithm
data_astar = {
    "Direction": ["N,E,S,W"] * 8,
    "Weight": [
        (10, -10, -10, 10), (-10, 10, 10, -10), (0, 10, 10, 0),
        (0, -10, -10, 0), (0, 0, 0, 0), (10, 10, 10, 10),
        (-10, 0, 0, -10), (10, 0, 0, 10)
    ],
    "Manhattan Path Length": [1187, 1187, 1075, None, 1837, 1075, None, 1075],
    "Manhattan Search Length": [8999, 5277, 9090, None, 4377, 8984, None, 8996],
    "Euclidean Path Length": [1173, 1795, 1075, None, 1841, 1075, None, 1075],
    "Euclidean Search Length": [8999, 5277, 9090, None, 4377, 8984, None, 8996],
    "Chebyshev Path Length": [1173, 1795, 1075, None, 1735, 1075, None, 1075],
    "Chebyshev Search Length": [8999, 5277, 9090, None, 4377, 8984, None, 8996]
}

# Create DataFrame
df_astar = pd.DataFrame(data_astar)

# Clean the data (drop rows with NaN values)
df_astar_clean = df_astar.dropna(subset=["Manhattan Path Length", "Euclidean Path Length", "Chebyshev Path Length"])

# Extract values for plotting
weights_astar = df_astar_clean["Weight"].apply(str).tolist()
manhattan_path_astar = df_astar_clean["Manhattan Path Length"].tolist()
euclidean_path_astar = df_astar_clean["Euclidean Path Length"].tolist()
chebyshev_path_astar = df_astar_clean["Chebyshev Path Length"].tolist()

manhattan_search_astar = df_astar_clean["Manhattan Search Length"].tolist()
euclidean_search_astar = df_astar_clean["Euclidean Search Length"].tolist()
chebyshev_search_astar = df_astar_clean["Chebyshev Search Length"].tolist()

# --- Plot 1: Path Lengths for A* ---
fig, ax1 = plt.subplots(figsize=(12, 8))

# Set the title and labels before plotting the data
ax1.set_title('A* Search Algorithm - Path Lengths for Different Heuristics')
ax1.set_xlabel('Weights')
ax1.set_ylabel('Path Length', color='black')
ax1.tick_params(axis='y', labelcolor='black')

# First plot the data points (markers) to avoid lines covering the points
ax1.scatter(weights_astar, euclidean_path_astar, label="Euclidean Path Length", color='green', marker='s', s=100)
ax1.scatter(weights_astar, manhattan_path_astar, label="Manhattan Path Length", color='blue', marker='o', s=100)
ax1.scatter(weights_astar, chebyshev_path_astar, label="Chebyshev Path Length", color='red', marker='^', s=100)


# Now plot the lines (after the data points)
ax1.plot(weights_astar, euclidean_path_astar, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax1.plot(weights_astar, manhattan_path_astar, color='blue', linestyle='-', linewidth=2, alpha=0.5)
ax1.plot(weights_astar, chebyshev_path_astar, color='red', linestyle='-.', linewidth=2, alpha=0.5)


ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Search Lengths for A* ---
fig, ax2 = plt.subplots(figsize=(12, 8))

# Set the title and labels before plotting the data
ax2.set_title('A* Search Algorithm - Search Lengths for Different Heuristics')
ax2.set_xlabel('Weights')
ax2.set_ylabel('Search Length', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# First plot the data points (markers) to avoid lines covering the points
ax2.scatter(weights_astar, euclidean_search_astar, label="Euclidean Search Length", color='green', marker='s', s=100)
ax2.scatter(weights_astar, manhattan_search_astar, label="Manhattan Search Length", color='blue', marker='o', s=100)
ax2.scatter(weights_astar, chebyshev_search_astar, label="Chebyshev Search Length", color='red', marker='^', s=100)


# Now plot the lines (after the data points)
ax2.plot(weights_astar, euclidean_search_astar, color='green', linestyle=':', linewidth=2, alpha=0.5)
ax2.plot(weights_astar, manhattan_search_astar, color='blue', linestyle=':', linewidth=2, alpha=0.5)
ax2.plot(weights_astar, chebyshev_search_astar, color='red', linestyle=':', linewidth=2, alpha=0.5)


ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()
