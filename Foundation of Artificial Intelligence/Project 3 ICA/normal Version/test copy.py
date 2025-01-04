# %%
import pandas as pd
import numpy as np
import random
import sys
# %%
# Load maze from maze.csv
maze_file_path = 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/normal Version/maze--2025-01-03--13-49-03.csv'
maze_df = pd.read_csv(maze_file_path, header=None, names=["cell", "E", "W", "N", "S"])
maze_array = maze_df.values
# %%
# Dimensions of the maze
rows, cols = maze_array.shape

# Set seed for reproducibility
random.seed(42)

# Function to check if a 2x2 block can be placed at a given position
def can_place_obstacle(x, y, maze, placed_obstacles):
    for dx in range(2):
        for dy in range(2):
            if x + dx >= rows or y + dy >= cols or maze[x + dx, y + dy] != 0 or (x + dx, y + dy) in placed_obstacles:
                return False
    return True
# %%
# Generate random obstacles as 2x2 blocks
obstacles = set()
max_attempts = 10000  # Increased limit to avoid infinite loop
total_obstacles = 100  # Total number of obstacles we want to generate
attempts = 0

print("Progress: ", end="")
# %%
while len(obstacles) < total_obstacles and attempts < max_attempts:  # 100 is the arbitrary number of obstacles
    x = random.randint(0, rows - 2)
    y = random.randint(0, cols - 2)
    
    if can_place_obstacle(x, y, maze_array, obstacles):
        for dx in range(2):
            for dy in range(2):
                obstacles.add((x + dx, y + dy))
        attempts += 1
    
    # Show progress
    progress = len(obstacles) / total_obstacles * 100
    sys.stdout.write(f"\rProgress: {len(obstacles)}/{total_obstacles} ({progress:.2f}%)")
    sys.stdout.flush()
# %%
# Prepare the obstacles in the required format
obstacle_list = []
for (x, y) in obstacles:
    cell = f"({x+1}, {y+1})"
    E, W, N, S = 1, 1, 1, 1  # Default values for a 2x2 block
    # Check for walls in the maze and adjust the E, W, N, S values accordingly
    if x > 0 and maze_array[x-1, y] == 1:
        W = 0
    if y > 0 and maze_array[x, y-1] == 1:
        N = 0
    if x + 1 < rows and maze_array[x+1, y] == 1:
        E = 0
    if y + 1 < cols and maze_array[x, y+1] == 1:
        S = 0
    
    obstacle_list.append([cell, E, W, N, S])
# %%
# Save obstacles to random_obstacles.csv
obstacles_df = pd.DataFrame(obstacle_list, columns=["cell", "E", "W", "N", "S"])
obstacles_df.to_csv('D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/normal Version/random_obstacles2.csv', index=False)

print(f"\nRandom obstacles created and saved: {len(obstacles)}")
