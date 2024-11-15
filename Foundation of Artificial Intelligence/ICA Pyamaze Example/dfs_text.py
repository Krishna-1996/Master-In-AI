from collections import deque

# Define the maze data (open directions for each cell)
# The maze data format: (E, W, N, S)
maze = {
    (1, 1): (1, 0, 0, 1), (2, 1): (0, 0, 1, 1), (3, 1): (1, 0, 1, 1), 
    (4, 1): (0, 0, 1, 1), (5, 1): (1, 0, 1, 1), (6, 1): (1, 0, 1, 0),
    # Add all cells here...
    (25, 18): (0, 1, 0, 1)  # Example start cell
}

# Directions: (East, West, North, South)
directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # E, W, N, S

# Goal
start = (25, 18)  # Starting point at bottom-right
goal = (1, 1)     # Goal point at top-left

# BFS implementation
def bfs(maze, start, goal):
    visited = set()  # To track visited cells
    queue = deque([(start, [])])  # Queue of tuples (current cell, path taken)
    
    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path  # Return the path if goal is reached
        
        if current in visited:
            continue
        
        visited.add(current)
        for i, (dx, dy) in enumerate(directions):
            if maze[current][i] == 1:  # Check if the direction is open
                next_cell = (current[0] + dx, current[1] + dy)
                if 1 <= next_cell[0] <= 25 and 1 <= next_cell[1] <= 18:  # In bounds
                    if next_cell not in visited:
                        queue.append((next_cell, path + [next_cell]))  # Append next cell with path taken

    return None  # If no path found

# Find the path from start to goal
path = bfs(maze, start, goal)
if path:
    print(f"Path from {start} to {goal}: {path}")
else:
    print("No path found.")
