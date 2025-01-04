import pandas as pd
import heapq
import numpy as np

# Load maze from maze.csv
maze_file_path = 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/normal Version/maze--2025-01-03--13-49-03.csv'
maze_df = pd.read_csv(maze_file_path, header=None, names=["cell", "E", "W", "N", "S"])
maze_array = maze_df.values

# Load obstacles from random_obstacles.csv
obstacles_df = pd.read_csv('D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/normal Version/random_obstacles.csv')
obstacles = set(map(tuple, obstacles_df.values))  # Convert to tuples for hashability

# Dimensions of the maze
rows, cols = maze_array.shape

def heuristic(a, b):
    """ Manhattan distance as heuristic for A* """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(maze, start, goal, obstacles):
    open_list = []
    closed_set = set()
    came_from = {}
    g_score = {tuple(cell): float('inf') for cell in maze}
    g_score[tuple(start)] = 0
    f_score = {tuple(cell): float('inf') for cell in maze}
    f_score[tuple(start)] = heuristic(start, goal)
    
    heapq.heappush(open_list, (f_score[tuple(start)], tuple(start)))
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == tuple(goal):
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(tuple(start))
            path.reverse()
            return path
        
        closed_set.add(current)
        
        neighbors = get_neighbors(current, maze, obstacles)
        
        for neighbor in neighbors:
            if neighbor in closed_set:
                continue
            
            tentative_g_score = g_score[current] + 1
            
            if neighbor not in open_list or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None  # No path found

def get_neighbors(node, maze, obstacles):
    """ Get valid neighbors considering maze walls and obstacles """
    x, y = node
    neighbors = []
    
    # Directions: N, S, E, W
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and maze[nx, ny] == 0 and (nx, ny) not in obstacles:
            neighbors.append((nx, ny))
    
    return neighbors

# Run the A* algorithm
start = (2, 2)  # Example start position
goal = (47, 95)  # Example goal position

print("Maze:")
print(maze_array)
print("Obstacles:")
print(obstacles)
print("Start:", start)
print("Goal:", goal)

path = a_star(maze_array, start, goal, obstacles)

if path:
    print("Path found:", path)
else:
    print("No path found.")
