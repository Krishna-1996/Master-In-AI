# 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv'
import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Load the maze from the CSV
def load_maze(file_path):
    maze = np.genfromtxt(file_path, delimiter=',', dtype=int)
    return maze

# BFS Algorithm
def bfs(maze, start, goal):
    rows, cols = maze.shape
    queue = deque([start])
    parent = {start: None}
    path = []

    while queue:
        current = queue.popleft()
        if current == goal:
            while current:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path

        # Explore neighbors (N, S, E, W)
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_cell = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= next_cell[0] < rows and 0 <= next_cell[1] < cols and maze[next_cell] == 0 and next_cell not in parent:
                queue.append(next_cell)
                parent[next_cell] = current
    return path

# Greedy BFS with Manhattan Distance Heuristic
def greedy_bfs(maze, start, goal):
    rows, cols = maze.shape
    frontier = []
    heapq.heappush(frontier, (0, start))  # (heuristic, position)
    parent = {start: None}
    path = []

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            while current:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path

        # Explore neighbors (N, S, E, W)
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_cell = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= next_cell[0] < rows and 0 <= next_cell[1] < cols and maze[next_cell] == 0 and next_cell not in parent:
                heapq.heappush(frontier, (heuristic(next_cell, goal), next_cell))
                parent[next_cell] = current
    return path

# A* Algorithm
def a_star(maze, start, goal):
    rows, cols = maze.shape
    frontier = []
    heapq.heappush(frontier, (0, start))  # (cost + heuristic, position)
    g_cost = {start: 0}
    parent = {start: None}
    path = []

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            while current:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path

        # Explore neighbors (N, S, E, W)
        for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_cell = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= next_cell[0] < rows and 0 <= next_cell[1] < cols and maze[next_cell] == 0:
                new_cost = g_cost[current] + 1
                if next_cell not in g_cost or new_cost < g_cost[next_cell]:
                    g_cost[next_cell] = new_cost
                    heapq.heappush(frontier, (new_cost + heuristic(next_cell, goal), next_cell))
                    parent[next_cell] = current
    return path

# Visualization
def plot_paths(maze, bfs_path, greedy_path, astar_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(maze, cmap="gray_r")  # Display maze with walls in black

    # Mark paths
    for (i, j) in bfs_path:
        ax.plot(j, i, color='blue', marker='o', markersize=4)
    for (i, j) in greedy_path:
        ax.plot(j, i, color='red', marker='o', markersize=4)
    for (i, j) in astar_path:
        ax.plot(j, i, color='green', marker='o', markersize=4)

    ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    plt.gca().invert_yaxis()  # Invert the y-axis to match the grid's origin
    plt.title('Pathfinding Algorithms: BFS (Blue), Greedy BFS (Red), A* (Green)')
    plt.show()

# Main function to run the algorithms and visualize results
def main():
    maze_file = 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv'  # Provide your maze file path here
    maze = load_maze(maze_file)

    start = (49, 119)  # Bottom-right corner
    goal1 = (0, 0)     # Example goal 1
    goal2 = (48, 1)    # Example goal 2
    goal3 = (0, 118)   # Example goal 3

    # Run algorithms for each goal position
    bfs_path = bfs(maze, start, goal1)
    greedy_path = greedy_bfs(maze, start, goal1)
    astar_path = a_star(maze, start, goal1)

    # Visualize the paths for each algorithm
    plot_paths(maze, bfs_path, greedy_path, astar_path)

if __name__ == "__main__":
    main()
