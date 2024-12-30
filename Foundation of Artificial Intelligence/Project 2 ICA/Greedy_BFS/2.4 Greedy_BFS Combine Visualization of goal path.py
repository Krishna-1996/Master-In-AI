import heapq
import time
import math
from pyamaze import maze, agent, COLOR, textLabel
import tkinter as tk
from tkinter import ttk

# Manhattan Heuristic Function
def manhattan_heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Euclidean Heuristic Function
def euclidean_heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Chebyshev Heuristic Function
def chebyshev_heuristic(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

# Get next cell in the maze based on direction
def get_next_cell(current, direction):
    x, y = current
    if direction == 'E':  # Move east
        return (x, y + 1)
    elif direction == 'W':  # Move west
        return (x, y - 1)
    elif direction == 'N':  # Move north
        return (x - 1, y)
    elif direction == 'S':  # Move south
        return (x + 1, y)
    return current

# Greedy BFS search algorithm with a specified heuristic
def greedy_bfs_search(maze_obj, start=None, goal=None, heuristic_method=manhattan_heuristic):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

    # Min-heap priority queue
    frontier = []
    heapq.heappush(frontier, (heuristic_method(start, goal), start))  # (f-cost, position)
    visited = {}
    exploration_order = []
    explored = set([start])

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                
                if next_cell not in explored:
                    heapq.heappush(frontier, (heuristic_method(next_cell, goal), next_cell))
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
                    explored.add(next_cell)

    # Reconstruct the path to the goal
    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

# Function to display the heuristic data as text labels in the maze window
def display_text_labels(m, heuristic_name, goal_position, path_length, search_length, execution_time):
    # Position where the text labels will appear
    x_position = 20
    y_position = 50  # Starting y-position for text labels

    # Display each information as text labels
    textLabel(m, f'{heuristic_name} Heuristic - Goal Position: {goal_position}', (x_position, y_position))
    y_position += 30
    textLabel(m, f'{heuristic_name} Heuristic - Path Length: {path_length}', (x_position, y_position))
    y_position += 30
    textLabel(m, f'{heuristic_name} Heuristic - Search Length: {search_length}', (x_position, y_position))
    y_position += 30
    textLabel(m, f'{heuristic_name} Heuristic - Execution Time (s): {round(execution_time, 4)}', (x_position, y_position))

# Main function
if __name__ == '__main__':
    # Create Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the Tkinter main window (we don't need it)

    # Create maze and set goal position
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/My_Maze.csv')  # Update with correct path

    goal_position = (1, 1)  # Example goal position

    # Start Timer for Manhattan Heuristic
    start_time = time.time()
    exploration_order_manhattan, visited_manhattan, path_to_goal_manhattan = greedy_bfs_search(m, goal=goal_position, heuristic_method=manhattan_heuristic)
    end_time = time.time()
    execution_time_manhattan = end_time - start_time
    search_length_manhattan = len(exploration_order_manhattan)
    path_length_manhattan = len(path_to_goal_manhattan) + 1  # Include the goal cell

    # Start Timer for Euclidean Heuristic
    start_time = time.time()
    exploration_order_euclidean, visited_euclidean, path_to_goal_euclidean = greedy_bfs_search(m, goal=goal_position, heuristic_method=euclidean_heuristic)
    end_time = time.time()
    execution_time_euclidean = end_time - start_time
    search_length_euclidean = len(exploration_order_euclidean)
    path_length_euclidean = len(path_to_goal_euclidean) + 1  # Include the goal cell

    # Start Timer for Chebyshev Heuristic
    start_time = time.time()
    exploration_order_chebyshev, visited_chebyshev, path_to_goal_chebyshev = greedy_bfs_search(m, goal=goal_position, heuristic_method=chebyshev_heuristic)
    end_time = time.time()
    execution_time_chebyshev = end_time - start_time
    search_length_chebyshev = len(exploration_order_chebyshev)
    path_length_chebyshev = len(path_to_goal_chebyshev) + 1  # Include the goal cell

    # Visualization setup for agents
    agent_goal_manhattan = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)  # Path with Manhattan Heuristic (Red)
    agent_goal_euclidean = agent(m, footprints=True, shape='square', color=COLOR.blue, filled=True)  # Path with Euclidean Heuristic (Blue)
    agent_goal_chebyshev = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)  # Path with Chebyshev Heuristic (Yellow)

    m.tracePath({agent_goal_manhattan: path_to_goal_manhattan}, delay=1)
    m.tracePath({agent_goal_euclidean: path_to_goal_euclidean}, delay=1)
    m.tracePath({agent_goal_chebyshev: path_to_goal_chebyshev}, delay=1)

    # Display heuristic information in a separate Tkinter window
    display_text_labels(m, "Manhattan", goal_position, path_length_manhattan, search_length_manhattan, execution_time_manhattan)
    display_text_labels(m, "Euclidean", goal_position, path_length_euclidean, search_length_euclidean, execution_time_euclidean)
    display_text_labels(m, "Chebyshev", goal_position, path_length_chebyshev, search_length_chebyshev, execution_time_chebyshev)

    m.run()
