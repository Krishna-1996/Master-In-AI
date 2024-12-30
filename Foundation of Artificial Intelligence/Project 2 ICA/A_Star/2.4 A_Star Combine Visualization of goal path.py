import heapq
import math
import tkinter as tk
from tkinter import ttk
from pyamaze import maze, agent, COLOR

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

# A* search algorithm with a specified heuristic
def a_star_search(maze_obj, start=None, goal=None, heuristic_method=manhattan_heuristic):
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
    g_costs = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                new_g_cost = g_costs[current] + 1  # +1 for each move (uniform cost)
                
                if next_cell not in explored or new_g_cost < g_costs.get(next_cell, float('inf')):
                    g_costs[next_cell] = new_g_cost
                    f_cost = new_g_cost + heuristic_method(next_cell, goal)
                    heapq.heappush(frontier, (f_cost, next_cell))
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

# Function to display the heuristic data in a tabular format using Tkinter Treeview
def display_info_window(path_lengths):
    # Create a new Tkinter window for displaying the results
    info_window = tk.Tk()
    info_window.title("Path Length Comparison for A* Heuristics")

    # Create a table (Treeview) widget for displaying the metrics
    table = ttk.Treeview(info_window, columns=("Heuristic", "Path Length"), show="headings")
    table.heading("Heuristic", text="Heuristic")
    table.heading("Path Length", text="Path Length")

    # Insert the data into the table for all three heuristics
    for heuristic, length in path_lengths.items():
        table.insert("", "end", values=(heuristic, length))

    # Pack the table into the window
    table.pack(fill=tk.BOTH, expand=True)

    # Run the Tkinter main loop to display the window
    info_window.mainloop()

# Main function
if __name__ == '__main__':
    # Create the maze
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/My_Maze.csv')  # Update with correct path

    goal_position = (1, 1)  # Example goal position

    # Run A* for Manhattan Heuristic
    exploration_order_manhattan, visited_manhattan, path_to_goal_manhattan = a_star_search(m, goal=goal_position, heuristic_method=manhattan_heuristic)
    path_length_manhattan = len(path_to_goal_manhattan) + 1  # Include the goal cell

    # Run A* for Euclidean Heuristic
    exploration_order_euclidean, visited_euclidean, path_to_goal_euclidean = a_star_search(m, goal=goal_position, heuristic_method=euclidean_heuristic)
    path_length_euclidean = len(path_to_goal_euclidean) + 1

    # Run A* for Chebyshev Heuristic
    exploration_order_chebyshev, visited_chebyshev, path_to_goal_chebyshev = a_star_search(m, goal=goal_position, heuristic_method=chebyshev_heuristic)
    path_length_chebyshev = len(path_to_goal_chebyshev) + 1

    # Visualization setup for agents
    agent_goal_manhattan = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)  # Red path for Manhattan
    agent_goal_euclidean = agent(m, footprints=True, shape='square', color=COLOR.blue, filled=True)  # Blue path for Euclidean
    agent_goal_chebyshev = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)  # Yellow path for Chebyshev

    # Trace paths
    m.tracePath({agent_goal_manhattan: path_to_goal_manhattan}, delay=1)
    m.tracePath({agent_goal_euclidean: path_to_goal_euclidean}, delay=1)
    m.tracePath({agent_goal_chebyshev: path_to_goal_chebyshev}, delay=1)

    # Store the path lengths for each heuristic
    path_lengths = {
        "Manhattan": path_length_manhattan,
        "Euclidean": path_length_euclidean,
        "Chebyshev": path_length_chebyshev
    }

    # Display the information in tabular format for all three heuristics
    display_info_window(path_lengths)

    # Run the PyAmaze visualization (ensure this is the last line)
    m.run()
