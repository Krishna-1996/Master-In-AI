import heapq
import time
from pyamaze import maze, agent, COLOR
import tkinter as tk
from tkinter import ttk  # For using the table-like grid in Tkinter
import math

# Define heuristic functions
def manhattan_heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def chebyshev_heuristic(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

# Directional weights
directional_weights = {
    'N': 15,  # Moving north costs
    'E': 15,  # Moving east costs
    'S': -10,  # Moving south costs
    'W': -10,  # Moving west costs
}

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

# A* search algorithm
def A_star_search(maze_obj, start=None, goal=None, heuristic_method=manhattan_heuristic):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

    frontier = []
    heapq.heappush(frontier, (0 + heuristic_method(start, goal), start))  # (f-cost, position)
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
                move_cost = directional_weights[direction]  # Use directional weight
                new_g_cost = g_costs[current] + move_cost

                if next_cell not in explored or new_g_cost < g_costs.get(next_cell, float('inf')):
                    g_costs[next_cell] = new_g_cost
                    f_cost = new_g_cost + heuristic_method(next_cell, goal)
                    heapq.heappush(frontier, (f_cost, next_cell))
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
                    explored.add(next_cell)

    return exploration_order

# Function to update the Tkinter window with heuristic data
def update_info_window(heuristic_name, search_length, weights):
    info_window = tk.Tk()
    info_window.title(f"{heuristic_name} Heuristic Information")

    table = ttk.Treeview(info_window, columns=("Metric", "Value"), show="headings")
    table.heading("Metric", text="Metric")
    table.heading("Value", text="Value")

    # Add information to the table
    table.insert("", "end", values=("Directional Weights", f"N={weights['N']}, E={weights['E']}, S={weights['S']}, W={weights['W']}"))
    table.insert("", "end", values=("Heuristic", heuristic_name))
    table.insert("", "end", values=("Search Length", search_length))

    table.pack(fill=tk.BOTH, expand=True)
    info_window.mainloop()

# Main function
if __name__ == '__main__':
    m = maze()  # Adjust maze size for testing
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/TEST_MAZE.csv')  # Adjust maze file path

    goal_position = (1, 1)  # Example goal position

    # Choose heuristic
    heuristic_function = manhattan_heuristic  # Change to euclidean_heuristic or chebyshev_heuristic for other heuristics

    start_time = time.time()
    exploration_order = A_star_search(m, goal=goal_position, heuristic_method=heuristic_function)
    end_time = time.time()

    search_length = len(exploration_order)

    # Create agent
    agent_algorithm = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)

    # Trace exploration path
    m.tracePath({agent_algorithm: exploration_order}, delay=1)

    # Update Tkinter window with information
    update_info_window(heuristic_function.__name__, search_length, directional_weights)

    m.run()
