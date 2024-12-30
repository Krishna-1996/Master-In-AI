# m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/My_Maze.csv')  

import heapq
import time
from pyamaze import maze, agent, COLOR
import tkinter as tk
from tkinter import ttk  # For using the table-like grid in Tkinter

# Manhattan Heuristic Function
def manhattan_heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

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

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

# Function to update the Tkinter window with heuristic data
def update_info_window(heuristic_name, goal_position, path_length, search_length, execution_time, weights):
    info_window = tk.Tk()
    info_window.title(f"{heuristic_name} Heuristic Information")

    table = ttk.Treeview(info_window, columns=("Metric", "Value"), show="headings")
    table.heading("Metric", text="Metric")
    table.heading("Value", text="Value")

    # Add information to the table
    table.insert("", "end", values=("Directional Weights", f"N={weights['N']}, E={weights['E']}, S={weights['S']}, W={weights['W']}"))
    table.insert("", "end", values=("Heuristic", heuristic_name))
    table.insert("", "end", values=("Goal Position", str(goal_position)))
    table.insert("", "end", values=("Path Length", path_length))
    table.insert("", "end", values=("Search Length", search_length))
    table.insert("", "end", values=("Execution Time (s)", round(execution_time, 4)))

    table.pack(fill=tk.BOTH, expand=True)
    info_window.mainloop()

# Main function
if __name__ == '__main__':
    m = maze()  # Adjust maze size for testing
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/TEST_MAZE.csv') # Automatically generate the maze

    goal_position = (1, 1)  # Example goal position

    start_time = time.time()
    exploration_order, visited_cells, path_to_goal = A_star_search(m, goal=goal_position, heuristic_method=manhattan_heuristic)
    end_time = time.time()

    execution_time = end_time - start_time
    search_length = len(exploration_order)
    path_length = len(path_to_goal) + 1

    # Create agents
    agent_explore = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
    agent_trace = agent(m, footprints=True, shape='square', color=COLOR.blue, filled=True)
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

    # Trace paths
    m.tracePath({agent_explore: exploration_order}, delay=1)
    m.tracePath({agent_trace: path_to_goal}, delay=1)

    # Update Tkinter window with information
    update_info_window("Manhattan", goal_position, path_length, search_length, execution_time, directional_weights)

    m.run()
