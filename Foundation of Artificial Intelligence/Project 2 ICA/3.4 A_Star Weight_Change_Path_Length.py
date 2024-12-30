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
    path_to_goal = {}
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
                    explored.add(next_cell)

    # Trace the path to the goal
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return path_to_goal

# Function to update the Tkinter window with heuristic data
def update_info_window(path_lengths, weights):
    info_window = tk.Tk()
    info_window.title("Heuristic Path Lengths")

    table = ttk.Treeview(info_window, columns=("Heuristic", "Path Length"), show="headings")
    table.heading("Heuristic", text="Heuristic")
    table.heading("Path Length", text="Path Length")

    # Add directional weights and heuristic data
    table.insert("", "end", values=("Directional Weights", f"N={weights['N']}, E={weights['E']}, S={weights['S']}, W={weights['W']}"))
    
    for heuristic, length in path_lengths.items():
        table.insert("", "end", values=(heuristic, length))

    table.pack(fill=tk.BOTH, expand=True)
    info_window.mainloop()

# Main function
if __name__ == '__main__':
    m = maze()  # Adjust maze size for testing
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/TEST_MAZE.csv')  # Adjust maze file path

    start_position = (m.rows, m.cols)  # Start at the bottom-right corner
    goal_position = (1, 1)  # Example goal position

    # Dictionary to store path lengths for each heuristic
    path_lengths = {}

    # Create a list of agent objects (one for each heuristic)
    agent_goal_manhattan = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)
    agent_goal_euclidean = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.blue, shape='square', filled=True)
    agent_goal_chebyshev = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.red, shape='square', filled=True)

    # Loop through each heuristic
    for heuristic_function, agent_goal in zip([manhattan_heuristic, euclidean_heuristic, chebyshev_heuristic],
                                               [agent_goal_manhattan, agent_goal_euclidean, agent_goal_chebyshev]):
        start_time = time.time()
        path_to_goal = A_star_search(m, start=start_position, goal=goal_position, heuristic_method=heuristic_function)
        end_time = time.time()

        path_length = len(path_to_goal) + 1  # Include the goal cell

        # Prepare the path in the correct order for tracing
        path = [start_position]  # Start with the start position
        while path[-1] != goal_position:
            current = path[-1]
            if current in path_to_goal:
                path.append(path_to_goal[current])
            else:
                break  # If path is not valid (shouldn't happen)

        # Trace the path for each agent
        print(f"Tracing path for {heuristic_function.__name__}...")
        m.tracePath({agent_goal: path}, delay=0.5)  # Adjust delay for better visualization of the agent

        # Store the path length for the current heuristic
        path_lengths[heuristic_function.__name__] = path_length

    # Update Tkinter window with all heuristic path lengths
    update_info_window(path_lengths, directional_weights)

    # Run the maze after all paths have been traced
    m.run()
