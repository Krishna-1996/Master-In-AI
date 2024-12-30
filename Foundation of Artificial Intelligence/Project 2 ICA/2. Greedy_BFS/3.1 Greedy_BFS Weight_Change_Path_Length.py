import heapq
import math
from pyamaze import maze, agent, COLOR
import tkinter as tk
from tkinter import ttk

# Heuristic Functions
def manhattan_heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def chebyshev_heuristic(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

# Directional weights
directional_weights = {
    'N': -200,  # Moving north costs
    'E': 110,  # Moving east costs
    'S': 110,  # Moving south costs
    'W': -20,  # Moving west costs
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

# Greedy BFS search algorithm
def greedy_bfs_search(maze_obj, start=None, goal=None, heuristic_method=manhattan_heuristic):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

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
                move_cost = directional_weights[direction]  # Use directional weight
                new_g_cost = g_costs[current] + move_cost

                if next_cell not in explored:
                    g_costs[next_cell] = new_g_cost
                    f_cost = heuristic_method(next_cell, goal)  # Greedy BFS uses only heuristic
                    heapq.heappush(frontier, (f_cost, next_cell))
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
                    explored.add(next_cell)

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    # Return the path to the goal and the path length
    return exploration_order, visited, path_to_goal, len(path_to_goal) + 1  # Include the goal cell

# Function to update the Tkinter window with heuristic data
def update_info_window(path_lengths):
    info_window = tk.Tk()
    info_window.title(f"Heuristic Comparison Information")

    table = ttk.Treeview(info_window, columns=("Heuristic", "Path Length"), show="headings")
    table.heading("Heuristic", text="Heuristic")
    table.heading("Path Length", text="Path Length")

    # Add information to the table
    for heuristic_name, path_length in path_lengths.items():
        table.insert("", "end", values=(heuristic_name, path_length))

    table.pack(fill=tk.BOTH, expand=True)
    info_window.mainloop()

# Main function
def run_maze_with_all_heuristics():
    m = maze()  # Adjust maze size for testing
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/My_Maze_2.csv')
    # m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/My_Maze.csv')  # Adjust maze file path

    goal_position = (1, 1)  # Example goal position
    start_position = (m.rows, m.cols)

    # Run for all heuristics
    heuristics = [
        (manhattan_heuristic, "Manhattan", COLOR.red),
        (euclidean_heuristic, "Euclidean", COLOR.green),
        (chebyshev_heuristic, "Chebyshev", COLOR.blue)
    ]

    path_lengths = {}

    # Create agents for each heuristic
    agent_explore_manhattan = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
    agent_explore_euclidean = agent(m, footprints=True, shape='square', color=COLOR.green, filled=True)
    agent_explore_chebyshev = agent(m, footprints=True, shape='square', color=COLOR.blue, filled=True)

    # Trace paths for each heuristic
    for heuristic, name, color in heuristics:
        # Run Greedy BFS search for each heuristic
        exploration_order, visited_cells, path_to_goal, path_length = greedy_bfs_search(m, start=start_position, goal=goal_position, heuristic_method=heuristic)
        
        # Trace the optimal path (only) for this heuristic in its designated color
        if name == "Manhattan":
            m.tracePath({agent_explore_manhattan: path_to_goal}, delay=1)
        elif name == "Euclidean":
            m.tracePath({agent_explore_euclidean: path_to_goal}, delay=1)
        elif name == "Chebyshev":
            m.tracePath({agent_explore_chebyshev: path_to_goal}, delay=1)

        # Store the path length
        path_lengths[name] = path_length

    # Update Tkinter window with path lengths for all heuristics
    update_info_window(path_lengths)

    m.run()

# Run the main function
if __name__ == '__main__':
    run_maze_with_all_heuristics()
