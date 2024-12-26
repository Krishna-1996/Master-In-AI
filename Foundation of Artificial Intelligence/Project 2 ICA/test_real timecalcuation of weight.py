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
    'N': 15,  # Moving north adds 15
    'E': 15,  # Moving east adds 15
    'S': -10,  # Moving south subtracts 10
    'W': -10,  # Moving west subtracts 10
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


# Real-time weight tracker with path tracing
def trace_path_with_weights(m, agent, path, weights, label, delay):
    total_weight = 0  # Initialize total weight
    current_cell = path[0]

    for next_cell in path[1:]:
        # Determine direction and calculate the weight
        if next_cell[0] == current_cell[0] - 1:  # Moved north
            direction = 'N'
        elif next_cell[0] == current_cell[0] + 1:  # Moved south
            direction = 'S'
        elif next_cell[1] == current_cell[1] + 1:  # Moved east
            direction = 'E'
        elif next_cell[1] == current_cell[1] - 1:  # Moved west
            direction = 'W'
        else:
            direction = None

        # Update weight
        if direction:
            move_weight = weights[direction]
            total_weight += move_weight
            label.config(
                text=f"Current Direction: {direction}, Move Weight: {move_weight}, Total Weight: {total_weight}"
            )
            label.update()  # Update label in real-time

        # Simulate movement with tracePath
        m.tracePath({agent: [next_cell]}, delay=delay)
        time.sleep(delay / 10)  # Pause to simulate movement

        # Update current cell
        current_cell = next_cell

    # Final update to show the total weight
    label.config(
        text=f"Path traversal completed. Final Total Weight: {total_weight}"
    )
    label.update()


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

    # Add weight tracker
    weight_label = tk.Label(info_window, text="Weight Calculation in Progress...")
    weight_label.pack()

    return weight_label, info_window


# Main function
if __name__ == '__main__':
    delay = 1  # Change delay time to control visualization speed

    m = maze(20, 20)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/TEST_MAZE.csv')

    goal_position = (1, 1)  # Example goal position

    start_time = time.time()
    exploration_order, visited_cells, path_to_goal = A_star_search(m, goal=goal_position, heuristic_method=manhattan_heuristic)
    end_time = time.time()

    execution_time = end_time - start_time

    # Adjust execution time based on delay
    adjusted_time = execution_time if delay == 1 else execution_time / delay

    search_length = len(exploration_order)
    path_length = len(path_to_goal) + 1

    # Create agents
    agent_explore = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
    agent_trace = agent(m, footprints=True, shape='square', color=COLOR.blue, filled=True)

    # Update Tkinter window with information
    weight_label, info_window = update_info_window("Manhattan", goal_position, path_length, search_length, adjusted_time, directional_weights)

    # Trace exploration path
    m.tracePath({agent_explore: exploration_order}, delay=delay)

    # Trace final path with weight updates
    trace_path_with_weights(
        m,
        agent_trace,
        list(path_to_goal.keys()) + [goal_position],
        directional_weights,
        weight_label,
        delay
    )

    info_window.mainloop()
    m.run()
