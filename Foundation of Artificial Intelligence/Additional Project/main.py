import heapq
import pandas as pd
import random
from pyamaze import maze, agent, COLOR

# Heuristic Functions
def manhattan_heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_heuristic(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def chebyshev_heuristic(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

# Directional weights
directional_weights = {
    'N': -10,  # Moving north costs
    'E': 10,  # Moving east costs
    'S': 10,  # Moving south costs
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
def A_star_search(maze_obj, start=None, goal=None, directional_weights=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

    frontier = []
    heapq.heappush(frontier, (0 + manhattan_heuristic(start, goal), start))  # (f-cost, position)
    visited = {}
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
                    f_cost = new_g_cost + manhattan_heuristic(next_cell, goal)
                    heapq.heappush(frontier, (f_cost, next_cell))
                    visited[next_cell] = current
                    explored.add(next_cell)

    # Reconstruct the path
    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    # Return the path to the goal and path length
    return path_to_goal, len(path_to_goal) + 1  # Include the goal cell

# Function to update Excel with results
def update_excel(results):
    df = pd.DataFrame(results, columns=["S.No", "Heuristic_Values", "Path_Length", "Search_Length"])
    df.to_excel("heuristic_results.xlsx", index=False)

# Function to run experiments with heuristic values from -10 to +10
def run_experiments():
    m = maze()  # Create maze
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/My_Maze.csv')

    # Range for heuristic values
    heuristic_range = range(-10, 11)  # Values from -10 to +10
    
    results = []
    serial_no = 1

    # Iterate through all combinations of heuristic values for N, E, S, W
    for n in heuristic_range:
        for e in heuristic_range:
            for s in heuristic_range:
                for w in heuristic_range:
                    directional_weights = {'N': n, 'E': e, 'S': s, 'W': w}

                    # Run A* search with the given directional weights
                    path_to_goal, path_length = A_star_search(m, start=(m.rows, m.cols), goal=(1, 1), directional_weights=directional_weights)

                    # Add result to list
                    results.append([serial_no, f"N:{n}, E:{e}, S:{s}, W:{w}", path_length, len(path_to_goal)])
                    serial_no += 1

                    # Print progress
                    if serial_no % 1000 == 0:
                        print(f"Processed {serial_no} combinations...")

    # Write the results to an Excel file
    update_excel(results)

# Main entry point
if __name__ == '__main__':
    run_experiments()
