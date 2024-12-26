# 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')

import heapq
import time
from pyamaze import maze, agent, COLOR

# Manhattan Heuristic Function
def manhattan_heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Directional weights
directional_weights = {
    'N': 1,  # Moving north costs 1
    'E': 1,  # Moving east costs 1
    'S': 5,  # Moving south costs 5
    'W': 2,  # Moving west costs 2
}

# Get next cell in the maze based on direction
def get_next_cell_with_direction(current, direction):
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

# Weighted cost function with direction
def get_directional_cost(direction, cell, weighted_map):
    # Base cost depends on direction
    direction_cost = directional_weights.get(direction, 1)  # Default cost is 1 if direction not defined
    cell_cost = weighted_map.get(cell, 0)  # Add additional terrain cost if applicable
    return direction_cost + cell_cost

# A* search algorithm with Directional and Weighted Costs
def A_star_search_directional(maze_obj, start=None, goal=None, heuristic_method=manhattan_heuristic, weighted_map=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

    if weighted_map is None:
        weighted_map = {}

    # Min-heap priority queue
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
                next_cell = get_next_cell_with_direction(current, direction)
                move_cost = get_directional_cost(direction, next_cell, weighted_map)  # Cost depends on direction
                new_g_cost = g_costs[current] + move_cost  # Cost = g_cost + move cost

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

# Main function
if __name__ == '__main__':
    # Create maze and set goal position
    m = maze(20, 20)  # Adjust maze size for testing
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/My_Maze.csv')  # Path updated  # Automatically generate the maze

    goal_position = (1, 1)  # Example goal position

    # Define a weighted map for cells (optional)
    weighted_map = {
        (5, 5): 5,  # Example: "Mud" cell with cost 5
        (10, 10): 10,  # Example: "Water" cell with cost 10
        (15, 15): 2,  # Example: A slightly harder terrain
    }

    start_time = time.time()
    exploration_order, visited_cells, path_to_goal = A_star_search_directional(
        m, goal=goal_position, heuristic_method=manhattan_heuristic, weighted_map=weighted_map
    )
    end_time = time.time()

    execution_time = end_time - start_time
    search_length = len(exploration_order)
    path_length = len(path_to_goal) + 1  # Include the goal cell

    # Visualization setup for agents
    agent_explore = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)  # Exploration path
    agent_trace = agent(m, footprints=True, shape='square', color=COLOR.blue, filled=True)  # Path to goal
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)  # Goal

    m.tracePath({agent_explore: exploration_order}, delay=10)
    m.tracePath({agent_trace: path_to_goal}, delay=10)

    print(f"Directional Weighted Costs - Heuristic: Manhattan")
    print(f"Goal Position: {goal_position}")
    print(f"Path Length: {path_length}")
    print(f"Search Length: {search_length}")
    print(f"Execution Time: {round(execution_time, 4)} seconds")

    m.run()
