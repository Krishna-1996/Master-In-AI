from pyamaze import maze, agent, COLOR, textLabel
import heapq
import math
import time

# Manhattan Heuristic Function
def manhattan_heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

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

# A* search algorithm with Manhattan heuristic
def A_star_search(maze_obj, start=None, goal=None, heuristic_method=manhattan_heuristic):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

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

# Main function
if __name__ == '__main__':
    # Create maze and set goal position
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')

    goal_position = (1, 1)  # Example goal, change to any valid coordinate

    start_time = time.time()
    exploration_order, visited_cells, path_to_goal = A_star_search(m, goal=goal_position, heuristic_method=manhattan_heuristic)
    end_time = time.time()

    execution_time = end_time - start_time
    search_length = len(exploration_order)
    path_length = len(path_to_goal) + 1  # Include the goal cell

    # Visualization setup
    agent_explore = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)  # Exploration path (red circle)
    agent_trace = agent(m, footprints=True, shape='square', color=COLOR.blue, filled=True)  # Path to goal (blue star)
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)  # Goal (green square)

    m.tracePath({agent_explore: exploration_order}, delay=1)
    m.tracePath({agent_trace: path_to_goal}, delay=1)
    m.tracePath({agent_goal: visited_cells}, delay=1)

    # Display heuristic information
    x_position = 20
    y_position = 50
    textLabel(m, f'Manhattan Heuristic - Goal Position: {goal_position}', (x_position, y_position))
    y_position += 30
    textLabel(m, f'Manhattan Heuristic - A* Path Length: {path_length}', (x_position, y_position))
    y_position += 30
    textLabel(m, f'Manhattan Heuristic - A* Search Length: {search_length}', (x_position, y_position))
    y_position += 30
    textLabel(m, f'Manhattan Heuristic - Execution Time (s): {round(execution_time, 4)}', (x_position, y_position))

    m.run()
