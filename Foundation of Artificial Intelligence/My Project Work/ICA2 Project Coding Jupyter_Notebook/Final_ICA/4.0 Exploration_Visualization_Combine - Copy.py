# %%
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque
import heapq

# %%
# Helper function to get the next cell based on direction
def get_next_cell(current, direction):
    """
    Returns the coordinates of the neighboring cell based on the direction.
    Directions are 'E' (East), 'W' (West), 'S' (South), 'N' (North).
    """
    row, col = current
    if direction == 'E':  # Move East
        return (row, col + 1)
    elif direction == 'W':  # Move West
        return (row, col - 1)
    elif direction == 'S':  # Move South
        return (row + 1, col)
    elif direction == 'N':  # Move North
        return (row - 1, col)

# %%
# BFS Algorithm
def BFS_search(maze_obj, start=None, goal=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    if goal is None:
        goal = maze_obj._goal  # Use the current goal position if none is passed

    frontier = deque([start])
    visited = {}
    exploration_order = []
    explored = set([start])

    while frontier:
        current = frontier.popleft()
        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                if next_cell not in explored:
                    frontier.append(next_cell)
                    explored.add(next_cell)
                    visited[next_cell] = current
                    exploration_order.append(next_cell)

    # Ensure goal cell is not included in exploration order
    if goal in exploration_order:
        exploration_order.remove(goal)

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

# %%
# Greedy BFS Algorithm (Greedy Best-First Search)
def heuristic(cell, goal):
    """
    Calculate the Manhattan distance from the current cell to the goal.
    This is the heuristic function used in Greedy BFS.
    """
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])
# %%
def greedy_bfs(m, start=None, goal=None):
    if start is None:
        start = (m.rows, m.cols)
    if goal is None:
        goal = m._goal  # Use the current goal position if none is passed

    f_costs = {cell: float('inf') for cell in m.grid}
    f_costs[start] = heuristic(start, goal)

    priority_queue = []
    heapq.heappush(priority_queue, (f_costs[start], start))

    came_from = {}
    exploration_order = []
    while priority_queue:
        _, current_cell = heapq.heappop(priority_queue)
        if current_cell == goal:
            break

        for d in 'ESNW':
            if m.maze_map[current_cell][d] == True:
                if d == 'E':  # East
                    neighbor_cell = (current_cell[0], current_cell[1] + 1)
                elif d == 'W':  # West
                    neighbor_cell = (current_cell[0], current_cell[1] - 1)
                elif d == 'S':  # South
                    neighbor_cell = (current_cell[0] + 1, current_cell[1])
                elif d == 'N':  # North
                    neighbor_cell = (current_cell[0] - 1, current_cell[1])

                if neighbor_cell not in came_from:
                    came_from[neighbor_cell] = current_cell
                    f_costs[neighbor_cell] = heuristic(neighbor_cell, goal)
                    heapq.heappush(priority_queue, (f_costs[neighbor_cell], neighbor_cell))
                    exploration_order.append(neighbor_cell)

    # Ensure goal cell is not included in exploration order
    if goal in exploration_order:
        exploration_order.remove(goal)

    path_to_goal = []
    cell = goal
    while cell != start:
        path_to_goal.append(cell)
        cell = came_from[cell]
    path_to_goal.append(start)
    path_to_goal.reverse()

    return exploration_order, came_from, path_to_goal

# %%
# A* Algorithm
def A_star_search(maze_obj, start=None, goal=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    if goal is None:
        goal = maze_obj._goal  # Use the current goal position if none is passed

    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")

    frontier = []
    heapq.heappush(frontier, (0 + heuristic(start, goal), start))
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
                new_g_cost = g_costs[current] + 1

                if next_cell not in explored or new_g_cost < g_costs.get(next_cell, float('inf')):
                    g_costs[next_cell] = new_g_cost
                    f_cost = new_g_cost + heuristic(next_cell, goal)
                    heapq.heappush(frontier, (f_cost, next_cell))
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
                    explored.add(next_cell)

    # Ensure goal cell is not included in exploration order
    if goal in exploration_order:
        exploration_order.remove(goal)

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

# %%
# Main Function
if __name__ == '__main__':
    # Create the maze
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')

    goal_position = (1, 1)  # You can change this to any valid goal position
    m._goal = goal_position  # Set the goal position in the maze

    # Perform BFS search on the maze
    exploration_order_bfs, visited_cells_bfs, path_to_goal_bfs = BFS_search(m, goal=goal_position)

    # Perform Greedy BFS search on the maze
    exploration_order_greedy, came_from_greedy, path_to_goal_greedy = greedy_bfs(m, goal=goal_position)

    # Perform A* search on the maze
    exploration_order_astar, visited_cells_astar, path_to_goal_astar = A_star_search(m, goal=goal_position)

    # Create agents to visualize the search processes
    agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red, filled='full')  # BFS
    agent_astar = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled='full')  # A*
    agent_greedyBFS = agent(m, footprints=True, shape='square', color=COLOR.blue, filled='full')  # Greedy BFS

    # Visualize BFS search order
    m.tracePath({agent_bfs: exploration_order_bfs}, delay=1)
    # Visualize A* search order
    m.tracePath({agent_astar: exploration_order_astar}, delay=1)
    # Visualize Greedy BFS search order
    m.tracePath({agent_greedyBFS: exploration_order_greedy}, delay=1)

    

    # Run the maze visualization
    m.run()
