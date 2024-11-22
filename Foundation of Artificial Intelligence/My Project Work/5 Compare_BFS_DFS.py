# Import necessary modules for maze creation and visualization
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque
import heapq  # For Dijkstra's and A* algorithms

# BFS search algorithm
def BFS_search(maze_obj, start=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    
    frontier = deque([start])
    visited = {}
    exploration_order = []
    explored = set([start])
    
    while frontier:
        current = frontier.popleft()
        
        if current == maze_obj._goal:
            break
        
        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                
                if next_cell not in explored:
                    frontier.append(next_cell)
                    explored.add(next_cell)
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
    
    path_to_goal = {}
    cell = maze_obj._goal
    while cell != (maze_obj.rows, maze_obj.cols):
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]
    
    return exploration_order, visited, path_to_goal

# DFS search algorithm
def dfs_search(maze_obj, start=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    stack = [start]
    visited = {}
    exploration_order = []
    explored = set([start])
    
    while stack:
        current = stack.pop()
        
        if current == maze_obj._goal:
            break
        
        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)

                if next_cell not in explored:
                    stack.append(next_cell)
                    explored.add(next_cell)
                    visited[next_cell] = current
                    exploration_order.append(next_cell)

    path_to_goal = {}
    cell = maze_obj._goal
    while cell != (maze_obj.rows, maze_obj.cols):
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

# A* search algorithm
def A_star_search(maze_obj, start=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    
    # Heuristic function: Manhattan distance
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_list = []
    heapq.heappush(open_list, (0, start))  # Push start node with priority 0
    g_score = {start: 0}  # Cost from start to current node
    f_score = {start: heuristic(start, maze_obj._goal)}  # Estimated cost from start to goal
    came_from = {}
    exploration_order = []

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == maze_obj._goal:
            break
        
        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                tentative_g_score = g_score[current] + 1

                if next_cell not in g_score or tentative_g_score < g_score[next_cell]:
                    came_from[next_cell] = current
                    g_score[next_cell] = tentative_g_score
                    f_score[next_cell] = tentative_g_score + heuristic(next_cell, maze_obj._goal)
                    heapq.heappush(open_list, (f_score[next_cell], next_cell))
                    exploration_order.append(next_cell)

    path_to_goal = {}
    cell = maze_obj._goal
    while cell != (maze_obj.rows, maze_obj.cols):
        path_to_goal[came_from[cell]] = cell
        cell = came_from[cell]

    return exploration_order, came_from, path_to_goal

# Dijkstra's search algorithm
def dijkstra_search(maze_obj, start=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    open_list = []
    heapq.heappush(open_list, (0, start))  # Push start node with priority 0
    dist = {start: 0}  # Distance from start to current node
    came_from = {}
    exploration_order = []

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == maze_obj._goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                tentative_dist = dist[current] + 1

                if next_cell not in dist or tentative_dist < dist[next_cell]:
                    came_from[next_cell] = current
                    dist[next_cell] = tentative_dist
                    heapq.heappush(open_list, (dist[next_cell], next_cell))
                    exploration_order.append(next_cell)

    path_to_goal = {}
    cell = maze_obj._goal
    while cell != (maze_obj.rows, maze_obj.cols):
        path_to_goal[came_from[cell]] = cell
        cell = came_from[cell]

    return exploration_order, came_from, path_to_goal

# Helper function to get the next cell based on the direction
def get_next_cell(current, direction):
    row, col = current
    if direction == 'E':
        return (row, col + 1)
    elif direction == 'W':
        return (row, col - 1)
    elif direction == 'S':
        return (row + 1, col)
    elif direction == 'N':
        return (row - 1, col)

# Main function to execute BFS, DFS, A*, and Dijkstra's search and compare
if __name__ == '__main__':
    # Create a 15x15 maze and load it from a CSV file
    m = maze(15, 15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/ICA Pyamaze Example/mazetest.csv')

    # Run BFS, DFS, A*, and Dijkstra's search on the maze
    exploration_order_bfs, visited_cells_bfs, path_to_goal_bfs = BFS_search(m)
    exploration_order_dfs, visited_cells_dfs, path_to_goal_dfs = dfs_search(m)
    exploration_order_astar, visited_cells_astar, path_to_goal_astar = A_star_search(m)
    exploration_order_dijkstra, visited_cells_dijkstra, path_to_goal_dijkstra = dijkstra_search(m)

    # Create agents to visualize the BFS, DFS, A*, and Dijkstra's search processes
    agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red)  # Visualize BFS search order
    agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.green)  # Visualize DFS search order
    agent_astar = agent(m, footprints=True, shape='square', color=COLOR.blue)  # Visualize A* search order
    agent_dijkstra = agent(m, footprints=True, shape='square', color=COLOR.cyan)  # Visualize Dijkstra search order

    # Full path agents
    agent_trace_bfs = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full BFS path
    agent_trace_dfs = agent(m, footprints=True, shape='star', color=COLOR.black, filled=False)  # Full DFS path
    agent_trace_astar = agent(m, footprints=True, shape='star', color=COLOR.red, filled=False)  # Full A* path
    agent_trace_dijkstra = agent(m, footprints=True, shape='star', color=COLOR.green, filled=False)  # Full Dijkstra path
    
    # Visualize the agents' movements along their respective paths
    m.tracePath({agent_bfs: exploration_order_bfs}, delay=5)  # BFS search order path
    m.tracePath({agent_dfs: exploration_order_dfs}, delay=5)  # DFS search order path
    m.tracePath({agent_astar: exploration_order_astar}, delay=5)  # A* search order path
    m.tracePath({agent_dijkstra: exploration_order_dijkstra}, delay=5)  # Dijkstra search order path

    # Visualize the full path for all algorithms from start to goal
    m.tracePath({agent_trace_bfs: path_to_goal_bfs}, delay=9)  # BFS path to the goal
    m.tracePath({agent_trace_dfs: path_to_goal_dfs}, delay=9)  # DFS path to the goal
    m.tracePath({agent_trace_astar: path_to_goal_astar}, delay=9)  # A* path to the goal
    m.tracePath({agent_trace_dijkstra: path_to_goal_dijkstra}, delay=9)  # Dijkstra path to the goal

    # Display the length of the paths and search steps for each algorithm
    textLabel(m, 'BFS Path Length', len(path_to_goal_bfs) + 1)  # BFS path length
    textLabel(m, 'DFS Path Length', len(path_to_goal_dfs) + 1)  # DFS path length
    textLabel(m, 'A* Path Length', len(path_to_goal_astar) + 1)  # A* path length
    textLabel(m, 'Dijkstra Path Length', len(path_to_goal_dijkstra) + 1)  # Dijkstra path length
    textLabel(m, 'BFS Search Length', len(exploration_order_bfs))  # BFS explored cells
    textLabel(m, 'DFS Search Length', len(exploration_order_dfs))  # DFS explored cells
    textLabel(m, 'A* Search Length', len(exploration_order_astar))  # A* explored cells
    textLabel(m, 'Dijkstra Search Length', len(exploration_order_dijkstra))  # Dijkstra explored cells

    # Run the maze visualization
    m.run()
