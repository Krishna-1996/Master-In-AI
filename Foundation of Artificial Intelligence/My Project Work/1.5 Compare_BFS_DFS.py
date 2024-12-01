# Import necessary modules for maze creation and visualization
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

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

# Main function to execute BFS and DFS search and compare
if __name__ == '__main__':
    # Create a 15x15 maze and load it from a CSV file
    m = maze(15, 15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv')

    # Run BFS and DFS search on the maze
    exploration_order_bfs, visited_cells_bfs, path_to_goal_bfs = BFS_search(m)
    exploration_order_dfs, visited_cells_dfs, path_to_goal_dfs = dfs_search(m)

    # Create agents to visualize the BFS and DFS search processes
    agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red)  # Visualize BFS search order
    agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.green)  # Visualize DFS search order
    agent_trace_bfs = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full BFS path
    agent_trace_dfs = agent(m, footprints=True, shape='star', color=COLOR.blue, filled=False)  # Full DFS path
    agent_goal = agent(m, 1, 1, footprints=True, color=COLOR.blue, shape='square', filled=True, goal=(m.rows, m.cols))  # Goal agent

    # Visualize the BFS and DFS agents' movements along their respective paths
    m.tracePath({agent_bfs: exploration_order_bfs}, delay=5)  # BFS search order path
    m.tracePath({agent_dfs: exploration_order_dfs}, delay=5)  # DFS search order path

    # Visualize the full path for both BFS and DFS from start to goal
    m.tracePath({agent_trace_bfs: path_to_goal_bfs}, delay=1)  # BFS path to the goal
    m.tracePath({agent_trace_dfs: path_to_goal_dfs}, delay=1)  # DFS path to the goal

    # Display the length of the BFS and DFS paths and search steps
    textLabel(m, 'BFS Path Length', len(path_to_goal_bfs) + 1)  # Length of the BFS path
    textLabel(m, 'DFS Path Length', len(path_to_goal_dfs) + 1)  # Length of the DFS path
    textLabel(m, 'BFS Search Length', len(exploration_order_bfs))  # Total number of explored BFS cells
    textLabel(m, 'DFS Search Length', len(exploration_order_dfs))  # Total number of explored DFS cells

    # Run the maze visualization
    m.run()
