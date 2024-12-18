# Importing required modules for maze creation and visualization
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

def DFS_search(maze_obj, start=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    
    # DFS stack, with the start position
    stack = [start]
    
    # To store the visited cells and path taken
    visited = set()
    parent = {}
    
    # Exploration order
    exploration_order = []
    
    # The DFS algorithm starts
    while stack:
        current = stack.pop()
        
        if current == maze_obj._goal:
            break
        
        # If not visited, mark it as visited
        if current not in visited:
            visited.add(current)
            exploration_order.append(current)
            
            # Explore all four directions: E, S, W, N
            for direction in 'ESNW':
                if maze_obj.maze_map[current][direction]:
                    next_cell = get_next_cell(current, direction)
                    if next_cell not in visited:
                        stack.append(next_cell)
                        parent[next_cell] = current

    # Path reconstruction from goal to start
    path_to_goal = {}
    cell = maze_obj._goal
    while cell != start:
        path_to_goal[parent[cell]] = cell
        cell = parent[cell]

    return exploration_order, parent, path_to_goal

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

# Main function to execute the maze creation and DFS search
if __name__ == '__main__':
    # Create a 50, 120 maze and load it from a CSV file
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence//My Project Work/maze_update2.csv')
    
    # Set the goal position to (49, 2) or any other position
    goal_position = (49, 2)
    m._goal = goal_position  # Update the goal position in the maze object

    # Set the start position to (50, 120) or any other valid position
    start_position = (50, 120)

    # Perform DFS search on the maze and get the exploration order and paths
    exploration_order, visited_cells, path_to_goal = DFS_search(m, start=start_position)

    # Create agents to visualize the DFS search process
    agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.red)  # Create the agent for DFS
    agent_dfs.position = start_position  # Set the agent's start position

    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full DFS path
    agent_trace.position = start_position  # Set the trace agent's start position

    agent_goal = agent(m, goal=goal_position, footprints=True, color=COLOR.blue, shape='square', filled=True)  # Goal agent
    agent_goal.position = goal_position  # Set the goal agent's position

    # Visualize the agents' movements along their respective paths
    m.tracePath({agent_dfs: exploration_order}, delay=1)  # DFS search order path
    m.tracePath({agent_goal: visited_cells}, delay=1)  # Trace the DFS path to the goal
    m.tracePath({agent_trace: path_to_goal}, delay=1)  # Trace the path from goal to start

    # Display the length of the DFS path and search steps
    textLabel(m, 'Goal Position', (goal_position))
    textLabel(m, 'DFS Path Length', len(path_to_goal) + 1)  # Length of the path from goal to start
    textLabel(m, 'DFS Search Length', len(exploration_order))  # Total number of explored cells

    # Run the maze visualization
    m.run()
