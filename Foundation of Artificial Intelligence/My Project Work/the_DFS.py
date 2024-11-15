# Import necessary modules for maze creation and visualization
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

def dfs_search(maze_obj, start=None):
    """
    Perform a Depth-First Search (DFS) on the maze to find a path from the start to the goal.
    If no start point is given, it defaults to the bottom-right corner of the maze.
    """
    # Default start position is the bottom-right corner if not provided
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    # Initialize DFS stack with the start point
    stack = [start]
    
    # Dictionary to store the path taken to reach each cell
    visited = {}
    
    # List to track the cells visited in the order of exploration
    exploration_order = []
    
    # Set of explored cells to avoid revisiting
    explored = set([start])
    
    while stack:
        # Pop the next cell from the stack (LIFO - Last In, First Out)
        current = stack.pop()

        # If we reached the goal, stop searching
        if current == maze_obj._goal:
            break
        
        # Explore all four directions (East, West, South, North)
        for direction in 'ESNW':
            # If movement is possible in this direction (no wall)
            if maze_obj.maze_map[current][direction]:
                # Calculate the coordinates of the next cell in that direction
                next_cell = get_next_cell(current, direction)

                # If the cell hasn't been visited yet, process it
                if next_cell not in explored:
                    stack.append(next_cell)  # Add to the stack
                    explored.add(next_cell)  # Mark as visited
                    visited[next_cell] = current  # Record the parent (current cell)
                    exploration_order.append(next_cell)  # Track exploration order

    # Reconstruct the path from the goal to the start using the visited dictionary
    path_to_goal = {}
    cell = maze_obj._goal
    while cell != (maze_obj.rows, maze_obj.cols):
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

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
    # Create a 15x15 maze and load it from a CSV file
    m = maze(15, 15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/ICA Pyamaze Example/mazetest.csv')

    # Perform DFS search on the maze and get the exploration order and paths
    exploration_order, visited_cells, path_to_goal = dfs_search(m)

    # Create agents to visualize the DFS search process
    agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.green)  # Visualize DFS search order
    agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=False)  # Full DFS path
    agent_goal = agent(m, 1, 1, footprints=True, color=COLOR.cyan, shape='square', filled=True, goal=(m.rows, m.cols))  # Goal agent

    # Visualize the agents' movements along their respective paths
    m.tracePath({agent_dfs: exploration_order}, delay=100)  # DFS search order path
    m.tracePath({agent_goal: visited_cells}, delay=100)  # Trace the DFS path to the goal
    m.tracePath({agent_trace: path_to_goal}, delay=100)  # Trace the path from goal to start

    # Display the length of the DFS path and search steps
    textLabel(m, 'DFS Path Length', len(path_to_goal) + 1)  # Length of the path from goal to start
    textLabel(m, 'DFS Search Length', len(exploration_order))  # Total number of explored cells

    # Run the maze visualization
    m.run()
