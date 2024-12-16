# Importing required modules for maze creation and visualization
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

def DFS_search(maze_obj, start=None):
    """
    Perform Depth-First Search (DFS) to solve the maze.
    """
    # Start position == Bottom-right corner.
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    # Initialize DFS stack with the start point
    stack = [start]

    # Dictionary to store the path taken to reach each cell
    visited = {}

    # List to track cells visited in the search process
    exploration_order = []

    # Set of explored cells to avoid revisiting
    explored = set([start])

    while stack:
        # Pop the next cell to process (DFS uses stack)
        current = stack.pop()

        # If the goal is reached, stop the search
        if current == maze_obj._goal:
            break

        # Check all four possible directions (East, West, South, North)
        for direction in 'ESNW':
            # If movement is possible in this direction (no wall)
            if maze_obj.maze_map[current][direction]:
                # Calculate the coordinates of the next cell in the direction
                next_cell = get_next_cell(current, direction)

                # If the cell hasn't been visited yet, process it
                if next_cell not in explored:
                    stack.append(next_cell)  # Add to the stack (DFS)
                    explored.add(next_cell)  # Mark as visited
                    visited[next_cell] = current  # Record the parent (current cell)
                    exploration_order.append(next_cell)  # Track the exploration order

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
    # Create a 50, 120 maze and load it from a CSV file
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence//My Project Work/maze_update2.csv')
    goal_position = ("49, 2")

    # Perform DFS search on the maze and get the exploration order and paths
    exploration_order, visited_cells, path_to_goal = DFS_search(m)

    # Create agents to visualize the DFS search process
    agent_dfs = agent(m, footprints=True, shape='square', 
                      color=COLOR.green)  # Visualize DFS search order
    agent_trace = agent(m, footprints=True, shape='star', 
                        color=COLOR.yellow, filled=False)  # Full DFS path
    agent_goal = agent(m, 49, 2, footprints=True, color=COLOR.blue, 
                       shape='square', filled=True, goal=(m.rows, m.cols))  # Goal agent

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
