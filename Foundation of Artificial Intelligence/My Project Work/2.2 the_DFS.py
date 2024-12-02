from pyamaze import maze, agent, COLOR, textLabel

def DFS_search(maze_obj, start=None, goal=None):
    # Default start position: Bottom-right corner
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    # Ensure goal is within the maze bounds
    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)  # Default to the middle of the maze
    # Check if the goal is valid (within bounds)
    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")

    # Initialize DFS stack with the start point
    stack = [start]

    # Dictionary to store the path taken to reach each cell
    visited = {}

    # List to track cells visited in the search process
    exploration_order = []

    # Set of explored cells to avoid revisiting
    explored = set([start])

    while stack:
        # Pop the next cell to process (LIFO order)
        current = stack.pop()

        # If the goal is reached, stop the search
        if current == goal:
            break

        # Check all four possible directions (East, West, South, North)
        for direction in 'ESNW':
            # If movement is possible in this direction (no wall)
            if maze_obj.maze_map[current][direction]:
                # Calculate the coordinates of the next cell in the direction
                next_cell = get_next_cell(current, direction)

                # If the cell hasn't been visited yet, process it
                if next_cell not in explored:
                    stack.append(next_cell)  # Add to the stack (LIFO)
                    explored.add(next_cell)   # Mark as visited
                    visited[next_cell] = current  # Record the parent (current cell)
                    exploration_order.append(next_cell)  # Track the exploration order

    # Reconstruct the path from the goal to the start using the visited dictionary
    path_to_goal = {}
    cell = goal
    while cell != start:
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
    # Create a 30x50 maze and load it from a CSV file
    m = maze(30, 50)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv')

    # Set your custom goal (within maze limits)
    goal_position = (26, 33)  # Example goal, you can change this to any valid coordinate

    # Perform DFS search on the maze and get the exploration order and paths
    exploration_order, visited_cells, path_to_goal = DFS_search(m, goal=goal_position)

    # Create agents to visualize the DFS search process
    agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.red)  # Visualize DFS search order
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full DFS path

    # Create the goal agent at the custom goal position
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

    # Visualize the agents' movements along their respective paths
    m.tracePath({agent_dfs: exploration_order}, delay=5)  # DFS search order path
    m.tracePath({agent_trace: path_to_goal}, delay=100)  # Trace the path from goal to start (final agent path)
    m.tracePath({agent_goal: visited_cells}, delay=100)  # Trace the DFS path to the goal

    # Add a text label to display the goal position on the maze
    textLabel(m, 'Goal Position', str(goal_position))

    # Display the length of the DFS path and search steps
    textLabel(m, 'DFS Path Length', len(path_to_goal) + 1)  # Length of the path from goal to start
    textLabel(m, 'DFS Search Length', len(exploration_order))  # Total number of explored cells

    # Run the maze visualization
    m.run()
