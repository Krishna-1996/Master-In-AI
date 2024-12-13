from pyamaze import maze, agent, COLOR, textLabel

def DFS_search(maze_obj, start, goal):
    """
    Perform DFS search on the maze to find the shortest path from start to goal.
    Arguments:
    - maze_obj: The maze object
    - start: Tuple (row, col) representing the start position
    - goal: Tuple (row, col) representing the goal position
    
    Returns:
    - exploration_order: List of cells visited during the search
    - path_to_goal: List of cells from start to goal
    """
    
    # Stack to hold the cells for DFS, initialized with the start position
    stack = [start]
    
    # Dictionary to store the parent of each visited cell (for path reconstruction)
    visited = {}
    
    # List to track the order in which cells are explored
    exploration_order = []
    
    # Set to track visited cells
    explored = set([start])

    while stack:
        # Pop the next cell from the stack (LIFO order)
        current = stack.pop()

        # If the goal is found, stop the search
        if current == goal:
            exploration_order.append(current)
            break

        # Check all four possible directions (East, West, South, North)
        for direction in 'ESNW':
            # If movement is possible (no wall in the direction)
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)

                # If the next cell hasn't been explored yet, process it
                if next_cell not in explored:
                    stack.append(next_cell)  # Add the next cell to the stack
                    explored.add(next_cell)  # Mark the next cell as explored
                    visited[next_cell] = current  # Record the parent of the next cell
                    exploration_order.append(next_cell)  # Track the exploration order

    # Reconstruct the path from the goal to the start using the visited dictionary
    path_to_goal = []
    if goal in visited:
        current = goal
        while current != start:
            path_to_goal.append(current)  # Add current to the path
            current = visited[current]  # Move to the parent
        path_to_goal.append(start)  # Add the start to the path
        path_to_goal.reverse()  # Reverse the path to get from start to goal
    
    return exploration_order, path_to_goal

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
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')

    # Set your custom goal (within maze limits)
    goal_position = (1, 10)  # Custom goal position (change this to any valid coordinate)

    # Set your custom start (bottom-right corner)
    start_position = (30, 50)  # Starting position is always bottom-right corner

    # Perform DFS search on the maze and get the exploration order and paths
    exploration_order, path_to_goal = DFS_search(m, start=start_position, goal=goal_position)

    # Create agents to visualize the DFS search process
    agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.red)  # Visualize DFS search order
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full DFS path

    # Create the goal agent at the custom goal position
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

    # Visualize the agents' movements along their respective paths
    m.tracePath({agent_dfs: exploration_order}, delay=4)  # DFS search order path
    m.tracePath({agent_trace: path_to_goal}, delay=100)  # Trace the path from goal to start (final agent path)
    m.tracePath({agent_goal: [goal_position]}, delay=100)  # Trace the DFS path to the goal

    # Add a text label to display the goal position on the maze
    textLabel(m, 'Goal Position', str(goal_position))

    # Display the length of the DFS path and search steps
    textLabel(m, 'DFS Path Length', len(path_to_goal))  # Length of the path from goal to start
    textLabel(m, 'DFS Search Length', len(exploration_order))  # Total number of explored cells

    # Run the maze visualization
    m.run()
