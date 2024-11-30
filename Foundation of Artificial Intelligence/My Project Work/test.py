# Importing required modules for maze creation and visualization
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

def BFS_search(maze_obj, start=None):

    # Start position == Bottom-right corner.
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    
    # Initialize BFS frontier with the start point
    frontier = deque([start])
    
    # Dictionary to store the path taken to reach each cell
    visited = {}
    
    # List to track cells visited in the search process
    exploration_order = []
    
    # Set of explored cells to avoid revisiting
    explored = set([start])
    
    while frontier:
        # Dequeue the next cell to process
        current = frontier.popleft()

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
                    frontier.append(next_cell)  # Add to the frontier
                    explored.add(next_cell)     # Mark as visited
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

# Main function to execute the maze creation and BFS search
if __name__ == '__main__':
    # Create a 15x15 maze and load it from a CSV file
    m = maze(15, 15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv')
    m._goal(15,15)

    # Perform BFS search on the maze and get the exploration order and paths
    exploration_order, visited_cells, path_to_goal = BFS_search(m)

    # Create agents to visualize the BFS search process
    agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red)  # Visualize BFS search order
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full BFS path
    agent_goal = agent(m, 1, 1, footprints=True, color=COLOR.blue, shape='square', filled=True, goal=(m.rows, m.cols))  # Goal agent

    # Visualize the agents' movements along their respective paths
    m.tracePath({agent_bfs: exploration_order}, delay=5)  # BFS search order path
    m.tracePath({agent_goal: visited_cells}, delay=100)  # Trace the BFS path to the goal
    m.tracePath({agent_trace: path_to_goal}, delay=100)  # Trace the path from goal to start

    # Display the length of the BFS path and search steps
    textLabel(m, 'BFS Path Length', len(path_to_goal) + 1)  # Length of the path from goal to start
    textLabel(m, 'BFS Search Length', len(exploration_order))  # Total number of explored cells

    # Run the maze visualization
    m.run()
