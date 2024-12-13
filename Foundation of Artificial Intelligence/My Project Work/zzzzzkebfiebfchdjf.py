
# Importing required modules for maze creation and visualization
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque



def BFS_search(maze_obj, start=None):
    # If no start position is provided, use the bottom-right corner of the maze as the start
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    
    # Initialize BFS frontier with the start point
    frontier = deque([start])  # BFS queue initialized with the start position
    
    # Dictionary to store the path taken to reach each cell
    visited = {}  # This will map each cell to its parent (previous cell)
    
    # List to track cells visited in the search process
    exploration_order = []  # This will store the order in which cells are explored
    
    # Set of explored cells to avoid revisiting
    explored = set([start])  # Start with the start cell marked as explored
    


    while frontier:  # Loop while there are still cells in the frontier to explore
        current = frontier.popleft()  # Dequeue the next cell to process

        # If the goal is reached, stop the search
        if current == maze_obj._goal:
            break
        
        # Check all four possible directions (East, West, South, North)
        for direction in 'ESNW':  # Loop through possible directions (East, South, North, West)
            # If movement is possible in this direction (no wall)
            if maze_obj.maze_map[current][direction]:  # Check if there's no wall in the current direction
                # Calculate the coordinates of the next cell in the direction
                next_cell = get_next_cell(current, direction)  # Get the next cell's coordinates
                
                # If the cell hasn't been visited yet, process it
                if next_cell not in explored:
                    frontier.append(next_cell)  # Add to the frontier (queue)
                    explored.add(next_cell)  # Mark the next cell as explored
                    visited[next_cell] = current  # Record the parent (current cell) for path reconstruction
                    exploration_order.append(next_cell)  # Track the order of exploration



    # Reconstruct the path from the goal to the start using the visited dictionary
    path_to_goal = {}  # Dictionary to store the reconstructed path from goal to start
    cell = maze_obj._goal  # Start from the goal cell
    while cell != (maze_obj.rows, maze_obj.cols):  # Continue until reaching the start cell
        path_to_goal[visited[cell]] = cell  # Map each cell to its predecessor
        cell = visited[cell]  # Move to the previous cell in the path

    # Return exploration order, visited cells, and the path to the goal
    return exploration_order, visited, path_to_goal  


def get_next_cell(current, direction):
    """
    Returns the coordinates of the neighboring cell based on the direction.
    Directions are 'E' (East), 'W' (West), 'S' (South), 'N' (North).
    """
    row, col = current  # Unpack current cell coordinates
    if direction == 'E':  # Move East
        return (row, col + 1)  # Move one column to the right
    elif direction == 'W':  # Move West
        return (row, col - 1)  # Move one column to the left
    elif direction == 'S':  # Move South
        return (row + 1, col)  # Move one row down
    elif direction == 'N':  # Move North
        return (row - 1, col)  # Move one row up

# Main function to execute the maze creation and BFS search
if __name__ == '__main__':
    # Create a 30 x 50 maze and load it from a CSV file
    m = maze(30, 50)  # Create a maze with 30 rows and 50 columns
    m.CreateMaze(loadMaze='.../My Project Work/maze--2024-11-30--21-36-21.csv')  # Load the maze from a CSV file
    goal_position = ("1, 1")  # Define the goal position (1,1)

    # Perform BFS search on the maze and get the exploration order and paths
    exploration_order, visited_cells, path_to_goal = BFS_search(m)  # Call BFS to get the search results

    # Create agents to visualize the BFS search process
    agent_bfs = agent(m, footprints=True, shape='square', 
                      color=COLOR.red)  # Agent for BFS search order visualization
    agent_trace = agent(m, footprints=True, shape='star', 
                        color=COLOR.yellow, filled=False)  # Agent for the full BFS path
    agent_goal = agent(m, 1, 1, footprints=True, color=COLOR.blue, shape='square', 
                       filled=True, goal=(m.rows, m.cols))  # Agent to mark the goal position

    # Visualize the agents' movements along their respective paths
    m.tracePath({agent_bfs: exploration_order}, delay=1)  # Visualize the BFS search order with a 1-second delay
    m.tracePath({agent_goal: visited_cells}, delay=1)  # Trace the BFS path to the goal
    m.tracePath({agent_trace: path_to_goal}, delay=1)  # Trace the path from the goal to the start

    # Display the length of the BFS path and search steps
    textLabel(m, 'Goal Position', (goal_position))  # Display the goal position on the maze
    textLabel(m, 'BFS Path Length', len(path_to_goal) + 1)  # Display the length of the BFS path (goal to start)
    textLabel(m, 'BFS Search Length', len(exploration_order))  # Display the total number of explored cells

    # Run the maze visualization
    m.run()  # Start the maze visualization
