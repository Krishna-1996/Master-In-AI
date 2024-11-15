# Import necessary modules for maze generation, BFS algorithm, and maze visualization
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

def BFS(m, start=None):
    """
    Perform a Breadth-First Search (BFS) on the given maze to find the shortest path.
    If a start point is not provided, BFS will start from the bottom-right corner.
    """
    
    # Set the starting point of the BFS, default is bottom-right corner if not specified
    if start is None:
        start = (m.rows, m.cols)  # Bottom-right corner
    
    # Initialize the frontier with the starting cell (deque acts like a queue)
    frontier = deque()
    frontier.append(start)
    
    # Dictionary to store the path taken to reach each cell
    bfsPath = {}

    # List to keep track of cells that have been explored
    explored = [start]
    
    # List to keep track of the order of cells explored during BFS
    bfsSearch = []

    # Continue processing until all reachable cells are explored or the goal is found
    while len(frontier) > 0:
        # Dequeue the current cell from the frontier (first cell in the queue)
        currCell = frontier.popleft()

        # If we've reached the goal, stop searching
        if currCell == m._goal:
            break

        # Explore the neighboring cells (up, down, left, right)
        for d in 'ESNW':  # Directions: East, South, North, West
            # Check if the current direction is passable (no wall)
            if m.maze_map[currCell][d] == True:
                
                # Calculate the coordinates of the neighboring cell based on direction
                if d == 'E':  # East
                    childCell = (currCell[0], currCell[1] + 1)
                elif d == 'W':  # West
                    childCell = (currCell[0], currCell[1] - 1)
                elif d == 'S':  # South
                    childCell = (currCell[0] + 1, currCell[1])
                elif d == 'N':  # North
                    childCell = (currCell[0] - 1, currCell[1])

                # If the child cell has already been explored, skip it
                if childCell in explored:
                    continue

                # Add the new cell to the frontier and mark it as explored
                frontier.append(childCell)
                explored.append(childCell)

                # Record the path taken to reach this cell (parent -> child)
                bfsPath[childCell] = currCell
                bfsSearch.append(childCell)  # Add to the order of exploration

    # Reconstruct the path from the goal to the start by following the parent pointers
    fwdPath = {}
    cell = m._goal  # Start from the goal
    while cell != (m.rows, m.cols):  # Keep going until we reach the start
        fwdPath[bfsPath[cell]] = cell
        cell = bfsPath[cell]

    # Return the order of exploration, the path taken, and the forward path to the goal
    return bfsSearch, bfsPath, fwdPath

# Main function to create and run the maze
if __name__ == '__main__':
    # Create a 15x15 maze and load a custom maze from a CSV file
    m = maze(15, 15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/ICA Pyamaze Example/mazetest.csv')

    # Perform BFS on the maze to find the search order and paths
    bfsSearch, bfsPath, fwdPath = BFS(m)

    # Create agents to visualize the maze solving process
    a = agent(m, footprints=True, shape='square', color=COLOR.green)  # Agent for BFS search order
    b = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=False)  # Path tracing agent
    c = agent(m, 1, 1, footprints=True, color=COLOR.cyan, shape='square', filled=True, goal=(m.rows, m.cols))  # Goal-seeking agent

    # Trace the agents' paths through the maze
    m.tracePath({a: bfsSearch}, delay=100)  # Trace BFS search order
    m.tracePath({c: bfsPath}, delay=100)  # Trace the path taken by BFS
    m.tracePath({b: fwdPath}, delay=100)  # Trace the forward path from goal to start

    # Display the lengths of the BFS search and forward paths as labels
    l = textLabel(m, 'BFS Path Length', len(fwdPath) + 1)  # Length of the path from goal to start
    l = textLabel(m, 'BFS Search Length', len(bfsSearch))  # Total number of cells explored

    # Now finally run the maze.
    m.run()
