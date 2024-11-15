# Import necessary modules.
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

def BFS(m, start=None):
    # Define the start cell, if not provided use the bottom-right cell.
    if start is None:
        start = (m.rows, m.cols)

    # Initialize the frontier as a deque (double-ended queue), start with the start cell.
    frontier = deque()
    frontier.append(start)

    # Dictionary to store the path taken by BFS.
    bfsPath = {}

    # List to keep track of the cells that have already been explored.
    explored = [start]

    # List to store the order of cells explored.
    bfsSearch = []

    # Process while there are cells in the frontier.
    while len(frontier) > 0:
        # Dequeue a cell from the frontier.
        currCell = frontier.popleft()

        # If the current cell is the goal, then stop processing.
        if currCell == m._goal:
            break

        # Check each direction from the current cell.
        for d in 'ESNW':
            # If the current direction is passable (no wall)...
            if m.maze_map[currCell][d] == True:
                # Determine the coordinates of the child cell based on the direction.
                if d == 'E':
                    childCell = (currCell[0], currCell[1] + 1)
                elif d == 'W':
                    childCell = (currCell[0], currCell[1] - 1)
                elif d == 'S':
                    childCell = (currCell[0] + 1, currCell[1])
                elif d == 'N':
                    childCell = (currCell[0] - 1, currCell[1])

                # If the child cell has already been explored, skip this iteration.
                if childCell in explored:
                    continue

                # Add the child cell to the frontier and mark it as explored.
                frontier.append(childCell)
                explored.append(childCell)

                # Add the child cell and its parent to the BFS path. Also, update the search order list.
                bfsPath[childCell] = currCell
                bfsSearch.append(childCell)

    # Trace the path from the goal to the start.
    fwdPath = {}
    cell = m._goal
    while cell != (m.rows, m.cols):
        fwdPath[bfsPath[cell]] = cell
        cell = bfsPath[cell]

    # Return the search order, path, and forward path.
    return bfsSearch, bfsPath, fwdPath

# Main function.
if __name__ == '__main__':
    # Create a maze.
    m = maze(15, 15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/ICA Pyamaze Example/mazetest.csv')

    # Perform BFS on the maze and retrieve the search order and paths.
    bfsSearch, bfsPath, fwdPath = BFS(m)

    # Create agents.
    a = agent(m, footprints=True, shape='square', color=COLOR.green)
    b = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=False)
    c = agent(m, 1, 1, footprints=True, color=COLOR.cyan, shape='square', filled=True, goal=(m.rows, m.cols))

    # Make agents trace the paths.
    m.tracePath({a: bfsSearch}, delay=100)
    m.tracePath({c: bfsPath}, delay=100)
    m.tracePath({b: fwdPath}, delay=100)

    l=textLabel(m, 'BFS Path Length',len(fwdPath)+1)
    l=textLabel(m, 'BFS Search Length',len(bfsSearch))

    # Run the maze.
    m.run()

