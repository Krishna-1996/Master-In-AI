from pyamaze import maze, agent, COLOR, textLabel
from queue import PriorityQueue

# Define the heuristic function.
def h(cell1, cell2):
    # Calculate Manhattan distance between two cells.
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x1-x2)+abs(y1-y2)

# A* algorithm function.
def aStar(m, start=None):
    # If no start cell is provided, choose the bottom right cell as the start cell.
    if start is None:    
        start=(m.rows,m.cols)
    # PriorityQueue of open cells.
    open=PriorityQueue()
    open.put((h(start, m._goal), h(start, m._goal), start))

    # Dictionary to store the path taken by A*.
    aPath={}

    # Initialize g_score (cost from start to cell) and f_score (g_score + heuristic) for each cell.
    g_score = {row: float("inf") for row in m.grid}
    g_score[start] = 0
    f_score = {row: float("inf") for row in m.grid}
    f_score[start]=h(start, m._goal)
    # List to store the order of cells explored.
    searchPath=[start]
    # Process while there are cells in the open PriorityQueue.
    while not open.empty():
        # Dequeue a cell from the PriorityQueue.
        currCell=open.get()[2]
        # Add the current cell to the searchPath.
        searchPath.append(currCell)
        # If the current cell is the goal, then stop processing.
        if currCell==m._goal:
            break
        # Check each direction from the current cell.
        for d in 'ESNW':
            # If the current direction is passable (no wall)...
            if m.maze_map[currCell][d]==True:
                # Calculate the coordinates of the child cell based on the direction.
                if d=='E':
                    childCell=(currCell[0],currCell[1]+1)
                if d=='W':
                    childCell=(currCell[0],currCell[1]-1)
                if d=='S':
                    childCell=(currCell[0]+1,currCell[1])
                if d=='N':
                    childCell=(currCell[0]-1,currCell[1])
                
                # Calculate tentative g_score and f_score for the child cell.
                temp_g_score = g_score[currCell] + 1
                temp_f_score = temp_g_score + h(childCell, m._goal)

                # If the new f_score is lower, update the path and scores for the child cell.
                if temp_f_score < f_score[childCell]:
                    aPath[childCell]=currCell
                    g_score[childCell] = temp_g_score
                    f_score[childCell] = temp_f_score
                    # Add the child cell to the PriorityQueue with the new f_score.
                    open.put((f_score[childCell], h(childCell,m._goal), childCell))
                    
    # Trace the path from the goal to the start.
    fwdPath={}
    cell=m._goal
    while cell!=start:
        fwdPath[aPath[cell]]=cell
        cell=aPath[cell]
    # Return the search order, path, and forward path.
    return searchPath,aPath,fwdPath

if __name__=='__main__':
    # Create a maze.
    m=maze(15,15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/ICA Pyamaze Example/mazetest.csv')

    # Run the A* algorithm on the maze and get the paths.
    searchPath, aPath, fwdPath = aStar(m, start=None)

    # Create agents.
    a=agent(m,footprints=True, color=COLOR.blue, filled=True)
    b=agent(m,1,1,footprints=True, color=COLOR.yellow, filled=True,goal=(m.rows,m.cols))
    c=agent(m,footprints=True, color=COLOR.red)

    # Have the agents trace the paths.
    m.tracePath({a:searchPath},delay=100)
    m.tracePath({b:aPath},delay=100)
    m.tracePath({c:fwdPath},delay=100)
    
    l=textLabel(m, 'A Star Path Length',len(fwdPath)+1)
    l=textLabel(m, 'A Star Search Length',len(searchPath))

    # Print the maze's map and grid for debugging.
    print(m.maze_map)
    print(m.grid)

    # Run the maze.
    m.run()
