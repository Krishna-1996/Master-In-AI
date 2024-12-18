from pyamaze import maze, agent, COLOR, textLabel

def DFS(m):
    # The starting point of the maze, in this case, the bottom right corner.
    start=(m.rows,m.cols)
    

    # List to keep track of the cells we have already explored.
    explored=[start]

    # Stack for implementing the DFS algorithm (Last In, First Out principle).
    frontier=[start]

    # Dictionary to keep track of the path from each node to its parent.
    dfsPath={}

    # List to keep track of the order in which nodes are visited.
    dSearch=[]

    # DFS loop that continues until all reachable nodes have been explored.
    while len(frontier)>0:
        # Pop the last node from the stack (depth-first principle).
        current=frontier.pop()
        dSearch.append(current)

        # If we reached the top left corner (goal), we break the loop.
        if current==m._goal:
            break

        # Check all 4 possible directions from the current cell.
        for d in 'ESNW':
            # If there's no wall in the current direction...
            if m.maze_map[current][d]==True:
                # ...calculate the coordinates of the child cell.
                if d=='E':
                    childCell=(current[0],current[1]+1)
                elif d=='W':
                    childCell=(current[0],current[1]-1)
                elif d=='S':
                    childCell=(current[0]+1,current[1])
                elif d=='N':
                    childCell=(current[0]-1,current[1])

                # If the child cell has already been explored, we ignore it.
                if childCell in explored:
                    continue

                # Otherwise, mark it as explored and add it to the frontier and path.
                explored.append(childCell)
                frontier.append(childCell)
                dfsPath[childCell]=current

    # Construct the shortest path from the start to the goal.
    fwdPath={}
    cell=m._goal
    while cell!=start:
        fwdPath[dfsPath[cell]]=cell
        cell=dfsPath[cell]

    # Return the search, path, and forward path.
    return dSearch,dfsPath,fwdPath

# Main script begins here.
if __name__ == '__main__':
    # Initialize a maze of size 15x15.
    m=maze(25,25)
    # Create the maze using the information from the csv file.
    # m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/ICA Pyamaze Example/mazetest.csv')
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence//My Project Work/maze_update2.csv')

    goal_position = (49, 2)
    # Run the DFS algorithm on the maze and retrieve the paths.
    dSearch,dfsPath,fwdPath=DFS(m)

    # Initialize the agents with their properties.
    a=agent(m, footprints=True, shape='square', color=COLOR.red)
    b=agent(m, goal=goal_position, footprints=True, color=COLOR.blue, shape='square', filled=True)
    c=agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)

    # Have the agents trace the paths.
    m.tracePath({a:dSearch},showMarked=True,delay=1)
    m.tracePath({b:dfsPath},delay=1)
    m.tracePath({c:fwdPath},delay=1)

    # Display some statistics about the path lengths.
    l=textLabel(m, 'DFS Path Length',len(fwdPath)+1)
    l=textLabel(m, 'DFS Search Length',len(dSearch))
    '''
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
    textLabel(m, 'DFS Search Length', len(exploration_order))  # Total number of explored cells'''

    # Run the visualization.
    m.run()
