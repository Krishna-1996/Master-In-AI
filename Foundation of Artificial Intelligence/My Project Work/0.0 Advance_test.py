from pyamaze import maze, agent, COLOR, textLabel

def DFS(m):
    # The starting point of the maze, in this case, the bottom right corner.
    start = (m.rows - 1, m.cols - 1)  # Starting at the bottom-right corner

    # List to keep track of the cells we have already explored.
    explored = [start]

    # Stack for implementing the DFS algorithm (Last In, First Out principle).
    frontier = [start]

    # Dictionary to keep track of the path from each node to its parent.
    dfsPath = {}

    # List to keep track of the order in which nodes are visited.
    dSearch = []

    # DFS loop that continues until all reachable nodes have been explored.
    while len(frontier) > 0:
        # Pop the last node from the stack (depth-first principle).
        current = frontier.pop()
        dSearch.append(current)

        # If we reached the goal, we break the loop.
        if current == m._goal:
            break

        # Check all 4 possible directions from the current cell.
        for d in 'ESNW':
            # If there's no wall in the current direction...
            if m.maze_map[current][d] == True:
                # ...calculate the coordinates of the child cell.
                if d == 'E':
                    childCell = (current[0], current[1] + 1)
                elif d == 'W':
                    childCell = (current[0], current[1] - 1)
                elif d == 'S':
                    childCell = (current[0] + 1, current[1])
                elif d == 'N':
                    childCell = (current[0] - 1, current[1])

                # If the child cell has already been explored, we ignore it.
                if childCell in explored:
                    continue

                # Otherwise, mark it as explored and add it to the frontier and path.
                explored.append(childCell)
                frontier.append(childCell)
                dfsPath[childCell] = current

    # Construct the shortest path from the start to the goal.
    fwdPath = {}
    cell = m._goal
    while cell != start:
        fwdPath[dfsPath[cell]] = cell
        cell = dfsPath[cell]

    # Return the search, path, and forward path.
    return dSearch, dfsPath, fwdPath

# Main script begins here.
if __name__ == '__main__':
    # Initialize a maze of size 50x120.
    m = maze(50, 120)

    # Create the maze using the information from the csv file.
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence//My Project Work/maze_update2.csv')

    # Ensure the goal is within the bounds of the maze (row, column).
    goal_x, goal_y = 2, 119  # New goal position (row 2, column 119)

    if goal_x < 0 or goal_x >= m.rows or goal_y < 0 or goal_y >= m.cols:
        raise ValueError(f"Goal position {goal_x, goal_y} is out of bounds!")

    m._goal = (goal_x, goal_y)  # Set the goal position

    # Run the DFS algorithm on the maze and retrieve the paths.
    dSearch, dfsPath, fwdPath = DFS(m)

    # Initialize the agents with their properties.
    a = agent(m, footprints=True, filled=True, shape='arrow', color=COLOR.red)

    # Set the new goal position for agent `b`
    b = agent(m, 1, 1, goal=(goal_x, goal_y), footprints=True, filled=True, color=COLOR.blue)

    c = agent(m, footprints=True, color=COLOR.yellow)

    # Have the agents trace the paths.
    m.tracePath({a: dSearch}, showMarked=True, delay=1)
    m.tracePath({b: dfsPath}, delay=1)
    m.tracePath({c: fwdPath}, delay=1)

    # Display some statistics about the path lengths.
    l = textLabel(m, 'DFS Path Length', len(fwdPath) + 1)
    l = textLabel(m, 'DFS Search Length', len(dSearch))

    # Run the visualization.
    m.run()
