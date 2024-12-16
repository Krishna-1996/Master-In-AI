from pyamaze import maze, agent, COLOR, textLabel
m = maze()
# print(dir(m))

maze.
# Import necessary modules (assuming the maze and agent are from a visualization library like "pyamaze" or similar)
from collections import deque

# DFS Algorithm to explore the maze and find the path to the goal
def DFS_search(maze, goal):
    start = maze.getStart()  # Starting point of the maze
    stack = [start]  # Stack for DFS, we start with the start cell
    visited_cells = []  # List to store the visited cells
    parent_map = {}  # Dictionary to map each cell to its parent for path reconstruction
    
    while stack:
        current_cell = stack.pop()  # Pop the top element (last visited cell)
        visited_cells.append(current_cell)  # Mark current cell as visited
        
        if current_cell == goal:  # If goal is found, stop searching
            break
        
        # Get the neighboring cells in the maze
        neighbors = maze.getNeighbors(current_cell)  
        
        for neighbor in neighbors:
            if neighbor not in visited_cells and neighbor not in stack:
                stack.append(neighbor)  # Add the unvisited neighbor to the stack
                parent_map[neighbor] = current_cell  # Set the parent for backtracking
    
    # Reconstruct the path to the goal
    path_to_goal = []
    current_cell = goal
    while current_cell != start:
        path_to_goal.append(current_cell)
        current_cell = parent_map[current_cell]
    path_to_goal.append(start)
    path_to_goal.reverse()  # Reverse the path to make it from start to goal
    
    return visited_cells, path_to_goal

# Main function to execute the maze creation and DFS search
if __name__ == '__main__':
    # Create a 50x120 maze and load it from a CSV file
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')

    # Set your custom goal (within maze limits)
    goal_position = (49, 2)  # Example goal, you can change this to any valid coordinate

    # Perform DFS search on the maze and get the exploration order and paths
    exploration_order, path_to_goal = DFS_search(m, goal=goal_position)

    # Create agents to visualize the DFS search process
    agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.red)  # Visualize DFS search order
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full DFS path

    # Create the goal agent at the custom goal position
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

    # Visualize the agents' movements along their respective paths
    m.tracePath({agent_dfs: exploration_order}, delay=1)  # DFS search order path
    m.tracePath({agent_trace: path_to_goal}, delay=1)  # Trace the path from goal to start (final agent path)
    m.tracePath({agent_goal: path_to_goal}, delay=1)  # Trace the DFS path to the goal

    # Display the length of the goal position, DFS path, and search steps
    textLabel(m, 'Goal Position', str(goal_position))
    textLabel(m, 'DFS Path Length', len(path_to_goal))  # Length of the path from start to goal
    textLabel(m, 'DFS Search Length', len(exploration_order))  # Total number of explored cells

    # Run the maze visualization
    m.run()

