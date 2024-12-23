# %%
# # Import necessary modules for maze generation, Greedy BFS algorithm, and maze visualization
from pyamaze import maze, agent, COLOR, textLabel
import heapq  # For the priority queue used in Greedy BFS algorithm

# %%
def heuristic(cell, goal):
    """
    Calculate the Manhattan distance from the current cell to the goal.
    This is the heuristic function used in Greedy BFS.
    """
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

def greedy_bfs(m, start=None):
    """
    Perform Greedy Best-First Search (Greedy BFS) Algorithm to find the shortest path in the maze.
    Greedy BFS algorithm uses only the heuristic to guide the search.
    """
    
    # Set the starting point of the Greedy BFS algorithm. Default is the bottom-right corner if not specified.
    if start is None:
        start = (m.rows, m.cols)  # Bottom-right corner
    
    # Initialize the distance dictionaries and priority queue
    f_costs = {cell: float('inf') for cell in m.grid}  # f(n): heuristic (no g(n) cost here)
    f_costs[start] = heuristic(start, m._goal)  # f(n) = h(n) for Greedy BFS
    
    # Priority queue (min-heap) for selecting the next node to explore based on f(n) = h(n)
    priority_queue = []
    heapq.heappush(priority_queue, (f_costs[start], start))  # Push the start cell with f(n)
    
    # Dictionary to store the path (previous cell) that leads to each cell
    came_from = {}
    
    # List to store the order in which cells are explored
    exploration_order = []
    
    # Process the priority queue until it's empty
    while priority_queue:
        _, current_cell = heapq.heappop(priority_queue)  # Get the cell with the lowest f(n)
        
        # If the current cell is the goal, we stop the algorithm
        if current_cell == m._goal:
            break
        
        # Explore the neighboring cells (up, down, left, right)
        for d in 'ESNW':  # Directions: East, South, North, West
            # Check if the current direction is passable (no wall)
            if m.maze_map[current_cell][d] == True:
                
                # Calculate the coordinates of the neighboring cell based on direction
                if d == 'E':  # East
                    neighbor_cell = (current_cell[0], current_cell[1] + 1)
                elif d == 'W':  # West
                    neighbor_cell = (current_cell[0], current_cell[1] - 1)
                elif d == 'S':  # South
                    neighbor_cell = (current_cell[0] + 1, current_cell[1])
                elif d == 'N':  # North
                    neighbor_cell = (current_cell[0] - 1, current_cell[1])
                
                # If this neighbor has not been explored yet, update its costs and add it to the queue
                if neighbor_cell not in came_from:
                    came_from[neighbor_cell] = current_cell
                    f_costs[neighbor_cell] = heuristic(neighbor_cell, m._goal)  # f(n) = h(n)
                    heapq.heappush(priority_queue, (f_costs[neighbor_cell], neighbor_cell))
                    exploration_order.append(neighbor_cell)  # Track the order of exploration

    # Reconstruct the path from the goal to the start by following the `came_from` dictionary
    path_to_goal = []
    cell = m._goal
    while cell != start:
        path_to_goal.append(cell)
        cell = came_from[cell]
    path_to_goal.append(start)
    path_to_goal.reverse()  # Reverse the path to get it from start to goal
    
    # Return the exploration order and the reconstructed path to the goal
    return exploration_order, came_from, path_to_goal
# %%
# Main function to create and run the maze
if __name__ == '__main__':
    # Create a 50, 120 maze and load a custom maze from a CSV file
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')
    goal_position = (1, 1)
    # Perform Greedy BFS algorithm on the maze to find the search order and paths
    exploration_order, came_from, path_to_goal = greedy_bfs(m)

    # Create agents to visualize the BFS search process
    agent_greedyBFS = agent(m, footprints=True, shape='square',color=COLOR.red)  # Visualize BFS search order
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full BFS path
    agent_goal = agent(m, 1, 1, footprints=True, color=COLOR.blue, shape='square', filled=True, goal=(m.rows, m.cols))  # Goal agent

    # Trace the agents' paths through the maze
    m.tracePath({agent_greedyBFS: exploration_order}, delay=1)  # Trace A* search order
    m.tracePath({agent_goal: came_from}, delay=1)  # Trace the path found by A*
    m.tracePath({agent_trace: path_to_goal}, delay=1)  # Trace the path from start to goal (final path)

    # Display the lengths of the Greedy BFS search and final paths as labels
    l = textLabel(m, 'Goal Position', str(goal_position))
    l = textLabel(m, 'Greedy BFS Path Length', len(path_to_goal))  # Length of the path from start to goal
    l = textLabel(m, 'Greedy BFS Search Length', len(exploration_order))  # Total number of cells explored

    # Run the maze simulation
    m.run()

# %%
