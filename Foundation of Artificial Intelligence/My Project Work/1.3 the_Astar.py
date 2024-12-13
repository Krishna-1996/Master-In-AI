# Import necessary modules for maze generation, A* algorithm, and maze visualization
from pyamaze import maze, agent, COLOR, textLabel
import heapq  # For the priority queue used in A* algorithm

def heuristic(cell, goal):
    """
    Calculate the Manhattan distance from the current cell to the goal.
    This is the heuristic function used in A*.
    """
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

def a_star(m, start=None):
    """
    Perform A* Algorithm to find the shortest path in the maze.
    A* algorithm uses both the distance from the start and a heuristic to guide the search.
    """

    # Set the starting point of the A* algorithm. Default is the bottom-right corner if not specified.
    if start is None:
        start = (m.rows, m.cols)  # Bottom-right corner
    
    # Initialize the distance dictionaries and priority queue
    g_costs = {cell: float('inf') for cell in m.grid}  # g(n): cost from start to current node
    g_costs[start] = 0  # Starting point has a g(n) cost of 0
    f_costs = {cell: float('inf') for cell in m.grid}  # f(n): g(n) + h(n)
    f_costs[start] = heuristic(start, m._goal)  # f(n) = g(n) + h(n)
    
    # Priority queue (min-heap) for selecting the next node to explore based on f(n)
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
                
                # Calculate the tentative g(n) cost to reach this neighbor
                tentative_g_cost = g_costs[current_cell] + 1  # All moves have equal cost (1)
                
                # If the new g(n) cost is lower, update the costs and add it to the priority queue
                if tentative_g_cost < g_costs[neighbor_cell]:
                    came_from[neighbor_cell] = current_cell
                    g_costs[neighbor_cell] = tentative_g_cost
                    f_costs[neighbor_cell] = tentative_g_cost + heuristic(neighbor_cell, m._goal)
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

# Main function to create and run the maze
if __name__ == '__main__':
    # Create a 30 x 50 maze and load a custom maze from a CSV file
    m = maze(30, 50)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')
    goal_position = (1,1)
    # Perform A* algorithm on the maze to find the search order and paths
    exploration_order, came_from, path_to_goal = a_star(m)

    # Create agents to visualize the maze solving process
    a = agent(m, footprints=True, shape='square', color=COLOR.green)  # Agent for A* search order
    b = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Path tracing agent
    c = agent(m, 1, 1, footprints=True, color=COLOR.blue, shape='square', filled=True, goal=(m.rows, m.cols))  # Goal-seeking agent

    # Trace the agents' paths through the maze
    m.tracePath({a: exploration_order}, delay=10)  # Trace A* search order
    m.tracePath({b: path_to_goal}, delay=100)  # Trace the path found by A*
    m.tracePath({c: path_to_goal}, delay=10)  # Trace the path from start to goal (final path)

    # Display the lengths of the A* search and final paths as labels
    l = textLabel(m, 'Goal Position', str(goal_position))
    l = textLabel(m, 'A* Path Length', len(path_to_goal))  # Length of the path from start to goal
    l = textLabel(m, 'A* Search Length', len(exploration_order))  # Total number of cells explored

    # Run the maze simulation
    m.run()
