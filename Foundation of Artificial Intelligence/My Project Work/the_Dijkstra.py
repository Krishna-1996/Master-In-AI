# Import necessary modules for maze generation, Dijkstra's algorithm, and maze visualization
from pyamaze import maze, agent, COLOR, textLabel
import heapq  # For the priority queue used in Dijkstra's algorithm

def dijkstra(m, start=None):
    """
    Perform Dijkstra's Algorithm to find the shortest path in the maze.
    Dijkstra's algorithm finds the shortest path from the start point to all reachable nodes in the maze.
    """

    # Set the starting point of the Dijkstra algorithm. Default is the bottom-right corner if not specified
    if start is None:
        start = (m.rows, m.cols)  # Bottom-right corner
    
    # Initialize the distance dictionary with infinite distance for all cells except the start cell
    distances = {cell: float('inf') for cell in m.grid}
    distances[start] = 0  # Distance to the start is 0

    # Initialize the priority queue (min-heap) for selecting the next node to explore
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))  # Push the start cell with distance 0
    
    # Dictionary to store the path (previous cell) that leads to each cell
    came_from = {}

    # List to store the order in which cells are explored
    exploration_order = []

    # Process the priority queue until it's empty
    while priority_queue:
        current_distance, current_cell = heapq.heappop(priority_queue)  # Get the cell with the lowest distance
        
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

                # Calculate the new distance to this neighbor
                new_distance = current_distance + 1  # Every move has equal cost (1)

                # If the new distance is shorter, update the neighbor's distance and add it to the priority queue
                if new_distance < distances[neighbor_cell]:
                    distances[neighbor_cell] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor_cell))
                    came_from[neighbor_cell] = current_cell
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
    # Create a 15x15 maze and load a custom maze from a CSV file
    m = maze(15, 15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/ICA Pyamaze Example/mazetest.csv')

    # Perform Dijkstra's algorithm on the maze to find the search order and paths
    exploration_order, came_from, path_to_goal = dijkstra(m)

    # Create agents to visualize the maze solving process
    a = agent(m, footprints=True, shape='square', color=COLOR.green)  # Agent for Dijkstra search order
    b = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=True)  # Path tracing agent
    c = agent(m, 1, 1, footprints=True, color=COLOR.cyan, shape='square', filled=True, goal=(m.rows, m.cols))  # Goal-seeking agent

    # Trace the agents' paths through the maze
    m.tracePath({a: exploration_order}, delay=100)  # Trace Dijkstra's search order
    m.tracePath({b: path_to_goal}, delay=100)  # Trace the path found by Dijkstra
    m.tracePath({c: path_to_goal}, delay=100)  # Trace the path from start to goal (final path)

    # Display the lengths of the Dijkstra search and final paths as labels
    l = textLabel(m, 'Dijkstra Path Length', len(path_to_goal))  # Length of the path from start to goal
    l = textLabel(m, 'Dijkstra Search Length', len(exploration_order))  # Total number of cells explored

    # Run the maze simulation
    m.run()
