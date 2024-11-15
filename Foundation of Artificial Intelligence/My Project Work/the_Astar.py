# Import necessary modules for maze generation, A* algorithm, and maze visualization
from pyamaze import maze, agent, COLOR, textLabel
import heapq  # Priority Queue for A* (min-heap)

# Heuristic function (Manhattan distance)
def heuristic(a, b):
    """
    Calculate the Manhattan distance between two points (a and b).
    This is the heuristic function used in A* to estimate the cost to the goal.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(m, start=None):
    """
    Perform A* (A-star) algorithm to find the shortest path in the maze.
    If no start point is specified, defaults to bottom-right corner of the maze.
    """

    # Set the starting point of A*, default is bottom-right corner if not specified
    if start is None:
        start = (m.rows, m.cols)  # Bottom-right corner
    
    # Initialize priority queue (min-heap) with the start cell
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, m._goal), 0, start))  # (f, g, cell)
    
    # Initialize dictionaries for storing path and costs
    came_from = {}  # To reconstruct the path (parent -> current cell)
    g_score = {start: 0}  # Cost from start to each cell
    f_score = {start: heuristic(start, m._goal)}  # Estimated cost to goal (f = g + h)

    # Set to store explored cells (closed list)
    closed_list = set()
    
    # List to track the order of cells explored
    exploration_order = []

    while open_list:
        # Pop the cell with the lowest f-score from the open list
        current_f, current_g, current_cell = heapq.heappop(open_list)

        # If the goal is reached, stop the search
        if current_cell == m._goal:
            break
        
        # Add the current cell to the closed list (already explored)
        closed_list.add(current_cell)

        # Explore neighboring cells (East, South, North, West)
        for direction in 'ESNW':  # Directions: East, South, North, West
            # Check if the direction is open (i.e., no wall)
            if m.maze_map[current_cell][direction]:
                # Get the coordinates of the neighboring cell based on direction
                neighbor = get_next_cell(current_cell, direction)

                # Skip cells that have already been explored
                if neighbor in closed_list:
                    continue

                # Calculate the tentative g-score (cost to reach this neighbor)
                tentative_g_score = g_score[current_cell] + 1  # Assuming uniform cost for each step

                # If the neighbor is not in the g_score or the new g_score is better
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_cell  # Track the parent cell
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, m._goal)

                    # Add the neighbor to the open list (priority queue)
                    heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))

                    # Add the neighbor to the exploration order list
                    exploration_order.append(neighbor)

    # Reconstruct the path from the goal to the start by following parent pointers
    path_to_goal = []
    cell = m._goal
    while cell != start:  # Keep going until we reach the start
        path_to_goal.append(cell)
        cell = came_from[cell]
    path_to_goal.append(start)  # Add the start cell to the path
    path_to_goal.reverse()  # Reverse the list to get the path from start to goal

    return exploration_order, came_from, path_to_goal

def get_next_cell(current, direction):
    """
    Get the coordinates of the neighboring cell based on the direction.
    Directions are 'E' (East), 'W' (West), 'S' (South), 'N' (North).
    """
    row, col = current
    if direction == 'E':  # Move East
        return (row, col + 1)
    elif direction == 'W':  # Move West
        return (row, col - 1)
    elif direction == 'S':  # Move South
        return (row + 1, col)
    elif direction == 'N':  # Move North
        return (row - 1, col)

# Main function to create and run the maze
if __name__ == '__main__':
    # Create a 15x15 maze and load a custom maze from a CSV file
    m = maze(15, 15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/ICA Pyamaze Example/mazetest.csv')

    # Perform A* on the maze to find the search order and paths
    exploration_order, came_from, path_to_goal = astar(m)

    # Create agents to visualize the maze solving process
    agent_astar = agent(m, footprints=True, shape='square', color=COLOR.green)  # A* search order agent
    agent_path = agent(m, footprints=True, shape='square', color=COLOR.red)  # Path agent (A* solution path)

    # Trace the agents' paths through the maze
    m.tracePath({agent_astar: exploration_order}, delay=100)  # Trace A* search order
    m.tracePath({agent_path: path_to_goal}, delay=150)  # Trace the final A* path

    # Display the lengths of the A* search and path as labels
    textLabel(m, 'A* Path Length', len(path_to_goal))  # Length of the path from start to goal
    textLabel(m, 'A* Search Length', len(exploration_order))  # Total number of cells explored

    # Run the maze simulation
    m.run()
