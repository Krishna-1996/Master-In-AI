def dfs_search(maze_obj, start=None, goal=None):
    """Depth-First Search algorithm."""
    start = (50, 100) # Default start position
    goal = (1,1 ) # Default goal position

def greedy_bfs_search(maze_obj, start=None, goal=None):
    start = (50, 100) # Default start position
    goal = (1,1 ) # Default goal position
    
    # Min-heap priority queue based on heuristic (Greedy BFS uses heuristic only)
    frontier = []  # The frontier will store cells to explore
    heapq.heappush(frontier, (heuristic(start, goal), start))  # Push start position to the queue (heuristic, start)
    visited = {}  # Dictionary to store visited cells and their predecessors
    exploration_order = []  # Order of exploration (for visualization)
    explored = set([start])  # Set to track the explored cells
    
    while frontier:
        _, current = heapq.heappop(frontier)  # Pop the cell with the lowest heuristic cost
        if current == goal:
            break  # Stop if the goal is reached
        for direction in 'ESNW':  # Explore all four directions (East, West, North, South)
            if maze_obj.maze_map[current][direction] == 1:  # If the direction is open (no wall)
                next_cell = get_next_cell(current, direction)  # Get the next cell based on the direction
                if next_cell not in explored:
                    heapq.heappush(frontier, (heuristic(next_cell, goal), next_cell))  # Add to frontier based on heuristic
                    visited[next_cell] = current  # Mark the predecessor of the next cell
                    exploration_order.append(next_cell)  # Add to exploration order
                    explored.add(next_cell)  # Mark as explored
    # If the goal is not reached, return empty results
    if goal not in visited:
        print("Goal is unreachable!")
        return [], {}, {}
    # Reconstruct the path from goal to start
    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal



# Heuristic function to calculate Manhattan distance between two points
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])








