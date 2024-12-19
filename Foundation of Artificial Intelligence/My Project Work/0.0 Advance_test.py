# Import necessary modules for maze creation and visualization
from pyamaze import maze, agent, COLOR, textLabel
import heapq

# Helper function to get the next cell based on direction
def get_next_cell(current, direction):
    x, y = current
    if direction == 'E':  # Move east
        return (x, y + 1)
    elif direction == 'W':  # Move west
        return (x, y - 1)
    elif direction == 'N':  # Move north
        return (x - 1, y)
    elif direction == 'S':  # Move south
        return (x + 1, y)
    return current  # Return the current cell if direction is invalid

# A* Algorithm
def heuristic(a, b):
    """
    Calculate the Manhattan distance heuristic.
    This is used in the A* algorithm to estimate the cost to reach the goal.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def A_star_search(maze_obj, start=None, goal=None):
    """
    Perform the A* search algorithm to find the shortest path in the maze.
    This algorithm uses both the cost to reach the current cell (g(n)) and
    the estimated cost to reach the goal (h(n)) to guide the search.
    """
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)  # Default to bottom-right corner
    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)  # Default goal to the center of the maze
    
    # Min-heap priority queue to explore the maze
    frontier_a_star = []
    heapq.heappush(frontier_a_star, (0 + heuristic(start, goal), start))  # (f-cost, position)
    
    visited_a_star = {}  # Dictionary to store the path (previous cell) leading to each cell
    exploration_order_a_star = []  # List to store the order in which cells are explored
    explored_a_star = set([start])  # Set of explored cells
    g_costs_a_star = {start: 0}  # Cost to reach each cell from the start

    while frontier_a_star:
        # Pop the cell with the lowest f-cost (f = g + h)
        _, current_a_star = heapq.heappop(frontier_a_star)
        
        if current_a_star == goal:
            break  # Stop once the goal is reached

        # Explore neighboring cells (East, South, North, West)
        for direction in 'ESNW':
            if maze_obj.maze_map[current_a_star][direction]:
                next_cell_a_star = get_next_cell(current_a_star, direction)
                new_g_cost = g_costs_a_star[current_a_star] + 1  # Uniform cost (1 per move)
                
                # If the cell hasn't been explored or we found a better path (lower g-cost)
                if next_cell_a_star not in explored_a_star or new_g_cost < g_costs_a_star.get(next_cell_a_star, float('inf')):
                    g_costs_a_star[next_cell_a_star] = new_g_cost
                    f_cost_a_star = new_g_cost + heuristic(next_cell_a_star, goal)  # f(n) = g(n) + h(n)
                    heapq.heappush(frontier_a_star, (f_cost_a_star, next_cell_a_star))  # Push to priority queue
                    visited_a_star[next_cell_a_star] = current_a_star  # Record the parent cell
                    exploration_order_a_star.append(next_cell_a_star)  # Add to exploration order
                    explored_a_star.add(next_cell_a_star)

    # Reconstruct the path from goal to start by following the visited dictionary
    path_to_goal_a_star = {}
    cell_a_star = goal
    while cell_a_star != start:
        path_to_goal_a_star[visited_a_star[cell_a_star]] = cell_a_star
        cell_a_star = visited_a_star[cell_a_star]

    return exploration_order_a_star, visited_a_star, path_to_goal_a_star

# Main function for A* search visualization
if __name__ == '__main__':
    # Create the maze object
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')
    
    goal_position = (1, 1)  # Example goal position (change as needed)
    
    # Perform A* search
    exploration_order_a_star, visited_a_star, path_to_goal_a_star = A_star_search(m, goal=goal_position)
    
    # Create agents for visualization
    agent_a_star = agent(m, footprints=True, shape='square', color=COLOR.red)  # Agent for A* search order
    agent_trace_a_star = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full path
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.blue, shape='square', filled=True)  # Goal agent

    # Visualize the agents' paths
    m.tracePath({agent_a_star: exploration_order_a_star}, delay=1)
    m.tracePath({agent_trace_a_star: path_to_goal_a_star}, delay=1)
    m.tracePath({agent_goal: path_to_goal_a_star}, delay=1)  # Trace the path from start to goal

    # Display the lengths of the A* search and path
    textLabel(m, 'Goal Position', str(goal_position))
    textLabel(m, 'A* Path Length', len(path_to_goal_a_star) + 1)
    textLabel(m, 'A* Search Length', len(exploration_order_a_star))

    # Run the maze simulation
    m.run()
