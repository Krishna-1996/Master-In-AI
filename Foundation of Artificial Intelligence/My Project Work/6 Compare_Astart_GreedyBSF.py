# Import necessary modules for maze generation, algorithms, and maze visualization
from pyamaze import maze, agent, COLOR, textLabel
import heapq  # For the priority queue used in both algorithms

def heuristic(cell, goal):
    """
    Calculate the Manhattan distance from the current cell to the goal.
    This heuristic is used in both A* and Greedy BFS algorithms.
    """
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

# A* Algorithm
def a_star(m, start=None):
    """
    Perform A* Algorithm to find the shortest path in the maze.
    A* algorithm uses both the distance from the start and a heuristic to guide the search.
    """
    if start is None:
        start = (m.rows, m.cols)  # Bottom-right corner
    
    g_costs = {cell: float('inf') for cell in m.grid}  # g(n): cost from start to current node
    g_costs[start] = 0  # Starting point has a g(n) cost of 0
    f_costs = {cell: float('inf') for cell in m.grid}  # f(n): g(n) + h(n)
    f_costs[start] = heuristic(start, m._goal)  # f(n) = g(n) + h(n)
    
    priority_queue = []
    heapq.heappush(priority_queue, (f_costs[start], start))  # Push the start cell with f(n)
    
    came_from = {}
    exploration_order = []
    
    while priority_queue:
        _, current_cell = heapq.heappop(priority_queue)
        
        if current_cell == m._goal:
            break
        
        for d in 'ESNW':  # Directions: East, South, North, West
            if m.maze_map[current_cell][d] == True:
                if d == 'E': 
                    neighbor_cell = (current_cell[0], current_cell[1] + 1)
                elif d == 'W': 
                    neighbor_cell = (current_cell[0], current_cell[1] - 1)
                elif d == 'S': 
                    neighbor_cell = (current_cell[0] + 1, current_cell[1])
                elif d == 'N': 
                    neighbor_cell = (current_cell[0] - 1, current_cell[1])
                
                tentative_g_cost = g_costs[current_cell] + 1
                if tentative_g_cost < g_costs[neighbor_cell]:
                    came_from[neighbor_cell] = current_cell
                    g_costs[neighbor_cell] = tentative_g_cost
                    f_costs[neighbor_cell] = tentative_g_cost + heuristic(neighbor_cell, m._goal)
                    heapq.heappush(priority_queue, (f_costs[neighbor_cell], neighbor_cell))
                    exploration_order.append(neighbor_cell)

    path_to_goal = []
    cell = m._goal
    while cell != start:
        path_to_goal.append(cell)
        cell = came_from[cell]
    path_to_goal.append(start)
    path_to_goal.reverse()
    
    return exploration_order, came_from, path_to_goal

# Greedy BFS Algorithm
def greedy_bfs(m, start=None):
    """
    Perform Greedy Best-First Search (Greedy BFS) Algorithm to find the shortest path in the maze.
    Greedy BFS algorithm uses only the heuristic to guide the search.
    """
    if start is None:
        start = (m.rows, m.cols)  # Bottom-right corner
    
    f_costs = {cell: float('inf') for cell in m.grid}
    f_costs[start] = heuristic(start, m._goal)
    
    priority_queue = []
    heapq.heappush(priority_queue, (f_costs[start], start))
    
    came_from = {}
    exploration_order = []
    
    while priority_queue:
        _, current_cell = heapq.heappop(priority_queue)
        
        if current_cell == m._goal:
            break
        
        for d in 'ESNW':
            if m.maze_map[current_cell][d] == True:
                if d == 'E': 
                    neighbor_cell = (current_cell[0], current_cell[1] + 1)
                elif d == 'W': 
                    neighbor_cell = (current_cell[0], current_cell[1] - 1)
                elif d == 'S': 
                    neighbor_cell = (current_cell[0] + 1, current_cell[1])
                elif d == 'N': 
                    neighbor_cell = (current_cell[0] - 1, current_cell[1])
                
                if neighbor_cell not in came_from:
                    came_from[neighbor_cell] = current_cell
                    f_costs[neighbor_cell] = heuristic(neighbor_cell, m._goal)
                    heapq.heappush(priority_queue, (f_costs[neighbor_cell], neighbor_cell))
                    exploration_order.append(neighbor_cell)

    path_to_goal = []
    cell = m._goal
    while cell != start:
        path_to_goal.append(cell)
        cell = came_from[cell]
    path_to_goal.append(start)
    path_to_goal.reverse()
    
    return exploration_order, came_from, path_to_goal

# Main function to create and run the maze with both algorithms
if __name__ == '__main__':
    m = maze(15, 15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv')
    
    # Run A* algorithm
    exploration_order_a_star, came_from_a_star, path_to_goal_a_star = a_star(m)

    # Run Greedy BFS algorithm
    exploration_order_greedy_bfs, came_from_greedy_bfs, path_to_goal_greedy_bfs = greedy_bfs(m)

    # Create agents for visualization
    a_star_agent = agent(m, footprints=True, shape='square', color=COLOR.green)  # A* search agent
    greedy_bfs_agent = agent(m, footprints=True, shape='square', color=COLOR.red )  # Greedy BFS search agent
    path_agent_a_star = agent(m, footprints=True, shape='star', color=COLOR.blue)  # A* path tracing agent
    path_agent_greedy_bfs = agent(m, footprints=True, shape='star', color=COLOR.yellow )  # Greedy BFS path tracing agent

    # Trace paths for each algorithm
    m.tracePath({a_star_agent: exploration_order_a_star}, delay=1)  # A* search order
    m.tracePath({greedy_bfs_agent: exploration_order_greedy_bfs}, delay=1)  # Greedy BFS search order
    m.tracePath({path_agent_a_star: path_to_goal_a_star}, delay=10)  # A* final path
    m.tracePath({path_agent_greedy_bfs: path_to_goal_greedy_bfs}, delay=10)  # Greedy BFS final path

    # Display the lengths of the paths for both algorithms
    l = textLabel(m, 'A* Path Length', len(path_to_goal_a_star))  # A* path length
    l = textLabel(m, 'Greedy BFS Path Length', len(path_to_goal_greedy_bfs))  # Greedy BFS path length
    l = textLabel(m, 'A* Search Length', len(exploration_order_a_star))  # A* search length
    l = textLabel(m, 'Greedy BFS Search Length', len(exploration_order_greedy_bfs))  # Greedy BFS search length

    # Run the maze simulation
    m.run()
