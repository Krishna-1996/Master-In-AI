# Import necessary modules for maze generation, A* algorithm, Dijkstra's algorithm, and maze visualization
from pyamaze import maze, agent, COLOR, textLabel
import heapq  # For the priority queue used in both algorithms

# Heuristic function for A* (Manhattan distance)
def heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

# A* Algorithm
def a_star(m, start=None):
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
                if d == 'E':  # East
                    neighbor_cell = (current_cell[0], current_cell[1] + 1)
                elif d == 'W':  # West
                    neighbor_cell = (current_cell[0], current_cell[1] - 1)
                elif d == 'S':  # South
                    neighbor_cell = (current_cell[0] + 1, current_cell[1])
                elif d == 'N':  # North
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

# Dijkstra's Algorithm
def dijkstra(m, start=None):
    if start is None:
        start = (m.rows, m.cols)  # Bottom-right corner
    
    distances = {cell: float('inf') for cell in m.grid}
    distances[start] = 0
    
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    
    came_from = {}
    exploration_order = []
    
    while priority_queue:
        current_distance, current_cell = heapq.heappop(priority_queue)
        
        if current_cell == m._goal:
            break
        
        for d in 'ESNW':  # Directions: East, South, North, West
            if m.maze_map[current_cell][d] == True:
                if d == 'E':  # East
                    neighbor_cell = (current_cell[0], current_cell[1] + 1)
                elif d == 'W':  # West
                    neighbor_cell = (current_cell[0], current_cell[1] - 1)
                elif d == 'S':  # South
                    neighbor_cell = (current_cell[0] + 1, current_cell[1])
                elif d == 'N':  # North
                    neighbor_cell = (current_cell[0] - 1, current_cell[1])

                new_distance = current_distance + 1
                if new_distance < distances[neighbor_cell]:
                    distances[neighbor_cell] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor_cell))
                    came_from[neighbor_cell] = current_cell
                    exploration_order.append(neighbor_cell)

    path_to_goal = []
    cell = m._goal
    while cell != start:
        path_to_goal.append(cell)
        cell = came_from[cell]
    path_to_goal.append(start)
    path_to_goal.reverse()
    
    return exploration_order, came_from, path_to_goal

# Main function to create and run the maze
if __name__ == '__main__':
    # Create a 15x15 maze and load a custom maze from a CSV file
    m = maze(15, 15)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/ICA Pyamaze Example/mazetest.csv')

    # Perform A* algorithm on the maze to find the search order and paths
    exploration_order_a_star, came_from_a_star, path_to_goal_a_star = a_star(m)

    # Perform Dijkstra's algorithm on the maze to find the search order and paths
    exploration_order_dijkstra, came_from_dijkstra, path_to_goal_dijkstra = dijkstra(m)

    # Create agents to visualize the maze solving process
    a_star_agent = agent(m, footprints=True, shape='square', color=COLOR.green)  # A* agent for search order
    dijkstra_agent = agent(m, footprints=True, shape='square', color=COLOR.red)  # Dijkstra agent for search order
    path_agent = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Path tracing agent

    # Trace the paths found by A* and Dijkstra
    m.tracePath({a_star_agent: exploration_order_a_star}, delay=10)  # A* search order
    m.tracePath({dijkstra_agent: exploration_order_dijkstra}, delay=10)  # Dijkstra search order
    m.tracePath({path_agent: path_to_goal_a_star}, delay=100)  # A* path
    m.tracePath({path_agent: path_to_goal_dijkstra}, delay=100)  # Dijkstra path

    # Display the lengths of the paths and the number of cells explored
    l_a_star = textLabel(m, 'A* Path Length', len(path_to_goal_a_star))
    l_dijkstra = textLabel(m, 'Dijkstra Path Length', len(path_to_goal_dijkstra))
    l_a_star_explored = textLabel(m, 'A* Search Length', len(exploration_order_a_star))
    l_dijkstra_explored = textLabel(m, 'Dijkstra Search Length', len(exploration_order_dijkstra))

    # Run the maze simulation
    m.run()
