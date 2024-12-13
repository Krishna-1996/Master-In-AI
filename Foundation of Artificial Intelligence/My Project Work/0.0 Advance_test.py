from pyamaze import maze, agent, COLOR, textLabel
import heapq  # Priority queue for A* algorithm

# Heuristic function for A* (Manhattan distance)
def heuristic(cell, goal):
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

# A* algorithm to find the shortest path in the maze
def a_star(m, start=None):
    # Set default start point if not provided (bottom-right corner)
    start = start or (m.rows, m.cols)
    
    # Initialize distances and priority queue
    g_costs = {cell: float('inf') for cell in m.grid}
    g_costs[start] = 0
    f_costs = {cell: float('inf') for cell in m.grid}
    f_costs[start] = heuristic(start, m._goal)
    
    # Priority queue to explore the cell with the lowest f(n)
    priority_queue = [(f_costs[start], start)]
    
    # Store the path and the order of exploration
    came_from = {}
    exploration_order = []
    
    # A* search loop
    while priority_queue:
        _, current_cell = heapq.heappop(priority_queue)
        
        # If goal is reached, stop
        if current_cell == m._goal:
            break
        
        # Explore neighboring cells (East, South, North, West)
        for d in 'ESNW':
            if m.maze_map[current_cell][d]:  # Check if move is valid
                # Get neighboring cell based on direction
                if d == 'E': neighbor_cell = (current_cell[0], current_cell[1] + 1)
                elif d == 'W': neighbor_cell = (current_cell[0], current_cell[1] - 1)
                elif d == 'S': neighbor_cell = (current_cell[0] + 1, current_cell[1])
                elif d == 'N': neighbor_cell = (current_cell[0] - 1, current_cell[1])
                
                # Calculate tentative g(n) cost
                tentative_g_cost = g_costs[current_cell] + 1
                
                # If found a better path, update the costs and add to priority queue
                if tentative_g_cost < g_costs[neighbor_cell]:
                    came_from[neighbor_cell] = current_cell
                    g_costs[neighbor_cell] = tentative_g_cost
                    f_costs[neighbor_cell] = tentative_g_cost + heuristic(neighbor_cell, m._goal)
                    heapq.heappush(priority_queue, (f_costs[neighbor_cell], neighbor_cell))
                    exploration_order.append(neighbor_cell)
    
    # Reconstruct the path from goal to start
    path_to_goal = []
    cell = m._goal
    while cell != start:
        path_to_goal.append(cell)
        cell = came_from[cell]
    path_to_goal.append(start)
    path_to_goal.reverse()  # Reverse path to get start-to-goal
    
    return exploration_order, came_from, path_to_goal

# Main function to create and visualize the maze
if __name__ == '__main__':
    # Create a maze and load a custom maze from a CSV file
    m = maze(30, 50)
    m.CreateMaze(loadMaze= 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv')

    # Run A* algorithm to get exploration order and path
    exploration_order, came_from, path_to_goal = a_star(m)

    # Create agents for visualization
    a = agent(m, footprints=True, shape='square', color=COLOR.green)  # A* search path agent
    b = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Path agent
    c = agent(m, 1, 1, footprints=True, color=COLOR.blue, shape='square', filled=True, goal=(m.rows, m.cols))  # Goal-seeking agent

    # Trace agents' paths
    m.tracePath({a: exploration_order}, delay= 1)  # A* exploration path
    m.tracePath({b: path_to_goal}, delay=1)  # Path from start to goal
    m.tracePath({c: path_to_goal}, delay=1)  # Trace final goal path

    # Display path lengths as labels
    textLabel(m, 'A* Path Length', len(path_to_goal))
    textLabel(m, 'A* Search Length', len(exploration_order))

    # Run the simulation
    m.run()
