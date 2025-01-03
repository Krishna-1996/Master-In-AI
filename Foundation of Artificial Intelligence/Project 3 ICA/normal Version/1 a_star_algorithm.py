from pyamaze import maze, agent, COLOR, textLabel
import heapq
import random
import csv

def heuristic(a, b):
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_next_cell(current, direction):
    """Calculate the next cell based on the current cell and direction."""
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

def load_maze_from_csv(file_path, maze_obj):
    """Load maze from a CSV file and update maze_obj's maze_map."""
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header

        for row in reader:
            # Extract the coordinates and direction info from the CSV row
            coords = eval(row[0])  # Converts string to tuple (row, col)
            E, W, N, S = map(int, row[1:])  # Convert direction values to integers
            
            # Update the maze map
            maze_obj.maze_map[coords] = {"E": E, "W": W, "N": N, "S": S}

def add_obstacles(maze_obj, obstacle_percentage=20):
    
    total_cells = maze_obj.rows * maze_obj.cols
    num_obstacles = int(total_cells * (obstacle_percentage / 100))

    # Create a list of all valid cell positions
    valid_cells = [(row, col) for row in range(maze_obj.rows) for col in range(maze_obj.cols)]

    # Randomly select obstacle positions
    blocked_cells = random.sample(valid_cells, num_obstacles)

    # Store obstacle locations for later use in visualization
    obstacle_locations = []

    for (row, col) in blocked_cells:
        if (row, col) in maze_obj.maze_map:
            # Randomly block directions (set E, W, N, S to 0 for obstacles)
            if random.choice([True, False]):
                maze_obj.maze_map[(row, col)]["E"] = 0
                maze_obj.maze_map[(row, col)]["W"] = 0
            if random.choice([True, False]):
                maze_obj.maze_map[(row, col)]["N"] = 0
                maze_obj.maze_map[(row, col)]["S"] = 0

            # Store the location of the obstacle
            obstacle_locations.append((row, col))

    return obstacle_locations  # Return all the blocked cells


def A_star_search(maze_obj, start=None, goal=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")

    # Min-heap priority queue
    frontier = []
    heapq.heappush(frontier, (0 + heuristic(start, goal), start))  # (f-cost, position)
    visited = {}
    exploration_order = []
    explored = set([start])
    g_costs = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for direction in 'ESNW':
            # Check if the move is valid (i.e., the cell is not blocked)
            if maze_obj.maze_map[current][direction] == 1:  # Only move if there is no obstacle
                next_cell = get_next_cell(current, direction)
                new_g_cost = g_costs[current] + 1  # +1 for each move (uniform cost)
                
                if next_cell not in explored or new_g_cost < g_costs.get(next_cell, float('inf')):
                    g_costs[next_cell] = new_g_cost
                    f_cost = new_g_cost + heuristic(next_cell, goal)
                    heapq.heappush(frontier, (f_cost, next_cell))
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
                    explored.add(next_cell)

    # Check if the goal is unreachable
    if goal not in visited:
        print("Goal is unreachable!")
        return [], {}, {}

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal


# Main function for A* search
if __name__ == '__main__':
    m = maze(50, 100)  # Maze size 50x100

    # Load maze from CSV file and update maze_map
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/normal Version/maze--2025-01-03--13-49-03.csv')

    # Load maze from CSV file and update maze_map
    load_maze_from_csv('D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/normal Version/maze--2025-01-03--13-49-03.csv', m)

    # Dynamically add obstacles
    obstacle_locations = add_obstacles(m, obstacle_percentage=25)  # Change obstacle percentage as needed
    
    goal_position = (1, 1)  # Example goal, change to any valid coordinate

    exploration_order, visited_cells, path_to_goal = A_star_search(m, goal=goal_position)

    # If a path is found, trace it; otherwise, print message
    if path_to_goal:
        agent_astar = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
        agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)
        agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

        m.tracePath({agent_astar: exploration_order}, delay=1)
        m.tracePath({agent_trace: path_to_goal}, delay=1)
        m.tracePath({agent_goal: visited_cells}, delay=1)

        textLabel(m, 'Goal Position', str(goal_position))
        textLabel(m, 'A* Path Length', len(path_to_goal) + 1)
        textLabel(m, 'A* Search Length', len(exploration_order))

    else:
        print("No path found to the goal!")

    m.run()
