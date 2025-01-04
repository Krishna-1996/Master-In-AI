import csv
import random
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

# Code 1 Functions
def load_maze_from_csv(file_path):
    """Load the maze from the main CSV file."""
    maze_map = {}
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        
        for row in reader:
            coords = eval(row[0])  # Convert string to tuple (row, col)
            E, W, N, S = map(int, row[1:])  # Convert direction values to integers
            maze_map[coords] = {"E": E, "W": W, "N": N, "S": S}
    
    return maze_map

def add_obstacles(maze_map, obstacle_percentage=20):
    """Add random obstacles to the maze."""
    total_cells = len(maze_map)
    num_obstacles = int(total_cells * (obstacle_percentage / 100))
    
    # Randomly select obstacle positions
    blocked_cells = random.sample(list(maze_map.keys()), num_obstacles)
    
    for (row, col) in blocked_cells:
        # Randomly block directions (E, W, N, S) for obstacles
        if random.choice([True, False]):
            maze_map[(row, col)]["E"] = 0
            maze_map[(row, col)]["W"] = 0
        if random.choice([True, False]):
            maze_map[(row, col)]["N"] = 0
            maze_map[(row, col)]["S"] = 0
    
    return maze_map

def save_maze_to_csv(maze_map, file_path):
    """Save the maze (with obstacles) to a CSV file."""
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['cell', 'E', 'W', 'N', 'S'])
        
        # Write maze data with obstacles
        for (row, col), directions in maze_map.items():
            E = directions.get('E', 1)  # Default to 1 (open) if not specified
            W = directions.get('W', 1)  # Default to 1 (open) if not specified
            N = directions.get('N', 1)  # Default to 1 (open) if not specified
            S = directions.get('S', 1)  # Default to 1 (open) if not specified
            writer.writerow([(row, col), E, W, N, S])

# Code 2 Functions
def get_next_cell(current, direction):
    """Calculate the next cell based on the current cell and direction."""
    x, y = current
    if direction == 'E':
        return (x, y + 1)
    elif direction == 'W':
        return (x, y - 1)
    elif direction == 'N':
        return (x - 1, y)
    elif direction == 'S':
        return (x + 1, y)
    return current

def bfs_search(maze_obj, start=None, goal=None):
    """Breadth-First Search algorithm."""
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    if goal is None:
        goal = (1, 1)  # Default goal position

    frontier = deque([start])  # Queue for BFS
    visited = {}
    exploration_order = []
    explored = set([start])

    while frontier:
        current = frontier.popleft()

        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction] == 1:
                next_cell = get_next_cell(current, direction)
                if next_cell not in explored:
                    frontier.append(next_cell)
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
                    explored.add(next_cell)

    if goal not in visited:
        return [], {}, {}

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

# Main Script
def main_loop():
    original_csv_path = 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/Advance Version of obstacle project/Maze_1_90_loopPercent.csv'
    modified_csv_path = 'maze_with_obstacles.csv'

    goal_position = (1, 1)
    success = False
    retry_count = 0

    while not success:
        retry_count += 1
        print(f"Retrying... ({retry_count})")
        
        # Step 1: Generate Maze with Obstacles
        maze_map = load_maze_from_csv(original_csv_path)
        maze_map_with_obstacles = add_obstacles(maze_map, obstacle_percentage=70)
        save_maze_to_csv(maze_map_with_obstacles, modified_csv_path)

        # Step 2: Run BFS Search
        m = maze(50, 100)
        m.CreateMaze(loadMaze=modified_csv_path)
        exploration_order, visited_cells, path_to_goal = bfs_search(m, goal=goal_position)

        if path_to_goal:
            print(f"Path found after {retry_count} attempts!")
            success = True
            
            # Visualize the solution
            agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
            m.tracePath({agent_bfs: path_to_goal}, delay=25)
            m.run()
        else:
            print("No path found.")

if __name__ == '__main__':
    main_loop()
