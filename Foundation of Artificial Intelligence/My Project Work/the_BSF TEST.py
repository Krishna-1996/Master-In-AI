from pyamaze import maze, agent, COLOR, textLabel
from collections import deque
import csv

# Function to load the maze from CSV
def load_maze_from_csv(filename):
    # First, read the CSV to determine maze dimensions
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        
        maze_map = {}
        max_row, max_col = 0, 0  # To track the max row and column for dynamic maze dimensions

        # Skip the header row (if any) and read each row in the CSV
        for row in reader:
            # Get the coordinates (row, col) in the format '(r, c)'
            cell = tuple(map(int, row[0][1:-1].split(',')))  # This will give you (row, col) tuple

            # Update max_row and max_col to figure out the size of the maze
            max_row = max(max_row, cell[0])  # Update max row
            max_col = max(max_col, cell[1])  # Update max column

            # Extract the wall data (East, West, North, South)
            E, W, N, S = int(row[1]), int(row[2]), int(row[3]), int(row[4])

            # Store the wall data in the maze_map
            maze_map[cell] = {'E': E, 'W': W, 'N': N, 'S': S}

    # Now create the maze object dynamically based on max_row and max_col
    m = maze(max_row + 1, max_col + 1)  # +1 because pyamaze uses 1-based indexing

    # Print out the maze dimensions (for debugging)
    print(f"Creating a maze of size {m.rows} x {m.cols}")

    # Now load the maze walls into the pyamaze maze object
    for (r, c), walls in maze_map.items():
        # Ensure that the row and column are within bounds before modifying the walls

        # Handling the East direction
        if walls['E'] == 1 and c < m.cols - 1:  # Ensure we are not at the far-right edge
            m.maze_map[(r, c)]['E'] = False
            m.maze_map[(r, c + 1)]['W'] = False  # Move to the next column

        # Handling the West direction
        if walls['W'] == 1 and c > 0:  # Ensure we are not at the far-left edge
            m.maze_map[(r, c)]['W'] = False
            m.maze_map[(r, c - 1)]['E'] = False  # Move to the previous column

        # Handling the North direction
        if walls['N'] == 1 and r > 0:  # Ensure we are not at the top edge
            m.maze_map[(r, c)]['N'] = False
            m.maze_map[(r - 1, c)]['S'] = False  # Move to the previous row

        # Handling the South direction
        if walls['S'] == 1 and r < m.rows - 1:  # Ensure we are not at the bottom edge
            m.maze_map[(r, c)]['S'] = False
            m.maze_map[(r + 1, c)]['N'] = False  # Move to the next row
    
    return m

# BFS Search Algorithm
def BFS_search(maze_obj, start=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    frontier = deque([start])
    visited = {}
    exploration_order = []
    explored = set([start])

    while frontier:
        current = frontier.popleft()

        if current == maze_obj._goal:
            break
        
        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                if next_cell not in explored:
                    frontier.append(next_cell)
                    explored.add(next_cell)
                    visited[next_cell] = current
                    exploration_order.append(next_cell)

    path_to_goal = {}
    cell = maze_obj._goal
    while cell != (maze_obj.rows, maze_obj.cols):
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

# Helper to get next cell based on direction
def get_next_cell(current, direction):
    row, col = current
    if direction == 'E':
        return (row, col + 1)
    elif direction == 'W':
        return (row, col - 1)
    elif direction == 'S':
        return (row + 1, col)
    elif direction == 'N':
        return (row - 1, col)

# Main function
if __name__ == '__main__':
    maze_filename = 'D:/Masters Projects/Master-In-AI/random_maze_format.csv'
    
    # Load the maze from CSV file and create maze object
    m = load_maze_from_csv(maze_filename)

    # Perform BFS search on the maze and get the exploration order and paths
    exploration_order, visited_cells, path_to_goal = BFS_search(m)

    # Create agents to visualize the BFS search process
    agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red)  
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  
    agent_goal = agent(m, 1, 1, footprints=True, color=COLOR.blue, shape='square', filled=True, goal=(m.rows, m.cols))

    # Visualize the agents' movements
    m.tracePath({agent_bfs: exploration_order}, delay=5)
    m.tracePath({agent_goal: visited_cells}, delay=100)
    m.tracePath({agent_trace: path_to_goal}, delay=100)

    # Display lengths of the BFS path and search steps
    textLabel(m, 'BFS Path Length', len(path_to_goal) + 1)
    textLabel(m, 'BFS Search Length', len(exploration_order))

    # Run the maze visualization
    m.run()
