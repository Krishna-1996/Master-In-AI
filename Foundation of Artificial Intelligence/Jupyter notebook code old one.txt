# file_path = 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/mazetest - Test.csv'
# *****************************************************************************
import csv
from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

def get_maze_dimensions(file_path):
    """Reads the maze file to determine its dimensions."""
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = sum(1 for _ in reader)  # Count the number of rows
        file.seek(0)  # Reset reader
        cols = len(next(reader))  # Count the number of columns in the first row
    return rows, cols

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

def load_maze_from_csv(file_path):
    """Load maze from a CSV file and create the maze object accordingly."""
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        maze_data = [row for row in reader]

    return maze_data

def create_maze_from_data(maze_data):
    """Create the maze object from the loaded maze data."""
    rows = len(maze_data)
    cols = len(maze_data[0])

    m = maze(rows, cols)  # Create maze with detected dimensions
    for r in range(rows):
        for c in range(cols):
            if maze_data[r][c] == '1':  # Wall
                m.walls[r, c] = True
            else:  # Path
                m.walls[r, c] = False

    return m

if __name__ == '__main__':
    file_path = 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/mazetest - Test.csv'
    
    # Load maze from CSV
    maze_data = load_maze_from_csv(file_path)
    rows, cols = len(maze_data), len(maze_data[0])  # Automatically get maze dimensions
    
    # Create maze object from loaded data
    m = create_maze_from_data(maze_data)

    # Set start and goal positions dynamically if required
    start = (0, 0)  # You can set it as per your maze design
    goal = (rows - 1, cols - 1)  # Set goal at bottom-right, adjust as necessary

    # Perform BFS search
    exploration_order, visited_cells, path_to_goal = BFS_search(m, start)

    # Set up agents and display the maze
    agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red)
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)
    agent_goal = agent(m, goal[0], goal[1], footprints=True, color=COLOR.blue, shape='square', filled=True, goal=(m.rows, m.cols))

    m.tracePath({agent_bfs: exploration_order}, delay=5)
    m.tracePath({agent_goal: visited_cells}, delay=100)
    m.tracePath({agent_trace: path_to_goal}, delay=100)

    # Display BFS path length and search length
    textLabel(m, 'BFS Path Length', len(path_to_goal) + 1)
    textLabel(m, 'BFS Search Length', len(exploration_order))

    # Run the maze simulation
    m.run()
  