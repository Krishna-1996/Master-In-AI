import random
import csv

# Function to generate a random 30x50 maze
def generate_random_maze(rows, cols):
    # Create an empty grid with all walls (1 represents wall, 0 represents path)
    maze = [[{'E': 1, 'W': 1, 'N': 1, 'S': 1} for _ in range(cols)] for _ in range(rows)]

    # Define the starting and ending points
    start = (1, 1)  # Top-left corner
    end = (rows-2, cols-2)  # Bottom-right corner

    # Set start and end points as open paths
    maze[start[0]][start[1]] = {'E': 0, 'W': 0, 'N': 0, 'S': 0}
    maze[end[0]][end[1]] = {'E': 0, 'W': 0, 'N': 0, 'S': 0}

    # Randomly open paths in the maze
    for r in range(1, rows-1, 2):  # Only place paths on odd rows
        for c in range(1, cols-1, 2):  # Only place paths on odd columns
            if random.random() > 0.3:  # 70% chance to make it a path
                maze[r][c] = {'E': 0, 'W': 0, 'N': 0, 'S': 0}

    # Function to carve a path between start and end using DFS
    def carve_path(start, end):
        stack = [start]
        visited = set()
        while stack:
            current = stack[-1]
            if current == end:
                break
            visited.add(current)
            neighbors = []
            # Explore the neighbors: up, down, left, right (2 steps away)
            for dr, dc, direction in [(-2, 0, 'N'), (2, 0, 'S'), (0, -2, 'W'), (0, 2, 'E')]:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 < nr < rows-1 and 0 < nc < cols-1 and (nr, nc) not in visited:
                    neighbors.append(((nr, nc), direction))  # Store as a tuple of (coordinates, direction)
            if neighbors:
                next_cell, direction = random.choice(neighbors)  # Unpack properly
                nr, nc = next_cell
                maze[nr][nc][direction] = 0
                # Break the wall between current and next cell
                maze[(nr + current[0]) // 2][(nc + current[1]) // 2][{'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}[direction]] = 0
                stack.append(next_cell)
            else:
                stack.pop()

    # Ensure a path is carved between start and end
    carve_path(start, end)

    return maze

# Function to save the maze to CSV in older format (Cell, E, W, N, S)
def save_maze_to_csv(maze, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for r in range(1, len(maze)-1):  # Ignore first and last row
            for c in range(1, len(maze[0])-1):  # Ignore first and last column
                cell = (r, c)
                E = maze[r][c]['E']
                W = maze[r][c]['W']
                N = maze[r][c]['N']
                S = maze[r][c]['S']
                writer.writerow([cell, E, W, N, S])

# Set maze dimensions
rows, cols = 30, 50

# Generate the maze
maze = generate_random_maze(rows, cols)

# Save the maze to a CSV file in the old format (Cell, E, W, N, S)
csv_filename = 'random_maze_old_format.csv'
save_maze_to_csv(maze, csv_filename)
print(f'Maze saved to {csv_filename}')
