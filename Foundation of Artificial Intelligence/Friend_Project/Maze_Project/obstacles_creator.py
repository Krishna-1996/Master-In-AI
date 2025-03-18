import random
import csv
import os

# Maze dimensions
ROWS = 60
COLS = 100

# Output directory
OUTPUT_DIR = "mazes"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_maze(obstacle_percentage):
    """
    Generates a maze with the given obstacle percentage.
    
    :param obstacle_percentage: Percentage of walls to add (0%, 10%, 30%, 50%).
    :return: A maze dictionary where each cell has walls (N, S, E, W).
    """
    maze = {}

    for r in range(ROWS):
        for c in range(COLS):
            maze[(r, c)] = {'N': 1, 'S': 1, 'E': 1, 'W': 1}

    # Create an open path (default: all walls intact)
    for r in range(ROWS):
        for c in range(COLS):
            if r < ROWS - 1:  # Link South
                maze[(r, c)]['S'] = 0
                maze[(r+1, c)]['N'] = 0
            if c < COLS - 1:  # Link East
                maze[(r, c)]['E'] = 0
                maze[(r, c+1)]['W'] = 0

    # Add obstacles randomly
    total_cells = ROWS * COLS
    obstacle_count = int((obstacle_percentage / 100) * total_cells)
    
    obstacle_cells = random.sample(list(maze.keys()), obstacle_count)

    for cell in obstacle_cells:
        direction = random.choice(['N', 'S', 'E', 'W'])
        maze[cell][direction] = 1  # Add a wall in one random direction

    return maze

def save_maze_to_csv(maze, filename):
    """
    Saves the generated maze into a CSV file.

    :param maze: The dictionary representing the maze.
    :param filename: The filename to save the maze as CSV.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Row", "Column", "E", "W", "N", "S"])
        
        for (r, c), walls in maze.items():
            writer.writerow([r, c, walls['E'], walls['W'], walls['N'], walls['S']])

    print(f"Saved maze to {filepath}")

def generate_all_mazes():
    """
    Generates all four maze files (0%, 10%, 30%, 50% obstacles).
    """
    for percentage in [0, 10, 30, 50]:
        maze = generate_maze(percentage)
        filename = f"maze_{percentage}p.csv"
        save_maze_to_csv(maze, filename)

if __name__ == "__main__":
    generate_all_mazes()
