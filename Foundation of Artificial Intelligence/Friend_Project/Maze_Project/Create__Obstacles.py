import csv
import random
from pyamaze import maze

ROWS, COLS = 70, 120
OBSTACLE_PERCENTAGES = [0, 10, 20, 40]

def generate_obstacles(maze_obj, percentage):
    total_cells = ROWS * COLS
    num_obstacles = int(total_cells * (percentage / 100))
    
    # Define the rectangle boundaries
    min_row, max_row = 3, 67
    min_col, max_col = 3, 117
    
    # Create list of cells within the rectangle
    all_cells = [(r, c) for r in range(min_row, max_row + 1) for c in range(min_col, max_col + 1)]
    
    # Randomly sample cells for obstacles
    obstacle_cells = random.sample(all_cells, num_obstacles)

    for (r, c) in obstacle_cells:
        if (r, c) != (1, 1) and (r, c) != (ROWS, COLS):
            if (r, c) in maze_obj.maze_map:
                for direction in ['E', 'W', 'N', 'S']:
                    maze_obj.maze_map[(r, c)][direction] = 0

def save_maze_to_csv(maze_obj, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cell", "E", "W", "N", "S"])
        for cell, walls in maze_obj.maze_map.items():
            writer.writerow([cell, walls["E"], walls["W"], walls["N"], walls["S"]])

if __name__ == "__main__":
    for percentage in OBSTACLE_PERCENTAGES:
        m = maze(ROWS, COLS)
        m.CreateMaze(loopPercent=80)
        generate_obstacles(m, percentage)
        filename = f"mazes/Obstacles_Design_{percentage}p.csv"
        save_maze_to_csv(m, filename)
        print(f"Saved: {filename}")
