import random
import csv
from pyamaze import maze

def create_obstacles_and_save_csv(file_path='obstacles.csv', maze_size=(50, 100), obstacle_percentage=45):
    m = maze(*maze_size)
    m.CreateMaze()
    
    # Inspect the maze map to verify obstacle coordinates
    for row in range(m.rows):
        for col in range(m.cols):
            print(f"Cell ({row}, {col}): {m.maze_map[(row, col)]}")
    
    total_cells = m.rows * m.cols
    num_obstacles = int(total_cells * (obstacle_percentage / 100))
    valid_cells = [(row, col) for row in range(m.rows) for col in range(m.cols)]
    blocked_cells = random.sample(valid_cells, num_obstacles)
    
    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Row, Col, E, W, N, S"])  # Header
        
        for (row, col) in blocked_cells:
            directions = m.maze_map[(row, col)]
            writer.writerow([row, col, directions['E'], directions['W'], directions['N'], directions['S']])

if __name__ == '__main__':
    create_obstacles_and_save_csv()