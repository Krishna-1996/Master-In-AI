import csv  # Import the csv module to save maze data into CSV format
import random  # Import the random module to generate random obstacles in the maze
from pyamaze import maze  # Import the maze class from the pyamaze library

# Constants defining the size of the maze and the obstacle percentages
ROWS, COLS = 70, 120  # Dimensions of the maze (70 rows and 120 columns)
OBSTACLE_PERCENTAGES = [0, 10, 20, 30]  # Different obstacle percentages to generate in the maze

# Function to generate obstacles in the maze
def generate_obstacles(maze_obj, percentage):
    total_cells = ROWS * COLS  # Total number of cells in the maze
    num_obstacles = int(total_cells * (percentage / 100))  # Calculate how many obstacles to place
    
    # Define the rectangle boundaries where obstacles can be placed
    min_row, max_row = 3, 67  # Row indices within which obstacles can be placed
    min_col, max_col = 3, 117  # Column indices within which obstacles can be placed
    
    # Create a list of all possible cells within the defined rectangular area
    all_cells = [(r, c) for r in range(min_row, max_row + 1) for c in range(min_col, max_col + 1)]
    
    # Randomly select cells for placing obstacles, ensuring we place the required number
    obstacle_cells = random.sample(all_cells, num_obstacles)

    # Loop through each selected cell and mark it as an obstacle
    for (r, c) in obstacle_cells:
        # Avoid placing obstacles on the start (1,1) and end (ROWS, COLS) points
        if (r, c) != (1, 1) and (r, c) != (ROWS, COLS):
            # If the cell exists in the maze, remove walls (set to 0)
            if (r, c) in maze_obj.maze_map:
                # Set all four possible directions (East, West, North, South) to 0 (removing walls)
                for direction in ['E', 'W', 'N', 'S']:
                    maze_obj.maze_map[(r, c)][direction] = 0

# Function to save the generated maze to a CSV file
def save_maze_to_csv(maze_obj, filename):
    # Open a CSV file in write mode
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)  # Create a CSV writer object
        # Write the header row in the CSV file
        writer.writerow(["Cell", "E", "W", "N", "S"])  
        # Loop through all cells and their corresponding walls, and write them into the CSV file
        for cell, walls in maze_obj.maze_map.items():
            # Write each cell and its wall states (East, West, North, South) into a row
            writer.writerow([cell, walls["E"], walls["W"], walls["N"], walls["S"]])

# Main execution block
if __name__ == "__main__":
    # Iterate over different obstacle percentages
    for percentage in OBSTACLE_PERCENTAGES:
        # Create a new maze object with the specified size (ROWS x COLS)
        m = maze(ROWS, COLS)
        # Create a maze with a 80% chance of creating loops (randomized paths)
        m.CreateMaze(loopPercent=80)
        # Generate obstacles in the maze based on the current percentage
        generate_obstacles(m, percentage)
        # Define the filename for saving the maze (with obstacle percentage in the filename)
        filename = f"mazes/Obstacles_Design_{percentage}p.csv"
        # Save the maze data to the CSV file
        save_maze_to_csv(m, filename)
        # Print a confirmation message indicating that the maze has been saved
        print(f"Saved: {filename}")
