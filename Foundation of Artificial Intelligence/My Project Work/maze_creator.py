import csv
import random

# Function to generate random walls for each cell
def generate_maze(rows, cols):
    maze = []
    
    for x in range(rows):
        for y in range(cols):
            # Randomly decide if there's a wall or open path in each direction
            east = random.choice([0, 1])  # 1 means wall, 0 means no wall
            west = random.choice([0, 1])
            north = random.choice([0, 1])
            south = random.choice([0, 1])
            
            # Append the cell's coordinates and wall information
            maze.append(((x+1, y+1), east, west, north, south))
    
    return maze

# Function to save the maze data to a CSV file
def save_to_csv(maze, filename='maze.csv'):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Cell', 'East', 'West', 'North', 'South']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for cell, east, west, north, south in maze:
            writer.writerow({'Cell': f'({cell[0]}, {cell[1]})', 
                             'East': east, 
                             'West': west, 
                             'North': north, 
                             'South': south})

# Set the size of the maze grid (rows and columns)
rows = 5
cols = 5

# Generate a random maze
maze = generate_maze(rows, cols)

# Save the maze to a CSV file
save_to_csv(maze, 'maze.csv')

print("Maze CSV file has been generated!")
