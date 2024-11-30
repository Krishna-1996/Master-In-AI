from pyamaze import maze

def create_and_store_maze(rows, cols):
    # Create a maze object with the given dimensions
    m = maze(rows, cols)
    
    # Generate a random maze layout
    m.CreateMaze()

    # Store the maze in a 2D list (variable)
    maze_layout = []
    for row in range(rows):
        row_data = []
        for col in range(cols):
            # Access the maze layout using the internal _maze attribute
            if m._maze[(row, col)] == 1:  # Wall
                row_data.append(1)
            else:  # Path
                row_data.append(0)
        maze_layout.append(row_data)

    return maze_layout

# Generate and store a 30x50 maze in a variable
maze_layout = create_and_store_maze(30, 50)

# Print the maze stored in the variable
for row in maze_layout:
    print(row)
