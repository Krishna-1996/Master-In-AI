import pickle

def create_maze(rows, cols):
    """Creates an empty maze of size rows x cols."""
    return [[0 for _ in range(cols)] for _ in range(rows)]

def save_maze(maze, filename):
    """Saves the maze to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(maze, f)

def load_maze(filename):
    """Loads the maze from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def print_maze(maze):
    """Prints the maze to the console."""
    for row in maze:
        print("".join(["#" if cell == 1 else "G" if cell == 2 else "S" if cell == 3 else "." if cell == 0 else "*" for cell in row]))
