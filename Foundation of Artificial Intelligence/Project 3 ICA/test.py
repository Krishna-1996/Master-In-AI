
# load_maze_from_csv('D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/maze--2025-01-03--10-31-10.csv')

import pygame
import csv

# Initialize pygame
pygame.init()

# Define window dimensions and cell size
window_width, window_height = 800, 600
cell_width, cell_height = 30, 30  # Cell size for maze visualization

# Create the maze object
rows, cols = 20, 50
maze_map = {}

# Create obstacles set
obstacles = set()

# Load maze from CSV
def load_maze_from_csv(file_path):
    global maze_map
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0].startswith("("):  # Ignore non-relevant rows
                coords = eval(row[0])  # Convert to tuple (e.g., "(1, 1)" -> (1, 1))
                E, W, N, S = map(int, row[1:])
                maze_map[coords] = {'E': E, 'W': W, 'N': N, 'S': S}

# Load maze from file
load_maze_from_csv('D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/maze--2025-01-03--10-31-10.csv')  # Use the correct path to your maze CSV

# Pygame window setup
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Interactive Maze")

# Draw the maze and obstacles
def draw_maze():
    screen.fill((255, 255, 255))  # Clear screen with white background
    
    for row in range(rows):
        for col in range(cols):
            x = col * cell_width
            y = row * cell_height

            # Draw obstacles in red
            if (row, col) in obstacles:
                pygame.draw.rect(screen, (255, 0, 0), (x, y, cell_width, cell_height))
            else:
                # Draw the walls of the maze
                if not maze_map.get((row + 1, col + 1), {}).get('E', 1):  # No East wall
                    pygame.draw.line(screen, (0, 0, 0), (x + cell_width, y), (x + cell_width, y + cell_height), 2)
                if not maze_map.get((row + 1, col + 1), {}).get('W', 1):  # No West wall
                    pygame.draw.line(screen, (0, 0, 0), (x, y), (x, y + cell_height), 2)
                if not maze_map.get((row + 1, col + 1), {}).get('S', 1):  # No South wall
                    pygame.draw.line(screen, (0, 0, 0), (x, y + cell_height), (x + cell_width, y + cell_height), 2)
                if not maze_map.get((row + 1, col + 1), {}).get('N', 1):  # No North wall
                    pygame.draw.line(screen, (0, 0, 0), (x, y), (x + cell_width, y), 2)

    # Draw buttons (for reset and run algorithm)
    pygame.draw.rect(screen, (0, 255, 0), (window_width - 150, window_height - 100, 130, 40))  # Run Algorithm
    pygame.draw.rect(screen, (255, 255, 0), (window_width - 150, window_height - 150, 130, 40))  # Reset Obstacles
    pygame.draw.rect(screen, (255, 0, 0), (window_width - 150, window_height - 200, 130, 40))  # Reset All

    pygame.display.flip()

# Place or remove obstacles when clicked
def place_obstacle():
    pos = pygame.mouse.get_pos()
    col = pos[0] // cell_width
    row = pos[1] // cell_height
    if (row, col) in obstacles:
        obstacles.remove((row, col))  # Remove obstacle
    else:
        obstacles.add((row, col))  # Add obstacle

# Reset the maze and obstacles
def reset_all():
    global obstacles
    obstacles = set()  # Clear obstacles
    load_maze_from_csv('D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/maze--2025-01-03--10-31-10.csv')  # Reload the maze from the CSV

# Reset obstacles only
def reset_obstacles():
    global obstacles
    obstacles = set()  # Remove obstacles from the set

# Main event loop for interaction
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.pos[0] < window_width - 150:  # Click to place obstacles only within maze area
                place_obstacle()
            else:
                # Handle button clicks
                if window_width - 150 <= event.pos[0] <= window_width - 20:
                    if window_height - 100 <= event.pos[1] <= window_height - 60:  # Run Algorithm
                        print("Run algorithm")  # Implement your algorithm logic here
                    elif window_height - 150 <= event.pos[1] <= window_height - 110:  # Reset Obstacles
                        reset_obstacles()
                    elif window_height - 200 <= event.pos[1] <= window_height - 160:  # Reset All
                        reset_all()

    draw_maze()  # Update display

# Close Pygame
pygame.quit()
