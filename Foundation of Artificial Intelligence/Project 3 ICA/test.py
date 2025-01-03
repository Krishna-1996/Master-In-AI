import pygame
from pyamaze import maze, agent, COLOR
from queue import PriorityQueue
import csv

# Initialize pygame
pygame.init()

# Maze configuration
window_width, window_height = 800, 600
cell_width, cell_height = 40, 40  # Cell size adjusted for better visibility

# Create the maze object
m = maze(20, 50)  # Assuming maze is 20x50 (rows x columns)
obstacles = set()

# Create maze map from CSV data
def load_maze_from_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0].startswith("("):  # To avoid header rows
                coords = eval(row[0])  # Convert string to tuple (e.g., "(1, 1)" -> (1, 1))
                E, W, N, S = map(int, row[1:])
                if coords not in m.maze_map:  # Ensure the cell exists in the map
                    m.maze_map[coords] = {'E': E, 'W': W, 'N': N, 'S': S}

# Loading maze from CSV
load_maze_from_csv('D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/maze--2025-01-03--10-31-10.csv')

# Pygame window setup
screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Interactive Maze")

# A* Algorithm for pathfinding
def A_star_search(maze_obj, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan Distance

    open_set = PriorityQueue()
    open_set.put((0, start))  # (f_score, position)
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    came_from = {}
    exploration_order = []

    while not open_set.empty():
        _, current = open_set.get()

        if current == goal:
            break

        exploration_order.append(current)

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                tentative_g_score = g_score[current] + 1

                if next_cell not in g_score or tentative_g_score < g_score[next_cell]:
                    came_from[next_cell] = current
                    g_score[next_cell] = tentative_g_score
                    f_score[next_cell] = g_score[next_cell] + heuristic(next_cell, goal)
                    open_set.put((f_score[next_cell], next_cell))

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[came_from[cell]] = cell
        cell = came_from[cell]

    return exploration_order, path_to_goal

# Get next cell based on direction
def get_next_cell(current, direction):
    row, col = current
    if direction == 'E':  # Move East
        return (row, col + 1)
    elif direction == 'W':  # Move West
        return (row, col - 1)
    elif direction == 'S':  # Move South
        return (row + 1, col)
    elif direction == 'N':  # Move North
        return (row - 1, col)

# Draw the maze and obstacles
def draw_maze():
    screen.fill((255, 255, 255))  # Clear screen with white background

    for row in range(m.rows):
        for col in range(m.cols):
            x = col * cell_width
            y = row * cell_height

            # Draw obstacles in red
            if (row, col) in obstacles:
                pygame.draw.rect(screen, (255, 0, 0), (x, y, cell_width, cell_height))
            else:
                # Draw the walls of the maze
                if not m.maze_map[(row + 1, col + 1)]['E']:  # No East wall
                    pygame.draw.line(screen, (0, 0, 0), (x + cell_width, y), (x + cell_width, y + cell_height), 2)
                if not m.maze_map[(row + 1, col + 1)]['W']:  # No West wall
                    pygame.draw.line(screen, (0, 0, 0), (x, y), (x, y + cell_height), 2)
                if not m.maze_map[(row + 1, col + 1)]['S']:  # No South wall
                    pygame.draw.line(screen, (0, 0, 0), (x, y + cell_height), (x + cell_width, y + cell_height), 2)
                if not m.maze_map[(row + 1, col + 1)]['N']:  # No North wall
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
    update_maze_with_obstacles()

# Update the maze with the new obstacles
def update_maze_with_obstacles():
    for row, col in obstacles:
        m.maze_map[(row + 1, col + 1)]['E'] = False
        m.maze_map[(row + 1, col + 1)]['W'] = False
        m.maze_map[(row + 1, col + 1)]['S'] = False
        m.maze_map[(row + 1, col + 1)]['N'] = False

# Reset the maze and obstacles
def reset_all():
    global obstacles
    obstacles = set()  # Clear obstacles
    load_maze_from_csv('D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/maze--2025-01-03--10-31-10.csv')  # Reload the maze
    update_maze_with_obstacles()  # Update maze walls after reset

# Reset obstacles only
def reset_obstacles():
    global obstacles
    obstacles = set()  # Remove obstacles from the set
    update_maze_with_obstacles()  # Update maze walls after reset

# Run the algorithm
def run_algorithm():
    start_position = (m.rows - 1, m.cols - 1)
    goal_position = (1, 1)

    exploration_order, path_to_goal = A_star_search(m, start_position, goal_position)

    # Visualize the exploration process using pyamaze agents
    agent_exploration = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
    agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

    m.tracePath({agent_exploration: exploration_order}, delay=0.1)
    m.tracePath({agent_trace: path_to_goal}, delay=0.1)

    # Run the visualization
    m.run()

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
                        run_algorithm()
                    elif window_height - 150 <= event.pos[1] <= window_height - 110:  # Reset Obstacles
                        reset_obstacles()
                    elif window_height - 200 <= event.pos[1] <= window_height - 160:  # Reset All
                        reset_all()

    draw_maze()  # Update display

# Close Pygame
pygame.quit()
