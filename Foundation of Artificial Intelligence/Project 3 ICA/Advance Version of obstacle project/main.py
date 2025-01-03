import pygame
import heapq
import random
import sys

# Constants
WIDTH, HEIGHT = 70, 30  # Grid dimensions
CELL_SIZE = 20  # Pixel size for each grid cell
START = (15, 1)  # Starting point
GOAL = (15, 67)  # Goal point

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Directions for movement (up, down, left, right)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE))
pygame.display.set_caption("Maze Pathfinding Algorithms")

# Initialize clock for framerate control
clock = pygame.time.Clock()

# Global Variables
grid = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]  # 0 represents free space, 1 represents obstacle
visited = [[False for _ in range(WIDTH)] for _ in range(HEIGHT)]  # To track visited cells
algorithm = 'A*'  # Default algorithm
running = True


# Heuristic function for A* and Greedy BFS
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# A* Algorithm
def astar():
    open_list = []
    closed_list = set()
    came_from = {}

    start_node = START
    goal_node = GOAL

    # Initialize open list
    heapq.heappush(open_list, (0 + heuristic(start_node, goal_node), 0, start_node))  # (f, g, node)

    g_costs = {start_node: 0}
    f_costs = {start_node: heuristic(start_node, goal_node)}

    while open_list:
        _, g, current = heapq.heappop(open_list)

        if current == goal_node:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1], len(closed_list)  # Path and exploration length

        closed_list.add(current)

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < HEIGHT and 0 <= neighbor[1] < WIDTH and grid[neighbor[0]][neighbor[1]] == 0:
                if neighbor in closed_list:
                    continue

                tentative_g = g + 1

                if neighbor not in g_costs or tentative_g < g_costs[neighbor]:
                    came_from[neighbor] = current
                    g_costs[neighbor] = tentative_g
                    f_costs[neighbor] = tentative_g + heuristic(neighbor, goal_node)
                    heapq.heappush(open_list, (f_costs[neighbor], tentative_g, neighbor))

    return [], len(closed_list)  # No path found


# Greedy BFS Algorithm
def greedy_bfs():
    open_list = []
    closed_list = set()
    came_from = {}

    start_node = START
    goal_node = GOAL

    # Initialize open list
    heapq.heappush(open_list, (heuristic(start_node, goal_node), start_node))

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal_node:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1], len(closed_list)  # Path and exploration length

        closed_list.add(current)

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < HEIGHT and 0 <= neighbor[1] < WIDTH and grid[neighbor[0]][neighbor[1]] == 0:
                if neighbor in closed_list:
                    continue
                came_from[neighbor] = current
                heapq.heappush(open_list, (heuristic(neighbor, goal_node), neighbor))

    return [], len(closed_list)  # No path found


# BFS Algorithm
def bfs():
    queue = [START]
    came_from = {START: None}
    closed_list = set([START])

    while queue:
        current = queue.pop(0)

        if current == GOAL:
            # Reconstruct the path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1], len(closed_list)  # Path and exploration length

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < HEIGHT and 0 <= neighbor[1] < WIDTH and grid[neighbor[0]][neighbor[1]] == 0:
                if neighbor not in closed_list:
                    queue.append(neighbor)
                    closed_list.add(neighbor)
                    came_from[neighbor] = current

    return [], len(closed_list)  # No path found


# DFS Algorithm
def dfs():
    stack = [START]
    came_from = {START: None}
    closed_list = set([START])

    while stack:
        current = stack.pop()

        if current == GOAL:
            # Reconstruct the path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1], len(closed_list)  # Path and exploration length

        for dx, dy in DIRECTIONS:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < HEIGHT and 0 <= neighbor[1] < WIDTH and grid[neighbor[0]][neighbor[1]] == 0:
                if neighbor not in closed_list:
                    stack.append(neighbor)
                    closed_list.add(neighbor)
                    came_from[neighbor] = current

    return [], len(closed_list)  # No path found


# Draw the maze on the screen
def draw_grid():
    screen.fill(WHITE)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            color = WHITE
            if grid[y][x] == 1:
                color = BLACK
            elif (y, x) == START:
                color = GREEN
            elif (y, x) == GOAL:
                color = RED
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    
    pygame.display.update()


# Handle user input for obstacle placement/removal
def handle_input():
    mouse_x, mouse_y = pygame.mouse.get_pos()
    grid_x, grid_y = mouse_x // CELL_SIZE, mouse_y // CELL_SIZE

    if pygame.mouse.get_pressed()[0]:  # Left click to place obstacle
        grid[grid_y][grid_x] = 1
    elif pygame.mouse.get_pressed()[2]:  # Right click to remove obstacle
        grid[grid_y][grid_x] = 0


# Run the selected algorithm and visualize the path
def run_algorithm():
    global algorithm
    if algorithm == 'A*':
        path, exploration_length = astar()
    elif algorithm == 'GreedyBFS':
        path, exploration_length = greedy_bfs()
    elif algorithm == 'BFS':
        path, exploration_length = bfs()
    elif algorithm == 'DFS':
        path, exploration_length = dfs()

    # Visualize the path
    for node in path:
        pygame.draw.rect(screen, YELLOW, (node[1] * CELL_SIZE, node[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
    pygame.display.update()

    print(f"Algorithm: {algorithm}")
    print(f"Path Length: {len(path)}")
    print(f"Exploration Length: {exploration_length}")


# Reset the maze
def reset_maze():
    global grid
    grid = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    draw_grid()


# Main game loop
def game_loop():
    global running, algorithm
    while running:
        handle_input()
        draw_grid()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Press 'r' to reset the maze
                    reset_maze()
                elif event.key == pygame.K_SPACE:  # Press 'space' to run the selected algorithm
                    run_algorithm()
                elif event.key == pygame.K_1:  # Press '1' to select A* algorithm
                    algorithm = 'A*'
                elif event.key == pygame.K_2:  # Press '2' to select Greedy BFS algorithm
                    algorithm = 'GreedyBFS'
                elif event.key == pygame.K_3:  # Press '3' to select BFS algorithm
                    algorithm = 'BFS'
                elif event.key == pygame.K_4:  # Press '4' to select DFS algorithm
                    algorithm = 'DFS'

        clock.tick(30)  # Control the frame rate

    pygame.quit()


# Run the game loop
game_loop()
