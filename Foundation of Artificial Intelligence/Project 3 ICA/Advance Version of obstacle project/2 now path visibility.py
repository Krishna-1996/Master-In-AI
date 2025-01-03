import pygame
import heapq
import random
import sys
import tkinter as tk
from tkinter import ttk
import threading

# Constants
WIDTH, HEIGHT = 70, 30  # Grid dimensions
CELL_SIZE = 20  # Pixel size for each grid cell
START = (15, 1)  # Starting point
GOAL = (28, 67)  # Goal point

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
running = True

# Global Variables to store paths and results
paths = {
    'A*': [],
    'GreedyBFS': [],
    'BFS': [],
    'DFS': []
}
results = {
    'A*': {'path_length': 0, 'exploration_length': 0},
    'GreedyBFS': {'path_length': 0, 'exploration_length': 0},
    'BFS': {'path_length': 0, 'exploration_length': 0},
    'DFS': {'path_length': 0, 'exploration_length': 0},
}

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

# Run all algorithms and visualize the paths
def run_all_algorithms():
    global results

    # Run A*, GreedyBFS, BFS, DFS
    path_a_star, exploration_a_star = astar()
    paths['A*'] = path_a_star
    results['A*'] = {'path_length': len(path_a_star), 'exploration_length': exploration_a_star}

    path_greedy_bfs, exploration_greedy_bfs = greedy_bfs()
    paths['GreedyBFS'] = path_greedy_bfs
    results['GreedyBFS'] = {'path_length': len(path_greedy_bfs), 'exploration_length': exploration_greedy_bfs}

    path_bfs, exploration_bfs = bfs()
    paths['BFS'] = path_bfs
    results['BFS'] = {'path_length': len(path_bfs), 'exploration_length': exploration_bfs}

    path_dfs, exploration_dfs = dfs()
    paths['DFS'] = path_dfs
    results['DFS'] = {'path_length': len(path_dfs), 'exploration_length': exploration_dfs}

    # Draw paths for all algorithms
    draw_all_paths()

    # Update the results table in Tkinter window
    update_results_table()

# Draw all paths for each algorithm
def draw_all_paths():
    for algo, path in paths.items():
        if algo == 'A*':
            color = YELLOW
        elif algo == 'GreedyBFS':
            color = BLUE
        elif algo == 'BFS':
            color = GREEN
        elif algo == 'DFS':
            color = RED

        for node in path:
            pygame.draw.rect(screen, color, (node[1] * CELL_SIZE, node[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.update()

# Initialize Tkinter window for results
def init_results_window():
    global root, tree
    root = tk.Tk()
    root.title("Algorithm Results")

    # Create treeview (table) to display results
    tree = ttk.Treeview(root, columns=("Algorithm", "Path Length", "Exploration Length"), show="headings")
    tree.heading("Algorithm", text="Algorithm")
    tree.heading("Path Length", text="Path Length")
    tree.heading("Exploration Length", text="Exploration Length")
    tree.pack(fill=tk.BOTH, expand=True)

    # Set window size and make it non-resizable
    root.geometry("400x400")
    root.resizable(True, True)

# Update the results table in Tkinter window
def update_results_table():
    for algo in results:
        path_length = results[algo]['path_length']
        exploration_length = results[algo]['exploration_length']
        # Check if algorithm already exists, if so, update it
        existing_item = next((item for item in tree.get_children() if tree.item(item)["values"][0] == algo), None)
        if existing_item:
            tree.item(existing_item, values=(algo, path_length, exploration_length))
        else:
            tree.insert("", "end", values=(algo, path_length, exploration_length))

# Reset the maze
def reset_maze():
    global grid, paths, results
    grid = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]  # Clear the grid
    paths = {
        'A*': [],
        'GreedyBFS': [],
        'BFS': [],
        'DFS': []
    }
    results = {
        'A*': {'path_length': 0, 'exploration_length': 0},
        'GreedyBFS': {'path_length': 0, 'exploration_length': 0},
        'BFS': {'path_length': 0, 'exploration_length': 0},
        'DFS': {'path_length': 0, 'exploration_length': 0},
    }
    draw_grid()
    update_results_table()

# Main game loop
def game_loop():
    global running
    while running:
        handle_input()
        draw_grid()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Press 'r' to reset the maze
                    reset_maze()
                elif event.key == pygame.K_SPACE:  # Press 'space' to run all algorithms
                    run_all_algorithms()

        clock.tick(300)  # Control the frame rate

    pygame.quit()
    root.quit()

# Run the game loop in a separate thread along with Tkinter window
def run():
    threading.Thread(target=game_loop, daemon=True).start()
    root.mainloop()

# Initialize Tkinter window and start the program
init_results_window()
run()
import pygame
import heapq
import random
import sys
import tkinter as tk
from tkinter import ttk
import threading

# Constants
WIDTH, HEIGHT = 70, 30  # Grid dimensions
CELL_SIZE = 20  # Pixel size for each grid cell
START = (15, 1)  # Starting point
GOAL = (28, 67)  # Goal point

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
running = True

# Global Variables to store paths and results
paths = {
    'A*': [],
    'GreedyBFS': [],
    'BFS': [],
    'DFS': []
}
results = {
    'A*': {'path_length': 0, 'exploration_length': 0},
    'GreedyBFS': {'path_length': 0, 'exploration_length': 0},
    'BFS': {'path_length': 0, 'exploration_length': 0},
    'DFS': {'path_length': 0, 'exploration_length': 0},
}

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

# Run all algorithms and visualize the paths
def run_all_algorithms():
    global results

    # Run A*, GreedyBFS, BFS, DFS
    path_a_star, exploration_a_star = astar()
    paths['A*'] = path_a_star
    results['A*'] = {'path_length': len(path_a_star), 'exploration_length': exploration_a_star}

    path_greedy_bfs, exploration_greedy_bfs = greedy_bfs()
    paths['GreedyBFS'] = path_greedy_bfs
    results['GreedyBFS'] = {'path_length': len(path_greedy_bfs), 'exploration_length': exploration_greedy_bfs}

    path_bfs, exploration_bfs = bfs()
    paths['BFS'] = path_bfs
    results['BFS'] = {'path_length': len(path_bfs), 'exploration_length': exploration_bfs}

    path_dfs, exploration_dfs = dfs()
    paths['DFS'] = path_dfs
    results['DFS'] = {'path_length': len(path_dfs), 'exploration_length': exploration_dfs}

    # Draw paths for all algorithms
    draw_all_paths()

    # Update the results table in Tkinter window
    update_results_table()

# Draw all paths for each algorithm
def draw_all_paths():
    for algo, path in paths.items():
        if algo == 'A*':
            color = YELLOW
        elif algo == 'GreedyBFS':
            color = BLUE
        elif algo == 'BFS':
            color = GREEN
        elif algo == 'DFS':
            color = RED

        for node in path:
            pygame.draw.rect(screen, color, (node[1] * CELL_SIZE, node[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.update()

# Initialize Tkinter window for results
def init_results_window():
    global root, tree
    root = tk.Tk()
    root.title("Algorithm Results")

    # Create treeview (table) to display results
    tree = ttk.Treeview(root, columns=("Algorithm", "Path Length", "Exploration Length"), show="headings")
    tree.heading("Algorithm", text="Algorithm")
    tree.heading("Path Length", text="Path Length")
    tree.heading("Exploration Length", text="Exploration Length")
    tree.pack(fill=tk.BOTH, expand=True)

    # Set window size and make it non-resizable
    root.geometry("400x400")
    root.resizable(False, False)

# Update the results table in Tkinter window
def update_results_table():
    for algo in results:
        path_length = results[algo]['path_length']
        exploration_length = results[algo]['exploration_length']
        # Check if algorithm already exists, if so, update it
        existing_item = next((item for item in tree.get_children() if tree.item(item)["values"][0] == algo), None)
        if existing_item:
            tree.item(existing_item, values=(algo, path_length, exploration_length))
        else:
            tree.insert("", "end", values=(algo, path_length, exploration_length))

# Reset the maze
def reset_maze():
    global grid, paths, results
    grid = [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]  # Clear the grid
    paths = {
        'A*': [],
        'GreedyBFS': [],
        'BFS': [],
        'DFS': []
    }
    results = {
        'A*': {'path_length': 0, 'exploration_length': 0},
        'GreedyBFS': {'path_length': 0, 'exploration_length': 0},
        'BFS': {'path_length': 0, 'exploration_length': 0},
        'DFS': {'path_length': 0, 'exploration_length': 0},
    }
    draw_grid()
    update_results_table()

# Main game loop
def game_loop():
    global running
    while running:
        handle_input()
        draw_grid()  # This will keep the grid and obstacles updated

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Press 'r' to reset the maze
                    reset_maze()
                elif event.key == pygame.K_SPACE:  # Press 'space' to run all algorithms
                    run_all_algorithms()

        clock.tick(30)  # Control the frame rate

    pygame.quit()
    root.quit()

# Run the game loop in a separate thread along with Tkinter window
def run():
    threading.Thread(target=game_loop, daemon=True).start()
    root.mainloop()

# Initialize Tkinter window and start the program
init_results_window()
run()
