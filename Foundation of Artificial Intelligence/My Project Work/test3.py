import matplotlib.pyplot as plt
import numpy as np
from pyamaze import maze, agent, COLOR
from collections import deque

def visualize_maze_with_labels(m):
    """
    Visualizes the maze with row and column labels using matplotlib.
    """
    maze_array = np.zeros((m.rows, m.cols))

    # Fill in the maze structure from the pyamaze object
    for r in range(1, m.rows + 1):
        for c in range(1, m.cols + 1):
            cell = m.maze_map[(r, c)]
            maze_array[r - 1, c - 1] = sum(cell.values())  # Sum of wall openings (0: closed, >0: open)

    # Plot the maze grid
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(maze_array, cmap='Greys', origin='upper')

    # Add row and column labels
    for i in range(m.rows):
        for j in range(m.cols):
            ax.text(j, i, f"{i + 1},{j + 1}", ha='center', va='center', color='red', fontsize=8)

    # Customize the grid and labels
    ax.set_xticks(np.arange(-0.5, m.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m.rows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    ax.set_xticks([])  # Remove x-axis major ticks
    ax.set_yticks([])  # Remove y-axis major ticks
    ax.set_title("Maze Visualization with Row and Column Labels", fontsize=14)
    plt.show()

def BFS_search(maze_obj, start=None, goal=None):
    """
    Breadth-First Search algorithm to find a path from start to goal in the maze.
    """
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    if goal is None:
        goal = (1, 1)

    frontier = deque([start])
    visited = {}
    exploration_order = []
    explored = set([start])

    while frontier:
        current = frontier.popleft()
        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                if next_cell not in explored:
                    frontier.append(next_cell)
                    explored.add(next_cell)
                    visited[next_cell] = current
                    exploration_order.append(next_cell)

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

def get_next_cell(current, direction):
    """
    Returns the coordinates of the neighboring cell based on the direction.
    Directions are 'E' (East), 'W' (West), 'S' (South), 'N' (North).
    """
    row, col = current
    if direction == 'E':  # Move East
        return (row, col + 1)
    elif direction == 'W':  # Move West
        return (row, col - 1)
    elif direction == 'S':  # Move South
        return (row + 1, col)
    elif direction == 'N':  # Move North
        return (row - 1, col)

if __name__ == '__main__':
    # Create a maze
    m = maze(10, 10)  # Example maze size
    m.CreateMaze()

    # Set custom goal position
    goal_position = (1, 1)  # Top-left corner as the goal

    # Perform BFS search
    exploration_order, visited_cells, path_to_goal = BFS_search(m, goal=goal_position)

    # Visualize the maze with BFS results
    visualize_maze_with_labels(m)

    # Visualize the BFS in pyamaze
    agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red)  # BFS path
    m.tracePath({agent_bfs: path_to_goal}, delay=100)
    m.run()
