from pyamaze import maze, agent, COLOR, textLabel
import csv
import random
from collections import deque

def get_next_cell(current, direction):
    """Calculate the next cell based on the current cell and direction."""
    x, y = current
    if direction == 'E':
        return (x, y + 1)
    elif direction == 'W':
        return (x, y - 1)
    elif direction == 'N':
        return (x - 1, y)
    elif direction == 'S':
        return (x + 1, y)
    return current

def load_maze_from_csv(file_path, maze_obj):
    """Load maze from CSV."""
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            coords = eval(row[0])  # Converts string to tuple
            E, W, N, S = map(int, row[1:])
            maze_obj.maze_map[coords] = {"E": E, "W": W, "N": N, "S": S}

def bfs_search(maze_obj, start=None, goal=None):
    """Breadth-First Search algorithm."""
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

    frontier = deque([start])  # Queue for BFS
    visited = {}
    exploration_order = []
    explored = set([start])

    while frontier:
        current = frontier.popleft()

        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction] == 1:
                next_cell = get_next_cell(current, direction)
                if next_cell not in explored:
                    frontier.append(next_cell)
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
                    explored.add(next_cell)

    if goal not in visited:
        print("Goal is unreachable!")
        return [], {}, {}

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

if __name__ == '__main__':
    m = maze(50, 100)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA//maze_with_obstacles.csv')
    goal_position = (1, 1)
    exploration_order, visited_cells, path_to_goal = bfs_search(m, goal=goal_position)

    if path_to_goal:
        agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
        agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)
        agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)
        m.tracePath({agent_bfs: exploration_order}, delay=1)
        m.tracePath({agent_trace: path_to_goal}, delay=1)
        m.tracePath({agent_goal: visited_cells}, delay=1)
        textLabel(m, 'Goal Position', str(goal_position))
        textLabel(m, 'BFS Path Length', len(path_to_goal) + 1)
        textLabel(m, 'BFS Search Length', len(exploration_order))
    else:
        print("No path found to the goal!")
    m.run()
