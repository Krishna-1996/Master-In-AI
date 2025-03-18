# A_Star.py
import heapq
from pyamaze import maze, COLOR

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_next_cell(current, direction):
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

def A_star_search(m, start=None, goal=None):
    if start is None:
        start = (m.rows-1, m.cols-1)  # Bottom-right cell
    if goal is None:
        goal = (0, 0)  # Top-left cell

    frontier = []
    heapq.heappush(frontier, (0 + heuristic(start, goal), start))
    visited = {}
    exploration_order = []
    explored = set([start])
    g_costs = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for direction in 'ESNW':
            if m.maze_map[current][direction] == 1:
                next_cell = get_next_cell(current, direction)
                new_g_cost = g_costs[current] + 1
                if next_cell not in explored or new_g_cost < g_costs.get(next_cell, float('inf')):
                    g_costs[next_cell] = new_g_cost
                    f_cost = new_g_cost + heuristic(next_cell, goal)
                    heapq.heappush(frontier, (f_cost, next_cell))
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
                    explored.add(next_cell)

    if goal not in visited:
        return [], {}, {}

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal
