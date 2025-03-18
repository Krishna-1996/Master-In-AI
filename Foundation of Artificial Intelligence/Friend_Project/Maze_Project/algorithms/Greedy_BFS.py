# Greedy_BFS.py
import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def greedy_bfs_search(m, start=None, goal=None):
    if start is None:
        start = (m.rows-1, m.cols-1)  # Bottom-right cell
    if goal is None:
        goal = (0, 0)  # Top-left cell

    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))
    visited = {}
    exploration_order = []

    while frontier:
        _, current = heapq.heappop(frontier)
        exploration_order.append(current)
        if current == goal:
            break

        for direction in 'ESNW':
            if m.maze_map[current][direction] == 1:
                next_cell = get_next_cell(current, direction)
                if next_cell not in visited:
                    visited[next_cell] = current
                    heapq.heappush(frontier, (heuristic(next_cell, goal), next_cell))

    if goal not in visited:
        return [], {}, {}

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal
