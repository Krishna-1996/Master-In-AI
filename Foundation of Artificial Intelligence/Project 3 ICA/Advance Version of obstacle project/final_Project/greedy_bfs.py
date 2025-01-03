import heapq

def run_algorithm(maze, start, goal):
    """Runs the Greedy BFS algorithm on the maze."""
    rows, cols = len(maze), len(maze[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    def in_bounds(x, y):
        return 0 <= x < rows and 0 <= y < cols

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {start: None}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if in_bounds(*neighbor) and maze[neighbor[0]][neighbor[1]] != 1 and neighbor not in came_from:
                heapq.heappush(open_set, (heuristic(neighbor, goal), neighbor))
                came_from[neighbor] = current

    return None  # No path found
