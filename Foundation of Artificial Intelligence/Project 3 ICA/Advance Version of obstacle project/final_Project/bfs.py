from collections import deque

def run_algorithm(maze, start, goal):
    """Runs the BFS algorithm on the maze."""
    rows, cols = len(maze), len(maze[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque([start])
    came_from = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and
                    maze[neighbor[0]][neighbor[1]] != 1 and neighbor not in came_from):
                queue.append(neighbor)
                came_from[neighbor] = current

    return None  # No path found
