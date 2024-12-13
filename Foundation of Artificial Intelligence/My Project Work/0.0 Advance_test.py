import heapq
import pandas as pd
def parse_maze(file_path):
    """Parse the maze CSV into a graph representation."""
    # 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv'
    df = pd.read_csv(file_path, header=None)
    graph = {}

    for _, row in df.iterrows():
        if row[0] == "cell":
            continue

        cell = eval(row[0])  # Convert "(x, y)" to tuple (x, y)
        neighbors = []

        if row[1]:  # East
            neighbors.append((cell[0], cell[1] + 1))
        if row[2]:  # West
            neighbors.append((cell[0], cell[1] - 1))
        if row[3]:  # North
            neighbors.append((cell[0] - 1, cell[1]))
        if row[4]:  # South
            neighbors.append((cell[0] + 1, cell[1]))

        graph[cell] = neighbors

    return graph

def bfs(graph, start, goal):
    """Breadth-First Search."""
    queue = [(start, [start])]
    visited = set()

    while queue:
        current, path = queue.pop(0)
        if current == goal:
            return path
        if current in visited:
            continue

        visited.add(current)

        for neighbor in graph[current]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return None

def dfs(graph, start, goal):
    """Depth-First Search."""
    stack = [(start, [start])]
    visited = set()

    while stack:
        current, path = stack.pop()
        if current == goal:
            return path
        if current in visited:
            continue

        visited.add(current)

        for neighbor in graph[current]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None

def heuristic(a, b):
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(graph, start, goal):
    """A* Search."""
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue

        visited.add(current)

        for neighbor in graph[current]:
            if neighbor not in visited:
                g = cost + 1
                f = g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, g, neighbor, path + [neighbor]))

    return None

def greedy_bfs(graph, start, goal):
    """Greedy Best-First Search."""
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start, [start]))
    visited = set()

    while open_set:
        _, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue

        visited.add(current)

        for neighbor in graph[current]:
            if neighbor not in visited:
                heapq.heappush(open_set, (heuristic(neighbor, goal), neighbor, path + [neighbor]))

    return None

# Main execution
maze_graph = parse_maze('D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv')
start = (30, 50)
goal = (10, 10)  # Example goal; change as needed

print("BFS Path:", bfs(maze_graph, start, goal))
print("DFS Path:", dfs(maze_graph, start, goal))
print("A* Path:", astar(maze_graph, start, goal))
print("Greedy BFS Path:", greedy_bfs(maze_graph, start, goal))
