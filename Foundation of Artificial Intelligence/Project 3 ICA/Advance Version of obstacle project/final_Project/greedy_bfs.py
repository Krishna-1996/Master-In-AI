from pyamaze import maze, agent, COLOR, textLabel
import heapq
import csv

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def load_maze_from_csv(file_path, maze_obj):
    with open(file_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            coords = eval(row[0])
            E, W, N, S = map(int, row[1:])
            maze_obj.maze_map[coords] = {"E": E, "W": W, "N": N, "S": S}

def add_obstacles(maze_obj, obstacle_percentage=20):
    total_cells = maze_obj.rows * maze_obj.cols
    num_obstacles = int(total_cells * (obstacle_percentage / 100))
    valid_cells = [(row, col) for row in range(maze_obj.rows) for col in range(maze_obj.cols)]
    blocked_cells = random.sample(valid_cells, num_obstacles)
    obstacle_locations = []
    
    for (row, col) in blocked_cells:
        if (row, col) in maze_obj.maze_map:
            if random.choice([True, False]):
                maze_obj.maze_map[(row, col)]["E"] = 0
                maze_obj.maze_map[(row, col)]["W"] = 0
            if random.choice([True, False]):
                maze_obj.maze_map[(row, col)]["N"] = 0
                maze_obj.maze_map[(row, col)]["S"] = 0
            obstacle_locations.append((row, col))
    return obstacle_locations

def greedy_bfs(maze_obj, start=None, goal=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)
    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")
    
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))
    visited = {}
    exploration_order = [start]
    explored = set([start])
    
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction] == 1:
                next_cell = get_next_cell(current, direction)
                if next_cell not in explored:
                    heapq.heappush(frontier, (heuristic(next_cell, goal), next_cell))
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
    m.CreateMaze(loadMaze='maze.csv')
    goal_position = (1, 1)
    exploration_order, visited_cells, path_to_goal = greedy_bfs(m, goal=goal_position)
    
    if path_to_goal:
        agent_greedy = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
        agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)
        agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)
        m.tracePath({agent_greedy: exploration_order}, delay=1)
        m.tracePath({agent_trace: path_to_goal}, delay=1)
        m.tracePath({agent_goal: visited_cells}, delay=1)
        textLabel(m, 'Goal Position', str(goal_position))
        textLabel(m, 'Greedy BFS Path Length', len(path_to_goal) + 1)
        textLabel(m, 'Greedy BFS Search Length', len(exploration_order))
    else:
        print("No path found to the goal!")
    m.run()
