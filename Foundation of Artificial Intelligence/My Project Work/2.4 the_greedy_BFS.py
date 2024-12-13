from pyamaze import maze, agent, COLOR, textLabel
import heapq

def get_next_cell(current, direction):
    """Calculate the next cell based on the current cell and direction."""
    x, y = current
    if direction == 'E':  # Move east
        return (x, y + 1)
    elif direction == 'W':  # Move west
        return (x, y - 1)
    elif direction == 'N':  # Move north
        return (x - 1, y)
    elif direction == 'S':  # Move south
        return (x + 1, y)
    return current  # Return the current cell if direction is invalid

def greedy_heuristic(a, b):
    # Manhattan distance heuristic for Greedy BFS
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def greedy_BFS_search(maze_obj, start=None, goal=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")

    # Min-heap priority queue
    frontier = []
    heapq.heappush(frontier, (greedy_heuristic(start, goal), start))  # (h-cost, position)
    visited = {}
    exploration_order = []
    explored = set([start])

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                if next_cell not in explored:
                    heapq.heappush(frontier, (greedy_heuristic(next_cell, goal), next_cell))
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
                    explored.add(next_cell)

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

# Main function for Greedy BFS search
if __name__ == '__main__':
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')

    goal_position = (1 ,1)  # Example goal, change to any valid coordinate

    exploration_order, visited_cells, path_to_goal = greedy_BFS_search(m, goal=goal_position)

    agent_greedy = agent(m, footprints=True, shape='square', color=COLOR.red)
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

    m.tracePath({agent_greedy: exploration_order}, delay=1)
    m.tracePath({agent_trace: path_to_goal}, delay=1)
    m.tracePath({agent_goal: visited_cells}, delay=1)

    textLabel(m, 'Goal Position', str(goal_position))
    textLabel(m, 'Greedy BFS Path Length', len(path_to_goal) + 1)
    textLabel(m, 'Greedy BFS Search Length', len(exploration_order))

    m.run()
