from pyamaze import maze, agent, COLOR, textLabel
import heapq

def heuristic(a, b):
    # Manhattan distance heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

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

def A_star_search(maze_obj, start=None, goal=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")

    # Min-heap priority queue
    frontier = []
    heapq.heappush(frontier, (0 + heuristic(start, goal), start))  # (f-cost, position)
    visited = {}
    exploration_order = []
    explored = set([start])
    g_costs = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                new_g_cost = g_costs[current] + 1  # +1 for each move (uniform cost)
                
                if next_cell not in explored or new_g_cost < g_costs.get(next_cell, float('inf')):
                    g_costs[next_cell] = new_g_cost
                    f_cost = new_g_cost + heuristic(next_cell, goal)
                    heapq.heappush(frontier, (f_cost, next_cell))
                    visited[next_cell] = current
                    exploration_order.append(next_cell)
                    explored.add(next_cell)

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

# Main function for A* search
if __name__ == '__main__':
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')

    goal_position = (1, 1)  # Example goal, change to any valid coordinate

    exploration_order, visited_cells, path_to_goal = A_star_search(m, goal=goal_position)

    # Create agents to visualize the BFS search process
    agent_A_star = agent(m, footprints=True, shape='square', 
                      color=COLOR.red)  # Visualize BFS search order
    agent_trace = agent(m, footprints=True, shape='star', 
                        color=COLOR.yellow, filled=False)  # Full BFS path
    agent_goal = agent(m, 1, 1, footprints=True, color=COLOR.blue, 
                       shape='square', filled=True, goal=(m.rows, m.cols))  # Goal agent

    # Trace the agents' paths through the maze
    m.tracePath({agent_A_star: exploration_order}, delay=1)  # Trace A* search order
    m.tracePath({agent_goal: visited_cells}, delay=1)  # Trace the path found by A*
    m.tracePath({agent_trace: path_to_goal}, delay=1)  # Trace the path from start to goal (final path)

    # Display the lengths of the Greedy BFS search and final paths as labels
    textLabel(m, 'Goal Position', str(goal_position))
    textLabel(m, 'Greedy A* Path Length', len(path_to_goal))  # Length of the path from start to goal
    textLabel(m, 'Greedy A* Search Length', len(exploration_order))  # Total number of cells explored

    m.run()
