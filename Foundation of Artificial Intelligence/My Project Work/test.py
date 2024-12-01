from pyamaze import maze, agent, COLOR, textLabel
from collections import deque
import heapq

def heuristic(a, b):
    # Manhattan distance heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

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

# Main code remains the same as in the original
# Main function to execute the maze creation and BFS search
if __name__ == '__main__':
    # Create a 30x50 maze and load it from a CSV file
    m = maze(30, 50)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv')

    # Set your custom goal (within maze limits)
    goal_position = (29, 1)  # Example goal, you can change this to any valid coordinate

    # Perform BFS search on the maze and get the exploration order and paths
    exploration_order, visited_cells, path_to_goal = A_star_search(m, goal=goal_position)

    # Create agents to visualize the BFS search process
    agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red)  # Visualize BFS search order
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full BFS path

    # Create the goal agent at the custom goal position
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

    # Visualize the agents' movements along their respective paths
    m.tracePath({agent_bfs: exploration_order}, delay=5)  # BFS search order path
    m.tracePath({agent_trace: path_to_goal}, delay=100)  # Trace the path from goal to start (final agent path)
    m.tracePath({agent_goal: visited_cells}, delay=100)  # Trace the BFS path to the goal

    # Display the length of the BFS path and search steps
    textLabel(m, 'BFS Path Length', len(path_to_goal) + 1)  # Length of the path from goal to start
    textLabel(m, 'BFS Search Length', len(exploration_order))  # Total number of explored cells

    # Run the maze visualization
    m.run()
