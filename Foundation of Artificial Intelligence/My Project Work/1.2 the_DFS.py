from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

def dfs_search(maze_obj, start=None, goal = (29, 1)):
    # Default start position: Bottom-right corner
    if start is None:
        start = (maze_obj.rows - 1, maze_obj.cols - 1)  # Correct for bottom-right corner

   

    # Ensure goal is within the maze bounds
    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")

    # Stack for DFS
    stack = [start]
    visited = {}
    exploration_order = []
    explored = set([start])

    while stack:
        current = stack.pop()

        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                if next_cell not in explored:
                    stack.append(next_cell)
                    explored.add(next_cell)
                    visited[next_cell] = current
                    exploration_order.append(next_cell)

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

def get_next_cell(current, direction):
    row, col = current
    if direction == 'E':
        return (row, col + 1)
    elif direction == 'W':
        return (row, col - 1)
    elif direction == 'S':
        return (row + 1, col)
    elif direction == 'N':
        return (row - 1, col)

# Main function to execute the maze creation and DFS search
if __name__ == '__main__':
    # Create a 30x50 maze and load it from a CSV file
    m = maze(30, 50)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv')

    goal_position = (29, 1)  # Set custom goal position

    # Perform DFS search on the maze and get the exploration order and paths
    exploration_order, visited_cells, path_to_goal = dfs_search(m, goal=goal_position)

    # Create agents to visualize the DFS search process
    agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.green)  # Visualize DFS search order
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)  # Full DFS path
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.blue, shape='square', filled=True)  # Goal agent

    # Visualize the agents' movements along their respective paths
    m.tracePath({agent_dfs: exploration_order}, delay=5)  # DFS search order path
    m.tracePath({agent_trace: path_to_goal}, delay=1)  # Trace the DFS path to the goal
    m.tracePath({agent_goal: visited_cells}, delay=1)  # Trace the path from goal to start

    # Display the length of the DFS path and search steps
    textLabel(m, 'DFS Path Length', len(path_to_goal) + 1)  # Length of the path from goal to start
    textLabel(m, 'DFS Search Length', len(exploration_order))  # Total number of explored cells

    # Run the maze visualization
    m.run()
