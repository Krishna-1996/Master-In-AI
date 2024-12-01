from pyamaze import maze, agent, COLOR, textLabel
from collections import deque

def DFS_search(maze_obj, start=None, goal=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)

    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)

    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")

    # Stack for DFS
    frontier = [start]
    visited = {}
    exploration_order = []
    explored = set([start])

    while frontier:
        current = frontier.pop()

        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                if next_cell not in explored:
                    frontier.append(next_cell)
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

# Main function for DFS search
if __name__ == '__main__':
    m = maze(30, 50)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv')

    goal_position = (29, 1)  # Example goal, change to any valid coordinate

    exploration_order, visited_cells, path_to_goal = DFS_search(m, goal=goal_position)

    agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.red)
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

    m.tracePath({agent_dfs: exploration_order}, delay=5)
    m.tracePath({agent_trace: path_to_goal}, delay=100)
    m.tracePath({agent_goal: visited_cells}, delay=100)

    textLabel(m, 'DFS Path Length', len(path_to_goal) + 1)
    textLabel(m, 'DFS Search Length', len(exploration_order))

    m.run()
