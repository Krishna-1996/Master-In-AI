from pyamaze import maze, agent, COLOR, textLabel

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

def DFS_search(maze_obj, start=None, goal=None):
    if start is None:
        start = (0, 0)  # Set a default valid start point

    if goal is None:
        goal = (maze_obj.rows // 2, maze_obj.cols // 2)  # Default goal in the center

    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")
    
    # Stack for DFS (Last-In-First-Out)
    frontier = [start]  # DFS uses a stack, so it's a list
    visited = {}
    exploration_order = []
    explored = set([start])

    while frontier:
        current = frontier.pop()  # Pop the last element (LIFO)

        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                
                if next_cell not in explored:
                    explored.add(next_cell)
                    frontier.append(next_cell)
                    visited[next_cell] = current
                    exploration_order.append(next_cell)

    # Reconstruct path to goal
    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[visited[cell]] = cell
        cell = visited[cell]

    return exploration_order, visited, path_to_goal

# Main function for DFS search
if __name__ == '__main__':
    m = maze(50, 120)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze_update2.csv')

    goal_position = (2, 119)  # Example goal, change to any valid coordinate

    exploration_order, visited_cells, path_to_goal = DFS_search(m, goal=goal_position)

    agent_dfs = agent(m, footprints=True, shape='square', color=COLOR.red)
    agent_trace = agent(m, footprints=True, shape='star', color=COLOR.yellow, filled=False)

    m.tracePath({agent_dfs: exploration_order}, delay=1)
    m.tracePath({agent_trace: path_to_goal}, delay=1)

    textLabel(m, 'Goal Position', str(goal_position))
    textLabel(m, 'DFS Path Length', len(path_to_goal) + 1)
    textLabel(m, 'DFS Search Length', len(exploration_order))

    m.run()
