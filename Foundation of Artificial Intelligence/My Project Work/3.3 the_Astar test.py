from pyamaze import maze, agent, COLOR, textLabel
import heapq

# Manhattan distance heuristic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Function to calculate the next cell based on direction
def get_next_cell(current, direction):
    x, y = current
    if direction == 'E':  # Move east
        return (x, y + 1)
    elif direction == 'W':  # Move west
        return (x, y - 1)
    elif direction == 'N':  # Move north
        return (x - 1, y)
    elif direction == 'S':  # Move south
        return (x + 1, y)
    return current

# A* search algorithm
def A_star_search(maze_obj, start=None, goal=None):
    if start is None:
        start = (maze_obj.rows, maze_obj.cols)
    if goal is None:
        goal = (2, 45)  # Default goal position

    if not (0 <= goal[0] < maze_obj.rows and 0 <= goal[1] < maze_obj.cols):
        raise ValueError(f"Invalid goal position: {goal}. It must be within the bounds of the maze.")
    
    # Min-heap priority queue for A* algorithm
    frontier = []
    heapq.heappush(frontier, (0 + heuristic(start, goal), start))  # (f-cost, position)
    visited = {}  # Tracks the path
    g_costs = {start: 0}  # Stores the g-cost for each position
    exploration_order = []  # Tracks the exploration order of cells

    while frontier:
        _, current = heapq.heappop(frontier)  # Get the cell with the lowest f-cost

        # If we reach the goal, stop the search
        if current == goal:
            break

        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:  # Check if move in direction is valid
                next_cell = get_next_cell(current, direction)
                new_g_cost = g_costs[current] + 1  # Uniform cost of 1 for each step

                # If the next cell is either unvisited or offers a better path
                if next_cell not in g_costs or new_g_cost < g_costs[next_cell]:
                    g_costs[next_cell] = new_g_cost
                    f_cost = new_g_cost + heuristic(next_cell, goal)
                    heapq.heappush(frontier, (f_cost, next_cell))
                    visited[next_cell] = current
                    exploration_order.append(next_cell)

    # Reconstruct path from goal to start (if the goal was found)
    path_to_goal = []
    if goal in visited:  # Check if the goal has been reached
        cell = goal
        while cell != start:
            path_to_goal.append(cell)
            cell = visited[cell]
        path_to_goal.append(start)
        path_to_goal.reverse()  # Reverse to get the path from start to goal
    else:
        print("Goal not reached. Path reconstruction failed.")
        path_to_goal = []

    return exploration_order, path_to_goal

# Main function to run the A* search and visualization
if __name__ == '__main__':
    # Create a maze instance and load the maze from a CSV file
    m = maze(30, 50)
    m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/My Project Work/maze--2024-11-30--21-36-21.csv')

    # Goal position (changeable within the code)
    goal_position = (2, 45)  # Example goal

    # Perform A* search
    start_position = (30, 50)  # Starting point: bottom-right corner
    exploration_order, path_to_goal = A_star_search(m, start=start_position, goal=goal_position)

    # Create the agent and path visualization
    agent_astar = agent(m, footprints=True, shape='square', color=COLOR.red)
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

    # Trace the path found by A* algorithm
    if path_to_goal:  # Only trace the path if it's valid
        m.tracePath({agent_astar: path_to_goal}, delay=5)

    # Display text labels on the maze
    textLabel(m, 'Goal Position', str(goal_position))
    textLabel(m, 'A* Path Length', len(path_to_goal))
    textLabel(m, 'A* Exploration Length', len(exploration_order))

    # Run the maze
    m.run()
