from pyamaze import maze, agent, COLOR, textLabel
from queue import PriorityQueue

# A* Algorithm Implementation
def A_star_search(maze_obj, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan Distance

    open_set = PriorityQueue()
    open_set.put((0, start))  # (f_score, position)
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    came_from = {}
    exploration_order = []

    while not open_set.empty():
        _, current = open_set.get()

        if current == goal:
            break

        exploration_order.append(current)
        
        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                tentative_g_score = g_score[current] + 1
                
                if next_cell not in g_score or tentative_g_score < g_score[next_cell]:
                    came_from[next_cell] = current
                    g_score[next_cell] = tentative_g_score
                    f_score[next_cell] = g_score[next_cell] + heuristic(next_cell, goal)
                    open_set.put((f_score[next_cell], next_cell))

    path_to_goal = {}
    cell = goal
    while cell != start:
        path_to_goal[came_from[cell]] = cell
        cell = came_from[cell]

    return exploration_order, path_to_goal

# Function to get the next cell based on direction
def get_next_cell(current, direction):
    row, col = current
    if direction == 'E':  # Move East
        return (row, col + 1)
    elif direction == 'W':  # Move West
        return (row, col - 1)
    elif direction == 'S':  # Move South
        return (row + 1, col)
    elif direction == 'N':  # Move North
        return (row - 1, col)

# Create your maze
m = maze(50, 120)
m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/maze--2025-01-03--10-31-10.csv')
# D:\Masters Projects\Master-In-AI\Foundation of Artificial Intelligence\Project 3 ICA

# The obstacles will be added here. Each obstacle is a tuple (row, col)
obstacles = set()

# You can create a function that handles the maze clicking to place obstacles
def place_obstacle(event):
    row, col = event.cell
    if (row, col) in obstacles:
        obstacles.remove((row, col))  # Remove the obstacle if clicked again
    else:
        obstacles.add((row, col))  # Add an obstacle at this position
    update_maze_with_obstacles()

# Update the maze with obstacles after placement
def update_maze_with_obstacles():
    for row, col in obstacles:
        m.maze_map[(row, col)]['E'] = False
        m.maze_map[(row, col)]['W'] = False
        m.maze_map[(row, col)]['S'] = False
        m.maze_map[(row, col)]['N'] = False

# Main function to run the A* algorithm and visualize the maze
def run_experiment():
    start_position = (m.rows-1, m.cols-1)
    goal_position = (1, 1)

    # Run A* algorithm
    exploration_order, path_to_goal = A_star_search(m, start_position, goal_position)

    # Visualize the exploration process
    agent_exploration = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
    agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)
    agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

    m.tracePath({agent_exploration: exploration_order}, delay=0.1)
    m.tracePath({agent_trace: path_to_goal}, delay=0.1)

    # Run the visualization
    m.run()

# Attach the click event to place obstacles
m.setEventCallback('click', place_obstacle)

# Run the experiment
run_experiment()
