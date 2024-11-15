from pyamaze import maze, agent

# Define the size of the maze.
maze_width = 25
maze_height = 25

# Create the maze object.
# The 'CreateMaze' function generates a random maze of the specified size.
my_maze = maze(maze_width, maze_height)
my_maze.CreateMaze(loopPercent=75, saveMaze=True)

# Define the agent that will navigate the maze.
# The 'footprints' parameter is set to True, which will leave a trail behind the agent.
# The 'shape' and 'color' parameters define the appearance of the agent.
my_agent = agent(my_maze, footprints=True, filled=True, shape='arrow', color='red')

my_maze.tracePath({my_agent:my_maze.path},delay=100)

# Start the simulation.
my_maze.run()
