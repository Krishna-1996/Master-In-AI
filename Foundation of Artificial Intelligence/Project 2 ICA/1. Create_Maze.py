
from pyamaze import maze

my_Maze = maze(60, 150)
my_Maze.CreateMaze(loopPercent=10, pattern= 'v', saveMaze=True)

# Display the maze
my_Maze.run()