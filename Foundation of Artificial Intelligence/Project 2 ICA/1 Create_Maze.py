
from pyamaze import maze

my_Maze = maze(20, 80)
my_Maze.CreateMaze(loopPercent=35, saveMaze=True)

# Display the maze
my_Maze.run()