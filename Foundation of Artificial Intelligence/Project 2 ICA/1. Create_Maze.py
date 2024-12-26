
from pyamaze import maze

my_Maze = maze(70, 180)
my_Maze.CreateMaze(loopPercent=35, saveMaze=True)

# Display the maze
my_Maze.run()