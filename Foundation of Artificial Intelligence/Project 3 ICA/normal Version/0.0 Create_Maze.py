
from pyamaze import maze

my_Maze = maze(50, 100)
my_Maze.CreateMaze(loopPercent=68, saveMaze=True)

# Display the maze
my_Maze.run()