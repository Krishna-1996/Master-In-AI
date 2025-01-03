
from pyamaze import maze

my_Maze = maze(20, 50)
my_Maze.CreateMaze(loopPercent=38, saveMaze=True)

# Display the maze
my_Maze.run()