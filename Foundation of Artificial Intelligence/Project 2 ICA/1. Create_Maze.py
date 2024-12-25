
from pyamaze import maze

my_Maze = maze(500, 500)
my_Maze.CreateMaze(loopPercent= 10)

# Display the maze
my_Maze.run()