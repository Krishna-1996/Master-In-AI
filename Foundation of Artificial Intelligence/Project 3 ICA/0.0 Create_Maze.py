

# %%
from pyamaze import maze

my_Maze = maze(50, 120)
my_Maze.CreateMaze(
    loopPercent=48, 
    saveMaze=True)

# Display the maze
my_Maze.run()