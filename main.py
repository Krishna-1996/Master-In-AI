from pyamaze import maze
m = maze(50, 120)
m.CreateMaze(loopPercent=30,saveMaze=True)
# m.CreateMaze(saveMaze=True)
m.run()