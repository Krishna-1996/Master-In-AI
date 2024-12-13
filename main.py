from pyamaze import maze
m = maze(30, 75)
m.CreateMaze(loopPercent=45,saveMaze=True)
# m.CreateMaze(saveMaze=True)
m.run()