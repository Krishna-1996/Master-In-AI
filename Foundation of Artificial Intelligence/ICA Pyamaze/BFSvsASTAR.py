# Import the necessary modules and functions
from bfsmazetest import BFS  # BFS algorithm implementation
from astarmazetest import aStar  # A* algorithm implementation
from pyamaze import maze, agent, COLOR, textLabel  # Pyamaze library for creating and visualizing mazes
from timeit import timeit  # Function for measuring execution time

# Create a maze instance
m = maze()

# Load the maze layout from a CSV file
m.CreateMaze(loadMaze='mazetest.csv')

# Run the BFS algorithm on the maze and get the paths
bSearch, bfsPath, fwdBFSPath = BFS(m)

# Display the lengths of the paths found by BFS on the maze
textLabel(m, 'BFS Path Length', len(fwdBFSPath)+1)
textLabel(m, 'BFS Search Length', len(bSearch)+1)

# Measure the execution time of BFS
t1 = timeit('BFS(m)', number=1000, globals=globals())  # Time for BFS

# Display the execution time on the maze
textLabel(m, 'BFS Time', t1)

# Create agents for visualizing the paths found by BFS
a = agent(m, footprints=True, color=COLOR.blue, filled=True)
b = agent(m,1,1, footprints=True, color=COLOR.yellow, filled=True, goal=(m.rows, m.cols))
c = agent(m, footprints=True, color=COLOR.red)

# Have the agents trace the paths found by BFS
m.tracePath({a: bSearch}, delay=50)
m.tracePath({b: bfsPath}, delay=100)
m.tracePath({c: fwdBFSPath}, delay=100)

# Run the A* algorithm on the maze and get the paths
aSearch, aStarPath, fwdAStarPath = aStar(m)

# Display the lengths of the paths found by A* on the maze
textLabel(m, 'A* Path Length', len(fwdAStarPath)+1)
textLabel(m, 'A* Search Length', len(aSearch)+1)

# Measure the execution time of A*
t2 = timeit('aStar(m)', number=1000, globals=globals())  # Time for A*

# Display the execution time on the maze
textLabel(m, 'A* Time', t2)

# Create agents for visualizing the paths found by A*
a = agent(m, footprints=True, color=COLOR.green, filled=True)
b = agent(m,1,1, footprints=True, color=COLOR.yellow, filled=True, goal=(m.rows, m.cols))
c = agent(m, footprints=True, color=COLOR.red)

# Have the agents trace the paths found by A*
m.tracePath({a: aSearch}, delay=50)
m.tracePath({b: aStarPath}, delay=100)
m.tracePath({c: fwdAStarPath}, delay=100)

# Start the visualization
m.run()
