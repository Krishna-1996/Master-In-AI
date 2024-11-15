# Import necessary modules and functions
from dfsmazetest import DFS  # DFS algorithm implementation
from bfsmazetest import BFS  # BFS algorithm implementation
from pyamaze import maze, agent, COLOR, textLabel  # Pyamaze library for creating and visualizing mazes
from timeit import timeit  # Function for measuring execution time

# Create a maze instance
m = maze()

# Load the maze layout from a CSV file
m.CreateMaze(loadMaze='D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/ICA Pyamaze Example/mazetest.csv')

# Run the DFS algorithm on the maze and get the paths
searchPath, dfsPath, fwdDFSPath = DFS(m)

# Run the BFS algorithm on the maze and get the paths
bSearch, bfsPath, fwdBFSPath = BFS(m)

# Display the lengths of the paths found by DFS and BFS on the maze
textLabel(m, 'DFS Path Length', len(fwdDFSPath)+1)
textLabel(m, 'BFS Path Length', len(fwdBFSPath)+1)
textLabel(m, 'DFS Search Length', len(searchPath)+1)
textLabel(m, 'BFS Search Length', len(bSearch)+1)

# Create two agents for visualizing the paths
a = agent(m, footprints=True, color=COLOR.cyan, filled=True)  # Agent for BFS
b = agent(m, footprints=True, color=COLOR.yellow)  # Agent for DFS

# Have the agents trace the paths found by BFS and DFS
m.tracePath({b: fwdDFSPath}, delay=100)
m.tracePath({a: fwdBFSPath}, delay=100)


# Measure the execution time of DFS and BFS
t1 = timeit('DFS(m)', number=1000, globals=globals())  # Time for DFS
t2 = timeit('BFS(m)', number=1000, globals=globals())  # Time for BFS

# Display the execution times on the maze
textLabel(m, 'DFS Time', t1)
textLabel(m, 'BFS Time', t2)

# Start the visualization
m.run()
