import maze_loader
from a_star import run_algorithm as a_star
from bfs import run_algorithm as bfs
from greedy_bfs import run_algorithm as greedy_bfs

def main():
    # Load the maze
    maze_file = input("Enter the maze file to load: ")
    maze = maze_loader.load_maze(maze_file)

    # Set start and goal
    start = tuple(map(int, input("Enter start position (row, col): ").split(',')))
    goal = tuple(map(int, input("Enter goal position (row, col): ").split(',')))

    # Choose the algorithm
    print("Choose the algorithm: 1) A* 2) BFS 3) Greedy BFS")
    choice = int(input("Enter your choice: "))
    
    if choice == 1:
        path = a_star(maze, start, goal)
    elif choice == 2:
        path = bfs(maze, start, goal)
    elif choice == 3:
        path = greedy_bfs(maze, start, goal)
    else:
        print("Invalid choice!")
        return

    # Show results
    if path:
        for x, y in path:
            if maze[x][y] == 0:  # Don't overwrite start/goal
                maze[x][y] = 4  # Mark path
        maze_loader.print_maze(maze)
    else:
        print("No path found!")

if __name__ == "__main__":
    main()
