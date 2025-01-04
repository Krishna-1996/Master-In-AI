from a_star import A_star_search
from bfs import BFS_search
from greedy_bfs import greedy_bfs

if __name__ == '__main__':
    goal_position = (1, 1)
    
    # Load maze and obstacles from the saved CSV
    obstacle_file = 'obstacles.csv'
    
    # A* Algorithm
    print("Running A* Algorithm")
    exploration_order_astar, visited_cells_astar, path_to_goal_astar = A_star_search(maze_file='maze.csv', goal=goal_position)
    
    # BFS Algorithm
    print("Running BFS Algorithm")
    exploration_order_bfs, visited_cells_bfs, path_to_goal_bfs = BFS_search(maze_file='maze.csv', goal=goal_position)
    
    # Greedy BFS Algorithm
    print("Running Greedy BFS Algorithm")
    exploration_order_greedy, visited_cells_greedy, path_to_goal_greedy = greedy_bfs(maze_file='maze.csv', goal=goal_position)
    
    # Display results for comparison
    print("\nA* Results:")
    print(f"Path length: {len(path_to_goal_astar)}")
    
    print("\nBFS Results:")
    print(f"Path length: {len(path_to_goal_bfs)}")
    
    print("\nGreedy BFS Results:")
    print(f"Path length: {len(path_to_goal_greedy)}")
