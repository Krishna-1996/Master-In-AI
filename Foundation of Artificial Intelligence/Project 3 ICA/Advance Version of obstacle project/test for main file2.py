import tkinter as tk
from tkinter import messagebox
from pyamaze import maze, agent, COLOR, textLabel
import importlib
import os

# Mapping obstacle percentage to CSV file names
obstacle_files = {
    0: "Obstacles_Design_1_0p.csv",
    10: "Obstacles_Design_2_10p.csv",
    30: "Obstacles_Design_3_30p.csv",
    50: "Obstacles_Design_4_50p.csv"
}

# Main class for the GUI
class MazeSolverApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Maze Solver")
        self.geometry("400x300")

        # Algorithm choice (default is BFS)
        self.algorithm = tk.StringVar(value="bfs")

        # Obstacle percentage choice (default is 0%)
        self.obstacle_percentage = tk.IntVar(value=0)

        # Dropdown for algorithm selection
        self.create_algorithm_selection()

        # Dropdown for obstacle percentage selection
        self.create_obstacle_selection()

        # Buttons to run the selected algorithm and reset the choices
        self.create_buttons()

    def create_algorithm_selection(self):
        """Create the dropdown for algorithm selection."""
        label = tk.Label(self, text="Select Algorithm:")
        label.pack(pady=10)

        algorithm_choices = ["BFS", "DFS", "A*", "Greedy BFS"]
        algorithm_menu = tk.OptionMenu(self, self.algorithm, *algorithm_choices)
        algorithm_menu.pack()

    def create_obstacle_selection(self):
        """Create the dropdown for obstacle percentage selection."""
        label = tk.Label(self, text="Select Obstacle Percentage:")
        label.pack(pady=10)

        obstacle_choices = [0, 10, 30, 50]
        obstacle_menu = tk.OptionMenu(self, self.obstacle_percentage, *obstacle_choices)
        obstacle_menu.pack()

    def create_buttons(self):
        """Create the 'Run' and 'Reset' buttons."""
        button_frame = tk.Frame(self)
        button_frame.pack(pady=20)

        # Run button
        run_button = tk.Button(button_frame, text="Run", command=self.run_algorithm)
        run_button.pack(side=tk.LEFT, padx=10)

        # Reset button
        reset_button = tk.Button(button_frame, text="Reset", command=self.reset_choices)
        reset_button.pack(side=tk.LEFT, padx=10)

    def reset_choices(self):
        """Reset the selections to their default values and stop the current maze."""
        # Reset the algorithm and obstacle percentage to default
        self.algorithm.set("bfs")
        self.obstacle_percentage.set(0)

        # Close the current maze window (reset Python Maze World visualization)
        self.quit()  # Close the current Tkinter window
        
        # Create a new Tkinter window to ensure it's a fresh start
        app = MazeSolverApp()
        app.mainloop()  # Start a new instance of the app

    def run_algorithm(self):
        """Run the selected algorithm on the maze."""
        # Get the algorithm and obstacle percentage
        algorithm_choice = self.algorithm.get().lower()
        obstacle_percentage = self.obstacle_percentage.get()

        # Path to the selected obstacle CSV file
        csv_file_path = os.path.join("D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 3 ICA/Advance Version of obstacle project/maze_csvs", obstacle_files[obstacle_percentage])

        try:
            # Load the maze
            m = maze(50, 100)
            m.CreateMaze(loadMaze=csv_file_path)

            # Define the goal position
            goal_position = (1, 1)

            # Dynamically import the algorithm based on user selection
            if algorithm_choice == "bfs":
                algorithm_module = importlib.import_module('algorithms.BFS_Algorithm')
                exploration_order, visited_cells, path_to_goal = algorithm_module.bfs_search(m, goal=goal_position)
            elif algorithm_choice == "dfs":
                algorithm_module = importlib.import_module('algorithms.DFS_Algorithm')
                exploration_order, visited_cells, path_to_goal = algorithm_module.dfs_search(m, goal=goal_position)
            elif algorithm_choice == "a*":
                algorithm_module = importlib.import_module('algorithms.A_Star')
                exploration_order, visited_cells, path_to_goal = algorithm_module.A_star_search(m, goal=goal_position)
            elif algorithm_choice == "greedy bfs":
                algorithm_module = importlib.import_module('algorithms.Greedy_BFS')
                exploration_order, visited_cells, path_to_goal = algorithm_module.greedy_bfs_search(m, goal=goal_position)
            else:
                raise ValueError("Unknown algorithm choice!")

            # If a path to goal is found, show it in the PYTHON MAZE WORLD window
            if path_to_goal:
                agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
                agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)
                agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

                m.tracePath({agent_bfs: exploration_order}, delay=1)
                m.tracePath({agent_trace: path_to_goal}, delay=1)
                m.tracePath({agent_goal: visited_cells}, delay=1)

                textLabel(m, 'Goal Position', str(goal_position))
                textLabel(m, f'{algorithm_choice.capitalize()} Path Length', len(path_to_goal) + 1)
                textLabel(m, f'{algorithm_choice.capitalize()} Search Length', len(exploration_order))

                # Run the maze visualization in the PYTHON MAZE WORLD window
                m.run()

            else:
                raise ValueError("No path found to the goal!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create and run the Tkinter app
if __name__ == "__main__":
    app = MazeSolverApp()
    app.mainloop()
