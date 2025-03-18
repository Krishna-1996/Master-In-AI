import tkinter as tk  # Import tkinter for GUI creation
from tkinter import messagebox  # Import messagebox for displaying error messages
from pyamaze import maze, agent, COLOR, textLabel  # Import necessary components from pyamaze library
import importlib  # Import importlib to dynamically load the algorithm modules
import os  # Import os to handle file paths

# Mapping obstacle percentage to CSV file names
obstacle_files = {
    0: "Obstacles_Design_0p.csv",  # File for 0% obstacles
    10: "Obstacles_Design_10p.csv",  # File for 10% obstacles
    20: "Obstacles_Design_20p.csv",  # File for 20% obstacles
    30: "Obstacles_Design_30p.csv"  # File for 30% obstacles
}

# Predefined goal positions
goal_positions = {
    "Top Left": (1, 1),  # Goal at the top-left corner
    "Top Right": (1, 99),  # Goal at the top-right corner
    "Bottom Left": (49, 1),  # Goal at the bottom-left corner
    "Bottom Right": (49, 99),  # Goal at the bottom-right corner
    "Center": (25, 50)  # Goal at the center of the maze
}

# Main class for the GUI
class MazeSolverApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Initialize the GUI window
        self.title("Maze Solver")
        self.geometry("1000x800")  # Adjusted window size for better maze display

        # Default algorithm choice (BFS)
        self.algorithm = tk.StringVar(value="BFS")

        # Default obstacle percentage (0%)
        self.obstacle_percentage = tk.IntVar(value=0)

        # Default goal position ("Top Left")
        self.goal_position = tk.StringVar(value="Top Left")

        # Create a frame to hold the dropdown menus and button
        self.dropdown_frame = tk.Frame(self)
        self.dropdown_frame.pack(pady=20)

        # Create and place the dropdowns and button
        self.create_algorithm_selection()
        self.create_obstacle_selection()
        self.create_goal_selection()
        self.create_run_button()

        # Scrollable Frame to display the maze
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas widget to hold the maze
        self.canvas = tk.Canvas(self.canvas_frame, width=1000, height=700)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a vertical scrollbar
        scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=scrollbar.set)

    def create_algorithm_selection(self):
        """Create the dropdown for algorithm selection."""
        label = tk.Label(self.dropdown_frame, text="Select Algorithm:", font=("Arial", 10, "bold"))
        label.grid(row=0, column=0, padx=10)

        algorithm_choices = ["BFS", "DFS", "A*", "Greedy BFS"]
        algorithm_menu = tk.OptionMenu(self.dropdown_frame, self.algorithm, *algorithm_choices)
        algorithm_menu.grid(row=0, column=1, padx=10)

    def create_obstacle_selection(self):
        """Create the dropdown for obstacle percentage selection."""
        label = tk.Label(self.dropdown_frame, text="Select Obstacle Percentage:", font=("Arial", 10, "bold"))
        label.grid(row=0, column=2, padx=10)

        obstacle_choices = [0, 10, 20, 30]
        obstacle_menu = tk.OptionMenu(self.dropdown_frame, self.obstacle_percentage, *obstacle_choices)
        obstacle_menu.grid(row=0, column=3, padx=10)

    def create_goal_selection(self):
        """Create the dropdown for goal selection."""
        label = tk.Label(self.dropdown_frame, text="Select Goal Position:", font=("Arial", 10, "bold"))
        label.grid(row=0, column=4, padx=10)

        goal_choices = list(goal_positions.keys())  # The keys of the goal_positions dictionary
        goal_menu = tk.OptionMenu(self.dropdown_frame, self.goal_position, *goal_choices)
        goal_menu.grid(row=0, column=5, padx=10)

    def create_run_button(self):
        """Create the 'Run' button to execute the algorithm."""
        run_button = tk.Button(self.dropdown_frame, text="Run", command=self.run_algorithm, 
                               font=("Arial", 12, "bold"), fg="white", bg="#4CAF50", relief="flat", width=10)
        
        # Add hover effect for the button
        run_button.bind("<Enter>", lambda e: self.on_hover(run_button))
        run_button.bind("<Leave>", lambda e: self.on_leave(run_button))
        
        run_button.grid(row=1, column=2, padx=10)

    def on_hover(self, button):
        """Change button color when the mouse hovers over it."""
        button.config(bg="#45a049")

    def on_leave(self, button):
        """Reset button color when the mouse leaves it."""
        button.config(bg="#4CAF50")

    def run_algorithm(self):
        """Run the selected algorithm on the maze."""
        # Get the algorithm, obstacle percentage, and goal position
        algorithm_choice = self.algorithm.get().lower()
        obstacle_percentage = self.obstacle_percentage.get()
        goal_choice = self.goal_position.get()
        goal_position = goal_positions[goal_choice]  # Get the selected goal position from the dictionary

        # Path to the selected obstacle CSV file
        csv_file_path = os.path.join("D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Friend_Project/Maze_Project/mazes", obstacle_files[obstacle_percentage])

        try:
            # Load the maze based on the selected obstacle CSV file
            m = maze(70, 120)
            m.CreateMaze(loadMaze=csv_file_path)

            # Dynamically import the algorithm based on user selection
            if algorithm_choice == "bfs":
                algorithm_module = importlib.import_module('algorithms.BFS_Algorithm')  # Import BFS algorithm
                exploration_order, visited_cells, path_to_goal = algorithm_module.bfs_search(m, goal=goal_position)
            elif algorithm_choice == "dfs":
                algorithm_module = importlib.import_module('algorithms.DFS_Algorithm')  # Import DFS algorithm
                exploration_order, visited_cells, path_to_goal = algorithm_module.dfs_search(m, goal=goal_position)
            elif algorithm_choice == "a*":
                algorithm_module = importlib.import_module('algorithms.A_Star')  # Import A* algorithm
                exploration_order, visited_cells, path_to_goal = algorithm_module.A_star_search(m, goal=goal_position)
            elif algorithm_choice == "greedy bfs":
                algorithm_module = importlib.import_module('algorithms.Greedy_BFS')  # Import Greedy BFS algorithm
                exploration_order, visited_cells, path_to_goal = algorithm_module.greedy_bfs_search(m, goal=goal_position)
            else:
                raise ValueError("Unknown algorithm choice!")

            # If a path to the goal is found, visualize it in the maze
            if path_to_goal:
                agent_bfs = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
                agent_trace = agent(m, footprints=True, shape='square', color=COLOR.yellow, filled=True)
                agent_goal = agent(m, goal_position[0], goal_position[1], footprints=True, color=COLOR.green, shape='square', filled=True)

                # Trace the agent's path through the maze
                m.tracePath({agent_bfs: exploration_order}, delay=1)
                m.tracePath({agent_trace: path_to_goal}, delay=1)
                m.tracePath({agent_goal: visited_cells}, delay=1)

                # Display additional information on the maze visualization
                textLabel(m, 'Goal Position', str(goal_position))
                textLabel(m, f'{algorithm_choice.capitalize()} Path Length', len(path_to_goal) + 1)
                textLabel(m, f'{algorithm_choice.capitalize()} Search Length', len(exploration_order))

                # Run the maze visualization in the Python Maze World window
                m.run()

            else:
                raise ValueError("No path found to the goal!")

        except Exception as e:
            # Display an error message if something goes wrong
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Create and run the Tkinter app
if __name__ == "__main__":
    app = MazeSolverApp()  # Instantiate the GUI application
    app.mainloop()  # Run the Tkinter event loop
