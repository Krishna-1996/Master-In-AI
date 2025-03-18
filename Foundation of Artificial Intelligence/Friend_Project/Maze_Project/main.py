import tkinter as tk
from tkinter import messagebox
from pyamaze import maze, agent, COLOR, textLabel
import importlib
import os

# Define CSV maze files
obstacle_files = {
    0: "maze_0p.csv",
    10: "maze_10p.csv",
    30: "maze_30p.csv",
    50: "maze_50p.csv"
}

# Define goal and start positions
goal_position = (0, 0)  # Top-left
start_position = (69, 119)  # Bottom-right

class MazeSolverApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Maze Solver")
        self.geometry("700x150")

        self.algorithm = tk.StringVar(value="A*")
        self.obstacle_percentage = tk.IntVar(value=0)

        self.dropdown_frame = tk.Frame(self)
        self.dropdown_frame.pack(pady=20)

        self.create_algorithm_selection()
        self.create_obstacle_selection()
        self.create_run_button()

    def create_algorithm_selection(self):
        label = tk.Label(self.dropdown_frame, text="Select Algorithm:", font=("Arial", 10, "bold"))
        label.grid(row=0, column=0, padx=10)

        algorithm_choices = ["A*", "BFS", "DFS", "Greedy BFS"]
        algorithm_menu = tk.OptionMenu(self.dropdown_frame, self.algorithm, *algorithm_choices)
        algorithm_menu.grid(row=0, column=1, padx=10)

    def create_obstacle_selection(self):
        label = tk.Label(self.dropdown_frame, text="Select Obstacle %:", font=("Arial", 10, "bold"))
        label.grid(row=0, column=2, padx=10)

        obstacle_menu = tk.OptionMenu(self.dropdown_frame, self.obstacle_percentage, 0, 10, 30, 50)
        obstacle_menu.grid(row=0, column=3, padx=10)

    def create_run_button(self):
        run_button = tk.Button(self.dropdown_frame, text="Run", command=self.run_algorithm, 
                               font=("Arial", 12, "bold"), fg="white", bg="#4CAF50", relief="flat", width=10)
        run_button.grid(row=1, column=2, padx=10)

    def run_algorithm(self):
        algorithm_choice = self.algorithm.get().lower().replace(" ", "_")
        obstacle_percentage = self.obstacle_percentage.get()

        csv_file_path = os.path.join("mazes", obstacle_files[obstacle_percentage])

        try:
            m = maze(70, 120)
            m.CreateMaze(loadMaze=csv_file_path)

            algorithm_module = importlib.import_module(f"algorithms.{algorithm_choice}")
            exploration_order, visited_cells, path_to_goal = getattr(algorithm_module, f"{algorithm_choice}_search")(m, start=start_position, goal=goal_position)

            if path_to_goal:
                agent_path = agent(m, footprints=True, shape='square', color=COLOR.red, filled=True)
                m.tracePath({agent_path: path_to_goal}, delay=1)

                textLabel(m, f'{algorithm_choice.capitalize()} Path Length', len(path_to_goal) + 1)
                textLabel(m, f'{algorithm_choice.capitalize()} Search Length', len(exploration_order))

                m.run()
            else:
                raise ValueError("No path found!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = MazeSolverApp()
    app.mainloop()
