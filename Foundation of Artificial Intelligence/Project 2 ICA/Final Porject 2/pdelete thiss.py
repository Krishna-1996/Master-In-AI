import os

# Path to the folder
folder_path = 'D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/Final Porject 2/New folder'
# D:/Masters Projects/Master-In-AI/Foundation of Artificial Intelligence/Project 2 ICA/Final Porject 2/Jupyter Notebook

# List all files in the folder
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Print the file names
for file_name in file_names:
    print(file_name)
