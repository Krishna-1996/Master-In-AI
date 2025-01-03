from flask import Flask, render_template, request, jsonify
import heapq
from pyamaze import maze

app = Flask(__name__)

# Global maze variable
m = None

def create_maze(rows=20, cols=20):
    global m
    m = maze(rows, cols)
    m.CreateMaze(loopPercent=40)
    return m

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(maze_obj, start, goal):
    frontier = []
    heapq.heappush(frontier, (0 + heuristic(start, goal), start))
    visited = {}
    g_costs = {start: 0}
    explored = set([start])

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for direction in 'ESNW':
            if maze_obj.maze_map[current][direction]:
                next_cell = get_next_cell(current, direction)
                new_g_cost = g_costs[current] + 1
                if next_cell not in explored or new_g_cost < g_costs.get(next_cell, float('inf')):
                    g_costs[next_cell] = new_g_cost
                    f_cost = new_g_cost + heuristic(next_cell, goal)
                    heapq.heappush(frontier, (f_cost, next_cell))
                    visited[next_cell] = current
                    explored.add(next_cell)

    path = []
    cell = goal
    while cell in visited:
        path.append(cell)
        cell = visited[cell]
    path.reverse()
    return path

def get_next_cell(current, direction):
    x, y = current
    if direction == 'E':
        return (x, y + 1)
    if direction == 'W':
        return (x, y - 1)
    if direction == 'N':
        return (x - 1, y)
    if direction == 'S':
        return (x + 1, y)
    return current

@app.route('/')
def home():
    create_maze(20, 20)
    return render_template('index.html')

@app.route('/run_astar', methods=['POST'])
def run_astar():
    global m
    start = (m.rows, m.cols)
    goal = (1, 1)
    path = a_star_search(m, start, goal)
    return jsonify({'path': path})

@app.route('/reset', methods=['POST'])
def reset():
    create_maze(20, 20)
    return jsonify({'status': 'Maze Reset'})

if __name__ == '__main__':
    app.run(debug=True)
