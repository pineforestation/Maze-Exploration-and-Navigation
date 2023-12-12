from enum import IntEnum
from queue import PriorityQueue

class OccupancyMap(IntEnum):
    UNKNOWN = 0
    UNVISITED = 1
    VISITED = 2
    OBSTACLE = 3

class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class AStar:
    def __init__(self, start, goal, occupancy_grid):
        self.occupancy_grid = occupancy_grid
        self.start = (*start, Direction.NORTH)
        self.goals = [(*goal, dir) for dir in Direction]

        height, width = occupancy_grid.shape
        self.height = height
        self.width = width

    def get_neighbors(self, coordinate):
        i, j, _direction = coordinate
        neighbors = []
        if i > 0 and self.occupancy_grid[i - 1, j] == OccupancyMap.UNVISITED:
            neighbors.append((i - 1, j, Direction.NORTH))
        if j > 0 and self.occupancy_grid[i, j - 1] == OccupancyMap.UNVISITED:
            neighbors.append((i, j - 1, Direction.WEST))
        if i < self.height - 1 and self.occupancy_grid[i + 1, j] == OccupancyMap.UNVISITED:
            neighbors.append((i + 1, j, Direction.SOUTH))
        if j < self.width - 1 and self.occupancy_grid[i, j + 1] == OccupancyMap.UNVISITED:
            neighbors.append((i, j + 1, Direction.EAST))
        return neighbors

    def get_cost(self, start, end):
        start_direction = start[2]
        end_direction = end[2]
        turning_cost = 0
        if start_direction == end_direction:
            turning_cost = 0
        elif (start_direction == end_direction) % 4 == 2:
            turning_cost = 72
        else:
            turning_cost = 36

        return 1 + turning_cost
        
    def heuristic(self, start, end):
        # L1 norm: Manhattan distance between two points
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def perform_search(self):
        goal_reached = False
        last_node = None
        
        frontier = PriorityQueue()
        reached_from = {}
        cost_to_reach = {}

        frontier.put((0, self.start))
        reached_from[self.start] = self.start
        cost_to_reach[self.start] = 0

        while not frontier.empty():
            (_priority, current_node) = frontier.get()
            if current_node in self.goals:
                goal_reached = True
                last_node = current_node
                break
            for next_node in self.get_neighbors(current_node):
                updated_cost = cost_to_reach[current_node] + self.get_cost(current_node, next_node)
                if (next_node not in reached_from or updated_cost < cost_to_reach[next_node]):
                    cost_to_reach[next_node] = updated_cost
                    priority = updated_cost + self.heuristic(next_node, self.goals[0])
                    frontier.put((priority, next_node))
                    reached_from[next_node] = current_node
    
        path = []
        # Reconstruct_best_path
        if goal_reached:
            current_node = last_node
            while current_node != self.start:
                path.append(current_node[0:2])
                current_node = reached_from[current_node]
            path.append(self.start[0:2])
            path.reverse()
        
        return path