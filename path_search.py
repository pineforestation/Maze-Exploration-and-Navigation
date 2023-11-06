from enum import IntEnum
from queue import PriorityQueue

class OccupancyMap(IntEnum):
    UNEXPLORED = 0
    VISITED = 1
    OBSTACLE = 2

class AStar:
    def __init__(self, start, goal, occupancy_grid):
        self.occupancy_grid = occupancy_grid
        self.start = start
        self.goal = goal

        height, width = occupancy_grid.shape
        self.height = height
        self.width = width

    def get_neighbors(self, coordinate):
        (i, j) = coordinate
        neighbors = []
        if i > 0 and self.occupancy_grid[i - 1, j] == OccupancyMap.VISITED:
            neighbors.append((i - 1, j))
        if j > 0 and self.occupancy_grid[i, j - 1] == OccupancyMap.VISITED:
            neighbors.append((i, j - 1))
        if i < self.height - 1 and self.occupancy_grid[i + 1, j] == OccupancyMap.VISITED:
            neighbors.append((i + 1, j))
        if j < self.width - 1 and self.occupancy_grid[i, j + 1] == OccupancyMap.VISITED:
            neighbors.append((i, j + 1))
        return neighbors

    def get_cost(self, _coordinate):
        # TODO might consider the cost of turning vs going in straight line
        return 1
        
    def heuristic(self, start, end):
        # L1 norm: Manhattan distance between two points
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def perform_search(self):
        goal_reached = False
        
        frontier = PriorityQueue()
        reached_from = {}
        cost_to_reach = {}

        frontier.put((0, self.start))
        reached_from[self.start] = self.start
        cost_to_reach[self.start] = 0

        while not frontier.empty():
            (_priority, current_node) = frontier.get()
            if current_node == self.goal:
                goal_reached = True
                break
            for next_node in self.get_neighbors(current_node):
                updated_cost = cost_to_reach[current_node] + self.get_cost(next_node)
                if (next_node not in reached_from or updated_cost < cost_to_reach[next_node]):
                    cost_to_reach[next_node] = updated_cost
                    priority = updated_cost + self.heuristic(next_node, self.goal)
                    frontier.put((priority, next_node))
                    reached_from[next_node] = current_node
    
        path = []
        # Reconstruct_best_path
        if goal_reached:
            current_node = self.goal
            while current_node != self.start:
                path.append(current_node)
                current_node = reached_from[current_node]
            path.append(self.start)
            path.reverse()
        
        return path