from enum import Enum, auto
import numpy as np
import random
from vis_nav_game import Action
from path_search import AStar, OccupancyMap
from base_player import BasePlayer


class ControlState(Enum):
    INIT = auto()
    FIND_PATH_TO_EXPLORE = auto()
    FOLLOW_EXPLORATION_PATH = auto()
    POST_EXPLORE = auto()
    FOLLOW_NAVIGATION_PATH = auto()


class FullAutoPlayer(BasePlayer):
    """
    Fully automatic exploration and navigation.
    """

    def __init__(self):
        super(FullAutoPlayer, self).__init__()
        self.keymap = None
        self.control_state = ControlState.INIT
        self.performed_scan = False
        self.steps_since_last_scan = 0
        self.steps_taken_on_path = 0
        self.failed_path_count = 0
        self.percent_to_explore = 75
        self.actual_map_area = (self.MAP_WIDTH // 2) ** 2
        self.good_checkpoints = [(self.get_map_coord_y(0), self.get_map_coord_x(0))]
        self.scanned_positions = set()


    def get_next_action(self):
        next_action = Action.IDLE
        if self.control_state == ControlState.INIT:
            next_action = self.initial_action()
        elif self.control_state == ControlState.FIND_PATH_TO_EXPLORE:
            next_action = self.find_path_to_explore()
        elif self.control_state == ControlState.FOLLOW_EXPLORATION_PATH:
            next_action = self.follow_exploration_path()
        elif self.control_state == ControlState.POST_EXPLORE:
            next_action = self.post_explore()
        elif self.control_state == ControlState.FOLLOW_NAVIGATION_PATH:
            next_action = self.follow_navigation_path()
        return next_action


    def initial_action(self):
        self.action_queue = [Action.IDLE] + [Action.LEFT] * 74 + [Action.IDLE] + [Action.FORWARD] * 10 + [Action.IDLE] + [Action.LEFT] * 73 + [Action.IDLE] + [Action.FORWARD] * 10
        self.control_state = ControlState.FIND_PATH_TO_EXPLORE
        return Action.IDLE


    def find_path_to_explore(self):
        next_action = Action.IDLE

        if self.steps_taken_on_path > 10:
            self.good_checkpoints.append(self.goal)
            print("good checkpoints: ", self.good_checkpoints)

        self.goal = None

        print(f"Finding a new path. Followed the previous path for {self.steps_taken_on_path} steps.")
        if self.steps_taken_on_path < 3:
            self.failed_path_count += 1
            self.refine_occupancy_map()
            print(f"PATH FAILED. Failed {self.failed_path_count} paths in a row.")
            if self.failed_path_count == 2:
                print("I might be stuck. Will try backing up.")
                self.action_queue = [Action.BACKWARD] * 2
            elif self.failed_path_count == 5:
                print("I might be stuck. Will try thinning out the wall markers.")
                self.erode_occupancy_map()
            elif self.failed_path_count == 6:
                print("I might be stuck. Will try moving somewhere close.")
                self.goal = (self.get_map_coord_y(self.y-1), self.get_map_coord_x(self.x))
            elif self.failed_path_count == 7:
                print("I might be stuck. Will try moving somewhere close.")
                self.goal = (self.get_map_coord_y(self.y+1), self.get_map_coord_x(self.x))
            elif self.failed_path_count == 8:
                print("I might be stuck. Will try moving somewhere close.")
                self.goal = (self.get_map_coord_y(self.y), self.get_map_coord_x(self.x-1))
            elif self.failed_path_count == 9:
                print("I might be stuck. Will try moving somewhere close.")
                self.goal = (self.get_map_coord_y(self.y-1), self.get_map_coord_x(self.x+1))
            elif self.failed_path_count == 10:
                print("I might be stuck. Going to go back to a previous spot.")
                self.goal = self.good_checkpoints.pop()
            # if self.failed_path_count >= 8:
            #     print("I might be stuck. The next path will be followed with less strict collision avoidance to try to get unstuck, but this risks odometry errors.")
        else:
            self.failed_path_count = 0

        self.steps_taken_on_path = 0

        time_remaining = self.get_state()[5]
        percent_explored = np.count_nonzero(self.occupancy_grid) / self.actual_map_area * 100.0
        print(f"Percent explored: {percent_explored:.1f}% (will stop at {self.percent_to_explore}%); " +
              f"Time remaining: {time_remaining:.0f}s (will stop with 5 minutes remaining)")
        if time_remaining < 300 or percent_explored >= self.percent_to_explore:
            print("Exploration complete")
            self.control_state = ControlState.POST_EXPLORE
            self.post_exploration_processing()
        else:
            start = (self.get_map_coord_y(self.y), self.get_map_coord_x(self.x))
            # Use a thickened version of the map for pathfinding, in order to avoid bumping walls
            thick_occ_grid = self.thicken_occupancy_map(num_dilations=(2 - self.failed_path_count // 6))

            if self.goal is None:
                # Find a good place to explore in an increasing radius from the current position
                d = 5
                path_found = False
                while(not path_found):
                    d += 1
                    dist = d // 2 + self.failed_path_count
                    x_min = max(0, start[0] - dist)
                    x_max = min(self.MAP_WIDTH-1, start[0] + dist)
                    y_min = max(0, start[1] - dist)
                    y_max = min(self.MAP_WIDTH-1, start[1] + dist)
                    self.goal = (random.randint(x_min, x_max), random.randint(y_min, y_max))
                    print(f"x={x_min}:{x_max}, y=x={y_min}:{y_max}, got goal={self.goal}")
                     # Check that this is unexplored and worth going to
                    if self.occupancy_grid[self.goal] != OccupancyMap.UNKNOWN:
                        print(f"Skippping {self.goal} since it isn't unkown; dist={dist}")
                        continue
                    # Don't try to go to places that are too closed to walls
                    if (thick_occ_grid[self.goal[0]-1:self.goal[0]+2, self.goal[1]-1:self.goal[1]+2] == OccupancyMap.OBSTACLE).any():
                        print(f"Skippping {self.goal} since it is too close to a wall; dist={dist}")
                        continue
                    self.path = AStar(start, self.goal, thick_occ_grid, allow_unknown=True).perform_search()
                    path_found = (len(self.path) > 0)

            else:
                self.path = AStar(start, self.goal, thick_occ_grid, allow_unknown=True).perform_search()

            if len(self.path) > 0:
                self.control_state = ControlState.FOLLOW_EXPLORATION_PATH
                shape = self.occupancy_grid.shape
                self.path_overlay = np.zeros(shape, dtype=np.uint8)
                for (y, x) in self.path:
                    self.path_overlay[y, x] = 1
                self.nav_point = self.path.pop(0)
        return next_action


    def follow_exploration_path(self):
        next_action = Action.IDLE

        # After a certain number of steps, rotate 360 degrees to get a good look at all the walls
        if self.performed_scan:
            self.steps_since_last_scan = 0
            self.performed_scan = False
            self.refine_occupancy_map()
        elif self.steps_since_last_scan > 10:
            if (self.x, self.y) not in self.scanned_positions:
                # Don't rescan same location
                self.action_queue = [Action.IDLE] + [Action.LEFT] * 147
                self.scanned_positions.add((self.x, self.y))
            self.performed_scan = True
            print("Scanning.")

        if self.nav_point is None:
            self.control_state = ControlState.FIND_PATH_TO_EXPLORE
        else:
            next_action, _goal_reached = self.follow_path(strict_collision_checking=(self.failed_path_count<=10))
            # next_action, _goal_reached = self.follow_path(strict_collision_checking=(True))
            if next_action == Action.FORWARD:
                self.steps_since_last_scan += 1
                self.steps_taken_on_path +=1
            elif next_action == Action.BACKWARD:
                self.steps_since_last_scan -= 1
                self.steps_taken_on_path -=1
        return next_action


    def post_explore(self):
        self.control_state = ControlState.FOLLOW_NAVIGATION_PATH
        return Action.QUIT


    def follow_navigation_path(self):
        next_action = Action.IDLE
        if self.nav_point is not None:
            next_action, goal_reached = self.follow_path(strict_collision_checking=False)
            if goal_reached:
                next_action = Action.CHECKIN
                end_time = self.get_state()[3]
                print(f"reached goal in {(end_time - self.set_target_img_timestamp):.2f} seconds")
        else:
            # TODO try to rescue this, or checkin if the target location is close enough
            print("I can't reach the goal ")
            next_action = Action.QUIT
        return next_action


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=FullAutoPlayer())
