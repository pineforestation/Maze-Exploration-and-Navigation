from enum import Enum, auto
import numpy as np
import random
from vis_nav_game import Action
from path_search import AStar
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
        self.steps_since_last_scan = 0


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
        if self.get_state()[2] > 200: # TODO map percent orr time remaining
            self.control_state = ControlState.POST_EXPLORE
            self.post_exploration_processing()
        else:
            start = (self.get_map_coord_y(self.y), self.get_map_coord_x(self.x))
            self.goal = (random.randint(0, self.MAP_WIDTH), random.randint(0, self.MAP_WIDTH)) #TODO check that this is unexplored
            self.path = AStar(start, self.goal, self.occupancy_grid, allow_unknown=True).perform_search()
            if len(self.path) > 0:
                self.control_state = ControlState.FOLLOW_EXPLORATION_PATH
                shape = self.occupancy_grid.shape
                self.path_overlay = np.zeros(shape, dtype=np.uint8)
                for (y, x) in self.path:
                    self.path_overlay[y, x] = 1
                self.nav_point = self.path.pop(0)
        return Action.IDLE


    def follow_exploration_path(self):
        next_action = Action.IDLE
        if self.nav_point is None:
            self.control_state = ControlState.FIND_PATH_TO_EXPLORE
        elif self.steps_since_last_scan > 10: # TODO don't rescan same location
            self.steps_since_last_scan = 0
            self.action_queue = [Action.IDLE] + [Action.LEFT] * 147
        else:
            next_action, _goal_reached = self.follow_path(strict_collision_checking=True)
            if next_action == Action.FORWARD:
                self.steps_since_last_scan += 1
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
