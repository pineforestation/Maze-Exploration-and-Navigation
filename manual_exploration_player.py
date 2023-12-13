from enum import Enum
import pygame

from vis_nav_game import Action, Phase
from base_player import BasePlayer


class CustomAction(Enum):
    QUARTER_TURN_LEFT = 33
    QUARTER_TURN_RIGHT = 34
    RESET_TRUE_NORTH = 39
    PROCESS_EXPLORATION_IMAGES = 40


class ManualExplorationPlayer(BasePlayer):
    """
    Control manually using the keyboard during the exploration phase,
    then automatically navigate to the target.
    """

    def __init__(self):
        super(ManualExplorationPlayer, self).__init__()
        self.keymap = None


    def reset(self):
        super(ManualExplorationPlayer, self).reset()
        self.keymap = {
            pygame.K_LEFT: CustomAction.QUARTER_TURN_LEFT,
            pygame.K_RIGHT: CustomAction.QUARTER_TURN_RIGHT,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
            pygame.K_q: Action.LEFT,
            pygame.K_e: Action.RIGHT,
            pygame.K_n: CustomAction.RESET_TRUE_NORTH,
            pygame.K_RETURN: CustomAction.PROCESS_EXPLORATION_IMAGES,
        }


    def get_next_action(self):
        phase = None
        state = self.get_state()
        if state is not None:
          phase = state[1]

        if phase == Phase.NAVIGATION and self.nav_point is not None:
            next_action, goal_reached = self.follow_path()
            if goal_reached:
                next_action = Action.CHECKIN
                end_time = self.get_state()[3]
                print(f"reached goal in {(end_time - self.set_target_img_timestamp):.2f} seconds")
        else:
            next_action = self.manual_action()
        return next_action


    def manual_action(self):
        """
            Let the player control using the keyboard.
        """
        next_action = Action.IDLE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return Action.QUIT
            elif event.type == pygame.KEYDOWN and event.key in self.keymap:
                action = self.keymap[event.key]
                if isinstance(action, CustomAction):
                    next_action = self.perform_custom_action(action)
                elif isinstance(action, Action):
                    next_action = action

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            next_action = Action.FORWARD
        elif keys[pygame.K_DOWN]:
            next_action = Action.BACKWARD

        return next_action


    def perform_custom_action(self, action):
        next_action = Action.IDLE

        if action == CustomAction.QUARTER_TURN_LEFT:
            if self.heading == self.HEADING_EAST:
                self.action_queue = [Action.IDLE] + [Action.LEFT] * 36
            else:
                self.action_queue = [Action.IDLE] + [Action.LEFT] * 37
        elif action == CustomAction.QUARTER_TURN_RIGHT:
            if self.heading == self.HEADING_NORTH:
                self.action_queue = [Action.IDLE] + [Action.RIGHT] * 36
            else:
                self.action_queue = [Action.IDLE] + [Action.RIGHT] * 37
        elif action == CustomAction.RESET_TRUE_NORTH:
            self.heading = 0
        elif action == CustomAction.PROCESS_EXPLORATION_IMAGES:
            self.post_exploration_processing()
            self.action_queue = [Action.QUIT]
        else:
            raise NotImplementedError(f"Unknown custom action: {action}")

        return next_action

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=ManualExplorationPlayer())
