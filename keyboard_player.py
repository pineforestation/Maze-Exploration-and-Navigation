from vis_nav_game import Player, Action, Phase
from enum import Enum
import pygame
import numpy as np
import cv2
import math
from time import sleep, strftime
import os
from path_search import OccupancyMap, AStar


def convert_opencv_img_to_pygame(opencv_image, bgr_to_rb = False):
    """
    Convert OpenCV images for Pygame.

    see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
    """
    if bgr_to_rb:
        opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
    shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
    pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

    return pygame_image

class CustomAction(Enum):
    QUARTER_TURN_LEFT = 33
    QUARTER_TURN_RIGHT = 34
    MARK_NORTH_WALL = 35
    MARK_WEST_WALL = 36
    MARK_SOUTH_WALL = 37
    MARK_EAST_WALL = 38
    RESET_TRUE_NORTH = 39

class KeyboardPlayerPyGame(Player):
    """
    Control manually using the keyboard.
    """

    MAP_WIDTH = 200

    HEADING_NORTH = 0
    HEADING_EAST = 36
    HEADING_SOUTH = 73
    HEADING_WEST = 110

    def __init__(self):
        self.fpv = None
        self.screen = None
        self.keymap = None
        self.other_keymap = None
        self.filepath = ''
        self.action_queue = []

        self.x = 0
        self.y = 0
        self.heading = 0

        self.target_location = None

        self.occupancy_grid = np.zeros(shape=(self.MAP_WIDTH, self.MAP_WIDTH), dtype=np.uint8)
        self.path = []
        self.nav_point = None
        self.path_overlay = None

        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.screen = None

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: CustomAction.QUARTER_TURN_LEFT,
            pygame.K_RIGHT: CustomAction.QUARTER_TURN_RIGHT,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
            pygame.K_q: Action.LEFT,
            pygame.K_e: Action.RIGHT,
            pygame.K_w: CustomAction.MARK_NORTH_WALL,
            pygame.K_a: CustomAction.MARK_WEST_WALL,
            pygame.K_s: CustomAction.MARK_SOUTH_WALL,
            pygame.K_d: CustomAction.MARK_EAST_WALL,
            pygame.K_n: CustomAction.RESET_TRUE_NORTH,
        }

        self.filepath = './data/exploration_views/' + strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.filepath, exist_ok=True)

    def pre_exploration(self):
        self.x = 0
        self.y = 0
        self.heading = 0
        self.action_queue = []

    def pre_navigation(self):
        self.x = 0
        self.y = 0
        self.heading = 0
        self.action_queue = []

    def get_map_coord_x(self, raw_coord_x):
        return self.MAP_WIDTH // 2 + round(raw_coord_x)
    
    def get_map_coord_y(self, raw_coord_y):
        return self.MAP_WIDTH // 2 - round(raw_coord_y)
    
    def act(self):
        grid_coord_x = self.get_map_coord_x(self.x)
        grid_coord_y = self.get_map_coord_y(self.y)

        step = 0
        phase = None
        state = self.get_state()
        if state is not None:
          step = state[2]
          phase = state[1]
        
        # At the start of the simulation, the robot takes a few seconds to settle into position.
        # Tracking gets messed up if you try to move before this is done.
        if step < 40:
            return Action.IDLE
        
        if phase == Phase.NAVIGATION and len(self.path) > 0:
            next_action = self.follow_path(grid_coord_x, grid_coord_y)
        else:
            next_action = self.manual_action()
        
        wall_ahead = (phase == Phase.EXPLORATION) and self.check_for_collision_ahead()
        if wall_ahead:
            # Draw the wall markers on the minimap
            # TODO: Sometimes these are too wide, e.g. when coming at parallel at a wall, 
            #       or when the wall is off to the side. We might want to make this more accurate.
            if self.heading == self.HEADING_NORTH:
                self.occupancy_grid[grid_coord_y - 8:grid_coord_y - 5, grid_coord_x-6:grid_coord_x+6] = OccupancyMap.OBSTACLE
            elif self.heading == self.HEADING_EAST:
                self.occupancy_grid[grid_coord_y-6:grid_coord_y+6, grid_coord_x + 5:grid_coord_x + 8] = OccupancyMap.OBSTACLE
            elif self.heading == self.HEADING_SOUTH:
                self.occupancy_grid[grid_coord_y + 5:grid_coord_y + 8, grid_coord_x-6:grid_coord_x+6] = OccupancyMap.OBSTACLE
            elif self.heading == self.HEADING_WEST:
                self.occupancy_grid[grid_coord_y-6:grid_coord_y+6, grid_coord_x - 8:grid_coord_x - 5] = OccupancyMap.OBSTACLE

        if next_action == Action.FORWARD:
            if wall_ahead:
                next_action = Action.IDLE
            else:
                converted_heading = self.heading / 147 * 2 * math.pi
                self.x += math.sin(converted_heading)
                self.y += math.cos(converted_heading)
        elif next_action == Action.BACKWARD:
            converted_heading = self.heading / 147 * 2 * math.pi
            self.x -= math.sin(converted_heading)
            self.y -= math.cos(converted_heading)
        elif next_action == Action.LEFT:
            self.heading = (self.heading - 1) % 147
        elif next_action == Action.RIGHT:
            self.heading = (self.heading + 1) % 147

        self.occupancy_grid[grid_coord_y-1:grid_coord_y+1, grid_coord_x-1:grid_coord_x+1] = OccupancyMap.VISITED

        # For now, set the target to be the last location visited in the first phase.
        # Later, this will be set by the result of Visual Place Recognition.
        if phase == Phase.EXPLORATION:
            self.target_location = (grid_coord_y, grid_coord_x)

        return next_action
    
    def follow_path(self, grid_coord_x, grid_coord_y):
        """ 
            Automatically follow the path to the target as return by A* or other algorithm.
        """
        # If the current nav point has been reached, get the next one
        if self.nav_point == (grid_coord_y, grid_coord_x):
            if len(self.path) == 0:
                return Action.CHECKIN
            else:
                self.nav_point = self.path.pop(0)

        nav_y, nav_x = self.nav_point
        if nav_y < grid_coord_y:
            if self.heading == self.HEADING_NORTH:
                return Action.FORWARD
            elif self.heading <= self.HEADING_SOUTH:
                return Action.LEFT
            else:
                return Action.RIGHT
        elif nav_y > grid_coord_y:
            if self.heading == self.HEADING_SOUTH:
                return Action.FORWARD
            elif self.heading < self.HEADING_SOUTH:
                return Action.RIGHT
            else:
                return Action.LEFT
        elif nav_x > grid_coord_x:
            if self.heading == self.HEADING_EAST:
                return Action.FORWARD
            elif self.heading < self.HEADING_EAST or self.heading > self.HEADING_WEST:
                return Action.RIGHT
            else:
                return Action.LEFT
        elif nav_x < grid_coord_x:
            if self.heading == self.HEADING_WEST:
                return Action.FORWARD
            elif self.heading < self.HEADING_WEST and self.heading > self.HEADING_EAST:
                return Action.RIGHT
            else:
                return Action.LEFT
        return Action.IDLE

    def manual_action(self):
        """
            Let the player control using the keyboard.
        """
        next_action = Action.IDLE

        if self.action_queue:
            next_action = self.action_queue.pop()
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return Action.QUIT
                elif event.type == pygame.KEYDOWN and event.key in self.keymap:
                    action = self.keymap[event.key]
                    if isinstance(action, CustomAction):
                        self.perform_custom_action(action)
                    elif isinstance(action, Action):
                        next_action = action

            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                next_action = Action.FORWARD
            elif keys[pygame.K_DOWN]:
                next_action = Action.BACKWARD
        return next_action
    
    def check_for_collision_ahead(self):
        """
            Check if there is floor visible at the bottom of the first-person view.
            If not, we are about to hit a wall, so prevent moving forward.
        """
        fpv = self.fpv
        w = fpv.shape[1]
        width_to_check = 40

        white_floor = [239, 239, 239]
        blue_floor = [224, 186, 162]

        bottom_row = fpv[-1, (w // 2 - width_to_check):(w // 2 + width_to_check), :]

        return not ((bottom_row == white_floor) | (bottom_row == blue_floor)).all()

    def perform_custom_action(self, action):
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
        elif action == CustomAction.MARK_NORTH_WALL:
            self.occupancy_grid[self.get_map_coord_y(self.y) - 1, :] = OccupancyMap.OBSTACLE
        elif action == CustomAction.MARK_SOUTH_WALL:
            self.occupancy_grid[self.get_map_coord_y(self.y) + 1, :] = OccupancyMap.OBSTACLE
        elif action == CustomAction.MARK_WEST_WALL:
            self.occupancy_grid[:, self.get_map_coord_x(self.x) - 1] = OccupancyMap.OBSTACLE
        elif action == CustomAction.MARK_EAST_WALL:
            self.occupancy_grid[:, self.get_map_coord_x(self.x) + 1] = OccupancyMap.OBSTACLE
        elif action == CustomAction.RESET_TRUE_NORTH:
            self.heading = 0
        else:
            raise NotImplementedError(f"Unknown custom action: {action}")
        
    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

        if images is None or len(images) <= 0:
            return
        
        start = (100, 100)
        self.path = AStar(start, self.target_location, self.occupancy_grid).perform_search()
        shape = self.occupancy_grid.shape
        self.path_overlay = np.zeros(shape, dtype=np.uint8)
        for (y, x) in self.path:
            self.path_overlay[y, x] = 1
        self.nav_point = self.path.pop(0)
    
    def convert_occupancy_to_cvimg(self):
        shape = self.occupancy_grid.shape
        height, width = shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[self.occupancy_grid == OccupancyMap.VISITED] = [255, 0, 0]
        image[self.occupancy_grid == OccupancyMap.OBSTACLE] = [0, 255, 0]
        image[self.path_overlay == 1] = [0, 0, 255]
        return image
        
    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        h, w, _ = fpv.shape
        if self.screen is None:
            self.screen = pygame.display.set_mode((4*w, 2*h))

        # Save camera observations (exploration phase only)
        # These can be used for SfM, SLAM, and visual place recognition
        step = 0
        phase = None
        state = self.get_state()
        if state is not None:
          step = state[2]
          phase = state[1]
        if phase == Phase.EXPLORATION and step % 5 == 1:
            cv2.imwrite(os.path.join(self.filepath, f"{step}_{self.x}_{self.y}_{self.heading}.png"), fpv)

        # Draw the heads up display (HUD) and the mini-map
        hud_img = np.zeros(shape=(h,w,3), dtype=np.uint8)
        w_offset = 10
        h_offset = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1
        color = (255, 255, 255)
        cv2.putText(hud_img, f"x={self.x:.2f}, y={self.y:.2f}", (w_offset, h_offset), font, size, color, stroke, line)
        cv2.putText(hud_img, f"heading={self.heading}", (w_offset, h_offset * 2), font, size, color, stroke, line)
        
        # Draw the position marker
        minimap = self.convert_occupancy_to_cvimg()
        marker_size = 10
        marker_img = np.zeros(shape=(marker_size, marker_size, 3), dtype=np.uint8)
        cv2.drawMarker(marker_img, (marker_size // 2, marker_size // 2), [0, 255, 0], cv2.MARKER_TRIANGLE_UP, marker_size)
        rotation_matrix = cv2.getRotationMatrix2D((marker_img.shape[1] // 2, marker_img.shape[0] // 2), (self.heading / 147 * -360), 1)
        rotated_marker = cv2.warpAffine(marker_img, rotation_matrix, (marker_img.shape[1], marker_img.shape[0]))
        x = self.get_map_coord_x(self.x) - marker_size // 2
        y = self.get_map_coord_y(self.y) - marker_size // 2
        minimap[y:y+rotated_marker.shape[0], x:x+rotated_marker.shape[1]][rotated_marker != [0, 0, 0]] = rotated_marker[rotated_marker != [0, 0, 0]]

        # Display everything
        pygame.display.set_caption(f"{self.__class__.__name__}:fpv; h: {h} w:{w}; step{step}")
        fpv_pygame = convert_opencv_img_to_pygame(cv2.resize(fpv, (2*w, 2*h)), True)
        hud_pygame = convert_opencv_img_to_pygame(hud_img, True)
        minimap_pygame = convert_opencv_img_to_pygame(cv2.resize(minimap, (2*w, 2*h)))
        self.screen.blit(fpv_pygame, (0, 0))
        self.screen.blit(hud_pygame, (2*w, 0))
        self.screen.blit(minimap_pygame, (2*w, 50))
        pygame.display.update()

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
