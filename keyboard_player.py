from vis_nav_game import Player, Action, Phase
from enum import Enum
import pygame
import numpy as np
import cv2
import math
from time import sleep, strftime, time
import os
from path_search import OccupancyMap, AStar
from visual_place_recognition import BovwPlaceRecognition


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
    PROCESS_EXPLORATION_IMAGES = 40


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

        self.vpr = None

        self.occupancy_grid = np.zeros(shape=(self.MAP_WIDTH, self.MAP_WIDTH), dtype=np.uint8)
        self.path = []
        self.nav_point = None
        self.path_overlay = None

        self.nav_phase_start = None

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
            pygame.K_RETURN: CustomAction.PROCESS_EXPLORATION_IMAGES,
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
        self.nav_phase_start = time()


    def get_map_coord_x(self, raw_coord_x):
        return self.MAP_WIDTH // 2 + round(raw_coord_x)


    def get_map_coord_y(self, raw_coord_y):
        return self.MAP_WIDTH // 2 - round(raw_coord_y)


    def add_obstacles(self, y_start, y_end, x_start, x_end):
        slice_to_update = self.occupancy_grid[y_start:y_end, x_start:x_end]
        mask = (slice_to_update != OccupancyMap.VISITED)
        self.occupancy_grid[y_start:y_end, x_start:x_end][mask] = OccupancyMap.OBSTACLE

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
        
        if phase == Phase.NAVIGATION and self.nav_point is not None:
            next_action = self.follow_path(grid_coord_x, grid_coord_y)
        else:
            next_action = self.manual_action()
        

        wall_ahead = (phase == Phase.EXPLORATION) and self.check_for_collision_ahead()
        if wall_ahead:
            # Draw the wall markers on the minimap
            # TODO: Sometimes these are too wide, e.g. when coming at parallel at a wall, 
            #       or when the wall is off to the side. We might want to make this more accurate.
            if self.heading == self.HEADING_NORTH:
                self.add_obstacles(grid_coord_y - 8, grid_coord_y - 5, grid_coord_x - 6, grid_coord_x + 6)
            elif self.heading == self.HEADING_EAST:
                self.add_obstacles(grid_coord_y - 6, grid_coord_y + 6, grid_coord_x + 5, grid_coord_x + 8)
            elif self.heading == self.HEADING_SOUTH:
                self.add_obstacles(grid_coord_y + 5, grid_coord_y + 8, grid_coord_x - 6, grid_coord_x + 6)
            elif self.heading == self.HEADING_WEST:
                self.add_obstacles(grid_coord_y - 6, grid_coord_y + 6, grid_coord_x - 8, grid_coord_x - 5)

        if next_action == Action.FORWARD:
            if wall_ahead:
                next_action = Action.IDLE
            else:
                converted_heading = self.heading / 147 * 2 * math.pi
                # self.x += math.sin(converted_heading)
                # self.y += math.cos(converted_heading)
                if self.heading == self.HEADING_WEST:
                    self.x -= 1
                elif self.heading == self.HEADING_NORTH:
                    self.y += 1
                elif self.heading == self.HEADING_SOUTH:
                    self.y -= 1
                elif self.heading == self.HEADING_EAST:
                    self.x += 1
        elif next_action == Action.BACKWARD:
            converted_heading = self.heading / 147 * 2 * math.pi
            self.x -= math.sin(converted_heading)
            self.y -= math.cos(converted_heading)
        elif next_action == Action.LEFT:
            self.heading = (self.heading - 1) % 147
        elif next_action == Action.RIGHT:
            self.heading = (self.heading + 1) % 147

        self.occupancy_grid[grid_coord_y, grid_coord_x] = OccupancyMap.VISITED

        return next_action


    def follow_path(self, grid_coord_x, grid_coord_y):
        """ 
            Automatically follow the path to the target as returned by A* or other algorithm.
        """
        # If the current nav point has been reached, get the next one
        if self.nav_point == (grid_coord_y, grid_coord_x):
            if len(self.path) == 0:
                self.nav_point = None
                end_time = time()
                print(f"reached goal in {end_time - self.nav_phase_start} seconds")
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
                        next_action = self.perform_custom_action(action)
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
        elif action == CustomAction.PROCESS_EXPLORATION_IMAGES:
            self.process_exploration_images()
            next_action = Action.QUIT
        else:
            raise NotImplementedError(f"Unknown custom action: {action}")

        return next_action

   
    def process_exploration_images(self):
        self.vpr = BovwPlaceRecognition()
        print("Begin building BOVW-VPR database")
        self.vpr.build_database(self.filepath)
        print("Finished building BOVW-VPR database")
        

    def show_target_images(self, all_matched_imgs, target_positions, target_guess):
        target_display = cv2.hconcat(all_matched_imgs)
        h, w = target_display.shape[:2]
        print(f"w:{w}, h:{h}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.9
        stroke = 1
        text_color = (255, 20, 20)
        line_color = (127, 127, 127)
        unused_coords_color = text_color
        target_coords_color = (20, 20, 255)

        cv2.putText(target_display, 'Front View', (0 + w // 16, h // 2 - 5), font, size, text_color, stroke, line)
        if (target_positions[0] == target_guess).all():
            coords_color = target_coords_color
        else:
            coords_color = unused_coords_color
        cv2.putText(target_display, f'x: {target_positions[0,0]}, y: {target_positions[0,1]}', (0 + w // 16, h - 20), font, size, coords_color, stroke, line)

        cv2.putText(target_display, 'Right View', (w // 4 + w // 16, h // 2 - 5), font, size, text_color, stroke, line)
        if (target_positions[1] == target_guess).all():
            coords_color = target_coords_color
        else:
            coords_color = unused_coords_color
        cv2.putText(target_display, f'x: {target_positions[1,0]}, y: {target_positions[1,1]}', (w // 4 + w // 16, h - 20), font, size, coords_color, stroke, line)

        cv2.putText(target_display, 'Back View', (w // 2 + w // 16, h // 2 - 5), font, size, text_color, stroke, line)
        if (target_positions[2] == target_guess).all():
            coords_color = target_coords_color
        else:
            coords_color = unused_coords_color
        cv2.putText(target_display, f'x: {target_positions[2,0]}, y: {target_positions[2,1]}', (w // 2 + w // 16, h - 20), font, size, coords_color, stroke, line)

        cv2.putText(target_display, 'Left View', (3 * w // 4 + w // 16, h // 2 - 5), font, size, text_color, stroke, line)
        if (target_positions[3] == target_guess).all():
            coords_color = target_coords_color
        else:
            coords_color = unused_coords_color
        cv2.putText(target_display, f'x: {target_positions[3,0]}, y: {target_positions[3,1]}', (3*w // 4 + w // 16, h - 20), font, size, coords_color, stroke, line)

        cv2.line(target_display, (w // 4, 0), (w // 4, h), line_color, 2)
        cv2.line(target_display, (w // 2, 0), (w // 2, h), line_color, 2)
        cv2.line(target_display, (3 * w // 4, 0), (3 * w // 4, h), line_color, 2)
        cv2.line(target_display, (0, h // 2), (w, h // 2), line_color, 2)
        cv2.imshow('Top: actual targets; Bottom: matched images', target_display)


    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)

        if images is None or len(images) <= 0:
            return

        all_matched_imgs = []
        target_positions = np.zeros((len(images), 2), dtype=np.int8)
        for i in range(4):
            match_filename, match_img, match_distance = self.vpr.query_by_image(images[i])
            all_matched_imgs.append(cv2.vconcat([images[i], match_img]))

            goal_x, goal_y, _heading = self.decode_filename(match_filename)
            print(f"Target_{i} (x,y): {goal_x}, {goal_y}; confidence is {match_distance}")
            target_positions[i] = (goal_x, goal_y)

        average_position = target_positions.mean(axis=0)
        target_guess = None
        min_distance = float('inf')
        for i in range(4):
            distance = np.linalg.norm(average_position - target_positions[i])
            if distance < min_distance:
                min_distance = distance
                target_guess = target_positions[i]
        print(f"Choosing {target_guess} as the target position.")
        self.show_target_images(all_matched_imgs, target_positions, target_guess)

        start = (self.get_map_coord_x(0), self.get_map_coord_y(0))
        goal = (self.get_map_coord_y(target_guess[1]), self.get_map_coord_x(target_guess[0]))

        self.path = AStar(start, goal, self.occupancy_grid).perform_search()
        if len(self.path) > 0:
            shape = self.occupancy_grid.shape
            self.path_overlay = np.zeros(shape, dtype=np.uint8)
            for (y, x) in self.path:
                self.path_overlay[y, x] = 1
            self.nav_point = self.path.pop(0)
        else:
            print(f"Could not find a path between #{start} and #{goal}")
    

    def convert_occupancy_to_cvimg(self):
        shape = self.occupancy_grid.shape
        height, width = shape
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[self.occupancy_grid == OccupancyMap.VISITED] = [255, 0, 0]
        image[self.occupancy_grid == OccupancyMap.OBSTACLE] = [0, 255, 0]
        image[self.path_overlay == 1] = [0, 0, 255]
        return image


    def decode_filename(self, filename):
        info = filename.split('_')
        x = int(info[1])
        y = int(info[2])
        heading = int(info[3])
        return x, y, heading

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        h, w, _ = fpv.shape
        if self.screen is None:
            self.screen = pygame.display.set_mode((4*w, 2*h))

        step = 0
        phase = None
        state = self.get_state()
        if state is not None:
          step = state[2]
          phase = state[1]

        # Save camera observations (exploration phase only)
        # These can be used for SfM, SLAM, and visual place recognition
        if phase == Phase.EXPLORATION and step % 5 == 1:
            cv2.imwrite(os.path.join(self.filepath, f"{step}_{round(self.x)}_{round(self.y)}_{self.heading}_.png"), fpv)

        # Draw the heads up display (HUD) and the mini-map
        if phase == Phase.EXPLORATION:
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
        minimap_pygame = convert_opencv_img_to_pygame(cv2.resize(minimap, (2*w, 2*h)))
        self.screen.blit(fpv_pygame, (0, 0))
        self.screen.blit(minimap_pygame, (2*w, 50))
        if phase == Phase.EXPLORATION:
            hud_pygame = convert_opencv_img_to_pygame(hud_img, True)
            self.screen.blit(hud_pygame, (2*w, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
