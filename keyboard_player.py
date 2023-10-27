from vis_nav_game import Player, Action, Phase
import pygame
import numpy as np
import cv2
import math
from time import sleep, strftime
import os


def convert_opencv_img_to_pygame(opencv_image):
    """
    Convert OpenCV images for Pygame.

    see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
    """
    opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
    shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
    pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

    return pygame_image

def convert_exploration_graph_to_pygame(exploration_graph):    
    shape = exploration_graph.shape[1::-1]
    pygame_image = pygame.image.frombuffer(exploration_graph, shape, 'RGB')

    return pygame_image

class KeyboardPlayerPyGame(Player):
    """
    This is a copy of the example player provided by the professor.
    Control manually using the keyboard.
    """

    def __init__(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.other_keymap = None
        self.filepath = ''

        self.x = 0
        self.y = 0
        self.heading = 0

        self.exploration_graph = np.zeros(shape=(300, 300, 3), dtype=np.uint8)

        for i in range(300):
            self.exploration_graph[0][i] = [0, 0, 255]
            self.exploration_graph[299][i] = [0, 0, 255]
            self.exploration_graph[i][0] = [0, 0, 255]
            self.exploration_graph[i][299] = [0, 0, 255]

        super(KeyboardPlayerPyGame, self).__init__()

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

        self.other_keymap = {
            pygame.K_w: "North wall",
            pygame.K_a: "West wall",
            pygame.K_s: "South wall",
            pygame.K_d: "East wall",
            pygame.K_n: "Reset true north",
        }

        self.filepath = './data/exploration_views/' + strftime("%Y%m%d-%H%M%S")
        os.makedirs(self.filepath, exist_ok=True)

    def pre_exploration(self):
        self.x = 0
        self.y = 0
        self.heading = 0
        self.last_act = Action.IDLE

    def pre_navigation(self):
        self.x = 0
        self.y = 0
        self.heading = 0
        self.last_act = Action.IDLE

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act = self.keymap[event.key]
                elif event.key in self.other_keymap:
                    if self.other_keymap[event.key] == "North wall":
                        for i in range(300):
                            self.exploration_graph[150 - round(self.y) - 1][i] = [0, 255, 0]
                    elif self.other_keymap[event.key] == "South wall":
                        for i in range(300):
                            self.exploration_graph[150 - round(self.y) + 1][i] = [0, 255, 0]
                    elif self.other_keymap[event.key] == "West wall":
                        for i in range(300):
                            self.exploration_graph[i][150 + round(self.x) - 1] = [0, 255, 0]
                    elif self.other_keymap[event.key] == "East wall":
                        for i in range(300):
                            self.exploration_graph[i][150 + round(self.x) + 1] = [0, 255, 0]
                    elif self.other_keymap[event.key] == "Reset true north":
                        self.heading = 0
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act = Action.IDLE
        
        if self.last_act == Action.FORWARD:
            converted_heading = self.heading / 147 * 2 * math.pi
            self.x += math.sin(converted_heading)
            self.y += math.cos(converted_heading)
        elif self.last_act == Action.BACKWARD:
            converted_heading = self.heading / 147 * 2 * math.pi
            self.x -= math.sin(converted_heading)
            self.y -= math.cos(converted_heading)
        elif self.last_act == Action.LEFT:
            self.heading = (self.heading - 1) % 147
        elif self.last_act == Action.RIGHT:
            self.heading = (self.heading + 1) % 147

        self.exploration_graph[150 - round(self.y)][150 + round(self.x)] = [255, 0, 0]

        sleep(0.01)
        return self.last_act

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
        minimap = np.copy(self.exploration_graph)
        marker_size = 10
        marker_img = np.zeros(shape=(marker_size, marker_size, 3), dtype=np.uint8)
        cv2.drawMarker(marker_img, (marker_size // 2, marker_size // 2), [0, 255, 0], cv2.MARKER_TRIANGLE_UP, marker_size)
        rotation_matrix = cv2.getRotationMatrix2D((marker_img.shape[1] // 2, marker_img.shape[0] // 2), (self.heading / 147 * -360), 1)
        rotated_marker = cv2.warpAffine(marker_img, rotation_matrix, (marker_img.shape[1], marker_img.shape[0]))
        x = 150 + round(self.x) - marker_size // 2
        y = 150 - round(self.y) - marker_size // 2
        minimap[y:y+rotated_marker.shape[0], x:x+rotated_marker.shape[1]] = rotated_marker

        # Display everything
        pygame.display.set_caption(f"{self.__class__.__name__}:fpv; h: {h} w:{w}; step{step}")
        fpv_doubled = cv2.resize(fpv, (2*w, 2*h))
        fpv_pygame = convert_opencv_img_to_pygame(fpv_doubled)
        hud_pygame = convert_opencv_img_to_pygame(hud_img)
        minimap_pygame = convert_exploration_graph_to_pygame(minimap)
        self.screen.blit(fpv_pygame, (0, 0))
        self.screen.blit(hud_pygame, (2*w, 0))
        self.screen.blit(minimap_pygame, (2*w+50, 50))
        pygame.display.update()

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
