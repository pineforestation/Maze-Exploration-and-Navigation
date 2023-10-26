from enum import IntEnum
from vis_nav_game import Player, Action, Phase
import pygame
import numpy as np
import cv2
from time import sleep, strftime
import os

class OccupancyMap(IntEnum):
    UNEXPLORED = 0
    VISITED = 1
    OBSTACLE = 2

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
    """
    Convert OpenCV images for Pygame.

    see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
    """
    # opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB

    color_mapping = {
        OccupancyMap.UNEXPLORED: (0, 0, 0),
        OccupancyMap.VISITED: (255, 0, 0),
        OccupancyMap.OBSTACLE: (0, 255, 255)
    }
    
    shape = exploration_graph.shape
    height, width = shape
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            image[i, j] = color_mapping[exploration_graph[i, j]]

    pygame_image = pygame.image.frombuffer(image, shape, 'RGB')

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
        self.filepath = ''

        self.x = 0
        self.y = 0
        self.heading = 0

        self.exploration_graph = np.zeros(shape=(200,200), dtype=np.uint8)

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
            pygame.K_ESCAPE: Action.QUIT
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
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act = Action.IDLE
        
        if self.last_act == Action.FORWARD:
            self.x += 1
        elif self.last_act == Action.BACKWARD:
            self.x -= 1
        elif self.last_act == Action.LEFT:
            self.heading = (self.heading - 1) % 147
        elif self.last_act == Action.RIGHT:
            self.heading = (self.heading + 1) % 147

        self.exploration_graph[100 - round(self.x)][100 - round(self.y)] = OccupancyMap.VISITED

        # sleep(0.01)
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
            self.screen = pygame.display.set_mode((2*w, h))

        step = 0
        phase = None
        state = self.get_state()
        if state is not None:
          step = state[2]
          phase = state[1]

        if phase == Phase.EXPLORATION and step % 5 == 1:
            cv2.imwrite(os.path.join(self.filepath, f"{step}_{self.x}_{self.y}_{self.heading}.png"), fpv)

        blank_img = np.zeros(shape=(h,w,3), dtype=np.uint8)
        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1
        color = (255, 255, 255)

        cv2.putText(blank_img, f"x={self.x}, y=0, heading={self.heading}", (h_offset, w_offset), font, size, color, stroke, line)

        pygame.display.set_caption(f"{self.__class__.__name__}:fpv; h: {h} w:{w}; step{step}")
        rgb = convert_opencv_img_to_pygame(fpv)
        minimap = convert_opencv_img_to_pygame(blank_img)
        minimap2 = convert_exploration_graph_to_pygame(self.exploration_graph)
        self.screen.blit(rgb, (0, 0))
        self.screen.blit(minimap, (w, 0))
        self.screen.blit(minimap2, (w+50, 50))
        pygame.display.update()

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
