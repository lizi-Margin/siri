import math
import threading

from siri.global_config import GloablStatus
from siri.utils.logger import lprint


def move_mouse(*args):
    # print(f"move_mouse called with args: {args}")
    pass


class Operator(threading.Thread):
    def __init__(self, draw_action_hook=None):
        super().__init__()
        self.obs: dict = None
        self.obs_ready_mutex = threading.Semaphore(value=0)
        self.draw_action_hook = draw_action_hook
    
    def run(self):
        lprint(self, "start")
        try:
            while True:
                self.obs_ready_mutex.acquire()
                assert self.obs is not None
                obs = self.obs; self.obs = None

                self.take_action(obs)
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            self.obs = None
        lprint(self, "finish")

    def see_obs(self, obs: dict):
        lprint(self, "see_obs called", debug=True)
        if obs is None:
            obs = {}  # step abort

        assert isinstance(obs, dict)
        last_obs = self.obs; self.obs = obs
        if last_obs is None:
            self.obs_ready_mutex.release()

    def take_action(self, obs: dict):
        assert obs is not None
        if obs == {}:
            return  # BUG, blank obs needed as well

        x, y = obs['xy']
        cls = obs['cls']

        move_x, move_y = self.calc_movement(x, y, cls)
        move_mouse(move_x, move_y)
        data = {
            'xy': (x, y,),
            'move': (move_x, move_y)
        }
        self.draw_action_hook(data)

    def calc_movement(self, target_x, target_y, target_cls):
        monitor = GloablStatus.monitor
        assert monitor is not None
        center_x, center_y = monitor['xy']
        screen_width, screen_height = monitor['width'], monitor['height']
        dpi = 1100
        mouse_sensitivity = 3.0
        fov_x = fov_y = 40

        

        offset_x = target_x - center_x
        offset_y = target_y - center_y
        distance = math.sqrt(offset_x**2 + offset_y**2)
        speed_multiplier = 1.

        degrees_per_pixel_x = fov_x / screen_width
        degrees_per_pixel_y = fov_y / screen_height

        move_x = offset_x * degrees_per_pixel_x
        move_y = offset_y * degrees_per_pixel_y

        move_x = (move_x / 360) * (dpi * (1 / mouse_sensitivity)) * speed_multiplier
        move_y = (move_y / 360) * (dpi * (1 / mouse_sensitivity)) * speed_multiplier

        return move_x, move_y

