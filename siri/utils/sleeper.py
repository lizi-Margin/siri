import time
from .logger import lprint
from siri.global_config import GlobalConfig as cfg


class Sleeper:
    def __init__(self, tick=None, user=None):
        self._start = time.time_ns()
        self.tick = tick if tick is not None else cfg.tick
        self.user=user
    
    def sleep(self):
        sleep_time = self.tick - (time.time_ns() - self._start)/1e9
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            buffer = 'warning: tick time out'
            if self.user is not None:
                buffer += f", caller is {self.user.__class__.__name__}"
            lprint(self, buffer)