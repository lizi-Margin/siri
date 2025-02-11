import os


def get_root_dir():
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.getcwd() + '/'
    return root_dir


class GlobalConfig:
    debug = False

    root_dir = get_root_dir()

    device = 'cuda:0'
    conf_threshold = 0.2
    half = True
    tick = 0.028
    sz_wh = (640, 360,)
    # sz_wh = (640, 640)

    manual_preprocess=False

    body_y_offset = 0.1

    plt = 'qt'

class GloablStatus:
    monitor = None