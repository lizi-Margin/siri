import time
from ultralytics import YOLO
from siri.global_config import GlobalConfig as cfg
from siri.vision.detector import ScrDetector
from siri.vision.visualizer import Visualizer
from siri.strategy.operator import Operator
from siri.utils.logger import print_obj


def start():
    print_obj(cfg)
    print('- - ' * 10)
    time.sleep(1)

    model = YOLO(f"{cfg.root_dir}/model/fe1/yolo11m/200/weights/best.pt", task='detect')
    visualizer = Visualizer()
    operator = Operator(draw_action_hook=visualizer.draw_action)
    detector = ScrDetector(model, obs_hook=operator.see_obs, sv_source_hook=visualizer.draw_sv_source)

    visualizer.start()
    time.sleep(0.1)
    operator.start()
    time.sleep(0.1)
    # detector.start()
    detector.start_session()
    

if __name__ == '__main__':
    start()
