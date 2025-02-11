import sys
import cv2
import threading
import numpy as np
import supervision as sv
from PyQt5.QtWidgets import QApplication
from matplotlib import pyplot as plt

from siri.utils.sleeper import Sleeper
from siri.utils.logger import lprint, print_obj
from siri.global_config import GloablStatus
from siri.global_config import GlobalConfig as cfg
from siri.utils.img_window import ImageWindow
from siri.vision.preprocess import to_int


class SV_Source:
    def __init__(self, frame: np.ndarray, detections: sv.Detections):
        assert frame is not None
        assert detections is not None
        self.frame = frame
        self.detections = detections


class Visualizer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.draw_mutex = threading.Semaphore(value=0)

        self.sv_source_queue: list[SV_Source] = []
        self.action_data = None

        self.plt = cfg.plt
            
    def run(self):
        lprint(self, "start")
        title = f"{self.__class__.__name__}"
        try:
            while True:
                # self.draw_mutex.acquire()

                sleeper = Sleeper(user=self)            
                if len(self.sv_source_queue) == 0:
                    sleeper.sleep()
                    continue 

                sv_source = self.sv_source_queue.pop(0); self.sv_source_queue = []
                action_data = self.action_data; self.action_data = None

                frame = sv_source.frame
                detections = sv_source.detections

                xy = None
                if action_data is not None:
                    xy = action_data['xy']

                annotated_frame = self.plot(frame, detections, target_xy=xy)

                if self.plt == 'plt':
                    if not hasattr(self, 'img'):
                        self.fig, ax = plt.subplots()
                        ax.set_title(title)
                        self.img = ax.imshow(annotated_frame)
                        ax.axis('off')
                        plt.ion()
                        plt.show()
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    self.img.set_data(annotated_frame)
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                elif self.plt == 'cv2':
                    cv2.imshow(title, annotated_frame)
                    cv2.waitKey(1)
                elif self.plt == 'qt':
                    if not hasattr(self, 'img'):
                        # BUG, WARNING: QApplication was not created in the main() thread.
                        self.app = QApplication(sys.argv)
                        self.img = ImageWindow(title=title)
                        self.img.show()
                    self.img.update_image(annotated_frame)
                    self.app.processEvents()
                else:
                    raise NotImplementedError
                sleeper.sleep()
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            self.sv_source_queue = []
            self.action_data = None
        lprint(self, "finish")

    def draw_sv_source(self, sv_source: SV_Source):
        lprint(self, "draw_sv_source called", debug=True)
        assert isinstance(sv_source, SV_Source)
        self.sv_source_queue.append(sv_source)
    
    def draw_action(self, data: dict):
        lprint(self, "draw_action called", debug=True)
        assert isinstance(data, dict) and data is not None
        last_action_data = self.action_data; self.action_data = data
        if last_action_data is None:
            self.draw_mutex.release()
    
    def plot(self, frame: np.ndarray, detections, target_xy=None):
        assert isinstance(frame, np.ndarray)
        assert detections is not None
        
        # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections)
        if target_xy is not None:
            target_xy = to_int(target_xy)
            annotated_frame = cv2.circle(
                img=annotated_frame,
                center=target_xy,
                radius=8,
                color=(0, 0, 0,),  # color BGR
                thickness=-1
            )

        return annotated_frame