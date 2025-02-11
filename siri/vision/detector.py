import cv2
import mss
import time
import torch
import threading
import subprocess
import numpy as np
import supervision as sv
from typing import Union, List

from siri.global_config import GlobalConfig as cfg
from siri.global_config import GloablStatus
from siri.vision.preprocess import preprocess, postprocess
from siri.vision.visualizer import SV_Source
from siri.utils.logger import lprint
from siri.utils.sleeper import Sleeper


class ObsMaker:
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        return self.make_obs(*args, **kwds)

    def make_obs(self, detections: sv.Detections):
        assert isinstance(detections, sv.Detections)
        
        if detections.xyxy.any():
            boxes_array, classes_tensor = self.convert_sv_to_tensor(detections)
            if classes_tensor.numel():
                target = self.find_best_target(boxes_array, classes_tensor)
            
                clss = [0.0, 1.0]
                # if cfg.hideout_targets:
                #     clss.extend([5.0, 6.0])
                # if not cfg.disable_headshot:
                #     clss.append(7.0)
                # if cfg.third_person:
                #     clss.append(10.0)
                
                if target.cls in clss:           
                    x, y = target.x, target.y
                    # x, y = predict_xy(x, y)
                    obs = {
                        'xy': (x, y,),
                        'cls': target.cls
                    }
                    return obs

        return None
    
    def find_best_target(self, boxes_array, classes_tensor):
        assert GloablStatus.monitor is not None
        return self.find_nearest_target_to(
            GloablStatus.monitor['xy'],
            boxes_array,
            classes_tensor
        )
    
    def find_nearest_target_to(self, xy:tuple, boxes_array, classes_tensor,):
        center = torch.tensor([xy[0], xy[1]], device=cfg.device)
        distances_sq = torch.sum((boxes_array[:, :2] - center) ** 2, dim=1)
        # weights = torch.ones_like(distances_sq)

        head_mask = classes_tensor == 7
        if head_mask.any():
            nearest_idx = torch.argmin(distances_sq[head_mask])
            nearest_idx = torch.nonzero(head_mask)[nearest_idx].item()
        else:
            nearest_idx = torch.argmin(distances_sq)

        target_data = boxes_array[nearest_idx, :4].cpu().numpy()
        target_class = classes_tensor[nearest_idx].item()
        """
        names:
            0: player
            1: bot
            2: weapon
            3: outline
            4: dead_body
            5: hideout_target_human
            6: hideout_target_balls
            7: head
            8: smoke
            9: fire
            10: third_person
        """
        return Target(*target_data, target_class)

    @staticmethod
    def convert_sv_to_tensor(frame: sv.Detections):
        assert frame.xyxy.any()
        xyxy = frame.xyxy
        xywh = torch.tensor(np.array(
            [(xyxy[:, 0] + xyxy[:, 2]) / 2,  
             (xyxy[:, 1] + xyxy[:, 3]) / 2,  
             xyxy[:, 2] - xyxy[:, 0],        
             xyxy[:, 3] - xyxy[:, 1]]
        ), dtype=torch.float32).to(cfg.device).T
        
        classes_tensor = torch.from_numpy(np.array(frame.class_id, dtype=np.float32)).to(cfg.device)
        return xywh, classes_tensor


class Target:
    def __init__(self, x, y, w, h, cls):
        self.x = x
        self.y = y if cls == 7 else (y - cfg.body_y_offset * h)
        self.w = w
        self.h = h
        self.cls = cls


class Detector(threading.Thread):
    def __init__(self, model, obs_hook=None, sv_source_hook=None):
        super().__init__()
        from ultralytics import YOLO
        assert isinstance(model, YOLO)
        self.model = model
        self.tracker = sv.ByteTrack()
        self.make_obs = ObsMaker()

        self.obs_hook = obs_hook
        self.sv_source_hook = sv_source_hook
    
    def run(self):
        lprint(self, "start")
        self.start_session()
        lprint(self, "finish")
    
    def start_session(self):
        raise NotImplementedError

    def _predict(self, frame_or_batch: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(frame_or_batch, np.ndarray):
            batch = [frame_or_batch]
        elif isinstance(frame_or_batch, list):
            assert isinstance(frame_or_batch[0], np.ndarray)
            batch = frame_or_batch
        else:
            assert False

        assert len(batch[0].shape) == 3
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # frame = pad(frame, to_sz_wh=cfg.sz_wh)
        # frame = cv2.resize(frame, cfg.sz_wh)
        if cfg.manual_preprocess: 
            batch = preprocess(batch)

        results = self.model.predict(
            batch,
            cfg=f"{cfg.root_dir}/siri/vision/game.yaml",
            imgsz=tuple(reversed(cfg.sz_wh)),
            stream=True,
            conf=cfg.conf_threshold,
            iou=0.5,
            device=cfg.device,
            half=cfg.half,
            max_det=20,
            agnostic_nms=False,
            augment=False,
            vid_stride=False,
            visualize=False,
            verbose=False,
            show_boxes=False,
            show_labels=False,
            show_conf=False,
            save=False,
            show=False,
            # batch=1
        )

        return results

    def predict_and_plot(self, frame):
        results = self._predict(frame)
        for result in results:
            result_frame = result.plot()
            cv2.imshow("tmp", result_frame)

            while True:
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
    
    def predict_and_make_obs(self, frame):
        results = self._predict(frame)
        # print(results)
        if isinstance(results, list):
            if len(results) > 0:
                result = results[0]
            else:
                lprint(self, "no result were returned by the model")
                return
        else:
            try:
                result = next(results)
            except StopIteration:
                lprint(self, "no result were returned by the model")
                return

        sv_detections = sv.Detections.from_ultralytics(result)
        if self.sv_source_hook is not None:
            self.sv_source_hook(SV_Source(frame, sv_detections))

        sv_detections = self.tracker.update_with_detections(sv_detections)
        obs = self.make_obs(sv_detections)
        if self.obs_hook is not None:
            self.obs_hook(obs)


class ScrDetector(Detector):
    @staticmethod
    def get_scrcpy_window_geometry(window_keyword='Phone'):
        result = subprocess.run(
            ['wmctrl', '-lG'],
            stdout=subprocess.PIPE,
            text=True
        )
        lines = result.stdout.splitlines()
        for line in lines:
            if window_keyword in line:
                parts = line.split()
                x, y = int(parts[2]), int(parts[3])
                scr_width, scr_height = int(parts[4]), int(parts[5])
                return x, y, scr_width, scr_height
        return None

    def start_session(self):
        geometry = self.get_scrcpy_window_geometry()
        if not geometry:
            lprint(self, "scrcpy window not found")
            lprint(self, "start_session failed")
            return

        assert GloablStatus.monitor is None
        left, top, scr_width, scr_height = geometry
        center_xy = (
            left + scr_width/2,
            top + scr_height/2,
        )

        GloablStatus.monitor = {
            "top": top,
            "left": left,
            "width": scr_width,
            "height": scr_height,
            "xy": center_xy
        }
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(
        #     'output.mp4',
        #     fourcc,
        #     1/cfg.tick,
        #     cfg.sz_wh if cfg.manual_preprocess else (scr_width, scr_height,)
        # )

        try:
            with mss.mss() as sct:
                while True:
                    sleeper = Sleeper(user=self)
                    screenshot = sct.grab(GloablStatus.monitor)
                    frame = np.array(screenshot)
                    if frame.shape[-1] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    assert frame.shape[-1] == 3, 'not a BGR format'
                    # print(frame.shape)

                    # Predict
                    self.predict_and_make_obs(frame)
                    # results = self._predict(frame)
                    # if isinstance(results, list):
                    #     assert len(results) == 1
                    #     result = results[0]
                    # else:
                    #     result = next(results)
                    
                    # # Output
                    # result_frame = result.plot()

                    # if cfg.manual_preprocess:
                    #     result_frame = postprocess(result_frame)

                    # # out.write(result_frame)
                    # cv2.imshow("tmp", result_frame)

                    # if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    #     break
                    sleeper.sleep()
        except KeyboardInterrupt:
            lprint(self, "Sig INT catched, stopping session.")
        finally:
            # cv2.destroyAllWindows()
            GloablStatus.monitor = None

