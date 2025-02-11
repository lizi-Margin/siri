import os
from siri.global_config import GlobalConfig as cfg

def get_action(yolo_result):
    enemy_box = yolo_result.boxes[0].xyxy  
    x1, y1, x2, y2 = enemy_box
    enemy_center_x = (x1 + x2) / 2
    enemy_center_y = (y1 + y2) / 2
    
    frame_width, frame_height = cfg.sz_wh
    screen_center_x = frame_width / 2
    screen_center_y = frame_height / 2
    
    offset_x = enemy_center_x - screen_center_x
    offset_y = enemy_center_y - screen_center_y

    # 方法一：使用 pyautogui
    # pyautogui.moveTo(screen_center_x, screen_center_y)
    # pyautogui.mouseDown()
    # pyautogui.moveRel(offset_x, offset_y, duration=0.05)
    # pyautogui.mouseUp()

    # 或方法二：使用 adb 命令模拟触摸滑动
    start_x = int(screen_center_x)
    start_y = int(screen_center_y)
    end_x = int(screen_center_x + offset_x)
    end_y = int(screen_center_y + offset_y)
    swipe_duration = cfg.tick * 1000  # ms
    os.system(f"adb shell input swipe {start_x} {start_y} {end_x} {end_y} {swipe_duration}")
