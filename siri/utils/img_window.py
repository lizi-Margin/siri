import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap


class ImageWindow(QWidget):
    def __init__(self, title):
        super().__init__()
        self.initUI(title)

    def initUI(self, title):
        self.setWindowTitle(title)
        self.label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

    def update_image(self, image_data):
        height, width, channel = image_data.shape
        bytes_per_line = 3 * width
        # q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_BGR888)  # use cv2 BGR format 
        pixmap = QPixmap.fromImage(q_image)
        self.label.setPixmap(pixmap)