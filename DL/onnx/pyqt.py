import sys
import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtCore import QBuffer, QByteArray
from PyQt5.QtGui import QPixmap
import os
import sys
os.chdir(sys.path[0])
import onnxruntime
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageQt
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
sys.path.append('../../')
from model.utils import PrintModelInfo,preprocess_image,min_max_normalize
ONNX_MODEL="./flame_detect.onnx"

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.camera = cv2.VideoCapture(0)  # Open default camera
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.frame = None

    def initUI(self):
        self.setWindowTitle('Flame Detection')
        self.setGeometry(0, 0, 800, 600)

        self.pic_label = QLabel(self)
        self.pic_label.setMinimumHeight(500)
        self.pic_label.setMaximumHeight(500)
        self.pic_label.setStyleSheet('border: 2px solid black; padding: 10px;')

        self.pic_button = QPushButton('Picture', self)
        self.pic_button.clicked.connect(self.load_image)

        self.camera_button = QPushButton('Camera', self)
        self.camera_button.clicked.connect(self.start_camera)

        self.detect_button = QPushButton('Detect', self)
        self.detect_button.clicked.connect(self.detect_flame)

        layout_left = QVBoxLayout()
        layout_left.addWidget(self.pic_label)

        layout_right = QVBoxLayout()
        layout_right.addWidget(self.pic_button)
        layout_right.addWidget(self.camera_button)
        layout_right.addWidget(self.detect_button)

        main_layout = QHBoxLayout()
        main_layout.addLayout(layout_left)
        main_layout.addLayout(layout_right)

        self.setLayout(main_layout)

    def load_image(self):
        image_path = 'path_to_your_image.jpg'  # Replace with your image path
        self.display_image(image_path)

    def start_camera(self):
        if not self.timer.isActive():
            self.timer.start(1000 // 30)  # Update every 30 ms (30 fps)

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            self.frame = frame
            self.display_frame()

    def detect_flame(self):
        if self.frame is not None:
            # Convert OpenCV BGR frame to RGB
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # Convert RGB frame to QImage
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            qimage = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Display QImage
            pixmap = QPixmap.fromImage(qimage)
            pixmap = pixmap.scaled(self.pic_label.size(), Qt.KeepAspectRatio)
            self.pic_label.setPixmap(pixmap)

            # Example: Convert QImage to PIL.Image
            image = self.convert_qimage_to_pil_image(qimage)
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor()
            ])
            image=transform(image)
            input = min_max_normalize(image)
            input=input.unsqueeze(0)
            input_array=np.array(input)
            onnx_model = onnxruntime.InferenceSession(ONNX_MODEL)
            input_name = onnx_model.get_inputs()[0].name
            out = onnx_model.run(None, {input_name:input_array})
            pred=np.argmax(out, axis=-1)
            print(pred)

    def display_image(self, image_path):
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(self.pic_label.size(), Qt.KeepAspectRatio)
        self.pic_label.setPixmap(pixmap)

    def display_frame(self):
        # Convert OpenCV BGR frame to QImage
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame_rgb.shape
        bytes_per_line = 3 * width
        qimage = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Display QImage
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaled(self.pic_label.size(), Qt.KeepAspectRatio)
        self.pic_label.setPixmap(pixmap)

    def convert_qimage_to_pil_image(self, qimage):
        qimage = qimage.convertToFormat(QImage.Format_RGBA8888)
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.constBits()
        ptr.setsize(qimage.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # 4 channels: RGBA
        arr = arr[..., :3]  # Drop the alpha channel
        pil_image = Image.fromarray(arr)
        return pil_image

    def closeEvent(self, event):
        self.timer.stop()
        self.camera.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
