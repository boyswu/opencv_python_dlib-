
from scipy.spatial import distance as dist

from UI import Ui_MainWindow
from PyQt5 import QtWidgets
import dlib
from PyQt5 import QtCore
import cv2
# 摄像头索引、面部标志检测器模型路径和警告声音文件路径
WEBCAM_INDEX = 0
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
ALARM_PATH = 'alarm.wav'
1

def eye_aspect_ratio(eye):
    # 计算眼睛纵横比
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
class EyeTracker (QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(EyeTracker, self).__init__(parent)
        self.setupUi(self)
        self.start.clicked.connect(self.start_tracking)
        self.stop.clicked.connect(self.stop_tracking)
    def start_tracking(self):
        # 打开摄像头
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        # 加载面部标志检测器
        self.detector = cv2.CascadeClassifier(SHAPE_PREDICTOR_PATH)


if __name__ == "__main__":
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

