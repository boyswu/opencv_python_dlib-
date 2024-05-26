
from scipy.spatial import distance as dist
from UI import Ui_MainWindow
from PyQt5 import QtWidgets
import dlib
import cv2
from imutils import face_utils
import imutils
from threading import Thread

# 摄像头索引、面部标志检测器模型路径和警告声音文件路径
WEBCAM_INDEX = 0
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
ALARM_PATH = 'alarm.wav'
# 眼睛纵横比阈值
EYE_AR_THRESH = 0.2
# 眼睛长宽比阈值
EYE_AR_CONSEC_FRAMES = 3
# 眼睛闭合阈值
EYE_AR_CLOSE_THRESH = 0.5
# 眼睛睁开阈值
EYE_AR_OPEN_THRESH = 0.3
# 眼睛闭合持续帧数
EYE_AR_CLOSE_FRAMES = 30
# 眼睛睁开持续帧数
EYE_AR_OPEN_FRAMES = 10
# 眼睛闭合持续帧数计数器
EYE_AR_CLOSE_COUNTER = 0
# 眼睛睁开持续帧数计数器
EYE_AR_OPEN_COUNTER = 0

# 定义眼睛跟踪器类
class EyeTracker (QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(EyeTracker, self).__init__(parent)
        self.setupUi(self)
        self.start.clicked.connect(self.start_tracking)
        self.stop.clicked.connect(self.stop_tracking)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    def start_tracking(self):
        # 打开摄像头
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        # 加载面部标志检测器
        self.detector = cv2.CascadeClassifier(SHAPE_PREDICTOR_PATH)
    def stop_tracking(self):
        # 关闭摄像头
        self.cap.release()
        cv2.destroyAllWindows()

    def eye_aspect_ratio(self, eye):
        # 计算眼睛纵横比
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def monitor_eyes(self):
        # 定义警报标志

        while self.monitoring:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 在灰度图上检测面部
            rects = detector(gray, 0)

            for rect in rects:
                # 获取面部关键点坐标
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # 提取左右眼区域
                left_eye = shape[self.lStart:self.lEnd]
                right_eye = shape[self.rStart:self.rEnd]

                # 计算左右眼的纵横比
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)

                # 计算平均纵横比
                ear = (left_ear + right_ear) / 2.0

                # 绘制眼睛区域的凸包
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                # 判断眼睛纵横比是否低于阈值
                if ear < EYE_AR_THRESH:
                    COUNTER += 1

                    # 判断是否连续帧数达到阈值，触发警报
                    if COUNTER >= EYE_AR_CONSE_FRAMES and not ALARM_ON:
                        ALARM_ON = True

                        # 播放警报声音（在新线程中以避免阻塞主线程）
                        if ALARM_PATH:
                            t = Thread(target=sound_alarm, args=(ALARM_PATH,))
                            t.daemon = True
                            t.start()

                        # 在视频帧上添加警告文本
                        cv2.putText(frame, '警告！', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    # 重置计数器和警报标志
                    COUNTER = 0
                    ALARM_ON = False

                # 在视频帧上添加当前眼睛纵横比的文本显示
                cv2.putText(frame, 'ERA:{:2f}'.format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 显示视频帧
            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF

            # 按下 'q' 键停止监测
            if key == ord('q'):
                self.stop_monitoring()

        # 关闭 OpenCV 窗口
        cv2.destroyAllWindows()




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

