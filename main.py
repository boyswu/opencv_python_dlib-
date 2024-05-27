
from playsound import playsound
from scipy.spatial import distance as dist
from UI import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui,QtCore
import dlib
import cv2
from imutils import face_utils
import imutils

# 摄像头索引、面部标志检测器模型路径和警告声音文件路径
watch_path = 0
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
ALARM_PATH = 'alarm.wav'


detector = dlib.get_frontal_face_detector()
# 加载面部标志检测器
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)


# 定义眼睛跟踪器类
class EyeTracker(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(EyeTracker, self).__init__(parent)
        self.setupUi(self)
        self.start.clicked.connect(self.start_tracking)
        self.stop.clicked.connect(self.stop_tracking)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        self.camera = None  # 视频流对象
        self.monitoring = False  # 监测标志

    def start_tracking(self):
        self.monitoring = True
        # self.camera = VideoStream(src=watch_path, usePiCamera=False).start()
        # 打开摄像头
        self.camera = cv2.VideoCapture(watch_path)
        while self.monitoring:
            ret, frame = self.camera.read()
            frame = imutils.resize(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('1.jpg', frame)
            cv2.imwrite('2.jpg', gray)
            if not ret:
                print("failed to grab frame")
                return
            else:
                # 在灰度图上检测面部
                rects = detector(gray, 0)
                ear = self.monitor_eyes(frame, gray, rects)
                self.judge_eyes(ear)

    def stop_tracking(self):
        # 关闭摄像头
        self.camera.release()
        cv2.destroyAllWindows()
        self.monitoring = False

    def eye_aspect_ratio(self, eye):
        # 计算眼睛纵横比
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def monitor_eyes(self,frame, gray, rects):
        # 监测眼睛
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
            # 绘制眼睛凸包
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

            # # 显示眼睛纵横比
            # cv2.putText(frame, "Left EAR: {:.2f}".format(left_ear), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            #             (0, 0, 255), 2)
            # cv2.putText(frame, "Right EAR: {:.2f}".format(right_ear), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            #             (0, 0, 255), 2)
            # cv2.putText(frame, "Average EAR: {:.2f}".format(ear), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            #             (0, 0, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
            frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                                         int(frame.shape[1]) * 3,
                                         QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.watch.setPixmap(QtGui.QPixmap.fromImage(frame))
            return ear # 往显示视频的Label里 显示QImage

            # 将 OpenCV 图像帧转换为 QImage 格式
            # h, w, ch = frame.shape
            # bytesPerLine = ch * w
            # convertToQtFormat = QtGui.QImage(frame.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            # p = convertToQtFormat.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
            # # 将 QImage 显示在 QLabel 控件上
            # self.watch.setPixmap(QtGui.QPixmap.fromImage(p))
    def judge_eyes(self, ear):
        # 眼睛纵横比阈值
        EYE_AR_THRESH = 0.2
        # 眼睛长宽比阈值
        EYE_AR_CONSEC_FRAMES = 3
        # 眼睛闭合阈值
        EYE_AR_CLOSE_THRESH = 0.5
        # 眼睛睁开阈值
        EYE_AR_OPEN_THRESH = 0.3
        # 眼睛闭合持续帧数
        EYE_AR_CLOSE_FRAMES = 5
        # 眼睛睁开持续帧数
        EYE_AR_OPEN_FRAMES = 20

        close_counter = 0  # 眼睛闭合持续帧数计数器

        open_counter = 0  # 眼睛睁开持续帧数计数器
        # 眼睛闭合判断
        if ear < EYE_AR_THRESH:
            close_counter += 1
            if close_counter >= EYE_AR_CLOSE_FRAMES:
                # 警报
                playsound(ALARM_PATH)
                close_counter = 0
        else:
            close_counter = 0

        # 眼睛睁开判断
        if ear > EYE_AR_OPEN_THRESH:
            open_counter += 1
            if open_counter >= EYE_AR_OPEN_FRAMES:
                # 警报
                playsound(ALARM_PATH)
                open_counter = 0
        else:
            open_counter = 0


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = EyeTracker()
    MainWindow.show()
    sys.exit(app.exec_())