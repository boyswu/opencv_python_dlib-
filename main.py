import time
from playsound import playsound
from scipy.spatial import distance as dist
from UI import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui
import dlib
import cv2
from imutils import face_utils
import pygame
import imutils

# 摄像头索引、面部标志检测器模型路径和警告声音文件路径
watch_path = 0
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
ALARM_PATH = 'alarm.wav'

# 加载面部检测器
detector = dlib.get_frontal_face_detector()
# 加载面部标志预测器
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)


# 定义眼睛跟踪器类
class EyeTracker(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(EyeTracker, self).__init__(parent)

        self.close_counter = 0
        self.open_counter = 0
        self.setupUi(self)
        self.start.clicked.connect(self.start_tracking)
        self.stop.clicked.connect(self.stop_tracking)
        self.save_img.clicked.connect(self.save_image)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        self.watch.setScaledContents(True)  # 图片自适应大小
        self.print_container = None  # 显示文本框
        self.camera = None  # 视频流对象
        self.monitoring = False  # 监测标志
        self.ear =0.25 # 眼睛纵横比

    def start_tracking(self):
        self.monitoring = True

        fps = 25.0  # 帧率
        # 初始化计数器和计时器
        frame_count = 0
        start_time = time.time()  # 使用time.time()来获取初始时间戳

        # 打开摄像头
        self.camera = cv2.VideoCapture(watch_path)
        while self.monitoring:
            ret, frame = self.camera.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 在每一帧的处理中增加帧计数器
            frame_count += 1
            # 在一定的时间间隔后计算帧率
            if time.time() - start_time >= 1:  # 每秒计算一次
                # 计算帧率
                fps = frame_count / (time.time() - start_time)
                # 重置计数器和计时器
                frame_count = 0
                start_time = time.time()  # 使用time.time()来获取新的时间戳
                # 打印帧率
                # print("FPS:", fps)
            if not ret:
                print("failed to gray frame")
                continue
            else:
                ear = self.monitor_eyes(frame, gray)
                if ear is not None:
                    self.judge_eyes(ear, fps)
                    # print("fps值: ", fps)
                else:
                    continue
        return None

    def stop_tracking(self):
        # 关闭摄像头
        self.camera.release()
        cv2.destroyAllWindows()
        self.monitoring = False
        self.watch.clear()
        self.txt.clear()

    def eye_aspect_ratio(self, eye):
        # 计算眼睛纵横比
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def save_image(self):
        try:
            self.camera = cv2.VideoCapture(watch_path)
            ret, frame = self.camera.read()
            self.camera.release()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            # print("检测到人脸数: ", len(rects))
            self.print_container = "检测到人脸数: " + str(len(rects)) + "\n"
            self.txt.setText(self.print_container)
            if len(rects) > 0:
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
                    self.ear = (left_ear + right_ear) / 2.0
                    self.print_container = str(self.print_container) + "正常情况下的平均眼睛纵横比: " + str(self.ear) + "\n"
                    self.txt.setText(self.print_container)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
            frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                                 int(frame.shape[1]) * 3,
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.watch.setPixmap(QtGui.QPixmap.fromImage(frame))


        except:
            self.camera.release()
            self.save_image()

    def monitor_eyes(self, frame, gray):
        # 在灰度图上检测面部
        rects = detector(gray, 0)
        # print("检测到人脸数: ", len(rects))
        self.print_container = "检测到人脸数: " + str(len(rects)) + "\n"
        self.txt.setText(self.print_container)

        if len(rects) > 0:
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

                container = "左眼纵横比: {:.2f}, 右眼纵横比: {:.2f}, 平均纵横比: {:.2f}".format(left_ear, right_ear,
                                                                                                ear)
                self.print_container = str(self.print_container) + container + "\n"
                self.txt.setText(self.print_container)
                # print("average eye aspect ratio: ", ear)

                # 绘制眼睛区域的凸包
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                # 在图像上绘制眼睛区域的凸包
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                # 显示眼睛纵横比
                cv2.putText(frame, "Left EAR: {:.2f}".format(left_ear), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.putText(frame, "Right EAR: {:.2f}".format(right_ear), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                cv2.putText(frame, "Average EAR: {:.2f}".format(ear), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
                frame = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                                     int(frame.shape[1]) * 3,
                                     QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
                self.watch.setPixmap(QtGui.QPixmap.fromImage(frame))
                cv2.waitKey(1)
                return ear  # 往显示视频的Label里 显示QImage

                # h, w, ch = frame.shape
                # bytesPerLine = ch * w
                # convertToQtFormat = QtGui.QImage(frame.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                # p = convertToQtFormat.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
                # # 将 QImage 显示在 QLabel 控件上
                # self.watch.setPixmap(QtGui.QPixmap.fromImage(p))
                # return ear  # 往显示视频的Label里 显示QImage
        else:
            print("没有检测到人脸")
            self.print_container = "没有检测到人脸" + "\n"
            self.txt.setText(self.print_container)

            return None

    def judge_eyes(self, ear, fps):
        # 眼睛纵横比阈值
        eve_ar_thresh = self.ear * 0.8 # 眼睛闭合阈值
        # 眼睛睁开阈值
        eve_ar_open_thresh = self.ear *0.8
        # # 眼睛闭合持续帧数
        # eve_ar_close_frames = 250 / fps
        # # 眼睛睁开持续帧数
        # eve_ar_open_frames = 250 / fps
        # 眼睛闭合持续帧数
        eve_ar_close_frames = 50
        # 眼睛睁开持续帧数
        eve_ar_open_frames = 100

        # 眼睛微闭判断
        if ear <= eve_ar_thresh:  # 判断眼睛纵横比是否小于阈值
            self.close_counter += 1  # 递增闭合持续帧数计数器
            if self.close_counter >= eve_ar_close_frames:  # 如果持续帧数超过阈值
                pygame.mixer.init()
                pygame.mixer.music.load(ALARM_PATH)
                print("闭眼！",ear)
                pygame.mixer.music.play()
                self.print_container = str(self.print_container) + "警报！" + "\n"  # 显示警报信息

                self.close_counter = 0  # 重置闭合持续帧数计数器
        else:
            self.close_counter = 0  # 如果眼睛并非闭合状态，则重置闭合持续帧数计数器
        # 眼睛睁开判断
        if ear >= eve_ar_open_thresh:  # 判断眼睛纵横比是否大于阈值
            self.open_counter += 1  # 递增睁开持续帧数计数器
            if self.open_counter >= eve_ar_open_frames:  # 如果持续帧数超过阈值
                pygame.mixer.init()
                pygame.mixer.music.load(ALARM_PATH)
                print("睁眼！",ear)
                pygame.mixer.music.play()
                self.print_container = str(self.print_container) + "警报！" + "\n"  # 显示警报信息
                self.open_counter = 0  # 重置睁开持续帧数计数器
        else:
            self.open_counter = 0  # 如果眼睛并非睁开状态，则重置睁开持续帧数计数器

        # print("close_counter: ", self.close_counter)
        # print("open_counter: ", self.open_counter)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = EyeTracker()
    MainWindow.show()
    sys.exit(app.exec_())
