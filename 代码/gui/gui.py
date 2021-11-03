import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import cv2
from net_util import detect_sex, detect_emotion


class MlGui(QMainWindow):
    def __init__(self):
        super(MlGui, self).__init__()
        self.init_UI()
        self.face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
        self.cameraCapture = None
        self.thd = None
        palette = QPalette()
        palette.setColor(QPalette.Background, QColor('#F0F8FF'))
        self.setPalette(palette)

    def closeEvent(self, a0) -> None:
        super(MlGui, self).closeEvent(a0)
        self.release()

    def init_UI(self):
        self.setWindowIcon(QIcon('images/icon.svg'))
        self.setWindowTitle('性别分类与表情识别系统')
        self.resize(1000, 600)
        init_widget = self.get_init_widget()
        self.setCentralWidget(init_widget)

    def get_init_widget(self):
        init_widget = QWidget()
        self.button_image = QPushButton(init_widget)
        self.button_image.setStyleSheet('QPushButton{border-image: url(images/picture.svg)}')
        self.button_image.resize(90, 90)
        self.button_image.move(100, 100)
        self.button_image.clicked.connect(self.load_image)
        self.button_image.setToolTip('加载图片')

        self.button_vedio = QPushButton(init_widget)
        self.button_vedio.setStyleSheet('QPushButton{border-image: url(images/film.svg)}')
        self.button_vedio.resize(90, 90)
        self.button_vedio.move(100, 250)
        self.button_vedio.clicked.connect(lambda: self.load_video('video'))
        self.button_vedio.setToolTip('加载视频')

        self.button_cameras = QPushButton(init_widget)
        self.button_cameras.setStyleSheet('QPushButton{border-image: url(images/camera.svg)}')
        self.button_cameras.resize(90, 90)
        self.button_cameras.move(100, 400)
        self.button_cameras.clicked.connect(lambda: self.load_video('camera'))
        self.button_cameras.setToolTip('打开本地摄像头')

        self.label = QLabel(init_widget)
        self.label.setScaledContents(True)
        self.label.setPixmap(QPixmap('./images/window.jpeg'))
        self.label.resize(600, 400)
        self.label.move(300, 90)

        self.start_button = QPushButton(init_widget)
        self.start_button.resize(90, 90)
        self.start_button.move(460, 500)
        self.start_button.setStyleSheet('QPushButton{border-image: url(images/play.svg)}')
        self.start_button.clicked.connect(self.start)
        self.start_button.setToolTip('播放')

        self.stop_button = QPushButton(init_widget)
        self.stop_button.resize(90, 90)
        self.stop_button.move(570, 500)
        self.stop_button.setStyleSheet('QPushButton{border-image: url(images/stop.svg)}')
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setToolTip('暂停')

        self.cancel_button = QPushButton(init_widget)
        self.cancel_button.resize(80, 80)
        self.cancel_button.move(700, 500)
        self.cancel_button.setStyleSheet('QPushButton{border-image: url(images/clean.svg)}')
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setToolTip('取消')

        self.male_label = QLabel(init_widget)
        self.male_label.setScaledContents(True)
        self.male_label.setPixmap(QPixmap('images/male.svg'))
        self.male_label.resize(50, 50)
        self.male_label.move(400, 30)

        self.female_label = QLabel(init_widget)
        self.female_label.setScaledContents(True)
        self.female_label.resize(50, 50)
        self.female_label.setPixmap(QPixmap('images/female.svg'))
        self.female_label.move(450, 30)

        self.smile_label = QLabel(init_widget)
        self.smile_label.setScaledContents(True)
        self.smile_label.resize(50, 50)
        self.smile_label.setPixmap(QPixmap('images/happy1.svg'))
        self.smile_label.move(550, 30)

        self.norm_label = QLabel(init_widget)
        self.norm_label.setScaledContents(True)
        self.norm_label.resize(50, 50)
        self.norm_label.setPixmap(QPixmap('images/unhappy1.svg'))
        self.norm_label.move(650, 30)

        self.bad_label = QLabel(init_widget)
        self.bad_label.setPixmap(QPixmap('images/sad1.svg'))
        self.bad_label.setScaledContents(True)
        self.bad_label.resize(50, 50)
        self.bad_label.move(750, 30)

        self.help_button = QPushButton(init_widget)
        self.help_button.setStyleSheet('QPushButton{border-image: url(images/prompt.svg)}')
        self.help_button.resize(50, 50)
        self.help_button.move(20, 530)
        self.help_button.clicked.connect(self.prompt_message)
        self.help_button.setToolTip('关于我们')

        self.info_label = QLabel(self)
        self.info_label.resize(280, 50)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setText('性别与表情识别系统')
        self.info_label.setFont(QFont('Microsoft YaHei', 16))

        self.face_label = QLabel(init_widget)
        self.face_label.setScaledContents(True)
        self.face_label.resize(50, 50)
        self.face_label.move(850, 30)
        self.face_label.setPixmap(QPixmap('images/snow.svg'))

        return init_widget

    def prompt_message(self):
        QMessageBox.information(self, '关于', '姓名：刘嘉伟\n学号：2018213106\n运行环境：Pytorch1.7.1', QMessageBox.Ok)

    def release(self):
        if self.cameraCapture is not None:
            self.cameraCapture.release()
        self.label.setPixmap(QPixmap('./images/window.jpeg'))

    def cancel(self):
        self.release()
        if self.thd:
            self.thd.stop()
        self.reload_ui()

    def stop(self):
        if self.thd:
            self.thd.stop()

    def start(self):
        if Thread.flag:
            return
        Thread.flag = 1
        if self.thd:
            self.thd.start()

    def load_image(self):
        self.release()
        fname, _ = QFileDialog.getOpenFileName(self, '打开文件', '.', '图像文件(*.jpg *.png)')
        img = cv2.imread(fname)
        if img is None:
            QMessageBox.critical(self, '错误', '请选择正确文件', QMessageBox.Ok)
            self.reload_ui()
            return
        face = self.detect_face(img)
        self.setPixel(face, self.label)

    def reload_ui(self):
        self.female_label.setVisible(True)
        self.male_label.setVisible(True)
        self.bad_label.setVisible(True)
        self.smile_label.setVisible(True)
        self.norm_label.setVisible(True)
        self.label.setPixmap(QPixmap('./images/window.jpeg'))
        self.face_label.setPixmap(QPixmap('images/snow.svg'))

    def detect_face(self, img):
        global crop
        faces = self.face_cascade.detectMultiScale(img, 1.3, 5)
        # 绘制矩形框
        male_dic = {'male': False, 'female': False}
        emotion_dic = {'happy': False, 'sad': False, 'normal': False}
        color = {'male': (255, 0, 0), 'female': (0, 0, 255)}
        for (x, y, w, h) in faces:
            crop = img[y:y + h, x:x + w]

            # cv2.imwrite('photo/' + str(time.time()) + '.jpg', crop)
            self.setPixel(crop, self.face_label)
            sex = detect_sex(crop)
            emotion = detect_emotion(crop)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color[sex], 2)
            font_size = 1
            img = cv2.putText(img, sex + ',' + emotion, (x, y - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_size,
                              color[sex],
                              2 * font_size,
                              cv2.LINE_AA)
            male_dic[sex] = True
            emotion_dic[emotion] = True
            # print(sex)
            # print(emotion)
        self.male_label.setVisible(male_dic['male'])
        self.female_label.setVisible(male_dic['female'])
        self.smile_label.setVisible(emotion_dic['happy'])
        self.norm_label.setVisible(emotion_dic['normal'])
        self.bad_label.setVisible(emotion_dic['sad'])

        return img

    def load_video(self, check):
        self.release()
        self.reload_ui()
        if check == 'video':
            fname, _ = QFileDialog.getOpenFileName(self, '打开文件', '.', '视频文件(*.mp4 *.avi)')
        else:
            fname = 0
        self.cameraCapture = cv2.VideoCapture(fname)
        if not self.cameraCapture.isOpened():
            QMessageBox.critical(self, '错误', '请选择正确文件', QMessageBox.Ok)
            self.reload_ui()
        self.thd = Thread(self)
        self.thd.changePixmap.connect(self.setPixel)
        self.thd.cap = self.cameraCapture
        self.thd.face_cascade = self.face_cascade
        self.thd.obj = self
        self.thd.start()

    def setPixel(self, img, label=None):
        if label is None:
            label = self.label
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * show.shape[2], QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(showImage))


class Thread(QThread):
    changePixmap = pyqtSignal(np.ndarray)
    flag = 1

    def stop(self):
        Thread.flag = 0

    def run(self):
        Thread.flag = 1
        while Thread.flag and self.cap.isOpened():
            ret, img = self.cap.read()
            if ret:
                # 进行人脸检测
                face = self.obj.detect_face(img)
                self.changePixmap.emit(face)
            else:
                Thread.flag = 0
                break


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MlGui()
    main.show()
    sys.exit(app.exec_())
