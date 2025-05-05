import random
import sys
import threading
import time
from collections import deque
import logging
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QGridLayout, \
    QPushButton, QMenu, QAction, QDialog, QLabel, QProgressBar, QFrame, QFileDialog, QMessageBox, QProgressDialog
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QImage
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from random import randint
from ultralytics import YOLO
from face_emotion_detect import Config
import face_emotion_detect.detect_tools as tools
from threading import Thread
from PyQt5.QtCore import QThread, pyqtSignal
import os
from stt_emotion import VideoAudioAnalyzer

SOUND_NAMES = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy',
               4: 'neutral', 5: 'other', 6: 'sad', 7: 'surprised', 8: 'unk'}
SOUND_COLORS = ['#FF6B6B', '#C77DFF', '#FFD166', '#06D6A0',
                '#118AB2', '#5E72E4', '#FF9F1C', '#EF476F', '#83C5BE']

# Text 配置
TEXT_NAMES = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy',
              4: 'neutral', 5: 'sadness', 6: 'shame', 7: 'surprise'}
TEXT_COLORS = ['#FF6B6B', '#C77DFF', '#FFD166', '#06D6A0',
               '#118AB2', '#5E72E4', '#FF9F1C', '#EF476F']
class StudentWidget(QLabel):
    def __init__(self, student_id, parent=None):
        super().__init__(parent)
        self.student_id = student_id
        self.setFixedSize(240, 180)  # 每个学生区域放大1.5倍
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: lightblue; margin: 5px; border: 1px solid gray;")

        self.file_path = None
        self.video_capture = None  # 用于播放视频
        self.timer = None  # 用于定时更新视频帧
        self.setPixmap(self.create_placeholder_image())

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

        # 加载模型
        self.face_model = YOLO('./face_emotion_detect/models/yolov8n-face.pt', task='detect')
        self.expression_model = YOLO('./face_emotion_detect/models/expression_cls.pt', task='classify')

    def create_placeholder_image(self):
        pixmap = QPixmap(180, 135)  # 每个学生区域大小放大1.5倍
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawRect(0, 0, 180, 135)

        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "Unlinked students")
        painter.end()
        return pixmap

    def show_context_menu(self, pos):
        context_menu = QMenu(self)

        link_student_action = QAction('link Student', self)
        link_student_action.triggered.connect(self.link_student)
        context_menu.addAction(link_student_action)

        remove_action = QAction('Remove Student', self)
        remove_action.triggered.connect(self.remove_student)
        context_menu.addAction(remove_action)

        analyze_action = QAction('Status Analysis', self)
        analyze_action.triggered.connect(self.show_emotion_analysis)
        context_menu.addAction(analyze_action)

        context_menu.exec_(self.mapToGlobal(pos))

    def link_student(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("图片或视频文件 (*.png *.jpg *.jpeg *.mp4 *.avi)")
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.file_path = selected_files[0]
                if self.file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.setPixmap(self.load_and_scale_image(self.file_path))
                elif self.file_path.lower().endswith(('.mp4', '.avi')):
                    self.setPixmap(self.create_video_placeholder())
                    self.video_capture = cv2.VideoCapture(self.file_path)
                    self.timer = QTimer(self)
                    self.timer.timeout.connect(self.update_video_frame)
                    self.timer.start(30)  # 每30毫秒更新一次

    def load_and_scale_image(self, file_path):
        pixmap = QPixmap(file_path)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)  # 保持比例缩放
        return scaled_pixmap

    def create_video_placeholder(self):
        pixmap = QPixmap(180, 135)
        pixmap.fill(Qt.gray)
        painter = QPainter(pixmap)

        # 创建新的字体对象，并设置为粗体
        font = QFont()
        font.setBold(True)
        painter.setFont(font)

        painter.setPen(QColor(255, 255, 255))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "视频文件")
        painter.end()
        return pixmap

    def update_video_frame(self):
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                self.current_frame = frame.copy()  # 保存当前帧
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QPixmap.fromImage(QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888))
                self.setPixmap(q_image.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))



    def remove_student(self):
        """移出学生并清理相关资源"""
        if self.file_path:
            # 如果是视频文件，停止播放并释放资源
            if self.file_path.lower().endswith(('.mp4', '.avi')):
                if self.timer:
                    self.timer.stop()
                if self.video_capture and self.video_capture.isOpened():
                    self.video_capture.release()
                self.video_capture = None
                self.timer = None
                self.setPixmap(self.create_placeholder_image())
            # 如果是图片文件，恢复为占位图
            elif self.file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.setPixmap(self.create_placeholder_image())

        # 移除学生展示
        parent_window = self.parent()
        if isinstance(parent_window, EmotionMonitoringUI):
            parent_window.remove_student(self.student_id)

    def show_emotion_analysis(self):
        """启动情感分析窗口，并更新学生图像"""
        emotion_window = EmotionDetailWindow(self.student_id, self)
        emotion_window.exec_()



    def update_student_image(self, processed_img):
        """接收处理后的图像并更新显示"""
        if processed_img is not None:
            # 确保将 BGR 转换为 RGB
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

            # 转换为 QPixmap 格式显示
            height, width, channel = processed_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(processed_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def face_detect(self, image):
        # 进行人脸检测，并截取人脸图片
        image = image.copy()
        results = self.face_model(image, conf=0.5)
        faces = []
        face_locations = []
        if len(results[0].boxes.data):
            face_locations_float = results[0].boxes.xyxy.tolist()
            for each in face_locations_float:
                face_locations.append(list(map(int, each)))
            for face_location in face_locations:
                left, top, right, bottom = face_location
                faces.append(image[top:bottom, left:right])
                image = cv2.rectangle(image, (left, top), (right, bottom), (50, 50, 250), 3)
            return image, faces, face_locations
        else:
            return image, None, None

    def analyze_face_emotions(self, image):
        """检测人脸并进行表情分析"""
        # 进行人脸检测
        face_cvimg, faces, locations = self.face_detect(image)
        if faces is not None:
            emotion_counts = [0] * len(Config.names)  # 创建情感计数器，数量与表情种类相同
            for i in range(len(faces)):
                left, top, right, bottom = locations[i]

                # 将彩色图片转换为灰度图
                img = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                # 将灰度图转换为3通道图像
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # 进行表情识别
                rec_res = self.expression_model(img)
                probs = rec_res[0].probs.data.tolist()  # 获取每个情感类别的概率
                num = np.argmax(probs)  # 获取最大概率对应的情感标签
                label = Config.names[num]  # 获取情感标签
                emotion_counts[num] += probs[num]  # 使用概率值来更新情感计数（而不是简单计数）

                # 在人脸区域绘制表情标签
                face_cvimg = cv2.putText(face_cvimg, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                         (0, 0, 250), 2, cv2.LINE_AA)
                # 在人脸框周围绘制矩形
                face_cvimg = cv2.rectangle(face_cvimg, (left, top), (right, bottom), (50, 50, 250), 3)  # 画出人脸框

            return face_cvimg, emotion_counts  # 返回处理后的图像和情感计数（概率）
        return image, None  # 如果没有检测到人脸，则返回原图和空的情感计数

    def analyze_face_emotions_counts(self, frame):
        """检测视频帧中的人脸并进行表情分析"""
        # 进行人脸检测
        result = self.face_detect(frame)
        if len(result) != 3:
            return frame, None  # 如果返回值不符合预期，返回原帧和None

        face_cvimg, faces, locations = result

        if faces is not None and locations is not None:
            emotion_counts = [0] * len(Config.names)  # 创建情感计数器，数量与表情种类相同
            for i in range(len(faces)):
                left, top, right, bottom = locations[i]

                # 将彩色图片转换为灰度图
                img = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                # 将灰度图转换为3通道图像
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # 进行表情识别
                rec_res = self.expression_model(img)
                probs = rec_res[0].probs.data.tolist()  # 获取每个情感类别的概率
                num = np.argmax(probs)  # 获取最大概率对应的情感标签
                label = Config.names[num]  # 获取情感标签
                emotion_counts[num] += probs[num]  # 使用概率值来更新情感计数（而不是简单计数）

                # 在人脸区域绘制表情标签
                face_cvimg = cv2.putText(face_cvimg, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                         (0, 0, 250), 2, cv2.LINE_AA)
                # 在人脸框周围绘制矩形
                face_cvimg = cv2.rectangle(face_cvimg, (left, top), (right, bottom), (50, 50, 250), 3)  # 画出人脸框

            return face_cvimg, emotion_counts  # 返回处理后的图像和情感计数（概率）
        return frame, None  # 如果没有检测到人脸，则返回原帧和空的情感计数



    def analyze_video_frame_emotions(self, frame):
        """检测视频帧中的人脸并进行表情分析"""
        # 进行人脸检测
        face_cvimg, faces, locations = self.face_detect(frame)
        if faces is not None:
            emotion_counts = [0] * len(Config.names)  # 创建情感计数器，数量与表情种类相同
            for i in range(len(faces)):
                left, top, right, bottom = locations[i]

                # 将彩色图片转换为灰度图
                img = cv2.cvtColor(faces[i], cv2.COLOR_BGR2GRAY)
                # 将灰度图转换为3通道图像
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # 进行表情识别
                rec_res = self.expression_model(img)
                probs = rec_res[0].probs.data.tolist()  # 获取每个情感类别的概率
                num = np.argmax(probs)  # 获取最大概率对应的情感标签
                label = Config.names[num]  # 获取情感标签
                emotion_counts[num] += probs[num]  # 使用概率值来更新情感计数（而不是简单计数）
                if not emotion_counts or any(not np.isfinite(x) for x in emotion_counts):
                    return None, [0.0] * len(Config.names)

                # 确保总和为1（概率分布）
                total = sum(emotion_counts)
                if total > 0:
                    emotion_counts = [x / total for x in emotion_counts]
                else:
                    emotion_counts = [1.0 / len(Config.names)] * len(Config.names)
                # 在人脸区域绘制表情标签
                face_cvimg = cv2.putText(face_cvimg, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                         (0, 0, 250), 2, cv2.LINE_AA)
                # 在人脸框周围绘制矩形
                face_cvimg = cv2.rectangle(face_cvimg, (left, top), (right, bottom), (50, 50, 250), 3)  # 画出人脸框

            return face_cvimg, emotion_counts  # 返回处理后的图像和情感计数（概率）
        return frame, None  # 如果没有检测到人脸，则返回原帧和空的情感计数

    def get_current_frame(self):
        """获取当前显示的帧（用于视频或静态图片）"""
        if self.file_path:
            if self.file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 如果是图片，直接返回读取的图像
                frame = cv2.imread(self.file_path)
                if frame is not None:
                    return frame
            elif self.file_path.lower().endswith(('.mp4', '.avi')):
                # 如果是视频，返回当前播放的帧
                if hasattr(self, 'current_frame') and self.current_frame is not None:
                    return self.current_frame
        return None

class EmotionPieChart(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(5, 4))
        super().__init__(fig)
        self.setParent(parent)
        self.update_pie_chart([1] * 7)  # 初始化均匀分布的饼图

    def update_pie_chart(self, emotion_counts=None):
        """更新饼图数据，使用实例自身的emotion_names和colors"""
        # 优先使用实例自身的配置，如果没有则使用Config默认配置
        emotions = list(getattr(self, 'emotion_names', Config.names).values())
        colors = getattr(self, 'colors', Config.colors)

        # 确保颜色数量与标签匹配
        if len(colors) < len(emotions):
            colors = colors * (len(emotions) // len(colors) + 1)
            colors = colors[:len(emotions)]

        if emotion_counts is None:
            emotion_counts = [1] * len(emotions)
        elif len(emotion_counts) != len(emotions):
            raise ValueError(f"情感数据维度不匹配: 预期 {len(emotions)} 维，得到 {len(emotion_counts)} 维")

        # 计算百分比
        total = sum(emotion_counts)
        percentages = [count / total * 100 if total > 0 else 0 for count in emotion_counts]

        # 清除旧图形
        self.ax.clear()

        # 绘制饼图（保持原有样式）
        wedges, texts, autotexts = self.ax.pie(
            percentages,
            labels=[emotion if p >= 3 else "" for emotion, p in zip(emotions, percentages)],
            autopct=lambda p: f'{p:.1f}%' if p >= 5 else '',
            startangle=90,
            colors=colors[:len(emotions)],  # 确保颜色数量匹配
            wedgeprops={'width': 0.6, 'edgecolor': 'white', 'linewidth': 0.5},
            textprops={'fontsize': 8, 'color': 'white', 'weight': 'bold'}
        )

        # 调整标签样式
        plt.setp(texts, size=9, color='black')
        self.ax.axis('equal')
        self.draw()


class EmotionLineChart(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(8, 6))
        super().__init__(fig)
        self.setParent(parent)

        # 初始化数据存储
        self.time_data = deque(maxlen=100)
        self.emotion_data = {emotion: deque(maxlen=100) for emotion in Config.names.values()}

        # 初始化线条
        self.lines = {}
        colors = Config.colors
        for i, emotion in enumerate(Config.names.values()):
            self.lines[emotion], = self.ax.plot(
                [], [],
                color=colors[i],
                linewidth=1.5,
                alpha=0.7,
                label=''
            )

        # 设置图表样式
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 100)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#f5f5f5')
        self.ax.legend(loc='upper right')
        self.draw()

    def update_data(self, emotion_values):
        """更新折线图数据"""
        # 确保输入是列表格式
        if isinstance(emotion_values, dict):
            emotion_values = [emotion_values.get(emotion, 0) for emotion in Config.names.values()]

        # 添加时间点
        self.time_data.append(len(self.time_data) + 1)

        # 更新每种情绪的数据
        for i, emotion in enumerate(Config.names.values()):
            self.emotion_data[emotion].append(emotion_values[i])

            # 更新线条数据
            self.lines[emotion].set_data(
                range(len(self.emotion_data[emotion])),
                list(self.emotion_data[emotion])
            )

        # 调整X轴范围
        if len(self.time_data) > 50:
            self.ax.set_xlim(
                len(self.time_data) - 50,
                len(self.time_data) + 10
            )

        self.draw()

    def clear_chart(self):
        """清空图表"""
        self.time_data.clear()
        for emotion in self.emotion_data:
            self.emotion_data[emotion].clear()

        for line in self.lines.values():
            line.set_data([], [])

        self.ax.set_xlim(0, 100)
        self.draw()


class EmotionMonitoringUI(QWidget):
    def __init__(self, switch_page):
        super().__init__()
        self.setWindowTitle('Classroom Emotion Monitoring System')
        self.setGeometry(100, 100, 1800, 800)
        self.switch_page = switch_page
        self.students = {}
        self.next_student_id = 0
        self.sound_pie_chart = None
        self.text_pie_chart = None
        # 定义配置为实例变量
        self.SOUND_NAMES = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy',
                            4: 'neutral', 5: 'other', 6: 'sad', 7: 'surprised', 8: 'unk'}
        self.SOUND_COLORS = ['#FF6B6B', '#C77DFF', '#FFD166', '#06D6A0',
                             '#118AB2', '#5E72E4', '#FF9F1C', '#EF476F', '#83C5BE']

        self.TEXT_NAMES = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy',
                           4: 'neutral', 5: 'sadness', 6: 'shame', 7: 'surprise'}
        self.TEXT_COLORS = ['#FF6B6B', '#C77DFF', '#FFD166', '#06D6A0',
                            '#118AB2', '#5E72E4', '#FF9F1C', '#EF476F']

        # 初始化数据
        self.time_data = []
        self.emotion_data = {emotion: [] for emotion in
                             ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']}

        # 主布局
        main_layout = QHBoxLayout()
        self.create_face_recognition_area(main_layout)

        # 右侧布局
        right_layout = QVBoxLayout()
        self.create_pie_chart_area(right_layout)
        self.create_color_indicator_area(right_layout)
        self.create_line_chart_area(right_layout)

        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def create_face_recognition_area(self, layout):
        face_recognition_layout = QVBoxLayout()
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.face_frame = QWidget()
        self.face_layout = QGridLayout()
        self.face_frame.setLayout(self.face_layout)

        self.scroll_area.setWidget(self.face_frame)
        face_recognition_layout.addWidget(self.scroll_area)
        self.create_buttons_area(face_recognition_layout)

        layout.addLayout(face_recognition_layout)

        # 默认添加一些座位
        for _ in range(15):
            self.add_seat()

    def add_seat(self):
        """添加一个座位，即在界面中添加一个新的学生展示区域"""
        student_id = self.next_student_id
        student_widget = StudentWidget(student_id, self)
        self.students[student_id] = student_widget

        rows, cols = 2, 3
        row = self.next_student_id // cols
        col = self.next_student_id % cols
        self.face_layout.addWidget(student_widget, row, col)
        self.next_student_id += 1

    def analyze_all_videos(self):
        """分析所有学生的视频文件，计算情感得分平均值并更新饼图"""
        try:
            # 获取所有学生的视频文件路径
            video_paths = []
            for student_id, student_widget in self.students.items():
                if student_widget.file_path and student_widget.file_path.lower().endswith(('.mp4', '.avi')):
                    video_paths.append(student_widget.file_path)

            if not video_paths:
                QMessageBox.warning(self, "Warning", "No video files found to analyze")
                return

            # 创建进度对话框
            progress = QProgressDialog("Analyzing video...", "Cancel", 0, len(video_paths), self)
            progress.setWindowTitle("Analysis progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # 初始化情感得分累加器
            total_audio_emotion = {emotion: 0 for emotion in self.SOUND_NAMES.values()}
            total_text_emotion = {emotion: 0 for emotion in self.TEXT_NAMES.values()}
            processed_count = 0

            # 创建分析器实例
            analyzer = VideoAudioAnalyzer()

            for i, video_path in enumerate(video_paths):
                if progress.wasCanceled():
                    break

                progress.setValue(i)
                QApplication.processEvents()  # 更新UI

                try:
                    # 分析当前视频
                    results = analyzer.analyze_video(video_path)

                    # 累加音频情感得分
                    for emotion, score in results["audio_emotion"].items():
                        if emotion in total_audio_emotion:
                            total_audio_emotion[emotion] += score

                    # 累加文本情感得分
                    for emotion, score in results["text_emotion"].items():
                        if emotion in total_text_emotion:
                            total_text_emotion[emotion] += score

                    processed_count += 1
                except Exception as e:
                    logging.error(f"Analyze Video {video_path} Failed: {e}")
                    continue

            progress.close()

            if processed_count == 0:
                QMessageBox.warning(self, "Warning", "No videos were successfully analyzed")
                return

            # 计算平均值
            avg_audio_emotion = {emotion: total / processed_count for emotion, total in total_audio_emotion.items()}
            avg_text_emotion = {emotion: total / processed_count for emotion, total in total_text_emotion.items()}

            # 更新sound饼图
            sound_values = [avg_audio_emotion.get(emotion, 0) for emotion in self.SOUND_NAMES.values()]
            self.sound_pie_chart.update_pie_chart(sound_values)

            # 更新text饼图
            text_values = [avg_text_emotion.get(emotion, 0) for emotion in self.TEXT_NAMES.values()]
            self.text_pie_chart.update_pie_chart(text_values)

            QMessageBox.information(self, "Finished", f" successfully analyzed {processed_count}/{len(video_paths)} Videos")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Parsing failed: {str(e)}")
    def create_pie_chart_area(self, layout):
        """创建包含三个水平排列饼图的区域"""
        pie_charts_container = QWidget(self)
        pie_charts_layout = QHBoxLayout(pie_charts_container)

        # 1. Visual 饼图
        self.create_single_pie_chart(pie_charts_layout, "Real-time visual sentiment", "visual")

        # 2. Sound 饼图
        sound_frame = QWidget()
        sound_frame.setFixedSize(350, 350)
        sound_button = QPushButton("Real-time sound sentiment", sound_frame)
        sound_button.clicked.connect(self.analyze_all_videos)  # 修改为调用分析所有视频的函数
        sound_button.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")

        sound_pie_layout = QVBoxLayout(sound_frame)
        sound_pie_layout.addWidget(sound_button)
        sound_pie_layout.addSpacing(10)

        self.sound_pie_chart = EmotionPieChart(sound_frame)
        self.sound_pie_chart.emotion_names = self.SOUND_NAMES  # 使用实例变量
        self.sound_pie_chart.colors = self.SOUND_COLORS
        sound_pie_layout.addWidget(self.sound_pie_chart)
        self.sound_pie_chart.update_pie_chart([1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.sound_pie_chart.setMinimumSize(300, 300)
        pie_charts_layout.addWidget(sound_frame)

        # 3. Text 饼图
        text_frame = QWidget()
        text_frame.setFixedSize(350, 350)
        text_button = QPushButton("Real-time text sentiment", text_frame)
        text_button.clicked.connect(self.analyze_all_videos)
        text_button.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")

        text_pie_layout = QVBoxLayout(text_frame)
        text_pie_layout.addWidget(text_button)
        text_pie_layout.addSpacing(10)

        self.text_pie_chart = EmotionPieChart(text_frame)
        self.text_pie_chart.emotion_names = self.TEXT_NAMES  # 使用实例变量
        self.text_pie_chart.colors = self.TEXT_COLORS
        self.text_pie_chart.update_pie_chart([1, 1, 1, 1, 1, 1, 1, 1])
        text_pie_layout.addWidget(self.text_pie_chart)
        self.text_pie_chart.setMinimumSize(300, 300)
        pie_charts_layout.addWidget(text_frame)

        layout.addWidget(pie_charts_container)




    def create_single_pie_chart(self, parent_layout, button_text, chart_type):
        """Helper method to create a single pie chart frame"""
        frame = QWidget()
        frame.setFixedSize(350, 350)

        # Create button
        button = QPushButton(button_text, frame)
        button.clicked.connect(lambda: self.show_pie_chart(chart_type))
        button.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")

        # Create pie chart layout
        pie_layout = QVBoxLayout(frame)
        pie_layout.addWidget(button)
        pie_layout.addSpacing(10)

        # Create and add pie chart
        pie_chart = EmotionPieChart(frame)
        pie_layout.addWidget(pie_chart)
        pie_chart.setMinimumSize(300, 300)

        # Store references based on chart type
        if chart_type == "visual":
            self.visual_pie_chart_frame = frame
            self.visual_pie_chart = pie_chart
        # elif chart_type == "sound":
        #     self.sound_pie_chart_frame = frame
        #     self.sound_pie_chart = pie_chart
        # elif chart_type == "text":
        #     self.text_pie_chart_frame = frame
        #     self.text_pie_chart = pie_chart

        parent_layout.addWidget(frame)

    def show_pie_chart(self, chart_type):
        """启动分析线程"""
        # 停止现有线程
        if hasattr(self, f'{chart_type}_analysis_thread'):
            getattr(self, f'{chart_type}_analysis_thread').stop()

        # 创建并启动新线程
        thread = VideoAnalysisThread(self.students, chart_type)
        thread.frame_processed.connect(self.update_realtime_chart)
        thread.analysis_finished.connect(self.update_final_chart)
        setattr(self, f'{chart_type}_analysis_thread', thread)
        thread.start()
        # self._show_loading_indicator(chart_type)

    def update_realtime_chart(self, frame_data, chart_type):
        """更新实时图表"""
        pie_chart = getattr(self, f'{chart_type}_pie_chart')
        pie_chart.update_pie_chart(list(frame_data.values()))

    def update_final_chart(self, values, chart_type):
        """更新最终结果"""
        pie_chart = getattr(self, f'{chart_type}_pie_chart')
        pie_chart.update_pie_chart(values)

    def _update_pie_chart_from_thread(self, values, chart_type):
        """接收线程结果并更新UI"""
        try:
            pie_chart = getattr(self, f'{chart_type}_pie_chart')
            pie_chart.update_pie_chart(values)
        except Exception as e:
            print(f"Error updating {chart_type} chart: {str(e)}")
            uniform = 1.0 / len(Config.names)
            pie_chart.update_pie_chart([uniform] * len(Config.names))

        # 隐藏加载状态
        self._hide_loading_indicator(chart_type)

    def _show_loading_indicator(self, chart_type):
        """显示分析中的加载状态"""
        frame = getattr(self, f'{chart_type}_pie_chart_frame')
        if not hasattr(frame, 'loading_label'):
            from PyQt5.QtWidgets import QLabel
            frame.loading_label = QLabel("Analyzing...", frame)
            frame.loading_label.setStyleSheet("font-size: 14px; color: gray;")
            frame.layout().addWidget(frame.loading_label)

    def _hide_loading_indicator(self, chart_type):
        """隐藏加载状态"""
        frame = getattr(self, f'{chart_type}_pie_chart_frame')
        if hasattr(frame, 'loading_label'):
            frame.loading_label.deleteLater()
            del frame.loading_label


    # line chart create&&update
    def create_line_chart_area(self, layout):
        """创建包含三种图表的区域"""
        charts_container = QWidget(self)
        charts_layout = QHBoxLayout(charts_container)

        # 1. Visual 折线图
        self.create_single_line_chart(charts_layout, "Visual sentiment change", "visual")

        # 2. Sound 柱状图
        sound_frame = QWidget()
        sound_frame.setFixedSize(350, 350)
        sound_button = QPushButton("Sound sentiment", sound_frame)
        sound_button.clicked.connect(lambda: self.show_emotion_chart("sound"))
        sound_button.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")

        sound_chart_layout = QVBoxLayout(sound_frame)
        sound_chart_layout.addWidget(sound_button)

        self.sound_bar_chart = EmotionBarChart(sound_frame, chart_type="sound")
        sound_chart_layout.addWidget(self.sound_bar_chart)
        charts_layout.addWidget(sound_frame)

        # 3. Text 柱状图
        text_frame = QWidget()
        text_frame.setFixedSize(350, 350)
        text_button = QPushButton("Text sentiment", text_frame)
        text_button.clicked.connect(lambda: self.show_emotion_chart("text"))
        text_button.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")

        text_chart_layout = QVBoxLayout(text_frame)
        text_chart_layout.addWidget(text_button)

        self.text_bar_chart = EmotionBarChart(text_frame, chart_type="text")
        text_chart_layout.addWidget(self.text_bar_chart)
        charts_layout.addWidget(text_frame)

        layout.addWidget(charts_container)

    def show_emotion_chart(self, chart_type):
        """显示情感图表（柱状图或折线图），分析所有视频文件并计算平均情感值"""
        # 创建进度对话框
        progress = QProgressDialog(f"Analyzing...{chart_type}emotion...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        try:
            QApplication.processEvents()  # 更新UI

            # 初始化分析器
            progress.setLabelText("Initialize analyzer...")
            progress.setValue(10)
            analyzer = VideoAudioAnalyzer()

            # 获取视频文件列表
            progress.setLabelText("Scan video...")
            progress.setValue(20)
            video_paths = [sw.file_path for sw in self.students.values()
                           if sw.file_path and sw.file_path.lower().endswith(('.mp4', '.avi'))]

            if not video_paths:
                progress.close()
                QMessageBox.warning(self, "Warning", "No video files found to analyze")
                return

            # 初始化累加器
            total_audio_emotion = {emotion: 0.0 for emotion in self.SOUND_NAMES.values()}
            total_text_emotion = {emotion: 0.0 for emotion in self.TEXT_NAMES.values()}
            valid_videos = 0
            total_videos = len(video_paths)

            # 分析每个视频
            for i, video_path in enumerate(video_paths):
                if progress.wasCanceled():
                    break

                progress.setLabelText(f"Analyze Video {i + 1}/{total_videos}...")
                progress.setValue(30 + int(i * 60 / total_videos))
                QApplication.processEvents()

                try:
                    # 更新状态信息
                    status_msg = f"Processing: {os.path.basename(video_path)}"
                    progress.setLabelText(status_msg)

                    # 分析视频
                    results = analyzer.analyze_video(video_path)
                    valid_videos += 1

                    # 累加音频情感值
                    for emotion, value in results["audio_emotion"].items():
                        if emotion in total_audio_emotion:
                            total_audio_emotion[emotion] += value

                    # 累加文本情感值
                    for emotion, value in results["text_emotion"].items():
                        if emotion in total_text_emotion:
                            total_text_emotion[emotion] += value

                except Exception as e:
                    print(f"Analyze Video {video_path} Failed: {str(e)}")
                    continue

            if progress.wasCanceled():
                progress.close()
                QMessageBox.information(self, "message", "Analysis canceled")
                return

            if valid_videos == 0:
                progress.close()
                QMessageBox.warning(self, "Warning", "All video analysis failed")
                return

            # 计算平均值
            progress.setLabelText("Calculation results...")
            progress.setValue(90)
            avg_audio_emotion = {emotion: value / valid_videos for emotion, value in total_audio_emotion.items()}
            avg_text_emotion = {emotion: value / valid_videos for emotion, value in total_text_emotion.items()}

            # 更新图表
            progress.setLabelText("Updating chart...")
            progress.setValue(95)
            if chart_type == "sound":
                sound_values = [avg_audio_emotion.get(emotion, 0) * 100 for emotion in self.SOUND_NAMES.values()]
                self.sound_bar_chart.update_bar_chart(sound_values)
            else:  # text
                text_values = [avg_text_emotion.get(emotion, 0) * 100 for emotion in self.TEXT_NAMES.values()]
                self.text_bar_chart.update_bar_chart(text_values)

            progress.setValue(100)
            progress.close()

            # 显示完成提示
            QMessageBox.information(self, "Complete", f"{chart_type}Sentiment analysis completed！\nTotally {valid_videos} Videos")

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"{chart_type}Analysis failed: {str(e)}")



    def create_single_line_chart(self, parent_layout, button_text, chart_type):
        """创建单个折线图框架"""
        frame = QWidget()
        frame.setFixedSize(350, 350)  # 稍微增大尺寸以容纳更多曲线

        # 创建按钮
        button = QPushButton(button_text, frame)
        button.clicked.connect(lambda: self.show_emotion_change_graph(chart_type))
        button.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")

        # 创建折线图布局
        line_layout = QVBoxLayout(frame)
        line_layout.addWidget(button)

        # 创建并添加折线图
        line_chart = EmotionLineChart(frame)
        line_layout.addWidget(line_chart)

        # 存储引用
        setattr(self, f'{chart_type}_line_chart', line_chart)
        parent_layout.addWidget(frame)

    def hide_legend(self):
        """Hide the legend from the pie chart"""
        if hasattr(self, 'ax'):
            self.ax.legend().set_visible(False)
        self.draw()

    def show_emotion_change_graph(self, chart_type):
        """启动情感变化图表"""
        line_chart = getattr(self, f'{chart_type}_line_chart')
        line_chart.clear_chart()

        # 创建并启动分析线程
        thread = EmotionAnalysisThread(
            students=self.students,
            chart_type=chart_type
        )
        thread.frame_processed.connect(
            lambda data, ct: self._update_line_chart(data, ct) if ct == chart_type else None
        )

        # 存储线程引用
        setattr(self, f'{chart_type}_analysis_thread', thread)
        thread.start()

    def _update_line_chart(self, data, chart_type):
        """更新折线图数据"""
        line_chart = getattr(self, f'{chart_type}_line_chart')

        # 确保数据是字典格式
        if not isinstance(data, dict):
            data = {emotion: data[i] if i < len(data) else 0
                    for i, emotion in enumerate(Config.names.values())}

        line_chart.update_data(data)

    def create_buttons_area(self, layout):
        button_layout = QHBoxLayout()

        add_seat_button = QPushButton('Add Seat', self)
        add_seat_button.clicked.connect(self.add_seat)
        button_layout.addWidget(add_seat_button)

        emotion_monitoring_button = QPushButton('Emotion Monitoring', self)
        emotion_monitoring_button.clicked.connect(self.emotion_monitoring_placeholder)
        button_layout.addWidget(emotion_monitoring_button)


        end_monitoring_button = QPushButton('End', self)
        end_monitoring_button.clicked.connect(self.end_monitoring_placeholder)
        button_layout.addWidget(end_monitoring_button)

        button_layout.setStretch(0, 1)
        button_layout.setStretch(1, 1)
        button_layout.setStretch(2, 1)
        button_layout.setStretch(3, 1)
        button_layout.setStretch(4, 1)

        layout.addLayout(button_layout)


    def emotion_monitoring_placeholder(self):
        """检测所有不为空的座位的图片或视频，并显示每一类处理的结果"""
        emotion_data = {emotion: 0 for emotion in Config.names}
        total_students = 0

        for student_id, student_widget in self.students.items():
            file_path = student_widget.file_path
            if file_path:
                total_students += 1

                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    cv_img = tools.img_cvread(file_path)
                    if cv_img is not None:
                        processed_img, emotion_counts = student_widget.analyze_face_emotions(cv_img)
                        if emotion_counts:
                            for i, emotion in enumerate(emotion_data):
                                emotion_data[emotion] += emotion_counts[i]
                        if processed_img is not None:
                            # Convert BGR to RGB
                            # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                            student_widget.update_student_image(processed_img)

                elif file_path.lower().endswith(('.mp4', '.avi')):
                    # Check if the video is already being processed
                    if not hasattr(student_widget,
                                   'processing_thread') or not student_widget.processing_thread.is_alive():
                        # Stop the existing video timer
                        if student_widget.timer and student_widget.timer.isActive():
                            student_widget.timer.stop()

                        # Start a new thread for video processing
                        student_widget.processing_thread = Thread(target=self.process_video,
                                                                  args=(student_widget, emotion_data))
                        student_widget.processing_thread.start()

        if total_students > 0:
            for emotion in emotion_data:
                emotion_data[emotion] /= total_students



    def process_video(self, student_widget, emotion_data):
        """Process video frames in a separate thread"""
        cap = cv2.VideoCapture(student_widget.file_path)
        student_widget.stop_processing = False  # Flag to control the processing loop

        while cap.isOpened() and not student_widget.stop_processing:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame, emotion_counts = student_widget.analyze_video_frame_emotions(frame)
            if emotion_counts:
                for i, emotion in enumerate(emotion_data):
                    emotion_data[emotion] += emotion_counts[i]
            if processed_frame is not None:
                # Convert BGR to RGB
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                # Convert the processed frame to QPixmap and update the display
                height, width, channel = processed_frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(processed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                student_widget.setPixmap(
                    pixmap.scaled(student_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

            cv2.waitKey(30)  # Control the frame rate
        cap.release()

        # Restart the video timer after processing
        if student_widget.timer:
            student_widget.timer.start(30)

    def end_monitoring_placeholder(self):
        """结束所有正在进行的情感监控并恢复原始视频播放"""
        for student_id, student_widget in self.students.items():
            # Signal the processing thread to stop
            if hasattr(student_widget, 'processing_thread') and student_widget.processing_thread.is_alive():
                student_widget.stop_processing = True  # Set the flag to stop processing
                student_widget.processing_thread.join(timeout=1.0)  # Wait for the thread to finish

            # Stop the video timer if it is active
            if student_widget.timer and student_widget.timer.isActive():
                student_widget.timer.stop()

            # Reset the display to the original video
            file_path = student_widget.file_path
            if file_path and file_path.lower().endswith(('.mp4', '.avi')):
                # Ensure the video capture is properly reset
                if hasattr(student_widget, 'video_capture'):
                    student_widget.video_capture.release()

                student_widget.video_capture = cv2.VideoCapture(file_path)
                student_widget.timer = QTimer(student_widget)
                student_widget.timer.timeout.connect(student_widget.update_video_frame)
                student_widget.timer.start(30)  # Resume playing the original video

            elif file_path and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                student_widget.setPixmap(student_widget.load_and_scale_image(file_path))

        print("所有监控已结束，恢复原始视频播放")

    def remove_student(self, student_id):
        student_widget = self.students.pop(student_id, None)
        if student_widget:
            student_widget.deleteLater()

    def create_color_indicator_area(self, layout):
        """Create a horizontal color indicator bar showing emotion-color mapping in English"""
        color_indicator = QWidget()
        color_layout = QHBoxLayout(color_indicator)  # 使用水平布局
        color_layout.setContentsMargins(80, 10, 0, 10)
        color_layout.setSpacing(20)  # 设置元素间距

        # 使用英文标签 (从Config.names字典中按顺序获取值)
        emotion_names = [Config.names[i] for i in range(len(Config.names))]

        # 为每个情感创建颜色标签和文字标签
        for emotion, color in zip(emotion_names, Config.colors):
            # 创建单个情感指示器的小容器
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(5)

            # 颜色方块
            color_label = QLabel()
            color_label.setFixedSize(50, 20)
            color_label.setStyleSheet(f"""
                background-color: {color}; 
                border: 1px solid black;
                border-radius: 3px;
            """)

            # 英文标签
            text_label = QLabel(emotion)
            text_label.setStyleSheet("""
                font-size: 12px; 
                min-width: 60px;
            """)

            # 添加到布局
            item_layout.addWidget(color_label)
            item_layout.addWidget(text_label)

            # 添加到主布局
            color_layout.addWidget(item_widget)

        # 添加弹性空间使内容居中
        color_layout.addStretch()

        # 添加到主布局
        layout.addWidget(color_indicator)




    def show_error_message(self, message):
        """显示错误信息的弹框"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("错误")
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()


class EmotionAnalysisThread(QThread):
    frame_processed = pyqtSignal(dict, str)  # 发送字典格式数据

    def __init__(self, students, chart_type):
        super().__init__()
        self.students = students
        self.chart_type = chart_type
        self.running = False

    def run(self):
        self.running = True
        try:
            while self.running:
                # 收集当前帧的情绪数据
                current_data = self._collect_emotion_data()
                if current_data:
                    # 转换为百分比并发送
                    processed_data = {
                        emotion: current_data[i] * 100
                        for i, emotion in enumerate(Config.names.values())
                    }
                    self.frame_processed.emit(processed_data, self.chart_type)

                time.sleep(0.1)  # 控制刷新频率

        except Exception as e:
            print(f"Error in analysis thread: {str(e)}")
        finally:
            self.running = False

    def _collect_emotion_data(self):
        """收集所有学生的情绪数据"""
        try:
            emotion_counts = [0.0] * len(Config.names)
            valid_students = 0

            for student_widget in self.students.values():
                if not student_widget.file_path:
                    continue

                # 根据类型分析数据
                counts = None
                if self.chart_type == "visual":
                    _, counts = student_widget.analyze_face_emotions_counts(
                        student_widget.get_current_frame()
                    )
                # 其他类型的分析...

                if counts and len(counts) == len(Config.names):
                    valid_students += 1
                    for i in range(len(counts)):
                        emotion_counts[i] += counts[i]

            if valid_students > 0:
                return [c / valid_students for c in emotion_counts]
            return None

        except Exception as e:
            print(f"Data collection error: {str(e)}")
            return None

    def stop(self):
        """安全停止线程"""
        self.running = False
        self.wait(2000)

class VideoAnalysisThread(QThread):
    frame_processed = pyqtSignal(dict, str)  # 实时帧数据信号
    analysis_finished = pyqtSignal(list, str)  # 最终结果信号

    def __init__(self, students, chart_type):
        super().__init__()
        self.students = students
        self.chart_type = chart_type
        self.running = False  # 初始状态为False
        self.lock = threading.Lock()  # 添加线程锁

    def run(self):
        self.running = True
        emotion_data = {emotion: 0.0 for emotion in Config.names}
        total_frames = 0

        try:
            for student_id, student_widget in self.students.items():
                if not self.running:
                    break

                if not student_widget.file_path:
                    continue

                file_path = student_widget.file_path.lower()
                if not file_path.endswith(('.mp4', '.avi')):
                    continue

                cap = None
                try:
                    cap = cv2.VideoCapture(student_widget.file_path)
                    if not cap.isOpened():
                        print(f"无法打开视频文件: {student_widget.file_path}")
                        continue

                    while self.running:
                        with self.lock:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            # 分析当前帧
                            counts = self.process_frame(student_widget, frame)
                            if counts is None:
                                continue

                            total_frames += 1
                            # 更新数据并发送信号
                            self.update_and_emit(emotion_data, counts, total_frames)

                except Exception as e:
                    print(f"处理视频时出错 {student_widget.file_path}: {str(e)}")
                finally:
                    if cap is not None:
                        cap.release()

            # 发送最终结果
            if self.running:
                self.emit_final_result(emotion_data, total_frames)

        except Exception as e:
            print(f"线程运行出错: {str(e)}")
        finally:
            self.running = False





    def process_frame(self, student_widget, frame):
        """处理单帧并返回计数"""
        try:
            if self.chart_type == "visual":
                _, counts = student_widget.analyze_face_emotions_counts(frame)
            # elif self.chart_type == "sound":
            #     counts = student_widget.analyze_sound_emotions(frame)
            # elif self.chart_type == "text":
            #     counts = student_widget.analyze_text_emotions(frame)
            # else:
            #     return None

                if counts and len(counts) == len(Config.names):
                    return counts
            return None
        except Exception as e:
            print(f"帧处理出错: {str(e)}")
            return None

    def update_and_emit(self, emotion_data, counts, total_frames):
        """更新数据并发送信号"""
        # 更新累计数据
        for i, emotion in enumerate(emotion_data):
            emotion_data[emotion] += counts[i]

        # 发送实时数据
        frame_data = {
            emotion: counts[i] * 100  # 转换为百分比
            for i, emotion in enumerate(Config.names.values())
        }
        self.frame_processed.emit(frame_data, self.chart_type)

        # 发送当前平均值
        current_avg = [emotion_data[e]/total_frames for e in Config.names]
        self.frame_processed.emit(
            {e: v*100 for e, v in zip(Config.names.values(), current_avg)},
            self.chart_type
        )

    def emit_final_result(self, emotion_data, total_frames):
        """发送最终结果"""
        if total_frames > 0:
            values = [emotion_data[e]/total_frames for e in Config.names]
        else:
            values = [1.0/len(Config.names)] * len(Config.names)
        self.analysis_finished.emit(values, self.chart_type)

    def stop(self):
        """安全停止线程"""
        with self.lock:
            self.running = False
        self.wait(2000)  # 等待2秒


class EmotionDetailWindow(QDialog):
    def __init__(self, student_id, parent_widget):
        super().__init__()
        self.setWindowTitle(f"Sentiment Analysis - Student {student_id}")
        self.setGeometry(200, 200, 400, 500)

        # 主布局
        layout = QVBoxLayout()

        # 创建进度条展示区域
        self.create_progress_frame(layout)

        # 设置窗口布局
        self.setLayout(layout)

        # 获取学生图片并进行情感分析
        #

        self.parent_widget = parent_widget
        self.is_video = parent_widget.file_path.lower().endswith(('.mp4', '.avi'))  # 检查是否为视频
        self.analyze_emotions(parent_widget)


    def create_progress_frame(self, layout):
        """创建进度条展示区域"""
        self.progress_frame = QFrame(self)
        self.progress_frame.setStyleSheet("background-color: lightgray;")
        self.progress_frame.setGeometry(10, 50, 380, 400)

        # 添加情绪进度条
        progress_layout = QVBoxLayout()

        # 情绪类别
        self.progress_bars = {}
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Focus', 'Sad', 'Surprised']

        for emotion in emotions:
            label = QLabel(emotion, self.progress_frame)
            progress_bar = QProgressBar(self.progress_frame)
            progress_bar.setMaximum(100)
            progress_bar.setTextVisible(True)
            progress_bar.setValue(0)

            progress_layout.addWidget(label)
            progress_layout.addWidget(progress_bar)
            self.progress_bars[emotion] = progress_bar

        self.progress_frame.setLayout(progress_layout)



    def analyze_emotions(self, parent_widget):
        """分析情感并更新进度条（支持视频处理）"""
        file_path = parent_widget.file_path

        if not file_path or file_path.strip() == "":
            self.show_error_message("请上传图片或视频")
            return
        print("type is ", self.is_video)
        if self.is_video:
            self.process_video(file_path, parent_widget)
        else:
            return self.process_image(file_path, parent_widget)

    def process_image(self, file_path, parent_widget):
        """处理单张图片进行情感分析"""
        cv_img = tools.img_cvread(file_path)

        if cv_img is None:
            self.show_error_message("无法读取图片，请检查文件格式或路径")
            return None  # 如果读取失败，返回 None

        face_cvimg, emotion_counts = parent_widget.analyze_face_emotions(cv_img)

        if emotion_counts is not None:
            for i, emotion in enumerate(self.progress_bars.keys()):
                self.progress_bars[emotion].setValue(int(emotion_counts[i] * 100))  # 更新进度条

        # 返回处理后的图像
        # cv2.imshow("img",face_cvimg)
        # cv2.waitKey(0)
        return face_cvimg  # 返回图像

    def process_video(self, file_path, parent_widget):
        """视频处理并计算平均情感值"""
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        emotion_sums = [0] * len(self.progress_bars)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 如果没有读取到帧，退出

            # 进行情感分析
            face_cvimg, emotion_counts = parent_widget.analyze_video_frame_emotions(frame)

            if emotion_counts is not None:
                # 累加每个情感类别的值
                for i, emotion in enumerate(emotion_counts):
                    emotion_sums[i] += emotion

        cap.release()  # 释放视频文件

        # 计算平均情感值
        if total_frames > 0:
            average_emotions = [s / total_frames for s in emotion_sums]
        else:
            average_emotions = emotion_sums

        # 更新进度条为平均值
        for i, emotion in enumerate(self.progress_bars.keys()):
            self.progress_bars[emotion].setValue(int(average_emotions[i] * 100))  # 更新进度条

        # 视频处理完成，显示进度页面
        # self.show_progress_page(average_emotions)


class EmotionBarChart(FigureCanvas):
    def __init__(self, parent=None, chart_type="sound"):
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        super().__init__(self.fig)
        self.setParent(parent)

        # 根据图表类型设置不同的配置
        if chart_type == "sound":
            self.emotion_names = {0: 'angry', 1: 'disgusted', 2: 'fearful', 3: 'happy',
                                  4: 'neutral', 5: 'other', 6: 'sad', 7: 'surprised', 8: 'unk'}
            self.colors = ['#FF6B6B', '#C77DFF', '#FFD166', '#06D6A0',
                           '#118AB2', '#5E72E4', '#FF9F1C', '#EF476F', '#83C5BE']
        else:  # text
            self.emotion_names = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy',
                                  4: 'neutral', 5: 'sadness', 6: 'shame', 7: 'surprise'}
            self.colors = ['#FF6B6B', '#C77DFF', '#FFD166', '#06D6A0',
                           '#118AB2', '#5E72E4', '#FF9F1C', '#EF476F']

        # 初始化空柱状图
        self.update_bar_chart([0] * len(self.emotion_names))

    def update_bar_chart(self, emotion_values):
        """更新柱状图数据"""
        self.ax.clear()

        # 确保数据长度与情感类别数量一致
        if len(emotion_values) != len(self.emotion_names):
            emotion_values = [0] * len(self.emotion_names)

        # 创建柱状图
        bars = self.ax.bar(
            range(len(self.emotion_names)),
            emotion_values,
            color=self.colors,
            alpha=0.7
        )

        # 设置图表样式
        self.ax.set_xticks(range(len(self.emotion_names)))
        self.ax.set_xticklabels([self.emotion_names[i] for i in range(len(self.emotion_names))], rotation=45)
        self.ax.set_ylim(0, 100)
        # self.ax.set_ylabel('Percentage (%)')
        self.ax.grid(True, alpha=0.3)

        # 在柱子上方显示数值
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只显示大于0的值
                self.ax.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:.1f}%', ha='center', va='bottom')

        self.fig.tight_layout()
        self.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionMonitoringUI()
    window.show()
    sys.exit(app.exec_())
