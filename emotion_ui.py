import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QGridLayout, \
    QPushButton, QMenu, QAction, QDialog, QLabel, QProgressBar, QFrame, QFileDialog, QMessageBox
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

        # 获取处理后的图像，并更新显示
        # processed_img = emotion_window.analyze_emotions(self)  # 获取处理后的图像
        # emotion_window.exec_()  # 显示窗口

        # if processed_img is not None:
        #     self.update_student_image(processed_img)  # 更新显示图像

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

class EmotionPieChart(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(5, 4))
        super().__init__(fig)
        self.setParent(parent)
        self.update_pie_chart([1, 1, 1, 1, 1, 1, 1])

    def update_pie_chart(self, emotion_counts=None):
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Focus', 'Sad', 'Surprised']
        if emotion_counts is None:
            emotion_counts = [1] * len(emotions)

        total_students = sum(emotion_counts)
        if total_students == 0:
            emotion_percentages = [0] * len(emotions)
        else:
            emotion_percentages = [count / total_students * 100 for count in emotion_counts]

        emotion_percentages = np.nan_to_num(emotion_percentages)

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        self.ax.clear()
        self.ax.pie(emotion_percentages,
                    startangle=90,
                    colors=plt.cm.Paired.colors,
                    wedgeprops=dict(width=0.6))  # 可以调整wedge宽度使饼图更美观

        self.ax.axis('equal')
        self.draw()

class EmotionLineChart(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(8, 6))
        super().__init__(fig)
        self.setParent(parent)
        self.time_data = []  # 存储时间数据
        self.emotion_data = {emotion: [] for emotion in
                              ['Angry', 'Disgust', 'Fear', 'Happy', 'Focus', 'Sad', 'Surprised']}  # 存储情感数据
        self.update_line_chart()

    def update_line_chart(self):
        """更新情感变化的折线图"""
        emotions = list(self.emotion_data.keys())
        for emotion in emotions:
            self.ax.plot(self.time_data, self.emotion_data[emotion], label=emotion)

        self.ax.set_title("Emotional change statistics")
        self.ax.set_ylim(0, 100)
        if not self.time_data:
            self.ax.set_xlim(1, 5)  # 默认显示1-5次检测
        else:
            self.ax.set_xlim(1, len(self.time_data))  # X轴范围=1到检测次数

        self.draw()

    def update_data(self, emotion_counts):
        """更新折线图的数据（自动计算检测次数）"""
        # X轴=检测次数（1, 2, 3...）
        self.time_data.append(len(self.time_data) + 1)

        # 更新情绪数据（确保emotion_counts是百分比0-100）
        for i, emotion in enumerate(self.emotion_data.keys()):
            self.emotion_data[emotion].append(emotion_counts[i])

        self.update_line_chart()

class EmotionMonitoringUI(QWidget):
    def __init__(self,switch_page):
        super().__init__()
        self.setWindowTitle('Classroom Emotion Monitoring System')
        self.setGeometry(100, 100, 1800, 800)
        self.switch_page = switch_page
        self.students = {}
        self.next_student_id = 0

        # 用于折线图的数据
        self.time_data = []  # 存储时间数据
        self.emotion_data = {emotion: [] for emotion in
                              ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']}  # 存储情感数据

        main_layout = QHBoxLayout()  # 保持主布局为水平布局
        self.create_face_recognition_area(main_layout)

        # 创建右侧区域的垂直布局
        right_layout = QVBoxLayout()
        self.create_pie_chart_area(right_layout)
        self.create_color_indicator_area(right_layout)
        self.create_line_chart_area(right_layout)

        # 将右侧区域的垂直布局添加到主布局的右侧
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

    def create_pie_chart_area(self, layout):
        # Create a container widget for all pie charts
        pie_charts_container = QWidget(self)
        pie_charts_layout = QHBoxLayout(pie_charts_container)

        # Original visual sentiment pie chart
        self.create_single_pie_chart(pie_charts_layout, "Real-time visual sentiment", "visual")

        # Sound sentiment pie chart
        self.create_single_pie_chart(pie_charts_layout, "Real-time sound sentiment", "sound")

        # Text sentiment pie chart
        self.create_single_pie_chart(pie_charts_layout, "Real-time text sentiment", "text")

        layout.addWidget(pie_charts_container)

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
        elif chart_type == "sound":
            self.sound_pie_chart_frame = frame
            self.sound_pie_chart = pie_chart
        elif chart_type == "text":
            self.text_pie_chart_frame = frame
            self.text_pie_chart = pie_chart

        parent_layout.addWidget(frame)

    def show_pie_chart(self, chart_type):
        """Show pie chart for the specified type (visual, sound, or text) - Video only version"""
        emotion_data = {emotion: 0.0 for emotion in Config.names}  # 初始化为浮点数0.0
        total_valid_students = 0  # 有效学生计数

        for student_id, student_widget in self.students.items():
            if not student_widget.file_path:
                continue

            file_path = student_widget.file_path.lower()
            if not file_path.endswith(('.mp4', '.avi')):
                continue

            try:
                cap = cv2.VideoCapture(student_widget.file_path)
                if not cap.isOpened():
                    continue

                ret, frame = cap.read()
                cap.release()

                if not ret:
                    continue

                # 根据类型调用不同分析函数
                if chart_type == "visual":
                    _, emotion_counts = student_widget.analyze_face_emotions_counts(frame)
                elif chart_type == "sound":
                    emotion_counts = student_widget.analyze_sound_emotions(frame)
                elif chart_type == "text":
                    emotion_counts = student_widget.analyze_text_emotions(frame)
                else:
                    continue

                if not emotion_counts or len(emotion_counts) != len(Config.names):
                    continue

                # 累加情绪数据
                total_valid_students += 1
                for i, emotion in enumerate(emotion_data):
                    emotion_data[emotion] += emotion_counts[i]

            except Exception as e:
                print(f"Error processing {student_widget.file_path}: {str(e)}")
                continue

        # 处理计算结果
        if total_valid_students > 0:
            # 计算平均值并确保没有NaN或inf
            for emotion in emotion_data:
                emotion_data[emotion] /= total_valid_students
                if not np.isfinite(emotion_data[emotion]):
                    emotion_data[emotion] = 0.0
        else:
            # 如果没有有效数据，使用均匀分布
            uniform_value = 1.0 / len(Config.names)
            emotion_data = {emotion: uniform_value for emotion in Config.names}

        # 转换为列表并验证总和
        values = list(emotion_data.values())
        total = sum(values)

        # 如果总和为0（不应该发生），则使用均匀分布
        if total <= 0:
            uniform_value = 1.0 / len(Config.names)
            values = [uniform_value] * len(Config.names)

        # 更新对应的饼图
        try:
            if chart_type == "visual":
                self.visual_pie_chart.update_pie_chart(values)
            elif chart_type == "sound":
                self.sound_pie_chart.update_pie_chart(values)
            elif chart_type == "text":
                self.text_pie_chart.update_pie_chart(values)
        except Exception as e:
            print(f"Error updating {chart_type} pie chart: {str(e)}")
            # 作为后备，使用均匀分布
            uniform_value = 1.0 / len(Config.names)
            backup_values = [uniform_value] * len(Config.names)
            if chart_type == "visual":
                self.visual_pie_chart.update_pie_chart(backup_values)
            elif chart_type == "sound":
                self.sound_pie_chart.update_pie_chart(backup_values)
            elif chart_type == "text":
                self.text_pie_chart.update_pie_chart(backup_values)

    def create_line_chart_area(self, layout):
        """Create area for all line charts"""
        line_charts_container = QWidget(self)
        line_charts_layout = QHBoxLayout(line_charts_container)

        # Original visual sentiment line chart
        self.create_single_line_chart(line_charts_layout, "Visual sentiment change", "visual")

        # Sound sentiment line chart
        self.create_single_line_chart(line_charts_layout, "Sound sentiment change", "sound")

        # Text sentiment line chart
        self.create_single_line_chart(line_charts_layout, "Text sentiment change", "text")

        layout.addWidget(line_charts_container)

    def create_single_line_chart(self, parent_layout, button_text, chart_type):
        """Helper method to create a single line chart frame"""
        frame = QWidget()
        frame.setFixedSize(350, 350)

        # Create button
        button = QPushButton(button_text, frame)
        button.clicked.connect(lambda: self.show_emotion_change_graph(chart_type))
        button.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")

        # Create line chart layout
        line_layout = QVBoxLayout(frame)
        line_layout.addWidget(button)

        # Create and add line chart
        line_chart = EmotionLineChart(frame)
        line_layout.addWidget(line_chart)

        # Remove legend from line chart (assuming your EmotionLineChart class has this method)


        # Store references based on chart type
        if chart_type == "visual":
            self.visual_line_chart_frame = frame
            self.visual_line_chart = line_chart
            self.visual_time_data = []
        elif chart_type == "sound":
            self.sound_line_chart_frame = frame
            self.sound_line_chart = line_chart
            self.sound_time_data = []
        elif chart_type == "text":
            self.text_line_chart_frame = frame
            self.text_line_chart = line_chart
            self.text_time_data = []

        parent_layout.addWidget(frame)

    def hide_legend(self):
        """Hide the legend from the pie chart"""
        if hasattr(self, 'ax'):
            self.ax.legend().set_visible(False)
        self.draw()


    def show_emotion_change_graph(self, chart_type):
        """Show line chart for the specified type (visual, sound, or text)"""
        emotion_data = {emotion: 0 for emotion in Config.names}
        total_students = 0

        for student_id, student_widget in self.students.items():
            if student_widget.file_path:
                total_students += 1
                file_path = student_widget.file_path

                if chart_type == "visual":
                    # Process visual data (same as original)
                    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        cv_img = tools.img_cvread(file_path)
                        if cv_img is not None:
                            _, emotion_counts = student_widget.analyze_face_emotions(cv_img)
                            if emotion_counts:
                                for i, emotion in enumerate(emotion_data):
                                    emotion_data[emotion] += emotion_counts[i]
                    elif file_path.lower().endswith(('.mp4', '.avi')):
                        cap = cv2.VideoCapture(file_path)
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            _, emotion_counts = student_widget.analyze_video_frame_emotions(frame)
                            if emotion_counts:
                                for i, emotion in enumerate(emotion_data):
                                    emotion_data[emotion] += emotion_counts[i]
                                break
                        cap.release()

                elif chart_type == "sound":
                    # Process sound data (you'll need to implement this)
                    if file_path.lower().endswith(('.wav', '.mp3')):
                        sound_emotion_counts = student_widget.analyze_sound_emotions(file_path)
                        if sound_emotion_counts:
                            for i, emotion in enumerate(emotion_data):
                                emotion_data[emotion] += sound_emotion_counts[i]

                elif chart_type == "text":
                    # Process text data (you'll need to implement this)
                    if file_path.lower().endswith(('.txt', '.doc', '.docx')):
                        text_emotion_counts = student_widget.analyze_text_emotions(file_path)
                        if text_emotion_counts:
                            for i, emotion in enumerate(emotion_data):
                                emotion_data[emotion] += text_emotion_counts[i]

        if total_students > 0:
            for emotion in emotion_data:
                emotion_data[emotion] /= total_students

        # Update the appropriate chart
        if chart_type == "visual":
            self.visual_time_data.append(len(self.visual_time_data))
            self.visual_line_chart.update_data(self.visual_time_data[-1], list(emotion_data.values()))
            self.refresh_line_chart(self.visual_line_chart)
        elif chart_type == "sound":
            self.sound_time_data.append(len(self.sound_time_data))
            self.sound_line_chart.update_data(self.sound_time_data[-1], list(emotion_data.values()))
            self.refresh_line_chart(self.sound_line_chart)
        elif chart_type == "text":
            self.text_time_data.append(len(self.text_time_data))
            self.text_line_chart.update_data(self.text_time_data[-1], list(emotion_data.values()))
            self.refresh_line_chart(self.text_line_chart)

    def refresh_line_chart(self, line_chart):
        """Refresh the specified line chart"""
        line_chart.draw()

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

    def show_error_message(self, message):
        """显示错误信息的弹框"""
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("错误")
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = EmotionMonitoringUI()
    window.show()
    sys.exit(app.exec_())
