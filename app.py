#!/usr/bin/env python3
# app.py - Single-file PyQt5 interface for gesture detection + history + recording
# Requirements: PyQt5, opencv-python, mediapipe, tensorflow (for TFLite interpreter), numpy, pandas

import sys
import csv
import os
import time
from collections import deque, Counter
from datetime import datetime

import cv2 as cv
import numpy as np
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets

# ====== If your helper functions (calc_landmark_list, pre_process_landmark, etc.)
# are already in this file, keep them. Otherwise import them from your existing app.py
# or modules.
#
# For this template I include minimal necessary helper functions. If you had richer
# drawing code before, either paste it here or import from your module.
# ======

# --- Minimal helper implementations (replace/extend with your existing functions) ---
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp = np.array(landmark_list, dtype=np.float32)
    base_x, base_y = temp[0]
    temp[:,0] -= base_x
    temp[:,1] -= base_y
    flat = temp.flatten().tolist()
    max_value = max(list(map(abs, flat))) if len(flat)>0 else 1.0
    return [x / max_value for x in flat]

def pre_process_point_history(point_history, image_shape):
    # point_history: deque of [x,y]
    h, w = image_shape[0], image_shape[1]
    temp = [list(pt) for pt in point_history]
    if len(temp)==0:
        return [0.0] * (16*2)  # fallback
    base_x, base_y = temp[0]
    out = []
    for p in temp:
        out.append((p[0] - base_x) / float(w))
        out.append((p[1] - base_y) / float(h))
    # pad if necessary
    while len(out) < 16*2:
        out.extend([0.0, 0.0])
    return out[:16*2]

# --- Ensure directories exist
os.makedirs('model/keypoint_classifier', exist_ok=True)
os.makedirs('model/point_history_classifier', exist_ok=True)
os.makedirs('history', exist_ok=True)

# ====== Import your KeyPointClassifier (TFLite) ======
# If it's in model/keypoint_classifier.py and class name KeyPointClassifier:
try:
    from model.keypoint_classifier import KeyPointClassifier
except Exception:
    # Fallback: define a dummy class to avoid immediate crash; replace with your import
    class KeyPointClassifier:
        def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.tflite', num_threads=1):
            pass
        def __call__(self, landmark_list):
            return 0

# ====== Video processing thread ======
class VideoWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)      # BGR image
    label_ready = QtCore.pyqtSignal(int, str)        # (index, label_name)
    history_ready = QtCore.pyqtSignal(list)          # row for history CSV

    def __init__(self, device=0, width=960, height=540, parent=None):
        super().__init__(parent)
        self.device = device
        self.width = width
        self.height = height
        self._running = False

        # Mediapipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        # Classifier
        self.kp_classifier = KeyPointClassifier()

        # Label loader
        self.label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
        self.labels = self._load_labels(self.label_path)

        # Point history for dynamic gestures
        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)

        # Logging / recording toggles
        self.record_mode = 0  # 0=off, 1=keypoint, 2=point_history
        self.record_label = None  # integer label to use when recording
        self.capture = None

    def _load_labels(self, path):
        if not os.path.exists(path):
            return []
        with open(path, encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
            return [r[0] for r in rows]

    def run(self):
        self._running = True
        self.capture = cv.VideoCapture(self.device)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

        while self._running:
            ret, image = self.capture.read()
            if not ret:
                time.sleep(0.01)
                continue
            image = cv.flip(image, 1)  # mirror
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.hands.process(rgb)
            rgb.flags.writeable = True

            predicted_index = None
            predicted_label = ""

            if results.multi_hand_landmarks:
                # use first hand only for simplicity
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_list = calc_landmark_list(image, hand_landmarks)
                pre_processed = pre_process_landmark(landmark_list)
                pre_processed_point_history = pre_process_point_history(self.point_history, image.shape)

                # If recording, append new sample to appropriate CSV
                if self.record_mode == 1 and self.record_label is not None:
                    # keypoint csv path
                    csv_path = 'model/keypoint_classifier/keypoint.csv'
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.record_label, *pre_processed])
                elif self.record_mode == 2 and self.record_label is not None:
                    csv_path = 'model/point_history_classifier/point_history.csv'
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.record_label, *pre_processed_point_history])

                # Prediction (ensure size matches expected model input)
                try:
                    pred_idx = int(self.kp_classifier(pre_processed))
                    predicted_index = pred_idx
                    if 0 <= pred_idx < len(self.labels):
                        predicted_label = self.labels[pred_idx]
                    else:
                        predicted_label = f"idx:{pred_idx}"
                except Exception as e:
                    predicted_label = f"err"
                    print("Prediction error:", e)

                # update point_history for dynamic gestures detection
                # push index finger tip landmark if keypoint classifier says point gesture (example)
                # For robustness, we push landmark 8 (index finger tip) always if exists
                if len(landmark_list) > 8:
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0,0])

                # Drawing overlay - simple
                for (x,y) in landmark_list:
                    cv.circle(image, (x,y), 3, (0,255,0), -1)
                bbox = cv.boundingRect(np.array(landmark_list))
                cv.rectangle(image, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255,0,0), 2)

            else:
                # no hand - keep history rolling
                self.point_history.append([0,0])

            # Overlay text: predicted label
            if predicted_label:
                cv.putText(image, f"Pred: {predicted_label}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Emit frame for GUI
            self.frame_ready.emit(image)

            # Emit label and history
            if predicted_index is not None:
                # write to history CSV (timestamp, label_index, label_name)
                row = [datetime.utcnow().isoformat(), predicted_index, predicted_label]
                # append to history file
                hist_path = 'history/gesture_log.csv'
                with open(hist_path, 'a', newline='', encoding='utf-8') as hf:
                    writer = csv.writer(hf)
                    writer.writerow(row)
                self.history_ready.emit(row)
                self.label_ready.emit(predicted_index, predicted_label)

            # small sleep to avoid hogging CPU
            time.sleep(0.01)

        # cleanup
        if self.capture:
            self.capture.release()
        self.hands.close()

    def stop(self):
        self._running = False
        self.wait(1000)

    def set_recording(self, mode:int, label:int):
        """
        :param mode: 0=off, 1=keypoint logging, 2=point_history logging
        :param label: integer label to write as first column in CSV
        """
        self.record_mode = mode
        self.record_label = label

    def update_labels(self):
        self.labels = self._load_labels(self.label_path)


# ====== Main UI ======
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Control Interface")
        self.setGeometry(100,100,1200,720)
        # central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: video display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(960, 540)
        self.video_label.setStyleSheet("background: black;")
        layout.addWidget(self.video_label)

        # Right: controls / history
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right)

        # Buttons
        self.btn_start = QtWidgets.QPushButton("Start Detection")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_record_toggle = QtWidgets.QPushButton("Start Recording Samples (Off)")
        self.btn_record_toggle.setCheckable(True)
        self.record_mode_combo = QtWidgets.QComboBox()
        self.record_mode_combo.addItems(["Keypoint (static)", "Point history (dynamic)"])

        # Label management
        self.label_list = QtWidgets.QListWidget()
        self.label_list.setFixedHeight(200)
        self.btn_add_label = QtWidgets.QPushButton("Add Label")
        self.input_label_name = QtWidgets.QLineEdit()
        self.input_label_name.setPlaceholderText("New label name (e.g., 'thumbs_up')")

        # History table
        self.history_table = QtWidgets.QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["timestamp_utc","label_index","label_name"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setMinimumWidth(300)
        self.load_history_button = QtWidgets.QPushButton("Load History")

        # Layout
        right.addWidget(self.btn_start)
        right.addWidget(self.btn_stop)
        right.addWidget(self.record_mode_combo)
        right.addWidget(self.btn_record_toggle)
        right.addSpacing(10)
        right.addWidget(QtWidgets.QLabel("Labels (order === model index):"))
        right.addWidget(self.label_list)
        right.addWidget(self.input_label_name)
        right.addWidget(self.btn_add_label)
        right.addSpacing(10)
        right.addWidget(QtWidgets.QLabel("History:"))
        right.addWidget(self.history_table)
        right.addWidget(self.load_history_button)
        right.addStretch()

        # Worker
        self.worker = VideoWorker()
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.label_ready.connect(self.on_label)
        self.worker.history_ready.connect(self.on_history_append)

        # Connect UI
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_record_toggle.toggled.connect(self.toggle_recording)
        self.btn_add_label.clicked.connect(self.add_label)
        self.load_history_button.clicked.connect(self.load_history)

        # Load labels into list
        self.label_csv = 'model/keypoint_classifier/keypoint_classifier_label.csv'
        self.load_labels()

        # history file ensure header
        hist_path = 'history/gesture_log.csv'
        if not os.path.exists(hist_path):
            with open(hist_path, 'w', newline='', encoding='utf-8') as hf:
                writer = csv.writer(hf)
                writer.writerow(['timestamp_utc','label_index','label_name'])

    def load_labels(self):
        self.label_list.clear()
        if os.path.exists(self.label_csv):
            with open(self.label_csv, encoding='utf-8-sig') as f:
                for row in csv.reader(f):
                    if row:
                        self.label_list.addItem(row[0])
        # inform worker to reload labels
        self.worker.update_labels()

    def add_label(self):
        name = self.input_label_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Validation", "Please enter a label name.")
            return
        # append to CSV (label ordering must match model training ordering)
        with open(self.label_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([name])
        self.input_label_name.clear()
        self.load_labels()
        QtWidgets.QMessageBox.information(self, "Label Added", f"Added label '{name}'.\n\nNote: ensure your training data uses the same index for this label.")

    def start_detection(self):
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_detection(self):
        self.worker.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def toggle_recording(self, checked):
        # When toggled on we will begin writing samples. We need a label index selection.
        if checked:
            # require a selected label
            sel = self.label_list.currentRow()
            if sel < 0:
                QtWidgets.QMessageBox.warning(self, "Select Label", "Select a label in the list to record samples to.")
                self.btn_record_toggle.setChecked(False)
                return
            mode = 1 if self.record_mode_combo.currentIndex() == 0 else 2
            self.worker.set_recording(mode, sel)
            self.btn_record_toggle.setText(f"Recording Samples (Label idx {sel})")
        else:
            self.worker.set_recording(0, None)
            self.btn_record_toggle.setText("Start Recording Samples (Off)")

    @QtCore.pyqtSlot(np.ndarray)
    def on_frame(self, frame):
        # Convert BGR ndarray -> QImage -> setPixmap
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio))

    @QtCore.pyqtSlot(int, str)
    def on_label(self, idx, name):
        # you can display status or use for other logic
        self.statusBar().showMessage(f"Predicted: {name} (idx {idx})")

    @QtCore.pyqtSlot(list)
    def on_history_append(self, row):
        # append new row to table UI
        # possible performance note: you may buffer updates in large streams
        self.append_history_row(row)

    def append_history_row(self, row):
        r = self.history_table.rowCount()
        self.history_table.insertRow(r)
        for c, v in enumerate(row):
            it = QtWidgets.QTableWidgetItem(str(v))
            self.history_table.setItem(r, c, it)

    def load_history(self):
        hist_path = 'history/gesture_log.csv'
        if not os.path.exists(hist_path):
            QtWidgets.QMessageBox.information(self, "No history", "No history found yet.")
            return
        with open(hist_path, encoding='utf-8-sig') as f:
            reader = list(csv.reader(f))
        # remove header if present
        if reader and reader[0][0] == 'timestamp_utc':
            reader = reader[1:]
        self.history_table.setRowCount(0)
        for r in reader:
            if not r:
                continue
            self.append_history_row(r)

# ====== Run App ======
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
