import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QWidget, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class BananaRipenessDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Banana Ripeness Detector")
        self.setGeometry(100, 100, 800, 600)

        # Load model kustom
        self.load_custom_model()

        # UI Elements
        self.original_image_label = QLabel("Original Image")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFixedSize(320, 240)

        self.processed_image_label = QLabel("Processed Image")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setFixedSize(320, 240)

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)

        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)

        self.detect_button = QPushButton("Detect Ripeness")
        self.detect_button.clicked.connect(self.detect_ripeness)

        self.ripeness_label = QLabel("Ripeness Level: -")
        self.ripeness_label.setAlignment(Qt.AlignCenter)

        self.color_values = QLabel("RGB Values: -")
        self.color_values.setAlignment(Qt.AlignCenter)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.original_image_label)
        layout.addWidget(self.processed_image_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.detect_button)
        layout.addWidget(self.ripeness_label)
        layout.addWidget(self.color_values)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.current_image = None

    def load_custom_model(self):
        self.model = tf.keras.models.load_model("banana_ripeness_model.h5")
        self.class_labels = ["Ripe", "Rotten", "Unripe"]

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.current_image = cv2.imread(file_name)
            self.display_image(self.current_image, self.original_image_label)
            self.processed_image_label.clear()
            self.ripeness_label.setText("Ripeness Level: -")
            self.color_values.setText("RGB Values: -")

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Cannot access camera.")
            return

        ret, frame = cap.read()
        cap.release()

        if ret:
            self.current_image = frame
            self.display_image(self.current_image, self.original_image_label)
            self.processed_image_label.clear()
            self.ripeness_label.setText("Ripeness Level: -")
            self.color_values.setText("RGB Values: -")
        else:
            QMessageBox.critical(self, "Capture Error", "Failed to capture image.")

    def detect_ripeness(self):
        if self.current_image is None:
            QMessageBox.warning(self, "No Image", "Please upload or capture an image first.")
            return

        try:
            img = cv2.resize(self.current_image, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            predictions = self.model.predict(img_array)
            predicted_index = np.argmax(predictions)
            predicted_label = self.class_labels[predicted_index]
            confidence = predictions[0][predicted_index]

            self.ripeness_label.setText(f"Ripeness Level: {predicted_label} ({confidence*100:.2f}%)")

            avg_color_per_row = np.average(img, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            r, g, b = avg_color
            self.color_values.setText(f"RGB Values: R={int(r)} G={int(g)} B={int(b)}")

            self.display_image(self.current_image, self.processed_image_label)

        except Exception as e:
            QMessageBox.critical(self, "Detection Error", f"Failed to detect ripeness: {str(e)}")

    def display_image(self, img, label):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(QPixmap.fromImage(p))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BananaRipenessDetector()
    window.show()
    sys.exit(app.exec_())
