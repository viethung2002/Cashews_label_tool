import sys
import os
import cv2
import numpy as np
import torch
import json
import datetime
from PyQt5.QtWidgets import (QMainWindow, QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QMenuBar, QMenu, QMessageBox, QShortcut)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence
from PyQt5.QtCore import Qt
from segment_anything import sam_model_registry, SamPredictor
from Contour_Detection import change_mode, load_current_image, select_folder, next_image, prev_image, display_saved_masks, load_samples_from_saved_masks, on_click,undo_last_mask,redo_last_mask
import Contour_Detection


class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
    def mousePressEvent(self, event):
        label_x = event.pos().x()
        label_y = event.pos().y()
        scale_x = Contour_Detection.image.shape[1] / self.width()
        scale_y = Contour_Detection.image.shape[0] / self.height()
        real_x = int(label_x * scale_x)
        real_y = int(label_y * scale_y)

        # Simulate an OpenCV left-button click event
        on_click(cv2.EVENT_LBUTTONDOWN, real_x, real_y, None, None)
        self.window().update_image()  # Refresh display immediately
        super().mousePressEvent(event)

class SAMTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM Labeling Tool")

        # Thiết lập kích thước tối thiểu (có thể thay đổi tuỳ theo yêu cầu)
        self.setMinimumSize(400, 300)  # Kích thước tối thiểu

        # Thiết lập kích thước tối đa (có thể thay đổi tuỳ theo yêu cầu)
        self.setMaximumSize(2000, 1500)  # Kích thước tối đa

        self.initUI()
    
    def initUI(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        open_action = file_menu.addAction("Open Folder")
        open_action.triggered.connect(lambda: (select_folder(), self.update_image()))
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

        main_layout = QVBoxLayout()

        self.image_label = ClickableLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_label)

        btn_layout = QHBoxLayout()

        btn_select_folder = QPushButton("Chọn thư mục ảnh")
        btn_select_folder.clicked.connect(lambda: (select_folder(), self.update_image()))
        btn_layout.addWidget(btn_select_folder)

        btn_prev = QPushButton("Ảnh trước")
        btn_prev.clicked.connect(lambda: (prev_image(), self.update_image()))
        btn_layout.addWidget(btn_prev)

        btn_next = QPushButton("Ảnh tiếp theo")
        btn_next.clicked.connect(lambda: (next_image(), self.update_image()))
        btn_layout.addWidget(btn_next)

        btn_view_masks = QPushButton("Xem mask đã lưu")
        btn_view_masks.clicked.connect(lambda: (display_saved_masks(), self.update_image()))
        btn_layout.addWidget(btn_view_masks)

        btn_load_samples = QPushButton("Load Samples")
        btn_load_samples.clicked.connect(lambda: (load_samples_from_saved_masks(), self.update_image()))
        btn_layout.addWidget(btn_load_samples)

        btn_load_samples = QPushButton("bắt đầu segment")
        btn_load_samples.clicked.connect(lambda: (change_mode("segment"), self.update_image()))
        btn_layout.addWidget(btn_load_samples)

        btn_load_samples = QPushButton("select_hat_dieu")
        btn_load_samples.clicked.connect(lambda: (change_mode("select_hat_dieu"), self.update_image()))
        btn_layout.addWidget(btn_load_samples)

        btn_load_samples = QPushButton("select_nen")
        btn_load_samples.clicked.connect(lambda: (change_mode("select_nen"), self.update_image()))
        btn_layout.addWidget(btn_load_samples)

        btn_undo_last_mask = QPushButton("Undo Last Mask")
        btn_undo_last_mask.clicked.connect(lambda: (undo_last_mask(), self.update_image()))
        btn_layout.addWidget(btn_undo_last_mask)

        btn_redo_last_mask = QPushButton("Redo Last Mask")
        btn_redo_last_mask.clicked.connect(lambda: (redo_last_mask(), self.update_image()))
        btn_layout.addWidget(btn_redo_last_mask)

        main_layout.addLayout(btn_layout)
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        QShortcut(QKeySequence("H"), self).activated.connect(lambda: change_mode("select_hat_dieu"))
        QShortcut(QKeySequence("N"), self).activated.connect(lambda: change_mode("select_nen"))
        QShortcut(QKeySequence("S"), self).activated.connect(lambda: change_mode("segment"))
        QShortcut(QKeySequence("O"), self).activated.connect(lambda: (select_folder(), self.update_image()))
        QShortcut(QKeySequence("."), self).activated.connect(lambda: (next_image(), self.update_image()))
        QShortcut(QKeySequence(">"), self).activated.connect(lambda: (next_image(), self.update_image()))
        QShortcut(QKeySequence(","), self).activated.connect(lambda: (prev_image(), self.update_image()))
        QShortcut(QKeySequence("<"), self).activated.connect(lambda: (prev_image(), self.update_image()))
        QShortcut(QKeySequence("V"), self).activated.connect(lambda: (display_saved_masks(), self.update_image()))
        QShortcut(QKeySequence("L"), self).activated.connect(lambda: (load_samples_from_saved_masks() and change_mode("segment"), self.update_image()))
        QShortcut(QKeySequence("D"), self).activated.connect(lambda: change_mode("delete_mask"))
        QShortcut(QKeySequence("Q"), self).activated.connect(lambda: (self.close(), sys.exit(0)))
        QShortcut(QKeySequence("U"), self).activated.connect(lambda: (undo_last_mask(), self.update_image()))
        QShortcut(QKeySequence("R"), self).activated.connect(lambda: (redo_last_mask(), self.update_image()))

    def update_image(self):
        if Contour_Detection.image is None:
            return

        # Lấy kích thước cửa sổ hiện tại
        window_size = self.size()  # Trả về QSize (width, height)
        
        # Chuyển đổi ảnh BGR thành RGB
        img_rgb = cv2.cvtColor(Contour_Detection.image, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        
        # Chuyển đổi sang QImage
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Tính toán kích thước ảnh sao cho vừa với cửa sổ
        new_w = window_size.width() - 18  # Giảm chiều rộng cửa sổ
        new_h = window_size.height() - 68  # Giảm chiều cao cửa sổ

        # Tạo QPixmap từ QImage và co giãn ảnh
        pixmap = QPixmap.fromImage(q_img).scaled(
            new_w,
            new_h,
            Qt.IgnoreAspectRatio,  # Không giữ tỷ lệ ảnh
            Qt.SmoothTransformation
        )
        
        # Cập nhật QPixmap lên QLabel
        self.image_label.setPixmap(pixmap)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # In ra kích thước cửa sổ hiện tại
        # window_size = self.size()  # Trả về QSize (width, height)
        # print(f"Window size: {window_size.width()}x{window_size.height()}")  # In ra kích thước cửa sổ
        self.update_image()  # Cập nhật lại hình ảnh khi cửa sổ thay đổi kích thước



    def show_about(self):
        QMessageBox.information(
            self,
            "About",
            "Hướng dẫn sử dụng:\n"
            "- Nhấn 'h' để chọn màu hạt điều\n"
            "- Nhấn 'n' để chọn màu nền\n"
            "- Nhấn 's' để bắt đầu segment\n"
            "- Nhấn 'o' để chọn thư mục ảnh\n"
            "- Nhấn '>' hoặc '.' để chuyển ảnh tiếp theo\n"
            "- Nhấn '<' hoặc ',' để chuyển ảnh trước đó\n"
            "- Nhấn 'v' để xem các mask đã lưu cho ảnh hiện tại\n"
            "- Nhấn 'l' để tự động lấy mẫu màu từ mask đã lưu\n"
            "- Nhấn 'q' để thoát\n"
            "- Nhấn 'd' để xóa mask\n"
            "- Nhấn 'u' để undo mask cuối cùng\n"
            "- Nhấn 'r' để redo mask cuối cùng\n"
        )
    
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_H:
            change_mode("select_hat_dieu")
        elif key == Qt.Key_N:
            change_mode("select_nen")
        elif key == Qt.Key_S:
            change_mode("segment")
        elif key == Qt.Key_O:
            if select_folder():
                self.update_image()
        elif key in [Qt.Key_Period, Qt.Key_Greater]:
            next_image()
            self.update_image()
        elif key in [Qt.Key_Comma, Qt.Key_Less]:
            prev_image()
            self.update_image()
        elif key == Qt.Key_V:
            display_saved_masks()
            self.update_image()
        elif key == Qt.Key_L:
            if load_samples_from_saved_masks():
                print("Đã tải mẫu màu thành công, bạn có thể bắt đầu segment ngay!")
                change_mode("segment")
                self.update_image()
            else:
                print("Không thể tự động tải mẫu màu, vui lòng chọn thủ công")
        elif key == Qt.Key_D:
            change_mode("delete_mask")
        elif key == Qt.Key_Q:
            self.close()
            sys.exit(0)
        elif key == Qt.Key_U:
            undo_last_mask()
            self.update_image()
        else:
            super().keyPressEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SAMTool()
    window.show()
    sys.exit(app.exec_())
