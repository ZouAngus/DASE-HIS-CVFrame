import sys
import os
import json
import cv2
import numpy as np
import pandas as pd

# Project root is one level up from this file (tools/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QProgressDialog,
    QFileDialog, QSlider, QSpinBox, QApplication, QMessageBox, QLineEdit, QGridLayout, QSizePolicy, QInputDialog, QMenuBar, QListWidget, QListWidgetItem, QCheckBox
)
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication
from PyQt5.QtCore import Qt, QTimer, QCoreApplication
from video_player import VideoPlayer  
from black_video_player import BlackVideoPlayer
from points3d_cache import Points3DCache

DEFAULT_POINTS_FPS = 60.0

# 定義關鍵點連接對，用於繪製骨架
# 17 keypoints
JOINT_PAIRS_17kp = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]

# 24 keypoints
JOINT_PAIRS_24kp = [
    (6,9),(12,9),(12,15),
    (20,18),(18,16),
    (16,13),(13,6),(14,6),(14,17),
    (17,19),(19,21),
    (3,6),(0,3),(1,0),(2,0),
    (10,7),(7,4),(4,1),
    (2,5),(5,8),(11,8)
]

# 骨架連接對的映射表
JOINT_PAIRS_MAP = {
    17: JOINT_PAIRS_17kp,
    24: JOINT_PAIRS_24kp,
}

class ProjectionWindow2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Points Projection ver2")
        self.setGeometry(100, 100, 1000, 600)
        self.setFocusPolicy(Qt.StrongFocus)  # Added to capture key events
        self.setFocus()                       # Ensure window has focus
        self.player = None 
        self.intrinsics = None
        self.extrinsics = None
        self.rvec = None
        self.tvec = None
        self.points3d = None
        self.frame_offset = 0
        self.points_frame_count = 0
        self.loaded_video_filename = ""
        self.loaded_video_path = ""
        self.recent_video_filename = ""
        self.recent_video_path = ""
        self.loaded_intrinsics_filename = ""
        self.loaded_extrinsics_filename = ""
        self.loaded_points_filename = ""
        self.is_playing = False
        self.max_frame_3d = 0

        # FPS of the 3D points data (used when we can't infer it from a loaded video).
        # If your 3D CSV/NPY is 60fps and your video is 30fps, keep this at 60.
        self.default_points_fps = DEFAULT_POINTS_FPS

        # Extrinsic convention: some toolchains store camera pose as camera->world.
        # OpenCV projectPoints expects world->camera (Xc = R*Xw + t).
        self._extrinsic_R = None
        self._extrinsic_t = None
        self._extrinsic_convention_fixed = False
        self._extrinsic_mirror_fixed = False
        
        # 添加多個3D資料檔案的支援
        self.loaded_points_files = []  # 存儲已加載的3D資料檔案資訊
        self.current_points_index = -1  # 當前選中的3D資料檔案索引
        self.visible_points_files = set()  # 存儲勾選顯示的檔案索引
        self.show_skeleton = True  # 控制是否顯示骨架連接線

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 初始化時自動加載內外參數
        self.update_camera_parameters()
        
        self.statusBar().showMessage(
            "Shortcut: Space - Play/Pause, A - Prev Frame, D - Next Frame, Q - Back 1s, E - Forward 1s, W - Increase Offset, S - Decrease Offset, R - Locate Frame, F - Locate Time, Z - Copy Offset"
        )

    @staticmethod
    def _normalize_dist_coeffs(dist_coeffs_value):
        if dist_coeffs_value is None:
            return None
        d = np.array(dist_coeffs_value, dtype=float)
        # Accept either [k1,k2,p1,p2,k3] or [[...]]
        d = d.reshape(-1)
        return d

    @staticmethod
    def _extract_extrinsic_3x4(data: dict):
        """Return a 3x4 extrinsic matrix from supported JSON formats."""
        if not isinstance(data, dict):
            raise ValueError("Extrinsics JSON must be an object")

        ext = None
        if "best_extrinsic" in data:
            ext = np.array(data["best_extrinsic"], dtype=float)
        elif "extrinsic" in data:
            ext = np.array(data["extrinsic"], dtype=float)
        elif "extrinsics" in data:
            # Many files store a list of per-frame [R|t]. Use the first by default.
            ext_list = data.get("extrinsics")
            if isinstance(ext_list, list) and ext_list:
                ext = np.array(ext_list[0], dtype=float)

        if ext is None:
            raise KeyError("No supported extrinsic matrix key found (best_extrinsic/extrinsic/extrinsics)")

        if ext.shape == (4, 4):
            ext = ext[:3, :]

        if ext.shape != (3, 4):
            raise ValueError(f"Unsupported extrinsic shape {ext.shape}; expected 3x4 or 4x4")

        return ext

    @staticmethod
    def _invert_world_camera(R: np.ndarray, t: np.ndarray):
        """Invert a world->camera transform into camera->world and back.

        If Xc = R*Xw + t, then Xw = R^T*(Xc - t).
        So the inverse (camera->world) has R_inv = R^T and t_inv = -R^T*t.
        """
        R_inv = R.T
        t_inv = -R_inv @ t
        return R_inv, t_inv

    def _set_extrinsic_Rt(self, R: np.ndarray, t: np.ndarray):
        self._extrinsic_R = np.array(R, dtype=float)
        self._extrinsic_t = np.array(t, dtype=float).reshape(3, 1)
        self.rvec, _ = cv2.Rodrigues(self._extrinsic_R)
        self.tvec = self._extrinsic_t
        self._extrinsic_convention_fixed = False
        self._extrinsic_mirror_fixed = False

    def _load_extrinsics_from_json(self, data: dict):
        """Load camera intrinsics/distortion/extrinsics from a JSON dict."""
        self.extrinsics = data
        ext3x4 = self._extract_extrinsic_3x4(data)
        R = ext3x4[:, :3]
        t = ext3x4[:, 3].reshape(3, 1)
        self._set_extrinsic_Rt(R, t)

    def _auto_fix_extrinsic_convention_if_needed(self, pts3d: np.ndarray):
        """If most points are behind the camera, try inverting [R|t].

        This addresses the common mismatch where JSON stores camera pose as camera->world
        but OpenCV expects world->camera.
        """
        if self._extrinsic_convention_fixed:
            return
        if self._extrinsic_R is None or self._extrinsic_t is None:
            return
        if pts3d is None or pts3d.size == 0:
            return

        try:
            pts = np.asarray(pts3d, dtype=float).reshape(-1, 3).T  # 3xN
            z_as_is = (self._extrinsic_R @ pts + self._extrinsic_t)[2, :]
            pos_as_is = float(np.mean(z_as_is > 1e-6))

            R_inv, t_inv = self._invert_world_camera(self._extrinsic_R, self._extrinsic_t)
            z_inv = (R_inv @ pts + t_inv)[2, :]
            pos_inv = float(np.mean(z_inv > 1e-6))

            # If inversion clearly puts more points in front, adopt it.
            if pos_inv > pos_as_is + 0.25:
                self._set_extrinsic_Rt(R_inv, t_inv)
            self._extrinsic_convention_fixed = True
        except Exception:
            # On any failure, don't block rendering.
            self._extrinsic_convention_fixed = True

    def _auto_fix_extrinsic_horizontal_mirror_if_needed(self, pts3d: np.ndarray, frame_bgr: np.ndarray):
        """If projections appear horizontally mirrored, try flipping camera X.

        Heuristic: choose the transform that yields more projected points within image bounds.
        This is intentionally conservative and only runs once per extrinsics load.
        """
        if self._extrinsic_mirror_fixed:
            return
        if self._extrinsic_R is None or self._extrinsic_t is None:
            self._extrinsic_mirror_fixed = True
            return
        if pts3d is None or pts3d.size == 0 or frame_bgr is None:
            self._extrinsic_mirror_fixed = True
            return
        if self.extrinsics is None:
            self._extrinsic_mirror_fixed = True
            return

        try:
            h, w = frame_bgr.shape[:2]
            if h <= 0 or w <= 0:
                self._extrinsic_mirror_fixed = True
                return

            if self.intrinsics is not None:
                cam_mtx = np.array(self.intrinsics["camera_matrix"], dtype=float)
                dcoeff = self._normalize_dist_coeffs(self.intrinsics.get("dist_coeffs"))
            else:
                cam_mtx = np.array(self.extrinsics["camera_matrix"], dtype=float)
                dcoeff = self._normalize_dist_coeffs(self.extrinsics.get("dist_coeffs"))
            if dcoeff is None:
                dcoeff = np.zeros(5, dtype=float)

            pts3d_reshaped = np.asarray(pts3d, dtype=float).reshape(-1, 1, 3)

            def in_frame_count(R, t):
                rvec, _ = cv2.Rodrigues(R)
                projected, _ = cv2.projectPoints(pts3d_reshaped, rvec, t, cam_mtx, dcoeff)
                p = projected.reshape(-1, 2)
                x = p[:, 0]
                y = p[:, 1]
                return int(np.sum((x >= 0) & (x < w) & (y >= 0) & (y < h)))

            count_as_is = in_frame_count(self._extrinsic_R, self._extrinsic_t)

            flip_x = np.diag([-1.0, 1.0, 1.0])
            R_fx = flip_x @ self._extrinsic_R
            t_fx = flip_x @ self._extrinsic_t
            count_fx = in_frame_count(R_fx, t_fx)

            # Only switch if it's clearly better.
            if count_fx > count_as_is + max(5, int(0.05 * pts3d_reshaped.shape[0])):
                self._set_extrinsic_Rt(R_fx, t_fx)

            self._extrinsic_mirror_fixed = True
        except Exception:
            self._extrinsic_mirror_fixed = True

    def _get_default_virtual_fps(self):
        """Choose a reasonable FPS for the virtual black background."""
        if 0 <= self.current_points_index < len(self.loaded_points_files):
            fps = self.loaded_points_files[self.current_points_index].get('fps')
            if fps and fps > 0:
                return float(fps)
        if self.loaded_points_files:
            fps_values = [fi.get('fps') for fi in self.loaded_points_files]
            fps_values = [float(f) for f in fps_values if f and f > 0]
            if fps_values:
                return max(fps_values)
        return float(self.default_points_fps)

    def _estimate_points_fps_from_video(self, points_frame_count: int):
        """Estimate points FPS so points duration matches the loaded video duration."""
        try:
            if not self.player or isinstance(self.player, BlackVideoPlayer):
                return None
            video_fps = float(self.player.fps) if self.player.fps else 0.0
            video_frames = int(self.player.frame_count) if self.player.frame_count else 0
            if video_fps <= 0 or video_frames <= 0:
                return None
            duration_sec = video_frames / video_fps
            if duration_sec <= 0:
                return None
            return float(points_frame_count) / duration_sec
        except Exception:
            return None

    def _refresh_points_fps_estimates_from_video(self):
        """When a real video is loaded, update all loaded point files' FPS estimates."""
        if not self.player or isinstance(self.player, BlackVideoPlayer):
            return
        for fi in self.loaded_points_files:
            fc = int(fi.get('frame_count') or 0)
            est = self._estimate_points_fps_from_video(fc)
            if est and est > 0:
                fi['fps'] = est

    @staticmethod
    def _map_video_frame_to_points_index(
        video_frame_idx: int,
        video_fps: float,
        points_fps: float,
        points_frame_count: int,
        offset_video_frames: int = 0,
    ):
        """Map a video frame index to a 3D-points frame index using FPS ratio.

        Offset is interpreted in *video frames* (keeps existing UI behavior).
        """
        if points_frame_count <= 0:
            return None

        try:
            vfps = float(video_fps) if video_fps else 0.0
        except Exception:
            vfps = 0.0
        try:
            pfps = float(points_fps) if points_fps else 0.0
        except Exception:
            pfps = 0.0

        if vfps <= 0:
            idx = int(video_frame_idx) + int(offset_video_frames)
        else:
            if pfps <= 0:
                pfps = vfps
            idx = int(round((int(video_frame_idx) + int(offset_video_frames)) * (pfps / vfps)))

        if idx < 0:
            return 0
        if idx >= points_frame_count:
            return points_frame_count - 1
        return idx

    ########## UI Components ##########
    def init_ui(self):
        widget = QWidget()
        self.setCentralWidget(widget)
        main_layout = QVBoxLayout()
        
        # 創建水平佈局來放置視頻和3D資料列表
        content_layout = QHBoxLayout()
        
        # Video layout: video display and overlaid info label
        video_layout = QVBoxLayout()
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(400, 300)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Ensure full horizontal fill
        video_layout.addWidget(self.video_label)
        self.info_label = QLabel(self.video_label)
        self.info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.info_label.setStyleSheet("background-color: rgba(0, 0, 0, 0.5); color: white; padding: 5px;")
        self.info_label.setText("00:00:00 / 00:00:00\nFrame: 0 / 0")
        
        # 3D Data Files List (右側)
        points_list_layout = QVBoxLayout()
        points_list_label = QLabel("Loaded 3D Data Files:")
        points_list_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        points_list_layout.addWidget(points_list_label)
        
        # 使用QListWidget with checkboxes
        self.points_list = QListWidget()
        self.points_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow both horizontal and vertical expansion
        self.points_list.itemChanged.connect(self.on_points_checkbox_changed)
        self.points_list.itemDoubleClicked.connect(self.on_points_file_double_clicked)
        points_list_layout.addWidget(self.points_list)
        
        # 添加3D資料檔案的按鈕
        add_points_btn = QPushButton("Add 3D Data")
        add_points_btn.clicked.connect(lambda: self.load_points(is_visible_by_default=True))
        add_points_btn.setMaximumWidth(250)
        points_list_layout.addWidget(add_points_btn)
        
        # 添加骨架顯示控制
        self.skeleton_checkbox = QCheckBox("Show Skeleton")
        self.skeleton_checkbox.setChecked(self.show_skeleton)
        self.skeleton_checkbox.stateChanged.connect(self.on_skeleton_checkbox_changed)
        self.skeleton_checkbox.setMaximumWidth(250)
        points_list_layout.addWidget(self.skeleton_checkbox)
        
        # 將視頻和3D資料列表添加到水平佈局
        content_layout.addLayout(video_layout, 4)  # 視頻佔4份空間
        content_layout.addLayout(points_list_layout, 1)  # 列表佔1份空間
        
        # Control layout: file load controls (wrapped in a fixed-height widget) on top and play buttons underneath
        control_layout = QVBoxLayout()
        file_grid = QGridLayout()
        self.create_file_widgets(file_grid)
        file_grid.setVerticalSpacing(5)
        file_grid.setHorizontalSpacing(5)
        file_grid_widget = QWidget()
        file_grid_widget.setLayout(file_grid)
        file_grid_widget.setFixedHeight(120)
        control_layout.addWidget(file_grid_widget)
        
        # --- Add synchronization offset controls ---
        sync_layout = QHBoxLayout()
        offset_label = QLabel("Frame Offset:")
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-20000, 20000)
        self.offset_spin.setValue(self.frame_offset)
        self.offset_spin.valueChanged.connect(self.change_offset)
        sync_layout.addWidget(offset_label)
        sync_layout.addWidget(self.offset_spin)
        
        # --- Add Flip Controls ---
        self.flip_x_cb = QCheckBox("Flip X")
        self.flip_y_cb = QCheckBox("Flip Y")
        self.flip_z_cb = QCheckBox("Flip Z")
        self.auto_fix_cb = QCheckBox("Auto Fix (Invert/Mirror)")
        self.auto_fix_cb.setChecked(False) # default disabled
        self.flip_x_cb.stateChanged.connect(self.update_frame)
        self.flip_y_cb.stateChanged.connect(self.update_frame)
        self.flip_z_cb.stateChanged.connect(self.update_frame)
        self.auto_fix_cb.stateChanged.connect(self.update_frame)
        sync_layout.addWidget(self.flip_x_cb)
        sync_layout.addWidget(self.flip_y_cb)
        sync_layout.addWidget(self.flip_z_cb)
        sync_layout.addWidget(self.auto_fix_cb)
        sync_layout.addStretch()
        
        control_layout.addLayout(sync_layout)
        # --- End synchronization offset controls ---
        
        play_controls_layout = QHBoxLayout()
        self.create_play_buttons(play_controls_layout)
        control_layout.addLayout(play_controls_layout)
        
        # Adjust stretch factors: content layout gets more space than controls
        main_layout.addLayout(content_layout, 4)
        main_layout.addLayout(control_layout, 1)
        widget.setLayout(main_layout)
        
        menu_bar = self.menuBar() if self.menuBar() else self.menuBar()  # ensure it exists

        # 新增 File 選單
        file_menu = menu_bar.addMenu("File")
        act_load_folder = file_menu.addAction("Load Folder (npy/csv)")
        act_load_folder.triggered.connect(self.load_folder)

        # --- Add Locate controls ---
        go_menu   = menu_bar.addMenu("Go")
        act_frame = go_menu.addAction("Locate Frame")
        act_time  = go_menu.addAction("Locate Time")
        act_frame.triggered.connect(self.locate_frame)
        act_time.triggered.connect(self.locate_time)
        
        # --- Add Camera Parameters menu ---
        camera_menu = menu_bar.addMenu("Camera")
        act_middle = camera_menu.addAction("Switch Perspective (middle)")
        act_left = camera_menu.addAction("Switch Perspective (left)")
        act_middle.triggered.connect(self.update_camera_parameters)
        act_left.triggered.connect(self.update_camera_parameters_left)

        # Add video menu
        video_menu = menu_bar.addMenu("Video")
        act_virtual = video_menu.addAction("Use Virtual Video")
        act_video_file = video_menu.addAction(f"Use Video file")
        act_virtual.triggered.connect(self.update_background_virtual)
        act_video_file.triggered.connect(self.update_background_real)

        # Add export menu
        export_menu = menu_bar.addMenu("Export")
        act_export = export_menu.addAction("Export Video")
        act_export.triggered.connect(self.export_video)

    def create_file_widgets(self, file_grid):
        # Simplified function to create all file loading buttons and labels
        self.btn_load_intr = self.create_button("Update Intrinsics", "No File Loaded", self.load_intrinsics)
        self.btn_load_extr = self.create_button("Update Extrinsics", "No File Loaded", self.load_extrinsics)
        self.btn_load_video = self.create_button("Load Video", "No File Loaded",self.load_video)
        #self.btn_load_points = self.create_button("Load 3D Data (CSV)", "No File Loaded",self.load_points)
        self.btn_load_points = self.create_button("Load 3D Data (NPY/CSV)", "No File Loaded", lambda: self.load_points(is_visible_by_default=True))
        
        file_grid.addWidget(self.btn_load_intr[0], 0, 0)
        file_grid.addWidget(self.btn_load_extr[0], 0, 1)
        file_grid.addWidget(self.btn_load_video[0], 1, 0)
        file_grid.addWidget(self.btn_load_points[0], 1, 1)
        
    def create_button(self, text, label_text, click_event):
        btn = QPushButton(text)
        btn.setFixedSize(150, 30)
        btn.clicked.connect(click_event)
        label = QLabel(label_text)
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(btn, alignment=Qt.AlignCenter)
        layout.addWidget(label, alignment=Qt.AlignCenter)
        widget.setLayout(layout)
        return widget, label

    def create_play_buttons(self, play_controls_layout):
        self.btn_jump_bwd = QPushButton("<< -1 sec")
        self.btn_prev = QPushButton("<<")
        self.btn_toggle = QPushButton("Play")  # combined play/pause button
        self.btn_next = QPushButton(">>")
        self.btn_jump_fwd = QPushButton(">> +1 sec")

        play_controls_layout.addWidget(self.btn_jump_bwd)
        play_controls_layout.addWidget(self.btn_prev)
        play_controls_layout.addWidget(self.btn_toggle)
        play_controls_layout.addWidget(self.btn_next)
        play_controls_layout.addWidget(self.btn_jump_fwd)

        self.btn_toggle.clicked.connect(self.toggle_playback)
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_jump_fwd.clicked.connect(lambda: self.jump_seconds(1))
        self.btn_jump_bwd.clicked.connect(lambda: self.jump_seconds(-1))

    ########## Loading Utilities ##########
    def load_intrinsics(self):
        """手動選擇並載入內參檔案"""
        filename, _ = QFileDialog.getOpenFileName(self, "Selfect Intrinsics JSON", "", "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'r') as fp:
                    data = json.load(fp)
                self.intrinsics = data
                self.loaded_intrinsics_filename = os.path.basename(filename)
                print(f"Loaded Intrinsics from {filename}")
                self.update_loaded_files_label()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load intrinsics:\n{str(e)}")

    def load_extrinsics(self):
        """手動選擇並載入外參檔案"""
        filename, _ = QFileDialog.getOpenFileName(self, "Select Extrinsics JSON", "", "JSON Files (*.json)")
        if filename:
            try:
                with open(filename, 'r') as fp:
                    data = json.load(fp)
                self._load_extrinsics_from_json(data)
                self.loaded_extrinsics_filename = os.path.basename(filename)
                print(f"Loaded Extrinsics from {filename}")
                self.update_loaded_files_label()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load extrinsics:\n{str(e)}")
    
    def update_background_virtual(self):
        # print(f"Creating virtual black video with {self.max_frame_3d} frame")
        if self.player is not None:
            self.player.release()
        self.player = BlackVideoPlayer(frame_count=self.max_frame_3d, fps=self._get_default_virtual_fps())
        self.recent_video_filename = "Virtual Black Video"
        self.recent_video_path = "Virtual Black Video"
        self.update_frame()
        self.update_loaded_files_label()
    
    def update_background_real(self):   
        try:
            if self.player is not None:
                self.player.release()
            self.player = VideoPlayer(self.loaded_video_path)
            self.recent_video_filename = self.loaded_video_filename
            self.recent_video_path = self.loaded_video_path
            self._refresh_points_fps_estimates_from_video()
            self.frame_offset = 0
            if hasattr(self, 'offset_spin'):
                self.offset_spin.setValue(0)
            self.update_frame()
            self.update_loaded_files_label()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open video file {self.loaded_video_path}.\n{str(e)}")     

    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi)")
        if filename:
            try:
                if self.player is not None:
                    self.player.release()
                self.player = VideoPlayer(filename)
                self.loaded_video_path = filename
                self.loaded_video_filename = os.path.basename(filename)
                self.recent_video_filename = self.loaded_video_filename
                self.recent_video_path = self.loaded_video_path
                self._refresh_points_fps_estimates_from_video()
                self.frame_offset = 0
                if hasattr(self, 'offset_spin'):
                    self.offset_spin.setValue(0)
                self.update_loaded_files_label()
                self.update_frame()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot open video file {filename}.\n{str(e)}")

    def load_points(self, filename=None, is_visible_by_default=True):
        # 如果沒有提供 filename，則開啟檔案選擇對話框
        if filename is None:
            filename, _ = QFileDialog.getOpenFileName(self, "Select 3D Data File", "", "3D Data Files (*.npy *.csv)")
            if not filename:
                return

        try:
            data = None
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension == ".npy":
                data = np.load(filename)
                print(f"Loaded NPY data shape: {data.shape}")
            elif file_extension == ".csv":
                # 根據之前查看的 extract_17_keypoint_from_csv.py 邏輯，
                # 假設 CSV 檔案的格式是每一行代表一幀，每三個列代表一個關鍵點的XYZ座標
                df = pd.read_csv(filename)
                # 確保所有列都是數值類型，非數值的轉換為NaN
                df = df.apply(pd.to_numeric, errors='coerce')
                
                # 從列名中解析出關鍵點數量，例如 '0_x', '0_y', '0_z', '1_x', ...
                # 假設列名格式為 "joint_idx_axis"
                num_cols = df.shape[1]
                if num_cols % 3 != 0:
                    QMessageBox.critical(self, "Error", f"CSV file has {num_cols} columns, which is not a multiple of 3. Expected (joints * 3).")
                    return
                num_joints = num_cols // 3
                
                # 將 DataFrame 重塑為 (frames, joints, 3)
                # 首先將所有XYZ座標合併成一個大的一維陣列，然後重塑
                # 注意：這裡需要確保df的順序是按照x,y,z順序排列
                data = df.values.reshape(-1, num_joints, 3)
                print(f"Loaded CSV data shape: {data.shape}")
            else:
                QMessageBox.critical(self, "Error", "Unsupported file type. Please select a .npy or .csv file.")
                return
            
            # 檢查載入的資料是否符合預期形狀 (frames, joints, 3)
            if data is not None and len(data.shape) == 3 and data.shape[2] == 3:
                frame_count = data.shape[0]
                print(f"Number of frames: {frame_count}")

                # Estimate the FPS of this 3D data. If a real video is loaded, match durations.
                estimated_points_fps = self._estimate_points_fps_from_video(frame_count)
                if not estimated_points_fps or estimated_points_fps <= 0:
                    estimated_points_fps = float(self.default_points_fps)

                # 如果沒有載入視頻，或者新增file have larger total frame，則創建虛擬視頻播放器
                if self.player is None or frame_count > self.max_frame_3d:
                    self.max_frame_3d = frame_count
                    if self.player is not None:
                        self.player.release()
                    print(f"Creating virtual black video with {self.max_frame_3d} frame")
                    self.player = BlackVideoPlayer(frame_count=self.max_frame_3d, fps=estimated_points_fps)
                    self.recent_video_filename = "Virtual Black Video"
                    self.recent_video_path = "Virtual Black Video"
                    self.update_loaded_files_label()
                    
                # 創建檔案資訊字典
                file_info = {
                    'filename': os.path.basename(filename),
                    'full_path': filename,
                    'data': data,
                    'frame_count': frame_count,
                    'fps': float(estimated_points_fps),
                    'color': self.get_next_color(len(self.loaded_points_files))  # 為每個檔案分配不同顏色
                }
                print(f"Assigning color {file_info['color']} to {file_info['filename']}")
                
                # 添加到已加載檔案列表
                self.loaded_points_files.append(file_info)
                
                # 根據 is_visible_by_default 決定是否勾選新加載的檔案
                if is_visible_by_default:
                    self.visible_points_files.add(len(self.loaded_points_files) - 1)
                
                # 更新列表顯示
                self.update_points_list()
                
                # 如果是第一個檔案，自動選中
                if len(self.loaded_points_files) == 1:
                    self.current_points_index = 0
                    self.points3d = data
                    self.points_frame_count = frame_count
                    self.frame_offset = 0
                    if hasattr(self, 'offset_spin'):
                        self.offset_spin.setValue(0)

                # QMessageBox.information(self, "Success", f"3D data loaded: {os.path.basename(filename)} ({frame_count} frames)")
                self.update_loaded_files_label()
                self.update_frame() 
            else:
                QMessageBox.critical(self, "Error", f"Unexpected data shape: {data.shape}. Expected 3D array (frames, joints, 3) with 3 coordinates per joint.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load 3D data:\n{str(e)}")

    def get_next_color(self, index):
        """為每個3D資料檔案分配不同的顏色"""
        colors = [
            (0, 0, 255),    # 紅色 (BGR)
            (0, 255, 0),    # 綠色 (BGR)
            (255, 0, 0),    # 藍色 (BGR)
            (0, 255, 255),  # 黃色 (BGR)
            (255, 0, 255),  # 洋紅色 (BGR)
            (255, 255, 0),  # 青色 (BGR)
            (128, 0, 128),  # 紫色 (BGR)
            (0, 165, 255),  # 橙色 (BGR)
            (0, 128, 0),    # 深綠色 (BGR)
            (0, 0, 128),    # 深紅色 (BGR)
        ]
        assigned_color = colors[index % len(colors)]
        print(f"Getting color for index {index}: {assigned_color} (BGR)")
        return assigned_color
    
    def update_points_list(self):
        """更新3D資料檔案列表顯示"""
        self.points_list.clear()
        for i, file_info in enumerate(self.loaded_points_files):
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            
            # 設定勾選狀態
            if i in self.visible_points_files:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
            
            # 設定顯示文字
            item_text = f"{file_info['filename']} ({file_info['frame_count']} frames)"
            '''
            if i == self.current_points_index:
                item_text += " [Active]"
            '''
            item.setText(item_text)
            
            self.points_list.addItem(item)
    
    def on_points_checkbox_changed(self, item):
        """當使用者勾選或取消勾選3D資料檔案時"""
        index = self.points_list.row(item)
        if 0 <= index < len(self.loaded_points_files):
            if item.checkState() == Qt.Checked:
                self.visible_points_files.add(index)
                print(f"Enabled 3D data: {self.loaded_points_files[index]['filename']}")
            else:
                self.visible_points_files.discard(index)
                print(f"Disabled 3D data: {self.loaded_points_files[index]['filename']}")
            
            # 立即更新幀顯示
            self.update_frame()

    def on_points_file_double_clicked(self, item):
        """當使用者雙擊3D資料檔案時，切換為當前激活檔案"""
        index = self.points_list.row(item)
        if 0 <= index < len(self.loaded_points_files):
            self.current_points_index = index
            file_info = self.loaded_points_files[index]
            self.points3d = file_info['data']
            self.points_frame_count = file_info['frame_count']
            self.loaded_points_filename = file_info['filename']
            self.frame_offset = 0
            if hasattr(self, 'offset_spin'):
                self.offset_spin.setValue(0)
            
            print(f"Switched to 3D data: {file_info['filename']}")
            self.update_points_list()  # 更新列表顯示
            self.update_loaded_files_label()

    ########## Video Control Functions ##########
    def update_loaded_files_label(self):
        # Update the info label with loaded file names  
        self.btn_load_video[1].setText(self.recent_video_filename if self.recent_video_filename else "No File Loaded")
        self.btn_load_intr[1].setText(self.loaded_intrinsics_filename if self.loaded_intrinsics_filename else "No File Loaded") 
        self.btn_load_extr[1].setText(self.loaded_extrinsics_filename if self.loaded_extrinsics_filename else "No File Loaded")
        
        # 顯示當前選中的3D資料檔案
        if self.current_points_index >= 0 and self.current_points_index < len(self.loaded_points_files):
            current_file = self.loaded_points_files[self.current_points_index]
            self.btn_load_points[1].setText(f"{current_file['filename']} ({current_file['frame_count']})")
        else:
            self.btn_load_points[1].setText(f"No File Loaded ({len(self.loaded_points_files)} files available)")

    def update_frame(self):
        # If playing, advance the frame
        if self.player and self.player.is_playing:
            self.player.next_frame()
            if self.player.current_frame >= self.player.frame_count - 1:
                self.player.is_playing = False
                self.btn_toggle.setText("Play")
                self.timer.stop()
        
        if not self.player:
            return

        frame = self.player.get_frame()
        if frame is None:
            return

        # Process the frame: convert to BGR for processing
        # 如果是虛擬視頻，直接使用黑色背景，不用轉換
        if isinstance(self.player, BlackVideoPlayer):
            frame_bgr = frame 
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Optionally undistort using intrinsics if available
        # if self.intrinsics is not None:
        #     cam_mtx = np.array(self.intrinsics["camera_matrix"])
        #     dcoeff = self.intrinsics["dist_coeffs"]
        #     if isinstance(dcoeff[0], list):
        #         dcoeff = np.array(dcoeff[0])
        #     else:
        #         dcoeff = np.array(dcoeff)
        #     frame_bgr = cv2.undistort(frame_bgr, cam_mtx, dcoeff)
        
        # Map 3D points onto the frame if available
        if self.extrinsics is not None and self.visible_points_files:
            # 顯示所有勾選的3D資料檔案
            for file_index in self.visible_points_files:
                if 0 <= file_index < len(self.loaded_points_files):
                    file_info = self.loaded_points_files[file_index]
                    points_data = file_info['data']
                    color = file_info['color']

                    points_idx = self._map_video_frame_to_points_index(
                        video_frame_idx=self.player.current_frame,
                        video_fps=self.player.fps,
                        points_fps=file_info.get('fps') or self.default_points_fps,
                        points_frame_count=points_data.shape[0],
                        offset_video_frames=self.frame_offset,
                    )

                    if points_idx is not None and 0 <= points_idx < points_data.shape[0]:
                        pts3d = points_data[points_idx].copy()
                        
                        if hasattr(self, 'flip_x_cb'):
                            if self.flip_x_cb.isChecked(): pts3d[:, 0] *= -1
                            if self.flip_y_cb.isChecked(): pts3d[:, 1] *= -1
                            if self.flip_z_cb.isChecked(): pts3d[:, 2] *= -1
                            
                        # if hasattr(self, 'auto_fix_cb') and self.auto_fix_cb.isChecked():
                        #     self._auto_fix_extrinsic_convention_if_needed(pts3d)
                        #     self._auto_fix_extrinsic_horizontal_mirror_if_needed(pts3d, frame_bgr)
                        if self.rvec is not None and self.tvec is not None:
                            if self.intrinsics is not None:
                                cam_mtx = np.array(self.intrinsics["camera_matrix"])
                            else:
                                cam_mtx = np.array(self.extrinsics["camera_matrix"])
                            dcoeff_ex = self._normalize_dist_coeffs(self.extrinsics.get("dist_coeffs"))
                            if dcoeff_ex is None:
                                dcoeff_ex = np.zeros(5, dtype=float)
                            pts3d_reshaped = pts3d.reshape(-1, 1, 3)
                            projected, _ = cv2.projectPoints(pts3d_reshaped, self.rvec, self.tvec, cam_mtx, dcoeff_ex)
                            projected = projected.squeeze().astype(int)
                            
                            # 畫3D點
                            for pt in projected:
                                x, y = pt
                                if 0 <= x < frame_bgr.shape[1] and 0 <= y < frame_bgr.shape[0]:
                                    cv2.circle(frame_bgr, (x, y), 4, color, -1)
                            
                            # 畫骨架
                            if self.show_skeleton:
                                num_joints = pts3d.shape[0]
                                joint_pairs = JOINT_PAIRS_MAP.get(num_joints) # 從映射表中獲取骨架連接對
                                if joint_pairs:
                                    for joint_pair in joint_pairs:
                                        if (joint_pair[0] < len(projected) and joint_pair[1] < len(projected)):
                                            pt1 = projected[joint_pair[0]]
                                            pt2 = projected[joint_pair[1]]
                                            x1, y1 = pt1
                                            x2, y2 = pt2
                                            if (0 <= x1 < frame_bgr.shape[1] and 0 <= y1 < frame_bgr.shape[0] and
                                                0 <= x2 < frame_bgr.shape[1] and 0 <= y2 < frame_bgr.shape[0]):
                                                line_color = tuple(int(c * 0.7) for c in color)
                                                cv2.line(frame_bgr, (x1, y1), (x2, y2), line_color, 2)
        
        # Convert processed frame back to RGB (if it was BGR, for QImage)
        # 虛擬視頻的frame_bgr已經是BGR，直接使用
        # if isinstance(self.player, BlackVideoPlayer):
        #     frame_rgb = frame_bgr 
        # else:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Cache label size to avoid unnecessary re-scaling.
        label_size = self.video_label.size()
        if not hasattr(self, '_cached_label_size') or self._cached_label_size != label_size:
            self._cached_label_size = label_size
            self._cached_pixmap = None  # Invalidate the cache if label size changes
        
        # Always update the pixmap for the new frame.
        self._cached_pixmap = QPixmap.fromImage(q_img).scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.FastTransformation  # Faster than SmoothTransformation
        )
        
        # Temporarily disable updates to avoid flicker.
        self.video_label.setUpdatesEnabled(False)
        self.video_label.setPixmap(self._cached_pixmap)
        self.video_label.setUpdatesEnabled(True)
        
        # Compute letterbox offsets (if needed later for click mapping, etc.)
        pixmap_width = self._cached_pixmap.width()
        pixmap_height = self._cached_pixmap.height()
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        self._video_offset_x = max(0, (label_width - pixmap_width) // 2)
        self._video_offset_y = max(0, (label_height - pixmap_height) // 2)
        
        # Update info label text if video is loaded
        if self.player and self.player.frame_count:
            current_time = self.player.get_current_time()
            total_time = self.player.frame_count / self.player.fps
            self.info_label.setText(
                f"{self.format_time(current_time)} / {self.format_time(total_time)}\n"
                f"Frame: {self.player.current_frame} / {self.player.frame_count}"
            )
        
        # Restore offset line: position info_label relative to displayed video area with a 10-pixel margin.
        offset_x = self._video_offset_x + 10
        offset_y = self._video_offset_y + 10
        self.info_label.move(offset_x, offset_y)

    def change_offset(self, value):
        self.frame_offset = value
        self.update_frame() 
        
    def toggle_playback(self):
        if self.player is None:
            QMessageBox.warning(self, "Warning", "Load a video or 3D data first.")
            return
        if self.timer.isActive():
            self.timer.stop()
            self.btn_toggle.setText("Play")
            self.is_playing = False
            self.player.is_playing = False  # added: stop player playback
        else:
            if self.player.frame_count > 0: # 只有有幀數才允許播放
                self.timer.start(1000 // int(self.player.fps))
                self.btn_toggle.setText("Pause")
                self.is_playing = True
                self.player.is_playing = True  # added: start player playback
            else:
                QMessageBox.warning(self, "Warning", "No frames to play.")

    def next_frame(self):
        if self.player:
            if self.is_playing:
                self.toggle_playback()
                self.is_playing = False
            self.player.next_frame()
            self.update_frame()

    def prev_frame(self):
        if self.player:
            if self.is_playing:
                self.toggle_playback()
                self.is_playing = False
            self.player.prev_frame()
            self.update_frame()

    def jump_seconds(self, seconds):
        if self.player:
            if self.is_playing:
                self.toggle_playback()
                self.is_playing = False
            self.player.jump_seconds(seconds)
            self.update_frame()

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"
    
        # --- new helper ---
    def locate_frame(self):
        """Jump to an exact frame number."""
        if self.player:
            current = self.player.current_frame
            frame, ok = QInputDialog.getInt(
                self, "Locate Frame", "Enter frame number:",
                min=0, max=self.player.frame_count - 1,
                value=current,
            )
            if ok:
                self.player.current_frame = frame
                self.update_frame()

    # --- new helper ---
    def locate_time(self):
        """Jump to an exact timestamp (HH:MM:SS)."""
        if self.player:
            cur_time = self.player.get_current_time()
            total_time = self.player.frame_count / self.player.fps
            time_str, ok = QInputDialog.getText(
                self, "Locate Time",
                f"Enter time (HH:MM:SS, max {self.format_time(total_time)}):",
                text=self.format_time(cur_time)
            )
            if ok:
                try:
                    h, m, s = map(int, time_str.split(':'))
                    target_sec   = h*3600 + m*60 + s
                    self.player.current_frame = int(
                        max(0, min(self.player.frame_count-1, target_sec * self.player.fps))
                    )
                    self.update_frame()
                except ValueError:
                    QMessageBox.warning(self, "Invalid Time",
                                        "Please enter time in HH:MM:SS format.")

    def copy_offset(self):
        """
        Copy current frame-offset value to clipboard.
        """
        QGuiApplication.clipboard().setText(str(self.frame_offset))
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.toggle_playback()
        elif event.key() == Qt.Key_A:
            self.prev_frame()
        elif event.key() == Qt.Key_D:
            self.next_frame()
        elif event.key() == Qt.Key_Q:
            self.jump_seconds(-1)
        elif event.key() == Qt.Key_E:
            self.jump_seconds(1)
        elif event.key() == Qt.Key_W:
            # Increase sync offset count by 1
            self.offset_spin.setValue(self.offset_spin.value() + 1)
        elif event.key() == Qt.Key_S:
            # Decrease sync offset count by 1
            self.offset_spin.setValue(self.offset_spin.value() - 1)
        elif event.key() == Qt.Key_R:
            self.locate_frame()
        elif event.key() == Qt.Key_F:
            self.locate_time()
        elif event.key() == Qt.Key_Z:
            self.copy_offset()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.player is not None:
            self.player.release()
        event.accept()
        
    def update_camera_parameters(self):
        """更新相机内外参数 - 使用预设的绝对路径"""
        # 预设的绝对路径，可以根据需要修改
        intrinsics_path = os.path.join(PROJECT_ROOT, "data", "intrinsic_middle.json")
        extrinsics_path = os.path.join(PROJECT_ROOT, "data", "extrinsics_middle.json")

        # Fallback to archive/ if data/ JSONs are not present in this workspace.
        if not os.path.exists(intrinsics_path):
            intrinsics_path = os.path.join(PROJECT_ROOT, "archive", "intrinsic_middle.json")
        if not os.path.exists(extrinsics_path):
            extrinsics_path = os.path.join(PROJECT_ROOT, "archive", "extrinsics_middle.json")
        
        # 加载内参
        if os.path.exists(intrinsics_path):
            try:
                with open(intrinsics_path, 'r') as fp:
                    data = json.load(fp)
                self.intrinsics = data
                self.loaded_intrinsics_filename = os.path.basename(intrinsics_path)
                print(f"Loaded Intrinsics from {intrinsics_path}")
            except Exception as e:
                print(f"Failed to load intrinsics: {str(e)}")
                self.intrinsics = None
                self.loaded_intrinsics_filename = ""
        else:
            print(f"Intrinsics file not found: {intrinsics_path}")
            self.intrinsics = None
            self.loaded_intrinsics_filename = ""
        
        # 加载外参
        if os.path.exists(extrinsics_path):
            try:
                with open(extrinsics_path, 'r') as fp:
                    data = json.load(fp)
                self._load_extrinsics_from_json(data)
                self.loaded_extrinsics_filename = os.path.basename(extrinsics_path)
                print(f"Loaded Extrinsics from {extrinsics_path}")
            except Exception as e:
                print(f"Failed to load extrinsics: {str(e)}")
                self.extrinsics = None
                self.rvec = None
                self.tvec = None
                self.loaded_extrinsics_filename = ""
        else:
            print(f"Extrinsics file not found: {extrinsics_path}")
            self.extrinsics = None
            self.rvec = None
            self.tvec = None
            self.loaded_extrinsics_filename = ""
        
        # 更新UI显示
        self.update_loaded_files_label()
        self.update_frame()

    
    def update_camera_parameters_left(self):
        """切換到left視角的相機內外參數"""
        intrinsics_path = os.path.join(PROJECT_ROOT, "data", "intrinsic_left.json")
        extrinsics_path = os.path.join(PROJECT_ROOT, "data", "extrinsics_left.json")

        # Fallback to archive/ if data/ JSONs are not present in this workspace.
        if not os.path.exists(intrinsics_path):
            intrinsics_path = os.path.join(PROJECT_ROOT, "archive", "intrinsic_left.json")
        if not os.path.exists(extrinsics_path):
            extrinsics_path = os.path.join(PROJECT_ROOT, "archive", "extrinsics_left.json")
        # 加载内参
        if os.path.exists(intrinsics_path):
            try:
                with open(intrinsics_path, 'r') as fp:
                    data = json.load(fp)
                self.intrinsics = data
                self.loaded_intrinsics_filename = os.path.basename(intrinsics_path)
                print(f"Loaded Intrinsics from {intrinsics_path}")
            except Exception as e:
                print(f"Failed to load intrinsics: {str(e)}")
                self.intrinsics = None
                self.loaded_intrinsics_filename = ""
        else:
            print(f"Intrinsics file not found: {intrinsics_path}")
            self.intrinsics = None
            self.loaded_intrinsics_filename = ""
        # 加载外参
        if os.path.exists(extrinsics_path):
            try:
                with open(extrinsics_path, 'r') as fp:
                    data = json.load(fp)
                self._load_extrinsics_from_json(data)
                self.loaded_extrinsics_filename = os.path.basename(extrinsics_path)
                print(f"Loaded Extrinsics from {extrinsics_path}")
            except Exception as e:
                print(f"Failed to load extrinsics: {str(e)}")
                self.extrinsics = None
                self.rvec = None
                self.tvec = None
                self.loaded_extrinsics_filename = ""
        else:
            print(f"Extrinsics file not found: {extrinsics_path}")
            self.extrinsics = None
            self.rvec = None
            self.tvec = None
            self.loaded_extrinsics_filename = ""
        # 更新UI显示
        self.update_loaded_files_label()
        self.update_frame()

    def on_skeleton_checkbox_changed(self, state):
        self.show_skeleton = state == Qt.Checked
        self.update_frame()

    def export_video(self):
        if not self.player or not self.recent_video_filename:
            QMessageBox.warning(self, "Warning", "Please load a video or 3D data first.")
            return
        
        # 如果是虚拟视频，直接创建新的黑视频
        if isinstance(self.player, BlackVideoPlayer):
            video_name = "virtual_black_video"
            fps = self.player.fps
            width = self.player.width
            height = self.player.height
            total_frames = self.player.frame_count
            is_virtual = True
        else:
            # 生成导出文件名
            video_name = self.recent_video_filename
            # 用cap读取原始帧，保证分辨率和像素排列不变
            cap = cv2.VideoCapture(self.loaded_video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, "Error", "Cannot open video file for export.")
                return
            fps = self.player.fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            is_virtual = False

        # 获取所有勾选的3D点文件名
        showing_files = [self.loaded_points_files[i]['filename'] for i in sorted(self.visible_points_files) if 0 <= i < len(self.loaded_points_files)]
        showing_files_str = '+'.join(showing_files) if showing_files else 'none'
        skeleton_status = 'true' if self.show_skeleton else 'false'
        default_name = f"exported_{video_name}+{showing_files_str}+{skeleton_status}.mp4"

        save_path, _ = QFileDialog.getSaveFileName(self, "Export Video", default_name, "MP4 Files (*.mp4)")
        if not save_path:
            if not is_virtual: cap.release() # 如果是实际视频，退出前释放cap
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        # print(f"Exporting: {self.recent_video_filename}")
        print(f"Exporting: fps={fps}, width={width}, height={height}, total_frames={total_frames}")

        progress = QProgressDialog("Exporting video...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # 保存当前帧
        original_frame = self.player.current_frame
        was_playing = self.is_playing
        self.is_playing = False
        self.player.is_playing = False

        for frame_idx in range(total_frames):
            if is_virtual:
                frame = np.zeros((height, width, 3), dtype=np.uint8) # 黑色背景
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            # frame 是 BGR 格式，直接在 frame 上绘制骨架
            self.player.current_frame = frame_idx
            frame_bgr = self.draw_3d_points_and_skeleton(frame, frame_idx, bgr=True)
            # 画timer和帧号
            current_time = frame_idx / fps
            total_time = total_frames / fps
            timer_text = f"{self.format_time(current_time)} / {self.format_time(total_time)}"
            frame_text = f"Frame: {frame_idx} / {total_frames}"
            cv2.putText(frame_bgr, timer_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame_bgr, frame_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            out.write(frame_bgr)
            progress.setValue(frame_idx)
            QCoreApplication.processEvents()
            if progress.wasCanceled():
                break
        
        if not is_virtual: cap.release()
        out.release()
        progress.close()
        # 恢复 current_frame
        self.player.current_frame = original_frame
        self.update_frame()
        self.is_playing = was_playing
        self.player.is_playing = was_playing
        QMessageBox.information(self, "Export Finished", f"Video exported to {save_path}")

    def draw_3d_points_and_skeleton(self, frame_bgr, frame_idx, bgr=False):
        """在frame_bgr上繪製所有勾選的3D點和骨架，frame_idx為當前幀號。bgr=True表示frame_bgr已經是BGR格式。"""
        # 如果不是BGR格式，先转BGR
        if not bgr:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
        if self.extrinsics is not None and self.visible_points_files:
            for file_index in self.visible_points_files:
                if 0 <= file_index < len(self.loaded_points_files):
                    file_info = self.loaded_points_files[file_index]
                    points_data = file_info['data']
                    color = file_info['color']

                    points_idx = self._map_video_frame_to_points_index(
                        video_frame_idx=frame_idx,
                        video_fps=self.player.fps if self.player else 0,
                        points_fps=file_info.get('fps') or self.default_points_fps,
                        points_frame_count=points_data.shape[0],
                        offset_video_frames=self.frame_offset,
                    )

                    if points_idx is not None and 0 <= points_idx < points_data.shape[0]:
                        pts3d = points_data[points_idx].copy()
                        
                        if hasattr(self, 'flip_x_cb'):
                            if self.flip_x_cb.isChecked(): pts3d[:, 0] *= -1
                            if self.flip_y_cb.isChecked(): pts3d[:, 1] *= -1
                            if self.flip_z_cb.isChecked(): pts3d[:, 2] *= -1
                            
                        if hasattr(self, 'auto_fix_cb') and self.auto_fix_cb.isChecked():
                            self._auto_fix_extrinsic_convention_if_needed(pts3d)
                            self._auto_fix_extrinsic_horizontal_mirror_if_needed(pts3d, frame_bgr)
                        if self.rvec is not None and self.tvec is not None:
                            if self.intrinsics is not None:
                                cam_mtx = np.array(self.intrinsics["camera_matrix"])
                            else:
                                cam_mtx = np.array(self.extrinsics["camera_matrix"])
                            dcoeff_ex = self._normalize_dist_coeffs(self.extrinsics.get("dist_coeffs"))
                            if dcoeff_ex is None:
                                dcoeff_ex = np.zeros(5, dtype=float)
                            pts3d_reshaped = pts3d.reshape(-1, 1, 3)
                            projected, _ = cv2.projectPoints(pts3d_reshaped, self.rvec, self.tvec, cam_mtx, dcoeff_ex)
                            projected = projected.squeeze().astype(int)
                            # 畫點
                            for pt in projected:
                                x, y = pt
                                if 0 <= x < frame_bgr.shape[1] and 0 <= y < frame_bgr.shape[0]:
                                    cv2.circle(frame_bgr, (x, y), 4, color, -1)
                            # 畫骨架
                            if self.show_skeleton:
                                num_joints = pts3d.shape[0]
                                joint_pairs = JOINT_PAIRS_MAP.get(num_joints) # 從映射表中獲取骨架連接對
                                if joint_pairs:
                                    for joint_pair in joint_pairs:
                                        if (joint_pair[0] < len(projected) and joint_pair[1] < len(projected)):
                                            pt1 = projected[joint_pair[0]]
                                            pt2 = projected[joint_pair[1]]
                                            x1, y1 = pt1
                                            x2, y2 = pt2
                                            if (0 <= x1 < frame_bgr.shape[1] and 0 <= y1 < frame_bgr.shape[0] and
                                                0 <= x2 < frame_bgr.shape[1] and 0 <= y2 < frame_bgr.shape[0]):
                                                line_color = tuple(int(c * 0.7) for c in color)
                                                cv2.line(frame_bgr, (x1, y1), (x2, y2), line_color, 2)
        return frame_bgr

    def load_folder(self):
        """載入資料夾中的所有 NPY/CSV 檔案"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select 3D Data Folder")
        if not folder_path:
            return

        loaded_count = 0
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(('.npy', '.csv')):
                    # 透過 load_folder 載入的檔案預設為不可見
                    self.load_points(filename=file_path, is_visible_by_default=False)
                    loaded_count += 1
        
        QMessageBox.information(self, "Load Folder", f"Finished loading {loaded_count} 3D data files from {os.path.basename(folder_path)}.")
        self.update_points_list() # 確保列表是最新的
        self.update_frame()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProjectionWindow2()
    # window.resize(800, 600)  # matches main_window size
    window.show()
    sys.exit(app.exec_())
