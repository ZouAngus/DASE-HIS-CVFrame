"""
clip_annotator.py - Action Clip Preview & Export Tool
Usage:  python clip_annotator.py
"""

import sys, os, json, copy
import cv2
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QFileDialog, QSlider, QSpinBox, QComboBox, QCheckBox,
    QProgressDialog, QMessageBox, QGroupBox, QFormLayout, QSizePolicy,
    QInputDialog, QMenu, QAction,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QCoreApplication

CAMERA_NAMES = [
    "topleft", "topcenter", "topright",
    "bottomleft", "bottomcenter", "bottomright",
    "diagonal",
]
DEFAULT_POINTS_FPS = 60.0

JOINT_PAIRS_24 = [
    (6,9),(12,9),(12,15),(20,18),(18,16),(16,13),(13,6),(14,6),(14,17),
    (17,19),(19,21),(3,6),(0,3),(1,0),(2,0),(10,7),(7,4),(4,1),(2,5),(5,8),(11,8),
]
JOINT_PAIRS_17 = [
    (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),(8,14),(14,15),(15,16),
]
JOINT_PAIRS_MAP = {17: JOINT_PAIRS_17, 24: JOINT_PAIRS_24}
PT_COLOR = (0, 0, 255)


def parse_excel_actions(xlsx_path, sheet_name):
    """Parse action rows. Handles variable column layouts and multi-repetition columns."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    hdr = {str(c).strip().lower(): i for i, c in enumerate(df.columns)}
    action_col = hdr.get("action")
    no_col = hdr.get("no.")
    if action_col is None:
        best = (-1, -1)
        for i in range(len(df.columns)):
            n = int(df.iloc[:, i].apply(lambda x: isinstance(x, str)).sum())
            if n > best[0]: best = (n, i)
        action_col = best[1]
    # Find all numeric columns with mean > 10 (frame-like values)
    num_cols = []
    for i in range(len(df.columns)):
        vals = pd.to_numeric(df.iloc[:, i], errors="coerce").dropna()
        if len(vals) >= 2 and float(vals.mean()) > 10: num_cols.append(i)
    if len(num_cols) < 2: return []
    # Pick the PRIMARY start/end pair: consecutive pair with the highest
    # combined mean (frame numbers >> time-in-seconds or row numbers)
    best_pair = (num_cols[-2], num_cols[-1])
    best_score = -1
    for pi in range(len(num_cols) - 1):
        ci, cj = num_cols[pi], num_cols[pi + 1]
        vi = pd.to_numeric(df.iloc[:, ci], errors="coerce").dropna()
        vj = pd.to_numeric(df.iloc[:, cj], errors="coerce").dropna()
        count = 0
        for idx in range(len(df)):
            sv = df.iloc[idx, ci]; ev = df.iloc[idx, cj]
            try:
                s = int(float(sv)); e = int(float(ev))
                if s > 0 and e > s: count += 1
            except (TypeError, ValueError): pass
        if count < 2: continue
        # Score: strongly prefer more rows; use mean as tiebreaker
        score = count * 100000 + float(vi.mean()) + float(vj.mean())
        if score > best_score:
            best_score = score; best_pair = (ci, cj)
    start_col, end_col = best_pair
    # Extra repetition columns: pairs of columns that come AFTER end_col
    extra_pairs = []
    remaining = [c for c in num_cols if c > end_col]
    for ri in range(0, len(remaining) - 1, 2):
        extra_pairs.append((remaining[ri], remaining[ri + 1]))
    variant_col = None
    cand = action_col + 1
    if cand < len(df.columns) and cand not in (start_col, end_col): variant_col = cand
    act_series = df.iloc[:, action_col].copy().ffill()
    rows = []
    for idx in range(len(df)):
        # Get base info for this row
        aname = str(act_series.iloc[idx]).strip() if pd.notna(act_series.iloc[idx]) else "?"
        variant = ""
        if variant_col is not None:
            v = df.iloc[idx, variant_col]
            if pd.notna(v): variant = str(v).strip()
        no_val = None
        if no_col is not None:
            nv = df.iloc[idx, no_col]
            if pd.notna(nv):
                try: no_val = int(float(nv))
                except Exception: pass
        # Primary start/end
        try:
            sf = int(float(df.iloc[idx, start_col]))
            ef = int(float(df.iloc[idx, end_col]))
        except (TypeError, ValueError): sf = ef = 0
        if sf > 0 and ef > sf:
            a = dict(no=no_val, action=aname, variant=variant, start=sf, end=ef)
            a["label"] = make_label(a)
            rows.append(a)
        # Extra repetition pairs from the same row
        for rep_i, (rc_s, rc_e) in enumerate(extra_pairs):
            try:
                rs = int(float(df.iloc[idx, rc_s]))
                re_ = int(float(df.iloc[idx, rc_e]))
            except (TypeError, ValueError): continue
            if rs > 0 and re_ > rs:
                a2 = dict(no=no_val, action=aname, variant=variant,
                          start=rs, end=re_, rep=f"rep{rep_i + 2}")
                a2["label"] = make_label(a2)
                rows.append(a2)
    return rows


def load_calibration(cal_dir, cam_name):
    intr = extr = None
    for fn in os.listdir(cal_dir):
        fl = fn.lower()
        if cam_name not in fl: continue
        fp = os.path.join(cal_dir, fn)
        try:
            with open(fp) as f: d = json.load(f)
        except Exception: continue
        if "intrinsic" in fl: intr = d
        elif "extrinsic" in fl: extr = d
    return intr, extr


def _rvec_tvec(extr):
    ext = None
    for k in ("best_extrinsic", "extrinsic", "extrinsics"):
        if k not in extr: continue
        v = extr[k]
        if k == "extrinsics" and isinstance(v, list) and v: v = v[0]
        ext = np.array(v, dtype=float); break
    if ext is None: return None, None
    if ext.shape == (4, 4): ext = ext[:3, :]
    if ext.shape != (3, 4): return None, None
    R = ext[:, :3]; t = ext[:, 3].reshape(3, 1)
    rv, _ = cv2.Rodrigues(R)
    return rv, t


def project_pts(pts3d, intr, extr, flip_x=False, flip_y=False, flip_z=False):
    pts = pts3d.copy().astype(float)
    if flip_x: pts[:, 0] *= -1
    if flip_y: pts[:, 1] *= -1
    if flip_z: pts[:, 2] *= -1
    rv, tv = _rvec_tvec(extr)
    if rv is None: return None
    cm = np.array(intr["camera_matrix"], dtype=float)
    dc_raw = intr.get("dist_coeffs") or extr.get("dist_coeffs")
    dc = np.array(dc_raw, dtype=float).reshape(-1) if dc_raw is not None else np.zeros(5)
    proj, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rv, tv, cm, dc)
    return proj.squeeze().astype(int)


def draw_skel(frame, proj, color=PT_COLOR):
    h, w = frame.shape[:2]; n = len(proj)
    bc = tuple(int(c * 0.7) for c in color)
    for pt in proj:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h: cv2.circle(frame, (x, y), 4, color, -1)
    for i, j in JOINT_PAIRS_MAP.get(n, []):
        if i < n and j < n:
            x1, y1 = int(proj[i][0]), int(proj[i][1])
            x2, y2 = int(proj[j][0]), int(proj[j][1])
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), bc, 2)


def v2p(vf, vfps, pfps, ptot, off=0):
    if vfps <= 0: vfps = 30.0
    if pfps <= 0: pfps = vfps
    idx = int(round((vf + off) * (pfps / vfps)))
    return max(0, min(ptot - 1, idx))


def fmt_time(sec):
    return f"{int(sec//3600):02d}:{int((sec%3600)//60):02d}:{int(sec%60):02d}"


def make_label(a, ov=None):
    """Build display label for an action entry."""
    if ov is None: ov = {}
    s = ov.get("start", a["start"]); e = ov.get("end", a["end"])
    rep = a.get("rep")  # repetition tag, e.g. "rep2"
    lbl = (f"#{a['no']} " if a.get("no") else "") + a["action"]
    if a.get("variant"): lbl += f" [{a['variant']}]"
    if rep: lbl += f" {rep}"
    lbl += f"  ({s}-{e})"
    off = ov.get("offset", 0)
    if off != 0: lbl += f" off={off}"
    return lbl


class ClipAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Clip Annotator - Multi-View Action Clip Tool")
        self.setGeometry(60, 60, 1440, 840)
        self.setFocusPolicy(Qt.StrongFocus)
        self.xlsx_path = self.sheet_name = self.data_folder = self.cal_folder = None
        self.actions = []
        self.cur_act = -1
        self.avail_cams = []
        self.active_cam = None
        self.cap = None
        self.vfps = 30.0
        self.vtotal = 0
        self.pts3d = None
        self.pfps = DEFAULT_POINTS_FPS
        self.calibs = {}
        self.scene_offset = 0
        self.overrides = {}
        self.cur_frame = 0
        self.clip_start = self.clip_end = 0
        self.playing = False
        self.show_skel = True
        self.flip = [False, False, False]
        self._suppress_spin = False
        # Frame cache: avoid redundant decoding
        self._cached_frame_idx = -1
        self._cached_frame = None
        self._build_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

    def _build_ui(self):
        mb = self.menuBar()
        fm = mb.addMenu("File")
        fm.addAction("Load Excel...", self._load_xlsx)
        fm.addAction("Load Data Folder...", self._load_data)
        fm.addAction("Load Calibration Folder...", self._load_cal)
        em = mb.addMenu("Export")
        em.addAction("Export Current Clip (all cams)...", lambda: self._export(False))
        em.addAction("Export ALL Clips (all cams)...", lambda: self._export(True))

        root = QWidget(); self.setCentralWidget(root)
        hl = QHBoxLayout(root); hl.setContentsMargins(4,4,4,4)

        # LEFT
        left = QWidget(); left.setFixedWidth(310)
        lv = QVBoxLayout(left); lv.setContentsMargins(0,0,0,0)
        sg = QGroupBox("Sources"); sf = QFormLayout(sg)
        self.lbl_xlsx = QLabel("-"); self.lbl_data = QLabel("-"); self.lbl_cal = QLabel("-")
        sf.addRow("Excel:", self.lbl_xlsx)
        sf.addRow("Data:", self.lbl_data)
        sf.addRow("Calib:", self.lbl_cal)
        lv.addWidget(sg)
        lv.addWidget(QLabel("Actions (right-click to add repetition):"))
        self.act_list = QListWidget()
        self.act_list.currentRowChanged.connect(self._on_act_sel)
        self.act_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.act_list.customContextMenuRequested.connect(self._act_context_menu)
        lv.addWidget(self.act_list)
        hl.addWidget(left)

        # CENTER
        center = QWidget(); cvl = QVBoxLayout(center); cvl.setContentsMargins(0,0,0,0)
        self.vid_lbl = QLabel()
        self.vid_lbl.setAlignment(Qt.AlignCenter)
        self.vid_lbl.setMinimumSize(640, 400)
        self.vid_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vid_lbl.setStyleSheet("background:black;")
        cvl.addWidget(self.vid_lbl)
        self.info_lbl = QLabel("Frame: - / -   Time: - / -")
        self.info_lbl.setStyleSheet("font-size:13px; padding:2px;")
        cvl.addWidget(self.info_lbl)
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider)
        cvl.addWidget(self.slider)
        br = QHBoxLayout()
        for txt, fn in [("<< -1s", lambda: self._jump(-1)), ("< Prev", self._prev),
                         ("Play / Pause", self._toggle_play), ("Next >", self._nxt),
                         ("+1s >>", lambda: self._jump(1))]:
            b = QPushButton(txt); b.clicked.connect(fn); br.addWidget(b)
        cvl.addLayout(br)
        hl.addWidget(center, stretch=1)

        # RIGHT
        right = QWidget(); right.setFixedWidth(260)
        rv = QVBoxLayout(right); rv.setContentsMargins(0,0,0,0)
        cg = QGroupBox("Camera"); cf = QFormLayout(cg)
        self.cam_combo = QComboBox()
        self.cam_combo.currentTextChanged.connect(self._on_cam)
        cf.addRow("View:", self.cam_combo)
        rv.addWidget(cg)

        og = QGroupBox("Sync (3D offset)"); of2 = QFormLayout(og)
        self.scene_off_spin = QSpinBox(); self.scene_off_spin.setRange(-50000, 50000)
        self.scene_off_spin.valueChanged.connect(self._on_scene_off)
        of2.addRow("Scene:", self.scene_off_spin)
        self.act_off_spin = QSpinBox(); self.act_off_spin.setRange(-50000, 50000)
        self.act_off_spin.valueChanged.connect(self._on_act_off)
        of2.addRow("Action:", self.act_off_spin)
        rv.addWidget(og)

        ag = QGroupBox("Action Override"); af = QFormLayout(ag)
        self.start_spin = QSpinBox(); self.start_spin.setRange(0, 9999999)
        self.start_spin.valueChanged.connect(self._on_start_ov)
        af.addRow("Start:", self.start_spin)
        self.end_spin = QSpinBox(); self.end_spin.setRange(0, 9999999)
        self.end_spin.valueChanged.connect(self._on_end_ov)
        af.addRow("End:", self.end_spin)
        rv.addWidget(ag)

        fg = QGroupBox("Display"); ff = QFormLayout(fg)
        self.skel_cb = QCheckBox("Show Skeleton"); self.skel_cb.setChecked(True)
        self.skel_cb.stateChanged.connect(lambda s: setattr(self, "show_skel", s == Qt.Checked) or self._show_frame())
        ff.addRow(self.skel_cb)
        for axis, idx in [("Flip X", 0), ("Flip Y", 1), ("Flip Z", 2)]:
            cb = QCheckBox(axis)
            cb.stateChanged.connect(lambda s, i=idx: self._set_flip(i, s == Qt.Checked))
            ff.addRow(cb)
        rv.addWidget(fg)
        rv.addStretch()
        hl.addWidget(right)

        self.statusBar().showMessage("Space=Play  A/D=Prev/Next  Q/E=-/+1s  W/S=SceneOffset  Up/Down=Action")

    def _set_flip(self, idx, val):
        self.flip[idx] = val; self._show_frame()

    # ---- File loaders ----

    def _load_xlsx(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Excel", "", "Excel (*.xlsx *.xls)")
        if not path: return
        try: xl = pd.ExcelFile(path)
        except Exception as e: QMessageBox.critical(self, "Error", str(e)); return
        sheet, ok = QInputDialog.getItem(self, "Sheet", "Select scene sheet:", xl.sheet_names, 0, False)
        if not ok: return
        self.xlsx_path = path; self.sheet_name = sheet
        self.actions = parse_excel_actions(path, sheet)
        self.overrides = {}
        self.lbl_xlsx.setText(f"{os.path.basename(path)} [{sheet}]")
        self._refresh_act_list()
        if self.actions: self.act_list.setCurrentRow(0)
        self.statusBar().showMessage(f"Loaded {len(self.actions)} actions from {sheet}")

    def _load_data(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if not folder: return
        self.data_folder = folder
        self.lbl_data.setText(os.path.basename(folder))
        self.avail_cams = []
        for cn in CAMERA_NAMES:
            for fn in os.listdir(folder):
                if cn in fn.lower() and fn.lower().endswith(".mp4"):
                    self.avail_cams.append(cn); break
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        self.cam_combo.addItems(self.avail_cams)
        self.cam_combo.blockSignals(False)
        self.pts3d = None
        for fn in os.listdir(folder):
            if fn.lower().startswith("extracted") and fn.lower().endswith(".csv"):
                csvp = os.path.join(folder, fn)
                df = pd.read_csv(csvp)
                df = df.apply(pd.to_numeric, errors="coerce")
                nc = df.shape[1]
                if nc % 3 == 0:
                    self.pts3d = df.values.reshape(-1, nc // 3, 3)
                break
        if self.avail_cams: self._switch_cam(self.avail_cams[0])
        self._estimate_pfps()
        self.statusBar().showMessage(f"Data: {len(self.avail_cams)} cams, pts={self.pts3d.shape if self.pts3d is not None else None}")

    def _load_cal(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Calibration Folder")
        if not folder: return
        self.cal_folder = folder
        self.lbl_cal.setText(os.path.basename(folder))
        self.calibs = {}
        for cn in CAMERA_NAMES:
            intr, extr = load_calibration(folder, cn)
            if intr and extr: self.calibs[cn] = (intr, extr)
        self.statusBar().showMessage(f"Loaded calibration for {len(self.calibs)} cameras")
        self._show_frame()

    def _estimate_pfps(self):
        if self.pts3d is not None and self.vtotal > 0 and self.vfps > 0:
            dur = self.vtotal / self.vfps
            if dur > 0: self.pfps = self.pts3d.shape[0] / dur; return
        self.pfps = DEFAULT_POINTS_FPS

    def _switch_cam(self, cam_name):
        """Switch camera: open new VideoCapture, invalidate frame cache."""
        if self.cap: self.cap.release(); self.cap = None
        self.active_cam = cam_name
        self._cached_frame_idx = -1
        self._cached_frame = None
        if not self.data_folder: return
        for fn in os.listdir(self.data_folder):
            if cam_name in fn.lower() and fn.lower().endswith(".mp4"):
                vpath = os.path.join(self.data_folder, fn)
                self.cap = cv2.VideoCapture(vpath)
                if self.cap.isOpened():
                    self.vfps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                    self.vtotal = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self._estimate_pfps()
                break
        self._read_frame(self.cur_frame)
        self._show_frame()

    # ---- Frame cache (key perf fix) ----

    def _read_frame(self, frame_idx):
        """Read a video frame. Uses cache to skip redundant seeks/decodes."""
        if frame_idx == self._cached_frame_idx and self._cached_frame is not None:
            return self._cached_frame
        if not self.cap or not self.cap.isOpened():
            return None
        cur_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        # If next frame is sequential, just read (no seek needed)
        if cur_pos != frame_idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self._cached_frame_idx = frame_idx
            self._cached_frame = frame.copy()
            return self._cached_frame
        return None

    # ---- Action list helpers ----

    def _refresh_act_list(self):
        self.act_list.clear()
        for i, a in enumerate(self.actions):
            ov = self.overrides.get(i, {})
            self.act_list.addItem(make_label(a, ov))

    def _act_context_menu(self, pos):
        """Right-click context menu on action list."""
        item = self.act_list.itemAt(pos)
        if item is None: return
        row = self.act_list.row(item)
        menu = QMenu(self)
        add_rep = menu.addAction("Add repetition (duplicate with new start/end)...")
        delete_act = menu.addAction("Delete this entry")
        chosen = menu.exec_(self.act_list.viewport().mapToGlobal(pos))
        if chosen == add_rep:
            self._add_repetition(row)
        elif chosen == delete_act:
            self._delete_action(row)

    def _add_repetition(self, src_row):
        """Duplicate an action entry with new start/end frames."""
        if src_row < 0 or src_row >= len(self.actions): return
        a = self.actions[src_row]
        # Ask for new start frame
        new_start, ok1 = QInputDialog.getInt(self, "New repetition",
            f"Start frame for new repetition of \"{a['action']}\":",
            value=a["end"] + 1, min=0, max=9999999)
        if not ok1: return
        new_end, ok2 = QInputDialog.getInt(self, "New repetition",
            f"End frame:", value=new_start + (a["end"] - a["start"]), min=new_start + 1, max=9999999)
        if not ok2: return
        # Count existing reps for this action
        base_action = a["action"]
        base_variant = a.get("variant", "")
        rep_count = 0
        for aa in self.actions:
            if aa["action"] == base_action and aa.get("variant", "") == base_variant:
                rep_count += 1
        new_a = dict(no=a["no"], action=a["action"], variant=a.get("variant", ""),
                     start=new_start, end=new_end, rep=f"rep{rep_count + 1}",
                     label="")
        new_a["label"] = make_label(new_a)
        # Insert right after the source row
        insert_idx = src_row + 1
        self.actions.insert(insert_idx, new_a)
        # Shift overrides indices
        new_ov = {}
        for k, v in self.overrides.items():
            if k >= insert_idx: new_ov[k + 1] = v
            else: new_ov[k] = v
        self.overrides = new_ov
        self._refresh_act_list()
        self.act_list.setCurrentRow(insert_idx)
        self.statusBar().showMessage(f"Added repetition: {new_a['label']}")

    def _delete_action(self, row):
        if row < 0 or row >= len(self.actions): return
        a = self.actions[row]
        reply = QMessageBox.question(self, "Confirm",
            f"Delete \"{make_label(a)}\"?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes: return
        self.actions.pop(row)
        self.overrides.pop(row, None)
        new_ov = {}
        for k, v in self.overrides.items():
            if k > row: new_ov[k - 1] = v
            elif k < row: new_ov[k] = v
        self.overrides = new_ov
        self._refresh_act_list()
        if self.actions:
            self.act_list.setCurrentRow(min(row, len(self.actions) - 1))

    # ---- Action selection & overrides ----

    def _get_effective_act_offset(self, row):
        """Get action offset: own override, or inherit from nearest previous action."""
        ov = self.overrides.get(row, {})
        if "offset" in ov:
            return ov["offset"]
        # Walk backwards to find nearest action with an explicit offset
        for r in range(row - 1, -1, -1):
            prev_ov = self.overrides.get(r, {})
            if "offset" in prev_ov:
                return prev_ov["offset"]
        return 0

    def _on_act_sel(self, row):
        if row < 0 or row >= len(self.actions): return
        self.cur_act = row
        a = self.actions[row]
        ov = self.overrides.get(row, {})
        s = ov.get("start", a["start"])
        e = ov.get("end", a["end"])
        self.clip_start = s; self.clip_end = e
        self._suppress_spin = True
        self.start_spin.setValue(s)
        self.end_spin.setValue(e)
        self.act_off_spin.setValue(self._get_effective_act_offset(row))
        # Scene offset is persistent - do NOT reset it here
        self._suppress_spin = False
        self.slider.setRange(s, e)
        self.cur_frame = s
        self.slider.setValue(s)
        self._read_frame(self.cur_frame)
        self._show_frame()

    def _on_start_ov(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        ov = self.overrides.setdefault(self.cur_act, {})
        ov["start"] = val
        self.clip_start = val
        self.slider.setRange(val, self.clip_end)
        self._update_act_label()

    def _on_end_ov(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        ov = self.overrides.setdefault(self.cur_act, {})
        ov["end"] = val
        self.clip_end = val
        self.slider.setRange(self.clip_start, val)
        self._update_act_label()

    def _on_act_off(self, val):
        if self._suppress_spin or self.cur_act < 0: return
        self.overrides.setdefault(self.cur_act, {})["offset"] = val
        self._show_frame()

    def _on_scene_off(self, val):
        self.scene_offset = val; self._show_frame()

    def _on_cam(self, text):
        if text and text != self.active_cam:
            self._switch_cam(text)

    def _update_act_label(self):
        if self.cur_act < 0: return
        a = self.actions[self.cur_act]
        ov = self.overrides.get(self.cur_act, {})
        item = self.act_list.item(self.cur_act)
        if item: item.setText(make_label(a, ov))

    # ---- Playback ----

    def _on_slider(self, val):
        self.cur_frame = val
        self._read_frame(val)
        self._show_frame()

    def _toggle_play(self):
        if self.playing:
            self.playing = False; self.timer.stop()
        else:
            if not self.cap: return
            self.playing = True
            # Seek to current position before starting sequential read
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame)
            self.timer.start(int(1000 / self.vfps))

    def _tick(self):
        """Timer tick: sequential read (no seek) for smooth playback."""
        if self.cur_frame >= self.clip_end:
            self.playing = False; self.timer.stop(); return
        self.cur_frame += 1
        # Sequential read - much faster than seek + read
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self._cached_frame_idx = self.cur_frame
                self._cached_frame = frame
        self.slider.blockSignals(True)
        self.slider.setValue(self.cur_frame)
        self.slider.blockSignals(False)
        self._show_frame()

    def _prev(self):
        if self.playing: self._toggle_play()
        if self.cur_frame > self.clip_start:
            self.cur_frame -= 1
            self._read_frame(self.cur_frame)
            self.slider.setValue(self.cur_frame)

    def _nxt(self):
        if self.playing: self._toggle_play()
        if self.cur_frame < self.clip_end:
            self.cur_frame += 1
            self._read_frame(self.cur_frame)
            self.slider.setValue(self.cur_frame)

    def _jump(self, secs):
        if self.playing: self._toggle_play()
        delta = int(secs * self.vfps)
        nf = max(self.clip_start, min(self.clip_end, self.cur_frame + delta))
        self.cur_frame = nf
        self._read_frame(nf)
        self.slider.setValue(nf)

    # ---- Rendering (display only, uses cached frame) ----

    def _show_frame(self):
        """Render the cached frame with skeleton overlay. No video I/O here."""
        raw = self._cached_frame
        if raw is None: return
        frame = raw.copy()
        # project 3D skeleton
        if (self.show_skel and self.pts3d is not None
                and self.active_cam and self.active_cam in self.calibs):
            intr, extr = self.calibs[self.active_cam]
            total_off = self.scene_offset + self._get_effective_act_offset(self.cur_act)
            pidx = v2p(self.cur_frame, self.vfps, self.pfps, self.pts3d.shape[0], total_off)
            pts = self.pts3d[pidx]
            proj = project_pts(pts, intr, extr, self.flip[0], self.flip[1], self.flip[2])
            if proj is not None:
                draw_skel(frame, proj)
        # convert + display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.vid_lbl.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.vid_lbl.setPixmap(pix)
        # info
        vf = self.vfps if self.vfps > 0 else 30.0
        t_cur = self.cur_frame / vf
        t_tot = self.vtotal / vf
        t_cs = self.clip_start / vf
        t_ce = self.clip_end / vf
        self.info_lbl.setText(
            f"Frame: {self.cur_frame} / {self.vtotal}   "
            f"Time: {fmt_time(t_cur)} / {fmt_time(t_tot)}   "
            f"Clip: {self.clip_start}-{self.clip_end} ({fmt_time(t_cs)}-{fmt_time(t_ce)})"
        )

    # ---- Keyboard ----

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Space: self._toggle_play()
        elif k == Qt.Key_A: self._prev()
        elif k == Qt.Key_D: self._nxt()
        elif k == Qt.Key_Q: self._jump(-1)
        elif k == Qt.Key_E: self._jump(1)
        elif k == Qt.Key_W: self.scene_off_spin.setValue(self.scene_off_spin.value() + 1)
        elif k == Qt.Key_S: self.scene_off_spin.setValue(self.scene_off_spin.value() - 1)
        elif k == Qt.Key_Up:
            r = max(0, self.act_list.currentRow() - 1)
            self.act_list.setCurrentRow(r)
        elif k == Qt.Key_Down:
            r = min(len(self.actions) - 1, self.act_list.currentRow() + 1)
            self.act_list.setCurrentRow(r)
        else: super().keyPressEvent(event)

    # ---- Export ----

    def _export(self, all_actions):
        if not self.actions:
            QMessageBox.warning(self, "Warning", "No actions loaded."); return
        if not self.data_folder:
            QMessageBox.warning(self, "Warning", "No data folder loaded."); return
        if not self.calibs:
            QMessageBox.warning(self, "Warning", "No calibration loaded."); return
        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not out_dir: return
        indices = list(range(len(self.actions))) if all_actions else (
            [self.cur_act] if self.cur_act >= 0 else [])
        if not indices:
            QMessageBox.warning(self, "Warning", "No action selected."); return
        total_ops = len(indices) * len(self.avail_cams)
        prog = QProgressDialog("Exporting...", "Cancel", 0, total_ops, self)
        prog.setWindowModality(Qt.WindowModal); prog.setMinimumDuration(0)
        op = 0
        for ai in indices:
            a = self.actions[ai]
            ov = self.overrides.get(ai, {})
            sf = ov.get("start", a["start"])
            ef = ov.get("end", a["end"])
            total_off = self.scene_offset + self._get_effective_act_offset(ai)
            safe = f"{ai:03d}_{a['action']}"
            if a.get("variant"): safe += f"_{a['variant']}"
            if a.get("rep"): safe += f"_{a['rep']}"
            safe = safe.replace(" ", "_").replace("/", "_")
            act_dir = os.path.join(out_dir, safe)
            os.makedirs(act_dir, exist_ok=True)
            # points csv
            if self.pts3d is not None:
                pi_s = v2p(sf, self.vfps, self.pfps, self.pts3d.shape[0], total_off)
                pi_e = v2p(ef, self.vfps, self.pfps, self.pts3d.shape[0], total_off)
                sl = self.pts3d[pi_s:pi_e+1]
                nj = sl.shape[1]
                cols = []
                for j in range(nj): cols.extend([f"{j}_x", f"{j}_y", f"{j}_z"])
                pd.DataFrame(sl.reshape(sl.shape[0], -1), columns=cols).to_csv(
                    os.path.join(act_dir, "points3d.csv"), index=False)
            # video per camera
            for cn in self.avail_cams:
                if prog.wasCanceled(): break
                vpath = None
                for fn in os.listdir(self.data_folder):
                    if cn in fn.lower() and fn.lower().endswith(".mp4"):
                        vpath = os.path.join(self.data_folder, fn); break
                if not vpath: op += 1; prog.setValue(op); continue
                cap2 = cv2.VideoCapture(vpath)
                if not cap2.isOpened(): op += 1; prog.setValue(op); continue
                fps = cap2.get(cv2.CAP_PROP_FPS) or 30.0
                w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_path = os.path.join(act_dir, f"{cn}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                ie = self.calibs.get(cn)
                cap2.set(cv2.CAP_PROP_POS_FRAMES, sf)
                for fi in range(sf, ef + 1):
                    ret, frm = cap2.read()
                    if not ret: break
                    if self.pts3d is not None and ie:
                        intr, extr = ie
                        pidx = v2p(fi, fps, self.pfps, self.pts3d.shape[0], total_off)
                        proj = project_pts(self.pts3d[pidx], intr, extr,
                                           self.flip[0], self.flip[1], self.flip[2])
                        if proj is not None and self.show_skel:
                            draw_skel(frm, proj)
                    t = fi / fps
                    cv2.putText(frm, f"{fmt_time(t)} F:{fi}", (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                    writer.write(frm)
                writer.release(); cap2.release()
                op += 1; prog.setValue(op)
                QCoreApplication.processEvents()
        prog.close()
        QMessageBox.information(self, "Done", f"Exported to {out_dir}")

    def closeEvent(self, event):
        if self.cap: self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ClipAnnotator()
    win.show()
    sys.exit(app.exec_())
