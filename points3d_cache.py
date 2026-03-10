import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QProgressDialog, QMessageBox
from PyQt5.QtCore import QCoreApplication

class Points3DCache:
    def __init__(self, df, start_f, type_list, parent=None, preload_size=1):
        """
        Initialize the Points3DCache object.

        :param df: The pandas DataFrame containing the 3D data.
        :param start_f: The starting frame index in the DataFrame.
        :param type_list: List of column types from the DataFrame.
        :param progress_dialog: QProgressDialog for showing progress during initialization.
        :param parent: Parent widget for QMessageBox (optional).
        :param preload_size: Number of frames to preload into memory.
        """
        self.df = df
        self.start_f = start_f
        self.type_list = type_list
        self.parent = parent
        self.preload_size = preload_size
        self.total_frames = len(df) - start_f
        self.cache = {}
        self.cache_range = (0, 0)  # (start, end) of cached frames
        self.shape = (self.total_frames, 24, 3)  # Assuming 24 joints and 3D coordinates

    def _load_frames(self, start, end):
        """
        Load frames into the cache.

        :param start: Start frame index.
        :param end: End frame index.
        """
        end = min(end, self.total_frames)
        data_ = np.zeros((end - start, 24, 3))
        for frame in range(start, end):
            for joint in range(len(TARGET_JOINTS_ORDERED)):
                if len(TARGET_JOINTS_ORDERED[joint]) > 1:
                    matching_columns_0 = [i for i, value in enumerate(self.df.iloc[0].values)
                if str(value)[13:] == TARGET_JOINTS_ORDERED[joint][0][0] and self.type_list[i].startswith(TARGET_JOINTS_ORDERED[joint][0][1])][-3:]
                    matching_columns_1 = [i for i, value in enumerate(self.df.iloc[0].values)
                if str(value)[13:] == TARGET_JOINTS_ORDERED[joint][1][0] and self.type_list[i].startswith(TARGET_JOINTS_ORDERED[joint][1][1])][-3:]
                    position_0 = self.df.iloc[self.start_f + frame][matching_columns_0].values.astype(float)
                    position_1 = self.df.iloc[self.start_f + frame][matching_columns_1].values.astype(float)
                    final_ = (position_0 + position_1) / 2
                    data_[frame - start, joint] = final_
                else:
                    matching_columns = [i for i, value in enumerate(self.df.iloc[0].values)
                                        if str(value)[13:] == TARGET_JOINTS_ORDERED[joint][0][0] and
                                        self.type_list[i].startswith(TARGET_JOINTS_ORDERED[joint][0][1])][-3:]
                    data_[frame - start, joint] = np.array(self.df.iloc[self.start_f + frame][matching_columns].values, dtype=float)
            # self.progress_dialog.setValue(frame)
            # QCoreApplication.processEvents()
        self.cache = {i: data_[i - start] for i in range(start, end)}
        self.cache_range = (start, end)

    def __getitem__(self, frame):
        """
        Get the 3D data for a specific frame.

        :param frame: Frame index.
        :return: 3D data for the frame.
        """
        if frame < 0 or frame >= self.total_frames:
            raise IndexError("Frame index out of range.")
        if frame < self.cache_range[0] or frame >= self.cache_range[1]:
            self._load_frames(frame, frame + self.preload_size)
        return self.cache[frame]

TARGET_JOINTS_ORDERED = {
    0:  [('Hip','Bone')],
    1:  [('LThigh','Bone')],
    2:  [('RThigh','Bone')],
    3:  [('Ab','Bone')],
    4:  [('LShin','Bone')],
    5:  [('RShin','Bone')],
    6:  [('BackLeft','Bone Marker'),('BackRight','Bone Marker')],
    7:  [('LFoot','Bone')],
    8:  [('RFoot','Bone')],
    9:  [('BackTop','Bone Marker')],
    10: [('LToe','Bone')],
    11: [('RToe','Bone')],
    12: [('Neck','Bone')], 
    13: [('LShoulder','Bone')],
    14: [('RShoulder','Bone')],
    15: [('Head','Bone')],
    16: [('LUArm','Bone')],
    17: [('RUArm','Bone')],
    18: [('LFArm','Bone')],
    19: [('RFArm','Bone')],
    20: [('LWristIn','Bone Marker'),('LWristOut','Bone Marker')],
    21: [('RWristIn','Bone Marker'),('RWristOut','Bone Marker')],
    22: [('RHandOut','Bone Marker')],
    23: [('RHandOut','Bone Marker')],
}