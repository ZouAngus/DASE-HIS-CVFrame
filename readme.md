# CVFrame2 User Guide

## Introduction
This project is a PyQt5-based tool for synchronized viewing of 3D skeleton point clouds and videos. It allows you to load videos, multiple sets of 3D keypoints (npy/csv), camera intrinsic and extrinsic parameters (json), and project 3D point clouds onto the video frames. It supports switching between multiple 3D datasets, skeleton visualization, synchronization offset adjustment, and video export.

## Main Features
- Load video files (mp4)
- Load 3D keypoint data (directory/npy/csv)
- Load camera intrinsic and extrinsic parameters (json)
- Support switching and overlay display of multiple 3D datasets
- Toggle skeleton connection display
- Video frame-by-frame playback, fast forward/rewind, jump to specific frame/time
- Support exporting videos (with projected skeleton)
[- Adjust synchronization offset between video and 3D data]

## Dependency Installation
First, install Python 3.7+ and then install the following packages:
```bash
pip install PyQt5 opencv-python numpy pandas
```

## File Structure Description
- `main.py`: Launches the main program and loads the main window.
- `projection_window2.py`: Main window and all feature logic.
- `black_video_player.py`: Virtual black bacground created automatically.
- `video_player.py`: Frame-by-frame video reading and caching.
- `points3d_cache.py`: 3D point cloud data caching and access.
- `data/`: Stores camera parameters (e.g., `intrinsic_left.json`, `extrinsics_left.json`).

## Usage
1. Run `main.py` to start the program:
   ```bash
   python main.py
   ```
2. Click the buttons on the interface in order to load:
   - Video (mp4)
   - 3D data (directory/npy/csv)
   - [Optional] Camera intrinsic parameters (intrinsic json)
   - [Optional] Camera extrinsic parameters (extrinsic json)
3. Use the menu "Camera" to easily switch camera perpectives (Middle/Left)
4. In the list on the right, check multiple 3D datasets to overlay display, and toggle skeleton display.
5. Supports shortcut keys (e.g., Space for play/pause, A/D for previous/next frame, Q/E for rewind/fast forward 1 second, etc.).
6. Export videos with projected skeletons via the menu "Export".

## Camera Parameter Format
- `intrinsic_left.json` example:
  ```json
  {
      "camera_matrix": [[...], [...], [...]],
      "dist_coeffs": [[...]]
  }
  ```
- `extrinsics_left.json` example:
  ```json
  {
      "camera_matrix": [[...], [...], [...]],
      "dist_coeffs": [...],
      "extrinsics": [[[...], ...], ...]
  }
  ```

## Notes
- Only npy/csv format is supported. Please prepare data in [T, J, 3] format, where T = total frames, J = total joints.
- If unable to open video or data files, please check that the file paths and formats are correct.