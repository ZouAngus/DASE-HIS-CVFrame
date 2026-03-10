# CVFrame2  3D Skeleton Projection Viewer

A PyQt5-based tool for synchronized viewing of 3D skeleton keypoint data projected onto video frames, with camera calibration support via OpenCV.

---

## Setup

### Requirements
- Python 3.7+
- Packages:

```bash
pip install PyQt5 opencv-python numpy pandas tqdm
```

### Running the App
make sure you modify the path in
```bash
cd CVFrame2-v3
python main.py
```

---

## Building a Standalone Executable

You can package CVFrame2 into a single executable (no Python installation needed on the target machine).

### 1. Install PyInstaller
```bash
pip install pyinstaller
```

### 2. Run the build script
```bash
python build/build.py
```

This will:
- Clean any previous `build/` and `dist/` folders
- Run PyInstaller using `CVFrame2.spec`
- Output the executable to `dist/`

### Output locations
| Platform | Output |
|----------|--------|
| Windows  | `dist/CVFrame2.exe` |
| macOS    | `dist/CVFrame2.app` |

### Notes
- The `archive/` folder (default camera JSONs) is automatically bundled into the executable.
- On **macOS**, the `.app` bundle can be dragged into `/Applications` like any normal app.
- On **Windows**, you can distribute `dist/CVFrame2.exe` as a single self-contained file.
- If you add new data files (e.g., additional JSON configs) that must ship with the app, add them to the `datas` list in `CVFrame2.spec`:
  ```python
  datas=[
      ('archive', 'archive'),
      ('your_folder', 'your_folder'),  # add here
  ],
  ```
- To add a custom app icon, place a `.ico` file (Windows) or `.icns` file (macOS) in the project root and set the `icon=` field in `CVFrame2.spec`.

---

## Project Structure

```
CVFrame2-v3/
 main.py                              # Entry point  launches the viewer window
 extract_24_keypoint_from_csv.py      # Independantly Converts raw OptiTrack CSV to (T, 24, 3) array
 flip_video.py                        # Independant utility to horizontally flip video files in-place
 tools/                               # All module source files
    projection_window2.py             # Main UI and all projection/display logic
    video_player.py                   # Frame-by-frame video reading and caching
    black_video_player.py             # Virtual black background video player
    points3d_cache.py                 # 3D point cloud data caching
 build/                               # Build scripts
    build.py                          # One-click build script
    CVFrame2.spec                     # PyInstaller spec (Windows .exe / macOS .app)
 archive/                             # Default fallback camera parameter JSONs
    intrinsic_left.json
    intrinsic_middle.json
    extrinsics_left.json
    extrinsics_middle.json
 data/                                # Per-session 3D point data
     sword_02/
        Sword_02.csv
     trove_15/
         Trove_15.csv
```

---

## Preparing 3D Point Data

> ⚠️ **Required step before running `main.py`:** Raw OptiTrack CSV files cannot be loaded directly into the viewer. They must first be converted into a `(T, 24, 3)` shaped array using `extract_24_keypoint_from_csv.py`. Only the resulting `.csv` (or `.npy`) output file should be loaded into the viewer.

Use `extract_24_keypoint_from_csv.py` from the command line:

```bash
python extract_24_keypoint_from_csv.py -input_csv path/to/raw.csv -output_csv path/to/output.csv
```

All arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `-input_csv` | `./Trove_15.csv` | Path to the raw OptiTrack CSV |
| `-output_csv` | `./trove_15_3d_points_out.csv` | Path to save the converted output |
| `-total_frames` | `-1` (all) | Limit number of frames to extract |
| `-skiprows` | `1` | Header rows to skip |
| `-offset` | `4` | Frame offset into the CSV |

Output format: `(T, 24, 3)` — T frames, 24 joints, XYZ per joint.

---

## Camera Parameter JSON Format

The viewer auto-loads from `data/` first, then falls back to `archive/`.

**Intrinsics** (`intrinsic_*.json`):
```json
{
    "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "dist_coeffs": [[k1, k2, p1, p2, k3]]
}
```

**Extrinsics** (`extrinsics_*.json`)  supports any of these keys:
```json
{
    "camera_matrix": [[...]],
    "dist_coeffs": [...],
    "best_extrinsic": [[r00,r01,r02,tx], [r10,r11,r12,ty], [r20,r21,r22,tz]]
}
```
Or use `"extrinsic"` (34) or `"extrinsics"` (list, first entry used).

>  Extrinsics must follow OpenCV convention: **world  camera** (`Xc = RXw + t`).

---

## Usage

### Loading Files
1. **Camera > Switch Perspective (middle/left)**  auto-loads the preset JSON files.
2. **Update Intrinsics / Update Extrinsics** buttons  manually pick any JSON.
3. **Load Video**  load an `.mp4` or `.avi` file.
4. **Add 3D Data**  load one `.npy` or `.csv` file (shape `TJ3`).
5. **File > Load Folder**  batch-load all `.npy`/`.csv` files in a folder (loaded unchecked by default).

### Playback Controls
| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `A` / `D` | Previous / Next frame |
| `Q` / `E` | Jump back / forward 1 second |
| `W` / `S` | Increase / Decrease frame offset by 1 |
| `R` | Jump to specific frame number |
| `F` | Jump to specific timestamp (HH:MM:SS) |
| `Z` | Copy current frame offset to clipboard |

### 3D Data Panel (right side)
- **Check/uncheck** files to toggle their overlay visibility.
- **Double-click** a file to set it as the active file (used for offset tracking).
- **Show Skeleton** checkbox  toggle skeleton bone connections.

### Axis Flip Controls
Located in the bottom control bar, next to Frame Offset:

| Control | Effect |
|---------|--------|
| **Flip X** | Multiplies all X coordinates by 1 |
| **Flip Y** | Multiplies all Y coordinates by 1 |
| **Flip Z** | Multiplies all Z coordinates by 1 |
| **Auto Fix (Invert/Mirror)** | Heuristically inverts extrinsics if points appear behind camera, or mirrors horizontally if out-of-frame. **Disabled by default**  only enable if you have not properly aligned your coordinate system. |

> **Note:** Apply the same axis flips in both calibration (`main_cali_all.py`) and here. OpenCV requires a **right-handed** coordinate system. If your mocap frame is left-handed (e.g., X-axis flipped), negate X on the mocap points before running `solvePnP` *and* before projecting here.

### Synchronization (Frame Offset)
Use the **Frame Offset** spinbox to shift the 3D data forward or backward in time relative to the video. Positive = 3D data starts later; Negative = 3D data starts earlier.

### Export
**Export > Export Video**  renders and saves an `.mp4` with all visible projected skeletons and a timestamp overlay burned in.

---

## Utilities

### flip_video.py — Horizontally Flip a Video In-Place

Useful when your recording camera is physically mirrored.

From the terminal (one or more files):
```bash
python flip_video.py -input video1.mp4 video2.mp4
```

For IDE use, edit the `IDE_INPUT_FILES` list directly inside the script:
```python
IDE_INPUT_FILES = [
    r"C:\path\to\video1.mp4",
    r"C:\path\to\video2.mp4",
]
```

The original file is overwritten with the flipped version.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Points appear mirrored horizontally | Mocap X-axis is flipped relative to OpenCV | Check **Flip X** |
| Points appear upside down | Y or Z axis mismatch | Check **Flip Y** or **Flip Z**. Also check whether the raw video is mirrored or rotated|
| Points are spatially correct but lag/lead in time | Video/mocap not synced | Adjust **Frame Offset** |
| Points look correct in calibration but shift in viewer | Axis flip applied in only one place | Apply the same flip in both `main_cali_all.py` and this viewer |
| Projection is slightly off everywhere | Residual calibration reprojection error | Re-calibrate with better corner detections or more images |
