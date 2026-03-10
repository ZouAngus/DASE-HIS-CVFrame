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
```bash
cd CVFrame2-v3
python main.py
```

---

## Project Structure

```
CVFrame2-v3/
 main.py                         # Entry point  launches the viewer window
 projection_window2.py           # Main UI and all projection/display logic
 video_player.py                 # Frame-by-frame video reading and caching
 black_video_player.py           # Virtual black background video player
 points3d_cache.py               # 3D point cloud data caching
 extract_24_keypoint_from_csv.py # Converts raw OptiTrack CSV to (T, 24, 3) array
 flip_video.py                   # Utility to horizontally flip video files in-place
 archive/                        # Default fallback camera parameter JSONs
    intrinsic_left.json
    intrinsic_middle.json
    extrinsics_left.json
    extrinsics_middle.json
 data/                           # Per-session 3D point data
     sword_02/
        Sword_02.csv
     trove_15/
         Trove_15.csv
```

---

## Preparing 3D Point Data

Raw motion capture CSVs from OptiTrack must be converted to a `(T, J, 3)` array before loading into the viewer.

Use `extract_24_keypoint_from_csv.py`:

```python
# Edit these values inside the script and run it
INPUT_CSV    = r"data/sword_02/Sword_02.csv"
OUTPUT_PATH  = r"sword_02_3d_points_out.csv"   # or .npy
TOTAL_FRAMES = -1   # -1 = all frames
SKIP_ROWS    = 1
OFFSET       = 0
```

```bash
python extract_24_keypoint_from_csv.py
```

Output format: `(T, 24, 3)`  T frames, 24 joints, XYZ per joint.

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

### flip_video.py  Horizontally Flip a Video In-Place

Useful when your recording camera is physically mirrored.

```python
# Edit inside the script for IDE use:
INPUT_FILES = [
    r"C:\path\to\video1.mp4",
    r"C:\path\to\video2.mp4",
]
```

Or from the terminal:
```bash
python flip_video.py video1.mp4 video2.mp4
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
