import cv2
import os
import sys

def flip_video_horizontally(input_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write to a temp file first, then replace the original
    dir_name = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    temp_path = os.path.join(dir_name, f"_temp_flip_{base_name}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    print(f"Flipping: {input_path}")
    print(f"  FPS={fps}, Size={width}x{height}, Frames={total_frames}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped = cv2.flip(frame, 1)  # 1 = horizontal flip
        out.write(flipped)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Progress: {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    # Replace original with flipped version
    os.replace(temp_path, input_path)
    print(f"Done. Saved flipped video to: {input_path}")


if __name__ == "__main__":
    # ── Edit these paths directly when running from the IDE ────────────────
    INPUT_FILES = [
        r"",
        # r"C:\path\to\another_video.mp4",
    ]
    # ───────────────────────────────────────────────────────────────────────

    # If command-line arguments are provided, they take priority over INPUT_FILES
    if len(sys.argv) >= 2:
        INPUT_FILES = sys.argv[1:]

    INPUT_FILES = [f for f in INPUT_FILES if f.strip()]

    if not INPUT_FILES:
        print("Usage: python flip_video.py <video1> [video2] ...")
        print("  Or set INPUT_FILES directly in the script.")
        sys.exit(1)

    for i, path in enumerate(INPUT_FILES):
        print(f"\n[{i+1}/{len(INPUT_FILES)}] {path}")
        flip_video_horizontally(path)
