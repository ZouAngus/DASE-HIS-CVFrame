import cv2

class VideoPlayer:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file '{video_path}'")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.is_playing = False

        # Cache for the most recent frame:
        self.cached_frame_index = -1
        self.cached_frame = None

    def get_frame(self):
        # Return the cached frame if already loaded
        if self.current_frame == self.cached_frame_index and self.cached_frame is not None:
            return self.cached_frame

        # For sequential access: if the current frame is exactly one after the cached frame,
        # then simply read the next frame without resetting the position.
        if self.cached_frame_index != -1 and self.current_frame == self.cached_frame_index + 1:
            ret, frame = self.cap.read()
            if ret:
                self.cached_frame_index += 1
                self.cached_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return self.cached_frame
            return None
        else:
            # For non-sequential access, set the frame position explicitly.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                self.cached_frame_index = self.current_frame
                self.cached_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return self.cached_frame
            return None

    def get_current_time(self):
        return self.current_frame / self.fps

    def next_frame(self):
        # Move forward one frame; if sequential, the cache remains valid.
        self.current_frame = min(self.frame_count - 1, self.current_frame + 1)

    def prev_frame(self):
        # When moving backward, the cached frame likely doesn't match.
        self.current_frame = max(0, self.current_frame - 1)

    def jump_seconds(self, seconds):
        new_frame = self.current_frame + int(seconds * self.fps)
        self.current_frame = max(0, min(self.frame_count - 1, new_frame))

    def release(self):
        self.cap.release()
