import numpy as np

# 新增一个虚拟的VideoPlayer，用于在没有视频加载时显示骨架
class BlackVideoPlayer:
    def __init__(self, frame_count, fps=30):
        self.frame_count = frame_count
        self.fps = fps
        self.current_frame = 0
        self.is_playing = False
        self.width = 1920  # 预设宽度
        self.height = 1080 # 预设高度

    def get_frame(self):
        # 创建一个黑色的帧
        black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return black_frame

    def get_current_time(self):
        return self.current_frame / self.fps

    def next_frame(self):
        self.current_frame = min(self.frame_count - 1, self.current_frame + 1)

    def prev_frame(self):
        self.current_frame = max(0, self.current_frame - 1)

    def jump_seconds(self, seconds):
        new_frame = self.current_frame + int(seconds * self.fps)
        self.current_frame = max(0, min(self.frame_count - 1, new_frame))

    def release(self):
        pass # 虚拟播放器不需要释放资源
