import threading
import time

import cv2


class VideoSource:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.fps or self.fps <= 0:
            self.fps = 25.0

        self.ret, self.frame = self.cap.read()
        self.running = True

        self.is_live = isinstance(self.source, int) or str(self.source).startswith(
            ("rtsp", "http")
        )

        if self.is_live:
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret, self.frame = ret, frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(1)

    def get_frame(self):
        if self.is_live:
            return self.ret, self.frame.copy() if self.ret else None
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            return ret, frame

    def stop(self):
        self.running = False
        if self.is_live:
            self.thread.join()
        self.cap.release()
