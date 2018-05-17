# 캠 영상 백그라운드에서 실시간으로 받게 해 주는 api
# 김진웅


import cv2
import threading


class VideoStream:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()

    def write(self, frame):
        if frame is None:
            print('[VideoStream] No Frame')
            return
        with self.frame_lock:
            self.frame = frame[:]

    def read(self):
        with self.frame_lock:
            if self.frame is None:
                return self.frame
            return self.frame[:]


class VideoWriteStream(VideoStream):
    def __init__(self, filesrc, fps=20.0):
        super().__init__()
        self.filesrc = filesrc
        self.fps = fps

    def initWrite(self, frame):
        isColor = False
        if frame.ndim == 3: isColor = True
        self.out = cv2.VideoWriter(self.filesrc, cv2.VideoWriter_fourcc(*'DIVX'), self.fps, (len(frame[0]), len(frame)),
                                   isColor)
        self.initWrite = (lambda x: None)

    def write(self, frame):
        self.initWrite(frame)
        if frame is None:
            print('[VideoStream] No Frame')
            return
        self.out.write(frame)
        with self.frame_lock:
            self.frame = frame

    def stop(self):
        self.out.release()

    def release(self):
        self.stop()


class WebCamVideoStream:
    def __init__(self, src, width, height):
        self.src = src
        self.width = width
        self.height = height
        self.frame_lock = threading.Lock()
        self.stop_fg = False
        self.writing = False

        self.stream = cv2.VideoCapture(self.src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.thread = threading.Thread(target=self.update)
        self.ret, self.frame = self.stream.read()
        self.out = None

    def start(self, filesrc=None, fps=60.0):
        if filesrc is not None:
            self.out = cv2.VideoWriter(filesrc, cv2.VideoWriter_fourcc(*'DIVX'), fps, (self.width, self.height))
            self.thread = threading.Thread(target=self.update_write)
            self.writing = True
        self.thread.start()

    def update(self):
        while True:
            ret, frame = self.stream.read()
            if frame is None:
                print('[WebCamVideoStream] No Frame')
                break
            with self.frame_lock:
                self.ret, self.frame = ret, frame
            if self.stop_fg is True:
                break

    def update_write(self):
        while True:
            ret, frame = self.stream.read()
            if frame is None:
                print('[WebCamVideoStream] No Frame')
                break
            self.out.write(frame)
            with self.frame_lock:
                self.ret, self.frame = ret, frame
            if self.stop_fg is True:
                break

    def read(self):
        with self.frame_lock:
            if self.frame is None:
                return self.ret, self.frame
            return self.ret, self.frame[:]

    def stop(self):
        self.stop_fg = True
        self.thread.join()
        self.stop_fg = False
        if self.writing is True:
            self.out.release()
            self.writing = False

    def release(self):
        self.stop()


if __name__ == "__main__":
    # grame = numpy.zeros((480, 640, 3), dtype=numpy.uint8)
    import time

    cap = WebCamVideoStream(0, 640, 480)
    cap.start('1.avi')
    fin = VideoWriteStream('2.avi')
    t = time.time()
    while time.time() - t < 2:
        tv = time.time()
        ret, frame = cap.read()
        final = cv2.flip(frame, 1)
        fin.write(final)
        time.sleep(0.01)
    print('OUTOUTOUTOUT')
    cap.release()
