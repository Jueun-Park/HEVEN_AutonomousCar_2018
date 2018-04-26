import threading
import cv2


class VideoStream:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()

        self.write_started = False
        self.out = None

    def _frame_set(self, frame):
        with self.frame_lock:
            self.frame = frame

    def _frame_get(self):
        with self.frame_lock:
            return self.frame[:]

    def write(self, frame):
        self._frame_set(frame)

    def read(self):
        return self._frame_get()

    def writeFileInit(self, src, width=None, height=None, fps=20):
        if width is None: width = len(src)
        if height is None: height = len(src[0])
        fcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter(src, fcc, fps, (width, height))
        self.write_started = True

    def writeFile(self, frame):
        if not self.write_started:
            print('[VideoStream] writeFileInit() must be called first')
            return
        self.out.write(frame)
        self.write(frame)

    def writeFileRelease(self):
        if not self.write_started:
            print('[VideoStream] writeFileInit() must be called first')
            return
        self.out.release()
        self.write_started = False


class WebcamVideoStream:
    def __init__(self, src, width, height, f=None):
        self.src = src
        self.width = width
        self.height = height
        self.frame_lock = threading.Lock()
        self.started_lock = threading.Lock()
        self.started = False
        self.writing = False
        self.f = f

        self.stream = None
        self.out = None
        self.ret = None
        self.frame = None
        self.thread = None

        self._start()

    def _start(self):
        if self._started_get():
            self._stop()
            print("[WebcamVideoStream] restart...")
        self.stream = cv2.VideoCapure(self.src)
        self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.height)
        self.ret, self.frame = self.stream.read()
        self.started = True
        if self.writing: self.thread = threading.Thread(target=self._write, daemon=True)
        else: self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _started_set(self, val):
        with self.started_lock:
            self.started = val

    def _started_get(self):
        with self.started_lock:
            return self.started

    def _frame_set(self, ret, frame):
        # called in self.thread
        with self.frame_lock:
            self.ret, self.frame = ret, frame

    def _frame_get(self,):
        with self.frame_lock:
            return self.ret, self.frame[:]

    def _update(self):
        # called in self.thread
        while self._started_get():
            ret, frame = self.stream.read()
            self._frame_set(ret, frame)

    def _write(self):
        # called in self.thread
        while self._started_get():
            ret, frame = self.stream.read()
            self.out.write(frame)
            self._frame_set(ret, frame)

    def _stop(self):
        self._started_set(False)
        self.thread.join()
        if self.writing:
            self.out.release()
            self.writing = False
        self.stream.release()

    def star(self):
        self._start()

    def read(self):
        return self._frame_get()

    def stop(self):
        self._stop()

    def writeFile(self, src, width=None, height=None, fps=20):
        if width is None: width = self.width
        if height is None: height = self.height
        fcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWrier(src, fcc, fps, (width, height))
        self.writing = True
        self._start()

    def xread(self, f, *param):
        if f is None: f = self.f
        return f(self.read(), *param)
