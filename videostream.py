import threading
import cv2


class VideoStream:
    def __init__(self):
        self.frame = None
        self.frame_lock = threading.Lock()

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


class WebcamVideoStream:
    def __init__(self, src, width, height, f=None):
        self.src = src
        self.width = width
        self.height = height
        self.frame_lock = threading.Lock()
        self.started_lock = threading.Lock()
        self.started = False
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
        self.stream = cv2.VideoCapture(self.src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.ret, self.frame = self.stream.read()
        self.started = True
        self.thread = threading.Thread(target=self._update)
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

    def _frame_get(self):
        with self.frame_lock:
            return self.ret, self.frame[:]

    def _update(self):
        # called in self.thread
        while self._started_get():
            ret, frame = self.stream.read()
            self._frame_set(ret, frame)

    def _stop(self):
        self._started_set(False)
        self.thread.join()
        self.stream.release()

    def start(self):
        self._start()

    def read(self):
        return self._frame_get()

    def stop(self):
        self._stop()

    def xread(self, *param):
        ret, frame = self.read()
        return ret, self.f(ret, frame, *param)

if __name__=="__main__":
    cap = WebcamVideoStream(0, 800, 448)
    output = cv2.VideoWriter('a.avi', cv2.VideoWriter_fourcc(*'XVID'), 60.0, (800, 448))
    #cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 448)

    while True:
        ret, frame = cap.read()
        final = cv2.flip(frame, 1)
        cv2.imshow('1', final)
        output.write(final)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.stop()