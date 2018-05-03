import cv2
import numpy as np


class Monitor:
    def concatenate(self, f1, f2, mode='h'):
        if f1 is None: return f2
        if f2 is None: return f1
        f, g = f1, f2
        if mode == 'h':
            if len(f1) != len(f2):
                if len(f1) > len(f2): f, g = f2, f1
                z = np.zeros((len(g) - len(f), len(f[0]), 3), dtype=np.uint8)
                f = np.concatenate((f, z), axis=0)
                if len(f1) > len(f2): f, g = g, f
            res = np.concatenate((f, g), axis=1)
        elif mode == 'v':
            if len(f1[0]) != len(f2[0]):
                if len(f1[0]) > len(f2[0]): f, g = f2, f1
                z = np.zeros((len(f), len(g[0]) - len(f[0]), 3), dtype=np.uint8)
                f = np.concatenate((f, z), axis=1)
                if len(f1[0]) > len(f2[0]): f, g = g, f
            res = np.concatenate((f, g), axis=0)
        else:
            return False
        return res

    def show(self, *frames):
        for i in range(len(frames)):
            if frames[i] is None: continue
            cv2.imshow(str(i), frames[i])

if __name__ == '__main__':
    import videostream
    video = videostream.WebcamVideoStream(0, 640, 480)
    video.start()
    monitor = Monitor()
    while True:
        ret, frame = video.read()
        final = cv2.flip(frame, 1)
        final = final[200:300, 100:500]
        result = monitor.concatenate(frame, final, 'v')
        monitor.show(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    video.release()