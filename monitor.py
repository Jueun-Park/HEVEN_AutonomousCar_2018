import cv2

class Monitor:
    def show(self, *frames):
        for i in range(len(frames)):
            if frames[i] is None: continue
            cv2.imshow(str(i), frames[i])