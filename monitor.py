# 주행 중 모니터링 프로그램
# 김진웅


import cv2
import numpy as np


class Monitor:
    def __init__(self):
        self.color_rgv = None
        self.color_hsv = None
        self.color_buf = []
        self.wname_buf = []
        self.windows_str = []
        self.windows_is = {}

    # color picker #
    def color_picker(self, event, x, y, flags, frame):
        self.color_rgv = frame[y][x]
        rgv_frame = np.uint8([[self.color_rgv]])
        hsv_frame = cv2.cvtColor(rgv_frame, cv2.COLOR_BGR2HSV)
        self.color_hsv = np.squeeze(hsv_frame)
        if event == cv2.EVENT_LBUTTONUP:
            rgv = frame[y][x]
            hsv = np.squeeze(hsv_frame)
            self.color_buf.append(list((rgv, hsv)))
        elif event == cv2.EVENT_RBUTTONUP:
            if len(self.color_buf) != 0:
                if flags == cv2.EVENT_FLAG_CTRLKEY:
                    self.color_buf = self.color_buf[1:]
                else:
                    self.color_buf.pop()

    def imcolorbuf(self):
        f = np.zeros((400, 280, 3), dtype=np.uint8)
        position = 1
        for color in self.color_buf:
            f = cv2.putText(f, '{:-3} {:-3} {:-3}'.format(color[0][0], color[0][1], color[0][2]), (0, 18 * position),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            position += 1
        position = 1
        for color in self.color_buf:
            f = cv2.putText(f, '{:-3} {:-3} {:-3}'.format(color[1][0], color[1][1], color[1][2]), (150, 18 * position),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            position += 1
        return f

    @classmethod
    def imcolor(cls, color):
        f = np.zeros((60, 450, 3), dtype=np.uint8)
        if color is not None:
            f = cv2.putText(f, '{:-3} {:-3} {:-3}'.format(color[0], color[1], color[2]), (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        return f
    # color picker #

    # platform status #
    @classmethod
    def imstatus(cls, gear, speed, steer, brake):
        from serialpacket import SerialPacket
        f = np.zeros((240, 400, 3), dtype=np.uint8)

        gear_str = ''
        if gear == SerialPacket.GEAR_FORWARD:
            gear_str = 'D'
        elif gear == SerialPacket.GEAR_NEUTRAL:
            gear_str = 'N'
        elif gear == SerialPacket.GEAR_BACKWARD:
            gear_str = 'R'

        speed_str = '{:6.2f}'.format(speed) + 'kph'

        steer_direction_str = 'L' if steer > 0 else 'R'
        if steer == 0: steer_direction_str = 'S'
        steer_str = steer_direction_str + '{:5.2f}'.format(abs(steer)) + 'deg'

        brake_str = '{:6.2f}'.format(brake) + 'brake'

        f = cv2.putText(f, gear_str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        f = cv2.putText(f, speed_str, (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        f = cv2.putText(f, steer_str, (0, 170), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        f = cv2.putText(f, brake_str, (0, 230), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        return f
    # platform status #

    # mission status #
    def immission(self, mission_num, control_status):
        f = np.zeros((240, 400, 3), dtype=np.uint8)

        return f
    # mission status #

    # monitor status #
    def immonitor(self):
        f = np.full((100, 200, 3), 50, dtype=np.uint8)
        numwin_str = str(sum(self.windows_is.values()))
        f = cv2.putText(f, numwin_str, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))
        idx = 10
        for window_str in self.windows_str:
            color = (255, 255, 255)
            if self.windows_is[window_str] is False: color = (0, 0, 255)
            cv2.putText(f, window_str, (50, idx), cv2.FONT_HERSHEY_PLAIN, 1, color)
            idx += 10
        return f
    # monitor status #

    @classmethod
    def concatenate(cls, f1, f2, mode='h'):
        if f1 is None: return f2
        if f2 is None: return f1
        if f1.ndim == 2:
            f1 = f1[:, :, np.newaxis]
            temp = np.concatenate((f1, f1), axis=2)
            f1 = np.concatenate((temp, f1), axis=2)
        if f2.ndim == 2:
            f2 = f2[:, :, np.newaxis]
            temp = np.concatenate((f2, f2), axis=2)
            f2 = np.concatenate((temp, f2), axis=2)
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

    @classmethod
    def concatenates(cls, *frames, mode='h'):
        frame = None
        for f in frames:
            frame = cls.concatenate(frame, f, mode=mode)
        return frame

    def initSetMouseCallback(self, winname, onMouse, userdata):
        for w in self.wname_buf:
            if w == winname:
                return
        self.wname_buf.append(winname)
        cv2.setMouseCallback(winname, onMouse, userdata)

    def show(self, wname, *frames, color_picker=False):
        wname_org = wname
        for i in range(len(frames)):
            wname = wname_org + ('-{}'.format(i) if i != 0 else '')
            if wname not in self.windows_str: self.windows_str.append(wname)
            if frames[i] is None: self.windows_is[wname] = False; continue
            self.windows_is[wname] = True
            cv2.imshow(wname, frames[i])
            if color_picker is True:
                self.initSetMouseCallback(wname, self.color_picker, frames[i])


if __name__ == '__main__':
    import time
    import video_stream

    # import communication
    # platform = communication.PlatformSerial('COM3')
    video = video_stream.WebCamVideoStream(0, 100, 200)
    video.start()
    monitor = Monitor()
    while True:
        # platform.recv()
        # platform.send()
        ret, frame = video.read()
        final = cv2.flip(frame, 1)
        final = final[:50, :100]
        result = monitor.concatenate(frame, final, 'v')
        # result = Monitor.concatenate(result, Monitor.imstatus(*platform.status()), 'h')
        monitor.show('1', result, color_picker=True)
        color_frame = monitor.concatenate(monitor.imcolor(monitor.color_rgv), monitor.imcolor(monitor.color_hsv),
                                          mode='v')
        monitor.show('color-picker', color_frame, monitor.imcolorbuf())
        monitor.show('monitor', monitor.immonitor())
        monitor.show('none', None)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    video.release()
    exit()
