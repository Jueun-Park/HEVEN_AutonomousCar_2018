import cv2
import numpy as np


class Monitor:
    @classmethod
    def imstatus(cls, gear, speed, steer, brake):
        from serialpacket import SerialPacket
        f = np.zeros((240, 400, 3), dtype=np.uint8)

        gear_str = ''
        if gear == SerialPacket.GEAR_FORWARD: gear_str = 'D'
        elif gear == SerialPacket.GEAR_NEUTRAL: gear_str = 'N'
        elif gear == SerialPacket.GEAR_BACKWARD: gear_str = 'R'

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

    @classmethod
    def show(cls, *frames):
        for i in range(len(frames)):
            if frames[i] is None: continue
            cv2.imshow(str(i), frames[i])


if __name__ == '__main__':
    import videostream
    import communication
    platform = communication.PlatformSerial('COM3')
    video = videostream.WebcamVideoStream(0, 100, 200)
    video.start()
    while True:
        platform.recv()
        platform.send()
        ret, frame = video.read()
        final = cv2.flip(frame, 1)
        final = final[:50, :100]
        result = Monitor.concatenate(frame, final, 'v')
        result = Monitor.concatenate(result, Monitor.imstatus(*platform.status()), 'h')
        Monitor.show(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    video.release()
