import numpy as np
import struct


class SerialPacket(object):
    AORM_MANUAL = 0x00; AORM_AUTO = 0x01
    ESTOP_OFF = 0x00; ESTOP_ON = 0x01
    GEAR_FORWARD = 0x00; GEAR_NEUTRAL = 0x01; GEAR_BACKWARD = 0x02
    BRAKE_NOBRAKE = 1; BRAKE_FULLBRAKE = 33

    def __init__(self, data=None, start_bytes=[0x53, 0x54, 0x58],
                 aorm=AORM_MANUAL, estop=ESTOP_OFF, gear=GEAR_FORWARD,
                 speed=0, steer=0, brake=BRAKE_NOBRAKE,
                 enc=0, alive=0,
                 end_bytes=[0x0D, 0x0A]):
        if data is not None: self.read_bytes(data); return
        self.start_bytes = start_bytes
        self.aorm = aorm
        self.estop = estop
        self.gear = gear
        self.speed = speed
        self.steer = steer
        self.brake = brake
        self.enc = enc
        self.alive = alive
        self.end_bytes = end_bytes

    def __setattr__(self, attr, v):
        if attr == 'start_bytes': super().__setattr__(attr, np.array(v, np.uint8)); return
        if attr == 'aorm': super().__setattr__(attr, np.uint8(v)); return
        if attr == 'estop': super().__setattr__(attr, np.uint8(v)); return
        if attr == 'gear': super().__setattr__(attr, np.uint8(v)); return
        if attr == 'speed': super().__setattr__(attr, np.uint16(v)); return
        if attr == 'steer': super().__setattr__(attr, np.int16(v)); return
        if attr == 'brake': super().__setattr__(attr, np.uint8(v)); return
        if attr == 'enc': super().__setattr__(attr, np.int32(v)); return
        if attr == 'alive': super().__setattr__(attr, np.uint8(v)); return
        if attr == 'end_bytes': super().__setattr__(attr, np.array(v, np.uint8)); return
        super().__setattr__(attr, v)

    def print_attr(self):
        print(self.start_bytes, self.aorm, self.estop, self.gear, self.speed, self.steer, self.brake, self.enc, self.alive, self.end_bytes)

    def read_bytes(self, b):
        try:
            u = struct.unpack('!3sBBBHhBiB2s', b)
        except:
            print(b)
            u = [b'STX', 0, 0, 0, 0, 0, 1, 0, 0, b'\r\n']

        self.start_bytes = bytearray(u[0])
        self.aorm = u[1]
        self.estop = u[2]
        self.gear = u[3]
        self.speed = u[4]
        self.steer = u[5]
        self.brake = u[6]
        self.enc = u[7]
        self.alive = u[8]
        self.end_bytes = bytearray(u[9])
        print(u)

    def write_bytes(self):
        b = struct.pack('!3sBBBHhBiB2s', bytes(self.start_bytes), self.aorm, self.estop, self.gear, self.speed, self.steer, self.brake, self.enc, self.alive, bytes(self.end_bytes))
        return b


if __name__ == '__main__':
    a = SerialPacket(bytearray.fromhex("53545800 00000000 00000100 00000000 0D0A"))
    a.read_bytes(bytearray.fromhex("53545800 00000000 00000100 00000000 0D0AFF"))
    print(a.start_bytes, a.end_bytes)
    print(str(a.write_bytes()))