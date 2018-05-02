import numpy as np
import struct


class SerialPacket(object):
    START_BYTES = [0x53, 0x54, 0x58]; END_BYTES = [0x0D, 0x0A]
    AORM_MANUAL = 0x00; AORM_AUTO = 0x01; AORM_DEFAULT = AORM_AUTO
    ESTOP_OFF = 0x00; ESTOP_ON = 0x01; ESTOP_DEFAULT = ESTOP_OFF
    GEAR_FORWARD = 0x00; GEAR_NEUTRAL = 0x01; GEAR_BACKWARD = 0x02; GEAR_DEFAULT = GEAR_FORWARD
    SPEED_MIN = 0
    STEER_MAXLEFT = -2000; STEER_STRAIGHT = 0; STEER_MAXRIGHT = 2000
    BRAKE_NOBRAKE = 1; BRAKE_FULLBRAKE = 33; BRAKE_DEFAULT = BRAKE_NOBRAKE; BRAKE_MAXBRAKE = 200

    def __init__(self, data=None, start_bytes=START_BYTES,
                 aorm=AORM_DEFAULT, estop=ESTOP_DEFAULT, gear=GEAR_DEFAULT,
                 speed=0, steer=0, brake=BRAKE_DEFAULT,
                 enc=0, alive=0,
                 end_bytes=END_BYTES):
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

    def default(self):
        self.start_bytes = SerialPacket.START_BYTES
        self.aorm = SerialPacket.AORM_DEFAULT
        self.estop = SerialPacket.ESTOP_DEFAULT
        self.gear = SerialPacket.GEAR_DEFAULT
        self.speed = 0
        self.steer = 0
        self.brake = SerialPacket.BRAKE_DEFAULT
        self.enc = 0
        self.alive = 0
        self.end_bytes = SerialPacket.END_BYTES

    def get_attr(self, mode=None):
        if mode == None: return self.gear, self.speed, self.steer, self.brake
        if mode == 'a': return self.aorm, self.estop, self.gear, self.speed, self.steer, self.brake, self.enc, self.alive
        if mode == 'ra':  return self.start_bytes, self.aorm, self.estop, self.gear, self.speed, self.steer, self.brake, self.enc, self.alive, self.end_bytes
        return 'wrong mode'

    def read_bytes(self, b):
        try:
            u = struct.unpack('<3sBBBHhBiB2s', b)
        except:
            print('[SerialPacket| READ ERROR:', b)
            print('-Set to default value]')
            self.default()
            return

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

    def write_bytes(self):
        try:
            b = struct.pack('!3sBBBHhBB2s', bytes(self.start_bytes), self.aorm, self.estop, self.gear, self.speed, self.steer, self.brake, self.alive, bytes(self.end_bytes))
        except:
            print('[SerialPacket| WRITE ERROR]')
            print('-Set to default value]')
            self.default()
            b = struct.pack('!3sBBBHhBB2s', bytes(self.start_bytes), self.aorm, self.estop, self.gear, self.speed, self.steer, self.brake, self.alive, bytes(self.end_bytes))
        return b

    def verify(self):
        if (self.start_bytes != SerialPacket.START_BYTES).any(): return False
        if (self.end_bytes != SerialPacket.END_BYTES).any(): return False
        return True

'''
if __name__ == '__main__':
    a = SerialPacket(bytearray.fromhex("53545800 00000000 00000100 00000000 0D0A"))
    a.read_bytes(bytearray.fromhex("53545800 00000000 00000100 00000000 0D0A"))
    a.default()
    print(a.start_bytes, a.end_bytes)
    print(str(a.write_bytes()))
'''