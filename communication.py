from serialpacket import SerialPacket
import serial


class PlatformSerial:
    def __init__(self, platform_port):
        self.port = platform_port
        try:
            self.ser = serial.Serial(self.port, 115200)
        except Exception as e:
            print('[PlatformSerial| INIT ERROR: ', e, ']')
        self.read_packet = SerialPacket()
        self.write_packet = SerialPacket()

    def send(self):
        self.write_packet.alive = self.read_packet.alive
        try:
            self.ser.write(self.write_packet.write_bytes())
        except Exception as e:
            print('[PlatformSerial| WRITE ERROR', e, ']')

    def recv(self):
        try:
            b = self.ser.read(18)
        except Exception as e:
            print('[PlatformSerial| READ ERROR', e, ']')
            return
        self.read_packet.read_bytes(b)

    def read(self):
        return self.read_packet.speed, self.read_packet.enc

    def write(self, gear, speed, steer, brake):
        self.write_packet.gear = gear
        self.write_packet.speed = speed
        self.write_packet.steer = steer
        self.write_packet.brake = brake

    def status(self):
        gear = self.read_packet.gear
        speed = self.read_packet.speed / 10
        steer = self.read_packet.steer / 71
        brake = (self.read_packet.brake - SerialPacket.BRAKE_NOBRAKE) / \
                (SerialPacket.BRAKE_MAXBRAKE - SerialPacket.BRAKE_NOBRAKE)
        print('[READ]')
        print(self.read_packet.get_attr(mode='a'))
        print(str(speed) + 'kph', str(round(steer, 4)) + 'deg', str(round(brake, 4)) + 'brake')
        print()
        return gear, speed, steer, brake

import time
def t_move():
    platform.write(SerialPacket.GEAR_FORWARD, 40, SerialPacket.STEER_STRAIGHT, SerialPacket.BRAKE_NOBRAKE)

def t_back():
    platform.write(SerialPacket.GEAR_BACKWARD, 60, SerialPacket.STEER_STRAIGHT, SerialPacket.BRAKE_NOBRAKE)

def t_stop():
    platform.write(SerialPacket.GEAR_NEUTRAL, 0, SerialPacket.STEER_STRAIGHT, 60)

def t_neutral():
    platform.write(SerialPacket.GEAR_NEUTRAL, 0, SerialPacket.STEER_STRAIGHT, SerialPacket.BRAKE_NOBRAKE)

def t_left():
    platform.write(SerialPacket.GEAR_NEUTRAL, 0, SerialPacket.STEER_MAXLEFT, SerialPacket.BRAKE_NOBRAKE)

def t_right():
    platform.write(SerialPacket.GEAR_NEUTRAL, 0, SerialPacket.STEER_MAXRIGHT, SerialPacket.BRAKE_NOBRAKE)


if __name__ == '__main__':
    platform = PlatformSerial('COM3')
    while True:
        platform.recv()
        platform.status()
        t_stop()
        platform.send()
        if platform.read_packet.aorm == SerialPacket.AORM_AUTO:
            t = time.time()
            while time.time() - t < 2:
                platform.recv()
                platform.status()
                t_move()
                platform.send()
            t = time.time()
            while time.time() - t < 2:
                platform.recv()
                platform.status()
                t_stop()
                platform.send()
