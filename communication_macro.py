from serialpacket import SerialPacket
from car_control_test import Control
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

    def print_status(self):
        speed = self.read_packet.speed / 10
        steer = self.read_packet.steer / 71
        brake = (self.read_packet.brake - SerialPacket.BRAKE_NOBRAKE) / \
                (SerialPacket.BRAKE_MAXBRAKE - SerialPacket.BRAKE_NOBRAKE)
        print('[READ]')
        print(self.read_packet.get_attr(mode='a'))
        print(str(speed) + 'kph', str(round(steer, 4)) + 'deg', str(round(brake, 4)) + 'brake')
        print()


if __name__ == '__main__':
    port = 'COM7'
    platform = PlatformSerial(port)
    control = Control()
    while True:
        if platform.read_packet.aorm == SerialPacket.AORM_AUTO:
            platform.recv()
            platform.print_status()
            control.read(*platform.read())
            control.mission(1, 0, 0)  # {주차 - 1, 유턴 - 6}
            control.change()
            if control.change_mission == 0:
                platform.write(*control.write())
                print("Doing Macro\n")
            else:
                platform.write(0, 0, 0, 0)
                print("End Macro\n")
            platform.send()
