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

    def _read(self, packet=SerialPacket()):
        try:
            b = self.ser.readline()
        except Exception as e:
            print('[PlatformSerial| READ ERROR', e, ']')
            return
        packet.read_bytes(b)

    def _write(self, packet=SerialPacket()):
        try:
            self.ser.write(packet.write_bytes())
        except Exception as e:
            print('[PlatformSerial| WRITE ERROR', e, ']')

    def send(self):
        self.write_packet.alive = self.read_packet.alive
        self._write(self.write_packet)

    def recv(self):
        self._read(self.read_packet)

    def set_automode(self):
        self.write_packet.aorm = SerialPacket.AORM_AUTO

    def set_manualmode(self):
        self.write_packet.aorm = SerialPacket.AORM_MANUAL

    def write(self, gear, speed, steer, brake):
        self.write_packet.gear = gear
        self.write_packet.speed = speed
        self.write_packet.steer = steer
        self.write_packet.brake = brake

    def print_status(self):
        speed = self.read_packet.speed / 10
        steer = self.read_packet.steer / 71
        brake = (self.read_packet.brake - SerialPacket.BRAKE_NOBRAKE) / \
                (SerialPacket.BRAKE_FULLBRAKE - SerialPacket.BRAKE_NOBRAKE)

        print(str(speed) + 'kph', str(steer) + 'deg', str(brake))

import time
if __name__ == '__main__':
    port = 'COM7'
    platform = PlatformSerial(port)
    while True:
        platform.recv()
        platform.print_status()
        platform.write(SerialPacket.GEAR_FORWARD, 1, 2, SerialPacket.BRAKE_NOBRAKE)
        platform.send()
