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
        print(self.write_packet.write_bytes())

    def recv(self):
        self._read(self.read_packet)
        print('read:', self.read_packet.get_attr())

import time
if __name__ == '__main__':
    port = 'COM7'
    platform = PlatformSerial(port)
    while True:
        t1=time.time()
        platform.recv()
        platform.write_packet = SerialPacket()
        platform.send()
        t2=time.time()
        print(t2-t1)