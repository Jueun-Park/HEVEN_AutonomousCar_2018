from serialpacket import SerialPacket
import serial

class PlatformSerial:
    def __init__(self, platform_port):
        self.port = platform_port
        try:
            self.ser = serial.Serial(self.port, 115200)
        except Exception as e:
            print('serial error')
            print(e)
        self.read_packet = SerialPacket()
        self.write_packet = SerialPacket()
        #self.ENC_with_time = (0, 0)

    def _read(self, packet=SerialPacket(), ser=serial.Serial()):
        packet.read_bytes(ser.readline())

        import math
        import time
        #DISTANCE_PER_ROTATION = 54.02 * math.pi
        #PULSE_PER_ROTATION = 100
        #DISTANCE_PER_PULSE = DISTANCE_PER_ROTATION / PULSE_PER_ROTATION

        #time_encoder = time.time()
        #speed_from_encoder = (self.reading_data.enc - self.ENC_with_time[0]) * DISTANCE_PER_PULSE / (
        #    time_encoder - self.ENC_with_time[1]) * 0.036

        #print('STEER = ', self.reading_data.steer, 'SPEED_ENC = ', speed_from_encoder)

        #self.ENC_with_time = (self.reading_data.enc, time_encoder)

    def _write(self, packet=SerialPacket(), ser=serial.Serial()):
        try:
            ser.write(packet.write_bytes())
        except Exception as e:
            print(e)
            print(' auto error')
            ser.write(bytearray(packet.write_bytes()))

    def send(self):
        self.write_packet.alive = self.read_packet.alive
        self._write(self.write_packet, self.ser)
        print('write:', self.write_packet.get_attr())

    def recv(self):
        self._read(self.read_packet, self.ser)
        print('read:', self.read_packet.get_attr())

import time
if __name__ == '__main__':
    port = 'COM7'
    platform = PlatformSerial(port)
    while True:
        platform.recv()
        platform.send()
        print()