# 통신 프로그램
# 제어에서 받은 정보로 통신 패킷 만들어서 플랫폼으로 보내기
# 플랫폼에서 통신 패킷 받아와서 제어로 보내기
# 패킷 세부 형식(byte array)은 책자 참조
# input: (from car_control)
# output: (to car_control)

import serial
import time
import math
import threading  # for test, main 코드에서는 멀티 프로세싱 사용하는 게 목표야.

from serialpacket import SerialPacket

# CONSTANTS for _read(), related with encoder
DISTANCE_PER_ROTATION = 54.02 * math.pi  # Distance per Rotation [cm]
PULSE_PER_ROTATION = 100.  # Pulse per Rotation
DISTANCE_PER_PULSE = DISTANCE_PER_ROTATION / PULSE_PER_ROTATION  # Distance per Pulse


class PlatformSerial:
    def __init__(self, platform_port):
        self.port = platform_port  # e.g. /dev/ttyUSB0 on GNU/Linux or COM3 on Windows.

        # 포트 오픈, 115200 사용. OS 내에서 시리얼 포트도 맞춰줄 것
        try:
            self.ser = serial.Serial(self.port, 115200)  # Baud rate such as 9600 or 115200 etc.
        except Exception as e:
            print('serial error ')
            print(e)

        self.reading_data = bytearray([0 for i in range(14)])

        # 쓰기 데이터 셋
        self.writing_data = bytearray.fromhex("5354580000000000000001000D0A")
        self.speed_for_write = 0
        self.steer_for_write = 0
        self.brake_for_write = 0
        self.gear_for_write = 0  # 0: 전진, 1: 중립(full_brake), 2: 후진
        self.check = 0
        self.present_time = 0
        self.past_time = 0

    def _read(self):  # read data from platform
        reading_data = bytearray(self.ser.readline())  # byte array 로 읽어옴
        self.reading_data = reading_data
        try:
            # data parsing, 패킷 설명은 책자 참조
            ETX1 = reading_data[17]
            AorM = reading_data[3]
            ESTOP = reading_data[4]
            GEAR = reading_data[5]
            SPEED = reading_data[6] + reading_data[7] * 256
            STEER = reading_data[8] + reading_data[9] * 256

            # STEER 범위 조정
            if STEER >= 32768:  # 65536 / 2 = 32768
                STEER = 65536 - STEER
            else:
                STEER = -STEER

            BRAKE = reading_data[10]
            time_encoder = time.time()

            # ENC0, ENC_with_time, ENC2, ENC3
            ENC = reading_data[11] + reading_data[12] * 256 + reading_data[13] * 65536 + reading_data[14] * 16777216
            if ENC >= 2147483648:
                ENC = ENC - 4294967296

            ALIVE = reading_data[15]  # 플랫폼 통신 주기 체크

            try:
                speed_from_encoder = (ENC - self.ENC_with_time[0]) * DISTANCE_PER_PULSE / (
                    time_encoder - self.ENC_with_time[1]) * 0.036
                print('STEER = ', STEER, ' SPEED_ENC = ', speed_from_encoder)
            except Exception as e:
                print(e)
                pass

            self.ENC_with_time = (ENC, time_encoder)

            self.speed_platform = SPEED

        except:
            pass

    def _write(self, speed_for_write=None, steer_for_write=None, gear_for_write=None):  # write data to platform
        if speed_for_write is not None:
            self.speed_for_write = speed_for_write
        if steer_for_write is not None:
            self.steer_for_write = steer_for_write
        if gear_for_write is not None:
            self.gear_for_write = gear_for_write

        try:
            self.steer_for_write = int(self.steer_for_write * 1.015)

            if self.steer_for_write < 0:
                self.steer_for_write = self.steer_for_write + 65536

            print("steer_for_write = ", self.steer_for_write, "/ speed_for_write = ", self.speed_for_write,
                  "/ BRAKE = ", self.brake_for_write, "/ GEAR =", self.gear_for_write)

            self.writing_data[3] = 1  # AorM
            self.writing_data[4] = 0  # E stop

            # gear 입력
            self.writing_data[5] = self.gear_for_write  # GEAR

            # speed 입력
            self.writing_data[6] = 0
            self.writing_data[7] = self.speed_for_write

            # steer 입력, 16진법 두 칸 전송
            self.writing_data[8] = int(self.steer_for_write / 256)
            self.writing_data[9] = self.steer_for_write % 256

            # brake 입력
            self.writing_data[10] = self.brake_for_write

            # 받은 데이터와 똑같이 전송, 플랫폼 자체적으로 데이터 수신 간격을 알기 위함
            self.writing_data[11] = self.reading_data[15]
            self.writing_data[12] = self.reading_data[16]
            self.writing_data[13] = self.reading_data[17]

            self.ser.write(bytearray(self.writing_data))  # 플랫폼에 시리얼 데이터 패킷 전송

        except Exception as e:
            print(e)
            print(' auto error')
            self.ser.write(bytearray(self.writing_data))
        pass

    def get_data_real_time(self):
        # _read() 를 이용해 플랫폼 데이터를 실시간으로 읽음
        try:
            while True:
                self._read()
        except KeyboardInterrupt:  # ctrl+C 누르면 탈출 - 안 되는데?
            pass
        self.ser.close()

    def test_write_to_platform(self):
        self.speed_for_write = 0
        self.brake_for_write = 0

        if self.check % 3 == 0:
            self.steer_for_write = -1900
        elif self.check % 3 == 1:
            self.steer_for_write = 0
        else:
            self.steer_for_write = 1900

        # 1초마다 steer 값 변경해서 테스트
        if self.present_time - self.past_time > 1:
            self.check += 1
            self.past_time = time.time()
            self.present_time = time.time()
        else:
            self.present_time = time.time()

    def test_communication_main(self):
        read_thread = threading.Thread(target=self._read)
        write_thread = threading.Thread(target=self._write)
        test_write_thread = threading.Thread(target=self.test_write_to_platform)

        read_thread.start()
        write_thread.start()
        test_write_thread.start()


if __name__ == '__main__':
    # port = '/dev/ttyUSB0'
    port = 'COM3'
    # e.g. /dev/ttyUSB0 on GNU/Linux or COM3 on Windows.
    platform = PlatformSerial(port)

    while True:
        t1 = time.time()
        platform.test_communication_main()
        t2 = time.time()
        print('time ', t2 - t1)
