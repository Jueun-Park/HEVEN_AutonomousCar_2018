# 통신 프로그램
# 제어에서 받은 정보로 통신 패킷 만들어서 플랫폼으로 보내기
# 플랫폼에서 통신 패킷 받아와서 제어로 보내기
# 패킷 세부 형식(string)은 책자 참조
# input: (from car_control)
# output: (to car_control)

import serial
import time
import math

port_PF = '/dev/ttyUSB0' # 플랫폼 포트

dpr = 54.02 * math.pi   # Distance per Rotation [cm]
ppr = 100.              # Pulse per Rotation
dpp = dpr/ppr           # Distance per Pulse

def read_PF():  # 플랫폼으로부터 컨트롤러(데스크탑)으로 데이터를 받음
    global aData, rData, STEER, SPEED, ENC1, SPEED_E
    ser_PF = serial.Serial(port=port_PF, baudrate=115200)  # open serial port
    rData = bytearray(ser_PF.readline())  # sting 으로 읽어옴
    # 패킷 설명 (책자) 참조
    try:
        ETX1 = rData[17]
        AorM = rData[3]
        ESTOP = rData[4]
        GEAR = rData[5]
        SPEED = rData[6] + rData[7] * 256
        STEER = rData[8] + rData[9] * 256
        if STEER >= 32768:
            STEER = 65536 - STEER
        else:
            STEER = -STEER
        BRAKE = rData[10]

        t_enc = time.time()
        ENC = rData[11] + rData[12] * 256 + rData[13] * 65536 + rData[14] * 16777216
        if ENC >= 2147483648:
            ENC = ENC - 4294967296
        ALIVE = rData[15]
        try:
            SPEED_E = (ENC - ENC1[0]) * dpp / (t_enc - ENC1[1]) * 0.036
        except Exception as e:
            print(e)
            pass
        ENC1 = [ENC, t_enc]
        # print SPEED, STEER
        print('STEER = ', STEER, ' SPEED_ENC = ', SPEED_E)

        # print STEER
    except:
        pass
    ser_PF.close()
    # threading.Timer(0.01, read_PF).start(


def write_PF():  # 컨트롤러에서 처리한 데이터를 플랫폼으로 전송
    global wData, wSTEER, wSPEED, aData, wBRAKE
    ser_PF = serial.Serial(port=port_PF, baudrate=115200)
    try:
        if wSTEER > 1970:
            wSTEER = 1970
        if wSTEER < -1970:
            wSTEER = -1970
        wSTEER = int(wSTEER * 1.015)

        if wSTEER < 0:
            wSTEER = wSTEER + 65536
        aData[6] = 0
        print("wSTEER = ", wSTEER, "/ wSPEED = ", wSPEED, "/ BRAKE = ", wBRAKE)
        aData[7] = wSPEED
        # 16진법 두 칸 전송
        aData[8] = wSTEER / 256
        aData[9] = wSTEER % 256

        wData[3] = 1
        wData[4] = 0  # E stop
        wData[5] = 0
        # 임시 데이터를 최종 데이터에 입력
        wData[6] = aData[6]
        wData[7] = aData[7]
        wData[8] = aData[8]
        wData[9] = aData[9]
        wData[10] = wBRAKE
        # 받은 데이터와 똑같이 전송, 플랫폼 자체적으로 데이터 수신 간격을 알기 위함
        wData[11] = rData[15]
        wData[12] = rData[16]
        wData[13] = rData[17]
        ser_PF.write(str(wData))

        # print str(wData)
        # f.write(str(wData))
        # print wData[8], wData[9]
    except Exception as e:
        print(e)
        print('auto error')
        ser_PF.write(str(wData))
    ser_PF.close()