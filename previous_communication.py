import serial
import time
import math
import threading

aData = [0] * 14
rData = [0] * 14
wSPEED = 0
wSTEER = 0
wBRAKE = 0
if wSTEER > 1970:
    wSTEER = 1970
if wSTEER < -1970:
    wSTEER = -1970
wSTEER = int(wSTEER * 1.015)

if wSTEER < 0:
    wSTEER = wSTEER + 65536

aData[6] = 0
aData[7] = wSPEED
aData[8] = wSTEER / 256
aData[9] = wSTEER % 256

wGEAR = 1
wBRAKE = 0

wData = bytearray.fromhex("5354580000000000000001000D0A")

dpr = 54.02 * math.pi  # Distance per Rotation [cm]
ppr = 100.  # Pulse per Rotation
dpp = dpr / ppr  # Distance per Pulse

port_PF = 'COM4'  # e.g. /dev/ttyUSB0 on GNU/Linux or COM3 on Windows.

check = 0
past_time = time.time()
present_time = time.time()


def read_PF():  # 플랫폼으로부터 컨트롤러(데스크탑)으로 데이터를 받음
    global aData, rData, STEER, SPEED, ENC1, SPEED_E
    ser_PF = serial.Serial(port_PF, 115200)  # 시리얼로 받는다
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


def write_PF():  # 컨트롤러에서 처리한 데이터를 플랫폼으로 전송
    global wData, wSTEER, wSPEED, aData, wBRAKE
    ser_PF = serial.Serial(port_PF, 115200)
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
        aData[8] = int(wSTEER / 256)  # an integer is required -> int() 추가
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

        ser_PF.write(bytearray(wData))  # byte array 로 입력

    except Exception as e:
        print(e)
        print('auto error')
        ser_PF.write(bytearray(wData))

    ser_PF.close()


def test_writePF():
    global wSTEER, wSPEED, wBRAKE, check, past_time, present_time
    wSPEED = 0
    wBRAKE = 0

    if check % 3 == 0:
        wSTEER = -1000
    elif check % 3 == 1:
        wSTEER = 0
    else:
        wSTEER = 1000

    if present_time - past_time > 1:
        check += 1
        past_time = time.time()
        present_time = time.time()
    else:
        present_time = time.time()


def main():
    PFread_thread = threading.Thread(target=read_PF())  # 플랫폼 Value 가져오기
    writePF_thread = threading.Thread(target=write_PF())  # 플랫폼에 Value 쓰기
    test_writePF_thread = threading.Thread(target=test_writePF())

    PFread_thread.start()
    writePF_thread.start()
    test_writePF_thread.start()


if __name__ == '__main__':
    t2 = 0
    while True:
        t1 = time.time()
        print('delay: ', t1 - t2)
        main()
        t2 = time.time()
