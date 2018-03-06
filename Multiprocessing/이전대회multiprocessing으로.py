######################### Module ############################
from multiprocessing import Process,Queue,Pipe,Pool
import serial
import socket
import threading
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import sys
import time
import copy
import Vision_0513 as VLD
import a_star_88_a as AS
import steering_yet_0515 as ST
# import parking as P
##import goal_selection2 as GS
from datetime import datetime

########################### global ##########################
global list_Lane, lane_Comb, lidar_Comb, points_Path, destination
global lane_Width, matrix_Show, stop_Line
global l_Exist, r_Exist, left_Bottom, right_Bottom
global park_detect, destination2
###########For test##########3
global Mission, img, ch
########################### server ##########################o

HOST = '115.145.177.1'
PORT = 2112
# HOST = '127.0.0.1'
# PORT = 10019
BUFF = 57600
MESG = chr(2) + 'sEN LMDscandata 1' + chr(3)

sock_lidar = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_lidar.connect((HOST, PORT))

sock_lidar.send(MESG)
fig = plt.figure()
# port_IMU = '/dev/ttyUSB2'
port_PF = '/dev/ttyUSB0'
# port_GPS = '/dev/ttyUSB0'
# ser_GPS = serial.Serial(port_GPS, 38400)
# ser_IMU = None
ser_PF = None

############################################################
buf_IMU = []
buf_PF = []
data_parsing = 0
lidar_Matrix = np.ones((80, 80))
Yaw = 0
Lat = 0
Lon = 0
STEER = 0
SPEED = 0
ENC1 = []
SPEED_E = 0
ch = 1

########################## Vision ###########################
u_turn_stop = 0
stop_Line = None
is_in_mission = True

cam = cv2.VideoCapture('/dev/video0')

cam.set(3, 480)
cam.set(4, 270)

if (not cam.isOpened()):
    print("cam open failed")
s, img = cam.read()
'''while True:
    s, img = cam.read()
    cv2.imshow('ere',img)
    print img
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break'''

####################### Sign Borad ##########################
'''crosswalkcascade = cv2.CascadeClassifier('crosswalk.xml')
narrowcascade = cv2.CascadeClassifier('narrow_0514.xml')
movingcascade = cv2.CascadeClassifier('moving.xml')
staticcascade = cv2.CascadeClassifier('static_0514.xml')
scurvecascade = cv2.CascadeClassifier('scurve_0514.xml')
uturncascade = cv2.CascadeClassifier('uturndetect.xml')
parkingcascade = cv2.CascadeClassifier('parkingdetect.xml')
'''
##uturncascade = cv2.CascadeClassifier('uturndetect.xml')
##
####################### Path Planning ########################
matrix_Show = np.zeros((80, 80))
list_Lane = np.zeros((80, 80))
lidar_Comb = np.zeros((80, 80), dtype=float)
lane_Comb = np.zeros((80, 80))
points_Path = [[39, 79], [39, 78], [39, 77], [39, 76], [39, 75]]
lane_Width = [15, 15]
l_Exist = 0
r_Exist = 0
left_Bottom = 0
right_Bottom = 0
c = 1
######################## Platform ###########################
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

#########For test##############
Mission = 1

'''
1 = Moving
2 = Narrow
3 = CrossWalk
4 = S-curve
5 = Uturn
9 = Static
7 = Parking'''
obs_Detect = False
u_Detect = False
threshold = 6
speed_Obs = 50  # Narrow/S-curve
speed_Default = 50  # Moving/CrossWalk/Static
t1_Cross = 0
t2_Cross = 0
t3_Cross = 0
time_Cross = 5
k_check = 0.2
dotted_Line = 0


########################## Funtion ###########################
def uturn_detect():
    global u_turn_stop, is_in_mission, Mission, img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if stop6 == 3 or (not in_Mission):
        pass
    else:
        print("Stop 6 is "), stop6
        uturn = uturncascade.detectMultiScale(gray, 1.08, 5)
        for (x, y, w, h) in uturn:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        rec6 = np.matrix(uturn)

        if np.sum(rec6) >= 1:
            detect_uturn = 6
            stop6 += 1
            Mission = 5
            if stop6 == 3:
                in_Mission = True
            print("Uturn!!!! "), detect_uturn


def openCam():
    global s, img, lane_Comb, stop_Line
    global l_Exist, r_Exist, left_Bottom, right_Bottom, img
    if (not cam.isOpened()):
        print("cam open failed")
    s, img = cam.read()
    # img = cv2.resize(img,(480,270))
    lane_c, stop_Line = VLD.lane_Detection(img)
    lane_Comb = lane_c[0]
    l_Exist, r_Exist, left_Bottom, right_Bottom = lane_c[2]
    print("Lane_Error"), l_Exist, r_Exist
    print("Bottom_Comb"), left_Bottom, right_Bottom


def read_Lidar():
    global data_parsing, lidar_Comb, data

    data = sock_lidar.recv(BUFF)
    ##    f.write(str(data))
    try:
        parse1 = data.split('sSN LMDscandata')[1]
        data_parsing = parse1[111: parse1.find('0') - 14]
        lidar_Comb = np.zeros((80, 80))
        distance = np.zeros(data_parsing.count(' ') + 2)
        # print data_parsing
        try:
            for i in range(89, 451):
                distance[i] = int(dat a_parsing.split(' ')[i], 16)
            for i in range(90, 450):
                theta = (i * 0.5 - 45) / 180 * math.pi
                x = math.cos(theta) * float(distance[i] / 1000.)
                y = math.sin(theta) * float(distance[i] / 1000.)

                try:
                    if y < 8 and y > 0.2 and x < 4 and x > -4:
                        lidar_Comb[int(-y * 80 / 8 + 79), int(x * 80 / 8 + 39)] = 1
                except:
                    pass
        except Exception as e:
            print(e)
            pass
    except:
        pass

    # lidar_Comb[45:53,destination[1]-3:destination[1]+3] = np.zeros((8,6),dtype = float)


def read_GPS():
    global Lat, Lon

    # ser_GPS = serial.Serial('COM4', 38400)

    gpsData = ser_GPS.readline()
    if gpsData.split(",")[0] == '$GPGGA':
        try:
            line1 = gpsData.split(",")
            Lat = float(line1[2])
            Lon = float(line1[4])
            LatH = Lat // 100.0
            LonH = Lon // 100.0
            LatM = (Lat - (LatH * 100)) / 60
            LonM = (Lon - (LonH * 100)) / 60
            Lat = LatH + LatM
            Lon = LonH + LonM
        except:
            pass

    # ser_GPS.close()


def read_IMU():
    global Yaw

    ser_IMU = serial.Serial(port_IMU, 115200)

    try:
        imuData = ser_IMU.readline()
        Yaw = float(imuData.split(',')[2])
        # print Yaw
    except:
        pass
    ser_IMU.close()
    # threading.Timer(0.02, read_IMU).start()


def read_PF():
    global aData, rData, STEER, SPEED, ENC1, SPEED_E
    ser_PF = serial.Serial(port_PF, 115200)
    rData = bytearray(ser_PF.readline())
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
        print('STEER'), STEER
        print('SPEED_ENC'), SPEED_E

        # print STEER
    except:
        pass
    ser_PF.close()
    # threading.Timer(0.01, read_PF).start(


def write_PF():
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
        print("wSTEER = "), wSTEER
        print("wSPEED = "), wSPEED
        aData[7] = wSPEED
        aData[8] = wSTEER / 256
        aData[9] = wSTEER % 256

        wData[3] = 1
        wData[4] = 0  # E stop
        wData[5] = 0
        wData[6] = aData[6]
        wData[7] = aData[7]
        wData[8] = aData[8]
        wData[9] = aData[9]
        print("BRAKE! "), wBRAKE
        wData[10] = wBRAKE
        wData[11] = rData[15]
        wData[12] = rData[16]
        wData[13] = rData[17]
        ser_PF.write(str(wData))

        # print str(wData)
        # f.write(str(wData))
        # print wData[8], wData[9]
    except Exception as e:
        print(e)
        print('autoerror')
        ser_PF.write(str(wData))
    ser_PF.close()


def matrix_Comb():
    global lidar_Comb, lane_Comb, list_Lane

    list_Lane = lidar_Comb + lane_Comb


def astar_Comb():
    global points_Path, list_Lane, lane_Width
    global l_Exist, r_Exist, left_Bottom, right_Bottom, wSPEED
    global lidar_Comb  # , obs_Detect, Mission#, speed_Obs, speed_Default
    ##    if Mission != 6:
    if Mission % 2 != 0 or obs_Detect == False:
        print("Default!")
        try:
            points = AS.pathFind(lidar_Comb, l_Exist, r_Exist, left_Bottom, right_Bottom, list_Lane, 39, 79)
            if points:
                points_Path = points

            else:
                pass
        except Exception as e:
            print("[Path Planning Error] ", e)
    else:  ### In Narrow, S-curve mission, when obstacle detected
        print("No Line")
        try:
            points = AS.pathFind(lidar_Comb, False, False, left_Bottom, right_Bottom, list_Lane, 39, 79)
            if points:
                points_Path = points

            else:
                pass

        except Exception as e:
            print("[Path Planning Error] ", e)


##    else:
##        pass
def steer_Comb():
    global wSTEER, wSPEED, wBRAKE, STEER, SPEED, points_Path, u_Detect, Mission, obs_Detect
    global t1_Cross, t2_Cross, t3_Cross, time_Cross, wBRAKE, k_check, stop_Line, dotted_Line
    global c, ch
    print("MISSION is = ", Mission)
    if Mission != 5:
        ch = 1
    if Mission == 5:
        if c:
            ch = 2  ##
            c = 0
        print("ch_1", ch)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!!!!!")
        dotted_Line = VLD.dotted_Detection()[0]
        print('dotted', dotted_Line)
        if ((dotted_Line < 160) and (dotted_Line > 100)):
            ch = 1
            print("Dotted Uturn!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Before Steer Mission = ", Mission)
    print("ch_2", ch)
    ch, wSTEER, wSPEED, Mission = ST.steering(Mission, ch, not obs_Detect, dotted_Line, points_Path,
                                              STEER)  # , speed_Default, speed_Obs)## Mission * not obs : 0 = Obs Detect, 5 = Uturn, 7 = Parking, Default = Normal
    print("ch_3", ch)
    print("After Steer Mission = ", Mission)
    ##    wSTEER = -wSTEER
    if Mission == 3:
        if stop_Line:
            t1_Cross = time.time()
            if t2_Cross == 0:
                t3_Cross = t1_Cross
                t2_Cross = 1
                wSPEED = 5
        if time.time() - t3_Cross > k_check:
            wSPEED = 1
            wSTEER = 0
            wBRAKE = 60
        if time.time() - t3_Cross - k_check > time_Cross:
            wSPEED = speed_Default
            wBRAKE = 1
            t1_Cross = 0
            t2_Cross = 0
            t3_Cross = 0
    elif Mission == 1:
        if obs_Detect:
            t1_Cross = time.time()
            if t2_Cross == 0:
                t3_Cross = t1_Cross
                t2_Cross = 1
                wSPEED = 20
        if time.time() - t3_Cross > k_check:
            wSPEED = 1
            wSTEER = 0
            wBRAKE = 45
        if not obs_Detect:  # time.time() - t3_Cross - k_check > time_Cross:
            wSPEED = speed_Default
            wBRAKE = 1
            t1_Cross = 0
            t2_Cross = 0
            t3_Cross = 0


def show_Path():
    global points_Path, matrix_Show, list_Lane
    matrix_Show = np.array(list_Lane)
    for i in range(0, len(points_Path)):
        matrix_Show[points_Path[i][1], points_Path[i][0]] = 3
    # matrix_Show[destination[1],destination[0]] = 3


def front_Detect():
    global lidar_Comb, obs_Detect, threshold, Mission
    if Mission % 2 != 0 and Mission != 1:
        obs_Detect = False
    else:
        # print "obs ",np.sum(lidar_Comb[60:80,20:59])
        if np.sum(lidar_Comb[45:80, 26:52]) > threshold:
            obs_Detect = True
        else:
            obs_Detect = False


def main():
    Cam_process=Process(target=openCam())
    Lidar_process = Process(target=read_Lidar())
    Matrix_process = Process(target=matrix_Comb())
    AS_process = Process(target=astar_Comb())
    # IMU_process = Process(target = read_IMU())
    PFread_process = Process(target=read_PF())
    # GPS_process = Process(target = read_GPS())
    STEER_process = Process(target=steer_Comb())
    PFwrite_process= Process(target=write_PF())
    Show_process= Process(target=show_Path())
    Obs_process= Process(target=front_Detect())
    # U_process = Process(target = uturn_detect())


    Cam_process.start()
    Lidar_process.start()
    Matrix_process.start()
    AS_process.start()
    #IMU_process.start()
    PFread_process.start()
    #GPS_process.start()
    STEER_process.start()
    PFwrite_process.start()
    Show_process.start()
    Obs_process.start()
    #U_process.start()

    #직접 차로 실험해봐야 오류가 날지 안날지 알 수 있음, 스레드와 달리 메모리오류가 날수도 있음
    #혹은 순차적으로 처리되지 않고 건너뛰는경우도 나올수도 있음
    #개별함수도 멀티프로세싱이 가능하다면 해볼예정
    Cam_process.join()
    Lidar_process.join()
    Matrix_process.join()
    AS_process.join()
    # IMU_process.join()
    PFread_process.join()
    # GPS_process.join()
    STEER_process.join()
    PFwrite_process.join()
    Show_process.join()
    Obs_process.join()
    # U_process.join()



'''
    Cam_thread = threading.Thread(target = openCam())
    Cam_thread = threading.Thread(target = openCam())
    Lidar_thread = threading.Thread(target = read_Lidar())
    Matrix_thread = threading.Thread(target = matrix_Comb())
    AS_thread = threading.Thread(target = astar_Comb())
    #IMU_thread = threading.Thread(target = read_IMU())
    PFread_thread = threading.Thread(target = read_PF())
    #GPS_thread = threading.Thread(target = read_GPS())
    STEER_thread = threading.Thread(target = steer_Comb())
    PFwrite_thread = threading.Thread(target = write_PF())
    Show_thread = threading.Thread(target = show_Path())
    Obs_thread = threading.Thread(target = front_Detect())
    #U_thread = threading.Thread(target = uturn_detect())

    Cam_thread.start()
    Lidar_thread.start()
    Matrix_thread.start()
    AS_thread.start()
    #IMU_thread.start()
    PFread_thread.start()
    #GPS_thread.start()
    STEER_thread.start()
    PFwrite_thread.start()
    Show_thread.start()
    Obs_thread.start()
    #U_thread.start()
    #print wSPEED
    '''



if __name__ == "__main__":
    t1 = 0
    t2 = 0
    while True:
        t2 = time.time()
        print(t1 - t2)
        t1 = time.time()
        # openCam()
        main()
        # astar_Comb()
        dim = (500, 500)
        # map_plot = cv2.resize(lane_Comb, dim, interpolation = cv2.INTER_AREA)
        # lidar_plot = cv2.resize(lidar_Comb, dim, interpolation = cv2.INTER_AREA)
        lane_plot = cv2.resize(matrix_Show, dim, interpolation=cv2.INTER_AREA)
        # cv2.imshow('map', map_plot)
        # cv2.imshow('lidar', lidar_plot)
        cv2.imshow('Path', lane_plot)
        # cv2.imshow('cam', img)
        if cv2.waitKey(1) == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)
