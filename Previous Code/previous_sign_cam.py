import numpy as np
import cv2
import threading

crosswalkcascade = cv2.CascadeClassifier('../sign_xml_files/crosswalk.xml')
narrowcascade = cv2.CascadeClassifier('../sign_xml_files/narrowno.xml')
movingcascade = cv2.CascadeClassifier('../sign_xml_files/moving_0510.xml')
staticcascade = cv2.CascadeClassifier('../sign_xml_files/static_0514.xml')
scurvecascade = cv2.CascadeClassifier('../sign_xml_files/scurve0517.xml')
uturncascade = cv2.CascadeClassifier('../sign_xml_files/uturndetect.xml')
parkingcascade = cv2.CascadeClassifier('../sign_xml_files/parkingdetect.xml')

cam = cv2.VideoCapture('0507_one_lap_normal.mp4')
# cam.set(3,480)
# cam.set(4, 270)
if (not cam.isOpened()):
    print("cam open failed")

ret, img = cam.read()
# x= 240; y= 50;
# w= 240; h= 220;
# img_trim=img[y:y+h,x:x+w]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

stop1 = 0
stop2 = 0
stop3 = 0
stop4 = 0
stop5 = 0
stop6 = 0
stop7 = 0
detect_crsosswalk = 0
detect_narrow = 0
detect_moving = 0
detect_static = 0
detect_scurve = 0
detect_uturn = 0
detect_parking = 0


def cam_open():
    global img, gray, img_trim
    ret, img = cam.read()
    x_trim = 240
    y_trim = 50
    w_trim = 240
    h_trim = 220
    img_trim = img[y_trim: y_trim + h_trim, x_trim: x_trim + w_trim]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crosswalk_detect():
    global stop1
    if stop1 == 3:
        return 0
    # print 't1', time.time()
    crosswalk = crosswalkcascade.detectMultiScale(gray, 1.07, 20)
    # print 't2', time.time()
    for (x, y, w, h) in crosswalk:
        cv2.rectangle(img_trim, (x, y), (x + w, y + h), (255, 0, 0), 2)
    rec1 = np.matrix(crosswalk)

    if np.sum(rec1) >= 1:
        detect_crosswalk = 1
        stop1 += 1
        print("CrossWalk!!! ", detect_crosswalk)


def narrow_detect():
    # print 'a'
    global stop2
    if stop2 == 3:
        return 0
    narrow = narrowcascade.detectMultiScale(gray, 1.06, 20)
    for (x, y, w, h) in narrow:
        cv2.rectangle(img_trim, (x, y), (x + w, y + h), (255, 255, 0), 2)
    rec2 = np.matrix(narrow)

    if np.sum(rec2) >= 1:
        detect_narrow = 2
        stop2 += 1
        print("Narrow!!!!! ", detect_narrow)


def moving_detect():
    # print 'b'
    global stop3
    if stop3 == 3:
        return 0
    moving = movingcascade.detectMultiScale(gray, 1.02, 5)
    for (x, y, w, h) in moving:
        cv2.rectangle(img_trim, (x, y), (x + w, y + h), (255, 150, 0), 2)
    rec3 = np.matrix(moving)

    if np.sum(rec3) >= 1:
        detect_moving = 3
        stop3 += 1
        print("Moving!!!! ", detect_moving)


def static_detect():
    # print 'c'
    global stop4
    if stop4 == 3:
        return 0
    static = staticcascade.detectMultiScale(gray, 1.3, 20)
    for (x, y, w, h) in static:
        cv2.rectangle(img_trim, (x, y), (x + w, y + h), (255, 255, 255), 2)
    rec4 = np.matrix(static)

    if np.sum(rec4) >= 1:
        detect_static = 4
        stop4 += 1
        print("Static!!!! ", detect_static)


def scurve_detect():
    global stop5
    if stop5 == 3:
        return 0
    scurve = scurvecascade.detectMultiScale(gray, 1.03, 20)
    for (x, y, w, h) in scurve:
        cv2.rectangle(img_trim, (x, y), (x + w, y + h), (255, 255, 120), 2)
    rec5 = np.matrix(scurve)

    if np.sum(rec5) >= 1:
        detect_scurve = 5
        stop5 += 1
        print("Scurve!!!! ", detect_scurve)


def uturn_detect():
    global stop6
    if stop6 == 3:
        return 0
    uturn = uturncascade.detectMultiScale(gray, 1.08, 5)
    for (x, y, w, h) in uturn:
        cv2.rectangle(img_trim, (x, y), (x + w, y + h), (255, 255, 0), 2)
    rec6 = np.matrix(uturn)

    if np.sum(rec6) >= 1:
        detect_uturn = 6
        stop6 += 1
        print("Uturn!!!! ", detect_uturn)


def parking_detect():
    global stop7
    if stop7 == 3:
        return 0

    parking = parkingcascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in parking:
        cv2.rectangle(img_trim, (x, y), (x + w, y + h), (255, 0, 0), 2)
    rec7 = np.matrix(parking)
    if np.sum(rec7) >= 1:
        detect_parking = 7
        stop7 += 1
        print("Parking!!!!! ", detect_parking)


def main():
    while True:
        # print 't1', time.time()
        cam_thread = threading.Thread(target=cam_open())
        cam_thread = threading.Thread(target=cam_open())
        cam_thread = threading.Thread(target=cam_open())

        crosswalk_thread = threading.Thread(target=crosswalk_detect())
        narrow_thread = threading.Thread(target=narrow_detect())
        moving_thread = threading.Thread(target=moving_detect())
        static_thread = threading.Thread(target=static_detect())
        scurve_thread = threading.Thread(target=scurve_detect())
        uturn_thread = threading.Thread(target=uturn_detect())
        parking_thread = threading.Thread(target=parking_detect())

        # cam_thread = threading.Thread(target = cam_open())

        cam_thread.start()
        crosswalk_thread.start()
        narrow_thread.start()
        moving_thread.start()
        static_thread.start()
        scurve_thread.start()
        uturn_thread.start()
        parking_thread.start()

        cv2.imshow('img', img_trim)
        # print 't2', time.time()
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
