import cv2
import time

face_cascade = cv2.CascadeClassifier('/home/heven2018/PycharmProjects/HEVEN_AutonomousCar_2018/venv1/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
check = 0
t1 = time.time()
cam = cv2.VideoCapture(0)
while True:
    t2 = time.time()
    check += 1
    ret, image = cam.read()
    image_umat = cv2.UMat(image)
    gray = cv2.cvtColor(image_umat, cv2.COLOR_BGR2GRAY)  # 영상을 흑백으로

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # print("Found {0} faces!".format(len(faces)))
    if check % 100 == 0:
        print(t2 - t1)
        t1 = time.time()

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 입력시 창 종료
        break

cam.release()
cv2.destroyAllWindows()
