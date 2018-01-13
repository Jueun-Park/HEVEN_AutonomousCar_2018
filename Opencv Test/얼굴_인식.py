import cv2

face_cascade = cv2.CascadeClassifier('C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
while True:
    ret, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 영상을 흑백으로

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Found {0} 횡단보도_표지판_위치_리스트!".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 입력시 창 종료
        break

cam.release()
cv2.destroyAllWindows()
