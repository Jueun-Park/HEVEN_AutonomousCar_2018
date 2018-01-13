import cv2

'''
안 되는데요?
'''

횡단보도_표지판_cascade = cv2.CascadeClassifier('../sign_xml_files/crosswalk.xml')

cam = cv2.VideoCapture(0)
while True:
    ret, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 영상을 흑백으로

    # <about detectMultiScale method>
    # scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
    # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    # minSize – Minimum possible object size. Objects smaller than that are ignored.
    # return: list of rectangles
    횡단보도_표지판_위치_리스트 = 횡단보도_표지판_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print("Found {0} 횡단보도 표지판!".format(len(횡단보도_표지판_위치_리스트)))

    for (x, y, w, h) in 횡단보도_표지판_위치_리스트:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # q 입력시 창 종료
        break

cam.release()
cv2.destroyAllWindows()
