# 표지판이 있는 위치를 자른 이미지 반환
# 이아영, 김윤진
# input: 캠 이미지
# output: 표지판 후보 이미지

import cv2

def shape_detect(img):
    sign = []

    if img is None:
        print("image is none")
    else:
        img5 = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 52, 104, apertureSize=3)
        image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            le = max(w, h) + 10

            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            if hull_area >0:
                solidity = int (100*area / hull_area)
                if  solidity>93 and w > 40 and h > 40:
                    x_1 = int(x + (w - le) / 2)
                    x_2 = int(x + (w + le) / 2)
                    y_1 = int(y + (h - le) / 2)
                    y_2 = int(y + (h + le) / 2)

                    if x_1 > 0 and 350 > y_2 and y_1 > 60:
                        img_trim = img[y_1: y_2, x_1:x_2]
                        cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (0, 0, 255), 4)
                        img_trim_resize = cv2.resize(img_trim, (32, 32))
                        sign.append(img_trim_resize)
    return sign


def main():
    Shape_detection = shape_detect(img)
    print(Shape_detection)


if __name__ == "__main__":
    # open cam
    cam = cv2.VideoCapture(2)
    cam.set(3, 800)
    cam.set(4, 448)

    if not cam.isOpened():
        print("cam open failed")
    while True:
        s, img = cam.read()
        main()
        cv2.imshow('cam', img)

        if cv2.waitKey(30) & 0xff == 27:
            break
    cam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)