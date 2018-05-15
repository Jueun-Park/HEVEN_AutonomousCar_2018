import cv2
import time

cap = cv2.VideoCapture(2)
cap.set(3, 800)
cap.set(4, 448)

if not cap.read():
    print("none")


if cap.read():
    count_2 = 0
    while True:
        time.sleep(0.3)
        count_2 +=1
        ret, img = cap.read()

        if img is None:
            print("image is none")
        else:
            img5 = cv2.GaussianBlur(img, (5, 5), 0)
            gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 52, 104, apertureSize=3)
            image, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count = 0
            for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                le = max(w,h)+10
                area = cv2.contourArea(cnt)  # Contour Line면적
                hull = cv2.convexHull(cnt)  # Convex hull line
                hull_area = cv2.contourArea(hull)  # Convex hull 면적

                if hull_area > 0:
                    solidity = int (100*(area) / hull_area)
                    if solidity>94 and w>42 and h>42:
                        x_1 = int (x+(w-le)/2)
                        x_2 = int (x+(w+le)/2)
                        y_1 = int (y+(h-le)/2)
                        y_2 = int (y+(h+le)/2)

                        if x_1 >300 and 290 >y_2 and 185> y_1 > 80:
                            cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (255, 0, 0), 4)
                            img_trim = img[y_1: y_2, x_1:x_2]
                            img_trim = cv2.resize(img_trim, (32,32))
                            name = "./images/" + str(solidity) +"_"+str(y_1)+"_"+str(y_2)+"_"+str(count) + "_" + str(count_2) + "_.png"
                            cv2.imwrite(name,img_trim)
                    count += 1

        cv2.imshow('img', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("fail")