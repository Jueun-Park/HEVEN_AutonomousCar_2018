from multiprocessing import Pool
import numpy as np
import cv2

def findCenterofMassX(y) :
    sum_of_x_mass_coordinates = 0
    for x in range(0,len(src[0])) :
        if[y][x] !=0 :
            sum_of_x_mass_coordinates += x

    return sum_of_x_mass_coordinates

def findCenterofMassY(y) :
    sum_of_y_mass_coordinates  = 0
    for x in range(0,len(black_image[0])) :
        if[y][x] !=0 :
            sum_of_y_mass_coordinates +=y

    return sum_of_y_mass_coordinates

def findCenterofMassP(y) :
    num_of_mass_points = 0
    for x in range(0, len(black_image[0])) :
        if [y][x] != 0:
            num_of_mass_points += 1

    return num_of_mass_points


def function_pool(src) :
    p=Pool(24)
    len_Y=len(src)
    len_X=len(src[0])
    num_of_mass_points = sum(p.map(findCenterofMassP, range(0, len_Y),len_X),0.00)
    center_of_mass_x = int(sum(p.map(findCenterofMassX, range(0, len_Y)),0.00)/num_of_mass_points)
    center_of_mass_y = int(sum(p.map(findCenterofMassY, range(0, len_Y)),0.00)/num_of_mass_points)
    print(center_of_mass_x,center_of_mass_y)
    return (center_of_mass_x,center_of_mass_y)


if __name__ == "__main__" :


    src = np.zeros((512, 512, 3), np.uint8)

    cv2.rectangle(src, (0, 0), (511, 511), (255, 0, 0), 3)
    cv2.circle(src, (256, 256), 256, (0, 0, 255), 1)
    cv2.ellipse(src, (256, 256), (256, 100), 0, 0, 360, (0, 255, 0), 1)

    cv2.imshow("imag", src)


    cv2.waitKey(0)
    function_pool(src)


    #src값은 정해져있는 값인가? 아님 계속 변동하는 값인가?
    #쿠다로 돌리는것도 짜보는중



