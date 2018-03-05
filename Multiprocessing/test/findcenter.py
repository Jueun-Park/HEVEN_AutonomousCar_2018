from multiprocessing import Pool,Queue,Process
import numpy as np
import cv2


def findCenterofMassX(src,output) :
    sum_of_x_mass_coordinates = 0
    for x in range(0,512) :
        for y in range(0,512) :
            if src[y][x] != 0 :
                sum_of_x_mass_coordinates += x

    output.put(sum_of_x_mass_coordinates)

def findCenterofMassY(src,output) :
    sum_of_y_mass_coordinates  = 0
    for x in range(0,512) :
        for y in range(0, 512):
            if src[y][x] != 0 :
                sum_of_y_mass_coordinates +=y

    output.put(sum_of_y_mass_coordinates)


def findCenterofMassP(src,output) :
    num_of_mass_points = 0
    for x in range(0, 512) :
        for y in range(0, 512):
            if src[y][x] != 0:
                num_of_mass_points += 1

    output.put(num_of_mass_points)

"""
def function_pool(src) :
    p=Pool(24)
    len_Y=len(src)
    len_X=len(src[0])
    num_of_mass_points = sum(p.map(findCenterofMassP, range(0, len_Y)))
    center_of_mass_x = int(sum(p.map(findCenterofMassX, range(0, len_Y)),0.00)/num_of_mass_points)
    center_of_mass_y = int(sum(p.map(findCenterofMassY, range(0, len_Y)),0.00)/num_of_mass_points)

    return (center_of_mass_x,center_of_mass_y)
    
"""


if __name__ == "__main__" :
    black_image = np.zeros((512, 512, 3), np.uint8)

    cv2.rectangle(black_image, (0, 0), (511, 511), (255, 0, 0), 3)
    cv2.circle(black_image, (256, 256), 256, (0, 0, 255), 1)
    cv2.ellipse(black_image, (256, 256), (256, 100), 0, 0, 360, (0, 255, 0), 1)

    # cv2.imshow( "image", black_image )
    process=[]
    output1 = Queue()
    output2 = Queue()
    output3 = Queue()

    process.append(Process(target=findCenterofMassX, args=(black_image, output1)))#512x512 이미지를 사용했음black_image
    process.append(Process(target=findCenterofMassY, args=(black_image, output2)))
    process.append(Process(target=findCenterofMassP, args=(black_image, output3)))

    for p in process: #프로세스 시작
        p.start()

    sum_of_x_mass_coordinates=output1.get()
    sum_of_y_mass_coordinates=output2.get()
    num_of_mass_points=output3.get()

    centerofX=(sum_of_x_mass_coordinates/num_of_mass_points)
    centerofY=(sum_of_y_mass_coordinates/num_of_mass_points)

    output1.close()
    output2.close()
    output3.close()

    for p in process:
         p.join()

    print(centerofX)
    print(centerofY)

    #src값은 정해져있는 값인가? 아님 계속 변동하는 값인가?
    #쿠다로 돌려야할듯 ㅠ

