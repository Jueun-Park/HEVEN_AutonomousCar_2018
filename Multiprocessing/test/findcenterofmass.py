from multiprocessing import Pool

def findCenterofMass(black_image)



    for y in range(0,len(src)) :
        for x in range(0,len(src[0])) :
            if[y][x] !=0 :
                sum_of_x_mass_coordinates += x
                sum_of_y_mass_coordinates += y
                num_of_mass_points+=1

    center_of_mass_x = int(sum_of_x_mass_coordinates / num_of_mass_points)
    center_of_mass_y = int(sum_of_y_mass_coordinates / num_of_mass_points)

    return (center_of_mass_x,center_of_mass_y)

def function_pool :
    p=Pool(10)

    sum_of_x_mass_coordinates = 0
    sum_of_y_mass_coordinates = 0
    num_of_mass_points = 0



if __name__ == "__main__" :
    import numpy as np
    import cv2

    black_image = np.zeros((512, 512, 3), np.uint8)

    cv2.rectangle(black_image, (0, 0), (511, 511), (255, 0, 0), 3)
    cv2.circle(black_image, (256, 256), 256, (0, 0, 255), 1)
    cv2.ellipse(black_image, (256, 256), (256, 100), 0, 0, 360, (0, 255, 0), 1)

    cv2.imshow("imag", black_image)

    cv2.waitKey(0)
    function_pool()




