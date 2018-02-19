from multiprocessing import Pool

def findCenterofMassX(y) :
    sum_of_x_mass_coordinates = 0
    for x in range(0,len(src[0])) :
        if[y][x] !=0 :
            sum_of_x_mass_coordinates += x

    return sum_of_x_mass_coordinates

def findCenterofMassY(y) :
    sum_of_y_mass_coordinates  = 0
    for x in range(0,len(src[0])) :
        if[y][x] !=0 :
            sum_of_y_mass_coordinates +=y

    return sum_of_y_mass_coordinates

def findCenterofMassP(y) :
    num_of_mass_points = 0
    for x in range(0, len(src[0])) :
        if [y][x] != 0:
            num_of_mass_points += 1

    return num_of_mass_points

"""
    for y in range(0,len(src)) :
        for x in range(0,len(src[0])) :
            if[y][x] !=0 :
                sum_of_x_mass_coordinates += x
                sum_of_y_mass_coordinates += y
                num_of_mass_points+=1

    center_of_mass_x = int(sum_of_x_mass_coordinates / num_of_mass_points)
    center_of_mass_y = int(sum_of_y_mass_coordinates / num_of_mass_points)

    return (center_of_mass_x,center_of_mass_y)
"""


def function_pool(src) :
    p=Pool(24)
    len_Y=len(src)
    len_X=len(src[0])
    num_of_mass_points = sum(p.map(findCenterofMassP, range(0, len_Y)),0.00)
    center_of_mass_x = int(sum(p.map(findCenterofMassX, range(0, len_Y)),0.00)/num_of_mass_points)
    center_of_mass_y = int(sum(p.map(findCenterofMassY, range(0, len_Y)),0.00)/num_of_mass_points)

    return (center_of_mass_x,center_of_mass_y)





    sum_of_x_mass_coordinates = 0
    sum_of_y_mass_coordinates = 0
    num_of_mass_points = 0



if __name__ == "__main__" :
    function_pool()



