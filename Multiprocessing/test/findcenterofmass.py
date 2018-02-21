from multiprocessing import Pool

def findCenterofMass(src)

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
    function_pool()




