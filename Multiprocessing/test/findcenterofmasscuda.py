import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ int findCenterofMass(src)
{
 for y in range(0,len(src)) :
        for x in range(0,len(src[0])) :
            if[y][x] !=0 :
                sum_of_x_mass_coordinates += x
                sum_of_y_mass_coordinates += y
                num_of_mass_points+=1

    center_of_mass_x = int(sum_of_x_mass_coordinates / num_of_mass_points)
    center_of_mass_y = int(sum_of_y_mass_coordinates / num_of_mass_points)

    return (center_of_mass_x,center_of_mass_y)
}
""")




findcenterofmass = mod.get_function("findcenterofmass")

a = numpy.random.randn(400).astype(numpy.float32)
b = numpy.random.randn(400).astype(numpy.float32)

dest = numpy.zeros_like(a)

multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1), grid=(1,1))

print (dest-a*b)

