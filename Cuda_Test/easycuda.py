import pycuda.autoinit
import pycuda.driver as cuda
import numpy

from pycuda.compiler import SourceModule

def getFunction(name, contents):
    mod = SourceModule(contents)
    return mod.get_function(name)