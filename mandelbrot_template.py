from numba import prange, njit #parallelization
from PIL import Image #image writing
from time import perf_counter #for testing length of code execution
from math import log10
import numpy as np #arrays

#globals
IMG_WIDTH = 800
IMG_HEIGHT = 800
CHANNELS = 3
MAX_ITERATIONS = 1000

#the bounds of where the Mandelbrot set lies mathematically
x_start = -2.0
x_stop = 1.0
y_start = -1.5
y_stop = 1.5

#proportionally maps a value from one range to another, ex: (0.5 from [0 to 1] to [0 to 100]) = 50
#useful for remapping a value in a mathematical range to where it proportionally belongs as a pixel
@njit
def remap(x, domain, range):
    return range[0] + (x - domain[0]) * (range[1] - range[0]) / (domain[1] - domain[0])

@njit
def colorize(i):
    #TODO: return a color value ex: (255, 255, 255) based on i
    pass

@njit(parallel=True)
def mandelbrot():
    img_array = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.uint8)

    #loop through each pixel (done in parallel using prange)
    for x in prange(IMG_WIDTH): 
        for y in range(IMG_HEIGHT):
            #TODO: define c and z based on what pixel we're generating

            for i in range(MAX_ITERATIONS):
                #TODO: if |z| > 2, pass how many iterations it took to colorize, assign the returned color to the pixel, and break
                #else z=z^2+c
                pass
    return img_array

mandelbrot()
start_time = perf_counter()
img = Image.fromarray(mandelbrot(), 'RGB')
end_time = perf_counter()

total_execution_time = end_time - start_time
print(f"Execution time: {total_execution_time:.2f}")

img.save("mandelbrot.png")