from numba import prange, njit #parallelization
from PIL import Image #image writing
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
    gray = log10(i)/log10(MAX_ITERATIONS) * 255
    return (gray, gray, gray)

@njit(parallel=True)
def mandelbrot():
    img_array = np.zeros((IMG_WIDTH, IMG_HEIGHT, 3), dtype=np.uint8)

    #loop through each pixel (done in parallel using prange)
    for x in prange(IMG_WIDTH): 
        for y in range(IMG_HEIGHT):
            #define c and z here
            c = complex(
                remap(x, [0,IMG_WIDTH-1], [x_start,x_stop]), 
                remap(y, [0,IMG_HEIGHT-1], [y_start,y_stop]) 
            )
            z = 0

            for i in range(MAX_ITERATIONS):
                if abs(z) > 2:
                    img_array[y, x] = colorize(i)
                    break
                z = z**2 + c
    
    return img_array


img = Image.fromarray(mandelbrot(), 'RGB')

img.save("mandelbrot.png")