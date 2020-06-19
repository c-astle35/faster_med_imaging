import dask
import math

#### delayed interp functions ####
@dask.delayed
def d_tstep(ndimphan):
    return 2.0 / (ndimphan - 1.0)

@dask.delayed
def d_r(x, tstep):
    return (x + 1.0) / tstep + 1.0

@dask.delayed
def d_floor(n):
    return math.floor(n)

@dask.delayed
def d_subtract(a,b):
    return a - b

@dask.delayed
def d_index_multiply(a, array, nx, ny):
    return a * array[nx-1, ny-1]

@dask.delayed
def d_add(a,b):
    return a + b

@dask.delayed
def d_multiply(a,b):
    return a * b
# # # # # # # # # # # # # # # # # # 

def interp(x, y, ndimphan, w):

    tstep = d_tstep(ndimphan)

    rx = d_r(x, tstep)
    ry = d_r(y, tstep)
    nx = d_floor(rx)
    ny = d_floor(ry)

    ax1 = d_subtract(rx, nx)
    ay1 = d_subtract(ry, ny)
    ax0 = d_subtract(1.0, ax1)
    ay0 = d_subtract(1.0, ay1)
    
    im_1 = d_index_multiply(ay0, w, nx, ny)
    im_2 = d_index_multiply(ay1, w, nx, ny+1)
    im_3 = d_index_multiply(ay0, w, nx+1, ny)
    im_4 = d_index_multiply(ay1, w, nx+1, ny+1)
    
    add_1 = d_add(im_1, im_2)
    add_2 = d_add(im_3, im_4)
    
    multiply_1 = d_multiply(ax0, add_1)
    multiply_2 = d_multiply(ax1, add_2)
    
    return d_add(multiply_1, multiply_2)
