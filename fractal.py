import timeit

import numpy as np
from numba import cuda

CALC_RESULT = 0
CALC_TIME = 1
CALC_TIME_MICROS = 2

#Kernel function for the GPU
@cuda.jit
def calculate(c, result, iters, floor):
    x, y = cuda.grid(2)
    if abs(c[x, y]) > 2:
        result[x, y] = floor
        return
    previous = complex(0, 0)
    for i in range(iters):
        z = previous ** 2 + c[x, y]
        if abs(z) > 2.0:
            result[x, y] = i
            return
        previous = z
    result[x, y] = floor


def get_fractal_data(iters = 20, n_thread_per_block_side = 25, dims = (800, 600), bounds = (-2, 1, -1, 1)):
    """
    Galculates fractal data for given parameters:
    :param iters: maximum iterations for each point for divergence test
    :param n_thread_per_block_side: side size of square
    :param dims: dimensions of fractal picture in pixels (width, height)
    :param bounds: lower and upper bounds of fractal picture in complex plane (x_lower, x_upper, i_lower, i_upper)
    :return: dictionary with CALC_RESULT, CALC_TIME, CALC_TIME_MICROS
    """
    total_points = dims[0] * dims[1] #Total poitns in the mesh
    d = ((bounds[1] - bounds[0]) / dims[0], (bounds[3] - bounds[2]) / dims[1]) #change of coordinates in both directions based on mesh and bounds
    #Initialize starting point array for GPU threads to use
    c = np.zeros(total_points).astype(complex).reshape(dims[0], dims[1])
    for i in range(dims[0]):
        for jj in range(dims[1]):
            c[i, jj] = complex((bounds[0] + i * d[0]), (bounds[3] - jj * d[1]))
    #Copy point and result arrays to GPU device
    device_c = cuda.to_device(c)
    device_result = cuda.device_array_like(np.zeros(total_points).reshape(dims[0], dims[1]))

    #Calculate thread grid clusters and their dimeniosn for calculation
    n_grid = (dims[0]//n_thread_per_block_side, dims[1]//n_thread_per_block_side)
    n_threads = (n_thread_per_block_side, n_thread_per_block_side)

    #Launch kernel, synchronize it with CPU and measure time elapsed
    start_time = timeit.default_timer()
    calculate[n_threads, n_grid](device_c, device_result, iters, 0.0000001)
    cuda.synchronize()
    calc_time = timeit.default_timer() - start_time
    #Copy/Return resulting iteration count array from GPU back to CPU
    result = device_result.copy_to_host()
    return {
        CALC_RESULT: result,
        CALC_TIME: calc_time * 1000,
        CALC_TIME_MICROS: calc_time * 1000 * 1000
    }