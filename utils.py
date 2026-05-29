import numpy as np
from numba import cuda

#default keys for card info return dict
ID = 0
NAME = 1
CLASS = 2
MP_COUNT = 3
CC_COUNT_ESTIMATE = 4

#
#Decoding dictionary for device class to "numba cores" per multiprocessor in that class
#As per https://stackoverflow.com/questions/63823395/how-can-i-get-the-number-of-cuda-cores-in-my-gpu-using-python-and-numba
#
cc_cores_per_SM_dict = {
    (2,0) : 32,
    (2,1) : 48,
    (3,0) : 192,
    (3,5) : 192,
    (3,7) : 192,
    (5,0) : 128,
    (5,2) : 128,
    (6,0) : 64,
    (6,1) : 128,
    (7,0) : 64,
    (7,5) : 64,
    (8,0) : 64,
    (8,6) : 128,
    (8,9) : 128,
    (9,0) : 128,
    (10,0) : 128,
    (12,0) : 128
    }

class Mesh():
    def __init__(self, bounds, dims):
        """
        Mesh object to describe bounds and resolution of fractal image complex field both in calculation and visualization
        :param bounds: bounds of fractal image complex field: (real_lower, real_upper, imag_lower, imag_upper)
        :param dims: dimensions/resolution of fractal image (width_px, height_px)
        """
        self.bounds = bounds
        self.dims = dims
        self.total_points = dims[0] * dims[1]
        self.d = ((bounds[1] - bounds[0]) / dims[0], (bounds[3] - bounds[2]) / dims[1])
        self.mesh = np.zeros(self.total_points).astype(complex).reshape(dims[0], dims[1])
        for i in range(dims[0]):
            for jj in range(dims[1]):
                self.mesh[i, jj] = complex((bounds[0] + i * self.d[0]), (bounds[3] - jj * self.d[1]))

    def get_updated_mesh_px(self, bounds_px):
        real_part = [self.mesh[bounds_px[0][0]][bounds_px[0][1]].real.item(), self.mesh[bounds_px[1][0]][bounds_px[1][1]].real.item()]
        imag_part = [self.mesh[bounds_px[0][0]][bounds_px[0][1]].imag.item(), self.mesh[bounds_px[1][0]][bounds_px[1][1]].imag.item()]
        real_part.sort()
        imag_part.sort()
        new_bounds = (real_part[0], real_part[1], imag_part[0], imag_part[1])
        return self.get_updated_mesh(new_bounds)

    def get_updated_mesh(self, bounds):
        return Mesh(bounds, self.dims)

def get_current_card():
    """
    :return: current active gpu device object
    """
    return cuda.get_current_device()


def get_card_info(device):
    """
    Gets info for the specified device
    :param device: device object for which info to derive
    :return: Returns device ID, NAME, CLASS, MP_COUNT, CC_COUNT_ESTIMATE
    """
    card_class = (device.COMPUTE_CAPABILITY_MAJOR, device.COMPUTE_CAPABILITY_MINOR)
    multiprocessor_count = device.MULTIPROCESSOR_COUNT
    cuda_core_count_estimate = multiprocessor_count * cc_cores_per_SM_dict[card_class]
    answer = {
        ID:int(device.id),
        NAME: device.name.decode('utf-8'),
        CLASS: card_class,
        MP_COUNT: multiprocessor_count,
        CC_COUNT_ESTIMATE: cuda_core_count_estimate
    }
    return answer

if __name__ == '__main__':
    device = cuda.get_current_device()
    print(get_card_info(device))
