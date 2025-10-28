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
