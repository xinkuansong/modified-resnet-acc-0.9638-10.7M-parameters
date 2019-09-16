"""
Utilization functions
"""
from numpy import pi
import numpy as np


def lr_stepwise(epoch):
    lr = 1e-1

    if epoch > 340:
        lr *= 0.5e-2
    elif epoch > 240:
        lr *= 1e-2
    elif epoch > 150:
        lr *= 1e-1

    print("Learning rate: {:0.6f}".format(lr))
    return lr


def lr_cosine_schedule_vary(epoch):
    lr_max = 0.01
    lr_min = 0.001
    lr_degrade = 0.1
    restart_each_nums_init = 80
    flag = True
    i = 1
    while flag:
        if not epoch > restart_each_nums_init * (2 ** i - 1):
            restart_each_nums = restart_each_nums_init * 2 ** (i - 1)
            restart_sum = restart_each_nums_init * (2 ** (i - 1) - 1)
            lr_max *= lr_degrade ** (i - 1)
            lr_min *= lr_degrade ** (i - 1)
            flag = False
        i += 1

    epoch -= restart_sum

    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(epoch / restart_each_nums * pi))

    print("Learning rate: {:0.6f}".format(lr))
    return lr

