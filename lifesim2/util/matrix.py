import numpy as np
from numpy import cos, sin, pi


def get_2d_rotation_matrix(time, rotation_period):
    argument = 2*pi/rotation_period*time
    return np.array([[cos(argument), -sin(argument)],
                     [sin(argument), cos(argument)]])