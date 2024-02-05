cimport numpy as cnp
cimport cython
import numpy as np
from libc.math cimport  cos, sin


ctypedef cnp.complex128_t DTYPE_t

@cython.cdivision(True)
@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef DTYPE_t[:,:,::1] get_input_complex_amplitudes(
                                  float time,
                                  float wavelength,
                                  double[::1] x_source_sky_coordinates,
                                  double[::1] y_source_sky_coordinates,
                                  double[::1] x_observatory_coordinates,
                                  double[::1] y_observatory_coordinates,
                                  float aperture_radius,
                                  int number_of_inputs,
                                  int grid_size,
                                  DTYPE_t[:,:,::1] out):
    cdef unsigned int ix, iy, index_input
    cdef double arg, cos_val, sin_val
    cdef double factor = 2 * 3.1415926536 / wavelength
    cdef DTYPE_t[:,:,::1] ou = out

    for index_input in range(number_of_inputs):
        for ix in range(grid_size):
            for iy in range(grid_size):
                arg = factor * (
                        x_observatory_coordinates[index_input] * x_source_sky_coordinates[ix] +
                        y_observatory_coordinates[index_input] * y_source_sky_coordinates[iy])

                cos_val = cos(arg)
                sin_val = sin(arg)

                ou[index_input,ix, iy] = aperture_radius * (cos_val + 1j * sin_val)
    return ou