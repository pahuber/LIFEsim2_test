cimport numpy as cnp
import cython
import numpy as np


ctypedef cnp.complex64_t cpl_t

# TODO: decide whether this is necessary
cpl = np.complex64
@cython.cdivision(True)
@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef get_input_complex_amplitude_vector(double wavelength,
                                    cnp.ndarray[double, ndim=1] x_observatory_coordinates,
                                    cnp.ndarray[double, ndim=1] y_observatory_coordinates,
                                    cnp.ndarray[double, ndim=2] source_sky_coordinate_map_x,
                                    cnp.ndarray[double, ndim=2] source_sky_coordinate_map_y,
                                    double aperture_radius,
                                    int number_of_inputs,
                                    int grid_size):
    """Return the unperturbed input complex amplitude vector, consisting of a flat wavefront per collector.

    :param time: The time for which the intensity response is calculated
    :param wavelength: The wavelength for which the intensity response is calculated
    :param source_sky_coordinate_maps: The sky coordinates of the source for which the intensity response is calculated
    :return: The input complex amplitude vector
    """
    # cdef cnp.ndarray[cpl_t, ndim=3] input_complex_amplitude_vector = np.zeros((number_of_inputs, grid_size, grid_size),dtype=cpl)
    cdef cpl_t [3] input_complex_amplitude_vector
    cdef int index_input
    cdef double pi = np.pi
    cdef double complex multiplier = 2j * pi / wavelength
    # cdef np.ndarray[DTYPE_float_t, ndim =1, mode='c' ] d = np.ones(n, dtype=DTYPE_float)
    # double *d = <double*>malloc(n*sizeof(double))

    for index_input in range(number_of_inputs):
        input_complex_amplitude_vector[index_input] = aperture_radius * np.exp(multiplier * (
                    x_observatory_coordinates[index_input] * source_sky_coordinate_map_x +
                    y_observatory_coordinates[index_input] * source_sky_coordinate_map_y))
    return input_complex_amplitude_vector