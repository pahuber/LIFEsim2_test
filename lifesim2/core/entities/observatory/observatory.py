from typing import Union

import astropy
from astropy import units as u

from lifesim2.core.entities.observatory.array_configurations import ArrayConfigurationEnum
from lifesim2.core.entities.observatory.beam_combination_schemes import BeamCombinationSchemeEnum
from lifesim2.core.entities.photon_sources.star import Star


class Observatory():
    """Class representation of an observatory.
    """

    def __init__(self):
        """Constructor method.
        """
        self.array_configuration = None
        self.beam_combination_scheme = None
        self.instrument_parameters = None

    def _get_optimal_baseline(self,
                              optimized_differential_output: int,
                              optimized_wavelength: astropy.units.Quantity,
                              optimized_angular_distance: astropy.units.Quantity):
        factors = 1,
        match (self.array_configuration.type, self.beam_combination_scheme.type):
            # 3 collector arrays
            case (ArrayConfigurationEnum.EQUILATERAL_TRIANGLE_CIRCULAR_ROTATION.value,
                  BeamCombinationSchemeEnum.KERNEL_3.value):
                factors = 0.67,
            # 4 collector arrays
            case (ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value,
                  BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value):
                factors = 0.6,
            case (ArrayConfigurationEnum.EMMA_X_CIRCULAR_ROTATION.value, BeamCombinationSchemeEnum.KERNEL_4.value):
                factors = 0.31, 1, 0.6
                print(
                    'The optimal baseline for Emma-X with kernel nulling is ill-defined for second differential output.')
            case (ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value, BeamCombinationSchemeEnum.DOUBLE_BRACEWELL.value):
                factors = 1,
                raise Warning('The optimal baseline for Emma-X with double stretching is not yet implemented.')
            case (ArrayConfigurationEnum.EMMA_X_DOUBLE_STRETCH.value, BeamCombinationSchemeEnum.KERNEL_4.value):
                factors = 1, 1, 1
                raise Warning('The optimal baseline for Emma-X with double stretching is not yet implemented.')
            # 5 collector arrays
            case (ArrayConfigurationEnum.REGULAR_PENTAGON_CIRCULAR_ROTATION.value,
                  BeamCombinationSchemeEnum.KERNEL_5.value):
                factors = 1.04, 0.67
        return factors[optimized_differential_output] * optimized_wavelength.to(
            u.m) / optimized_angular_distance.to(u.rad) * u.rad

    def set_optimal_baseline(self,
                             star: Star,
                             optimized_differential_output: int,
                             optimized_wavelength: astropy.units.Quantity,
                             optimized_star_separation: Union[str, astropy.units.Quantity],
                             baseline_minimum: astropy.units.Quantity,
                             baseline_maximum: astropy.units.Quantity):
        """Set the baseline to optimize for the habitable zone, if it is between the minimum and maximum allowed
        baselines.

        :param star: The star object
        :param optimized_differential_output: The optimized differential output index
        :param optimized_wavelength: The optimized wavelength
        :param optimzied_star_separation: The angular radius of the habitable zone
        """
        # Get the optimized separation in angular units, if it is not yet in angular units
        if optimized_star_separation == 'habitable-zone':
            optimized_star_separation = star.habitable_zone_central_angular_radius
        elif optimized_star_separation.unit.is_equivalent(u.m):
            optimized_star_separation = optimized_star_separation.to(u.m) / star.distance.to(u.m) * u.rad

        # Get the optimal baseline and check if it is within the allowed range
        optimal_baseline = self._get_optimal_baseline(optimized_differential_output=optimized_differential_output,
                                                      optimized_wavelength=optimized_wavelength,
                                                      optimized_angular_distance=optimized_star_separation).to(u.m)

        if (baseline_minimum.to(u.m).value <= optimal_baseline.value
                and optimal_baseline.value <= baseline_maximum.to(u.m).value):
            self.array_configuration.baseline = optimal_baseline
        else:
            raise ValueError(
                f'Optimal baseline of {optimal_baseline} is not within allowed ranges of baselines {self.array_configuration.baseline_minimum}-{self.array_configuration.baseline_maximum}')
