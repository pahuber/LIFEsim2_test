from typing import Any, Optional

from pydantic import BaseModel

from sygn.core.entities.noise import Noise


class Settings(BaseModel):
    """Class representation of the simulation configurations.

    """
    grid_size: int
    time_steps: int
    planet_orbital_motion: bool
    noise: Optional[Noise]
    integration_time: Any
    number_of_inputs: int
    time_step: Any = None

    def __init__(self, **data):
        """Constructor method.

        :param data: Data to initialize the star class.
        """
        super().__init__(**data)
        self.time_step = self.integration_time / self.time_steps
        self.noise.phase_perturbation_distribution = self.noise.get_perturbation_distribution(
            self.number_of_inputs,
            self.time_step,
            self.time_steps,
            self.noise.phase_perturbations.rms,
            self.noise.phase_perturbations.power_law_exponent)
        self.noise.polarization_perturbation_distribution = self.noise.get_perturbation_distribution(
            self.number_of_inputs,
            self.time_step,
            self.time_steps,
            self.noise.polarization_perturbations.rms,
            self.noise.polarization_perturbations.power_law_exponent)

        # # Plot
        # plt.plot(self.noise.phase_perturbation_distribution, color='#008080', lw=2)
        # plt.title('OPD Perturbation Time Series')
        # plt.xlabel('Time Steps (a.u.)')
        # plt.ylabel('OPD Perturbation Amplitude (nm)')
        # plt.savefig('phase_perturbation.pdf', bbox_inches='tight')
        # plt.show()
        #
        # a = (abs(fft(self.noise.phase_perturbation_distribution)) ** 2)
        # plt.plot(a[:50], color='#008080', label='PSD')
        # plt.plot([0] + np.max(a) / np.linspace(1, 49, 49), color='k', ls='--', label='1/f')
        # plt.title('OPD Perturbation PSD')
        # plt.xlabel('Frequency (a.u.)')
        # plt.ylabel('OPD Perturbation Power (a.u.)')
        # plt.legend()
        # plt.savefig('phase_perturbation_psd.pdf', bbox_inches='tight')
        # plt.show()
        # a = 0
