import os
import numpy as np
import pandas as pd

from scipy import integrate
from scipy import optimize

from typing import Dict
from numba import njit
from pint import UnitRegistry

_ureg = UnitRegistry()


@njit
def _get_distribution(x: float, coreIntensity: float, coreSize: float, tailIntensity: float, tailSize: float,
                      elens: float, tcp: float, tau: float):
    ref = coreIntensity * np.exp(-(x ** 2) / (2 * coreSize ** 2)) / (
            coreSize * np.sqrt(2 * np.pi)) + tailIntensity * np.exp(-(x ** 2) / (2 * tailSize ** 2)) / (
                  tailSize * np.sqrt(2 * np.pi))
    if elens > x > -elens:
        return ref
    elif (elens <= x < tcp) or (-elens >= x > -tcp):
        tails = np.exp(-((np.abs(x) - elens) / (tcp - elens)) / tau)
        return tails * ref
    else:
        return 0


@njit
def _gaussian(x: float, intensity: float, size: float):
    return intensity * np.exp(-(x ** 2) / (2 * size ** 2)) / (size * np.sqrt(2 * np.pi))


class LHCBeamError(Exception):
    pass


class LHCBeam:
    def __init__(self,
                 twiss: Dict = None,
                 coreIntensity: float = 1,
                 coreSize: float = 1.0,
                 tailIntensity: float = 0,
                 tailSize: float = 1.0,
                 elens: float = 4.7,
                 tcp: float = 6.7,
                 eta: float = 0.6
                 ):
        """
        twiss: input twiss parameter (beta_h * m, alpha_h, emit_h, beta_v * m, alpha_v, emit_v). Emittance is geometric emittance
        coreIntensity: Intensity of the core
        coreSize:
        tailIntensity:
        tailSize:
        elens:
        tcp:
        eta:
        """
        if twiss is None:
            twiss = {'beta_x': 0.1500121538058829 * _ureg.m,
                     'beta_y': 0.1500002840043648 * _ureg.m,
                     'alpha_x': 1.513983407593508e-05,
                     'alpha_y': -4.460920354037292e-05,
                     'emit_x': 0.0003351 * _ureg.mm * _ureg.mrad,
                     'emit_y': 0.0003351 * _ureg.mm * _ureg.mrad,
                     'energy': 7 * _ureg.TeV}
        self._twiss = twiss
        self._coreIntensity = coreIntensity
        self._coreSize = coreSize
        self._tailIntensity = tailIntensity
        self._tailSize = tailSize
        self._elens = elens
        self._tcp = tcp
        self._eta = eta

        self._tau = self.compute_tau()
        self._mat_denorm = self.set_denormalization_matrix()

    @property
    def coreIntensity(self):
        return self._coreIntensity

    @coreIntensity.setter
    def coreIntensity(self, val):
        self._coreIntensity = val

    def core(self, x):
        return _gaussian(x, 1, self._coreSize)

    @staticmethod
    @njit
    def _get_reference(x: float, coreIntensity: float, coreSize: float, tailIntensity: float, tailSize: float):
        return _gaussian(x, coreIntensity, coreSize) + _gaussian(x, tailIntensity, tailSize)

    @staticmethod
    @njit
    def get_tails(x: float, tau: float, elens: float, tcp: float):
        return np.exp(-((np.abs(x) - elens) / (tcp - elens)) / tau)

    def compute_tau(self):
        def exponentialTailCleaning(self, t):
            exponentialFactorIntegral = integrate.quad(
                lambda x: _get_distribution(x, self._coreIntensity, self._coreSize, self._tailIntensity, self._tailSize,
                                            self._elens, self._tcp, t[0]), self._elens, self._tcp)[0]
            exponentialTailCleaning = exponentialFactorIntegral / (integrate.quad(
                lambda x: self._get_reference(x, self._coreIntensity, self._coreSize, self._tailIntensity,
                                              self._tailSize), self._elens, self._tcp)[0])
            return exponentialTailCleaning

        return optimize.root(lambda t: exponentialTailCleaning(self, t) - (1 - self._eta), 1, method='hybr').x[0]

    @staticmethod
    @njit
    def get_particles(nparticles: int, coreIntensity: float, coreSize: float, tailIntensity: float, tailSize: float,
                      elens: float, tcp: float, tau: float):
        pts_x = np.zeros(nparticles)
        pts_y = np.zeros(nparticles)
        i = 0
        while i < nparticles:
            u1 = -tcp + (2 * tcp * np.random.rand())
            u2 = 0.4 * np.random.rand()  # TODO get maximum of distribution
            y_max = _get_distribution(u1, coreIntensity, coreSize, tailIntensity, tailSize, elens, tcp, tau)
            if u2 <= y_max:
                pts_x[i] = u1
                pts_y[i] = u2
                i += 1
        return [pts_x, pts_y]

    def generate(self, nparticles: int = int(1e4)):
        # TODO
        params = np.array([self._coreIntensity, self._coreSize, self._tailIntensity,
                                    self._tailSize, self._elens, self._tcp, self._tau])

        self.x = self.get_particles(nparticles, self._coreIntensity, self._coreSize, self._tailIntensity,
                                    self._tailSize, self._elens, self._tcp, self._tau)
        self.xp = self.get_particles(nparticles, self._coreIntensity, self._coreSize, self._tailIntensity,
                                     self._tailSize, self._elens, self._tcp, self._tau)
        self.y = self.get_particles(nparticles, self._coreIntensity, self._coreSize, self._tailIntensity,
                                    self._tailSize, self._elens, self._tcp, self._tau)
        self.yp = self.get_particles(nparticles, self._coreIntensity, self._coreSize, self._tailIntensity,
                                     self._tailSize, self._elens, self._tcp, self._tau)

        self.beam = np.stack(np.dot(self._mat_denorm,
                                    np.array([self.x[0], self.xp[0], self.y[0], self.yp[0]])),
                             axis=1)

    def set_denormalization_matrix(self):
        return np.array(
            [[np.sqrt(self._twiss['emit_x'].m_as("m * rad")) * np.sqrt(self._twiss['beta_x'].m_as("m")), 0, 0, 0],
             [-np.sqrt(self._twiss['emit_x'].m_as("m * rad")) * self._twiss['alpha_x'] / np.sqrt(
                 self._twiss['beta_x'].m_as("m")),
              np.sqrt(self._twiss['emit_x'].m_as("m")) / np.sqrt(self._twiss['beta_x'].m_as("m")), 0, 0],
             [0, 0, np.sqrt(self._twiss['emit_y'].m_as("m * rad")) * np.sqrt(self._twiss['beta_y'].m_as("m")), 0],
             [0, 0, -np.sqrt(self._twiss['emit_y'].m_as("m * rad")) * self._twiss['alpha_y'] / np.sqrt(
                 self._twiss['beta_y'].m_as("m")),
              np.sqrt(self._twiss['emit_y'].m_as("m * rad")) / np.sqrt(self._twiss['beta_y'].m_as("m"))]]
        )

    def check_integrals(self):
        int_red_curve = integrate.quad(
            lambda x: self._get_reference(x, self._coreIntensity, self._coreSize, self._tailIntensity,
                                          self._tailSize), self._elens, self._tcp)[0]
        int_blue_curve = integrate.quad(lambda x: _get_distribution(x, self._coreIntensity, self._coreSize,
                                                                    self._tailIntensity, self._tailSize, self._elens,
                                                                    self._tcp, self._tau), self._elens, self._tcp)[0]
        if not np.isclose(int_blue_curve / int_red_curve, 1 - self._eta):
            raise LHCBeamError('Integrals are not correct.')
        else:
            return "No error"

    def write_for_bdsim(self, path: str = '.', filename: str = 'lhc_beam', nslices: int = 1, compression: str = None):
        idx = 0
        for d_beam in np.split(self.beam, nslices):
            df = pd.DataFrame(data={'x': d_beam[:, 0],
                                    'xp': d_beam[:, 1],
                                    'y': d_beam[:, 2],
                                    'yp': d_beam[:, 3],
                                    })
            df['E'] = self._twiss['energy'].m_as("GeV")
            df.to_csv(f"{os.path.join(path, filename)}_{idx}.dat",
                      header=False,
                      sep=' ',
                      index=False,
                      compression=compression)
            idx += 1

    def get_reference(self, x):
        return _gaussian(x, self._coreIntensity, self._coreSize) + _gaussian(x, self._tailIntensity, self._tailSize)

    def get_distribution(self, x):
        return np.vectorize(_get_distribution)(x, self._coreIntensity, self._coreSize, self._tailIntensity, self._tailSize, self._elens, self._tcp, self._tau)

    def plot(self):
        pass
