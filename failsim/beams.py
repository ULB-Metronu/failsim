import os
import numpy as np
import pandas as pd
from scipy import integrate
from scipy import optimize
from typing import Dict
from numba import njit
from pint import UnitRegistry
import matplotlib.pyplot as plt

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
        twiss: input twiss parameter (betx * m, alfx, emitx, bety, alfy, emity, px, py). Emittance is geometric emittance
        coreIntensity: Intensity of the core
        coreSize:
        tailIntensity:
        tailSize:
        elens:
        tcp:
        eta:
        """
        if twiss is None:
            raise LHCBeamError("No Twiss provided. You should run sequence.get_initial_twiss() or give a dictionnary")
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

        self.x = np.array([])
        self.xp = np.array([])
        self.y = np.array([])
        self.yp = np.array([])
        self.t = np.array([])
        self.pt = np.array([])
        self.beam = np.array([])

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
                lambda x: _get_distribution(x, *params_distri, t[0]), self._elens, self._tcp)[0]
            exponentialTailCleaning = exponentialFactorIntegral / (integrate.quad(
                lambda x: self._get_reference(x, *params_ref), self._elens, self._tcp)[0])
            return exponentialTailCleaning

        params_distri = (self._coreIntensity, self._coreSize, self._tailIntensity,
                         self._tailSize, self._elens, self._tcp)
        params_ref = (self._coreIntensity, self._coreSize, self._tailIntensity, self._tailSize)
        return optimize.root(lambda t: exponentialTailCleaning(self, t) - (1 - self._eta), 1, method='hybr').x[0]

    @staticmethod
    @njit
    def get_particles(nparticles: int, params: np.array, xmin: float, xmax: float, ymin: float, ymax: float):
        pts_x = np.zeros(nparticles)
        pts_y = np.zeros(nparticles)
        i = 0
        while i < nparticles:
            u1 = xmin - (xmin - xmax) * np.random.rand()
            u2 = ymin - (ymin - ymax) * np.random.rand()
            if u2 <= _get_distribution(u1, *params):
                pts_x[i] = u1
                pts_y[i] = u2
                i += 1
        return [pts_x, pts_y]

    def generate(self, nparticles: int = int(1e4), xmin: float = None, xmax: float = None):
        if xmin is not None and xmax is not None:
            self._xmin = xmin
            self._xmax = xmax
        else:
            self._xmin = -self._tcp
            self._xmax = self._tcp

        params = (self._coreIntensity, self._coreSize, self._tailIntensity,
                  self._tailSize, self._elens, self._tcp, self._tau)

        # compute ymin and ymax of the distribution for performances
        if self._xmin < 0 and self._xmax < 0:
            ymin = _get_distribution(self._xmin, *params)
            ymax = _get_distribution(self._xmax, *params)
        elif self._xmin < 0 and self._xmax > 0:
            ymin = _get_distribution(self._xmin, *params)
            ymax = _get_distribution(0, *params)
        else:
            ymax = _get_distribution(self._xmin, *params)
            ymin = _get_distribution(self._xmax, *params)

        self.x = self.get_particles(nparticles, params, self._xmin, self._xmax, ymin, ymax)
        self.xp = self.get_particles(nparticles, params, self._xmin, self._xmax, ymin, ymax)
        self.y = self.get_particles(nparticles, params, self._xmin, self._xmax, ymin, ymax)
        self.yp = self.get_particles(nparticles, params, self._xmin, self._xmax, ymin, ymax)
        self.t = np.zeros(nparticles)
        self.pt = np.zeros(nparticles)

        self.beam = np.stack(np.dot(self._mat_denorm,
                                    np.array([self.x[0],
                                              self.xp[0],
                                              self.y[0],
                                              self.yp[0],
                                              self.t,
                                              self.pt])),
                             axis=1)

    def set_denormalization_matrix(self):
        return np.array(
            [[np.sqrt(self._twiss['emit_x'] * self._twiss['bet_x']), 0, 0, 0, 0, 0],
             [-np.sqrt(self._twiss['emit_x']) * self._twiss['alf_x'] / np.sqrt(
                 self._twiss['bet_x']),
              np.sqrt(self._twiss['emit_x']) / np.sqrt(self._twiss['bet_x']), 0, 0, 0, 0],
             [0, 0, np.sqrt(self._twiss['emit_y'] * self._twiss['bet_y']), 0, 0, 0],
             [0, 0, -np.sqrt(self._twiss['emit_y']) * self._twiss['alf_y'] / np.sqrt(
                 self._twiss['bet_y']),
              np.sqrt(self._twiss['emit_y']) / np.sqrt(self._twiss['bet_y']), 0, 0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
             ]
        )

    def check_integrals(self):
        int_red_curve = integrate.quad(
            lambda x: self._get_reference(x, self._coreIntensity, self._coreSize, self._tailIntensity,
                                          self._tailSize), self._elens, self._tcp)[0]
        int_blue_curve = integrate.quad(lambda x: _get_distribution(x, self._coreIntensity, self._coreSize,
                                                                    self._tailIntensity, self._tailSize, self._elens,
                                                                    self._tcp, self._tau), self._elens, self._tcp)[0]
        if np.isclose(int_blue_curve / int_red_curve, 1 - self._eta):
            print("No error.")
        else:
            raise LHCBeamError('Integrals are not correct.')

    def get_weight(self):
        return integrate.quad(lambda x: _get_distribution(x, self._coreIntensity, self._coreSize,
                                                          self._tailIntensity, self._tailSize, self._elens,
                                                          self._tcp, self._tau), self._xmin, self._xmax)[0]

    def write_for_madx(self, path: str = '.', filename: str = 'lhc_beam'):
        madx_beam = self.beam
        madx_beam[:, 1] = self.beam[:, 1] + self._twiss['p_x']
        madx_beam[:, 3] = self.beam[:, 3] + self._twiss['p_y']
        filename = os.path.join(path, filename)
        np.save(filename, madx_beam)

    def write_for_bdsim(self, path: str = '.', filename: str = 'lhc_beam', nslices: int = 1, compression: str = None):
        idx = 0
        w = self.get_weight()
        for d_beam in np.split(self.beam, nslices):
            df = pd.DataFrame(data={'x': d_beam[:, 0],
                                    'xp': d_beam[:, 1],
                                    'y': d_beam[:, 2],
                                    'yp': d_beam[:, 3],
                                    })
            df['E'] = self._twiss['energy']
            df['W'] = w
            df.to_csv(f"{os.path.join(path, filename)}_{idx}.dat",
                      header=False,
                      sep=' ',
                      index=False,
                      compression=compression)
            idx += 1

    def get_reference(self, x):
        return _gaussian(x, self._coreIntensity, self._coreSize) + _gaussian(x, self._tailIntensity, self._tailSize)

    def get_distribution(self, x):
        return np.vectorize(_get_distribution)(x, self._coreIntensity, self._coreSize, self._tailIntensity,
                                               self._tailSize, self._elens, self._tcp, self._tau)

    def plot(self, axes, with_samples: bool = True):
        x = np.linspace(-self._tcp, self._tcp, 1000)

        if with_samples:
            axes.plot(self.x[0], self.x[1], '.', markersize=0.05, label='sampled data')
        axes.plot(x, self.core(x), color='g', lw=2, label='core')
        axes.plot(x, self.get_reference(x), color='r', lw=2, label='reference')
        axes.plot(x, self.get_distribution(x), color='b', lw=2, label='elens')

        axes.axvline(self._elens, 1e-8, 1, color='black', lw=2, ls='--')
        axes.axvline(-self._elens, 1e-8, 1, color='black', lw=2, ls='--')

        axes.set_xlabel('Beam size (sigma)')
        axes.set_yscale('log')
        axes.grid(True)
        axes.legend()
        axes.set_xlim([-self._tcp, self._tcp])
