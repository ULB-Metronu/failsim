from __future__ import annotations
from typing import Optional, Dict, Tuple
import os
import numpy as np
import pandas as pd
import scipy.stats

def generate_multivariate_uniform_ring(ndims: int, nsamples: int=1, inner_radius: float=0.0, outer_radius:float=1.0, **kwargs):
    _ = np.random.normal(size=(nsamples, ndims))
    _ = _ / np.linalg.norm(_, axis=-1, keepdims=True)
    __ = np.random.uniform(size=(nsamples, ))
    __ = (__ * (outer_radius**ndims - inner_radius**ndims) + inner_radius**ndims)**(1/ndims)

    return _ * __[:, None]


def generate_double_multivariate_gaussian(ndims: int, nsamples: int=1, weight: float=0.8, sigma2: float=2.0, **kwargs):
    _ = np.random.uniform(size=int(nsamples))
    return np.concatenate(
        (
            np.random.multivariate_normal(mean=np.zeros(ndims), cov=np.eye(ndims),        size=int(nsamples))[_ <= weight], 
            np.random.multivariate_normal(mean=np.zeros(ndims), cov=sigma2*np.eye(ndims), size=int(nsamples))[_ >  weight]
        )
    )

class PDF:
    def __call__(self, x: np.ndarray):
        return self._pdf(x[:, 0:4])

    def __str__(self):
        return f"{self.__class__.__name__}"


class UniformPDF(PDF):
    def __init__(self):
        self._pdf = lambda _: np.ones(len(_))

class GaussianPDF(PDF):
    def __init__(
        self,
        intensity: float=0.8,
        size: float=1.0,
        ):
        self._pdf = lambda _: intensity * scipy.stats.multivariate_normal(mean=np.zeros(4), cov=size*np.eye(4)).pdf(_)

class DoubleGaussianPDF(PDF):
    def __init__(
        self,
        coreIntensity: float=0.8,
        coreSize: float=1.0,
        tailIntensity: float=0.2,
        tailSize: float=2.0,
        ):
        self._pdf = lambda _: coreIntensity * scipy.stats.multivariate_normal(mean=np.zeros(4), cov=coreSize**2*np.eye(4)).pdf(_) \
            + tailIntensity * scipy.stats.multivariate_normal(mean=np.zeros(4), cov=tailSize**2*np.eye(4)).pdf(_)


class DoubleGaussianELensHardEdgePDF(PDF):
    def __init__(
        self,
        coreIntensity: float=0.8,
        coreSize: float=1.0,
        tailIntensity: float=0.2,
        tailSize: float=2.0,
        elens: float=4.7
        ):
        def pdf(_):            
            r = coreIntensity * scipy.stats.multivariate_normal(mean=np.zeros(4), cov=coreSize**2*np.eye(4)).pdf(_) \
            + tailIntensity * scipy.stats.multivariate_normal(mean=np.zeros(4), cov=tailSize**2*np.eye(4)).pdf(_)
            r[np.linalg.norm(_, axis=1) > elens] = 0.0
            return r

        self._pdf = pdf

class DoubleGaussianElensExponentialTailDepletionPDF(DoubleGaussianPDF):
    def __init__(
        self,
        coreIntensity: float=0.8,
        coreSize: float=1.0,
        tailIntensity: float=0.2,
        tailSize: float=2.0,
        elens: float=4.7,
        tcp: float=6.7,
        eta: float=0.8
        ):
            super().__init__(coreIntensity, coreSize, tailIntensity, tailSize)
            self.coreIntensity = coreIntensity
            self.coreSize = coreSize
            self.tailIntensity = tailIntensity
            self.tailSize = tailSize
            self.elens = elens
            self.tcp = tcp
            self.eta = eta
            self.tau = self.compute_tau()

            spdf = self._pdf
            def pdf(_):
                return spdf(_) * self.depletion_factor(np.linalg.norm(_, axis=1))
            
            self._pdf = pdf

    def __str__(self):
        return f"{self.__class__.__name__}{self.eta}" 
        
    def radial_distribution(self, x, depleted: bool=True):
        factor = 1.0
        if depleted:
            factor = np.exp(-(x - self.elens) / (self.tcp - self.elens) / self.tau)
        factor[x < self.elens] = 1.0
        return  factor * (
              self.coreIntensity * scipy.stats.chi(4).pdf(x/self.coreSize) / self.coreSize
            + self.tailIntensity * scipy.stats.chi(4).pdf(x/self.tailSize) / self.tailSize
        )
    
    def depletion_factor(self, x):
        f = np.exp(-(x - self.elens) / (self.tcp - self.elens) / self.tau)
        f[x < self.elens] = 1.0
        f[x > self.tcp] = 0.0
        return f
        
    def compute_tau(self):
        def exponentialTailCleaning(tau):
            ref = (
                self.coreIntensity * (scipy.stats.chi(4).cdf(self.tcp/self.coreSize) - scipy.stats.chi(4).cdf(self.elens/self.coreSize)) / self.coreSize
              + self.tailIntensity * (scipy.stats.chi(4).cdf(self.tcp/self.tailSize) - scipy.stats.chi(4).cdf(self.elens/self.tailSize)) / self.tailSize
            )
            return ((
                (self.coreIntensity * scipy.integrate.quad(
                    lambda _: scipy.stats.chi(4).pdf(_) * np.exp(-((_ - self.elens/self.coreSize) / (self.tcp/self.coreSize - self.elens/self.coreSize)) / tau), 
                    self.elens/self.coreSize,
                    self.tcp/self.coreSize
                )[0] / self.coreSize)
              + (self.tailIntensity * scipy.integrate.quad(
                    lambda _: scipy.stats.chi(4).pdf(_) * np.exp(-((_ - self.elens/self.tailSize) / (self.tcp/self.tailSize - self.elens/self.tailSize)) / tau), 
                    self.elens/self.tailSize,
                    self.tcp/self.tailSize
                )[0] / self.tailSize))
                / ref)

        return scipy.optimize.fsolve(lambda t: exponentialTailCleaning(t) - (1 - self.eta),
                             np.array([0.1]))[0]

class QGaussianPDF(PDF):
    pass


class LHCBeamError(Exception):
    pass


class Beam:
    GENERATOR = None

    def __init__(self, closed_orbit: Optional[np.ndarray] = None, twiss: Optional[Dict] = None, model: Optional[PDF]=None):
        self._closed_orbit= closed_orbit if closed_orbit is not None else np.zeros(6)
        self._twiss = twiss or {}
        self._model = model
        self._weights = None

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, new_model: PDF):
        self._model = new_model
        self._weights = self.compute_weights(self._normalized_distribution)

    @property
    def weights(self):
        return self._weights

    @property
    def distribution(self):
        return self._beam

    @property
    def normalized_distribution(self):
        return self._normalized_distribution

    @property
    def distribution_with_weights(self):
        if self._model is not None:
            return np.hstack((self.distribution, self.weights.reshape((self.weights.shape[0], 1))))
        else:
            return np.hstack((self.distribution, np.ones((self.distribution.shape[0], 1))))

    @property
    def emittances(self):
        return (
            np.sqrt(np.linalg.det(np.cov(self.distribution[:, 0:2], rowvar=False, aweights=self.weights))),
            np.sqrt(np.linalg.det(np.cov(self.distribution[:, 2:4], rowvar=False, aweights=self.weights)))
        )

    def compute_weights(self, distribution):
        if self._model is not None:
            return self._model(distribution)

    def denormalization_matrix(self):
        t = self._twiss
        if not t:
            return np.eye(6)
        return np.array([
            [np.sqrt(t['emit_x'] * t['bet_x']), 0, 0, 0, 0, 0],
            [-np.sqrt(t['emit_x']) * t['alf_x'] / np.sqrt(t['bet_x']), np.sqrt(t['emit_x']) / np.sqrt(t['bet_x']), 0, 0, 0, 0],
            [0, 0, np.sqrt(t['emit_y'] * t['bet_y']), 0, 0, 0],
            [0, 0, -np.sqrt(t['emit_y']) * t['alf_y'] / np.sqrt(t['bet_y']), np.sqrt(t['emit_y']) / np.sqrt(t['bet_y']), 0, 0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])

    def weight_from_denormalized_distribution(self, distribution: np.ndarray):
        return self.compute_weights(np.dot(distribution - self._closed_orbit, np.linalg.inv(self.denormalization_matrix()).T))

    def generate(self, nparticles: int, **kwargs):
        self._normalized_distribution = np.hstack((
            self.__class__.GENERATOR(ndims=4, nsamples=nparticles, **kwargs),
            np.zeros((nparticles, 2))
            ))
        self._weights = self.compute_weights(self._normalized_distribution)
        self._beam = np.dot(self._normalized_distribution, self.denormalization_matrix().T) + self._closed_orbit
        return self

    def filter_by_indices(self, indices: np.array):
        self._beam = self.beam[indices]
        self._weights = self._weights[indices]
        self._normalized_distribution = self._normalized_distribution[indices]

class UniformBeam(Beam):

    GENERATOR = generate_multivariate_uniform_ring

    def generate(self, nparticles: int, inner_radius: float=0.0, outer_radius: float=1.0):
        args = locals()
        del args['self']
        return super().generate(**args)


class DoubleGaussianBeam(Beam):

    GENERATOR = generate_double_multivariate_gaussian

    def generate(self, nparticles: int, weight=0.8, **kwargs):
        args = locals()
        del args['self']
        args['weight'] = weight
        args['sigma2'] = 2.0
        return super().generate(**args)

class GaussianBeam(DoubleGaussianBeam):

    def generate(self, nparticles: int):
        args = locals()
        del args['self']
        return super().generate(weight=1.0, **args)
