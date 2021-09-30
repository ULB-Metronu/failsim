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


def generate_double_multivariate_gaussian(ndims: int, nsamples: int=1, weight: float=0.8, sigma2: float=2.0):
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

class DoubleGaussianPDF(PDF):
    def __init__(
        self,
        coreIntensity: float=1.0,
        coreSize: float=1.0,
        tailIntensity: float=0.0,
        tailSize: float=2.0,
        ):
        self._pdf = lambda _: coreIntensity * scipy.stats.multivariate_normal(mean=np.zeros(4), cov=coreSize*np.eye(4)).pdf(_) \
            + tailIntensity * scipy.stats.multivariate_normal(mean=np.zeros(4), cov=tailSize*np.eye(4)).pdf(_)

class ExponentialTailDepletionPDF(PDF):
    pass

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


class UniformBeam(Beam):

    GENERATOR = generate_multivariate_uniform_ring

    def generate(self, nparticles: int, inner_radius: float=0.0, outer_radius: float=1.0):
        args = locals()
        del args['self']
        return super().generate(**args)


class DoubleGaussianBeam(Beam):

    GENERATOR = generate_double_multivariate_gaussian

    def generate(self, nparticles: int, inner_radius: float=0.0, outer_radius: float=1.0):
        args = locals()
        del args['self']
        print(args)
        return super().generate(**args)

class GaussianBeam(DoubleGaussianBeam):

    def generate(self, nparticles: int, inner_radius: float=0.0, outer_radius: float=1.0):
        args = locals()
        del args['self']
        print(args)
        return super().generate(**args)
