from __future__ import annotations
from typing import Optional, Dict, Tuple
import os
import numpy as np
import pandas as pd
import scipy.stats

def generate_multivariate_uniform_ring(ndims: int, center: Optional[np.ndarray]=None, inner_radius: float=0.0, outer_radius:float=1.0, nsamples: int=1):
    if center is None:
        center = np.zeros(ndims)

    _ = np.random.normal(size=(nsamples, ndims))
    _ = _ / np.linalg.norm(_, axis=-1, keepdims=True)
    __ = np.random.uniform(size=(nsamples, ))
    __ = (__ * (outer_radius**ndims - inner_radius**ndims) + inner_radius**ndims)**(1/ndims)

    return _ * __[:, None] + center

class PDF:
    def __call__(self, x: np.ndarray):
        return self._pdf(x[:, 0:4])

class DoubleGaussianPDF(PDF):
    def __init__(
        self,
        coreIntensity: float=0.8,
        coreSize: float=1.0,
        tailIntensity: float=0.2,
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
    def __init__(self, closed_orbit: Tuple, twiss: Dict, model: PDF):
        self._orbit= closed_orbit
        self._twiss = twiss
        self._model = model

    def generate(self, nparticles: int, inner_radius: float=0.0, outer_radius: float=1.0):
        _ = np.hstack((
            generate_multivariate_uniform_ring(ndims=4, inner_radius=inner_radius, outer_radius=outer_radius, nsamples=nparticles), 
            np.zeros((nparticles, 2))
            ))
        self._weights = self._compute_weights(_)
        self._beam = np.dot(_, self.denormalization_matrix().T)
        return self

    @property
    def weights(self):
        return self._weights

    @property
    def distribution(self):
        return self._beam

    @property
    def distribution_with_weights(self):
        return np.hstack((self.distribution, self.weights.reshape((self.weights.shape[0], 1))))

    def _compute_weights(self, distribution):
        return self._model(distribution)

    def denormalization_matrix(self):
        t = self._twiss
        return np.array([
            [np.sqrt(t['emit_x'] * t['bet_x']), 0, 0, 0, 0, 0],
            [-np.sqrt(t['emit_x']) * t['alf_x'] / np.sqrt(t['bet_x']), np.sqrt(t['emit_x']) / np.sqrt(t['bet_x']), 0, 0, 0, 0],
            [0, 0, np.sqrt(t['emit_y'] * t['bet_y']), 0, 0, 0],
            [0, 0, -np.sqrt(t['emit_y']) * t['alf_y'] / np.sqrt(t['bet_y']), np.sqrt(t['emit_y']) / np.sqrt(t['bet_y']), 0, 0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ])

    def weight_from_denormalized_distribution(self, distribution: np.ndarray):
        return self._compute_weights(np.dot(distribution, np.linalg.inv(self.denormalization_matrix()).T))
        
