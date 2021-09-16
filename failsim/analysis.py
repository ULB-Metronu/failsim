from __future__ import annotations
from typing import List
import os
import re
import glob
import pickle
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hist


class AnalysisHistogram:
    def __init__(self, hist_histogram=None):
        self._h = hist_histogram

    def __add__(self, other):
        return self.__class__(self._h + other._h)

    @classmethod
    def combine(cls, *histograms: List[AnalysisHistogram]):
        return functools.reduce(lambda x, y: x + y, histograms)


class LossPerTurnHistogram(AnalysisHistogram):
    def __init__(self, hist_histogram=None):
        if hist_histogram:
            self._h = hist_histogram
        else:
            self._h = hist.Hist(hist.axis.Regular(50, 0.5, 50.5, underflow=False, overflow=False, name="Turns"))

    def fill(self, data):
        self._h.fill(data['turn'])

    def plot(self):
        self._h.plot()


class Analysis:
    """Base class to support the analysis module."""

    def __init__(self, path: str):
        self._path = path
        self._histograms = None

    def save(self, filename: str):
        """Save the combined analysis data (histograms, plots, etc.) in a pickle format."""
        with open(os.path.join(self._path, filename), "wb") as f:
            if self._histograms is not None:
                pickle.dump(self._histograms, f)

    def __getitem__(self, k):
        return self._histograms[k]


class EventAnalysis(Analysis):
    SUFFIXES = {
        'loss': "loss",
        'tracking': "track",
        'beam': "beam",
        'summary': "summ",
        'info': "info",
        'twiss': "twiss",
    }

    def __init__(self, prefix: str = '', histograms: Optional[List[AnalysisHistogram]] = None, path: str = '.'):
        super().__init__(path)
        self._prefix = prefix
        self._histograms = histograms or []

    def __call__(self):
        loss_data = pd.read_parquet(os.path.join(self._path, f"{self._prefix}-{self.SUFFIXES['loss']}.parquet"))

        def _preprocess_data():
            """Preprocessing applied to the loss dataframe."""
            loss_data["group"] = loss_data.apply(lambda _: re.compile(r"^.*?(?=\.)").findall(_["element"])[0], axis=1)

        def _process_data():
            for h in self._histograms:
                h.fill(loss_data)

        _preprocess_data()
        _process_data()

    def save(self):
        super().save(filename=f"{self._prefix}-analysis.pkl")

    @property
    def results(self):
        return self._histograms


class AnalysisCombine(Analysis):
    def __init__(self, path: str = '.'):
        super().__init__(path)
        self._files = glob.glob(os.path.join(self._path, "*-analysis.pkl"))
        self._event_histograms = None
        self._histograms = None

        for i, filename in enumerate(self._files):
            with open(filename, 'rb') as f:
                _ = np.array(pickle.load(f))
                if self._event_histograms is None:
                    self._event_histograms = np.empty((len(self._files), len(_)), dtype='O')
                self._event_histograms[i, :] = _

    def __call__(self):
        self._histograms = np.apply_along_axis(lambda _: AnalysisHistogram.combine(*_), 0, self._event_histograms)

    def save(self, filename: str = 'combined.pkl'):
        super().save(filename)

    @property
    def results(self):
        return self._histograms