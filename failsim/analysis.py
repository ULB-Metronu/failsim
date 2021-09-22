from __future__ import annotations
from typing import List
import os
import re
import glob
import yaml
import pickle
import functools
import pkg_resources
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import hist


class AnalysisHistogram:
    def __init__(self, hist_histogram=None):
        self._h = hist_histogram
        self._load_parameters()

    def __add__(self, other):
        return self.__class__(self._h + other._h)

    def _load_parameters(self):
        self.parameters = yaml.safe_load(open(os.path.join(pkg_resources.resource_filename('failsim', "data"),
                         'analysis_parameters.yaml')))

    @classmethod
    def combine(cls, *histograms: List[AnalysisHistogram]):
        return functools.reduce(lambda x, y: x + y, histograms)


class LossPerTurnHistogram(AnalysisHistogram):
    def __init__(self, hist_histogram=None, cumulative:bool =False):
        self.cumulative = cumulative
        if hist_histogram:
            self._h = hist_histogram
        else:
            self._h = hist.Hist(hist.axis.Regular(50, 0.5, 50.5, underflow=False, overflow=False, name="Turns"))

    def fill(self, data):
        self._h.fill(data["turn"]);
        if self.cumulative:
            self._h[:] = np.cumsum(self._h[:])
    def plot(self):
        self._h.plot(histtype="fill")


class LossPerTurnByGroupHistogram(AnalysisHistogram):
    def __init__(self, hist_histogram=None, groupby: str= "element", cumulative:bool =False):
        self.groupby = groupby
        self.cumulative = cumulative
        if hist_histogram:
          self._h = hist_histogram
        else:
            self._h = hist.Hist(
                hist.axis.Regular(50, 0.5, 50.5, underflow=False, overflow=False, name="Turns"),
                hist.axis.StrCategory(AnalysisHistogram().parameters["groupby"][self.groupby], label="Collimators")
            )

    def fill(self, data):
        self._h.fill(data["turn"], data[self.groupby])
        if self.cumulative:
            for n in range(len(self._h[0, :].values())):
                self._h[:, n] = np.cumsum(self._h[:, n])

        # df = data.groupby(["element"])
        # elements = list(df.groups.keys())
        # [self._h.fill(df.get_group(item)["turn"], item) for item in elements]

    def plot(self):
        self._h.stack(1)
        self._h.plot(stack=True, histtype="fill")


class ImpactParameterHistogram(AnalysisHistogram):
    def __init__(self, hist_histogram=None, groupby: str= "element", cumulative:bool =False):
        self.groupby = groupby
        self.cumulative = cumulative
        if hist_histogram:
          self._h = hist_histogram
        else:
            self._h = hist.Hist(
                axis.Regular(200, 0.0, 0.0022, underflow=False, overflow=False, name="x"),
                axis.Regular(200, 0.0, 0.0022, underflow=False, overflow=False, name="y"),
                axis.StrCategory(params["groupby"]["element"], label="Collimators", name="Collimators")
            )

    def fill(self, data):
        self._h.fill(data["x"], data["y"], data[self.groupby])

    def plot(self, projection: str="x", element: str=None):
        self._h.project(projection, "Collimators")
        if element is not None:
            self._h[:, element].plot(stack=True, histtype="fill")
        else:
            self._h.plot(stack=True, histtype="fill")


class Analysis:
    """Base class to support the analysis module."""

    def __init__(self, path: str):
        self._path = path
        self._histograms = None

    def save(self, filename: str, property: str = '_histograms'):
        """Save the combined analysis data (histograms, plots, etc.) in a pickle format."""
        with open(os.path.join(self._path, filename), "wb") as f:
            if self._histograms is not None:
                pickle.dump(getattr(self, property), f)

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
        try:
            loss_data = pd.read_parquet(os.path.join(self._path, f"{self._prefix}-{self.SUFFIXES['loss']}.parquet"))
        except FileNotFoundError:
            loss_data = None

        def _preprocess_data():
            """Preprocessing applied to the loss dataframe."""
            loss_data["family"] = loss_data.apply(lambda _: re.compile(r"^.*?(?=\.)").findall(_["element"])[0], axis=1)

        def _process_data():
            for h in self._histograms:
                h.fill(loss_data)

        if loss_data is not None:
            _preprocess_data()
            _process_data()

    def save(self):
        super().save(filename=f"{self._prefix}-analysis.pkl")

    @property
    def results(self):
        return self._histograms


class AnalysisCombineLosses(Analysis):
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


class AnalysisCombineTracks(Analysis):
    def __init__(self, path: str = '.', filters: Optional[List]=None):
        super().__init__(path)
        self._files = glob.glob(self._path + "/**track.parquet")
        nthreads = 1 if filters is not None else 64
        self._dataset = pq.ParquetDataset(
            self._files,
            metadata_nthreads=1,
            filters=filters,
            use_legacy_dataset=filters is None,
        )
        self._data = None

    def __call__(self, columns: Optional[List[str]]=None):
        self._data = self._dataset.read(columns=columns, use_threads=True)

    def save(self, filename: str = 'combined-tracks.parquet'):
        pq.write_table(self._data, os.path.join(self._path, filename))

    @property
    def results(self):
        return self._data