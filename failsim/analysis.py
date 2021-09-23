from __future__ import annotations
from typing import Optional, List
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
    def __init__(self, hist_histogram=None):
        super().__init__(hist_histogram)
        if hist_histogram:
            self._h = hist_histogram
        else:
            self._h = hist.Hist(hist.axis.Regular(50, 0.5, 50.5, underflow=False, overflow=False, name="Turns"))

    def fill(self, data):
        self._h.fill(data["turn"])

    def plot(self, ax, cumulative: bool = False):
        if cumulative:
            self._h[:] = np.cumsum(self._h[:])
        self._h.plot(histtype="fill", ax=ax)


class LossPerTurnByGroupHistogram(AnalysisHistogram):
    def __init__(self, hist_histogram=None, groupby: str = "element"):
        super().__init__(hist_histogram)
        self.groupby = groupby
        if hist_histogram:
            self._h = hist_histogram
        else:
            self._h = hist.Hist(
                hist.axis.Regular(50, 0.5, 50.5, underflow=False, overflow=False, name="Turns"),
                hist.axis.StrCategory(self.parameters["groupby"][self.groupby], label="Collimators")
            )

    def fill(self, data):
        self._h.fill(data["turn"], data[self.groupby])

    def plot(self, ax, cumulative: bool = False):
        if cumulative:
            for n in range(len(self._h[0, :].values())):
                self._h[:, n] = np.cumsum(self._h[:, n])

        def legend(data, h):
            if h.sum() != 0:
                if h[:, [data[0]]].sum() * 100 / h.sum() > 1:
                    return data[0]

        legend_parameters = pd.DataFrame(AnalysisHistogram().parameters["groupby"][self.groupby]
                                         ).apply(legend, args=(self._h,), axis=1).dropna()
        self._h.stack(1).plot(stack=True, histtype="fill", legend=True, ax=ax)

        artists = np.take(ax.get_legend_handles_labels()[0], -legend_parameters.index - 1)
        labels = np.take(ax.get_legend_handles_labels()[1], -legend_parameters.index - 1)
        ax.legend(artists,
                  labels,
                  prop={'size': 30},
                  loc='center left',
                  bbox_to_anchor=(0.965, 0.5),
                  fancybox=True,
                  shadow=True,
                  ncol=1)


class ImpactParameterHistogram(AnalysisHistogram):
    def __init__(self, hist_histogram=None, groupby: str = "element"):
        super().__init__(hist_histogram)
        self.groupby = groupby
        if hist_histogram:
            self._h = hist_histogram
        else:
            self._h = hist.Hist(
                hist.axis.Regular(400, -0.0022, 0.0022, underflow=False, overflow=False, name="x"),
                hist.axis.Regular(400, -0.0022, 0.0022, underflow=False, overflow=False, name="y"),
                hist.axis.StrCategory(self.parameters["groupby"][self.groupby], label="Collimators",
                                      name="Collimators")
            )

    def fill(self, data):
        def _postprocess_data(df):
            return df["x"] - np.sign(df["x"]) * df["aper_1"]

        data["x"] = data.apply(func=_postprocess_data, axis=1)

        self._h.fill(data["x"], data["y"], data[self.groupby])

    def plot(self, ax, projection: str = "x", group: str = None, ):
        def legend(data, h):
            if h.sum() != 0:
                if h[:, :, [data[0]]].sum() * 100 / h.sum() > 1:
                    return data[0]

        legend_parameters = pd.DataFrame(self.parameters["groupby"]["element"]
                                         ).apply(legend, args=(self._h,), axis=1).dropna()

        if group is not None:
            self._h.project(projection, "Collimators")[:, group].plot(stack=True, histtype="fill", ax=ax)
        else:
            self._h.project(projection, "Collimators").stack(1).plot(stack=True, histtype="fill", legend=True, ax=ax)

            artists = np.take(ax.get_legend_handles_labels()[0], -legend_parameters.index - 1)
            labels = np.take(ax.get_legend_handles_labels()[1], -legend_parameters.index - 1)

            ax.legend(artists,
                      labels,
                      prop={'size': 30},
                      loc='center left',
                      bbox_to_anchor=(0.965, 0.5),
                      fancybox=True,
                      shadow=True,
                      ncol=1)


class Analysis:
    """Base class to support the analysis module."""

    def __init__(self, path: str):
        self._path = path
        self._histograms = None

    def save(self, filename: str, property_name: str = '_histograms'):
        """Save the combined analysis data (histograms, plots, etc.) in a pickle format."""
        with open(os.path.join(self._path, filename), "wb") as f:
            if self._histograms is not None:
                pickle.dump(getattr(self, property_name), f)

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
        'twiss_pre_thin': 'twiss_pre_thin_lhcb1',
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
        try:
            twiss_pre_thin = pd.read_parquet(
                os.path.join(self._path, f"{self._prefix}-{self.SUFFIXES['twiss_pre_thin']}.parquet"))
        except FileNotFoundError:
            twiss_pre_thin = None

        def _preprocess_data():
            """Preprocessing applied to the loss dataframe."""
            loss_data["family"] = loss_data.apply(lambda _: re.compile(r"^.*?(?=\.)").findall(_["element"])[0], axis=1)
            loss_data["aper_1"] = twiss_pre_thin["aper_1"]

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
    def __init__(self, path: str = '.', filters: Optional[List] = None):
        super().__init__(path)
        self._files = glob.glob(self._path + "/**track.parquet")
        nthreads = 1 if filters is not None else 64
        self._dataset = pq.ParquetDataset(
            self._files,
            metadata_nthreads=nthreads,
            filters=filters,
            use_legacy_dataset=filters is None,
        )
        self._data = None

    def __call__(self, columns: Optional[List[str]] = None):
        self._data = self._dataset.read(columns=columns, use_threads=True)

    def save(self, filename: str = 'combined-tracks.parquet'):
        pq.write_table(self._data, os.path.join(self._path, filename))

    @property
    def results(self):
        return self._data
