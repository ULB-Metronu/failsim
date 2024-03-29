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
import scipy
import pyarrow.parquet as pq
import hist
from .results import TrackingResult
from .beams import PDF


class AnalysisHistogram:
    def __init__(self, hist_histogram=None, maximum_turns=100):
        self._h = hist_histogram
        self._max_turns = maximum_turns
        self._load_parameters()

    def __add__(self, other):
        if other is None:
            r = self.__class(self._h)
        else:
            r = self.__class__(self._h + other.h)
        if self.groupby != other.groupby:
            raise Exception("Trying to add histograms of different types")
        else:
            r.groupby = self.groupby
        return r

    def _load_parameters(self):
        self.parameters = yaml.safe_load(open(os.path.join(pkg_resources.resource_filename('failsim', "data"), 'analysis_parameters.yaml')))

    @property
    def accumulator(self):
        return self

    @property
    def h(self):
        return self._h

    @classmethod
    def combine(cls, *histograms: List[AnalysisHistogram]):
        combined_histogram = functools.reduce(lambda x, y: x + y, histograms)
        return combined_histogram


class LossPerTurnHistogram(AnalysisHistogram):
    def __init__(self, hist_histogram=None, maximum_turns=100):
        super().__init__(hist_histogram)
        self._max_turns = maximum_turns
        self.groupby = None
        if hist_histogram:
            self._h = hist_histogram
        else:
            self._h = hist.Hist(hist.axis.Regular(
                self._max_turns, 0.5, self._max_turns + 0.5, underflow=False, overflow=False, name="Turns"
            ))

    @property
    def accumulator(self):
        return LossPerTurnHistogramAccumulator(self) + self

    def fill(self, data, beam_weight: Optional[float] = None):
        self._beam_weight = beam_weight
        if not data.empty:
            self._h.fill(data["turn"], weight=data["weight"].values)

    def plot(self, ax, total_weight: float = None, normalization: float = 1.0, cumulative: bool = False, **kwargs):
        tmp_hist = self._h.copy()
        if total_weight is not None:
            tmp_hist = normalization * tmp_hist / total_weight
        if cumulative:
            tmp_hist[:] = np.cumsum(tmp_hist[:])
        tmp_hist.plot(ax=ax, **kwargs)

    def threshold(self, value: float, total_weight: float = None, normalization: float = 1.0, cumulated: bool=True):
        def coarse():
            tmp = np.argwhere(cumulative > value)
            if len(tmp) > 0:
                return np.min(tmp)
            else:
                return np.nan
            
        def refined(coarse_value):
            return scipy.optimize.broyden1(
                scipy.interpolate.UnivariateSpline(self._h.axes[0].centers, cumulative - value, bounds_error=False, kind='quadratic'),
                coarse_value
            )
        
        cumulative = normalization * np.cumsum(self._h[:]) / total_weight
        coarse = coarse()
        if not np.isnan(coarse):
            try:
                return refined(coarse)
            except Exception:
                return coarse
        else:
            return coarse

class LossPerTurnHistogramAccumulator(AnalysisHistogram):
    def __init__(self, histogram: LossPerTurnHistogram):
        self._h = hist.Hist(*histogram._h.axes, storage=hist.storage.WeightedMean())
        self._weights = []

    def __add__(self, other):
        if not isinstance(other, LossPerTurnHistogram):
            raise Exception("Trying to accumulate histograms of different types")
        self._weights.append(other._beam_weight)
        self._h.fill(
            other._h.axes.centers[0][other._h.view() > 0], 
            sample=other._h.view()[other._h.view() > 0],
            weight=1#other._beam_weight
        )

        return self

    @property
    def total_weight(self):
        return np.sum(self._weights)

    def plot(self, ax, normalization: float = 1.0, cumulative: bool = False, **kwargs):
        tmp_hist = self._h.copy()
        tmp_hist = normalization * tmp_hist / self.total_weight
        tmp_hist.plot(ax=ax, **kwargs)

    def plot_cumulative(self, ax, normalization: float = 1.0, **kwargs):
        tmp_hist = self._h.copy()
        tmp_hist = normalization * tmp_hist / self.total_weight
        tmp_hist[:] = np.cumsum(tmp_hist[:])
        tmp_hist.plot(ax=ax, **kwargs)
        
class LossPerTurnByGroupHistogram(AnalysisHistogram):
    def __init__(self, hist_histogram=None, groupby: str = "element", maximum_turns=100):
        super().__init__(hist_histogram=hist_histogram)
        self._max_turns = maximum_turns
        self.groupby = groupby

        if hist_histogram:
            self._h = hist_histogram
        else:
            self._h = hist.Hist(
                hist.axis.Regular(
                    self._max_turns, 0.5, self._max_turns + 0.5, underflow=False, overflow=False, name="Turns"
                ),
                hist.axis.StrCategory(
                    self.parameters["groupby"][self.groupby], label="Collimators"
                )
            )

    def fill(self, data,beam_weight: Optional[float] = None):
        self._beam_weight = beam_weight
        self._h.fill(data["turn"], data[self.groupby], weight=data["weight"].values)

    def plot(self, ax, total_weight=None, cumulative: bool = False, legend_filter=1, **kwargs):
        temp_hist = self._h.copy()
        if total_weight is not None:
            temp_hist = 700 * temp_hist / total_weight

        def legend(data, h):
            if h.sum() != 0:
                if h[:, [data[0]]].sum() * 100 / h.sum() > legend_filter:
                    return data[0]

        legend_parameters = pd.DataFrame(AnalysisHistogram().parameters["groupby"][self.groupby]
                                         ).apply(legend, args=(temp_hist,), axis=1).dropna()

        if cumulative:
            for n in range(len(self._h[0, :].values())):
                temp_hist[:, n] = np.cumsum(temp_hist[:, n])

        temp_hist.stack(1).plot(stack=True, histtype="fill", legend=True, ax=ax, **kwargs)

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
                hist.axis.StrCategory(self.parameters["groupby"][self.groupby], label="Collimators", name="Collimators")
            )

    def fill(self, data,beam_weight: Optional[float] = None):
        self._beam_weight = beam_weight
        def _postprocess_data(df):
            return df["x"] - np.sign(df["x"]) * df["aper_1"]

        data["x_impact"] = data.apply(func=_postprocess_data, axis=1)

        self._h.fill(data["x_impact"], data["y"], data[self.groupby], weight=data["weight"].values)

    def plot(self, ax, total_weight=None, projection: str = "x", group: str = None, legend_filter=1):
        if total_weight is not None:
            self._h = 700 * self._h / total_weight

        def legend(data, h):
            if h.sum() != 0:
                if h[:, :, [data[0]]].sum() * 100 / h.sum() > legend_filter:
                    return data[0]

        legend_parameters = pd.DataFrame(self.parameters["groupby"][self.groupby]
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


class LossMap(AnalysisHistogram):
    def __init__(self, hist_histogram=None, groupby: str = "element", maximum_turns=100):
        """'Groupby' can be 'element', 'family' or 'turn' """
        super().__init__(hist_histogram)
        self._max_turns = maximum_turns
        self.groupby = groupby
        if hist_histogram:
            self._h = hist_histogram
        else:
            if self.groupby == "turn":
                self._h = hist.Hist(
                    hist.axis.Regular(26000, 0, 26658.8832, underflow=False, overflow=False, name="s", label="s"),
                    hist.axis.IntCategory(np.array(range(self._max_turns)) + 1, label="Turn", name="Turn"))
            else:
                self._h = hist.Hist(
                    hist.axis.Regular(26000, 0, 26658.8832, underflow=False, overflow=False, name="s", label="s"),
                    hist.axis.StrCategory(self.parameters["groupby"][self.groupby], label="Collimators")
                )

    def fill(self, data, beam_weight: Optional[float] = None):
        self._beam_weight = beam_weight
        self._h.fill(data["s"], data[self.groupby], weight=data["weight"].values)

    def plot(self, ax, total_weight=None, legend_filter=1):
        """ Use this to "zoom in" the axes: ``ax.xlim(19700, 20000)`` """
        if total_weight is not None:
            self._h = 700 * self._h / total_weight

        def legend(data, h):
            if h.sum() != 0:
                if h[:, [data[0]]].sum() * 100 / h.sum() > legend_filter:
                    return data[0]

        if self.groupby == "turn":
            legend_parameters = pd.DataFrame(
                range(self._h.axes[1][-1])
            ).apply(legend, args=(self._h,), axis=1).dropna()
        else:
            legend_parameters = pd.DataFrame(
                AnalysisHistogram().parameters["groupby"][self.groupby]
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
        ymin, ymax = ax.get_ylim()

        ax.axvline(x=26658.8832, ymin=0, ymax=1, color='red')
        ax.axvline(x=0, ymin=0, ymax=1, color='red')
        ax.text(0.1, ymax / 2, 'ip1', rotation=90, fontsize=30, clip_on=True)
        ax.axvline(x=3332.436584, ymin=0, ymax=1, color='red')
        ax.text(3332.536584, ymax / 2, 'ip2', rotation=90, fontsize=30, clip_on=True)
        ax.axvline(x=6664.7208, ymin=0, ymax=1, color='red')
        ax.text(6664.8208, ymax / 2, 'ip3', rotation=90, fontsize=30, clip_on=True)
        ax.axvline(x=9997.005016, ymin=0, ymax=1, color='red')
        ax.text(9997.105016, ymax / 2, 'ip4', rotation=90, fontsize=30, clip_on=True)
        ax.axvline(x=13329.289233, ymin=0, ymax=1, color='red')
        ax.text(13329.389233, ymax / 2, 'ip5', rotation=90, fontsize=30, clip_on=True)
        ax.axvline(x=16661.725816, ymin=0, ymax=1, color='red')
        ax.text(16661.825816, ymax / 2, 'ip6', rotation=90, fontsize=30, clip_on=True)
        ax.axvline(x=19994.1624, ymin=0, ymax=1, color='red')
        ax.text(19994.2624, ymax / 2, 'ip7', rotation=90, fontsize=30, clip_on=True)
        ax.axvline(x=23315.378984, ymin=0, ymax=1, color='red')
        ax.text(23315.478984, ymax / 2, 'ip8', rotation=90, fontsize=30, clip_on=True)


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
        'twiss_pre_thin': '_twiss_pre_thin_lhcb1',
    }

    def __init__(self, prefix: str = '', histograms: Optional[List[AnalysisHistogram]] = None,
                 beam_model: Optional[PDF] = None, path: str = '.'):
        super().__init__(path)
        self._prefix = prefix
        self._histograms = histograms or []
        self._beam_model = beam_model
        self._total_weight = 1.0

    def __call__(self):
        try:
            twiss = pd.read_parquet(
                os.path.join(self._path, "1-twiss.parquet"))
        except FileNotFoundError:
            twiss = None

        self._tr = TrackingResult.load_data(
            path=self._path,
            suffix=self._prefix,
            load_twiss=False,
            load_summ=False,
            load_info=False,
            load_track=True,
            load_loss=True,
            load_beam=True
        )
        self._tr.beam_distribution.model = self._beam_model
        self._total_weight = self._tr.compute_weights()  # Adds the weight column and return the total weight

        def _preprocess_data():
            """Preprocessing applied to the loss dataframe."""
            if not self._tr.loss_df.empty:
                self._tr.loss_df["family"] = self._tr.loss_df.apply(
                    lambda _: re.compile(r"^.*?(?=\.)").findall(_["element"])[0], axis=1)
                if twiss is not None:
                    self._tr.loss_df["aper_1"] = twiss["aper_1"]

        def _process_data():
            for h in self._histograms:
                h.fill(self._tr.loss_df, beam_weight=self._total_weight)
            self._histograms.append(self._total_weight)

        if self._tr.loss_df is not None:
            _preprocess_data()
            _process_data()

    def save(self):
        self._tr.save_data(path=self._path, suffix=self._prefix, only_tracking=True, beam_suffix=str(self._beam_model))
        np.save(os.path.join(self._path, f"{self._prefix}-weights-{self._beam_model}.npy"), self._tr.track_df.query("turn == 0.0")[['number', 'weight']].values)
        super().save(filename=f"{self._prefix}-analysis-{self._beam_model}.pkl")

    @property
    def results(self):
        return self._histograms


class AnalysisCombineLosses(Analysis):
    def __init__(self, path: str = '.', beam_model: str = 'DoubleGaussianPDF', last_file_prefix: Optional[int] = None):
        super().__init__(path)
        self._files = glob.glob(os.path.join(self._path, f"*-analysis-{beam_model}.pkl"))
        self._last_file_prefix = last_file_prefix
        self._histograms = []
    
    def __call__(self):
        for filename in self._files:
            if self._last_file_prefix is not None:
                r = re.match(r".*\/(\d{1,3})-analysis.*.pkl", filename)
                if not r:
                    continue
                if int(r.groups()[0]) > self._last_file_prefix:
                    continue
            with open(filename, 'rb') as f:
                _ = np.array(pickle.load(f))
            for j, h in enumerate(_):
                try:
                    self._histograms[j] += h
                except IndexError:
                    self._histograms.append(getattr(h, 'accumulator', h))

    def save(self, filename: str = 'combined.pkl'):
        super().save(filename)

    @property
    def results(self):
        return self._histograms


class AnalysisCombineTracks(Analysis):
    def __init__(self, path: str = '.', filters: Optional[List]=None, beam_model: str='DoubleGaussianPDF'):
        super().__init__(path)
        self._files = glob.glob(self._path + f"/**track-{beam_model}.parquet")
        self._beam_model = beam_model
        self._filters = filters
        nthreads = 1 if filters is not None else 64
        self._dataset = pq.ParquetDataset(
            self._files,
            metadata_nthreads=nthreads,
            filters=filters,
            use_legacy_dataset=filters is None,
        )
        self._data = None

    def __call__(self, columns: Optional[List[str]] = None):
        print(f"Reading the data using filters: {self._filters}")
        self._data = self._dataset.read(columns=columns, use_threads=True)

    def save(self, filename: str = 'combined-tracks.parquet'):
        pq.write_table(self._data, os.path.join(self._path, filename))

    @property
    def results(self):
        return self._data
