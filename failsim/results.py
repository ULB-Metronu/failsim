from typing import Optional, List, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np

import plotly.graph_objects as go


@dataclass
class Result:
    """Docstring for Result. """

    twiss_df: pd.DataFrame
    summ_df: pd.DataFrame
    run_version: int
    hllhc_version: float

    def __init__(
        self,
        twiss_df: pd.DataFrame,
        summ_df: pd.DataFrame,
        run_version: int = 0,
        hllhc_version: float = 0.0,
    ):
        """TODO: Docstring

        Args:
            twiss_df (TODO): TODO
            summ_df (TODO): TODO

        Kwargs:
            run_version (TODO): TODO
            hllhc_version (TODO): TODO


        """
        self.twiss_df = twiss_df
        self.summ_df = summ_df
        self.run_version = run_version
        self.hllhc_version = hllhc_version

        self.fix_twiss_name()

    def fix_twiss_name(self):
        """TODO: Docstring for fix_twiss_name.
        Returns: TODO

        """
        self.twiss_df["name"] = self.twiss_df.apply(lambda x: x["name"][:-2], axis=1)


@dataclass
class TrackingResult(Result):

    """Docstring for TrackingResult. """

    track_df: pd.DataFrame

    def __init__(
        self,
        twiss_df: pd.DataFrame,
        summ_df: pd.DataFrame,
        track_df: pd.DataFrame,
        run_version: int = 0,
        hllhc_version: float = 0.0,
        eps_n: float = 2.5e-6,
        nrj: float = 7000,
    ):
        """TODO: Docstring

        Args:
            twiss_df (TODO): TODO
            summ_df (TODO): TODO
            track_df (TODO): TODO

        Kwargs:
            run_version (TODO): TODO
            hllhc_version (TODO): TODO
            eps_n (TODO): TODO
            nrj (TODO): TODO


        """
        Result.__init__(self, twiss_df, summ_df, run_version, hllhc_version)

        self.track_df = track_df

        self.normalize_track(eps_n, nrj)

    def normalize_track(self, eps_n: float = 2.5e-6, nrj: float = 7000):
        """TODO: Docstring for normalize_track.

        Kwargs:
            eps_n (float): TODO
            nrj (float): TODO

        Returns: TODO

        """
        gamma = nrj / 0.938
        eps_g = eps_n / gamma

        data_out = pd.DataFrame()

        for obs in set(self.track_df.index):
            data = self.track_df.loc[obs].copy()
            if type(data) != pd.DataFrame:
                continue

            betx = self.twiss_df.loc[obs]["betx"]
            alfx = self.twiss_df.loc[obs]["alfx"]
            bety = self.twiss_df.loc[obs]["bety"]
            alfy = self.twiss_df.loc[obs]["alfy"]

            data["xn"] = data.apply(lambda x: x["x"] / np.sqrt(eps_g * betx), axis=1)

            data["pxn"] = data.apply(
                lambda x: (x["x"] * alfx / np.sqrt(betx) + x["px"] * np.sqrt(betx))
                / np.sqrt(eps_g),
                axis=1,
            )

            data["yn"] = data.apply(lambda x: x["y"] / np.sqrt(eps_g * bety), axis=1)

            data["pyn"] = data.apply(
                lambda x: (x["y"] * alfy / np.sqrt(bety) + x["py"] * np.sqrt(bety))
                / np.sqrt(eps_g),
                axis=1,
            )

            data_out = data_out.append(data)

        self.track_df = data_out

    def calculate_action(self):
        """TODO: Docstring for calculate_action.
        Returns: TODO

        """
        actionx = np.sqrt(self.track_df["xn"] ** 2 + self.track_df["pxn"] ** 2)
        actiony = np.sqrt(self.track_df["yn"] ** 2 + self.track_df["pyn"] ** 2)
        actionr = np.sqrt(actionx ** 2 + actiony ** 2)

        return {"x": actionx, "y": actiony, "r": actionr}

    def plot_orbit_excursion(
        self,
        observation_filter: Optional[Union[str, List[str]]] = None,
        trace_name: Optional[str] = None,
        save_path: Optional[str] = None,
        figure: Optional[go.Figure] = None,
        **kwargs,
    ):
        """TODO: Docstring for plot_orbit_excursion.

        Args:
            kwargs: TODO

        Kwargs:
            observation_filter (TODO): TODO
            save_path (TODO): TODO
            figure (TODO): TODO

        Returns: TODO

        """
        action = self.calculate_action()

        x_data = self.track_df["turn"]
        y_data = action["r"]

        if observation_filter:
            try:
                x_data = x_data.loc[observation_filter]
                y_data = y_data.loc[observation_filter]
            except KeyError:  # Filter not in data
                return

        if figure is None:
            figure = go.Figure()

        data = go.Scatter(x=x_data, y=y_data, mode="lines+markers", name=trace_name)

        figure.add_trace(data)

        figure.update_layout(kwargs)

        if save_path is not None:
            if not save_path.endswith(".html"):
                save_path += ".html"
            figure.write_html(save_path)

        return figure
