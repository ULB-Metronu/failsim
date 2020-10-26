"""
Module containing classes that contain and handle data.
"""


from .failsim import FailSim
from typing import Optional, List, Union, Dict
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os

import plotly.graph_objects as go


@dataclass
class Result:
    """
    Base result class containing 3 DataFrames:

    Args:
        twiss_df: DataFrame containing the twiss table.
        summ_df: DataFrame containing the summ table.
        run_version: The LHC run used, otherwise 0.
        hllhc_version: The HLLHC version used, otherwise 0.
        eps_n: The normalized emmitance.
        nrj: Particle energy in GeV.

    Attributes:
        twiss_df: DataFrame containing the twiss table.
        summ_df: DataFrame containing the summ table.
        info_df: DataFrame containing the following information:
            **run_version**: The LHC run used, otherwise 0.
            **hllhc_version**: The HLLHC version used, otherwise 0.
            **eps_n**: The normalized emmitance.
            **nrj**: Particle energy in GeV.

    Note:
        Either run_version or hllhc_version has to be specified. An AssertionError will be thrown if none or both are specified.

    """

    def __init__(
        self,
        twiss_df: pd.DataFrame,
        summ_df: pd.DataFrame,
        run_version: int = 0,
        hllhc_version: float = 0.0,
        eps_n: float = 2.5e-6,
        nrj: float = 7000,
    ):
        self.twiss_df = twiss_df
        self.summ_df = summ_df

        self.info_df = pd.DataFrame(
            dict(
                run_version=run_version,
                hllhc_version=hllhc_version,
                eps_n=eps_n,
                nrj=nrj,
            ),
            index=["info"],
        )

        self.fix_twiss_name()

    def fix_twiss_name(self):
        """
        Removes last two characters from each index name in the twiss_df DataFrame.

        This method is called in the constructor, since Mad-X currently adds :0 and :1 to the end of the twiss DataFrame index names, which does not match with other DataFrames.
        """
        self.twiss_df["name"] = self.twiss_df.apply(
            lambda x: x["name"][:-2], axis=1)

    def calculate_betabeating(self, reference: Optional[pd.DataFrame] = None):
        """
        Calculates the beta beating of Twiss data

        Args:
            reference: Allows specification of a reference Twiss table. If no reference Twiss table is specified, the first turn of the internal twiss data will be used as reference.

        Returns:
            pd.DataFrame: DataFrame containing the beta beating.

        """
        turns = set(self.twiss_df["turn"])

        if reference is None:
            reference = self.twiss_df.loc[self.twiss_df["turn"] == min(turns)]

        res = pd.DataFrame()
        for turn in turns:
            data = self.twiss_df.loc[self.twiss_df["turn"] == turn]
            temp_res = pd.DataFrame()

            temp_res["betx"] = data["betx"] / reference["betx"]
            temp_res["alfx"] = data["alfx"] / reference["alfx"]
            temp_res["mux"] = data["mux"] / reference["mux"]

            temp_res["bety"] = data["bety"] / reference["bety"]
            temp_res["alfy"] = data["alfy"] / reference["alfy"]
            temp_res["muy"] = data["muy"] / reference["muy"]

            temp_res["name"] = reference["name"]
            temp_res["s"] = reference["s"]
            temp_res["turn"] = data["turn"]

            temp_res.set_index("name", inplace=True)

            res = res.append(temp_res)

        return res


@dataclass
class TrackingResult(Result):

    """
    Dataclass containing tracking data. The class inherits from Result, as it is also convenient to keep the tracking data together with twiss and summ data.

    Attributes:
        track_df: DataFrame containing the tracking data.

    Args:
        twiss_df: DataFrame containing the twiss table.
        summ_df: DataFrame containing the summ table.
        track_df: DataFrame containing the tracking data.
        run_version: The LHC run used, otherwise 0.
        hllhc_version: The HLLHC version used, otherwise 0.
        eps_n: The normalized emmitance.
        nrj: Particle energy in GeV.

    """

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
        Result.__init__(self, twiss_df, summ_df, run_version, hllhc_version)

        self.track_df = track_df

        self.normalize_track(eps_n, nrj)

    def normalize_track(self, eps_n: float = 2.5e-6, nrj: float = 7000):
        """
        Adds 4 columns to track_df:

        - **xn**: The normalized horizontal transverse position.
        - **pxn**: The normalized horizontal transverse velocity.
        - **yn**: The normalized vertical transverse position.
        - **pyn**: The normalized vertical transverse velocity.

        Args:
            eps_n: The normalized emmitance
            nrj: Particle energy in GeV

        """
        gamma = self.info_df["nrj"] / 0.938
        eps_g = self.info_df["eps_n"] / gamma

        data_out = pd.DataFrame()

        for obs in set(self.track_df.index):
            data = self.track_df.loc[obs].copy()
            if type(data) != pd.DataFrame:
                continue

            betx = self.twiss_df.loc[obs]["betx"]
            alfx = self.twiss_df.loc[obs]["alfx"]
            bety = self.twiss_df.loc[obs]["bety"]
            alfy = self.twiss_df.loc[obs]["alfy"]

            data["xn"] = data.apply(
                lambda x: x["x"] / np.sqrt(eps_g * betx), axis=1)

            data["pxn"] = data.apply(
                lambda x: (x["x"] * alfx / np.sqrt(betx) +
                           x["px"] * np.sqrt(betx))
                / np.sqrt(eps_g),
                axis=1,
            )

            data["yn"] = data.apply(
                lambda x: x["y"] / np.sqrt(eps_g * bety), axis=1)

            data["pyn"] = data.apply(
                lambda x: (x["y"] * alfy / np.sqrt(bety) +
                           x["py"] * np.sqrt(bety))
                / np.sqrt(eps_g),
                axis=1,
            )

            data_out = data_out.append(data)

        self.track_df = data_out

    def calculate_action(self):
        """Calculates and returns the orbit excursion / action of the tracking data.

        Returns:
            Dict: Dictionary containing the following key/value pairs:

                {
                'x': Horizontal action,
                'y': Vertical action,
                'r': Radial action
                }

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
        """
        Plots orbit excursion either to a new figure, or adds the plot to an existing figure.

        Note:
            The kwargs can be used to give layout/trace specifications to plotly.
            Each key must have either "layout_" or "trace_" as suffix.
            If a key has "layout_" as suffix, the key will be sent to update_layout(),
            whereas a key with the suffix "trace_" will be sent to Scatter().

        Args:
            observation_filter: Filters tracking data indexes by each item. Can either be a list of observation points, or a single observation point.
            save_path: Path and filename of where to save the figure. If save_path is None, the plot is not saved.
            figure: The figure to add the plot to. If figure is None, a new plotly Figure object is created.

        Returns:
            go.Figure: Returns either the newly created figure if no figure was specified, or the figure the plot was added to.

        """
        layout_kwargs = {x[7:]: kwargs[x]
                         for x in kwargs if x.startswith("layout_")}
        trace_kwargs = {x[6:]: kwargs[x]
                        for x in kwargs if x.startswith("trace_")}

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

        data = go.Scatter(
            trace_kwargs, x=x_data, y=y_data, mode="lines+markers", name=trace_name
        )

        figure.add_trace(data)

        figure.update_layout(
            layout_kwargs,
            xaxis_title=r"$\text{Time} [\text{LHC Turn}]$",
            yaxis_title=r"$\text{Radial orbit excursion} [\sigma_r]$",
        )

        if save_path is not None:
            if not save_path.endswith(".html"):
                save_path += ".html"
            figure.write_html(save_path)

        return figure

    def save_data(self, path: str, suffix: str = ""):
        """
        Saves the TrackingResult data in 4 disctinct files:

        - **info.parquet**: Contains miscellaneous table
        - **summ.parquet**: Contains the summ table
        - **track.parquet**: Contains the tracking data
        - **twiss.parquet**: Contains the twiss table

        Args:
            path: The directory in which to save the data. Can be either absolute or relative to cwd.
            suffix: Allows specification of a suffix. The suffix will be prepended to each of the 4 saved files.

        """

        if not path.startswith("/"):
            path = FailSim.path_to_cwd(path)

        # Save twiss
        twiss_name = os.path.join(path, suffix + "twiss.parquet")
        self.twiss_df.to_parquet(twiss_name)

        # Save track
        track_name = os.path.join(path, suffix + "track.parquet")
        self.track_df.to_parquet(track_name)

        # Save summ
        summ_name = os.path.join(path, suffix + "summ.parquet")
        self.summ_df.to_parquet(summ_name)

        # Save extra info
        info_name = os.path.join(path, suffix + "info.parquet")
        self.info_df.to_parquet(info_name)

    @classmethod
    def load_data(cls, path: str, suffix: str = ""):
        """Classmethod that loads data from the directory specified and returns a TrackingResult object.

        Note:
            The path specified by path must contain the following files:

            - info.parquet
            - summ.parquet
            - track.parquet
            - twiss.parquet

        Args:
            path: The directory containing data. Can be either absolute or relative to cwd.
            suffix: Allows specification of a suffix. The method will look for the same for files, only with the specified suffix prepended.

        Returns:
            TrackingResult: A TrackingResult instance containing the loaded data.

        """
        # Load twiss
        twiss_df = pd.read_parquet(
            os.path.join(path, suffix + "twiss.parquet"))

        # Load track
        track_df = pd.read_parquet(
            os.path.join(path, suffix + "track.parquet"))

        # Load summ
        summ_df = pd.read_parquet(os.path.join(path, suffix + "summ.parquet"))

        # Load info
        info_df = pd.read_parquet(os.path.join(path, suffix + "info.parquet"))

        # Create instance
        inst = TrackingResult(
            twiss_df=twiss_df,
            summ_df=summ_df,
            track_df=track_df,
            run_version=info_df["run_version"],
            hllhc_version=info_df["hllhc_version"],
            eps_n=info_df["eps_n"],
            nrj=info_df["nrj"],
        )

        return inst


@dataclass
class TwissResult(Result):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
