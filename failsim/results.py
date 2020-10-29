"""
Module containing classes that contain and handle data.
"""


from .failsim import FailSim
from typing import Optional, List, Union, Dict, Tuple, Callable
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

    def _plot(
        self,
        x_data: pd.Series,
        y_data: pd.Series,
        trace_name: Optional[str] = None,
        save_path: Optional[str] = None,
        figure: Optional[go.Figure] = None,
        center_elem: str = None,
        width: float = None,
        **kwargs,
    ):
        """ Private method that provides basic plotting utility. """
        layout_kwargs = {x[7:]: kwargs[x] for x in kwargs if x.startswith("layout_")}
        trace_kwargs = {x[6:]: kwargs[x] for x in kwargs if x.startswith("trace_")}

        if figure is None:
            figure = go.Figure()

        data = go.Scatter(
            trace_kwargs, x=x_data, y=y_data, mode="lines", name=trace_name
        )

        figure.add_trace(data)

        figure.update_layout(layout_kwargs)

        if center_elem is not None:
            assert width is not None, "width must be specified when using center_elem"
            elem_s = self.twiss_df.loc[center_elem]["s"].iloc[0]
            figure.update_layout(
                xaxis_range=(
                    elem_s - width / 2.0,
                    elem_s + width / 2.0,
                )
            )

        if save_path is not None:
            if not save_path.endswith(".html"):
                save_path += ".html"
            if not save_path.startswith("/"):
                save_path = FailSim.path_to_output(save_path)
            figure.write_html(save_path)

        return figure

    def fix_twiss_name(self):
        """
        Removes last two characters from each index name in the twiss_df DataFrame.

        This method is called in the constructor, since Mad-X currently adds :0 and :1 to the end of the twiss DataFrame index names, which does not match with other DataFrames.
        """
        self.twiss_df["name"] = self.twiss_df.apply(lambda x: x["name"][:-2], axis=1)

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

    def get_phase_advance(self, reference: str, elements: Union[str, List[str]]):
        """Get the phase advance between reference element and other elements.

        Args:
            reference: TODO
            elements: TODO

        Returns:
            Union[dict[str, float], List[dict[str, float]]]: Either a dictionary containing the keys 'x' and 'y', each mapping to the horizontal and vertical phase advance respectively, or a dictionary containing such dictionaries, with each element as key.

        """
        ref_mux = self.twiss_df.loc[reference]["mux"]
        ref_muy = self.twiss_df.loc[reference]["muy"]

        el_mux = self.twiss_df.loc[elements]["mux"]
        el_muy = self.twiss_df.loc[elements]["muy"]

        diff_mux = ref_mux - el_mux
        diff_muy = ref_muy - el_muy

        return {
            "x": diff_mux,
            "y": diff_muy,
        }

    def plot_beta_beating(
        self,
        axis: str = "x",
        element: Optional[str] = None,
        observation_filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
        trace_name: Optional[str] = None,
        save_path: Optional[str] = None,
        figure: Optional[go.Figure] = None,
        center_elem: str = None,
        width: float = None,
        **kwargs,
    ):
        """
        Plots beta beating either to a new figure, or adds the plot to an existing figure.

        Note:
            The kwargs can be used to give layout/trace specifications to plotly.
            Each key must have either "layout_" or "trace_" as suffix.
            If a key has "layout_" as suffix, the key will be sent to update_layout(),
            whereas a key with the suffix "trace_" will be sent to Scatter().

        Args:
            axis: Can either be 'x' or 'y' to specify which axis the betabeating should be calculated from.
            element: If element is not None, the method will plot the beating of a single element.
            observation_filter: Filters tracking data indexes by each item. Can either be a list of observation points, or a single observation point.
            save_path: Path and filename of where to save the figure. If save_path is None, the plot is not saved.
            figure: The figure to add the plot to. If figure is None, a new plotly Figure object is created.
            center_elem: Element on which the plot will be centered. If no element is specified, the method will not center any specific element. If center_elem is specified, width must not be None.
            width: The difference between the leftmost and rightmost points on the plot. Is meant to be used in conjuction with center_elem. If no center_elem is specified, width does nothing.

        Returns:
            go.Figure: Returns either the newly created figure if no figure was specified, or the figure the plot was added to.

        """
        twiss = self.twiss_df.copy()
        betabeating = self.calculate_betabeating()

        if observation_filter is not None:
            twiss = twiss.loc[observation_filter(twiss)]
            betabeating = betabeating.loc[observation_filter(betabeating)]

        if element is None:
            x_data = twiss["s"]
            y_data = betabeating[f"bet{axis}"]
        else:
            x_data = betabeating.loc[element]["turn"]
            y_data = betabeating.loc[element][f"bet{axis}"]

        return self._plot(
            x_data=x_data,
            y_data=y_data,
            observation_filter=observation_filter,
            trace_name=trace_name,
            save_path=save_path,
            figure=figure,
            center_elem=center_elem,
            width=width,
            **kwargs,
        )

    def plot_beta_function(
        self,
        axis: str = "x",
        observation_filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
        trace_name: Optional[str] = None,
        save_path: Optional[str] = None,
        figure: Optional[go.Figure] = None,
        center_elem: str = None,
        width: float = None,
        **kwargs,
    ):
        """
        Plots the beta function either to a new figure, or adds the plot to an existing figure.

        Note:
            The kwargs can be used to give layout/trace specifications to plotly.
            Each key must have either "layout_" or "trace_" as suffix.
            If a key has "layout_" as suffix, the key will be sent to update_layout(),
            whereas a key with the suffix "trace_" will be sent to Scatter().

        Args:
            axis: Can either be 'x' or 'y' to specify which axis to plot.
            observation_filter: Filters tracking data indexes by each item. Can either be a list of observation points, or a single observation point.
            save_path: Path and filename of where to save the figure. If save_path is None, the plot is not saved.
            figure: The figure to add the plot to. If figure is None, a new plotly Figure object is created.
            center_elem: Element on which the plot will be centered. If no element is specified, the method will not center any specific element. If center_elem is specified, width must not be None.
            width: The difference between the leftmost and rightmost points on the plot. Is meant to be used in conjuction with center_elem. If no center_elem is specified, width does nothing.

        Returns:
            go.Figure: Returns either the newly created figure if no figure was specified, or the figure the plot was added to.

        """
        twiss = self.twiss_df.copy()

        if observation_filter is not None:
            twiss = twiss.loc[observation_filter(twiss)]

        x_data = twiss["s"]
        y_data = twiss[f"bet{axis}"]

        return self._plot(
            x_data=x_data,
            y_data=y_data,
            observation_filter=observation_filter,
            trace_name=trace_name,
            save_path=save_path,
            figure=figure,
            center_elem=center_elem,
            width=width,
            **kwargs,
        )

    def plot_effective_gap(
        self,
        element: str,
        aperture: float,
        axis: str = "x",
        observation_filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
        trace_name: Optional[str] = None,
        save_path: Optional[str] = None,
        figure: Optional[go.Figure] = None,
        center_elem: str = None,
        width: float = None,
        **kwargs,
    ):
        """
        Plots the effective gap of the selected element.

        Args:
            element: The element to plot.
            aperture: The aperture of the element in metres.
            axis: Can either be 'x' or 'y' to specify which axis to plot.
            observation_filter: Filters tracking data indexes by each item. Can either be a list of observation points, or a single observation point.
            save_path: Path and filename of where to save the figure. If save_path is None, the plot is not saved.
            figure: The figure to add the plot to. If figure is None, a new plotly Figure object is created.
            center_elem: Element on which the plot will be centered. If no element is specified, the method will not center any specific element. If center_elem is specified, width must not be None.
            width: The difference between the leftmost and rightmost points on the plot. Is meant to be used in conjuction with center_elem. If no center_elem is specified, width does nothing.

        Returns:
            go.Figure: Returns either the newly created figure if no figure was specified, or the figure the plot was added to.

        """
        twiss = self.twiss_df.copy()

        if observation_filter is not None:
            twiss = twiss.loc[observation_filter(twiss)]

        gamma = self.info_df["nrj"]["info"] / 0.938
        eps_g = self.info_df["eps_n"]["info"] / gamma

        beta_elem = twiss.loc[element]
        beta_ref = beta_elem.loc[beta_elem["turn"] == 1]

        sig_elem = np.sqrt(eps_g * beta_elem[f"bet{axis}"])
        sig_ref = np.sqrt(eps_g * beta_ref[f"bet{axis}"])

        effective_gap = aperture * sig_ref / sig_elem

        x_data = beta_elem["turn"]
        y_data = effective_gap

        return self._plot(
            x_data=x_data,
            y_data=y_data,
            observation_filter=observation_filter,
            trace_name=trace_name,
            save_path=save_path,
            figure=figure,
            center_elem=center_elem,
            width=width,
            **kwargs,
        )

    @classmethod
    def create_cartouche(
        cls,
        fig_range: Tuple[float, float] = None,
        height: float = 0.08,
        y_start: float = 1.02,
    ):
        """TODO: Docstring for create_cartouche.

        Args:
            fig_range: The longitudinal range in which to draw the elements of the sequence.
            height: The height of the cartouche part of the plot.
            y_start: The height at which the cartouche starts to be drawn.

        Returns:
            go.Figure: Figure with sequence objects drawn above plot.

        """
        twiss_thick = pd.read_parquet(FailSim.path_to_output("twiss_pre_thin.parquet"))

        twiss_thick = twiss_thick.loc[
            ~twiss_thick["keyword"].isin(
                ["drift", "marker", "placeholder", "monitor", "instrument"]
            )
        ]

        if fig_range is not None:
            twiss_thick = twiss_thick.loc[
                (twiss_thick["s"] > fig_range[0]) & (twiss_thick["s"] < fig_range[1])
            ]

        fig = go.Figure()

        colors = dict(
            quadrupole="red",
            sextupole="gray",
            octupole="gray",
            multipole="gray",
            hkicker="gray",
            vkicker="gray",
            tkicker="gray",
            solenoid="orange",
            rfcavity="gray",
            rcollimator="black",
            rbend="lightblue",
            sbend="blue",
        )

        middle = y_start + height / 2

        shapes = [
            go.layout.Shape(
                type="line",
                yref="paper",
                y0=middle,
                y1=middle,
                x0=0,
                x1=max(twiss_thick["s"]),
                line_width=0.5,
            )
        ]
        for _, row in twiss_thick.iterrows():
            if row["keyword"] in ["rbend", "sbend", "rcollimator"]:
                y0 = y_start
                y1 = y_start + height
            elif row["keyword"] in ["quadrupole"]:
                y0 = middle - height / 4 + row["polarity"] * height / 4
                y1 = middle + height / 4 + row["polarity"] * height / 4
            else:
                y0 = middle - height / 4
                y1 = middle + height / 4

            shapes.append(
                go.layout.Shape(
                    type="rect",
                    yref="paper",
                    y0=y0,
                    y1=y1,
                    x0=row["s"] - row["l"],
                    x1=row["s"],
                    line_width=0,
                    fillcolor=colors[row["keyword"]],
                )
            )

        fig.update_layout(shapes=shapes)

        return fig


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

    def calculate_action(self, track: Optional[pd.DataFrame] = None):
        """Calculates and returns the orbit excursion / action of the tracking data.

        Args:
            track: Set track dataframe to calculate action of. If None is specified, the internal tracking data will be used.

        Returns:
            Dict: Dictionary containing the following key/value pairs:

                {
                'x': Horizontal action,
                'y': Vertical action,
                'r': Radial action
                }

        """
        if track is None:
            track = self.track_df

        actionx = np.sqrt(track["xn"] ** 2 + track["pxn"] ** 2)
        actiony = np.sqrt(track["yn"] ** 2 + track["pyn"] ** 2)
        actionr = np.sqrt(actionx ** 2 + actiony ** 2)

        return {"x": actionx, "y": actiony, "r": actionr}

    def plot_orbit_excursion(
        self,
        observation_filter: Optional[Union[str, List[str]]] = None,
        trace_name: Optional[str] = None,
        save_path: Optional[str] = None,
        figure: Optional[go.Figure] = None,
        center_elem: str = None,
        width: float = None,
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
            center_elem: Element on which the plot will be centered. If no element is specified, the method will not center any specific element. If center_elem is specified, width must not be None.
            width: The difference between the leftmost and rightmost points on the plot. Is meant to be used in conjuction with center_elem. If no center_elem is specified, width does nothing.

        Returns:
            go.Figure: Returns either the newly created figure if no figure was specified, or the figure the plot was added to.

        """
        track = self.track_df.copy()

        if observation_filter is not None:
            track = track.loc[observation_filter(track)]

        x_data = track["turn"]
        y_data = self.calculate_action(track)["r"]

        return self._plot(
            x_data=x_data,
            y_data=y_data,
            trace_name=trace_name,
            save_path=save_path,
            figure=figure,
            center_elem=center_elem,
            width=width,
            **kwargs,
        )

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
        twiss_df = pd.read_parquet(os.path.join(path, suffix + "twiss.parquet"))

        # Load track
        track_df = pd.read_parquet(os.path.join(path, suffix + "track.parquet"))

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
