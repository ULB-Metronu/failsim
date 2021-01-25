"""
Module containing classes that contain and handle data.
"""


from __future__ import annotations

from ._artist import _Artist
from .failsim import FailSim

from typing import Optional, List, Union, Dict, Tuple, Callable, Type, TYPE_CHECKING
from dataclasses import dataclass

import pandas as pd
import numpy as np
import plotly.graph_objects as go

import os
import re

# Type checking imports
if TYPE_CHECKING:
    from .lhc_sequence import CollimatorHandler


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

        if twiss_df["name"].iloc[0].endswith(":1"):
            self.fix_twiss_name()

    def append(self, other: Result, sort: bool = False):
        """TODO: Docstring for append.

        Args:
            other (TODO): TODO

        Kwargs:
            sort (TODO): TODO

        Returns: TODO

        """
        self.twiss_df = self.twiss_df.append(other.twiss_df)
        if sort:
            self.twiss_df = self.twiss_df.sort_values(["turn", "s"])

    def save_data(self, path: str, suffix: str = ""):
        """
        Saves the Result data in 3 disctinct files:

        - **info.parquet**: Contains miscellaneous table
        - **summ.parquet**: Contains the summ table
        - **twiss.parquet**: Contains the twiss table

        Args:
            path: The directory in which to save the data. Can be either absolute or relative to cwd.
            suffix: Allows specification of a suffix. The suffix will be prepended to each of the 4 saved files.

        """
        if not path.startswith("/"):
            path = FailSim.path_to_cwd(path)

        os.makedirs(path, exist_ok=True)

        # Save twiss
        twiss_name = os.path.join(path, suffix + "twiss.parquet")
        self.twiss_df.to_parquet(twiss_name)

        # Save summ
        summ_name = os.path.join(path, suffix + "summ.parquet")
        self.summ_df.to_parquet(summ_name)

        # Save extra info
        info_name = os.path.join(path, suffix + "info.parquet")
        self.info_df.to_parquet(info_name)

    @classmethod
    def load_data(cls, path: str, suffix: str = ""):
        """Classmethod that loads data from the directory specified and returns a Result object.

        Note:
            The path specified by path must contain the following files:

            - info.parquet
            - summ.parquet
            - twiss.parquet

        Args:
            path: The directory containing data. Can be either absolute or relative to cwd.
            suffix: Allows specification of a suffix. The method will look for the same for files, only with the specified suffix prepended.

        Returns:
            Result: A Result instance containing the loaded data.

        """
        # Load twiss
        twiss_df = pd.read_parquet(os.path.join(path, suffix + "twiss.parquet"))

        # Load summ
        summ_df = pd.read_parquet(os.path.join(path, suffix + "summ.parquet"))

        # Load info
        info_df = pd.read_parquet(os.path.join(path, suffix + "info.parquet"))

        # Create instance
        inst = cls(
            twiss_df=twiss_df,
            summ_df=summ_df,
            run_version=info_df["run_version"],
            hllhc_version=info_df["hllhc_version"],
            eps_n=info_df["eps_n"],
            nrj=info_df["nrj"],
        )

        return inst

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
            reference: Name of element to reference as 0 mu.
            elements: Can either be the name of a single element or a list of elements. These are the elements of which the relative phase advance will be calculated.

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
        loss_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__(
            twiss_df, summ_df, run_version, hllhc_version, eps_n=eps_n, nrj=nrj
        )

        self.track_df = track_df
        self.loss_df = loss_df

        self.plot = _TrackArtist(self)

    def append(self, other: TrackingResult, sort: bool = False):
        """TODO: Docstring for append.

        Args:
            other (TODO): TODO

        Kwargs:
            sort (TODO): TODO

        Returns: TODO

        """
        self.track_df = self.track_df.append(other.track_df)
        if not (self.loss_df is None or other.loss_df is None):
            self.loss_df.append(other.loss_df)
            if sort:
                self.loss_df = self.loss_df.sort_values(["turn", "s"])
        if sort:
            self.track_df = self.track_df.sort_values(["turn", "s"])

    def normalize_track(self):
        """
        Creates and returns new DataFrame based on track_df with four columns added:

        - **xn**: The normalized horizontal transverse position.
        - **pxn**: The normalized horizontal transverse velocity.
        - **yn**: The normalized vertical transverse position.
        - **pyn**: The normalized vertical transverse velocity.

        Args:
            eps_n: The normalized emmitance
            nrj: Particle energy in GeV

        Returns:
            pd.DataFrame: Tracking DataFrame with normalized columns added.

        """
        twiss_df = self.twiss_df
        track_df = self.track_df

        gamma = self.info_df["nrj"]["info"] / 0.938
        eps_g = self.info_df["eps_n"]["info"] / gamma

        data_out = pd.DataFrame()

        for obs in set(track_df.index):
            data = track_df.loc[obs].copy()
            if type(data) != pd.DataFrame:
                continue

            betx = twiss_df.loc[obs]["betx"]
            alfx = twiss_df.loc[obs]["alfx"]
            bety = twiss_df.loc[obs]["bety"]
            alfy = twiss_df.loc[obs]["alfy"]

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

        os.makedirs(path, exist_ok=True)

        super().save_data(path, suffix)

        # Save track
        track_name = os.path.join(path, suffix + "track.parquet")
        self.track_df.to_parquet(track_name)

        # Save loss if it is not None
        if self.loss_df is not None:
            loss_name = os.path.join(path, suffix + "loss.parquet")
            self.loss_df.to_parquet(loss_name)

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

        # Load summ
        summ_df = pd.read_parquet(os.path.join(path, suffix + "summ.parquet"))

        # Load info
        info_df = pd.read_parquet(os.path.join(path, suffix + "info.parquet"))

        # Load track
        track_df = pd.read_parquet(os.path.join(path, suffix + "track.parquet"))

        # Load loss
        loss_df = None
        if os.path.exists(os.path.join(path, suffix + "loss.parquet")):
            loss_df = pd.read_parquet(os.path.join(path, suffix + "loss.parquet"))

        # Create instance
        inst = TrackingResult(
            twiss_df=twiss_df,
            summ_df=summ_df,
            track_df=track_df,
            loss_df=loss_df,
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

        self.plot = _TwissArtist(self)


class _TwissArtist(_Artist):

    """Docstring for _TwissArtist. """

    element_colors = dict(
        quadrupole="red",
        sextupole="green",
        octupole="orange",
        multipole="gray",
        hkicker="gray",
        vkicker="gray",
        tkicker="gray",
        solenoid="brown",
        rfcavity="purple",
        rcollimator="black",
        rbend="yellow",
        sbend="blue",
        instrument="coral",
    )

    beam_colors = dict(
        orbitx="#a100ff",
        orbity="#ffa100",
    )

    aper_style = dict(
        mode="lines",
        marker_color="black",
        line_width=1,
        fill="toself",
        showlegend=False,
    )

    def __init__(self, parent: Result):
        _Artist.__init__(self)

        self._parent = parent

        self._center_elem = None
        self._width = None
        self._filter = None

    def centered_element(self, element: str, width: float = 1000):
        """TODO: Docstring for centered_element.

        Args:

        Returns: TODO

        """
        self._center_elem = element
        self._width = width

    def observation_filter(self, filter: Callable[[pd.DataFrame], pd.DataFrame]):
        """TODO: Docstring for observation_filter.

        Args:

        Returns: TODO

        """
        self._filter = filter

    def _apply_observation_filter(self, data: pd.DataFrame):
        """TODO: Docstring for _apply_observation_filter.

        Args:

        Returns: TODO

        """
        if self._filter is None:
            return data
        return data.loc[self._filter(data)]

    def get_centered_range(self):
        """TODO: Docstring for get_centered_range.

        Args:

        Returns: TODO

        """
        center_s = self._parent.twiss_df.loc[self._center_elem]["s"]
        try:
            center_s = center_s.iloc[0]
        except AttributeError:
            pass
        return center_s - self._width / 2, center_s + self._width / 2

    def _crop_to_centered(self, data: pd.DataFrame):
        """TODO: Docstring for _crop_to_centered.

        Args:

        Returns: TODO

        """
        if self._center_elem is not None:
            center_range = self.get_centered_range()
            return data[(data["s"] > center_range[0]) & (data["s"] < center_range[1])]
        return data

    def twiss_column(
        self,
        columns: Union[str, List[str]],
        xaxis_column: str = "s",
        animate_column: Optional[str] = None,
        crop_data: bool = False,
        reference: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        """TODO: Docstring for twiss_column.

        Args:
        function (TODO): TODO

        Returns: TODO

        """
        if type(columns) is str:
            columns = [columns]

        twiss = self._apply_observation_filter(self._parent.twiss_df)

        # Remove negative turns; this is a result of the time
        # dependencies being delayed by a single turn in the
        # twiss command, in order to ensure a single clean turn
        twiss = twiss.loc[twiss["turn"] >= 0]

        if self._center_elem is not None:
            center_range = self.get_centered_range()
            col, row = self._plot_pointer
            self.plot_layout(
                xaxis={
                    "range": (
                        center_range[0] * self._subplots[col][row]["factor"]["x"],
                        center_range[1] * self._subplots[col][row]["factor"]["x"],
                    )
                },
            )
            if crop_data:
                twiss = twiss[
                    (twiss["s"] > center_range[0]) & (twiss["s"] < center_range[1])
                ]

        for column in columns:
            style = dict(name=column)
            style.update(kwargs)

            if animate_column is not None:
                col, row = self._plot_pointer
                frame_trace = 0
                for _row in range(self._rows):
                    for _col in range(self._cols):
                        if _row * self._cols + _col > col * self._cols + col:
                            continue
                        frame_trace += len(self._subplots[_col][_row]["data"])
                for idx, (x, split) in enumerate(
                    dict(tuple(twiss.groupby(animate_column))).items()
                ):
                    x_data = split[xaxis_column]
                    y_data = split[column]

                    if reference is not None:
                        # y_data = (y_data / reference[column]).dropna()
                        temp = 100 * y_data / reference[column]
                        y_data = temp.loc[y_data.index]

                    if idx == 0:
                        self.add_data(
                            **style,
                            x=x_data,
                            y=y_data,
                        )

                    self.add_frame(
                        **style,
                        x=x_data,
                        y=y_data,
                        frame_name=x,
                        frame_trace=frame_trace,
                    )
            else:
                x_data = twiss[xaxis_column]
                y_data = twiss[column]

                if reference is not None:
                    y_data = 100 * (y_data / reference[column]).dropna()

                self.add_data(
                    **style,
                    x=x_data,
                    y=y_data,
                )

    def cartouche(self, twiss_path: Optional[Tuple[str, str]] = None):
        """TODO: Docstring for cartouche.

        Args:

        Returns: TODO

        """
        for ss in [f"lhcb{i}" for i in range(1, 3)]:
            # Read twiss data
            if twiss_path is None:
                twiss = pd.read_parquet(
                    FailSim.path_to_output(f"twiss_pre_thin_{ss}.parquet")
                )
            else:
                path = twiss_path[int(ss[-1]) - 1]
                if not path.startswith("/"):
                    path = FailSim.path_to_cwd(path)
                twiss = pd.read_parquet(path)

            # Remove elements that shouldn't be shown in cartouche
            twiss = twiss[
                ~twiss["keyword"].isin(
                    ["drift", "marker", "placeholder", "monitor", "instrument"]
                )
            ]

            # Interpolate mech_sep between each rbend element
            twiss["beam_sep"] = twiss.apply(
                lambda x: x["mech_sep"] if x["keyword"] == "rbend" else float("nan"),
                axis=1,
            )
            beam_sep = twiss.set_index("s")["beam_sep"].interpolate(
                method="index",
                limit_direction="both",
            )

            # Filter if center_elem is defined
            if self._center_elem is not None:
                center_range = self.get_centered_range()
                twiss = twiss[
                    (twiss["s"] > center_range[0]) & (twiss["s"] < center_range[1])
                ]
                beam_sep = beam_sep[
                    (beam_sep.index > center_range[0])
                    & (beam_sep.index < center_range[1])
                ]

            # Normalize beam separation
            if not max(abs(beam_sep)) == 0:
                beam_sep = beam_sep / max(abs(beam_sep))
            beam_sep = beam_sep.set_axis(twiss.index)

            # Draw beam line
            self.add_data(
                x=twiss["s"],
                y=beam_sep,
                showlegend=False,
                marker_color="gray",
                hoverinfo="skip",
                to_bottom=True,
                mode="lines",
            )

            for _, row in twiss.iterrows():
                if (
                    not ".b1:" in row["name"]
                    and not ".b2:" in row["name"]
                    and ss == "lhcb2"
                ):
                    continue

                x0 = row["s"] - (row["l"] if row["l"] != 0 else 0.1)
                x1 = row["s"]
                dy = 0.9
                y0 = beam_sep.loc[row["name"][:-2]] - dy / 2

                style = self.aper_style.copy()
                style.update(
                    fillcolor=self.element_colors[row["keyword"]], name=row["name"]
                )

                # Draw rbend and sbend
                if row["keyword"] in ["rbend", "sbend"]:
                    self.add_data(
                        x=[x0, x1, x1, x0, x0],
                        y=[y0, y0, y0 + dy, y0 + dy, y0],
                        **style,
                    )
                # Draw collimators
                elif row["keyword"] in ["rcollimator"]:
                    self.add_data(
                        x=[x0, x1, x1, x0, x0],
                        y=[y0, y0, y0 + dy / 3, y0 + dy / 3, y0],
                        **style,
                    )
                    self.add_data(
                        x=[x0, x1, x1, x0, x0],
                        y=[
                            y0 + dy * 2 / 3,
                            y0 + dy * 2 / 3,
                            y0 + dy,
                            y0 + dy,
                            y0 + dy * 2 / 3,
                        ],
                        **style,
                    )
                # Draw quadrupole
                elif row["keyword"] in ["quadrupole"]:
                    y0_pol = y0 + dy / 4 + row["polarity"] * dy / 4
                    y1_pol = y0 + dy * 3 / 4 + row["polarity"] * dy / 4
                    self.add_data(
                        x=[x0, x1, x1, x0, x0],
                        y=[y0_pol, y0_pol, y1_pol, y1_pol, y0_pol],
                        **style,
                    )
                # Draw everything else
                else:
                    self.add_data(
                        x=[x0, x1, x1, x0, x0],
                        y=[
                            y0 + dy / 4,
                            y0 + dy / 4,
                            y0 + dy * 3 / 4,
                            y0 + dy * 3 / 4,
                            y0 + dy / 4,
                        ],
                        **style,
                    )

    def aperture(
        self,
        axis: str,
        twiss_path: Optional[str] = None,
        **kwargs,
    ):
        """TODO: Docstring for aperture.

        Args:

        Returns: TODO

        """
        if twiss_path is None:
            twiss_thick = pd.read_parquet(
                FailSim.path_to_output("twiss_pre_thin_lhcb1.parquet")
            )
        else:
            if not twiss_path.startswith("/"):
                twiss_path = FailSim.path_to_cwd(twiss_path)
            twiss_thick = pd.read_parquet(twiss_path)

        # Filter if center_elem is defined
        if self._center_elem is not None:
            center_range = self.get_centered_range()
            twiss_thick = twiss_thick[
                (twiss_thick["s"] > center_range[0])
                & (twiss_thick["s"] < center_range[1])
            ]

        twiss_thick = twiss_thick.loc[
            ~twiss_thick["keyword"].isin(
                ["drift", "marker", "placeholder", "monitor", "instrument"]
            )
        ]

        for _, row in twiss_thick.iterrows():
            if "aper_1" not in row.index:
                continue

            x0 = row["s"] - (row["l"] if row["l"] != 0 else 0.1)
            x1 = row["s"]
            y0 = abs(row[f"aper_{1 if axis == 'x' else 2}"])
            y1 = 1

            # Move elements with 0 mm or infinite aperture to 200 mm
            if y0 == 0 or y0 == float("inf"):
                y0 = 200e-3

            style = self.aper_style.copy()
            style.update(
                fillcolor=self.element_colors[row["keyword"]],
                name=f"{row['name']}: {y0} mm",
                mode="lines",
            )

            for pol in [-1, 1]:
                self.add_data(
                    x=[x0, x1, x1, x0, x0],
                    y=[pol * y0, pol * y0, pol * y1, pol * y1, pol * y0],
                    **style,
                )
            style.update(
                line_width=0,
                opacity=0.1,
                mode="lines",
            )
            self.add_data(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, -y0, -y0, y0],
                **style,
            )

    def orbit(
        self,
        axis: str,
        beam_magnitudes: List[float] = [1, 4.7, 6.7],
        animate_column: Optional[str] = None,
        crop_data: bool = False,
        **kwargs,
    ):
        """TODO: Docstring for orbit.

        Args:

        Returns: TODO

        """
        twiss = self._parent.twiss_df.copy()
        twiss = self._apply_observation_filter(twiss)

        # Remove negative turns; this is a result of the time
        # dependencies being delayed by a single turn in the
        # twiss command, in order to ensure a single clean turn
        twiss = twiss.loc[twiss["turn"] >= 0]

        if self._center_elem is not None:
            center_range = self.get_centered_range()
            col, row = self._plot_pointer
            self.plot_layout(
                xaxis={
                    "range": (
                        center_range[0] * self._subplots[col][row]["factor"]["x"],
                        center_range[1] * self._subplots[col][row]["factor"]["x"],
                    )
                },
            )
            if crop_data:
                twiss = twiss[
                    (twiss["s"] > center_range[0]) & (twiss["s"] < center_range[1])
                ]

        gamma = self._parent.info_df["nrj"]["info"] / 0.938
        eps_g = self._parent.info_df["eps_n"]["info"] / gamma
        dpp = 1e-4

        twiss["envelope"] = np.sqrt(
            eps_g * twiss[f"bet{axis}"] + dpp ** 2 * twiss[f"d{axis}"] ** 2
        )

        # Get color, and convert from hex to string formatted as "r,g,b"
        beam_color = ",".join(
            [
                str(int(self.beam_colors[f"orbit{axis}"][i : i + 2], 16))
                for i in range(1, 6, 2)
            ]
        )
        beam_magnitudes += [-x for x in beam_magnitudes if -x not in beam_magnitudes]
        beam_magnitudes.sort()
        for idx, mag in enumerate(beam_magnitudes):
            alpha = 0.75 * (1 - 2 * abs(idx / len(beam_magnitudes) - 0.5))
            if animate_column is None:
                self.add_data(
                    **kwargs,
                    x=twiss["s"],
                    y=mag * twiss["envelope"] + twiss[axis],
                    line_width=0,
                    fill=None if idx == 0 else "tonexty",
                    fillcolor=f"rgba({beam_color},{alpha})",
                    marker_color=f"rgba({beam_color},{alpha})",
                    name=f"Orbit {axis}: {abs(mag)} sigma",
                    hoverinfo="skip" if mag < 0 else None,
                    showlegend=True
                    if mag == min(abs(np.array(beam_magnitudes)))
                    else False,
                )
            else:
                frame_trace = 0
                for _col in range(self._cols):
                    for _row in range(self._rows):
                        frame_trace += len(self._subplots[_col][_row]["data"])

                for idx2, (col_val, data) in enumerate(twiss.groupby(animate_column)):
                    style = dict(
                        line_width=0,
                        fill=None if idx == 0 else "tonexty",
                        fillcolor=f"rgba({beam_color},{alpha})",
                        marker_color=f"rgba({beam_color},{alpha})",
                        name=f"Orbit {axis}: {abs(mag)} sigma",
                        hoverinfo="skip" if mag < 0 else None,
                        showlegend=True
                        if mag == min(abs(np.array(beam_magnitudes)))
                        else False,
                    )
                    if idx2 == 0:
                        self.add_data(
                            **kwargs,
                            x=data["s"],
                            y=mag * data["envelope"] + data[axis],
                            **style,
                        )
                    self.add_frame(
                        **kwargs,
                        x=data["s"],
                        y=mag * data["envelope"] + data[axis],
                        frame_name=col_val,
                        frame_trace=frame_trace,
                        **style,
                    )

    def twiss_beating(
        self,
        column: str,
        reference: Optional[pd.DataFrame] = None,
        elements: Optional[List[str]] = None,
        start_col: str = "#000000",
        end_col: str = "#ff0000",
        crop_data: bool = False,
        **kwargs,
    ):
        """TODO: Docstring for twiss_beating.

        Args:

        Returns: TODO

        """
        twiss = self._parent.twiss_df.copy()

        if self._center_elem is not None:
            center_range = self.get_centered_range()
            col, row = self._plot_pointer
            self.plot_layout(
                xaxis={
                    "range": (
                        center_range[0] * self._subplots[col][row]["factor"]["x"],
                        center_range[1] * self._subplots[col][row]["factor"]["x"],
                    )
                },
            )
            if crop_data:
                twiss = twiss[
                    (twiss["s"] > center_range[0]) & (twiss["s"] < center_range[1])
                ]

        # Calculate beating
        if reference is None:
            reference = twiss[twiss["turn"] == min(twiss["turn"])]

        # Remove negative turns; this is a result of the time
        # dependencies being delayed by a single turn in the
        # twiss command, in order to ensure a single clean turn
        twiss = twiss.loc[twiss["turn"] >= 0]

        if elements is None:
            for idx, turn in enumerate(set(twiss["turn"])):
                data = twiss[twiss["turn"] == turn]
                res = 100 * data[column] / reference[column]

                col = _Artist.lerp_hex_color(
                    start_col, end_col, idx / len(set(twiss["turn"]))
                )

                self.add_data(
                    **kwargs,
                    x=data["s"],
                    y=res,
                    name=f"Turn {turn}",
                    marker_color=col,
                )
        else:
            for element in elements:
                data = twiss.loc[element]
                res = 100 * data[column] / reference.loc[element][column]

                self.add_data(
                    **kwargs,
                    x=data["turn"],
                    y=res,
                    name=element,
                )

    def effective_half_gap(
        self,
        elements: List[str],
        axis: str,
        twiss_path: str = "output/twiss_pre_thin_lhcb1.parquet",
        use_excursion: bool = True,
        suffix: Optional[str] = None,
        parallel: bool = False,
        trace_kwargs: Optional[List[dict]] = None,
        only_worst: bool = False,
        collimator_handler: Optional[CollimatorHandler] = None,
        **kwargs,
    ):
        """TODO: Docstring for effective_half_gap.

        Args:
        function (TODO): TODO

        Returns: TODO

        """
        if not twiss_path.startswith("/"):
            twiss_path = FailSim.path_to_cwd(twiss_path)

        twiss_thick = pd.read_parquet(twiss_path)
        twiss_df = self._parent.twiss_df.copy()

        # Remove negative turns; this is a result of the time
        # dependencies being delayed by a single turn in the
        # twiss command, in order to ensure a single clean turn
        twiss = twiss.loc[twiss["turn"] >= 0]

        gamma = self._parent.info_df["nrj"]["info"] / 0.938
        eps_g = self._parent.info_df["eps_n"]["info"] / gamma

        if only_worst:
            collect = pd.DataFrame()

        if collimator_handler is not None:
            collimators = collimator_handler.compute_settings(
                twiss_df[twiss_df["turn"] == 1],
                self._parent.info_df["eps_n"],
                self._parent.info_df["nrj"],
            )

        for idx, element in enumerate(elements):
            data_thick = twiss_thick.loc[element]
            data = twiss_df.loc[element]

            if (
                collimator_handler is not None
                and element.lower() in collimators.index.str.lower()
            ):
                el_col_data = collimators.loc[element.lower()]
                angle = float(el_col_data["angle"])
                halfgap = float(el_col_data["half_gap"])

                beta_skew = abs(
                    data["betx"] * np.cos(angle) + data["bety"] * np.sin(angle)
                )

                sig_elem = np.sqrt(eps_g * beta_skew)

                xn = data["x"] / np.sqrt(eps_g * data["betx"])
                yn = data["y"] / np.sqrt(eps_g * data["bety"])
                excursion = halfgap - abs(
                    np.sqrt(xn ** 2 + yn ** 2) * np.cos(np.arctan(yn / xn) - angle)
                )
            else:
                halfgap = data_thick[f"aper_{1 if axis == 'x' else 2}"]

                sig_elem = np.sqrt(eps_g * data[f"bet{axis}"])

                xn = data["x"] / np.sqrt(eps_g * data["betx"])
                yn = data["y"] / np.sqrt(eps_g * data["bety"])
                excursion = np.sqrt(xn ** 2 + yn ** 2)

            if use_excursion:
                effective_gap = halfgap / sig_elem - excursion
            else:
                effective_gap = halfgap / sig_elem
            effective_gap = effective_gap.clip(lower=0)

            if only_worst:
                collect[element] = list(effective_gap)
                continue

            if parallel:
                start_col = "#000000"
                end_col = "#ff0000"
                x = idx
                for idx2, gap in enumerate(effective_gap):
                    col = _Artist.lerp_hex_color(
                        start_col, end_col, idx2 / len(effective_gap)
                    )
                    kwarg = {"mode": "lines"}
                    kwarg.update(trace_kwargs)
                    self.add_data(
                        x=[x - 0.25, x, x + 0.25],
                        y=[gap] * 3,
                        marker_color=col,
                        showlegend=False,
                        name=f"Turn {idx2}",
                        **kwarg,
                    )

                    self.plot_layout(
                        xaxis={
                            "tickmode": "array",
                            "tickvals": list(range(len(elements))),
                            "ticktext": elements,
                        }
                    )
            else:
                elem_name = element if suffix is None else suffix + element

                self.add_data(
                    x=data["turn"],
                    y=effective_gap,
                    name=elem_name,
                    **trace_kwargs[element] if trace_kwargs is not None else {},
                )

        if only_worst:
            worst = []
            for _, row in collect.iterrows():
                worst.append(min(row))
            self.add_data(
                x=data["turn"],
                y=worst,
                **kwargs,
            )

    def clear_figure(self):
        """TODO: Docstring for clear_figure.

        Args:

        Returns: TODO

        """
        super().clear_figure()
        self._center_elem = None

    @property
    def figure(self):
        """TODO: Docstring for figure.

        Args:

        Returns: TODO

        """
        fig = super().figure

        return fig


class _TrackArtist(_TwissArtist):

    """Docstring for _TrackArtist. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def orbit_excursion(self, axis: str = "r", **kwargs):
        """TODO: Docstring for orbit_excursion.

        Args:
            axis: TODO

        Returns: TODO

        """
        track = self._apply_observation_filter(self._parent.track_df.copy())

        x_data = track["turn"]
        y_data = self._parent.calculate_action(track)[axis]

        self.add_data(**kwargs, x=x_data, y=y_data)

    def loss_histogram(self, mode: str = "stacked", by_group: bool = True, **kwargs):
        """TODO: Docstring for loss_histogram.

        Args:
        function (TODO): TODO

        Returns: TODO

        """
        loss = self._parent.loss_df
        track = self._parent.track_df
        twiss = self._parent.twiss_df

        # Remove negative turns; this is a result of the time
        # dependencies being delayed by a single turn in the
        # twiss command, in order to ensure a single clean turn
        twiss = twiss.loc[twiss["turn"] >= 0]

        if by_group:
            re_group = re.compile(r"^.*?(?=\.)")
            loss["group"] = loss.apply(
                lambda x: re_group.findall(x["element"])[0], axis=1
            )

        if mode == "parallel":
            start_col = "#000000"
            end_col = "#ff0000"

            for turn, data in loss.groupby("turn"):
                col = _Artist.lerp_hex_color(
                    start_col, end_col, turn / max(loss["turn"])
                )
                count = data.value_counts(
                    "group" if by_group else "element", sort=False
                )

                count = 100 * count / max(track["number"])

                self.add_data(
                    x=count.index,
                    y=count,
                    name=f"Turn {int(turn)}",
                    marker_color=col,
                    type="bar",
                    showlegend=False,
                    xaxis={"tickson": "boundaries"},
                )
        elif mode == "stacked":
            for k, v in loss.groupby("group" if by_group else "element", sort=False):
                data = v.value_counts("turn", sort=False)
                data = data.sort_index()
                data = 100 * data / max(track["number"])
                self.add_data(x=data.index, y=data, name=k, type="bar")

            self._global_layout.update(barmode="stack")
        elif mode == "longitudinal":
            data = loss.copy()
            data = self._crop_to_centered(data)
            data = data.value_counts("s", sort=False) * 100 / max(track["number"])
            lengths = []
            if len(data) != 0:
                lengths = [
                    max(twiss.loc[twiss["s"] == data.index[i][0]]["lrad"])
                    for i in range(len(data))
                ]

            self.add_data(
                x=[float(x[0]) for x in data.index],
                y=data,
                type="bar",
                name="Loss",
                width=lengths,
            )

    def boxplot_envelope(
        self,
        twiss_path: str,
        element: str,
        normalize: bool = False,
        reference: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        """TODO: Docstring for boxplot_envelope.

        Args:
            twiss_path (TODO): TODO
            element (TODO): TODO

        Returns:
            str: Returns either 'Vertical' or 'Horizontal' depending on which axis has the larger aperture.

        """
        if normalize:
            turns = set(self._parent.twiss_df["turn"])
            if reference is None:
                reference = self._parent.twiss_df.loc[
                    self._parent.twiss_df["turn"] == min(turns)
                ]
            ref = reference.loc[element]

        twiss_thick = pd.read_parquet(twiss_path)

        data = self._parent.track_df.loc[element].copy()
        twiss_data = twiss_thick.loc[element].copy()

        gamma = self._parent.info_df["nrj"]["info"] / 0.938
        eps_g = self._parent.info_df["eps_n"]["info"] / gamma

        aper = "1" if twiss_data["aper_1"] < twiss_data["aper_2"] else "2"
        axis, vh = ("x", "Horizontal") if aper == "1" else ("y", "Vertical")

        if normalize:
            data[axis] = data[axis].div(np.sqrt(eps_g * ref[f"bet{axis}"]))
            twiss_data[f"aper_{aper}"] /= np.sqrt(eps_g * ref[f"bet{axis}"])

        sig = data.loc[data["turn"] == min(data["turn"])][axis].std()

        mean = data.groupby("turn").mean()

        style = dict(
            boxpoints=False,
            boxmean="sd",
            marker_size=1,
        )
        style.update(kwargs)

        # Boxplot
        self.add_data(
            **style,
            x=data["turn"],
            y=data[axis],
            type="box",
            showlegend=True,
            name="Beam distribution",
        )

        # Aperture
        self.add_data(
            x=[min(data["turn"]) - 0.5]
            + list(data["turn"])
            + [max(data["turn"]) + 0.5],
            y=[twiss_data[f"aper_{aper}"]] * (len(data["turn"]) + 2),
            marker_color="rgba(255,50,50,0.75)",
            showlegend=False,
            name=f"{element} {vh.lower()} aperture",
            legendgroup="aperture",
            line_width=2,
        )
        self.add_data(
            x=[min(data["turn"]) - 0.5]
            + list(data["turn"])
            + [max(data["turn"]) + 0.5],
            y=[-twiss_data[f"aper_{aper}"]] * (len(data["turn"]) + 2),
            marker_color="rgba(255,50,50,0.75)",
            showlegend=True,
            name=f"{element} {vh.lower()} aperture",
            legendgroup="aperture",
            line_width=2,
        )

        # Nominal width
        self.add_data(
            x=mean.index,
            y=[sig*2]*len(mean),
            base=mean[axis]-sig,
            type="bar",
            name="Nominal beam sigma",
            marker_opacity=0.25,
        )

        return vh
