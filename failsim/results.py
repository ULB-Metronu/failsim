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

        self.fix_twiss_name()

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
            TrackingResult: A TrackingResult instance containing the loaded data.

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

    @staticmethod
    def lerp_hex_color(col1: str, col2: str, factor: float):
        r1, g1, b1 = (
            int(col1[1:3], 16),
            int(col1[3:5], 16),
            int(col1[5:], 16),
        )
        r2, g2, b2 = (
            int(col2[1:3], 16),
            int(col2[3:5], 16),
            int(col2[5:], 16),
        )

        r3 = r1 + (r2 - r1) * factor
        g3 = g1 + (g2 - g1) * factor
        b3 = b1 + (b2 - b1) * factor

        return f"#{int(r3):02x}{int(g3):02x}{int(b3):02x}"

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
    ):
        Result.__init__(self, twiss_df, summ_df, run_version, hllhc_version)

        self.track_df = track_df

        self.track_df = self.normalize_track(self.twiss_df, self.track_df, eps_n, nrj)

        self.plot = _TrackArtist(self)

    @staticmethod
    def normalize_track(
        twiss_df: pd.DataFrame,
        track_df: pd.DataFrame,
        eps_n: float = 2.5e-6,
        nrj: float = 7000,
    ):
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
        gamma = nrj / 0.938
        eps_g = eps_n / gamma

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

        return data_out

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

    def _get_centered_range(self):
        """TODO: Docstring for _get_centered_range.

        Args:

        Returns: TODO

        """
        center_s = self._parent.twiss_df.loc[self._center_elem]["s"].iloc[0]
        return (center_s - self._width / 2, center_s + self._width / 2)

    def _crop_to_centered(self, data: pd.DataFrame):
        """TODO: Docstring for _crop_to_centered.

        Args:

        Returns: TODO

        """
        center_range = self._get_centered_range()
        return data[(data["s"] > center_range[0]) & (data["s"] < center_range[1])]

    def twiss_column(self, columns: Union[str, List[str]], **kwargs):
        """TODO: Docstring for twiss_column.

        Args:
        function (TODO): TODO

        Returns: TODO

        """
        if type(columns) is str:
            columns = [columns]

        twiss = self._apply_observation_filter(self._parent.twiss_df)

        for column in columns:
            x_data = twiss["s"]
            y_data = twiss[column]

            self.add_data(
                **kwargs,
                x=x_data,
                y=y_data,
                name=column,
            )

            if self._center_elem is not None:
                center_range = self._get_centered_range()
                col, row = self._plot_pointer
                self.plot_settings(
                    xaxis={
                        "range": (
                            center_range[0] * self._subplots[col][row]["factor"]["x"],
                            center_range[1] * self._subplots[col][row]["factor"]["x"],
                        )
                    },
                )

    def cartouche(self, twiss_path: Tuple[str, str] = None):
        """TODO: Docstring for cartouche.

        Args:

        Returns: TODO

        """
        for ss in ["1", "2"]:
            # Read twiss data
            if twiss_path is None:
                twiss = pd.read_parquet(
                    FailSim.path_to_output(f"twiss_pre_thin_b{ss}.parquet")
                )
            else:
                path = twiss_path[int(ss) - 1]
                if not path.startswith("/"):
                    path = FailSim.path_to_cwd(path)
                twiss = pd.read_parquet(path)

            # Filter if center_elem is defined
            if self._center_elem is not None:
                center_range = self._get_centered_range()
                twiss = twiss[
                    (twiss["s"] > center_range[0]) & (twiss["s"] < center_range[1])
                ]

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
                    and ss == "2"
                ):
                    continue

                x0 = row["s"] - (row["l"] if row["l"] != 0 else 0.5)
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
        beam_magnitudes: List[float] = [1, 5, 10],
        collimator_handler: CollimatorHandler = None,
        twiss_path: str = None,
    ):
        """TODO: Docstring for aperture.

        Args:

        Returns: TODO

        """
        if twiss_path is None:
            twiss_thick = pd.read_parquet(
                FailSim.path_to_output("twiss_pre_thin_b1.parquet")
            )
        else:
            if not twiss_path.startswith("/"):
                twiss_path = FailSim.path_to_cwd(twiss_path)
            twiss_thick = pd.read_parquet(twiss_path)

        twiss = self._parent.twiss_df.copy()

        twiss = self._apply_observation_filter(twiss)

        # Filter if center_elem is defined
        if self._center_elem is not None:
            center_range = self._get_centered_range()
            twiss_thick = twiss_thick[
                (twiss_thick["s"] > center_range[0])
                & (twiss_thick["s"] < center_range[1])
            ]
            col, row = self._plot_pointer
            self.plot_settings(
                xaxis={
                    "range": (
                        center_range[0] * self._subplots[col][row]["factor"]["x"],
                        center_range[1] * self._subplots[col][row]["factor"]["x"],
                    )
                },
            )

        if collimator_handler is not None:
            settings = collimator_handler.compute_settings(
                twiss, self._parent.info_df["eps_n"], self._parent.info_df["nrj"]
            )
            for _, row in settings.iterrows():
                if row.name.lower() in twiss_thick.index:
                    twiss_thick.at[row.name.lower(), "aper_1"] = row["gaph"]
                    twiss_thick.at[row.name.lower(), "aper_2"] = row["gapv"]

        twiss_thick = twiss_thick.loc[
            ~twiss_thick["keyword"].isin(
                ["drift", "marker", "placeholder", "monitor", "instrument"]
            )
        ]

        for _, row in twiss_thick.iterrows():
            x0 = row["s"] - (row["l"] if row["l"] != 0 else 0.5)
            x1 = row["s"]
            y0 = abs(row[f"aper_{1 if axis == 'x' else 2}"])
            y1 = 1

            # Move elements with 0 mm aperture to 50 mm
            if y0 == 0 or y0 == float("inf"):
                y0 = 200e-3

            style = self.aper_style.copy()
            style.update(
                fillcolor=self.element_colors[row["keyword"]],
                name=f"{row['name']}: {y0} mm",
            )

            for pol in [-1, 1]:
                self.add_data(
                    x=[x0, x1, x1, x0, x0],
                    y=[pol * y0, pol * y0, pol * y1, pol * y1, pol * y0],
                    **style,
                )
            style.update(
                line_width=1,
                opacity=0.1,
            )
            self.add_data(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, -y0, -y0, y0],
                **style,
            )

        gamma = self._parent.info_df["nrj"]["info"] / 0.938
        eps_g = self._parent.info_df["eps_n"]["info"] / gamma
        dpp = 1e-4

        envelope = np.sqrt(
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
            alpha = 1 - 2 * abs(idx / len(beam_magnitudes) - 0.5)
            self.add_data(
                x=twiss["s"],
                y=mag * envelope + twiss[axis],
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


class _TrackArtist(_TwissArtist):

    """Docstring for _TrackArtist. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def orbit_excursion(self, **kwargs):
        """TODO: Docstring for orbit_excursion.

        Args:

        Returns: TODO

        """
        track = self._apply_observation_filter(self._parent.track_df.copy())

        x_data = track["turn"]
        y_data = self._parent.calculate_action(track)["r"]

        self.add_data(**kwargs, x=x_data, y=y_data)
