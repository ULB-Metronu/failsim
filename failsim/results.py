"""
Module containing classes that contain and handle data.
"""


from .failsim import FailSim
from typing import Optional, List, Union, Dict, Tuple, Callable, Type
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

    def _plot(
        self,
        x_data: pd.Series,
        y_data: pd.Series,
        trace_name: Optional[str] = None,
        save_path: Optional[str] = None,
        figure: Optional[go.Figure] = None,
        center_elem: str = None,
        width: float = None,
        plot_type: Type = go.Scatter,
        **kwargs,
    ):
        """ Private method that provides basic plotting utility. """
        layout_kwargs = {x[7:]: kwargs[x] for x in kwargs if x.startswith("layout_")}
        trace_kwargs = {x[6:]: kwargs[x] for x in kwargs if x.startswith("trace_")}

        if figure is None:
            figure = go.Figure()

        data = plot_type(trace_kwargs, x=x_data, y=y_data, name=trace_name)

        figure.add_trace(data)

        figure.update_layout(layout_kwargs)

        if center_elem is not None:
            assert width is not None, "width must be specified when using center_elem"
            elem_s = self.twiss_df.loc[center_elem]["s"]
            if type(elem_s) is pd.Series:
                elem_s = elem_s.iloc[0]
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
        plot_type: Type = go.Scatter,
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
            plot_type: Allows specification of a plotly plot type. Accepts any graph_objects plotting class that takes x and y keyword arguments.

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
            plot_type=plot_type,
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
        plot_type: Type = go.Scatter,
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
            plot_type: Allows specification of a plotly plot type. Accepts any graph_objects plotting class that takes x and y keyword arguments.

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
            plot_type=plot_type,
            **kwargs,
        )

    def plot_effective_gap(
        self,
        elements: Union[str, List[str]],
        apertures: Union[float, List[float]],
        axis: str = "x",
        observation_filter: Callable[[pd.DataFrame], pd.DataFrame] = None,
        save_path: Optional[str] = None,
        figure: Optional[go.Figure] = None,
        center_elem: str = None,
        width: float = None,
        parallel: bool = False,
        **kwargs,
    ):
        """
        Plots the effective gap of the selected elements.

        Args:
            elements: The elements to plot.
            apertures: The apertures of the elements in metres.
            axis: Can either be 'x' or 'y' to specify which axis to plot.
            observation_filter: Filters tracking data indexes by each item. Can either be a list of observation points, or a single observation point.
            save_path: Path and filename of where to save the figure. If save_path is None, the plot is not saved.
            figure: The figure to add the plot to. If figure is None, a new plotly Figure object is created.
            center_elem: Element on which the plot will be centered. If no element is specified, the method will not center any specific element. If center_elem is specified, width must not be None.
            width: The difference between the leftmost and rightmost points on the plot. Is meant to be used in conjuction with center_elem. If no center_elem is specified, width does nothing.
            parallel: TODO

        Returns:
            go.Figure: Returns either the newly created figure if no figure was specified, or the figure the plot was added to.

        """
        twiss = self.twiss_df.copy()

        if observation_filter is not None:
            twiss = twiss.loc[observation_filter(twiss)]

        if not isinstance(elements, list):
            elements = [elements]

        if not isinstance(apertures, list):
            apertures = [apertures]

        if figure is None:
            figure = go.Figure()

        for element, aperture in zip(elements, apertures):
            gamma = self.info_df["nrj"]["info"] / 0.938
            eps_g = self.info_df["eps_n"]["info"] / gamma

            beta_elem = twiss.loc[element]
            beta_ref = beta_elem.loc[beta_elem["turn"] == 1]

            sig_elem = np.sqrt(eps_g * beta_elem[f"bet{axis}"])
            sig_ref = np.sqrt(eps_g * beta_ref[f"bet{axis}"])

            xn = beta_elem["x"] / np.sqrt(eps_g * beta_ref["betx"])
            yn = beta_elem["y"] / np.sqrt(eps_g * beta_ref["bety"])
            excursion = np.sqrt(xn ** 2 + yn ** 2)

            effective_gap = (aperture - excursion) * sig_ref / sig_elem

            start_col = "#000000"
            end_col = "#ff5511"
            if parallel:
                for idx, val in enumerate(effective_gap):
                    col = Result.lerp_hex_color(
                        start_col,
                        end_col,
                        idx / len(effective_gap),
                    )

                    x_data = [element]
                    y_data = [val]

                    self._plot(
                        x_data=x_data,
                        y_data=y_data,
                        observation_filter=observation_filter,
                        trace_name=element,
                        save_path=save_path,
                        figure=figure,
                        center_elem=center_elem,
                        width=width,
                        plot_type=go.Box,
                        trace_showlegend=False,
                        trace_marker_color=col,
                        trace_hovertemplate="%{x}",
                        layout_yaxis_title=r"$\text{Effective half-gap} \: [\sigma]$",
                        layout_xaxis_showgrid=False,
                        **kwargs,
                    )
            else:
                x_data = beta_elem["turn"]
                y_data = effective_gap

                self._plot(
                    x_data=x_data,
                    y_data=y_data,
                    observation_filter=observation_filter,
                    trace_name=element,
                    save_path=save_path,
                    figure=figure,
                    center_elem=center_elem,
                    width=width,
                    plot_type=go.Scatter,
                    trace_mode="lines",
                    layout_xaxis_title=r"$\text{Time} \: [\text{LHC turn}]$",
                    layout_yaxis_title=r"$\text{Effective half-gap} \: [\sigma]$",
                    **kwargs,
                )

        return figure

    @classmethod
    def create_cartouche(
        cls,
        fig_range: Tuple[float, float] = None,
        center_elem: str = None,
        width: float = None,
    ):
        """Creates a cartouche plot.

        Args:
            fig_range: The longitudinal range in which to draw the elements of the sequence.
            center_elem: Name of element that should be the center of the drawn elements. If this parameter is specified, width has to be specified.
            width: Width around center_elem that elements should be drawn. Must be specified if center_elem is specified.

        Returns:
            go.Figure: Figure with sequence objects drawn above plot.

        """
        fig = go.Figure()

        colors = dict(
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

        for ss in ["1", "2"]:
            # Read twiss and survey data
            twiss = pd.read_parquet(
                FailSim.path_to_output(f"twiss_pre_thin_b{ss}.parquet")
            )
            survey = pd.read_parquet(
                FailSim.path_to_output(f"survey_pre_thin_b{ss}.parquet")
            )

            # Filter data around center_elem
            if center_elem is not None:
                assert (
                    width is not None
                ), "width must be specified when using center_elem"
                elem_s = twiss.loc[center_elem]["s"]
                if type(elem_s) is pd.Series:
                    elem_s = elem_s.iloc[0]
                twiss = twiss.loc[
                    (twiss["s"] > elem_s - width / 2)
                    & (twiss["s"] < elem_s + width / 2)
                ]
                survey = survey.loc[
                    (survey["s"] > elem_s - width / 2)
                    & (survey["s"] < elem_s + width / 2)
                ]

            # Remove drift, marker, placeholder and monitor elements from data
            twiss = twiss.loc[
                ~twiss["keyword"].isin(["drift", "marker", "placeholder", "monitor"])
            ]
            survey = survey.loc[
                ~survey["keyword"].isin(["drift", "marker", "placeholder", "monitor"])
            ]

            # Filter data to fit within fig_range
            if fig_range is not None:
                twiss = twiss.loc[
                    (twiss["s"] > fig_range[0]) & (twiss["s"] < fig_range[1])
                ]
                survey = survey.loc[
                    (survey["s"] > fig_range[0]) & (survey["s"] < fig_range[1])
                ]

            # Interpolate mech_sep between each rbend element
            survey["beam_sep"] = survey.apply(
                lambda x: x["mech_sep"] if x["keyword"] == "rbend" else float("nan"),
                axis=1,
            )
            beam_sep = survey.set_index("s")["beam_sep"].interpolate(
                method="index",
                limit_direction="both",
            )

            # Normalize beam separation
            beam_sep = beam_sep / max(abs(beam_sep))
            beam_sep = beam_sep.set_axis(survey.index)

            # Draw beam line underneath all other plots
            fig.data = fig.data[::-1]
            fig.add_trace(
                go.Scatter(
                    x=twiss["s"],
                    y=beam_sep,
                    yaxis="y2",
                    marker_color="gray",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig.data = fig.data[::-1]

            for _, row in twiss.iterrows():
                # Skip drawing beam 2 when both beams are in same pipe
                if beam_sep.loc[row["name"][:-2]] == 0 and ss == "2":
                    continue

                x0 = row["s"] - (row["l"] if row["l"] != 0 else 0.5)
                x1 = row["s"]
                dy = 0.9
                y0 = beam_sep.loc[row["name"][:-2]] - dy / 2

                # Draw rbend and sbend
                if row["keyword"] in ["rbend", "sbend"]:
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1, x1, x0, x0],
                            y=[y0, y0, y0 + dy, y0 + dy, y0],
                            yaxis="y2",
                            showlegend=False,
                            marker_color=colors[row["keyword"]],
                            fillcolor=colors[row["keyword"]],
                            mode="lines",
                            fill="toself",
                            name=row["name"],
                            line_width=0,
                        )
                    )
                # Draw collimators
                elif row["keyword"] in ["rcollimator"]:
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1, x1, x0, x0],
                            y=[y0, y0, y0 + dy / 3, y0 + dy / 3, y0],
                            yaxis="y2",
                            showlegend=False,
                            marker_color=colors[row["keyword"]],
                            fillcolor=colors[row["keyword"]],
                            mode="lines",
                            fill="toself",
                            name=row["name"],
                            line_width=0,
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1, x1, x0, x0],
                            y=[
                                y0 + dy * 2 / 3,
                                y0 + dy * 2 / 3,
                                y0 + dy,
                                y0 + dy,
                                y0 + dy * 2 / 3,
                            ],
                            yaxis="y2",
                            showlegend=False,
                            marker_color=colors[row["keyword"]],
                            fillcolor=colors[row["keyword"]],
                            mode="lines",
                            fill="toself",
                            name=row["name"],
                            line_width=0,
                        )
                    )
                # Draw quadrupole
                elif row["keyword"] in ["quadrupole"]:
                    y0_pol = y0 + dy / 4 + row["polarity"] * dy / 4
                    y1_pol = y0 + dy * 3 / 4 + row["polarity"] * dy / 4
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1, x1, x0, x0],
                            y=[y0_pol, y0_pol, y1_pol, y1_pol, y0_pol],
                            yaxis="y2",
                            showlegend=False,
                            marker_color=colors[row["keyword"]],
                            fillcolor=colors[row["keyword"]],
                            mode="lines",
                            fill="toself",
                            name=row["name"],
                            line_width=0,
                        )
                    )
                # Draw everything else
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1, x1, x0, x0],
                            y=[
                                y0 + dy / 4,
                                y0 + dy / 4,
                                y0 + dy * 3 / 4,
                                y0 + dy * 3 / 4,
                                y0 + dy / 4,
                            ],
                            yaxis="y2",
                            showlegend=False,
                            marker_color=colors[row["keyword"]],
                            fillcolor=colors[row["keyword"]],
                            mode="lines",
                            fill="toself",
                            name=row["name"],
                            line_width=0,
                        )
                    )

        # Small layout specifications
        fig.update_layout(
            yaxis=dict(
                domain=[0, 0.75],
            ),
            yaxis2=dict(
                domain=[0.8, 1],
                visible=False,
                fixedrange=True,
            ),
        )

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

        self.track_df = self.normalize_track(eps_n, nrj, self.twiss_df, self.track_df)

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

    def plot_orbit_excursion(
        self,
        observation_filter: Optional[Union[str, List[str]]] = None,
        trace_name: Optional[str] = None,
        save_path: Optional[str] = None,
        figure: Optional[go.Figure] = None,
        center_elem: str = None,
        width: float = None,
        plot_type: Type = go.Scatter,
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
            plot_type: Allows specification of a plotly plot type. Accepts any graph_objects plotting class that takes x and y keyword arguments.

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
            plot_type=plot_type,
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
