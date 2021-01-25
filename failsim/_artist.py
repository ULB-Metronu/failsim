from typing import List, Tuple, Dict, Optional, Union

import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline

import numpy as np
import pandas as pd

import os
import pickle


class _Artist:

    """Docstring for _Artist. """

    default_layout = dict(
        title=dict(
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=0.95,
            font_size=20,
        ),
        margin=dict(
            l=50,
            r=50,
            t=50,
            b=50,
        ),
        font_size=10,
        legend=dict(
            xanchor="right",
            yanchor="middle",
            x=0.95,
            y=0.5,
            bordercolor="black",
            borderwidth=1,
        ),
        xaxis=dict(
            showgrid=True,
            showline=True,
            zeroline=False,
            mirror="all",
            ticks="inside",
            exponentformat="E",
        ),
        yaxis=dict(
            showgrid=True,
            showline=True,
            zeroline=False,
            mirror="all",
            ticks="inside",
            exponentformat="E",
        ),
    )

    def __init__(self):
        self._rows = 1
        self._cols = 1

        self._plot_pointer = [0, 0]

        self._global_layout = {}
        self._frames = []
        self._annotations = []
        self._subplots = np.array([[0]], dtype=object)

        self._populate_subplots()

        failsim_template = dict(layout=go.Layout(self.default_layout))
        pio.templates["failsim"] = failsim_template

    def __getitem__(self, index):
        assert (type(index) == tuple) and (
            len(index) == 2
        ), "Index must specified as follows: [x,y]"
        self._plot_pointer = list(index)
        return self

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

    def global_layout(
        self,
        cols: int = None,
        rows: int = None,
        **kwargs,
    ):
        """TODO: Docstring for global_layout.

        Args:

        Returns: TODO

        """
        self._global_layout.update(kwargs)

        if cols is not None or rows is not None:
            self._rows = rows if rows is not None else self._rows
            self._cols = cols if cols is not None else self._cols

            self._subplots.resize((self._cols, self._rows), refcheck=False)
            self._populate_subplots()

            if self._plot_pointer[0] >= self._cols:
                self._plot_pointer[0] = self._cols - 1
            if self._plot_pointer[1] >= self._rows:
                self._plot_pointer[1] = self._rows - 1

    def plot_layout(
        self,
        xaxis: Dict = {},
        yaxis: Dict = {},
        share_xaxis: Tuple[int, int] = None,
        share_yaxis: Tuple[int, int] = None,
        axis_factors: Tuple[float, float] = None,
        data_modifiers: Dict = None,
    ):
        """TODO: Docstring for plot_layout.

        Args:

        Returns: TODO

        """
        col, row = self._plot_pointer

        idx = col + row * self._cols
        str_idx = "" if (idx + 1) == 1 else str(idx + 1)

        self._subplots[col][row][f"xaxis{str_idx}"].update(xaxis)
        self._subplots[col][row][f"yaxis{str_idx}"].update(yaxis)

        if share_xaxis is not None:
            self._subplots[col][row]["share"]["x"] = share_xaxis
        if share_yaxis is not None:
            self._subplots[col][row]["share"]["y"] = share_yaxis

        if axis_factors is not None:
            self._subplots[col][row]["factor"] = {
                "x": axis_factors[0],
                "y": axis_factors[1],
            }

    def _populate_subplots(self):
        plot_width = 1 / float(self._cols)
        plot_height = 1 / float(self._rows)

        for i in range(self._rows):
            for j in range(self._cols):
                idx = j + i * self._cols
                str_idx = "" if (idx + 1) == 1 else str(idx + 1)

                self._subplots[j][i] = {
                    "data": [],
                    f"xaxis{str_idx}": {
                        "domain": [
                            plot_width * j + 0.04,
                            plot_width * (j + 1) - 0.04,
                        ],
                        "anchor": f"y{str_idx}",
                    },
                    f"yaxis{str_idx}": {
                        "domain": [
                            plot_height * i + 0.04,
                            plot_height * (i + 1) - 0.04,
                        ],
                        "anchor": f"x{str_idx}",
                    },
                    "factor": {"x": 1, "y": 1},
                    "share": {
                        "x": None,
                        "y": None,
                    },
                }

    def clear_figure(self):
        """TODO: Docstring for clear_figure.

        Returns: TODO

        """
        self.global_layout(1, 1)
        self._global_layout = {}
        self._subplots.fill(0)
        self._populate_subplots()

    def clear_data(self):
        """TODO: Docstring for clear_data.

        Args:

        Returns: TODO

        """
        for col in range(self._cols):
            for row in range(self._rows):
                self._subplots[col][row]["data"] = []
                self._frames = []

    def save(
        self,
        name: str,
        as_pickle: bool = False,
        ext_figure: Optional[Union[Dict, go.Figure]] = None,
    ):
        """TODO: Docstring for save.

        Args:

        Returns: TODO

        """
        fig = self.figure if ext_figure is None else ext_figure

        if type(fig) is dict:
            fig = go.Figure(fig)

        # Create directory if it doesn't exist
        file_dir = os.path.dirname(name)
        if file_dir != "":
            os.makedirs(file_dir, exist_ok=True)

        if as_pickle:
            if not name.endswith(".pickle"):
                name += ".pickle"
            with open(name, "wb") as fd:
                pickle.dump(fig.to_dict(), fd)
        else:
            div = plotly.offline.plot(
                fig,
                include_plotlyjs="cdn",
                include_mathjax="cdn",
                output_type="div",
            )
            if not name.endswith(".html"):
                name += ".html"
            with open(name, "w") as fd:
                fd.write(div)

    def render(self, name: str, ext_figure: Optional[Union[Dict, go.Figure]] = None, **kwargs):
        """TODO: Docstring for render_png.

        Args:
            name (TODO): TODO

        Returns: TODO

        """
        fig = self.figure if ext_figure is None else ext_figure

        if type(fig) is dict:
            fig = go.Figure(fig)

        render_kwargs = dict(format="png", scale=8)
        render_kwargs.update(kwargs)
        if not name.endswith(f".{render_kwargs['format']}"):
            name += f".{render_kwargs['format']}"
        fig.write_image(file=name, **render_kwargs)

    def _process_data(
        self, x: pd.Series, y: pd.Series, xaxis: Dict = {}, yaxis: Dict = {}, **kwargs
    ):
        """TODO: Docstring for _process_data.

        Args:
        function (TODO): TODO

        Returns: TODO

        """
        col, row = self._plot_pointer

        idx = col + row * self._cols
        str_idx = "" if (idx + 1) == 1 else str(idx + 1)

        # Apply scaling factor to x and y.
        if type(x) is list:
            x = [v * self._subplots[col][row]["factor"]["x"] for v in x]
        elif type(x) is pd.Series:
            x = x.copy() * self._subplots[col][row]["factor"]["x"]
        if type(y) is list:
            y = [v * self._subplots[col][row]["factor"]["y"] for v in y]
        elif type(y) is pd.Series:
            y = y.copy() * self._subplots[col][row]["factor"]["y"]

        data_dict = {
            "x": x,
            "y": y,
            "xaxis": f"x{str_idx}",
            "yaxis": f"y{str_idx}",
        }

        # Specify shared axis for x.
        share_xaxis = self._subplots[col][row]["share"]["x"]
        if share_xaxis is not None:
            l_idx = share_xaxis[0] + share_xaxis[1] * self._cols
            l_str_idx = "" if (l_idx + 1) == 1 else str(l_idx + 1)
            data_dict["xaxis"] = f"x{l_str_idx}"
            self._subplots[col][row][f"yaxis{str_idx}"]["anchor"] = f"x{l_str_idx}"

        # Specify shared axis for y.
        share_yaxis = self._subplots[col][row]["share"]["y"]
        if share_yaxis is not None:
            l_idx = share_yaxis[0] + share_yaxis[1] * self._cols
            l_str_idx = "" if (l_idx + 1) == 1 else str(l_idx + 1)
            data_dict["yaxis"] = f"y{l_str_idx}"
            self._subplots[col][row][f"xaxis{str_idx}"]["anchor"] = f"y{l_str_idx}"

        # Add subplot specific axis layout options.
        self._subplots[col][row][f"xaxis{str_idx}"].update(xaxis)
        self._subplots[col][row][f"yaxis{str_idx}"].update(yaxis)

        data_dict.update(kwargs)

        return data_dict

    def add_annotation(self, **kwargs):
        """TODO: Docstring for add_annotation.

        Args:
        function (TODO): TODO

        Returns: TODO

        """
        self._annotations.append(kwargs)

    def add_data(
        self,
        x: pd.Series,
        y: pd.Series,
        xaxis: Dict = {},
        yaxis: Dict = {},
        to_bottom: bool = False,
        **kwargs,
    ):
        """TODO: Docstring for add_data.

        Args:

        Returns: TODO

        """
        col, row = self._plot_pointer

        data_dict = self._process_data(**kwargs, x=x, y=y, xaxis=xaxis, yaxis=yaxis)

        if to_bottom:
            self._subplots[col][row]["data"].insert(0, data_dict)

            # Increment each trace in each frame
            # for frame in self._frames:
            #     frame["traces"] = [x + 1 for x in frame["traces"]]

        else:
            self._subplots[col][row]["data"].append(data_dict)

    def add_frame(
        self,
        x: pd.Series,
        y: pd.Series,
        frame_name: str,
        frame_trace: int,
        xaxis: Dict = {},
        yaxis: Dict = {},
        **kwargs,
    ):
        """TODO: Docstring for add_frame.

        Args:
        function (TODO): TODO

        Returns: TODO

        """
        data_dict = self._process_data(**kwargs, x=x, y=y, xaxis=xaxis, yaxis=yaxis)

        # Make sure lines arent simplified.
        # This makes sure animations don't jump around strangely.
        if "line" in data_dict:
            data_dict["line"].update(simplify=False)
        else:
            data_dict.update(line={"simplify": False})

        # Loop through all frames to see if frame with
        # frame_name already exists. If one already exists,
        # add the data to this frame instead of creating a
        # new one.
        frame_found = False
        for frame in self._frames:
            if frame["name"] == frame_name:
                frame["data"].append(data_dict)
                frame["traces"].append(frame_trace)
                frame_found = True
        if not frame_found:
            frame = {
                "name": frame_name,
                "data": [data_dict],
                "traces": [frame_trace],
            }
            self._frames.append(frame)

        # If "sliders" key doesn't exist in global_layout,
        # add this key to global_layout with default values.
        if "sliders" not in self._global_layout:
            self._global_layout["sliders"] = [{"active": 0, "steps": []}]

        # Loop through all steps in global slider to see
        # if step for given frame_name already exists.
        # If it doesn't already, append new step with
        # given frame_name to steps. If it exists, do nothing.
        step_found = False
        for step in self._global_layout["sliders"][0]["steps"]:
            if step["args"][0][0] == frame_name:
                step_found = True
                break
        if not step_found:
            step = {
                "args": [
                    [frame_name],
                    {"frame": {"duration": 100, "redraw": True}},
                ],
                "label": frame_name,
                "method": "animate",
            }
            self._global_layout["sliders"][0]["steps"].append(step)

    def set_yrange_for_xrange(self, xrange: [Tuple[float, float]]):
        """TODO: Docstring for set_yrange_for_xrange.

        Args:

        Returns: TODO

        """
        col, row = self._plot_pointer
        plot = self._subplots[col, row]
        factor = plot["factor"]

        val_min = float("inf")
        val_max = -float("inf")
        for data in plot["data"]:
            range_check = [
                xrange[1] * factor["x"] > x > xrange[0] * factor["x"] for x in data["x"]
            ]
            range_data = [
                data["y"][i] for i in range(len(data["x"])) if range_check[i] == True
            ]
            if len(range_data) == 0:
                continue
            val_min = val_min if min(range_data) > val_min else min(range_data)
            val_max = val_max if max(range_data) < val_max else max(range_data)

        val_min -= (val_max - val_min) * 0.1
        val_max += (val_max - val_min) * 0.1

        idx = col + row * self._cols
        str_idx = "" if (idx + 1) == 1 else str(idx + 1)

        self._subplots[col][row][f"yaxis{str_idx}"].update(range=(val_min, val_max))

    @property
    def figure(
        self,
    ):
        """TODO: Docstring for figure.

        Args:

        Returns: TODO

        """
        fig = {}
        fig["data"] = []
        fig["frames"] = []
        fig["layout"] = {}

        fig["layout"].update(self._global_layout)

        for plot in self._subplots.reshape(1, -1)[0]:
            for axis, values in {k: v for k, v in plot.items() if "axis" in k}.items():
                if axis in fig["layout"].keys():
                    fig["layout"][axis].update(values)
                else:
                    fig["layout"].update({axis: values})
            for data in plot["data"]:
                fig["data"].append(data)

        fig["frames"] = self._frames
        fig["annotations"] = self._annotations

        return go.Figure(fig)

    def external_figure(self, figure: go.Figure):
        """TODO: Docstring for external_figure.

        Args:

        Returns: TODO

        """
        figure.update_layout(self._global_layout)

        for plot in self._subplots.reshape(1, -1)[0]:
            figure.update_layout({k: v for k, v in plot.items() if "axis" in k})
            for data in plot["data"]:
                figure.add_trace(data)

        for frame_self in self._frames:
            frame_found = False
            for frame in figure.frames:
                if str(frame["name"]) == str(frame_self["name"]):
                    frame["data"] = list(frame["data"]) + frame_self["data"]
                    frame["traces"] = list(frame["traces"]) + frame_self["traces"]
                    frame_found = True
            if not frame_found:
                figure.frames = list(figure.frames) + [frame_self]

        return figure
