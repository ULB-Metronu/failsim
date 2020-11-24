from typing import List, Tuple, Dict

import plotly.graph_objects as go
import plotly.offline

import numpy as np
import pandas as pd

import os


class _Artist:

    """Docstring for _Artist. """

    def __init__(self):
        self._rows = 1
        self._cols = 1

        self._plot_pointer = [0, 0]

        self._global_layout = {}
        self._subplots = np.array([[0]], dtype=object)

        self._populate_subplots()

    def __getitem__(self, index):
        assert (type(index) == tuple) and (
            len(index) == 2
        ), "Index must specified as follows: [x,y]"
        self._plot_pointer = list(index)
        return self

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

    def save(self, name: str):
        """TODO: Docstring for save.

        Args:

        Returns: TODO

        """
        div = plotly.offline.plot(
            self.figure,
            include_plotlyjs="cdn",
            include_mathjax="cdn",
            output_type="div",
        )
        os.makedirs(os.path.dirname(name), exist_ok=True)
        with open(name, "w") as fd:
            fd.write(div)

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

        idx = col + row * self._cols
        str_idx = "" if (idx + 1) == 1 else str(idx + 1)

        # Apply factor
        if type(x) is list:
            x = [v * self._subplots[col][row]["factor"]["x"] for v in x]
        else:
            x = x.copy() * self._subplots[col][row]["factor"]["x"]
        if type(y) is list:
            y = [v * self._subplots[col][row]["factor"]["y"] for v in y]
        else:
            y = y.copy() * self._subplots[col][row]["factor"]["y"]

        data_dict = {
            "x": x,
            "y": y,
            "xaxis": f"x{str_idx}",
            "yaxis": f"y{str_idx}",
        }

        share_xaxis = self._subplots[col][row]["share"]["x"]
        if share_xaxis is not None:
            l_idx = share_xaxis[0] + share_xaxis[1] * self._cols
            l_str_idx = "" if (l_idx + 1) == 1 else str(l_idx + 1)
            data_dict["xaxis"] = f"x{l_str_idx}"
            self._subplots[col][row][f"yaxis{str_idx}"]["anchor"] = f"x{l_str_idx}"

        share_yaxis = self._subplots[col][row]["share"]["y"]
        if share_yaxis is not None:
            l_idx = share_yaxis[0] + share_yaxis[1] * self._cols
            l_str_idx = "" if (l_idx + 1) == 1 else str(l_idx + 1)
            data_dict["yaxis"] = f"y{l_str_idx}"
            self._subplots[col][row][f"xaxis{str_idx}"]["anchor"] = f"y{l_str_idx}"

        self._subplots[col][row][f"xaxis{str_idx}"].update(xaxis)
        self._subplots[col][row][f"yaxis{str_idx}"].update(yaxis)

        data_dict.update(kwargs)
        if to_bottom:
            self._subplots[col][row]["data"].insert(0, data_dict)
        else:
            self._subplots[col][row]["data"].append(data_dict)

    @property
    def figure(
        self,
    ):
        """TODO: Docstring for figure.

        Args:

        Returns: TODO

        """
        fig = go.Figure()

        fig.update_layout(self._global_layout)

        for plot in self._subplots.reshape(1, -1)[0]:
            fig.update_layout({k: v for k, v in plot.items() if "axis" in k})
            for data in plot["data"]:
                fig.add_trace(data)

        return fig

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

        return figure
