from typing import List, Tuple, Dict

import plotly.graph_objects as go

import numpy as np
import pandas as pd


class _Artist:

    """Docstring for _Artist. """

    def __init__(self):
        self.figure = go.Figure()

        self._rows = 1
        self._cols = 1

        self._global_layout = {}
        self._subplots = np.array([[0]], dtype=object)

    def global_settings(
        self,
        rows: int = None,
        cols: int = None,
        **kwargs,
    ):
        """TODO: Docstring for global_settings.

        Args:

        Returns: TODO

        """
        self._global_layout.update(kwargs)

        if cols is not None or rows is not None:
            self._rows = rows if rows is not None else self._rows
            self._cols = cols if cols is not None else self._cols

            self._subplots.resize((self._cols, self._rows))
            self._populate_subplots()

    def plot_settings(self, row: int, col: int, xaxis: Dict = {}, yaxis: Dict = {}):
        """TODO: Docstring for plot_settings.

        Args:
        function (TODO): TODO

        Returns: TODO

        """
        idx = col + row * self._cols
        str_idx = "" if (idx + 1) == 1 else str(idx + 1)

        self._subplots[col][row][f"xaxis{str_idx}"].update(xaxis)
        self._subplots[col][row][f"yaxis{str_idx}"].update(yaxis)

    def _populate_subplots(self):
        plot_width = 1 / float(self._cols)
        plot_height = 1 / float(self._rows)

        for i in range(self._rows):
            for j in range(self._cols):
                idx = j + i * self._cols
                str_idx = "" if (idx + 1) == 1 else str(idx + 1)

                if self._subplots[j][i] == 0:
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
                    }

    def clear_figure(self):
        """TODO: Docstring for clear_figure.

        Returns: TODO

        """
        self.figure = go.Figure()

    def add_data(
        self,
        x: pd.Series,
        y: pd.Series,
        col: int = 0,
        row: int = 0,
        share_xaxis: Tuple[int, int] = None,
        share_yaxis: Tuple[int, int] = None,
        xaxis: Dict = {},
        yaxis: Dict = {},
        **kwargs,
    ):
        """TODO: Docstring for add_data.

        Args:

        Returns: TODO

        """
        idx = col + row * self._cols
        str_idx = "" if (idx + 1) == 1 else str(idx + 1)

        data_dict = {
            "x": x,
            "y": y,
            "xaxis": f"x{str_idx}",
            "yaxis": f"y{str_idx}",
        }

        if share_xaxis is not None:
            l_idx = share_xaxis[0] + share_xaxis[1] * self._cols
            l_str_idx = "" if (l_idx + 1) == 1 else str(l_idx + 1)
            data_dict["xaxis"] = f"x{l_str_idx}"
            self._subplots[col][row][f"yaxis{str_idx}"]["anchor"] = f"x{l_str_idx}"

        if share_yaxis is not None:
            l_idx = share_yaxis[0] + share_yaxis[1] * self._cols
            l_str_idx = "" if (l_idx + 1) == 1 else str(l_idx + 1)
            data_dict["yaxis"] = f"y{l_str_idx}"
            self._subplots[col][row][f"xaxis{str_idx}"]["anchor"] = f"y{l_str_idx}"

        self._subplots[col][row][f"xaxis{str_idx}"].update(xaxis)
        self._subplots[col][row][f"yaxis{str_idx}"].update(yaxis)

        data_dict.update(kwargs)
        self._subplots[col][row]["data"].append(data_dict)

    def get_figure(
        self,
    ):
        """TODO: Docstring for get_figure.

        Args:

        Returns: TODO

        """
        fig = go.Figure()

        fig.update_layout(self._global_layout)

        for plot in self._subplots.reshape(1, -1)[0]:
            fig.update_layout({k: v for k, v in plot.items() if k != "data"})
            for data in plot["data"]:
                fig.add_trace(data)

        return fig
