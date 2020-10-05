# TODO Function to find specific elements, and show surrounding elements
# TODO Function to install elements
# TODO Function to check sequence length

from dataclasses import dataclass
import pandas as pd
import numpy as np


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
    ):
        """TODO: Docstring

        Args:
            twiss_df (TODO): TODO
            summ_df (TODO): TODO
            track_df (TODO): TODO

        Kwargs:
            run_version (TODO): TODO
            hllhc_version (TODO): TODO


        """
        Result.__init__(self, twiss_df, summ_df, run_version, hllhc_version)

        self.track_df = track_df

    def normalize_track(self, eps_n: float = 2.5e-6, nrj: float = 7000):
        """TODO: Docstring for normalize_track.

        Kwargs:
            eps_n (float): TODO
            nrj (float): TODO

        Returns: TODO

        """
        beta = 0.998
        gamma = 1 / np.sqrt(1 - beta ** 2)
        eps_g = eps_n / (gamma * beta)

        data_out = pd.DataFrame()

        for obs in set(self.track_df.index):
            data = self.track_df.loc[obs].copy()

            betx = self.twiss_df.loc[obs]["betx"]
            alfx = self.twiss_df.loc[obs]["alfx"]
            bety = self.twiss_df.loc[obs]["bety"]
            alfy = self.twiss_df.loc[obs]["alfy"]

            data["xn"] = data.apply(lambda x: x["x"] / np.sqrt(eps_g * betx), axis=1)

            data["pxn"] = data.apply(
                lambda x: x["x"] * alfx / np.sqrt(eps_g * betx)
                + x["px"] * np.sqrt(betx),
                axis=1,
            )

            data["yn"] = data.apply(lambda x: x["y"] / np.sqrt(eps_g * bety), axis=1)

            data["pyn"] = data.apply(
                lambda x: x["y"] * alfy / np.sqrt(eps_g * bety)
                + x["py"] * np.sqrt(bety),
                axis=1,
            )

            data_out = data_out.append(data)

        self.track_df = data_out

    def calculate_action(self):
        """TODO: Docstring for calculate_action.
        Returns: TODO

        """
        pass
