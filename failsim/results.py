
from dataclasses import dataclass
import pandas


@dataclass
class Result:

    """Docstring for Result. """

    twiss_df: pandas.DataFrame
    summ_df: pandas.DataFrame
    run_version: int
    hllhc_version: float

    def __init__(self,
                 twiss_df: pandas.DataFrame,
                 summ_df: pandas.DataFrame,
                 run_version: int = 0,
                 hllhc_version: float = 0.0):
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


class TrackingResult(Result):

    """Docstring for TrackingResult. """

    track_df: pandas.DataFrame

    def __init__(self,
                 twiss_df: pandas.DataFrame,
                 summ_df: pandas.DataFrame,
                 track_df: pandas.DataFrame,
                 run_version: int = 0,
                 hllhc_version: float = 0.0):
        """TODO: Docstring

        Args:
            twiss_df (TODO): TODO
            summ_df (TODO): TODO
            track_df (TODO): TODO

        Kwargs:
            run_version (TODO): TODO
            hllhc_version (TODO): TODO


        """
        Result.__init__(self, twiss_df,
                        summ_df, run_version,
                        hllhc_version)

        self.track_df = track_df
