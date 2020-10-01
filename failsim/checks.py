"""
File: checks.py
Author: Oskari Tuormaa
Email: oskari.kristian.tuormaa@cern.ch
Github: https://github.com/Oskari-Tuormaa
Description: TODO
"""

from typing import List
import pymask
import pandas


class OpticsChecks:

    """TODO: Docstring for OpticsChecks. """

    def __init__(
        self,
        separation: bool = True,
        beta: bool = True,
        tol_sep: List = [1e-3, 10e-2, 1e-3, 1e-2],
        tol_beta: List = [1e-6, 1e-6, 1e-6, 1e-6],
        save_twiss_files: bool = True,
    ):
        """TODO: Docstring for __init__.

        Kwargs:
            separation (TODO): TODO
            beta (TODO): TODO
            tol_sep (TODO): TODO
            tol_beta (TODO): TODO
            save_twiss_files (TODO): TODO

        Returns: TODO

        """
        self.separation = separation
        self.beta = beta
        self.tol_sep = tol_sep
        self.tol_beta = tol_beta
        self.save_twiss_files = save_twiss_files

    def check_separations(
        self,
        twiss_df_b1: pandas.DataFrame,
        twiss_df_b2: pandas.DataFrame,
        variables_dict: dict,
    ):
        """TODO: Docstring for check_separations.

        Args:
            twiss_df_b1 (TODO): TODO
            twiss_df_b2 (TODO): TODO
            variables_dict (TODO): TODO

        Returns: TODO

        """
        separations_to_check = []
        for iip, ip in enumerate([1, 2, 5, 8]):
            for plane in ["x", "y"]:
                # (*) Adapet based on knob definitions
                separations_to_check.append(
                    {
                        "element_name": f"ip{ip}:1",
                        "scale_factor": -2 * 1e-3,
                        "plane": plane,
                        # knobs like on_sep1h, onsep8v etc
                        "varname": f"on_sep{ip}" + {"x": "h", "y": "v"}[plane],
                        "tol": self.tol_sep[iip],
                    }
                )
        pymask.check_separations_against_madvars(
            separations_to_check, twiss_df_b1, twiss_df_b2, variables_dict
        )

    def check_betas(self, beam: str, twiss_df: pandas.DataFrame, variables_dict: dict):
        """TODO: Docstring for check_betas.

        Args:
            beam (TODO): TODO
            twiss_df (TODO): TODO
            variables_dict (TODO): TODO

        Returns: TODO

        """
        twiss_value_checks = []
        for iip, ip in enumerate([1, 2, 5, 8]):
            for plane in ["x", "y"]:
                # (*) Adapt based on knob definitions
                twiss_value_checks.append(
                    {
                        "element_name": f"ip{ip}:1",
                        "keyword": f"bet{plane}",
                        "varname": f"bet{plane}ip{ip}b{beam}",
                        "tol": self.tol_beta[iip],
                    }
                )
        pymask.check_twiss_against_madvars(twiss_value_checks, twiss_df, variables_dict)

    def __call__(
        self, mad: pymask.Madxp, sequences: List[str], twiss_name: str = "twiss"
    ):
        """TODO: Docstring for __call__.

        Args:
            mad (TODO): TODO
            sequences (TODO): TODO

        Returns: TODO

        """
        var_dict = mad.get_variables_dicts()
        twiss_dfs = {}
        summ_dfs = {}
        for ss in sequences:
            mad.use(ss)
            mad.twiss()
            tdf = mad.get_twiss_df("twiss")
            twiss_dfs[ss] = tdf
            sdf = mad.get_summ_df("summ")
            summ_dfs[ss] = sdf

        if self.save_twiss_files:
            for ss in sequences:
                tt = twiss_dfs[ss]
                tt.to_parquet(twiss_name + f"_seq_{ss}.parquet")

        if self.beta:
            for ss in sequences:
                tt = twiss_dfs[ss]
                self.check_betas(beam=ss[-1], twiss_df=tt, variables_dict=var_dict)

        if self.separation:
            twiss_df_b1 = twiss_dfs["lhcb1"]
            twiss_df_b2 = twiss_dfs["lhcb2"]
            self.check_separations(twiss_df_b1, twiss_df_b2, var_dict)

        other_data = {}
        other_data.update(var_dict)
        other_data["summ_dfs"] = summ_dfs

        return twiss_dfs, other_data
