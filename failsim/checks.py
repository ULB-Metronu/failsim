"""
Contains classes that handle tolerance checking of sequences.
"""


from .failsim import FailSim

from typing import List, Dict
import pymask as pm
import pandas as pd
import logging


class OpticsChecks:
    """
    Class for handling tolerance checks of sequences.

    Is used internally by LHCSequence to run checks.

    Args:
        separation: Whether beam seperation should be checked.
        beta: Whether betas should be checked.
        tol_sep: List containing the tolerances for beam separation.
        tol_beta: List containing the tolerances for betas.
        save_twiss_files: Whether the intermediate twiss files should be saved.

    """

    def __init__(
        self,
        separation: bool = True,
        beta: bool = True,
        tol_sep: List[float] = [1e-3, 10e-2, 1e-3, 1e-2],
        tol_beta: List[float] = [1e-6, 1e-6, 1e-6, 1e-6],
        save_twiss_files: bool = True,
    ):
        self.separation = separation
        self.beta = beta
        self.tol_sep = tol_sep
        self.tol_beta = tol_beta
        self.save_twiss_files = save_twiss_files

    def check_separations(
        self,
        twiss_df_b1: pd.DataFrame,
        twiss_df_b2: pd.DataFrame,
        variables_dict: Dict,
    ):
        """
        Checks seperation against tolerances specified in the constructor.

        Args:
            twiss_df_b1: DataFrame containing the twiss data for beam 1.
            twiss_df_b2: DataFrame containing the twiss data for beam 2.
            variables_dict: Dictionary containing Mad-X variables.

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
        pm.check_separations_against_madvars(
            separations_to_check, twiss_df_b1, twiss_df_b2, variables_dict
        )

    def check_betas(self, beam: str, twiss_df: pd.DataFrame, variables_dict: Dict):
        """
        Checks betas against tolerances specified in the constructor.

        Args:
            beam: Which beam to check. Must be a single number; either 1, 2 or 4.
            twiss_df: DataFrame containing twiss data for the given beam.
            variables_dict: Dictionary containing Mad-X variables.

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
        pm.check_twiss_against_madvars(twiss_value_checks, twiss_df, variables_dict)

    def __call__(self, mad: pm.Madxp, sequences: List[str], twiss_name: str = "twiss"):
        """
        Method that runs a check.

        Args:
            mad: Mad-X instance to check.
            sequences: List of sequences to check.
            twiss_name: The name of the saved twiss table.

        Returns:
            tuple: Tuple containing:

                twiss_df: Twiss generated during checks.
                other_data: Dictionary containing the summ dataframe and all Mad-X variables.

        """
        if not self.beta and not self.separation:
            return tuple()
            
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
            if not twiss_name.startswith("/"):
                if "/" in twiss_name:
                    twiss_name = FailSim.path_to_cwd(twiss_name)
                else:
                    twiss_name = FailSim.path_to_output(twiss_name)

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
