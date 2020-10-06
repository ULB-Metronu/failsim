## Moved the load_metadata to LHCSequence
## LHCSequence has to call madx before build to be able to run checks

from typing import Optional, List, Union
from .failsim import FailSim
from .checks import OpticsChecks
import pymask as pm
import functools
import pkg_resources
import yaml
import os


class LHCSequence:

    """Docstring for LHCSequence. """

    def __init__(
        self,
        beam_mode: str,
        sequence_key: Optional[str] = None,
        optics_key: Optional[str] = None,
        check_betas_at_ips: bool = True,
        check_separations_at_ips: bool = True,
        tolerances_beta: List[float] = [1, 1, 1, 1],
        tolerances_seperation: List[float] = [1, 1, 1, 1],
        failsim: Optional[FailSim] = None,
        verbose: bool = True,
    ):
        """TODO: to be defined.

        Args:
            beam_mode (TODO): TODO

        Kwargs:
            sequence_key (TODO): TODO
            optics_key (TODO): TODO
            check_betas_at_ips (TODO): TODO
            check_separations_at_ips (TODO): TODO
            tolerances_beta (TODO): TODO
            tolerances_seperation (TODO): TODO
            failsim (TODO): TODO
            verbose (TODO): TODO

        """
        self._beam_mode = beam_mode
        self._check_betas_at_ips = check_betas_at_ips
        self._check_separations_at_ips = check_separations_at_ips
        self._tolerances_beta = tolerances_beta
        self._tolerances_seperation = tolerances_seperation
        self._verbose = verbose

        self._metadata = None
        self._custom_sequence = None
        self._run_version = None
        self._hllhc_version = None
        self._sequence_paths = None
        self._optics_base_path = None
        self._sequence_data = None
        self._custom_optics = None
        self._optics_path = None
        self._beam_to_configure = None
        self._sequences_to_check = None
        self._sequence_to_track = None
        self._generate_b4_from_b2 = None
        self._track_from_b4_mad_instance = None
        self._enable_bb_python = None
        self._enable_bb_legacy = None
        self._force_disable_check_separations_at_ips = None

        self.load_metadata()
        self.init_check()
        self.get_mode_configuration()

        if sequence_key is not None:
            self.select_sequence(sequence_key)

        if optics_key is not None:
            self.select_optics(optics_key)

        if failsim is None:
            self._failsim = FailSim()
        else:
            self._failsim = failsim

    def _print_info(func):
        """Decorator to print LHCSequence debug information"""

        @functools.wraps(func)
        def wrapper_print_info(self, *args, **kwargs):
            if self._verbose:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                print(f"LHCSequence -> {func.__name__}({signature})")
            func(self, *args, **kwargs)

        return wrapper_print_info

    @_print_info
    def load_metadata(self):
        """TODO: Docstring for load_metadata.
        Returns: TODO

        """
        metadata_stream = pkg_resources.resource_stream(__name__, "data/metadata.yaml")
        self._metadata = yaml.safe_load(metadata_stream)

        return self

    @_print_info
    def select_sequence(
        self,
        sequence: Union[str, List[str]],
        custom: bool = False,
        run_version: Optional[int] = None,
        hllhc_version: Optional[float] = None,
        optics_base_path: Optional[str] = None,
    ):
        """TODO: Docstring for select_sequence.

        Args:
            sequence (TODO): TODO

        Kwargs:
            custom (TODO): TODO
            run_version (TODO): TODO
            hllhc_version (TODO): TODO
            optics_base_path (TODO): TODO

        Returns: TODO

        """
        if custom:
            assert (
                run_version or hllhc_version
            ), "You must specify either run or hllhc version when using custom sequence"
            assert not (
                run_version and hllhc_version
            ), "You cant specify both run and hllhc version when using custom sequence"
            assert (
                optics_base_path
            ), "You must specify an optics base path when using custom sequence"

            self._custom_sequence = True

            self._run_version = run_version
            self._hllhc_version = hllhc_version
            self._sequence_paths = sequence
            self._optics_base_path = optics_base_path

        else:
            self._custom_sequence = False

            self._sequence_data = self._metadata[sequence]

            rel_paths = self._sequence_data["sequence_filenames"]
            self._sequence_paths = [
                pkg_resources.resource_filename(__name__, x) for x in rel_paths
            ]

            self._run_version = self._sequence_data["run"]
            self._hllhc_version = self._sequence_data["version"]
            self._optics_base_path = self._sequence_data["optics_base_path"]

        return self

    @_print_info
    def select_optics(self, optics: str, custom: bool = False):
        """TODO: Docstring for select_optics.

        Args:
            optics (TODO): TODO

        Kwargs:
            custom (TODO): TODO

        Returns: TODO

        """
        if custom:
            self._custom_optics = True
            self._optics_path = optics
        else:
            self._custom_optics = False

            rel_path = os.path.join(
                self._optics_base_path,
                self._sequence_data["optics"][optics]["strength_file"],
            )
            self._optics_path = pkg_resources.resource_filename(__name__, rel_path)
        return self

    @_print_info
    def load_sequences_and_optics(self):
        """TODO: Docstring for load_sequences_and_optics.
        Returns: TODO

        """
        for seq in self._sequence_paths:
            self._failsim.mad_call_file(seq)
        self._failsim.mad_call_file(self._optics_path)

        self._failsim.mad_input(f'ver_lhc_run = {self._run_version}')
        self._failsim.mad_input(f'ver_hllhc_run = {self._hllhc_version}')

    @_print_info
    def get_mode_configuration(self):
        """TODO: Docstring for get_mode_configuration.
        Returns: TODO

        """
        (
            self._beam_to_configure,
            self._sequences_to_check,
            self._sequence_to_track,
            self._generate_b4_from_b2,
            self._track_from_b4_mad_instance,
            self._enable_bb_python,
            self._enable_bb_legacy,
            self._force_disable_check_separations_at_ips,
        ) = pm.get_pymask_configuration(self._beam_mode)

        if self._force_disable_check_separations_at_ips:
            self._check_separations_at_ips = False
            self._check.check_separations = False

    @_print_info
    def init_check(self):
        """TODO: Docstring for init_check.
        Returns: TODO

        """
        self._check = OpticsChecks(
            separation=self._check_separations_at_ips,
            beta=self._check_betas_at_ips,
            tol_sep=self._tolerances_seperation,
            tol_beta=self._tolerances_beta,
        )

        return self

    @_print_info
    def run_check(self):
        """TODO: Docstring for run_check.
        Returns: TODO

        """
        self._check(mad=self._failsim._mad, sequences=self._sequences_to_check)

        return self
