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
        self._sequence_key = None
        self._run_version = None
        self._hllhc_version = None
        self._sequence_paths = None
        self._optics_base_path = None
        self._sequence_data = None
        self._custom_optics = None
        self._optics_key = None
        self._optics_path = None
        self._beam_to_configure = None
        self._sequences_to_check = None
        self._sequence_to_track = None
        self._generate_b4_from_b2 = None
        self._track_from_b4_mad_instance = None
        self._enable_bb_python = None
        self._enable_bb_legacy = None
        self._force_disable_check_separations_at_ips = None
        self._mask_path = None
        self._do_cycle = None

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
            self._sequence_key = sequence

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
            self._optics_key = optics
        return self

    @_print_info
    def load_sequences_and_optics(self):
        """TODO: Docstring for load_sequences_and_optics.
        Returns: TODO

        """
        if not self._custom_sequence:
            sequence_data = self._metadata[self._sequence_key]
            self._run_version = sequence_data['run']
            self._hllhc_version = sequence_data['version']
            self._optics_base_path = sequence_data['optics_base_path']

            rel_paths = sequence_data['sequence_filenames']
            self._sequence_paths = [
                pkg_resources.resource_filename(__name__, x)
                for x in rel_paths
            ]

        if not self._custom_optics:
            assert (
                not self._custom_sequence
            ), "You can't use non-custom optics with custom sequence"

            rel_path = os.path.join(
                    self._optics_base_path,
                    sequence_data['optics'][self._optics_key]['strength_file']
                )
            self._optics_path = pkg_resources.resource_filename(
                __name__,
                rel_path
            )

        for seq in self._sequence_paths:
            self._failsim.mad_call_file(seq)
        self._failsim.mad_call_file(self._optics_path)

        self._failsim.mad_input(f"ver_lhc_run = {self._run_version}")
        self._failsim.mad_input(f"ver_hllhc_run = {self._hllhc_version}")

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

    @_print_info
    def make_thin(self):
        """TODO: Docstring for make_thin.
        Returns: TODO

        """
        twiss_df, summ_df = self._failsim.twiss_and_summ(self._sequence_to_track)

    @_print_info
    def cycle(self):
        """TODO: Docstring for cycle.
        Returns: TODO

        """
        pass

    @_print_info
    def set_mask_parameter_path(self, path: str):
        """TODO: Docstring for set_mask_parameter_path.

        Args:
            path (TODO): TODO

        Returns: TODO

        """
        if path.startswith("/"):
            self._mask_path = path
        else:
            self._mask_path = os.path.join(self._cwd, path)

    @_print_info
    def load_mask_parameters(self):
        """TODO: Docstring for load_mask_parameters.
        Returns: TODO

        """
        if self._mask_path is None:
            self._mask_path = os.path.join(self._cwd, "./mask_parameters.py")

        with open(self._mask_path, "r") as fd:
            mask_parameters = yaml.safe_load(fd)

        if not self._enable_bb_legacy and not self._enable_bb_python:
            mask_parameters["par_on_bb_switch"] = 0.0

        pm.checks_on_parameter_dict(mask_parameters)

        self._failsim._mad.set_variables_from_dict(params=mask_parameters)

    @_print_info
    def set_knob_parameter_path(self, path: str):
        """TODO: Docstring for set_knob_parameter_path.

        Args:
            path (TODO): TODO

        Returns: TODO

        """
        if path.startswith("/"):
            self._knob_path = path
        else:
            self._knob_path = os.path.join(self._cwd, path)

    @_print_info
    def load_knob_parameters(self):
        """TODO: Docstring for load_knob_parameters.
        Returns: TODO

        """
        if self._knob_path is None:
            self._knob_path = os.path.join(self._cwd, "./knob_parameters.py")

        with open(self._knob_path, "r") as fd:
            knob_parameters = yaml.safe_load(fd)

        # PATCH!!!!!!! for leveling not working for b4
        # Copied from optics_specific_tools example of pymask
        if self._beam_mode == "b4_without_bb":
            knob_parameters["par_sep8"] = -0.03425547139366354
            knob_parameters["par_sep2"] = 0.14471680504084292

        self._failsim._mad.set_variables_from_dict(params=knob_parameters)

        # Set IP knobs
        self._failsim._mad.globals["on_x1"] = knob_parameters["par_x1"]
        self._failsim._mad.globals["on_sep1"] = knob_parameters["par_sep1"]
        self._failsim._mad.globals["on_x2"] = knob_parameters["par_x2"]
        self._failsim._mad.globals["on_sep2"] = knob_parameters["par_sep2"]
        self._failsim._mad.globals["on_x5"] = knob_parameters["par_x5"]
        self._failsim._mad.globals["on_sep5"] = knob_parameters["par_sep5"]
        self._failsim._mad.globals["on_x8"] = knob_parameters["par_x8"]
        self._failsim._mad.globals["on_sep8"] = knob_parameters["par_sep8"]
        self._failsim._mad.globals["on_a1"] = knob_parameters["par_a1"]
        self._failsim._mad.globals["on_o1"] = knob_parameters["par_o1"]
        self._failsim._mad.globals["on_a2"] = knob_parameters["par_a2"]
        self._failsim._mad.globals["on_o2"] = knob_parameters["par_o2"]
        self._failsim._mad.globals["on_a5"] = knob_parameters["par_a5"]
        self._failsim._mad.globals["on_o5"] = knob_parameters["par_o5"]
        self._failsim._mad.globals["on_a8"] = knob_parameters["par_a8"]
        self._failsim._mad.globals["on_o8"] = knob_parameters["par_o8"]
        self._failsim._mad.globals["on_crab1"] = knob_parameters["par_crab1"]
        self._failsim._mad.globals["on_crab5"] = knob_parameters["par_crab5"]
        self._failsim._mad.globals["on_disp"] = knob_parameters["par_on_disp"]

        # A check
        if self._failsim._mad.globals.nrj < 500:
            assert knob_parameters["par_on_disp"] == 0

        # Spectrometers at experiments
        if knob_parameters["par_on_alice"] == 1:
            self._failsim._mad.globals.on_alice = (
                7000.0 / self._failsim._mad.globals.nrj
            )
        if knob_parameters["par_on_lhcb"] == 1:
            self._failsim._mad.globals.on_lhcb = 7000.0 / self._failsim._mad.globals.nrj

        # Solenoids at experiments
        self._failsim._mad.globals.on_sol_atlas = knob_parameters["par_on_sol_atlas"]
        self._failsim._mad.globals.on_sol_cms = knob_parameters["par_on_sol_cms"]
        self._failsim._mad.globals.on_sol_alice = knob_parameters["par_on_sol_alice"]

    @_print_info
    def call_pymask_module(self, module: str):
        """TODO: Docstring for call_pymask_module.

        Args:
            module (TODO): TODO

        Returns: TODO

        """
        path = pkg_resources.resource_filename(
            "pymask", "../" + module
        )
        self._failsim.mad_call_file(path)

    @_print_info
    def build(self):
        """TODO: Docstring for build.
        Returns: TODO

        """
        self.load_sequences_and_optics()
        self.call_pymask_module("submodule_01a_preparation.madx")
        self.call_pymask_module("submodule_01b_beam.madx")
        self._failsim.use(self._sequence_to_track)
        self.make_thin()
        if self._do_cycle:
            self.cycle()
        self.load_mask_parameters()
        self.load_knob_parameters()
