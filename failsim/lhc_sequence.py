"""
Module containing the class LHCSequence.
"""

from .failsim import FailSim
from .checks import OpticsChecks
from .sequence_tracker import SequenceTracker
from .globals import FSGlobals
from .helpers import print_info

from typing import Optional, List, Union, Dict
import pymask as pm
import functools
import pkg_resources
import yaml
import os


class LHCSequence:

    """
    Class for handling a Mad-X sequence.
    This class is used to setup and alter a sequence, after which build_tracker can be called to freeze the sequence.

    Note:
        If either sequence_key or optics_key isn't specified, initial_build has to be called after select_sequence or select_optics has been called respectively.

    Args:
        beam_mode: Selects the beam mode to use. Available modes can be found [here](http://lhcmaskdoc.web.cern.ch/pymask/#selecting-the-beam-mode).
        sequence_key: The sequence to use. Must be one of the sequence keys found in metadata.yaml. If sequence_key isn't specified, select_sequence must be called later.
        optics_key: The optics to use. Can only be used if a sequence key has been specified. Must be one of the optics keys under the selected sequence found in metadata.yaml. If optics_key isn't specified, select_optics must be called later.
        check_betas_at_ips: Whether run_check checks beta tolerances.
        check_separations_at_ips: Whether run_check checks separation tolerances.
        tolerances_beta: The beta function tolerances used by run_check. The values must correspond to: [IP1, IP2, IP5, IP8].
        tolerances_seperation: The separation tolerances used by run_check. The values must correspond to: [IP1, IP2, IP5, IP8].
        failsim: The FailSim instance to use. If failsim is None, LHCSequence will initialize a default instance.
        verbose: Whether LHCSequence outputs a message each time a method is called.
        input_param_path: Allows specification of the path to the input_parameters.yaml file. If no path is given, LHCSequence will assume that input_parameters.yaml is in the cwd.

    """

    def __init__(
        self,
        beam_mode: str,
        sequence_key: Optional[str] = None,
        optics_key: Optional[str] = None,
        check_betas_at_ips: bool = True,
        check_separations_at_ips: bool = True,
        tolerances_beta: List[float] = [1e-3, 10e-2, 1e-3, 1e-2],
        tolerances_separation: List[float] = [1e-6, 1e-6, 1e-6, 1e-6],
        failsim: Optional[FailSim] = None,
        verbose: bool = False,
        input_param_path: Optional[str] = None,
    ):
        self._beam_mode = beam_mode
        self._check_betas_at_ips = check_betas_at_ips
        self._check_separations_at_ips = check_separations_at_ips
        self._tolerances_beta = tolerances_beta
        self._tolerances_separation = tolerances_separation
        self._verbose = verbose
        self._input_param_path = input_param_path

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
        self._do_cycle = None
        self._sequences_to_cycle = None
        self._cycle_target = None

        self._load_metadata()
        self._init_check()
        self._get_mode_configuration()

        if failsim is None:
            self._failsim = FailSim()
        else:
            self._failsim = failsim

        if sequence_key is not None:
            self.select_sequence(sequence_key)

        if optics_key is not None:
            self.select_optics(optics_key)

        if sequence_key is not None and optics_key is not None:
            self.initial_build()

    @print_info("LHCSequence")
    def _get_mode_configuration(self):
        """Loads the mode configuration.

        Note:
            This function is automatically called by the constructor and isn't meant to be called by the user.

        Returns:
            LHCSequence: Returns self

        """
        self._mode_configuration = {}

        (
            self._mode_configuration["beam_to_configure"],
            self._mode_configuration["sequences_to_check"],
            self._mode_configuration["sequence_to_track"],
            self._mode_configuration["generate_b4_from_b2"],
            self._mode_configuration["track_from_b4_mad_instance"],
            self._mode_configuration["enable_bb_python"],
            self._mode_configuration["enable_bb_legacy"],
            self._mode_configuration["force_disable_check_separations_at_ips"],
        ) = pm.get_pymask_configuration(self._beam_mode)

        if self._mode_configuration["force_disable_check_separations_at_ips"]:
            self._check_separations_at_ips = False
            self._check.check_separations = False

    @print_info("LHCSequence")
    def _init_check(self):
        """Initializes the internal OpticsChecks instance.

        Note:
            This function is automatically called by the constructor and isn't meant to be called by the user.

        Returns:
            LHCSequence: Returns self

        """
        self._check = OpticsChecks(
            separation=self._check_separations_at_ips,
            beta=self._check_betas_at_ips,
            tol_sep=self._tolerances_separation,
            tol_beta=self._tolerances_beta,
        )

        return self

    @print_info("LHCSequence")
    def _load_input_parameters(self):
        """Loads input_parameters.yaml.

        Note:
            This function is automatically called by initial_build and isn't meant to be called by the user.

        Returns:
            Dict: Returns a dictionary containing the input parameters.

        """
        if self._input_param_path is None:
            self._input_param_path = self._failsim.path_to_cwd(
                "./input_parameters.yaml"
            )

        with open(self._input_param_path, "r") as fd:
            input_parameters = yaml.safe_load(fd)

        return input_parameters

    @print_info("LHCSequence")
    def _load_mask_parameters(self, mask_parameters: Dict):
        """Loads mask_parameters.yaml.

        Note:
            This function is automatically called by _load_input_parameters and isn't meant to be called by the user.

        Args:
            mask_parameters: A dict containing the mask parameters.

        Returns:
            LHCSequence: Returns self

        """
        pm.checks_on_parameter_dict(mask_parameters)

        self._failsim._mad.set_variables_from_dict(params=mask_parameters)

        return self

    @print_info("LHCSequence")
    def _load_knob_parameters(self, knob_parameters: Dict):
        """Loads knob_parameters.yaml.

        Note:
            This function is automatically called by _load_input_parameters and isn't meant to be called by the user.

        Args:
            knob_parameters: A dict containing the knob parameters.

        Returns:
            LHCSequence: Returns self

        """
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

        return self

    @print_info("LHCSequence")
    def _load_metadata(self):
        """Loads the metadata.yaml file.

        Returns:
            LHCSequence: Returns self

        """
        metadata_stream = pkg_resources.resource_stream(__name__, "data/metadata.yaml")
        self._metadata = yaml.safe_load(metadata_stream)

        return self

    @print_info("LHCSequence")
    def select_sequence(
        self,
        sequence: Union[str, List[str]],
        run_version: Optional[int] = None,
        hllhc_version: Optional[float] = None,
    ):
        """Sets the selected sequence. Can be either a sequence found in metadata.yaml by specifying a valid key, or a custom list of .madx files containing sequence definitions.

        Note:
            If a list of .madx files is passed to sequence, either run_version or hllhc_version has to be specified. These variables are used internally by pymask. Note that specifying both parameters will lead to an AssertionError.

        Args:
            sequence: Can either be a sequence key or a list of .madx files.
            run_version: The LHC run that is being used.
            hllhc_version: The HLLHC version that is being used.

        Returns:
            LHCSequence: Returns self

        """
        if type(sequence) is str:
            self._custom_sequence = False
            self._sequence_key = sequence

        else:
            assert (
                run_version or hllhc_version
            ), "You must specify either run or hllhc version when using custom sequence"
            assert not (
                run_version and hllhc_version
            ), "You cant specify both run and hllhc version when using custom sequence"

            self._custom_sequence = True

            self._run_version = run_version
            self._hllhc_version = hllhc_version
            self._sequence_paths = sequence

        return self

    @print_info("LHCSequence")
    def select_optics(self, optics: str, custom: bool = False):
        """Sets the selected optics. The selected optics can either be an optics key found in metadata.yaml by specifying a valid key, or a custom optics strength file by specifying a path.

        Args:
            optics: Can either be an optics key or the path of a .madx file.
            custom: Chooses whether the optics parameter is interpreted as a key or a path. If custom is True, optics is interpreted as a key, if custom if False, optics is interpreted as a path.

        Returns:
            LHCSequence: Returns self

        """
        self._custom_optics = custom
        if custom:
            if not optics.startswith("/"):
                optics = self._failsim.path_to_cwd(optics)
            self._optics_path = optics
        else:
            self._optics_key = optics

        return self

    @print_info("LHCSequence")
    def initial_build(self):
        """Does the initial build of the sequence.

        Note:
            Specifically does the following:

            1. If a sequence key has been specified, loads the relevant sequence data.
            1. If an optics key has been specified, loads the relevant optics data.
            1. Calls all sequence files sequentially.
            1. Calls optics strength file.
            1. Inputs *mylhcbeam*, *ver_lhc_run* and *ver_hllhc_optics* into the Mad-X instance.
            1. Loads mask_parameters.yaml.
            1. Calls *submodule_01a_preparation.madx* and *submodule_01b_beam.madx*
                - These set basic internal Mad-X variables and define the beam.
            1. Makes all sequences thin.
            1. Loads knob_parameters.yaml

        Returns:
            LHCSequence: Returns self

        """
        if not self._custom_sequence:
            sequence_data = self._metadata[self._sequence_key]
            self._run_version = sequence_data["run"]
            self._hllhc_version = sequence_data["version"]
            self._optics_base_path = sequence_data["optics_base_path"]

            rel_paths = sequence_data["sequence_filenames"]
            self._sequence_paths = [
                pkg_resources.resource_filename(__name__, x) for x in rel_paths
            ]

        if not self._custom_optics:
            assert (
                not self._custom_sequence
            ), "You can't use non-custom optics with custom sequence"

            rel_path = os.path.join(
                self._optics_base_path,
                sequence_data["optics"][self._optics_key]["strength_file"],
            )
            self._optics_path = pkg_resources.resource_filename(__name__, rel_path)

        for seq in self._sequence_paths:
            self._failsim.mad_call_file(seq)
        self._failsim.mad_call_file(self._optics_path)

        self._failsim.mad_input(
            f"mylhcbeam = {self._mode_configuration['beam_to_configure']}"
        )
        self._failsim.mad_input(f"ver_lhc_run = {self._run_version}")
        self._failsim.mad_input(f"ver_hllhc_optics = {self._hllhc_version}")

        input_parameters = self._load_input_parameters()

        self._load_mask_parameters(input_parameters["mask_parameters"])
        self._failsim.call_pymask_module("submodule_01a_preparation.madx")
        self._failsim.call_pymask_module("submodule_01b_beam.madx")
        for seq in self._mode_configuration["sequences_to_check"]:
            self._failsim.make_thin(seq[-1])
        self._load_knob_parameters(input_parameters["knob_parameters"])

        return self

        return self

    @print_info("LHCSequence")
    def run_check(self):
        """Runs a check using the internal OpticsChecks instance.

        Returns:
            LHCSequence: Returns self

        """
        self._check(
            mad=self._failsim._mad,
            sequences=self._mode_configuration["sequences_to_check"],
        )

        return self

    @print_info("LHCSequence")
    def call_pymask_module(self, module: str):
        """Calls a pymask module using the internal Mad-X instance.

        Args:
            module: The name of the module to call.

        Returns:
            LHCSequence: Returns self

        """
        self._failsim.call_pymask_module(module)

        return self

    @print_info("LHCSequence")
    def cycle(self, sequences: List[str], target: str):
        """Cycles the specified sequences to start at the target element.

        Args:
            sequences: A list of sequences to cycle.
            target: The name of the element the sequences should start at.

        Returns:
            LHCSequence: Returns self

        """
        for seq in sequences:
            self._failsim.mad_input(
                f"seqedit, sequence={seq}; flatten; cycle, start={target}; flatten; endedit"
            )

        return self

    @print_info("LHCSequence")
    def set_input_parameter_path(self, path: str):
        """Sets the input_parameters.yaml path.

        Args:
            path: The path to input_parameters.yaml. Can be either absolute or relative.

        Returns:
            LHCSequence: Returns self

        """
        if path.startswith("/"):
            self._input_param_path = path
        else:
            self._input_param_path = self._failsim.path_to_cwd(path)

        return self

    @print_info("LHCSequence")
    def build_tracker(self, verbose: bool = False):
        """Builds a SequenceTracker instance.

        Args:
            verbose: Whether the generated SequenceTracker object should be verbose or not.

        Returns:
            SequenceTracker: A SequenceTracker instance containing this sequence.

        """
        new_fs = self._failsim.duplicate()
        tracker = SequenceTracker(
            new_fs, self._mode_configuration["sequence_to_track"], verbose=verbose
        )
        return tracker

    @print_info("LHCSequence")
    def crossing_save(self):
        """Saves the current crossing settings in internal Mad-X variables.

        Returns:
            LHCSequence: Returns self

        """
        self._failsim.mad_input("exec, crossing_save")

        return self

    @print_info("LHCSequence")
    def set_crossing(self, crossing_on: bool):
        """Either enables or disables crossing depending on the crossing_on parameter.

        Note:
            Crossing can only be enabled is crossing_save was called before the crossings were disabled.

        Args:
            crossing_on: Turns crossings on if True, turns crossings off if False.

        Returns:
            LHCSequence: Returns self

        """
        state = "restore" if crossing_on else "disable"
        self._failsim.mad_input(f"exec, crossing_{state}")

        return self

    @print_info("LHCSequence")
    def call_file(self, file_path: str):
        """Forwards the file_path to failsim.mad_call_file.

        Args:
            file_path: The path of the file to call. Can be either absolute or relative.

        Returns:
            LHCSequence: Returns self

        """
        self._failsim.mad_call_file(file_path)

        return self

    @print_info("LHCSequence")
    def call_files(self, file_paths: List[str]):
        """Calls multiple files using the internal Mad-X instance.

        Args:
            file_paths: A list of paths to call. The paths can be either absolute or relative.

        Returns:
            LHCSequence: Returns self

        """
        for file in file_paths:
            self.call_file(file)

        return self
