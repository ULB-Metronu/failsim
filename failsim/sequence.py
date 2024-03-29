"""
Module containing the class LHCSequence.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Union, Dict, Callable
import re
import os
import glob
import functools
import pkg_resources
import yaml
import numpy as np
import pandas as pd
import pymask as pm
from .failsim import FailSim
from .checks import OpticsChecks
from .tracker import Tracker
from .helpers import print_info
from .results import TwissResult
from .aperture import Aperture


def reset_state(build: bool, check: bool):
    def inner_reset_state(func):
        @functools.wraps(func)
        def wrapper_reset_state(self, *args, **kwargs):
            self._built = False if build else self._built
            self._checked = False if check else self._checked
            val = func(self, *args, **kwargs)
            return val

        return wrapper_reset_state

    return inner_reset_state


def ensure_build(func):
    @functools.wraps(func)
    def wrapper_ensure_build(self, *args, **kwargs):
        if not self._built:
            self.build()
        if not self._checked:
            self.run_check()
        val = func(self, *args, **kwargs)
        return val

    return wrapper_ensure_build


class LHCSequence:
    """
    Class for handling a MAD-X sequence.
    This class is used to setup and alter a sequence, after which build_tracker can be called to freeze the sequence.

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
        failsim_verbose: Sets the verbosity of the FailSim instance.
        madx_verbose: Sets the verbosity of MAD-X itself (false is equivalent to the 'mute' option in FailSim).
        input_param_path: Allows specification of the path to the input_parameters.yaml file. If no path is given, LHCSequence will assume that input_parameters.yaml is in the cwd.

    """

    def __init__(
        self,
        beam_mode: str,
        sequence_key: Optional[str] = None,
        optics_key: Optional[str] = None,
        collimation_key: Optional[str] = None,
        check_betas_at_ips: bool = True,
        check_separations_at_ips: bool = True,
        tolerances_beta: List[float] = [1e-3, 10e-2, 1e-3, 1e-2],
        tolerances_separation: List[float] = [1e-6, 1e-6, 1e-6, 1e-6],
        failsim: Optional[FailSim] = None,
        verbose: bool = False,
        failsim_verbose: bool = False,
        madx_verbose: bool = False,
        input_param_path: Optional[str] = None,
    ):
        self._beam_mode = beam_mode
        self._check_betas_at_ips = check_betas_at_ips
        self._check_separations_at_ips = check_separations_at_ips
        self._tolerances_beta = tolerances_beta
        self._tolerances_separation = tolerances_separation
        self._verbose = verbose
        self._input_param_path = input_param_path

        self._modules = {}
        self._pre_thin_scripts = []

        self._metadata = None
        self._custom_sequence = None
        self._sequence_key = None
        self._run_version = None
        self._hllhc_version = None
        self._sequence_paths = None
        self._optics_base_path = None
        self._sequence_data = None
        self._optics_key = None
        self._optics_path = None
        self._apertures = None
        self._thin_apertures = None
        self.collimation = None
        self._collimation_key = None
        self._collimation_path = None
        self._cycle = None
        self._bb_dfs = None
        self._twiss_pre_thin_paths = {}

        self._has_built_once = False
        self._built = False
        self._checked = False

        self._metadata = self.get_metadata()

        self._check = OpticsChecks(
            separation=self._check_separations_at_ips,
            beta=self._check_betas_at_ips,
            tol_sep=self._tolerances_separation,
            tol_beta=self._tolerances_beta,
        )

        self._get_mode_configuration()
        self._initialize_mask_dictionary()

        self._failsim = failsim
        self._failsim_verbose = failsim_verbose
        if madx_verbose:
            self._madx_verbose = ''
        else:
            self._madx_verbose = 'mute'

        if sequence_key is not None:
            self.select_sequence(sequence_key)

        if optics_key is not None:
            self.select_optics(optics_key)

        if collimation_key is not None:
            self.select_collimation(collimation_key)

        self._aperture_model = None

    @property
    def mad(self):
        return self._failsim.mad

    @property
    def configuration(self) -> Dict:
        return self._mode_configuration

    def _initialize_mask_dictionary(self):
        """Fills the internal dictionary _modules with default values for each pymask module.

        Note:
            The dictionary is set up in such a way, that each module has 3 aliases with which the values of the given module can be accessed with. For example, the module submodule_01a_preparation.madx can be accessed with the module number "01a", the module number and name "01a_preparation" or the entire module filename "submodule_01a_preparation.madx".
        """
        module_dir = pkg_resources.resource_filename(__name__, "data/pymask")
        modules = glob.glob(os.path.join(module_dir, "*.madx"))

        # Regex to get module number and name
        re_num = re.compile(r"(?<=_)(?:\d+\w?_?)+(?=_)")
        re_name = re.compile(r"(?<=_)\D+(?=\.madx)")

        for module in modules:
            num = re_num.findall(module)[0]
            name = re_name.findall(module)[0]
            self._modules.update(
                dict.fromkeys(
                    [num, num + "_" + name, os.path.basename(module)],
                    {"enabled": False, "called": False, "path": module},
                )
            )

        # Set default modules
        self._modules["01a_preparation"]["enabled"] = True
        self._modules["01b_beam"]["enabled"] = True
        self._modules["01c_phase"]["enabled"] = True
        self._modules["01d_crossing"]["enabled"] = True
        self._modules["01e_final"]["enabled"] = True
        self._modules["05a_MO"]["enabled"] = True

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

    def _load_input_parameters(self):
        """Loads input_parameters.yaml.

        Note:
            This function is automatically called by build and isn't meant to be called by the user.

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

    def _load_mask_parameters(self, mask_parameters: Dict):
        """Loads mask_parameters.yaml.

        Note:
            This function is automatically called by _load_input_parameters and isn't meant to be called by the user.

        Args:
            mask_parameters: A dict containing the mask parameters.

        Returns:
            LHCSequence: Returns self

        """
        if (
            not self._mode_configuration["enable_bb_legacy"]
            and not self._mode_configuration["enable_bb_python"]
        ):
            mask_parameters["par_on_bb_switch"] = 0.0

        self._failsim.mad.set_variables_from_dict(params=mask_parameters)

        return self

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

        self._failsim.mad.set_variables_from_dict(params=knob_parameters)

        # Set IP knobs
        self.mad.globals["on_x1"] = knob_parameters["par_x1"]
        self.mad.globals["on_x2"] = knob_parameters["par_x2"]
        self.mad.globals["on_sep1"] = knob_parameters["par_sep1"]
        self.mad.globals["on_sep2"] = knob_parameters["par_sep2"]
        self.mad.globals["on_x5"] = knob_parameters["par_x5"]
        self.mad.globals["on_sep5"] = knob_parameters["par_sep5"]
        self.mad.globals["on_x8"] = knob_parameters["par_x8"]
        self.mad.globals["on_sep8"] = knob_parameters["par_sep8"]
        self.mad.globals["on_a1"] = knob_parameters["par_a1"]
        self.mad.globals["on_o1"] = knob_parameters["par_o1"]
        self.mad.globals["on_a2"] = knob_parameters["par_a2"]
        self.mad.globals["on_o2"] = knob_parameters["par_o2"]
        self.mad.globals["on_a5"] = knob_parameters["par_a5"]
        self.mad.globals["on_o5"] = knob_parameters["par_o5"]
        self.mad.globals["on_a8"] = knob_parameters["par_a8"]
        self.mad.globals["on_o8"] = knob_parameters["par_o8"]
        self.mad.globals["on_crab1"] = knob_parameters["par_crab1"]
        self.mad.globals["on_crab5"] = knob_parameters["par_crab5"]
        self.mad.globals["on_disp"] = knob_parameters["par_on_disp"]

        # A check
        if self._failsim.mad.globals.nrj < 500:
            assert knob_parameters["par_on_disp"] == 0

        # Spectrometers at experiments
        if knob_parameters["par_on_alice"] == 1:
            self._failsim.mad.globals.on_alice = (
                7000.0 / self._failsim.mad.globals.nrj
            )
        if knob_parameters["par_on_lhcb"] == 1:
            self._failsim.mad.globals.on_lhcb = 7000.0 / self._failsim.mad.globals.nrj

        # Solenoids at experiments
        self._failsim.mad.globals.on_sol_atlas = knob_parameters["par_on_sol_atlas"]
        self._failsim.mad.globals.on_sol_cms = knob_parameters["par_on_sol_cms"]
        self._failsim.mad.globals.on_sol_alice = knob_parameters["par_on_sol_alice"]

        return self

    def _check_call_module(self, module: str):
        """Checks whether the given module is enabled, and whether the module has already been called. If the module is enabled and hasn't been called yet, the module is called.

        Args:
            module: String containing a key to the module to check.

        """
        if self._modules[module]["enabled"] and not self._modules[module]["called"]:
            if module == '05c_limit':  # Because that submodule calls a file with hardcoded path
                self.call_file(pkg_resources.resource_filename(
                    __name__, self._metadata[self._sequence_key]["optics_base_path"] + "errors/corr_limit.madx")
                )
            elif module == '05e_corrvalue':  # Because that submodule calls a file with hardcoded path
                self.call_file(pkg_resources.resource_filename(
                    __name__, self._metadata[self._sequence_key]["optics_base_path"] + "errors/corr_value_limit.madx")
                )
            else:
                self.call_pymask_module(os.path.basename(self._modules[module]["path"]))
            self._modules[module]["called"] = True

    def _set_modules_called(self, modules: List[str], called: bool = False):
        """ Sets the "called" flag for a list of modules.

        Args:
            modules (List[str]): List of modules to for which to set the "called" flag.

        Kwargs:
            called (bool): The boolean value to set for each module.

        Returns: None

        """
        for module in modules:
            self._modules[module]["called"] = called

    @classmethod
    def get_metadata(cls):
        return yaml.safe_load(
            pkg_resources.resource_stream(__name__, "data/metadata.yaml")
        )

    @classmethod
    def get_sequences(cls):
        return list(cls.get_metadata().keys())

    @classmethod
    def get_optics(cls, sequence_key):
        return list(cls.get_metadata()[sequence_key]["optics"].keys())

    @classmethod
    def get_collimations(cls, sequence_key):
        return list(cls.get_metadata()[sequence_key]['collimation'].keys())

    @classmethod
    def get_modules(cls):
        module_dir = pkg_resources.resource_filename(__name__, "data/pymask")
        modules = glob.glob(os.path.join(module_dir, "*.madx"))
        return [os.path.basename(x) for x in modules]

    @classmethod
    def get_collimator_handler(cls, sequence: str, collimation_key: str):
        """Returns CollimatorHandler object for given sequence and collimation key.

        Args:
            sequence: Which sequence key to use.
            collimation_key: Which collimation key to use.

        Returns:
            CollimatorHandler: Handler object loaded with proper collimation data.

        """
        metadata = cls.get_metadata()
        settings_path = os.path.join(
            metadata[sequence]["collimation_base_path"],
            metadata[sequence]["collimation"][collimation_key],
        )
        return CollimatorHandler(
            pkg_resources.resource_filename(__name__, settings_path)
        )

    def _load_and_set_collimators(self, _vertical_aperture: float = 10.0, with_pre_thin_twiss: bool = False):
        _tw = self.twiss()
        self.collimation = CollimatorHandler(self._collimation_path)
        openings = self.collimation.compute_openings(_tw)
        offsets = self.collimation.compute_offsets(_tw)
        settings = pd.concat([openings, offsets], axis=1)

        if with_pre_thin_twiss:
            twiss_thick = pd.read_parquet(
                self._twiss_pre_thin_paths[self._mode_configuration["sequence_to_track"]]
            )
        for _, row in settings.iterrows():
            if with_pre_thin_twiss:
                if row.name.lower() in twiss_thick.index:
                    twiss_thick.at[row.name.lower(), "aper_1"] = row['half_gap']
                    twiss_thick.at[row.name.lower(), "aper_2"] = _vertical_aperture
                    twiss_thick.at[row.name.lower(), "apoff_1"] = row['offset_x']
                    twiss_thick.at[row.name.lower(), "apoff_2"] = row['offset_y']
                    twiss_thick.at[row.name.lower(), "tilt"] = row['angle']

            self._failsim.mad_input(
                f"{row.name}, APERTYPE=RECTANGLE, "
                f"APER_OFFSET={{ {row['offset_x']}, {row['offset_y']} }}"
                f"APERTURE={{ {row['half_gap']}, {_vertical_aperture}, 0.0, 0.0 }}, TILT={row['angle']}"
            )
        if with_pre_thin_twiss:
            twiss_thick.to_parquet(
                self._twiss_pre_thin_paths[self._mode_configuration["sequence_to_track"]]
            )

    def _prepare_bb_dataframes(self):
        """ Prepares the beam-beam dataframes. """
        if self._mode_configuration["enable_bb_python"]:
            self._bb_dfs = pm.generate_bb_dataframes(
                self._failsim.mad,
                ip_names=["ip1", "ip2", "ip5", "ip8"],
                harmonic_number=35640,
                numberOfLRPerIRSide=[25, 20, 25, 20],
                bunch_spacing_buckets=10,
                numberOfHOSlices=11,
                bunch_population_ppb=None,
                sigmaz_m=None,
                z_crab_twiss=0,
                remove_dummy_lenses=True,
            )

    def _install_bb_lenses(self):
        """ Installs the beam-beam lenses if beam-beam has been enabled by mode. """
        # Python approach
        if self._mode_configuration["enable_bb_python"]:
            if self._mode_configuration["track_from_b4_mad_instance"]:
                bb_df_track = self._bb_dfs["b4"]
                assert self._mode_configuration["sequence_to_track"] == "lhcb2"
            else:
                bb_df_track = self._bb_dfs["b1"]
                assert self._mode_configuration["sequence_to_track"] == "lhcb1"

            pm.install_lenses_in_sequence(
                self._failsim.mad,
                bb_df_track,
                self._mode_configuration["sequence_to_track"],
            )

            # Disable bb (to be activated later)
            self._failsim.mad.globals.on_bb_charge = 0
        else:
            bb_df_track = None

        # Legacy bb macros
        if self._mode_configuration["enable_bb_legacy"]:
            assert self._mode_configuration["beam_to_configure"] == 1
            assert not (self._mode_configuration["track_from_b4_mad_instance"])
            assert not (self._mode_configuration["enable_bb_python"])
            self._failsim.mad_call_file("modules/module_03_beambeam.madx")

    def _sigma_to_length(self, beam_sigmas: float, element: str, axis: str = "x"):
        """
        Takes a length defined in beam sigmas, and returns the actual length at the given element.

        Args:
            sigma: Length expressed in beam sigmas.
            element: The element, at which the length should be calculated.
            axis: The axis of beta function to use.

        Returns:
            float: Length expressed in metres.

        """
        twiss = self.twiss()
        eps_g = twiss.info_df["eps_n"] / twiss.summ_df["gamma"]

        sigma = np.sqrt(eps_g * twiss.twiss_df.loc[element][f"bet{axis}"])
        return beam_sigmas * sigma

    @reset_state(True, True)
    @print_info("LHCSequence")
    def set_modules_enabled(self, modules: List[str], enabled: bool = True):
        """Either enables or disables a list of modules based upon the parameter "enabled".

        Args:
            modules: A list containing keys of the modules to either enable or disable.
            enabled: Whether the modules specified by "modules" should be enabled or disabled. True enables modules, while False disables.

        Returns:
            LHCSequence: Returns self.

        """
        for module in modules:
            assert module in self._modules.keys(), (
                f"Module {module} is not a valid module.\n"
                "Valid modules are:\n{}".format(
                    "\n".join([x for x in self._modules if x.endswith(".madx")])
                )
            )
            self._modules[module]["enabled"] = enabled

        return self

    @reset_state(True, True)
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

    @reset_state(True, True)
    @print_info("LHCSequence")
    def select_optics(self, optics: str, custom: bool = False):
        """Sets the selected optics. The selected optics can either be an optics key found in metadata.yaml by specifying a valid key, or a custom optics strength file by specifying a path.

        Args:
            optics: Can either be an optics key or the path of a .madx file.
            custom:

        Kwargs:
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

    @reset_state(True, True)
    @print_info("LHCSequence")
    def select_collimation(self, collimation: str, custom: bool = False):
        """Sets the collimation settings. The selected collimation can either be a collimation key found in metadata.yaml by specifying a valid key, or a custom collimation file by specifying a path.

        Args:
            collimation: Can either be a collimation key or the path to collimation file.
            custom:

        Kwargs:
            custom: Must be True if collimation is a path to a file, and False if collimation is a key.

        Returns:
            LHCSequence: Returns self.

        """
        self._custom_collimation = custom
        if custom:
            if not collimation.startswith("/"):
                collimation = self._failsim.path_to_cwd(collimation)
            self._collimation_path = collimation
        else:
            self._collimation_key = collimation

        return self

    @reset_state(True, True)
    @print_info("LHCSequence")
    def add_apertures(self):
        """Add the aperture definitions.

        The LHC reference aperture files are first loaded, and the sequence specific aperture files are then called.

        :return:
            LHCSequence: Returns self.
        """
        if self._run_version != 0 and self._metadata[self._sequence_key]['optics_base_path'].startswith('/'):
            base = self._metadata[self._sequence_key]["optics_base_path"]
            self._apertures = [
                os.path.join(base, 'aperture', 'aperture.b1.madx'),
                os.path.join(base, 'aperture', 'aperture.b2.madx'),
                os.path.join(base, 'aperture', 'aper_tol.b1.madx'),
                os.path.join(base, 'aperture', 'aper_tol.b2.madx'),
            ]
            for file in self._metadata[self._sequence_key]["aperture_files"]:
                self._apertures.append(
                    os.path.join(base, file),
                )
            self._thin_apertures = []
            for file in self._metadata[self._sequence_key]["thin_aperture_files"]:
                self._thin_apertures.append(
                    os.path.join(base, file),
                )
        else:
            self._apertures = [
                pkg_resources.resource_filename(__name__, "data/aperture/aperture.b1.madx"),
                pkg_resources.resource_filename(__name__, "data/aperture/aperture.b2.madx"),
                pkg_resources.resource_filename(__name__, "data/aperture/aper_tol.b1.madx"),
                pkg_resources.resource_filename(__name__, "data/aperture/aper_tol.b2.madx"),
            ]
            for file in self._metadata[self._sequence_key]["aperture_files"]:
                self._apertures.append(
                    pkg_resources.resource_filename(
                        __name__, self._metadata[self._sequence_key]["optics_base_path"] + file
                    )
                )
            self._thin_apertures = []
            for file in self._metadata[self._sequence_key]["thin_aperture_files"]:
                self._thin_apertures.append(
                    pkg_resources.resource_filename(
                        __name__, self._metadata[self._sequence_key]["optics_base_path"] + file
                    )
                )
        return self

    @print_info("LHCSequence")
    def build(self, thick: bool = False, with_pre_thin_twiss: bool = False):
        """Does the build of the sequence.

        Note:
            Specifically does the following:

            1. If a sequence key has been specified, loads the relevant sequence data.
            2. If an optics key has been specified, loads the relevant optics data.
            3. Calls all sequence files sequentially.
            4. Calls optics strength file.
            5. Inputs *mylhcbeam*, *ver_lhc_run* and *ver_hllhc_optics* into the Mad-X instance.
            6. Loads mask_parameters.yaml.
            7. Calls the submodules *01a_preparation* and *01b_beam* if these modules are enabled.
                - These set basic internal Mad-X variables and define the beam.
            8. Saves twiss table to use for cartouche plots (requires a thick Twiss table)
            9. Makes all sequences thin.
            10. Loads `knob_parameters.yaml`
            11. Cycles sequence if specified
            12. Runs any modules that have been enabled
            13. Calls the modules *01c_phase*, *01d_crossing*, *01e_final* and *02_lumilevel* if these modules are enabled.
            14. Installs BB lenses, if BB lenses have been specified.
            15. Checks each remaining module sequentially, and calls the module if it is both enabled and hasn't been called yet.

        Returns:
            LHCSequence: Returns self

        """
        BUILD_INFO_MESSAGE = "LHCSequence - build"
        self._built = True

        # If sequence has already been built once,
        # the madx instance has to be reset, and the
        # "called" flag for each module should be reset
        if self._has_built_once:
            self._failsim.reset_mad()
            self._set_modules_called(self.get_modules(), False)
        self._has_built_once = True

        @print_info(BUILD_INFO_MESSAGE)
        def step_01__load_macros(self):
            for macro in sequence_data['macros']:
                if macro.startswith('/'):
                    self._failsim.mad_call_file(macro)
                else:
                    self._failsim.mad_call_file(
                        pkg_resources.resource_filename(
                            __name__, sequence_data["optics_base_path"] + macro
                        )
                    )

        @print_info(BUILD_INFO_MESSAGE)
        def step_02__load_sequence(self):
            for seq in sequence_data["sequence_filenames"]:
                self._failsim.mad_call_file(pkg_resources.resource_filename(__name__, seq))

        @print_info(BUILD_INFO_MESSAGE)
        def step_03__load_apertures(self):
            if self._apertures is not None:
                for file in self._apertures:
                    self._failsim.mad_call_file(file)

        @print_info(BUILD_INFO_MESSAGE)
        def step_04__prepare(self):
            self._failsim.mad_input(f"mylhcbeam = {self._mode_configuration['beam_to_configure']}")
            self._failsim.mad_input(f"ver_lhc_run = {self._run_version}")
            self._failsim.mad_input(f"ver_hllhc_optics = {self._hllhc_version}")

            input_parameters = self._load_input_parameters()
            self._load_mask_parameters(input_parameters["mask_parameters"])
            self._check_call_module("01a_preparation")
            self._load_knob_parameters(input_parameters["knob_parameters"])

            self._check_call_module("01b_beam")
            import time
            unique_hash = hash(time.time()+float(id(self)))
            if with_pre_thin_twiss:
                for ss in self._mode_configuration["sequences_to_check"]:
                    twiss_df, _ = self._failsim.twiss_and_summ(ss)
                    self._twiss_pre_thin_paths[ss] = FailSim.path_to_output(
                        f"{unique_hash}_twiss_pre_thin_{ss}.parquet"
                    )
                    twiss_df.to_parquet(self._twiss_pre_thin_paths[ss])

                    survey_df = self._failsim.mad.survey(sequence=ss).dframe()
                    survey_df.to_parquet(
                        FailSim.path_to_output(f"{unique_hash}_survey_pre_thin_{ss}.parquet")
                    )

        @print_info(BUILD_INFO_MESSAGE)
        def step_05__cycle(self):
            if self._cycle is not None:
                for seq in self._cycle["sequences"]:
                    self._failsim.mad_input(
                        f"seqedit, sequence={seq}; flatten; cycle, start={self._cycle['target']}; flatten; endedit"
                    )

        @print_info(BUILD_INFO_MESSAGE)
        def step_06__thin(self):
            if not thick:
                for ff in self._pre_thin_scripts:
                    self._failsim.mad_call_file(ff)
                self._failsim.make_thin(make_end_markers=True)

                if self._thin_apertures is not None:
                    for file in self._thin_apertures:
                        self._failsim.mad_call_file(file)

        @print_info(BUILD_INFO_MESSAGE)
        def step_07__load_optics(self):
            rel_path = os.path.join(
                self._optics_base_path,
                sequence_data["optics"][self._optics_key]["strength_file"],
            )
            if self._optics_base_path.startswith("/"):
                self._optics_path = rel_path
            else:
                self._optics_path = pkg_resources.resource_filename(__name__, rel_path)
            self._failsim.mad_call_file(self._optics_path)

        @print_info(BUILD_INFO_MESSAGE)
        def step_08__mask_module_01_orbit(self):
            self._check_call_module("01c_phase")
            self._check_call_module("01d_crossing")
            self._check_call_module("01e_final")
            self._failsim.mad_input("exec, twiss_opt;")

        @print_info(BUILD_INFO_MESSAGE)
        def step_09__mask_module_02_lumilevel(self):
            self._check_call_module("02_lumilevel")

        @print_info(BUILD_INFO_MESSAGE)
        def step_10__mask_module_03_beambeam(self):
            self._prepare_bb_dataframes()
            self._install_bb_lenses()

        @print_info(BUILD_INFO_MESSAGE)
        def step_11__mask_module_04_errors(self):
            pass

        @print_info(BUILD_INFO_MESSAGE)
        def step_12__mask_module_05_tuning(self):
            self._check_call_module("05a_MO")
            self._check_call_module("05b_coupling")
            self._check_call_module("05c_limit")
            self._check_call_module("05d_matching")
            self._check_call_module("05e_corrvalue")
            self._check_call_module("05f_final")

        @print_info(BUILD_INFO_MESSAGE)
        def step_13__adjust_collimators(self):
            if self._collimation_key is not None:
                rel_path = os.path.join(
                    sequence_data["collimation_base_path"],
                    sequence_data["collimation"][self._collimation_key],
                )
                self._collimation_path = pkg_resources.resource_filename(__name__, rel_path)
                self._load_and_set_collimators(with_pre_thin_twiss=with_pre_thin_twiss)

        if self._failsim is None:
            self._failsim = FailSim(failsim_verbosity=self._failsim_verbose, madx_verbosity=self._madx_verbose)

        sequence_data = self._metadata[self._sequence_key]
        self._run_version = sequence_data["run"]
        self._hllhc_version = sequence_data["version"]
        self._optics_base_path = sequence_data["optics_base_path"]

        self._failsim.set_mad_mute(True)
        step_01__load_macros(self)
        step_02__load_sequence(self)
        step_03__load_apertures(self)
        if self._madx_verbose:
            self._failsim.set_mad_mute(False)
        step_04__prepare(self)
        step_05__cycle(self)
        step_06__thin(self)
        step_07__load_optics(self)
        step_08__mask_module_01_orbit(self)
        step_09__mask_module_02_lumilevel(self)
        step_10__mask_module_03_beambeam(self)
        step_11__mask_module_04_errors(self)
        step_12__mask_module_05_tuning(self)
        step_13__adjust_collimators(self)

        return self

    @reset_state(True, True)
    @print_info("LHCSequence")
    def add_pre_thin_script(self, scripts: List[str]):
        """Adds a list of files to be called before making the sequence thin.

        Args:
            scripts: List of files to call.

        Returns:
            LHCSequence: Returns self.

        """
        self._pre_thin_scripts += scripts

        return self

    @print_info("LHCSequence")
    def run_check(self):
        """Runs a check using the internal OpticsChecks instance.

        Returns:
            LHCSequence: Returns self

        """
        self._check(
            mad=self._failsim.mad,
            sequences=self._mode_configuration["sequences_to_check"],
        )
        self._checked = True

        return self

    @reset_state(False, True)
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

    @reset_state(True, True)
    @print_info("LHCSequence")
    def cycle(self, sequences: List[str], target: str):
        """Cycles the specified sequences to start at the target element.

        Args:
            sequences: A list of sequences to cycle.
            target: The name of the element the sequences should start at.

        Returns:
            LHCSequence: Returns self

        """
        self._cycle = dict(target=target, sequences=sequences)
        return self

    @reset_state(True, True)
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

    @reset_state(False, True)
    @print_info("LHCSequence")
    def crossing_save(self):
        """Saves the current crossing settings in internal MAD-X variables.

        Returns:
            LHCSequence: Returns self

        """
        self._failsim.mad_input("exec, crossing_save")
        return self

    @reset_state(False, True)
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

    @reset_state(False, True)
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

    @reset_state(False, True)
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

    @ensure_build
    @print_info("LHCSequence")
    def build_tracker(self, detached: bool = True, verbose: bool = False):
        """Builds a Tracker instance.

        Args:
            detached:
            verbose: Whether the generated Tracker object should be verbose or not.

        Returns:
            Tracker: A Tracker instance containing this sequence.

        """
        if detached:
            fs = self._failsim.duplicate()
        else:
            fs = self._failsim
        tracker = Tracker(
            fs,
            self._mode_configuration["sequence_to_track"],
            verbose=verbose,
        )
        return tracker

    @ensure_build
    @print_info("LHCSequence")
    def get_aperture_model(self):
        if self._aperture_model is None:
            _ = self._metadata[self._sequence_key]["aperture_model"]
            logging.warning(f"Loading the aperture model from: {_}")
            self._aperture_model = Aperture(self, _)
        return self._aperture_model

    def get_around_element(
        self,
        element: str,
        width: float,
        dataframe_filter: Callable[[pd.DataFrame], pd.Series] = None,
    ):
        """Returns a DataFrame containing twiss data for elements around the given element at a specific width.

        Args:
            element: Name of element around which elements will be returned.
            width: The longitudinal distance around the given element to filter elements by.
            dataframe_filter: Allows specification of a function that will be passed to the .loc method of the twiss DataFrame. Can be used to filter the DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing elements in the given width around the given element.

        """
        twiss = self.twiss().twiss_df
        twiss_filt = twiss.loc[dataframe_filter(twiss)].copy()
        return twiss_filt.loc[
            (twiss_filt["s"] > twiss.loc[element]["s"] - width)
            & (twiss_filt["s"] < twiss.loc[element]["s"] + width)
        ]

    @ensure_build
    def twiss(self):
        """Calculates and returns twiss table.

        Returns:
            TwissResult: TwissResult containing result.

        """
        twiss_df, summ_df = self._failsim.twiss_and_summ(
            self._mode_configuration["sequence_to_track"]
        )

        eps_n = self._failsim.mad.globals["par_beam_norm_emit"] * 1e-6
        run_version = self._failsim.mad.globals["ver_lhc_run"]
        hllhc_version = self._failsim.mad.globals["ver_hllhc_optics"]

        twiss_df["turn"] = 0

        return TwissResult(
            twiss_df=twiss_df,
            summ_df=summ_df,
            run_version=run_version,
            hllhc_version=hllhc_version,
            eps_n=eps_n,
            beam=dict(self._failsim.mad.sequence[self._mode_configuration["sequence_to_track"]].beam.items())
        )

    @ensure_build
    def get_initial_twiss(self) -> Dict:
        """ Return the initial Twiss parameters

        Returns:
            FailSim: Returns a dictionary with the Twiss and the normalized emittance
        """
        twiss0 = self.twiss().twiss_df.iloc[0]
        return {'bet_x': twiss0['betx'],
                'bet_y': twiss0['bety'],
                'alf_x': twiss0['alfx'],
                'alf_y': twiss0['alfy'],
                'x': twiss0['x'],
                'y': twiss0['y'],
                'p_x': twiss0['px'],
                'p_y': twiss0['py'],
                'emit_x': self.mad.globals["epsx"],
                'emit_y': self.mad.globals["epsy"],
                'energy': self.mad.globals['par_beam_energy_tot']
                }

    def set_collimators(self, handler: CollimatorHandler):
        """Sets collimators using specified CollimatorHandler.

        Args:
            handler: Handler to use.

        Returns:
            LHCSequence: Returns self.

        """
        twiss = self.twiss()

        settings = handler.compute_openings(twiss.twiss_df)

        for _, row in settings.iterrows():
            cmd = f"{row.name}, apertype=rectangle, aperture={{ {row['half_gaph']['info']}, {row['half_gapv']['info']}}}"
            self._failsim.mad_input(cmd)

        return self


class HLLHCSequence(LHCSequence):
    def __init__(self, *args, **kwargs):
        if kwargs.get("sequence_key") and (kwargs["sequence_key"] != "HLLHCV1.4" and kwargs["sequence_key"] != "HLLHCV1.5"):
            raise Exception("Invalid sequence key")
        super().__init__(*args, **kwargs)


class CollimatorHandler:
    """Class for loading collimation settings files, and converting collimator settings from beam sigma to mm."""

    def __init__(self, settings_file: str, check_against: pd.DataFrame = None):
        with open(settings_file, "r") as fd:
            col_settings = fd.readlines()
        col_settings = [x.split() for x in col_settings]
        self._collimator_df = (
            pd.DataFrame(col_settings[1:], columns=col_settings[0])
            .set_index("Name")
            .drop(columns="#")
        )
        self._collimator_df.index = self._collimator_df.index.str.lower()

    def compute_offsets(self, twiss: TwissResult):
        """Computes and returns horizontal and vertical offsets. The collimators are centered around the reference
        closed-orbit and not around the machine center. So we need to offset them with respect to the computed
        closed-orbit.

        Args:
            twiss: Dataframe containing twiss data.

        Returns:
            pd.DataFrame: Dataframe containing results.

        """
        centers_h = []
        centers_v = []
        for _, row in self._collimator_df.iterrows():
            x, y = twiss.twiss_df[twiss.twiss_df['turn'] == 0].loc[row.name.lower()][['x', 'y']]
            centers_h.append(x)
            centers_v.append(y)

        res = pd.DataFrame()
        res["offset_x"] = centers_h
        res["offset_y"] = centers_v
        res = res.set_index(self._collimator_df.index)
        return res

    def compute_openings(self, twiss: TwissResult):
        """Computes and returns horizontal, vertical and radial half-gap for each collimator, based on data given in twiss dataframe.

        Args:
            twiss: Dataframe containing twiss data.

        Returns:
            pd.DataFrame: Dataframe containing results.

        """
        eps_g = twiss.info_df.at['info', 'eps_n'] / twiss.beam['gamma']

        gap = []
        for _, row in self._collimator_df.iterrows():
            beta_x, beta_y = twiss.twiss_df[twiss.twiss_df['turn'] == 0].loc[row.name.lower()][['betx', 'bety']]
            angle = float(row["angle[rad]"])
            gap.append(
                float(row['nsig']) * np.sqrt(eps_g) * np.sqrt(beta_x * np.cos(angle)**2 + beta_y * np.sin(angle)**2)
            )

        res = pd.DataFrame()
        res["angle"] = self._collimator_df["angle[rad]"]
        res["nsig"] = self._collimator_df["nsig"]
        res["half_gap"] = gap
        return res
