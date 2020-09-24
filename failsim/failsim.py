"""
File: failsim.py
Author: Oskari Tuormaa
Email: oskari.kristian.tuormaa@cern.ch
Github: https://gitlab.cern.ch/machine-protection/fast-beam-failures/
Description: TODO
"""


from typing import Optional, List, Union
from .checks import OpticsChecks
import yaml
import imp
import os
import pkg_resources
import pymask as pm


class FailSim:

    """TODO: Docstring for FailSim. """

    mad_verbosity: str = 'echo, warn, info'
    failsim_verbosity: bool = True

    def __init__(self, mode: str,
                 save_intermediate_twiss: bool = True,
                 check_betas_at_ips: bool = True,
                 check_separations_at_ips: bool = True,
                 tol_beta: List = [1e-3, 10e-2, 1e-3, 1e-2],
                 tol_sep: List = [1e-6, 1e-6, 1e-6, 1e-6]):
        """TODO: Docstring for constructor

        Args:
            mode (TODO): TODO

        Kwargs:
            save_intermediate_twiss (TODO): TODO
            check_betas_at_ips (TODO): TODO
            check_separations_at_ips (TODO): TODO
            tol_beta (TODO): TODO
            tol_sep (TODO): TODO


        """
        self._mode = mode
        self._tol_beta = tol_beta
        self._tol_sep = tol_sep
        self._save_intermediate_twiss = save_intermediate_twiss
        self._check_betas_at_ips = check_betas_at_ips
        self._check_separations_at_ips = check_separations_at_ips

    def init_check(self):
        """TODO: Docstring for init_check.
        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Initializing check class')

        self._check = OpticsChecks(
            separation=self._check_separations_at_ips,
            beta=self._check_betas_at_ips,
            tol_sep=self._tol_sep,
            tol_beta=self._tol_beta,
            save_twiss_files=self._save_intermediate_twiss
        )

    def load_metadata(self):
        """TODO: Docstring for load_metadata.
        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Loading metadata')

        metadata_stream = pkg_resources.resource_stream(__name__,
                                                        'data/metadata.yaml')
        self._metadata = yaml.safe_load(metadata_stream)
        return (self._metadata['OPTICS'], self._metadata['SEQUENCES'])

    def select_sequence(self, sequence_key: str):
        """TODO: Docstring for select_sequence.

        Args:
            sequence_key (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Setting sequence: %s' % sequence_key)

        self._seq_key = sequence_key
        self._custom_sequence = False

    def select_custom_sequence(self, sequence_paths: List[str],
                               run_version: int = 0,
                               hllhc_version: float = 0.0):
        """TODO: Docstring for select_custom_sequence.

        Args:
            sequence_path (TODO): TODO
            sequence_path (TODO): TODO
            sequence_path (TODO): TODO

        Kwargs:
            run_version (TODO): TODO
            hllhc_version (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print("""FailSim -> Setting custom sequence:
                  \tPath: %s
                  \trun_version: %d
                  \thllhc_version: %.1f""" % (sequence_paths, run_version,
                                              hllhc_version))

        self._sequence_paths = sequence_paths
        self._run_version = run_version
        self._hllhc_version = hllhc_version
        self._custom_sequence = True

    def select_optics(self, optics_key: str):
        """TODO: Docstring for select_optics.

        Args:
            optics_key (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Setting optics: %s' % optics_key)

        self._opt_key = optics_key
        self._custom_optics = False

    def select_custom_optics(self, optics_path: str,
                             optics_type: str,
                             beta_ip5: float):
        """TODO: Docstring for select_custom_optics.

        Args:
            optics_path (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print("FailSim -> Setting custom optics: %s" % optics_path)

        self._opt_path = optics_path
        self._opt_type = optics_type
        self._beta_ip5 = beta_ip5
        self._custom_optics = True

    def init_mad_instance(self):
        """TODO: Docstring for init_mad_instance.

        Returns:
            None

        """
        if self.failsim_verbosity:
            print('FailSim -> Initializing MAD instance')

        self._mad = pm.Madxp()

    def mad_call_file(self, file_path: str):
        """TODO: Docstring for mad_call_file.

        Args:
            file_path (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Calling MAD with file: %s' % file_path)

        self._mad.input('option, %s' % self.mad_verbosity)
        self._mad.call(file_path)

    def fetch_mode_configuration(self):
        """TODO: Docstring for fetch_mode_configuration.

        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Fetching mode configuration')

        (self._beam_to_configure,
         self._sequences_to_check,
         self._sequence_to_track,
         self._generate_b4_from_b2,
         self._track_from_b4_mad_instance,
         self._enable_bb_python,
         self._enable_bb_legacy,
         self._force_disable_check_separations_at_ips,
         ) = pm.get_pymask_configuration(self._mode)

        if self._force_disable_check_separations_at_ips:
            self._check_separations_at_ips = False
            self._check.check_separations = False

    def load_sequences_and_optics(self,
                                  pre_scripts: Optional[List[str]] = None,
                                  post_scripts: Optional[List[str]] = None):
        """TODO: Docstring for load_sequences_and_optics.

        Kwargs:
            pre_scripts (TODO): TODO
            post_scripts (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Loading sequence and optics')

        # Load sequence and optics data
        if self._custom_sequence:
            seq_data = dict(sequence_filenames=self._sequence_paths,
                            run=self._run_version,
                            version=self._hllhc_version)
        else:
            seq_data = self._metadata['SEQUENCES'][self._seq_key]

        if self._custom_optics:
            opt_data = dict(type=self._opt_type,
                            beta_ip5=self._beta_ip5,
                            strength_file=self._opt_path)
        else:
            opt_data = self._metadata['OPTICS'][self._opt_key]

        # Set machine versions
        self._mad.input('ver_lhc_run = %s' % seq_data['run'])
        self._mad.input('ver_hllhc_optics = %s' % seq_data['version'])

        # Run pre_scripts if any
        if pre_scripts is not None:
            for script in pre_scripts:
                self.mad_call_file(script)

        # Run all sequences
        for sequence in seq_data['sequence_filenames']:
            if self._custom_sequence:
                seq_path = sequence
            else:
                seq_path = pkg_resources.resource_filename(__name__, sequence)
            self.mad_call_file(seq_path)

        # Load optics
        opt_path = pkg_resources.resource_filename(
            __name__,
            self._metadata['OPTICS_BASE_PATH'] + opt_data['strength_file']
        )
        self.mad_call_file(opt_path)

        # Run post_scripts if any
        if post_scripts is not None:
            for script in post_scripts:
                self.mad_call_file(script)

    def make_sequence_thin(self,
                           script_path: Optional[str] = None):
        """TODO: Docstring for make_sequence_thin.

        Kwargs:
            script_path (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Making sequence thin')

        # If no script_path given, do nominal slice
        if script_path is None:
            macro_path = pkg_resources.resource_filename(__name__,
                                                         'data/hllhc14/'
                                                         'toolkit/macro.madx')
            self.mad_call_file(macro_path)
            self._mad.input('exec, myslice')
        else:
            self._mad.input(script_path)

    def cycle_sequence(self, sequence: Union[str, List], start: str):
        """TODO: Docstring for cycle_sequence.

        Args:
            sequence (TODO): TODO
            start (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Cycling sequence')

        # Check whether multiple or single sequences should be cycled
        if isinstance(sequence, str):
            for seq in sequence:
                self._mad.input(f'seqedit, sequence={seq}; flatten;'
                                f'cycle, start={start}; flatten; endedit;')
        else:
            self._mad.input(f'seqedit, sequence={sequence}; flatten;'
                            f'cycle, start={start}; flatten; endedit;')

    def load_mask_parameters(self, mask_path: Optional[str] = None):
        """TODO: Docstring for load_mask_parameters.

        Kwargs:
            mask_path (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Loading mask parameters')

        # If no specific path is given to mask_parameters,
        # assume that mask_parameters.py is in same directory
        if mask_path is None:
            from mask_parameters import mask_parameters
        else:
            # Load mask_parameters from path
            mask_params_source = imp.load_source(os.path.basename(mask_path),
                                                 os.path.abspath(mask_path))
            mask_parameters = mask_params_source.mask_parameters

        if not self._enable_bb_legacy and not self._enable_bb_python:
            mask_parameters['par_on_bb_switch'] = 0.

        pm.checks_on_parameter_dict(mask_parameters)

        self._mad.set_variables_from_dict(params=mask_parameters)

    def prepare_and_attach_beam(self):
        """TODO: Docstring for prepare_and_attach_beam.
        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Calling preparation and beam modules')

        submod_prep = pkg_resources.resource_filename('pymask', '../submodule'
                                                      '_01a_preparation.madx')
        submod_beam = pkg_resources.resource_filename('pymask', '../submodule'
                                                      '_01b_beam.madx')

        self.mad_call_file(submod_prep)
        self.mad_call_file(submod_beam)

        self._check(self._mad, self._sequences_to_check,
                    twiss_name='twiss_from_optics')

    def twiss(self, sequence: Union[List, str]):
        """TODO: Docstring for twiss.

        Args:
            sequence (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Doing twiss')

        if isinstance(sequence, list):
            res = dict()
            for seq in sequence:
                self._mad.use(seq)
                self._mad.twiss()
                twiss_df = self._mad.get_twiss_df('twiss')
                summ_df = self._mad.get_summ_df('summ')
                res[seq] = {'twiss': twiss_df, 'summ': summ_df}
            return res
        else:
            self._mad.use(sequence)
            self._mad.twiss()
            twiss_df = self._mad.get_twiss_df('twiss')
            summ_df = self._mad.get_summ_df('summ')
            return (twiss_df, summ_df)

    def load_knob_parameters(self, knob_path: Optional[str] = None):
        """TODO: Docstring for load_knob_parameters.

        Kwargs:
            knob_path (TODO): TODO

        Returns: TODO

        """
        if self.failsim_verbosity:
            print('FailSim -> Loading knob parameters')

        # If no specific path is given to knob_parameters,
        # assume that knob_parameters.py is in same directory
        if knob_path is None:
            from knob_parameters import knob_parameters
        else:
            # Load mask_parameters from path
            knob_params_source = imp.load_source(os.path.basename(knob_path),
                                                 os.path.abspath(knob_path))
            knob_parameters = knob_params_source.knob_parameters

        # PATCH!!!!!!! for leveling not working for b4
        if self._mode == 'b4_without_bb':
            knob_parameters['par_sep8'] = -0.03425547139366354
            knob_parameters['par_sep2'] = 0.14471680504084292

        self._mad.set_variables_from_dict(params=knob_parameters)

        # Set IP knobs
        self._mad.globals['on_x1'] = knob_parameters['par_x1']
        self._mad.globals['on_sep1'] = knob_parameters['par_sep1']
        self._mad.globals['on_x2'] = knob_parameters['par_x2']
        self._mad.globals['on_sep2'] = knob_parameters['par_sep2']
        self._mad.globals['on_x5'] = knob_parameters['par_x5']
        self._mad.globals['on_sep5'] = knob_parameters['par_sep5']
        self._mad.globals['on_x8'] = knob_parameters['par_x8']
        self._mad.globals['on_sep8'] = knob_parameters['par_sep8']
        self._mad.globals['on_a1'] = knob_parameters['par_a1']
        self._mad.globals['on_o1'] = knob_parameters['par_o1']
        self._mad.globals['on_a2'] = knob_parameters['par_a2']
        self._mad.globals['on_o2'] = knob_parameters['par_o2']
        self._mad.globals['on_a5'] = knob_parameters['par_a5']
        self._mad.globals['on_o5'] = knob_parameters['par_o5']
        self._mad.globals['on_a8'] = knob_parameters['par_a8']
        self._mad.globals['on_o8'] = knob_parameters['par_o8']
        self._mad.globals['on_crab1'] = knob_parameters['par_crab1']
        self._mad.globals['on_crab5'] = knob_parameters['par_crab5']
        self._mad.globals['on_disp'] = knob_parameters['par_on_disp']

        # A check
        if self._mad.globals.nrj < 500:
            assert knob_parameters['par_on_disp'] == 0

        # Spectrometers at experiments
        if knob_parameters['par_on_alice'] == 1:
            self._mad.globals.on_alice = 7000./self._mad.globals.nrj
        if knob_parameters['par_on_lhcb'] == 1:
            self._mad.globals.on_lhcb = 7000./self._mad.globals.nrj

        # Solenoids at experiments
        self._mad.globals.on_sol_atlas = knob_parameters['par_on_sol_atlas']
        self._mad.globals.on_sol_cms = knob_parameters['par_on_sol_cms']
        self._mad.globals.on_sol_alice = knob_parameters['par_on_sol_alice']

    def crossing_save(self, store_optics: bool = True):
        """TODO: Docstring for crossing_save.

        Kwargs:
            store_optics (TODO): TODO

        Returns: TODO

        """
        self._mad.input('exec, crossing_save')
        if store_optics:
            mod_path = pkg_resources.resource_filename('pymask', '../submodule'
                                                       '_01e_final.madx')
            self._mad.call(mod_path)
