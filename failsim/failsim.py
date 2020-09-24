"""
File: failsim.py
Author: Oskari Tuormaa
Email: oskari.kristian.tuormaa@cern.ch
Github: https://gitlab.cern.ch/machine-protection/fast-beam-failures/
Description: TODO
"""


from typing import Optional, List, Union
import yaml
import imp
import os
import pkg_resources
import pymask as pm


class FailSim:

    """TODO: Docstring for FailSim. """

    verbosity: str = 'echo, warn, info'

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

    def load_metadata(self):
        """TODO: Docstring for load_metadata.
        Returns: TODO

        """
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
        self._seq_key = sequence_key

    def select_optics(self, optics_key: str):
        """TODO: Docstring for select_optics.

        Args:
            optics_key (TODO): TODO

        Returns: TODO

        """
        self._opt_key = optics_key

    def init_mad_instance(self):
        """TODO: Docstring for init_mad_instance.

        Returns:
            None

        """
        self._mad = pm.Madxp()

    def mad_call_file(self, file_path: str):
        """TODO: Docstring for mad_call_file.

        Args:
            file_path (TODO): TODO

        Returns: TODO

        """
        self._mad.input('option, %s' % self.verbosity)
        self._mad.call(file_path)

    def fetch_mode_configuration(self):
        """TODO: Docstring for fetch_mode_configuration.

        Returns: TODO

        """
        (self._beam_to_configure,
         self._sequences_to_check,
         self._sequence_to_track,
         self._generate_b4_from_b2,
         self._track_from_b4_mad_instance,
         self._enable_bb_python,
         self._enable_bb_legacy,
         self._force_disable_check_separations_at_ips,
         ) = pm.get_pymask_configuration(self._mode)

    def load_sequences_and_optics(self,
                                  pre_scripts: Optional[List[str]] = None,
                                  post_scripts: Optional[List[str]] = None):
        """TODO: Docstring for load_sequences_and_optics.

        Kwargs:
            lhc_version (TODO): TODO
            hllhc_version (TODO): TODO
            pre_scripts (TODO): TODO
            post_scripts (TODO): TODO

        Returns: TODO

        """
        # Load sequence and optics data
        seq_data = self._metadata['SEQUENCES'][self._seq_key]
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
        # If no specific path is given to mask_parameters,
        # assume that mask_parameters.py is in same directory
        if mask_path is None:
            from mask_parameters import mask_parameters
        else:
            # Load mask_parameters from path
            mask_params_source = imp.load_source(os.path.basename(mask_path),
                                                 os.path.abspath(mask_path))
            mask_parameters = mask_params_source.mask_parameters

        if self._force_disable_check_separations_at_ips:
            self._check_seperations_at_ips = False

        if not self._enable_bb_legacy and not self._enable_bb_python:
            mask_parameters['par_on_bb_switch'] = 0.

        pm.checks_on_parameter_dict(mask_parameters)

        self._mad.set_variables_from_dict(params=mask_parameters)

    def prepare_and_attach_beam(self):
        """TODO: Docstring for prepare_and_attach_beam.
        Returns: TODO

        """
        submod_prep = pkg_resources.resource_filename('pymask', '../submodule'
                                                      '_01a_preparation.madx')
        submod_beam = pkg_resources.resource_filename('pymask', '../submodule'
                                                      '_01b_beam.madx')

        self.mad_call_file(submod_prep)
        self.mad_call_file(submod_beam)

    def twiss(self, sequence: Union[List, str]):
        """TODO: Docstring for twiss.

        Args:
            sequence (TODO): TODO

        Returns: TODO

        """
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
