from .helpers import OutputSuppressor, ArrayFile

from typing import Optional, List, Union
import numpy as np
import pymask as pm
import functools
import os
import pkg_resources
import yaml
import shutil
import io


class FailSim:

    """TODO: Docstring for FailSim. """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        cwd: Optional[str] = None,
        madx_verbosity: str = "echo warn info",
        failsim_verbosity: bool = True,
        extra_macro_files: Optional[List[str]] = None,
        command_log: Optional[ArrayFile] = None,
    ):
        """TODO: Docstring

        Kwargs:
            output_dir (TODO): TODO
            cwd (TODO): TODO
            madx_verbosity (TODO): TODO
            failsim_verbosity (TODO): TODO
            extra_macro_files (TODO): TODO
            command_log (TODO): TODO

        """
        self._madx_verbosity = madx_verbosity
        self._extra_macro_files = extra_macro_files
        self._failsim_verbosity = failsim_verbosity
        self._mad = None

        if command_log is None:
            self._master_mad_command_log = ArrayFile()
        else:
            self._master_mad_command_log = command_log

        # Setup cwd
        if cwd is None:
            self._cwd = os.getcwd()
        else:
            self._cwd = cwd

        # Setup output directory
        if output_dir is None:
            self._output_dir = self._cwd
        else:
            if output_dir.startswith("/"):
                self._output_dir = output_dir
            else:
                self._output_dir = os.path.join(self._cwd, output_dir)

        # Setup output suppressor
        if madx_verbosity == "mute":
            self._madx_mute = OutputSuppressor(True)
        else:
            self._madx_mute = OutputSuppressor(False)

        self.initialize_mad()
        self.load_macros()

        if extra_macro_files is not None:
            for file in extra_macro_files:
                self.mad_call_file(file)

        if command_log is not None:
            for command in command_log.copy().read():
                self.mad_input(command)

    def _print_info(func):
        """Decorator to print FailSim debug information"""

        @functools.wraps(func)
        def wrapper_print_info(self, *args, **kwargs):
            if self._failsim_verbosity:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                print(f"FailSim -> {func.__name__}({signature})")
            func(self, *args, **kwargs)

        return wrapper_print_info

    @_print_info
    def initialize_mad(self):
        """TODO: Docstring for initialize_mad.
        Returns: TODO

        """
        self._mad = pm.Madxp(
            stdout=self._madx_mute, command_log=self._master_mad_command_log
        )
        self._mad.chdir(self._cwd)

        if self._madx_verbosity != "mute":
            self.mad_input("option, " + self._madx_verbosity)

        return self

    @_print_info
    def set_mad_mute(self, is_muted: bool):
        """TODO: Docstring for set_mad_mute.

        Args:
            is_muted (TODO): TODO

        Returns: TODO

        """
        self._madx_mute.set_enabled(is_muted)

        return self

    @_print_info
    def load_macros(self):
        """TODO: Docstring for load_macros.
        Returns: TODO

        """
        macro_path = pkg_resources.resource_filename(
            __name__, "data/hllhc14/" "toolkit/macro.madx"
        )
        self.mad_call_file(macro_path)

        return self

    @_print_info
    def mad_call_file(self, path: str):
        """TODO: Docstring for mad_call_file.

        Args:
            path (TODO): TODO

        Returns: TODO

        """
        pre_files = os.listdir(self._cwd)

        call_path = path
        if not path.startswith("/"):
            call_path = os.path.join(self._cwd, path)
        self._mad.call(call_path)

        post_files = os.listdir(self._cwd)

        new_files = np.setdiff1d(post_files, pre_files)
        for file in new_files:
            shutil.move(file, self._output_dir)

        return self

    @_print_info
    def mad_input(self, command: str):
        """TODO: Docstring for mad_input.

        Args:
            command (TODO): TODO

        Returns: TODO

        """
        pre_files = os.listdir(self._cwd)

        self._mad.input(command)

        post_files = os.listdir(self._cwd)

        new_files = np.setdiff1d(post_files, pre_files)
        for file in new_files:
            shutil.move(file, self._output_dir)

        return self

    @_print_info
    def duplicate(self):
        """TODO: Docstring for duplicate.
        Returns: TODO

        """
        return FailSim(
            output_dir=self._output_dir,
            cwd=self._cwd,
            madx_verbosity=self._madx_verbosity,
            failsim_verbosity=self._failsim_verbosity,
            command_log=self._master_mad_command_log.copy(),
        )

    @_print_info
    def use(self, seq: str):
        """TODO: Docstring for use.

        Args:
            seq (TODO): TODO

        Returns: TODO

        """
        self._mad.use(seq)

    @_print_info
    def twiss_and_summ(self, seq: str):
        """TODO: Docstring for twiss_and_summ.

        Args:
            seq (TODO): TODO

        Returns: TODO

        """
        self._mad.twiss(
            sequence=seq
        )
        #return (
            #self._mad.get_twiss_df,
            #self._mad.get_summ_df
        #)
