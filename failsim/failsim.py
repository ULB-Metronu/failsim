"""
Contains the class FailSim.
"""
from __future__ import annotations
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import pymask as pm
import os
import pkg_resources
import yaml
import shutil
import io
import copy


from .helpers import OutputSuppressor, ArrayFile, print_info, MoveNewFiles
from .globals import FailSimGlobals

class FailSim:
    """
    This class is the interface to the Mad-X instance in `cpymad`.

    Note:
        The output directory and the cwd can also be set globally by setting the static variables `output_dir` and `cwd` in the FailSimGlobals class.
        If an output directory or cwd are specified in the constructor, the ones specified in the constructor will take priority.

    Args:
        output_dir: Sets the desired output directory. If output_dir is None, and FailSimGlobals.output_dir is None, FailSim outputs all files in the cwd.
        cwd: Sets the desired cwd. If cwd is None, FailSim uses os.getcwd() to set the cwd.
        madx_verbosity: Sets the verbosity of Mad-X. If this parameter is "mute", FailSim will use OutputSuppressor to completely mute Mad-X output.
        failsim_verbosity: Enables or disables stdout output from FailSim.
        macro_files: An optional list of .madx files that should be called when MAD-X is initialized.
        command_log: If command_log is not None, FailSim will input each of the commands in the log into the initialized MAD-X instance.
        log_file: If a path to a file is given, all MAD-X commands will be logged in given file.
    """

    cwd: str = os.getcwd()
    output_dir: str = None

    def __init__(
        self,
        output_dir: Optional[str] = None,
        cwd: Optional[str] = None,
        madx_verbosity: str = "mute",
        failsim_verbosity: bool = False,
        macro_files: Optional[List[str]] = None,
        command_log: Optional[ArrayFile] = None,
        log_file: Optional[str] = None,
    ):
        self._madx_verbosity = madx_verbosity
        self._macro_files = macro_files
        self._verbose = failsim_verbosity
        self._mad = None

        # Setup cwd
        if cwd is None:
            if FailSimGlobals.cwd is None:
                self._cwd = os.getcwd()
            else:
                self._cwd = FailSimGlobals.cwd
        else:
            self._cwd = cwd

        # Setup output directory
        if output_dir is None:
            if FailSimGlobals.output_dir is None:
                self._output_dir = self._cwd
            else:
                self._output_dir = FailSimGlobals.output_dir
        else:
            self._output_dir = output_dir

        # Make output directory path absolute
        if not self._output_dir.startswith("/"):
            self._output_dir = os.path.join(self._cwd, self._output_dir)

        # Create output directory if it doesn't exist
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)

        # Set static variables
        FailSim.cwd = self._cwd
        FailSim.output_dir = self._output_dir

        # Setup output suppressor
        if madx_verbosity == "mute" or not FailSimGlobals.verbose:
            self._madx_mute = OutputSuppressor(True, log_file=log_file)
        else:
            self._madx_mute = OutputSuppressor(False, log_file=log_file)

        self._command_log = ArrayFile()
        self.initialize_mad(replay_log=command_log)

        if macro_files is not None:
            for file in extra_macro_files:
                self.mad_call_file(file)

    @property
    def mad(self):
        """Provides the running MAD-X instance."""
        return self._mad

    @print_info("FailSim")
    def initialize_mad(self, replay_log: Optional[ArrayFile] = None):
        """Initializes the Mad-X instance.
        Also sets the cwd and verbosity of the instance.

        Returns:
            FailSim: Returns self

        """
        if self._mad: del self._mad
        self._mad = pm.Madxp(stdout=self._madx_mute, command_log=self._command_log)
        self.mad.chdir(self._cwd)
        if replay_log is not None:
            for command in replay_log.read():
                if command.startswith("chdir"):
                    continue
                self.mad_input(command)

        if self._madx_verbosity != "mute" and self._madx_verbosity != '':
            self.mad_input("option, " + self._madx_verbosity)

        return self

    @print_info("FailSim")
    def set_mad_mute(self, is_muted: bool):
        """Enables/disables the internal OutputSuppressor.

        Args:
            is_muted: Whether the Mad-X instance should be muted or not.

        Returns:
            FailSim: Returns self

        """
        self._madx_mute.set_enabled(is_muted)

        return self

    @print_info("FailSim")
    def mad_call_file(self, path: str):
        """Method to call a file using the Mad-X instance.
        If an output directory has been specified, FailSim will move any new files or directories to the output directory.

        Args:
            path: The path of the file to call. If path doesn't start with "/", FailSim will prepend the cwd to the path.

        Returns:
            FailSim: Returns self

        """
        with MoveNewFiles(self._cwd, self._output_dir):
            if not path.startswith("/"):
                path = os.path.join(self._cwd, path)
            self.mad.call(path)

        return self

    @print_info("FailSim")
    def mad_input(self, command: str):
        """Method to input a command into the Mad-X instance.
        If an output directory has been specified, FailSim will move any new files or directories to the output directory.

        Args:
            command: Command to input.

        Returns:
            FailSim: Returns self

        """
        with MoveNewFiles(self._cwd, self._output_dir):
            self.mad.input(command + ";")

        return self

    @print_info("FailSim")
    def duplicate(self) -> FailSim:
        """Duplicates the FailSim instance.

        Returns:
            FailSim: The new copy

        """
        return FailSim(
            output_dir=copy.copy(self._output_dir),
            cwd=copy.copy(self._cwd),
            madx_verbosity=copy.copy(self._madx_verbosity),
            failsim_verbosity=self._verbose,
            command_log=copy.copy(self._command_log),
        )

    @print_info("FailSim")
    def use(self, seq: str):
        """Method to use a sequence in the Mad-X instance.

        Args:
            seq: Sequence to use.

        Returns:
            FailSim: Returns self

        """
        self.mad.use(seq)

        return self

    @print_info("FailSim")
    def twiss_and_summ(self, seq: str, flags: List[str] = None) -> Tuple[pd.DataFrame]:
        """Performs a Twiss with the given sequence on the Mad-X instance.

        Args:
            seq: Sequence to run twiss on.
            flags: Additional flags to pass to the twiss command.

        Returns:
            tuple: Tuple containing:

                pandas.DataFrame: DataFrame containing the twiss table
                pandas.DataFrame: DataFrame containing the summ table

        """
        flags = flags or []
        self.use(seq)
        self.mad_input(f"{', '.join(['twiss', f'sequence={seq}'] + flags)}")
        return self.mad.table["twiss"].dframe(), self.mad.table["summ"].dframe()

    @print_info("FailSim")
    def call_pymask_module(self, module: str):
        """Calls a pymask module using the Mad-X instance.

        Args:
            module: Module to call

        Returns:
            FailSim: Returns self

        """
        path = pkg_resources.resource_filename(
            __name__, "data/pymask/" + module
        )
        self.mad_call_file(path)

        return self

    @print_info("FailSim")
    def make_thin(self):
        """Makes the given sequence thin using the `myslice` macro.

        Args:
            sequence: The sequence that must be converted to a thin sequence.

        Returns:
            FailSim: Returns self

        """
        self.mad_input(f"exec myslice")
        return self

    @classmethod
    def path_to_cwd(cls, path: str):
        """Class method that prepends the cwd to a given path.

        Args:
            path: The path to alter.

        Returns:
            str: Modified path with cwd prepended.

        """
        return os.path.join(cls.cwd, path)

    @classmethod
    def path_to_output(cls, path: str):
        """Class method that prepends the output directory to a given path.

        Args:
            path: The path to alter.

        Returns:
            str: Modified path with the output directory prepended.

        """
        if cls.output_dir is None:
            if FailSimGlobals.output_dir is None:
                return path
            else:
                return os.path.join(FailSimGlobals.output_dir, path)
        return os.path.join(cls.output_dir, path)
