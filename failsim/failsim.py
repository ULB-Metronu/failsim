"""
Contains the class FailSim.
"""


from .helpers import OutputSuppressor, ArrayFile
from .globals import FSGlobals

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

    """
    This class is the interface to the Mad-X instance.

    Note:
        The output directory and the cwd can also be set globally by setting the static variables output_dir and cwd in the [FSGlobals](failsim.globals.FSGlobals) class.
        If an output directory or cwd are specified in the constructor, the ones specified in the constructor will take priority.

    Args:
        output_dir: Sets the desired output directory. If output_dir is None, and FSGlobals.output_dir is None, FailSim outputs all files in the cwd.
        cwd: Sets the desired cwd. If cwd is None, FailSim uses os.getcwd() to set the cwd.
        madx_verbosity: Sets the verbosity of Mad-X. If this parameter is "mute", FailSim will use [OutputSuppressor](failsim.helpers.OutputSuppressor) to completely mute Mad-X output.
        failsim_verbosity: Enables or disables stdout output from FailSim.
        extra_macro_files: An optional list of .madx files that should be called when Mad-X is initialized.
        command_log: Is command_log is not None, FailSim will input each of the commands in the log into the initialized Mad-X instance.

    """

    cwd: str = os.getcwd()
    output_dir: str = None

    def __init__(
        self,
        output_dir: Optional[str] = None,
        cwd: Optional[str] = None,
        madx_verbosity: str = "echo warn info",
        failsim_verbosity: bool = True,
        extra_macro_files: Optional[List[str]] = None,
        command_log: Optional[ArrayFile] = None,
    ):
        self._madx_verbosity = madx_verbosity
        self._extra_macro_files = extra_macro_files
        self._failsim_verbosity = failsim_verbosity
        self._mad = None
        self._macros_loaded = False

        if command_log is None:
            self._master_mad_command_log = ArrayFile()
        else:
            self._master_mad_command_log = command_log

        # Setup cwd
        if cwd is None:
            if FSGlobals.cwd is None:
                self._cwd = os.getcwd()
            else:
                self._cwd = FSGlobals.cwd
        else:
            self._cwd = cwd

        # Setup output directory
        if output_dir is None:
            if FSGlobals.output_dir is None:
                self._output_dir = self._cwd
            else:
                self._output_dir = FSGlobals.output_dir
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
        if madx_verbosity == "mute" or not FSGlobals.verbose:
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
        """ Decorator to print FailSim debug information """

        @functools.wraps(func)
        def wrapper_print_info(self, *args, **kwargs):
            if self._failsim_verbosity and FSGlobals.verbose:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                print(f"FailSim -> {func.__name__}({signature})")
            val = func(self, *args, **kwargs)
            return val

        return wrapper_print_info

    @_print_info
    def initialize_mad(self):
        """Initializes the Mad-X instance.
        Also sets the cwd and verbosity of the instance.

        Returns:
            FailSim: Returns self

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
        """Enables/disables the internal [OutputSuppressor](failsim.helpers.OutputSuppressor).

        Args:
            is_muted: Whether the Mad-X instance should be muted or not.

        Returns:
            FailSim: Returns self

        """
        self._madx_mute.set_enabled(is_muted)

        return self

    @_print_info
    def load_macros(self):
        """Loads macro.madx into the Mad-X instance.

        Returns:
            FailSim: Returns self

        """
        macro_path = pkg_resources.resource_filename(
            __name__, "data/hllhc14/" "toolkit/macro.madx"
        )
        self.mad_call_file(macro_path)
        self._macros_loaded = True

        return self

    @_print_info
    def mad_call_file(self, path: str):
        """Method to call a file using the Mad-X instance.
        If an output directory has been specified, FailSim will move any new files or directories to the output directory.

        Args:
            path: The path of the file to call. If path doesn't start with "/", FailSim will prepend the cwd to the path.

        Returns:
            FailSim: Returns self

        """
        pre_files = os.listdir(self._cwd)

        call_path = path
        if not path.startswith("/"):
            call_path = os.path.join(self._cwd, path)
        self._mad.call(call_path)

        post_files = os.listdir(self._cwd)

        if self._output_dir is not None:
            new_files = np.setdiff1d(post_files, pre_files)
            for file in new_files:
                shutil.move(
                    file, os.path.join(self._output_dir, os.path.basename(file))
                )

        return self

    @_print_info
    def mad_input(self, command: str):
        """Method to input a command into the Mad-X instance.
        If an output directory has been specified, FailSim will move any new files or directories to the output directory.

        Args:
            command: Command to input.

        Returns:
            FailSim: Returns self

        """
        pre_files = os.listdir(self._cwd)

        self._mad.input(command)

        post_files = os.listdir(self._cwd)

        if self._output_dir is not None:
            new_files = np.setdiff1d(post_files, pre_files)
            for file in new_files:
                shutil.move(
                    file, os.path.join(self._output_dir, os.path.basename(file))
                )

        return self

    @_print_info
    def duplicate(self):
        """Duplicates the FailSim instance.

        Returns:
            FailSim: The new copy

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
        """Method to use a sequence in the Mad-X instance.

        Args:
            seq: Sequence to use.

        Returns:
            FailSim: Returns self

        """
        self._mad.use(seq)

        return self

    @_print_info
    def twiss_and_summ(self, seq: str):
        """Does a twiss with the given sequence on the Mad-X instance.

        Args:
            seq: Sequence to run twiss on.

        Returns:
            tuple: Tuple containing:

                pandas.DataFrame: DataFrame containing the twiss table
                pandas.DataFrame: DataFrame containing the summ table

        """
        self._mad.twiss(sequence=seq)
        return (self._mad.table["twiss"].dframe(), self._mad.table["summ"].dframe())

    @_print_info
    def call_pymask_module(self, module: str):
        """Calls a pymask module using the Mad-X instance.

        Args:
            module: Module to call

        Returns:
            FailSim: Returns self

        """
        path = pkg_resources.resource_filename("pymask", "../" + module)
        self.mad_call_file(path)

        return self

    @_print_info
    def make_thin(self, beam: str):
        """Makes the given sequence thin using the myslice macro.

        Args:
            beam: The lhc beam to make thin. Can be either 1, 2 or 4.

        Returns:
            FailSim: Returns self

        """
        self.use(f"lhcb{beam}")

        twiss_df, summ_df = self.twiss_and_summ(f"lhcb{beam}")
        pre_len = summ_df["length"][0]

        if not self._macros_loaded:
            self.load_macros()

        self.mad_input(f"use, sequence=lhcb{beam}; exec myslice")

        twiss_df, summ_df = self.twiss_and_summ(f"lhcb{beam}")
        post_len = summ_df["length"][0]

        assert post_len == pre_len, "Length of sequence changed by makethin"

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
            return path
        return os.path.join(cls.output_dir, path)
