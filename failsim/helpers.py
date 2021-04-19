

"""
This module contains classes for miscellaneous tasks that don't fit in anywhere else.
"""


from .globals import FailSimGlobals
from typing import ByteString, Callable, List, Optional
import functools
import os
import re
import shutil
import numpy as np


class OutputSuppressor:

    """Can be used to intercept output to stdout, and only prints if not enabled.

    Is used by FailSim to suppress Mad-X output.

    Args:
        enabled: Whether or not OutputSuppressor should initially suppress incoming messages or not.
        log_file: If a path to a file is given, all MAD-X commands will be logged in given file.

    """

    def __init__(self, enabled: bool = True, log_file: Optional[str] = None):
        self._enabled = enabled
        self._buffer = []
        self._buffer_maxlen = 100
        self._log_file = log_file

    def set_enabled(self, enabled: bool):
        """Enables or disables OutputSuppressor.

        Args:
            enabled: Whether or not OutputSupressor should suppress incoming messages or not.

        """
        self._enabled = enabled

    def write(self, string: ByteString):
        """Prints written content to stdout if self._enabled is False.

        Args:
            string: ByteString to print.

        """
        self._buffer.append(string)

        while len(self._buffer) > self._buffer_maxlen:
            self._buffer.pop(0)

        if not self._enabled:
            print(string.decode("utf-8"))

        if self._log_file is not None:
            with open(self._log_file, "a") as fd:
                fd.write(string.decode("utf-8"))

    def read(self):
        """Returns what's currently in the buffer.

        Returns:
            List[str]: Returns what's currently in the buffer.

        """
        return self._buffer


class ArrayFile:

    """Array that behaves like a file.

    Is used by FailSim to keep track of commands sent to Mad-X.

    Examples:
        A simple example showing writing to and reading from an ArrayFile object.

        >>> af = ArrayFile() # Create ArrayFile object
        >>> af("Hello") # Write to ArrayFile
        >>> af(" world!")
        >>> x = af.read() # Read from ArrayFile
        >>> print(x)
        Hello world!
    """

    def __init__(self):
        self._lines = []

    def __call__(self, string: ByteString):
        """Saves written content to internal array.

        Args:
            string: ByteString to print.

        """
        self._lines.append(string)

    def read(self):
        """Returns internal array containing all written content.

        Returns:
            List[ByteString]: List containing data that has been written to this ArrayFile instance.

        """
        return self._lines

    def copy(self):
        """Copies this ArrayFile instance.

        Returns:
            ArrayFile: Copy of this ArrayFile.

        """
        temp = ArrayFile()
        temp._lines = self._lines.copy()
        return temp


class MoveNewFiles:
    """
    Moves files created in source directory while in context to destination.

    Args:
        source: Path to source directory. This is the directory that is watched for new files.
        destination: Path to destination directory. This is the directory that new files are moved to.

    """

    exclude = [r"^\..*", "temp"]

    def __init__(self, source: str, destination: str):
        self._source = source
        self._destination = destination

    def __enter__(self):
        self._pre_files = os.listdir(self._source)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        post_files = os.listdir(self._source)

        new_files = np.setdiff1d(post_files, self._pre_files)
        new_files = self.filter_exclude(new_files)
        for file in new_files:
            shutil.move(file, os.path.join(self._destination, os.path.basename(file)))

    def filter_exclude(self, files: List[str]):
        for ex in self.exclude:
            r = re.compile(ex)
            res = [r.findall(x) for x in files]
            res = [x[0] for x in res if len(x) > 0]
            files = list(set(files) - set(res))
        return files


def print_info(name: str):
    """
    Decorator to print debug information.

    Args:
        name: Name of that class to use in the printed statement.

    """

    def _print_info(func):
        @functools.wraps(func)
        def _wrapper_print_info(self, *args, **kwargs):
            if self._verbose and FailSimGlobals.verbose:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                print(f"{name} -> {func.__name__}({signature})")
            val = func(self, *args, **kwargs)
            return val

        return _wrapper_print_info

    return _print_info
