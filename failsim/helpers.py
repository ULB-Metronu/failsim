"""
File: helpers.py
Author: Oskari Tuormaa
Email: oskari.kristian.tuormaa@cern.ch
Github: https://github.com/Oskari-Tuormaa
Description: TODO
"""


from typing import ByteString


class OutputSuppressor:

    """Can be used to intercept output to stdout, and only prints
    if not enabled."""

    _enabled: bool

    def __init__(self, enabled: bool = True):
        """Initializes OutputSuppressor.

        Kwargs:
            enabled (bool): Whether or not OutputSuppressor should
                initially suppress incoming messages or not.


        """
        self._enabled = enabled

    def set_enabled(self, enabled: bool):
        """Enables or disables OutputSuppressor.

        Args:
            enabled (bool): Whether or not OutputSupressor should
                suppress incoming messages or not.

        Returns: None

        """
        self._enabled = enabled

    def write(self, string: ByteString):
        """Prints written content to stdout if self._enabled is False.

        Args:
            string (ByteString): ByteString to print

        Returns: None

        """
        if not self._enabled:
            print(string.decode("utf-8"))


class ArrayFile:

    """Array that behaves like a file. """

    def __init__(self):
        """Initializes ArrayFile.
        """
        self._lines = []

    def __call__(self, string: ByteString):
        """Saves written content to internal array.

        Args:
            string (ByteString): ByteString to print

        Returns: None

        """
        self._lines.append(string)

    def read(self):
        """Returns internal array containing all written content.
        Returns: List[str]

        """
        return self._lines

    def copy(self):
        """Copies this ArrayFile instance.
        Returns: Copy of this ArrayFile

        """
        temp = ArrayFile()
        temp._lines = self._lines.copy()
        return temp
