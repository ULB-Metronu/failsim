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
        """Initialized OutputSuppressor.

        Kwargs:
            enabled (bool): Whether or not OutputSuppressor should
                initially suppress incoming messages or not.


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
