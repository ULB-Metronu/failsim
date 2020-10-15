"""
This module contains classes for miscellaneous tasks that don't fit in anywhere else.
"""


from typing import ByteString


class OutputSuppressor:

    """Can be used to intercept output to stdout, and only prints if not enabled.

    Is used by [FailSim](failsim.failsim.FailSim) to suppress Mad-X output.

    Args:
        enabled: Whether or not OutputSuppressor should initially suppress incoming messages or not.

    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._buffer = []
        self._buffer_maxlen = 100

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

    def read(self):
        """Returns what's currently in the buffer.

        Returns:
            List[str]: Returns what's currently in the buffer.

        """
        return self._buffer


class ArrayFile:

    """Array that behaves like a file.

    Is used by [FailSim](failsim.failsim.FailSim) to keep track of commands sent to Mad-X.

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
