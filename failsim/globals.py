"""
Contains global settings
"""

from dataclasses import dataclass


@dataclass
class FSGlobals:

    """
    Static dataclass that holds global variables.

    Attributes:
        output_dir: Global output directory default. Does not take priority over instance specific output directories.
        cwd: Global working directory. Does not take priority over instance specific working directories.
        verbose: Global verbosity toggle. If False, mutes every class in the FailSim package.
    """

    output_dir: str = None
    cwd: str = None
    verbose: bool = True
