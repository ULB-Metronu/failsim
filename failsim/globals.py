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
        verbose: Global verbosity toggle. If False, mutes every class in the FailSim package.
    """

    output_dir: str = None
    verbose: bool = True
