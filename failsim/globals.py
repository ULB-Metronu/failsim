"""
Contains global settings.
"""
from dataclasses import dataclass


@dataclass
class FailSimGlobals:
    """
    Static dataclass that holds global variables.

    Attributes:
        output_dir: Global output directory default. Does not take priority over instance specific output directories.
        cwd: Global working directory. Does not take priority over instance specific working directories.
        verbose: Global verbosity toggle. If False, mutes every class in the FailSim package.
        tmp_direcotry: The name of the directory where the temporary files are stored.
    """
    output_dir: str = None
    cwd: str = None
    verbose: bool = True
    tmp_directory: str = 'tmp'