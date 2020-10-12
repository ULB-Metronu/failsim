from typing import List, Optional
from .failsim import FailSim
from .results import TrackingResult
import functools
import os


class SequenceTracker:

    """Docstring for SequenceTracker. """

    def __init__(self, failsim: FailSim, sequence_to_track: str, verbose: bool = True):
        """TODO: to be defined.

        Args:
            failsim (TODO): TODO
            sequence_to_track (TODO): TODO

        Kwargs:
            verbose (TODO): TODO


        """
        self._failsim = failsim
        self._sequence_to_track = sequence_to_track
        self._verbose = verbose

        self._time_dependencies = []
        self._observation_points = []
        self._track_flags = ["onetable"]
        self._mask_values = {}

    def _print_info(func):
        """Decorator to print SequenceTracker debug information"""

        @functools.wraps(func)
        def wrapper_print_info(self, *args, **kwargs):
            if self._verbose:
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                print(f"SequenceTracker -> {func.__name__}({signature})")
            val = func(self, *args, **kwargs)
            return val

        return wrapper_print_info

    @_print_info
    def track(self, turns: int = 40):
        """TODO: Docstring for track.

        Kwargs:
            turns (int): TODO

        Returns: TODO

        """
        self._failsim.use(self._sequence_to_track)

        if len(self._time_dependencies) != 0:
            time_depen = []
            tmp_files = []
            for idx, file in enumerate(self._time_dependencies):
                # Subsistute keys for values
                with open(file, "r") as fd:
                    filedata = fd.read()
                for key, value in self._mask_values.items():
                    filedata = filedata.replace(key, value)
                with open(f"tmp_{idx}.txt", "w") as fd:
                    fd.write(filedata)

                tmp_files.append(f"tmp_{idx}.txt")
                time_depen.append(f"call, file='tmp_{idx}.txt';")

            # Create tr$macro
            self._track_flags.append("update")
            time_depen = " ".join(time_depen)
            self._failsim.mad_input(
                f"tr$macro(turn): macro = {{comp=turn; {time_depen} }}"
            )

        twiss_df, summ_df = self._failsim.twiss_and_summ(self._sequence_to_track)
        run_version = self._failsim._mad.globals["ver_lhc_run"]
        hllhc_version = self._failsim._mad.globals["ver_hllhc_optics"]

        flags = ", ".join(self._track_flags)
        self._failsim.mad_input(f"track, {flags}")
        self._failsim.mad_input("start")
        for obs in self._observation_points:
            self._failsim.mad_input(f"observe, place='{obs}'")
        self._failsim.mad_input(f"run, turns={turns}")
        self._failsim.mad_input("endtrack")

        track_df = self._failsim._mad.table["trackone"].dframe()

        eps_n = self._failsim._mad.globals["par_beam_norm_emit"] * 1e-6
        nrj = self._failsim._mad.globals["nrj"]

        res = TrackingResult(
            twiss_df, summ_df, track_df, run_version, hllhc_version, eps_n, nrj
        )

        for file in tmp_files:
            os.remove(file)

        return res

    @_print_info
    def add_track_flags(self, flags: List[str]):
        """TODO: Docstring for add_track_flags.

        Args:
            flags (TODO): TODO

        Returns: TODO

        """
        self._track_flags.extend(flags)

    @_print_info
    def add_time_dependence(self, file_paths: List[str]):
        """TODO: Docstring for add_time_dependence.

        Args:
            file_paths (TODO): TODO

        Returns: TODO

        """
        fixed_paths = []
        for path in file_paths:
            if not path.startswith("/"):
                path = self._failsim.path_to_cwd(path)
            fixed_paths.append(path)

        self._time_dependencies.extend(fixed_paths)

    @_print_info
    def add_observation_points(self, points: List[str]):
        """TODO: Docstring for add_observation_points.

        Args:
            points (TODO): TODO

        Returns: TODO

        """
        self._observation_points.extend(points)

    @_print_info
    def add_mask_keys(
        self,
        keys: Optional[List[str]] = None,
        values: Optional[List[str]] = None,
        **kwargs,
    ):
        """TODO: Docstring for add_mask_keys.

        Args:
            **kwargs: TODO

        Kwargs:
            keys (TODO): TODO
            values (TODO): TODO

        Returns: TODO

        """
        if keys and values:
            for key, value in zip(keys, values):
                self._mask_values[key] = value
        for key, value in kwargs.items():
            self._mask_values[key] = value
