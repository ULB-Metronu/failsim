from .failsim import FailSim
from .results import TrackingResult
import functools


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
    def track(self):
        """TODO: Docstring for track.
        Returns: TODO

        """
        self._failsim.use(self._sequence_to_track)
        self._failsim.mad_input("track, onetable; start; run, turns=40; endtrack")
        track_df = self._failsim._mad.table["trackone"].dframe()
        twiss_df, summ_df = self._failsim.twiss_and_summ(self._sequence_to_track)
        run_version = self._failsim._mad.globals["ver_lhc_run"]
        hllhc_version = self._failsim._mad.globals["ver_hllhc_optics"]

        eps_n = self._failsim._mad.globals["par_beam_norm_emit"]
        nrj = self._failsim._mad.globals["nrj"]

        res = TrackingResult(
            twiss_df, summ_df, track_df, run_version, hllhc_version, eps_n, nrj
        )

        return res
