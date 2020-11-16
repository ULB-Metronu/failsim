"""
Module containing the class SequenceTracker
"""

from .failsim import FailSim
from .results import TrackingResult, TwissResult
from .globals import FSGlobals
from ._helpers import print_info

from typing import List, Optional, Tuple
import pandas as pd
import functools
import os


class SequenceTracker:

    """
    This class handles tracking of particles.

    Note:
        This class should not be created by the user, and should only be instantiated through build_tracker.

    Args:
        failsim: The FailSim instance to use.
        sequence_to_track: The sequence to track.
        verbose: Whether SequenceTracker should output a message each time a method is called.

    """

    def __init__(self, failsim: FailSim, sequence_to_track: str, verbose: bool = False):
        self._failsim = failsim
        self._sequence_to_track = sequence_to_track
        self._verbose = verbose

        self._time_dependencies = set()
        self._observation_points = []
        self._track_flags = ["onetable"]
        self._mask_values = {}

        self._particles = None

    @print_info("SequenceTracker")
    def twiss(self, turns: int):
        """
        Does Twiss with current sequence.
        If time dependencies have been defined, the method does a multi-turn twiss, calling the time dependies each iteration.

        Args:
            turns: Amount of turns to do multi-turn twiss.

        Returns:
            TwissResult: DataClass containing twiss data.

        """
        time_depen = []

        if len(self._time_dependencies) == 0:
            twiss_df, summ_df = self._failsim.twiss_and_summ(self._sequence_to_track)
            twiss_df["turn"] = 1

        else:
            for idx, file in enumerate(self._time_dependencies):
                # Subsistute keys for values
                with open(file, "r") as fd:
                    filedata = fd.read()
                for key, value in self._mask_values.items():
                    filedata = filedata.replace(key, value)
                with open(f"tmp_{idx}.txt", "w") as fd:
                    fd.write(filedata)

                time_depen.append(f"tmp_{idx}.txt")

            twiss_df = pd.DataFrame()
            for i in range(turns):
                self._failsim.mad_input(f"comp={i-1}")

                self._failsim.mad_input(
                    f"savebeta, label=end_{i}, place=#e, sequence={self._sequence_to_track}"
                )

                if i > 0:
                    for file in time_depen:
                        self._failsim.mad_call_file(file)

                if i == 0:
                    temp_df, summ_df = self._failsim.twiss_and_summ(
                        seq=self._sequence_to_track
                    )
                else:
                    temp_df, summ_df = self._failsim.twiss_and_summ(
                        seq=self._sequence_to_track, flags=[f"beta0=end_{i-1}"]
                    )

                temp_df["turn"] = i + 1

                twiss_df = twiss_df.append(temp_df)

                del temp_df

        for file in time_depen:
            os.remove(file)

        eps_n = self._failsim._mad.globals["par_beam_norm_emit"] * 1e-6
        nrj = self._failsim._mad.globals["nrj"]
        run_version = self._failsim._mad.globals["ver_lhc_run"]
        hllhc_version = self._failsim._mad.globals["ver_hllhc_optics"]

        return TwissResult(twiss_df, summ_df, run_version, hllhc_version, eps_n, nrj)

    @print_info("SequenceTracker")
    def track(self, turns: int = 40):
        """
        Does a tracking simulation using the current setup.

        Args:
            turns: How many turns to track.

        Returns:
            TrackingResult: Returns the resulting tracking data.

        """
        self._failsim.use(self._sequence_to_track)

        tmp_files = []
        if len(self._time_dependencies) != 0:
            time_depen = []
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

        if self._particles is not None:
            for particle in self._particles:
                self._failsim.mad_input(
                    "start, "
                    f"x = {particle[0]}"
                    f"xn = {particle[1]}"
                    f"y = {particle[2]}"
                    f"yn = {particle[3]}"
                    f"t = {particle[4]}"
                    f"tn = {particle[5]}"
                )
        else:
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

    @print_info("SequenceTracker")
    def set_particles(
        self, particles: List[Tuple[float, float, float, float, float, float]]
    ):
        """
        Sets initial starting positions for particles.

        Args:
            particles: List of particles to track. Each entry must a list with 6 values: [x, xn, y, yn, t, tn].

        """
        self._particles = particles

    @print_info("SequenceTracker")
    def add_track_flags(self, flags: List[str]):
        """
        Method for adding additional flags to the Mad-X *track* command.

        Args:
            flags: List of flags to add.

        Returns:
            SequenceTracker: Returns self

        """
        self._track_flags.append(flags)

        return self

    @print_info("SequenceTracker")
    def clear_time_dependence(self):
        """
        Clears the list of time dependence files.

        Returns:
            SequenceTracker: Returns self

        """
        self._time_dependencies = set()

        return self

    @print_info("SequenceTracker")
    def add_time_dependence(self, file_paths: List[str]):
        """
        Adds a list of files to be called on each iteration of the track.

        Args:
            file_paths: List of files to call each iteration. Paths can be either absolute or relative.

        Returns:
            SequenceTracker: Returns self

        """
        for path in file_paths:
            if not path.startswith("/"):
                path = self._failsim.path_to_cwd(path)
            self._time_dependencies.add(path)

        return self

    @print_info("SequenceTracker")
    def add_observation_points(self, points: List[str]):
        """
        Adds observation points to the track.

        Args:
            points: List of element names to observe during tracking.

        Returns:
            SequenceTracker: Returns self

        """
        self._observation_points.extend(points)

        return self

    @print_info("SequenceTracker")
    def add_mask_keys(
        self,
        keys: Optional[List[str]] = None,
        values: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Adds mask key/value pairs to replace in the time dependence files.

        Note:
            The length of keys and values must be equal, as each index in keys is to be replaced with the corresponding value at the same index in the values list.

        Note:
            Kwargs can be used in this case to do some smarter key/value pairing in case the key is a valid python parameter name. The method can therefore be used as follows:

                >>> sequence_tracker.add_mask_keys(key=value)

        Example:
            Say we have a file called *time_dependence.txt*, which looks like this:

                "Hello %s!"

            We can then specify add_mask_keys in the following manner:

                >>> sequence_tracker.add_mask_keys(keys=["%s"], values=["world"])

            Which would result the *time_dependence.txt* looking like this:

                "Hello world!"

        Args:
            keys: List of keys.
            values: List of values.

        Returns:
            SequenceTracker: Returns self

        """
        if keys and values:
            assert len(keys) == len(
                values
            ), "The length of keys and values must be equal"
            for key, value in zip(keys, values):
                self._mask_values[key] = value
        for key, value in kwargs.items():
            self._mask_values[key] = value

        return self
