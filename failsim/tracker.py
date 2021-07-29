"""
Module containing the class Tracker
"""

from .failsim import FailSim
from .results import TrackingResult, TwissResult
from .helpers import print_info
from .globals import FailSimGlobals

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import functools
import multiprocessing
import os
import gc
import re


class Tracker:

    """
    This class handles tracking of particles.

    Note:
        This class should not be created by the user, and should only be instantiated through build_tracker.

    Args:
        failsim: The FailSim instance to use.
        sequence_to_track: The sequence to track.
        verbose: Whether Tracker should output a message each time a method is called.

    """

    def __init__(self, failsim: FailSim, sequence_to_track: str, verbose: bool = False):
        self._failsim = failsim
        self._sequence_to_track = sequence_to_track
        self._verbose = verbose

        self._time_dependencies = set()
        self._observation_points = []
        self._track_flags = []
        self._mask_values = {}

        self._particles = None

    @print_info("Tracker")
    def twiss(self, turns: Optional[int] = None):
        """
        Does Twiss with current sequence.
        If time dependencies have been defined, the method does a multi-turn twiss, calling the time dependies each iteration.

        Note:
            Time dependencies will be delayed by a single turn, to ensure a clean first turn that can be used for reference. This will be compensated internally by subtracting 1 from each turn number, such that the reference turn will be turn -1, and the time dependencies will be called from turn 0.

        Args:
            turns: Amount of turns to do multi-turn twiss.

        Returns:
            TwissResult: DataClass containing twiss data.

        """
        time_depen = []

        if len(self._time_dependencies) == 0 or turns is None:
            twiss_df, summ_df = self._failsim.twiss_and_summ(self._sequence_to_track)
            twiss_df["turn"] = 0

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
            for i in range(turns+1):
                self._failsim.mad_input(f"comp={i-1}")

                self._failsim.mad_input(
                    f"savebeta, label=end_{i}, place=#e, sequence={self._sequence_to_track}"
                )

                if i > 0:
                    for file in time_depen:
                        self._failsim.mad_call_file(file)

                try:
                    if i == 0:
                        temp_df, summ_df = self._failsim.twiss_and_summ(
                            seq=self._sequence_to_track
                        )
                    else:
                        temp_df, summ_df = self._failsim.twiss_and_summ(
                            seq=self._sequence_to_track, flags=[f"beta0=end_{i-1}"]
                        )
                except KeyError:
                    break

                temp_df["turn"] = i - 1

                twiss_df = twiss_df.append(temp_df)

                del temp_df
                gc.collect()

        for file in time_depen:
            os.remove(file)

        try:
            eps_n = self._failsim.mad.globals["par_beam_norm_emit"] * 1e-6
        except KeyError:
            ...

        run_version = self._failsim.mad.globals["ver_lhc_run"]
        hllhc_version = self._failsim.mad.globals["ver_hllhc_optics"]
        beam=dict(self._failsim.mad.sequence[self._sequence_to_track].beam.items())

        return TwissResult(twiss_df, summ_df, run_version, hllhc_version, eps_n, beam)

    def fork(self):
        new_fs = self._failsim.fork()
        new_tracker = Tracker(new_fs, self._sequence_to_track, self._verbose)
        new_tracker.set_particles(self._particles)
        new_tracker.set_track_flags(self._track_flags)
        new_tracker.set_observation_points(self._observation_points)
        new_tracker.set_time_dependence(self._time_dependencies)
        new_tracker.add_mask_keys(keys=self._mask_values.keys(),
                values=self._mask_values.values())
        return new_tracker

    @print_info("Tracker")
    def track_multithreaded(self, turns: int = 40, nthreads: int = 8):
        # Basic assertions
        assert (nthreads%2 == 0), \
            ("nthreads must be a multiple of 2")
        # assert (self._particles != None) and (len(self._particles) != 0), \
            # ("Multithreaded tracking only works with a list of particles to track")
        ninstances = nthreads//2 # Number of MAD-X instances to create

        def track_thread(tracker, procnum, output_path):
            """Does single tracking pass"""
            output = os.path.join(output_path, str(procnum))
            res = tracker.track(turns)
            res.save_data(output)
            print(f"Track {procnum} done!")

        # Define and create temporary output path
        output_path = f"temp/{hash(self)}/track/"
        os.makedirs(output_path, exist_ok=True)

        # Split particles and create jobs
        jobs = []
        for idx, part in enumerate(np.array_split(self._particles, ninstances)):
            new_tracker = self.fork()
            new_tracker.set_particles(part)
            proc = multiprocessing.Process(target=track_thread,
                    args=(new_tracker, idx, output_path))
            proc.start()
            jobs.append(proc)

        # Wait for jobs to finish
        for proc in jobs:
            proc.join()

        for f in os.listdir(output_path):
            df_temp = pd.read_parquet(output_path+f+"/track.parquet")
            try:
                df_temp["number"] += max(df["number"])
                df = df.append(df_temp)
            except NameError:
                df = df_temp

        res = TrackingResult.load_data(output_path+"0")
        res.track_df = df
        return res

    @print_info("Tracker")
    def track(self, turns: int = 40):
        """
        Does a tracking simulation using the current setup.

        Does the following:
        - Goes through each time dependency file and replaces all keys specified by add_mask_keys by their corresponding values, and saves the new files in temporary files.
        - Creates tr$macro for calling files each turn.
        - Add flags specified by set_track_flags.
        - Adds particles specified by set_particles.
        - Adds observation points specified by set_observation_points.
        - Runs track.

        Args:
            turns: How many turns to track.

        Returns:
            TrackingResult: Returns the resulting tracking data.

        """
        self._failsim.use(self._sequence_to_track)

        tmp_files = []
        if len(self._time_dependencies) != 0:
            # Hash for keeping temporary tracker files
            # seperate per tracker instance
            unique_hash = hash(self)
            time_depen = []
            for idx, file in enumerate(self._time_dependencies):
                # Substitute keys for values
                with open(file, "r") as fd:
                    filedata = fd.read()
                for key, value in self._mask_values.items():
                    filedata = filedata.replace(key, value)
                file_name = f"temp/{unique_hash}_tmp_{idx}.txt"
                with open(file_name, "w") as fd:
                    fd.write(filedata)

                tmp_files.append(file_name)
                time_depen.append(f"call, file='{tmp_files[-1]}';")

            # Create tr$macro
            self._track_flags.append("onepass")
            self._track_flags.append("update")
            time_depen = " ".join(time_depen)
            self._failsim.mad_input(
                f"tr$macro(turn): macro = {{comp=turn; {time_depen} }}"
            )

        twiss_df, summ_df = self._failsim.twiss_and_summ(self._sequence_to_track)
        twiss_df["turn"] = 0
        run_version = self._failsim.mad.globals["ver_lhc_run"]
        hllhc_version = self._failsim.mad.globals["ver_hllhc_optics"]

        self._track_flags.append("onetable")
        flags = ", ".join(self._track_flags)
        self._failsim.mad_input(f"track, {flags}")

        if self._particles is not None:
            for particle in self._particles:
                self._failsim.mad_input(
                    "start, "
                    f"x = {particle[0]},"
                    f"px = {particle[1]},"
                    f"y = {particle[2]},"
                    f"py = {particle[3]},"
                    f"t= {particle[4]},"
                    f"pt = {particle[5]}"
                )
        else:
            self._failsim.mad_input("start")

        for obs in self._observation_points:
            self._failsim.mad_input(f"observe, place='{obs}'")

        self._failsim.mad_input(f"run, turns={turns}")
        self._failsim.mad_input("endtrack")
        if len(self._time_dependencies) != 0:
            self._failsim.mad_input("exec tr$macro(0);")

        track_df = self._failsim.mad.table["trackone"].dframe()

        try:
            eps_n = self._failsim.mad.globals["par_beam_norm_emit"] * 1e-6
        except KeyError:
            eps_n = 2.5e-6

        loss_df = None
        if "trackloss" in self._failsim.mad.table.keys():
            try:
                loss_df = self._failsim.mad.table["trackloss"].dframe()
            except KeyError:
                pass

        res = TrackingResult(
            twiss_df,
            summ_df,
            track_df,
            run_version,
            hllhc_version,
            eps_n,
            dict(self._failsim.mad.sequence[self._sequence_to_track].beam.items()),
            loss_df=loss_df,
        )

        for file in tmp_files:
            try:
                os.remove(file)
            except FileNotFoundError:
                continue

        return res

    @print_info("Tracker")
    def save_values(self, regex: str):
        """Saves values for all elements matching given regex in a dictionary.

        Args:
            regex: Regex that will be matched against each element in sequence.

        Returns:
            Dict: Dictionary containing the saved values.

        """
        reg = re.compile(regex)
        mad_vars = self._failsim.mad.get_variables_dicts()["all_variables_val"]
        el_vars = {x: mad_vars[x] for x in mad_vars.keys() if len(reg.findall(x)) != 0}
        return el_vars

    @print_info("Tracker")
    def restore_values(self, val_dict: dict):
        """Restores values from a dictionary created from a previous call to save_values.

        Args:
            val_dict: Dictionary containing saved values.

        Returns:
            Tracker: Returns self

        """
        for var, val in val_dict.items():
            self._failsim.mad_input(f"{var} = {val}")

    @print_info("Tracker")
    def set_particles(
        self, particles: List[Tuple[float, float, float, float, float, float]]
    ):
        """
        Sets initial starting positions for particles.

        Args:
            particles: List of particles to track. Each entry must a list with 6 values: [x, px, y, py, t, pt].

        """
        self._particles = particles

    @print_info("Tracker")
    def set_track_flags(self, flags: List[str]):
        """
        Method for adding additional flags to the Mad-X *track* command.

        Args:
            flags: List of flags to add.

        Returns:
            Tracker: Returns self

        """
        self._track_flags = flags

        return self

    @print_info("Tracker")
    def clear_time_dependence(self):
        """
        Clears the list of time dependence files.

        Returns:
            Tracker: Returns self

        """
        self._time_dependencies = set()

        return self

    @print_info("Tracker")
    def set_time_dependence(self, file_paths: List[str]):
        """
        Sets a list of files to be called on each iteration of the track.

        Args:
            file_paths: List of files to call each iteration. Paths can be either absolute or relative.

        Returns:
            Tracker: Returns self

        """
        self._time_dependencies = []
        for path in file_paths:
            if not path.startswith("/"):
                path = self._failsim.path_to_cwd(path)
            self._time_dependencies.append(path)

        return self

    @print_info("Tracker")
    def set_observation_points(self, points: List[str]):
        """
        Adds observation points to the track.

        Args:
            points: List of element names to observe during tracking.

        Returns:
            Tracker: Returns self

        """
        self._observation_points = points

        return self

    @print_info("Tracker")
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
            Tracker: Returns self

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
