from __future__ import annotations
from typing import TYPE_CHECKING, List
import os
import logging
import pkg_resources
import pandas as _pd
import scipy.interpolate
if TYPE_CHECKING:
    from .sequence import LHCSequence

APERTURE_LHC_APER1 = 0.022
APERTURE_LHC_APER2 = 0.01715
APERTURE_LHC_APER3 = 0.022
APERTURE_LHC_APER4 = 0.022


class Aperture:
    def __init__(self, sequence: LHCSequence, aperture_model: str):
        """

        Args:
            sequence:
            aperture_model:
        """
        self._sequence = sequence

        self._arc_starts = []
        self._arc_ends = []
        for ip in range(1, 8 + 1):
            self._arc_starts.append(self._find_location(f's.ds.r{ip}.b1'))
            self._arc_ends.append(self._find_location(f'e.ds.l{(ip % 8 + 1)}.b1'))

        self._ldb_apertures = _pd.read_parquet(os.path.join(pkg_resources.resource_filename('failsim', "data"),
                                                            aperture_model)).query("BEAM != 'B2'").sort_values(
            by='S_FROM_IP1').reset_index()
        self._ldb_apertures_a = scipy.interpolate.interp1d(self._ldb_apertures['S_FROM_IP1'],
                                                           self._ldb_apertures['ELEM_ELLIPSE_PARAM_A'],
                                                           kind='next', bounds_error=False, fill_value=(
                self._ldb_apertures.at[0, 'ELEM_ELLIPSE_PARAM_A'],
                self._ldb_apertures.iloc[-1]['ELEM_ELLIPSE_PARAM_A']),
                                                           assume_sorted=True)
        self._ldb_apertures_b = scipy.interpolate.interp1d(self._ldb_apertures['S_FROM_IP1'],
                                                           self._ldb_apertures['ELEM_ELLIPSE_PARAM_B'],
                                                           kind='next', bounds_error=False, fill_value=(
                self._ldb_apertures.at[0, 'ELEM_ELLIPSE_PARAM_B'],
                self._ldb_apertures.iloc[-1]['ELEM_ELLIPSE_PARAM_B']),
                                                           assume_sorted=True)
        self._ldb_apertures_c = scipy.interpolate.interp1d(self._ldb_apertures['S_FROM_IP1'],
                                                           self._ldb_apertures['ELEM_ELLIPSE_PARAM_C'],
                                                           kind='next', bounds_error=False, fill_value=(
                self._ldb_apertures.at[0, 'ELEM_ELLIPSE_PARAM_C'],
                self._ldb_apertures.iloc[-1]['ELEM_ELLIPSE_PARAM_C']),
                                                           assume_sorted=True)
        self._ldb_apertures_d = scipy.interpolate.interp1d(self._ldb_apertures['S_FROM_IP1'],
                                                           self._ldb_apertures['ELEM_ELLIPSE_PARAM_D'],
                                                           kind='next', bounds_error=False, fill_value=(
                self._ldb_apertures.at[0, 'ELEM_ELLIPSE_PARAM_D'],
                self._ldb_apertures.iloc[-1]['ELEM_ELLIPSE_PARAM_D']),
                                                           assume_sorted=True)

    @staticmethod
    def convert_layout_database_aperture_files(files: List[str], output_filename: str = 'aperture.parquet'):
        _ = _pd.concat((_pd.read_csv(f) for f in files)).query("BEAM != 'B2'").sort_values(
            by='S_FROM_IP1').reset_index()
        _.to_parquet(
            output_filename)

    def _is_in_arc(self, name: str) -> bool:
        in_arc = False
        for s, e in zip(self._arc_starts, self._arc_ends):
            if s <= self._find_location(name) <= e:
                in_arc = True
        return in_arc

    def _find_location(self, name: str) -> float:
        sequence = self._sequence.configuration['sequence_to_track']
        try:
            return self._sequence.mad.sequence[sequence].expanded_element_positions()[
                self._sequence.mad.sequence[sequence].expanded_element_names().index(name + '[0]')
            ]
        except ValueError:
            return self._sequence.mad.sequence[sequence].expanded_element_positions()[
                self._sequence.mad.sequence[sequence].expanded_element_names().index(name)
            ]

    def __call__(self, element_name: str):
        if self._is_in_arc(element_name):
            logging.info(f"Default aperture for element {element_name}: 'lhc', {APERTURE_LHC_APER1}, "
                         f"{APERTURE_LHC_APER2}, {APERTURE_LHC_APER3}, {APERTURE_LHC_APER4}")
            return 'lhc', APERTURE_LHC_APER1, APERTURE_LHC_APER2, APERTURE_LHC_APER3, APERTURE_LHC_APER4
        else:
            location = self._find_location(element_name)
            a = float(self._ldb_apertures_a(location))
            b = float(self._ldb_apertures_b(location))
            c = float(self._ldb_apertures_c(location))
            d = float(self._ldb_apertures_d(location))
            logging.warning(f"Aperture for element {element_name}: 'rectellipse', {a}, {b}, {c}, {d}")
            return 'rectellipse', a, b, c, d

