# coding=utf-8
# -------------------------------------------------------------------------------------------------
# Additional hydrologic tools developed by Oriane Strouk (2026).
#
# This module provides utilities to derive watershed hydrologic coefficients from landuse
# rasters and supports multiple initial abstraction parameterizations (runoff coefficient from the Rational
# Method, SCS Curve Number).
# -------------------------------------------------------------------------------------------------

from __future__ import (absolute_import, division, print_function) #, unicode_literals)
from builtins import *

from .algorithms import label
import numpy as np
import logging
import csv

logger = logging.getLogger(__name__)


def load_coefficient_table(csv_path):
    """Load a landuse-based hydrologic coefficient lookup table.

    The table associates each landuse code with a set of hydrologic coefficient
    values representing uncertainty bounds.

    Expected columns: landuse_code, landuse_category, lower_bound,
    default_value, upper_bound.

    Parameters
    ----------
    csv_path : str
        Path to the tab-separated lookup table.

    Returns
    -------
    dict
        Mapping {landuse_code: {"lower_bound": float, "default_value": float,
        "upper_bound": float}}.
    """

    table = {}

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            code = int(row["landuse_code"])

            table[code] = {
                "lower_bound": float(row["lower_bound"]),
                "default_value": float(row["default_value"]),
                "upper_bound": float(row["upper_bound"]),
            }

    return table


def coefficient_map_from_table(coefficient_table, scenario='default_value'):
    """Build a {landuse_code: coefficient} mapping for a chosen
    uncertainty scenario.

    Parameters
    ----------
    coefficient_table : dict
        Output of load_coefficient_table().
    scenario : str, optional
        Which bound to use: "lower_bound", "default_value", or "upper_bound".
        Default "default_value".

    Returns
    -------
    dict
        Mapping {landuse_code: coefficient}, usable directly as
        coefficient_map for landuse_to_initial_abstraction_coefficient /
        HydrologicCoefficientTool.
    """
    if scenario not in ('lower_bound', 'default_value', 'upper_bound'):
        raise ValueError(f"scenario must be 'lower_bound', 'default_value' or 'upper_bound', got {scenario!r}")

    return {landuse_code: entry[scenario] for landuse_code, entry in coefficient_table.items()}


def landuse_to_initial_abstraction_coefficient(landuse, coefficient_map, default_value=None):
    """Convert a landuse raster (integer landuse codes) to hydrologic coefficients."""

    landuse_arr = np.asarray(landuse)

    # Check that all raster codes exist in lookup table
    raster_codes = np.unique(landuse_arr)
    lookup_codes = np.array(list(coefficient_map.keys()))

    unknown_codes = np.setdiff1d(raster_codes, lookup_codes)

    if len(unknown_codes) > 0:
        raise ValueError(
            f"Unknown landuse codes found in raster: {unknown_codes.tolist()}. "
            f"Missing correspondence in lookup table."
        )

    coefficients = np.full(landuse_arr.shape, float(default_value), dtype=np.float64)

    for code, value in coefficient_map.items():
        coefficients[landuse_arr == code] = float(value)

    return coefficients

def label_mean_raster(labels, values, background=0):
    """Build a raster where each label cell receives the mean of values within its label."""
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if labels.shape != values.shape:
        raise ValueError("labels and values must have the same shape")

    flattened_labels = labels.ravel()
    flattened_values = values.ravel()
    nlabels = int(flattened_labels.max())

    sum_by_label = np.bincount(flattened_labels, weights=flattened_values, minlength=nlabels + 1)
    count_by_label = np.bincount(flattened_labels, minlength=nlabels + 1)
    means = np.zeros(nlabels + 1, dtype=np.float64)
    valid = count_by_label > 0
    means[valid] = sum_by_label[valid] / count_by_label[valid]
    means[background] = np.nan  # Set background label to NaN

    return label.set_label_to_value(labels, means)

class HydrologicCoefficientTool(object):
    """
    Compute hydrologic coefficients (runoff coefficient or curve number) for watershed rasters.

    Parameters
    ----------
    initial_abstraction_method : str
        "runoff_coefficient" or "curve_number"
    input_landuse : rasterreader
    input_watershed_labels : rasterreader
    coefficient_map : dict
        Mapping from landuse code to coefficient
    output_watershed_raster : rasterwriter
    """

    def __init__(self,
                 initial_abstraction_method,
                 scenario,
                 input_landuse,
                 input_watershed_labels,
                 output_watershed_raster):

        self.initial_abstraction_method = initial_abstraction_method
        self.scenario = scenario
        self.input_landuse = input_landuse
        self.input_watershed_labels = input_watershed_labels
        self.output_watershed_raster = output_watershed_raster

        # Load coefficient lookup table
        lookup_files = {
            "runoff_coefficient": "tests/data/runoff_coefficient_lookup.txt",
            "curve_number": "tests/data/curve_number_lookup.txt",
        }

        coefficient_table = load_coefficient_table(lookup_files[self.initial_abstraction_method])

        self.coefficient_map = coefficient_map_from_table(coefficient_table, scenario=self.scenario)

        self.logger = logging.getLogger(__name__)

    def _landuse_to_coefficient(self, landuse):
        default_values = {
            "runoff_coefficient": 0.75,
            "curve_number": 75,
        }
        return landuse_to_initial_abstraction_coefficient(landuse, self.coefficient_map, default_value=default_values[self.initial_abstraction_method])

    def process(self):
        """Compute hydrologic coefficient rasters."""

        self.logger.info(f"Running HydrologicCoefficientTool with initial_abstraction_method={self.initial_abstraction_method}")

        landuse = self.input_landuse.read()
        watershed_labels = self.input_watershed_labels.read()

        if landuse.shape != watershed_labels.shape:
            raise ValueError("Input rasters must have the same shape")

        coeff_raster = self._landuse_to_coefficient(landuse)

        self.logger.info("Computing watershed aggregated coefficients")
        watershed = label_mean_raster(watershed_labels, coeff_raster)
        self.output_watershed_raster.write(watershed)

