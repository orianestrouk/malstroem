# coding=utf-8
# -------------------------------------------------------------------------------------------------
# Copyright (c) 2020
# Developed by Septima.dk. This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 2 of the License, or (at you option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PORPOSE. See the GNU Gene-
# ral Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not,
# see http://www.gnu.org/licenses/.
# -------------------------------------------------------------------------------------------------
from __future__ import (absolute_import, division, print_function)
from builtins import *

import numpy as np
import logging
from collections import namedtuple

from malstroem.algorithms.label import label_stats, label_data
from malstroem.hyps import histogram_bins, Histogram, HistogramBinsInfo, cumulative_volume, hypsometrystats_from_flatdict
from malstroem.scripts._utils import rasterize_sumps

logger = logging.getLogger(__name__)

DrainageStats = namedtuple("DrainageStats", [
    "zhistogram",
    "wetted_area_cumulative",
    "diffuse_capacity_cumulative",
    "sump_capacity_cumulative"
])


def landuse_to_diffuse_rate(landuse, diffuse_rate_map, default_value=0.0):
    """Convert a landuse raster to a diffuse drainage rate raster.

    Mirrors landuse_to_manning: each landuse code is mapped to a diffuse
    drainage rate (m3/s per m2 of wetted surface).

    Parameters
    ----------
    landuse : array-like
        Raster of land-use codes.
    diffuse_rate_map : dict
        Mapping of land-use codes to diffuse drainage rate values.
    default_value : float, optional
        Diffuse rate for codes not found in the mapping. Default is 0.0.

    Returns
    -------
    np.ndarray
        Array of diffuse drainage rate values with the same shape as `landuse`.
    """
    landuse_arr = np.asarray(landuse)
    diffuse_rate = np.full(landuse_arr.shape, float(default_value), dtype=np.float64)

    for code, value in diffuse_rate_map.items():
        mask = np.isclose(landuse_arr, float(code))
        diffuse_rate[mask] = float(value)

    return diffuse_rate

def COTQ_landuse_diffuse_rate_map():
    """Return diffuse drainage rate (m3/s per m2) per COTQ landuse code.

    Based on typical infiltration rate orders of magnitude (mm/h) from
    USDA/SWMM literature, converted to m3/s/m2 (rate_mm_h / 3.6e6).
    """
    return {
        1: 0.0,           # Artificialisé dense -> impervious, no infiltration
        3: 2.5e-6,         # Couvert arboré -> forest, ~9mm/h
        4: 1.4e-6,         # Végétation basse -> shrub, ~5mm/h
        5: 0.3e-6,         # Terre -> bare soil, ~1mm/h
        6: 0.0,            # Roche -> rock, no infiltration
        7: 0.0,            # Route -> impervious
        8: 0.2e-6,         # Artificialisé -> low/medium urban, ~0.7mm/h
        9: 0.0,            # Eau -> open water, no infiltration
        10: 0.8e-6,        # Milieu humide -> wetland, ~3mm/h
        11: 0.9e-6,        # Terre avec végétation basse -> pasture, ~3mm/h
        12: 1.4e-6,        # Roche avec végétation basse -> shrub, ~5mm/h
    }

def drainage_capacity_by_volume(drain_stats, hyps_stats, cell_area):
    """Reindex drainage capacity curve from z-axis to volume-axis.

    Uses the hypsometry of the same bluespot to map each z-bin to a
    cumulative volume, producing a (volume, drainage_capacity) curve
    directly usable by FinalStateCalculator.

    Parameters
    ----------
    drain_stats : DrainageStats
        Output of bluespot_drainage_stats for one bluespot.
    hyps_stats : HypsometryStats
        Output of bluespot_hypsometry_stats for the same bluespot.
    cell_area : float
        Cell area in m2.

    Returns
    -------
    volumes : np.ndarray
        Cumulative volume (m3) at each z-bin.
    capacity : np.ndarray
        Total drainage capacity (m3/s) at each z-bin, indexed on the same axis.
    """
    volumes = cumulative_volume(hyps_stats.zhistogram, cell_area)
    capacity = drainage_capacity_curve(drain_stats)
    return volumes, capacity

def bluespot_drainage_io(bluespots_reader, dem_reader, sumps_reader, resolution,
                          pourpoints_reader, hyps_reader,
                          drainage_writer,
                          output_sump_capacity_raster,
                          landuse_reader=None,
                          diffuse_rate_map=None,
                          default_diffuse_rate=0.0):
    """Calculate drainage capacity per water level for each bluespot.

    Parameters
    ----------
    bluespots_reader : rasterreader
        Bluespot label raster.
    dem_reader : rasterreader
        DEM raster (same shape as bluespots).
    sumps_reader : vectorreader
        Point vector layer with sump (puisard) locations.
    resolution : float
        Z resolution for histogram bins (same convention as hypsometry resolution).
    pourpoints_reader : vectorreader
        Pourpoints, one feature per bluespot, used as the base feature to append properties.
    hyps_reader : vectorreader
        Hypsometry stats for each bluespot, used to convert z to volume.
    drainage_writer : vectorwriter
        Output writer for drainage stats per bluespot.
    output_sump_capacity_raster : rasterwriter
        Output writer for the sump capacity raster.
    landuse_reader : rasterreader, optional
        Landuse raster (same shape as bluespots), used to derive diffuse drainage rate.
    diffuse_rate_map : dict, optional
        Mapping of landuse code to diffuse drainage rate (m3/s per m2).
    default_diffuse_rate : float, optional
        Diffuse rate for landuse codes not found in diffuse_rate_map. Default 0.0.
    """
    assert bluespots_reader.shape == dem_reader.shape, "Dimension mismatch between dem and bluespot rasters"
    if landuse_reader is not None:
        assert bluespots_reader.shape == landuse_reader.shape, \
            "Dimension mismatch between landuse and bluespot rasters"
    for r0, r1 in zip(bluespots_reader.resolution, dem_reader.resolution):
        np.testing.assert_almost_equal(r0, r1, err_msg="Resolution mismatch between dem and bluespot rasters")

    cell_area = bluespots_reader.resolution[0] * bluespots_reader.resolution[1]

    logger.debug("Reading input rasters")
    bluespotlabels = bluespots_reader.read()
    dem = dem_reader.read()

    logger.debug("Rasterizing sump points to capacity grid")
    sump_capacity = rasterize_sumps(sumps_reader,
        dem_reader.transform, dem_reader.shape, dem_reader.crs,
        output_sump_capacity_raster
    )

    if diffuse_rate_map is not None and landuse_reader is not None:
        logger.debug("Deriving diffuse drainage rate from landuse")
        landuse = landuse_reader.read()
        diffuse_rate_raster = landuse_to_diffuse_rate(landuse, diffuse_rate_map, default_diffuse_rate)
        diffuse_capacity_raster = diffuse_rate_raster * cell_area
    else:
        logger.debug("No diffuse rate map provided, diffuse drainage set to 0")
        diffuse_capacity_raster = np.zeros(dem_reader.shape, dtype=np.float64)

    pourpoints_index = {gjn['properties']['bspot_id']: gjn for gjn in pourpoints_reader.read_geojson_features()}
    labels_max = np.max(bluespotlabels)

    hyps_index = {f['properties']['bspot_id']: f['properties']
                  for f in hyps_reader.read_geojson_features()}

    for bs_id, stats in bluespot_drainage_stats(
    bluespotlabels, dem, diffuse_capacity_raster, sump_capacity,
    resolution, labels_max, background=0):
        if bs_id == 0:  # skip background
            continue
        pp = pourpoints_index[bs_id]
        hyps_props = hyps_index.get(bs_id)
        if hyps_props:
            hyps_stats = hypsometrystats_from_flatdict(hyps_props)
            add_props = drainagestats_to_flatdict(stats, hyps_stats, cell_area)
        else:
            # Junction node without bluespot — no drainage curve
            add_props = {}
        pp["properties"].update(add_props)

    logger.debug("Writing features")
    drainage_writer.write_geojson_features(pourpoints_index.values())


def bluespot_drainage_stats(bluespotlabels, dem, diffuse_capacity_raster, sump_capacity_raster,
                             resolution, labels_max=None, background=0):
    """Yield, for each bluespot label, the z-histogram of wetted cells, cumulative
    diffuse drainage capacity, and cumulative sump capacity submerged at each water level.

    Both diffuse_capacity_raster and sump_capacity_raster are expected to already
    hold per-cell capacity values (m3/s), summed cell by cell into the same z-bins
    used for the wetted area histogram.
    """
    if labels_max is None:
        labels_max = np.max(bluespotlabels)

    logger.debug("Calculate z stats for each bluespot")
    label_z_stats = label_stats(dem, bluespotlabels, labels_max)

    logger.debug("Collecting per-label dem, diffuse capacity, and sump capacity values")
    label_dem_values = label_data(dem, bluespotlabels, labels_max, background=background)
    label_diffuse_values = label_data(diffuse_capacity_raster, bluespotlabels, labels_max, background=background)
    label_sump_values = label_data(sump_capacity_raster, bluespotlabels, labels_max, background=background)

    logger.debug("Calculating histograms")
    for label, dem_values in enumerate(label_dem_values):
        if label == background:
            bins = HistogramBinsInfo(0, 0, 0, -1)
            wetted_counts = []
            diffuse_capacity_per_bin = np.array([])
            sump_capacity_per_bin = np.array([])
        else:
            bins = histogram_bins(label_z_stats[label]["min"], label_z_stats[label]["max"], resolution)
            wetted_counts, _ = np.histogram(dem_values, bins.num_bins, (bins.lower_bound, bins.upper_bound))

            diffuse_values = label_diffuse_values[label]
            sump_values = label_sump_values[label]

            if len(dem_values):
                diffuse_capacity_per_bin, _ = np.histogram(
                    dem_values, bins.num_bins, (bins.lower_bound, bins.upper_bound), weights=diffuse_values
                )
                sump_capacity_per_bin, _ = np.histogram(
                    dem_values, bins.num_bins, (bins.lower_bound, bins.upper_bound), weights=sump_values
                )
            else:
                diffuse_capacity_per_bin = np.zeros(bins.num_bins)
                sump_capacity_per_bin = np.zeros(bins.num_bins)

        wetted_area_cumulative = np.cumsum(wetted_counts)
        diffuse_capacity_cumulative = np.cumsum(diffuse_capacity_per_bin)
        sump_capacity_cumulative = np.cumsum(sump_capacity_per_bin)

        yield label, DrainageStats(
            Histogram(wetted_counts, bins),
            wetted_area_cumulative,
            diffuse_capacity_cumulative,
            sump_capacity_cumulative
        )


def drainage_capacity_curve(stats):
    """Total drainage capacity per bin: diffuse + sump, already cell-by-cell summed."""
    return stats.diffuse_capacity_cumulative + stats.sump_capacity_cumulative


def drainagestats_to_flatdict(drain_stats, hyps_stats, cell_area):
    """Serialize drainage stats indexed on volume axis.

    Parameters
    ----------
    drain_stats : DrainageStats
    hyps_stats : HypsometryStats
        Hypsometry of the same bluespot — used to convert z to volume.
    cell_area : float
    """
    volumes, capacity = drainage_capacity_by_volume(drain_stats, hyps_stats, cell_area)
    return {
        "drain_volumes":               "|".join([str(x) for x in volumes]),
        "drain_capacity_curve":        "|".join([str(x) for x in capacity]),
        "drain_diffuse_capacity_cum":  "|".join([str(x) for x in drain_stats.diffuse_capacity_cumulative]),
        "drain_sump_capacity_cum":     "|".join([str(x) for x in drain_stats.sump_capacity_cumulative]),
        "drain_hist_num_bins":         drain_stats.zhistogram.bins.num_bins,
        "drain_hist_lower_bound":      drain_stats.zhistogram.bins.lower_bound,
        "drain_hist_upper_bound":      drain_stats.zhistogram.bins.upper_bound,
        "drain_hist_resolution":       drain_stats.zhistogram.bins.resolution,
    }