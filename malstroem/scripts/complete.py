# coding=utf-8
# -------------------------------------------------------------------------------------------------
# Copyright (c) 2016
# Developed by Septima.dk and Thomas Balstrøm (University of Copenhagen) for the Danish Agency for
# Data Supply and Efficiency. This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the Free Software Foundation,
# either version 2 of the License, or (at you option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PORPOSE. See the GNU Gene-
# ral Public License for more details.
# You should have received a copy of the GNU General Public License along with this program. If not,
# see http://www.gnu.org/licenses/.
# -------------------------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------------------------
# Modified to add additional optional hydrologic processing developed by Oriane Strouk (2026).
#
# Initial abstraction
# -------------------
# Initial abstraction before runoff generation can be accounted for at the watershed scale by
# providing a landuse raster. Two initial abstraction methods are currently implemented:
#   - Runoff Coefficient (Rational Method)
#   - SCS Curve Number
#
# A hydrologic coefficient is computed for each watershed from the landuse raster and propagated
# through the runoff network. It is then used when converting the input rainfall into the effective
# runoff volume entering each bluespot. Landuse codes must correspond to the lookup tables provided
# in tests/data/{initial_abstraction_method}_lookup.txt. Uncertainty can be evaluated by selecting one of
# the available coefficient scenarios:
#   - lower_bound
#   - default_value
#   - upper_bound
#
# Drainage
# --------
# Bluespot drainage can be simulated by providing a vector layer of sump locations with a
# "capacity" attribute (m³/s). The drainage calculation uses the bluespot hypsometry to derive
# drainage capacity curves. The simulation duration can be controlled with "simulation_duration_s".
#
# The "allow_initial_spillover" option controls how water exceeding the bluespot storage capacity
# is handled:
#   - True: excess water spills immediately and cannot be drained (conservative flash-flood
#     assumption, suitable for intense rainfall events).
#   - False: excess water may be drained according to the drainage capacity curve (continuous
#     mass-balance assumption, suitable for lower flows).
#
# Since drainage relies on the hypsometric curves, the hypsometry calculation is performed before
# the drainage step in the processing workflow.
# -------------------------------------------------------------------------------------------------

from __future__ import (absolute_import, division, print_function) #, unicode_literals)
from builtins import *

import click
import click_log

from malstroem import dem as demtool, bluespots, hydrology, io, streams, rain as raintool, network, hyps, approx
from malstroem.drainage import bluespot_drainage_io
from malstroem.vector import vectorize_labels_file_io
from ._utils import parse_filter
from osgeo import ogr, osr
import os

import logging
logger = logging.getLogger(__name__)

@click.command('complete')

@click.option('-dem', type=click.Path(exists=True), help='DEM raster file. Horisontal and vertical units must be meters')
@click.option('-outdir', type=click.Path(exists=True), help='Output directory')
@click.option('-mm', required=True, multiple=False, type=float, help='Rain incident in [mm]')
@click.option('-zresolution', required=True, type=float, help='Resolution in [m] when collecting statistics used for estimating water level for partially filled bluespots')
@click.option('-accum', is_flag=True, help='Calculate accumulated flow')
@click.option('-vector', is_flag=True, help='Vectorize bluespots and watersheds')
@click.option('-filter', help='Filter bluespots by area, maximum depth and volume. Format: '
                               '"area > 20.5 and (maxdepth > 0.05 or volume > 2.5)"')

# New options added by Oriane Strouk (2026) related to initial abstraction
# ------------------------------------------------------------------------
@click.option('-landuse', type=click.Path(exists=True), help='Landuse raster file containing integer landuse codes. Used to derive initial abstraction coefficients for each watershed.')
@click.option('-initial_abstraction_method',
              type=click.Choice(['none', 'runoff_coefficient', 'curve_number'], case_sensitive=False),
              default='none',
              help='Method used to account for initial abstraction. "none" means no initial abstraction, "runoff_coefficient" means using runoff coefficients C for the Rational Method and "curve_number" means using the SCS Curve Number method.')
@click.option('-scenario',
              type=click.Choice(['lower_bound', 'default_value', 'upper_bound'], case_sensitive=False),
              default='default_value',
              help='Uncertainty scenario used to select coefficient values from the lookup table.')

# New options added by Oriane Strouk (2026) related to drainage
# ------------------------------------------------------------- 
@click.option('-sumps', type=click.Path(exists=True), help='Sump point vector file with a flow capacity attribute')
@click.option('-simulation_duration_s', type=float, default=3600.0, show_default=True, help='Duration of the rain event in seconds.')
@click.option('-allow_initial_spillover', type=click.Choice(['YES', 'NO'], case_sensitive=False), default='YES', help='Instantly spill over water volumes exceeding the bluespot max capacity before drainage calculations.')

@click_log.simple_verbosity_option()

def process_all(dem, outdir, accum, filter, mm, zresolution, vector, landuse, initial_abstraction_method, scenario, sumps, simulation_duration_s, allow_initial_spillover):
    """Quick option to run all processes.

    \b
    Example:
    malstroem complete -mm 20 -filter "volume > 2.5" -dem dem.tif  -zresolution 0.1 -outdir ./outdir/
    """
    # Check that outdir exists and is empty
    if not os.path.isdir(outdir) or not os.path.exists(outdir) or os.listdir(outdir):
        logger.error("outdir isn't an empty directory")
        return 1
    
    # Checks and utilities added by Oriane Strouk (2026)
    # --------------------------------------------------

    # Check that landuse is provided if initial abstraction method is used
    if initial_abstraction_method.lower() != "none" and not landuse:
        logger.error("landuse must be provided when using initial abstraction method: {}".format(initial_abstraction_method))
        return 1

    # Check if explicitly requested (YES or NO) sumps must be active
    # Fetch the context directly inside the method body securely
    ctx = click.get_current_context()

    # Check if the user explicitly provided the parameter (CLI option or environment variable)
    param_source = ctx.get_parameter_source('allow_initial_spillover')
    is_explicitly_set = param_source not in (click.core.ParameterSource.DEFAULT, None)

    # Throw error if explicitly requested (YES or NO) but sumps are not active
    if is_explicitly_set and not sumps:
        logger.error("-allow_initial_spillover can only be set when a -sumps file is provided.")
        return 1

    # Convert to python boolean
    initial_spillover_bool = (allow_initial_spillover.upper() == 'YES')

    # End of checks and utilities added by Oriane Strouk (2026)
    # ---------------------------------------------------------

    outvector = os.path.join(outdir, 'malstroem.gpkg')
    ogr_drv = 'gpkg'
    ogr_dsco = []
    ogr_lco = ["SPATIAL_INDEX=NO"]
    nodatasubst = -999


    filter_function = parse_filter(filter)
    dem_reader = io.RasterReader(dem, nodatasubst=nodatasubst)
    tr = dem_reader.transform
    crs = dem_reader.crs

    logger.info('Processing')
    logger.info('   dem: {}'.format(dem))
    logger.info('   outdir: {}'.format(outdir))
    logger.info('   mm: {}mm'.format(mm))
    logger.info('   zresolution: {}m'.format(zresolution))
    logger.info('   accum: {}'.format(accum))
    logger.info('   filter: {}'.format(filter))
    logger.info('   landuse: {}'.format(landuse))
    logger.info('   initial_abstraction_method: {}'.format(initial_abstraction_method))
    logger.info('   scenario: {}'.format(scenario))
    logger.info('   sumps: {}'.format(sumps))
    logger.info('   simulation_duration_s: {}s'.format(simulation_duration_s))
    logger.info('   allow_initial_spillover: {}'.format(allow_initial_spillover))
    # Process DEM
    filled_writer = io.RasterWriter(os.path.join(outdir, 'filled.tif'), tr, crs, nodatasubst)
    flowdir_writer = io.RasterWriter(os.path.join(outdir, 'flowdir.tif'), tr, crs)
    depths_writer = io.RasterWriter(os.path.join(outdir, 'bs_depths.tif'), tr, crs)
    accum_writer = io.RasterWriter(os.path.join(outdir, 'accum.tif'), tr, crs) if accum else None

    dtmtool = demtool.DemTool(dem_reader, filled_writer, flowdir_writer, depths_writer, accum_writer)
    dtmtool.process()

    # Process bluespots
    depths_reader = io.RasterReader(depths_writer.filepath)
    flowdir_reader = io.RasterReader(flowdir_writer.filepath)
    accum_reader = io.RasterReader(accum_writer.filepath) if accum_writer else None
    pourpoint_writer = io.VectorWriter(ogr_drv, outvector, 'pourpoints', None, ogr.wkbPoint, crs, dsco=ogr_dsco, lco = ogr_lco)
    watershed_writer = io.RasterWriter(os.path.join(outdir, 'watersheds.tif'), tr, crs, 0)
    watershed_vector_writer = io.VectorWriter(ogr_drv, outvector, 'watersheds', None, ogr.wkbPolygon, crs, dsco=ogr_dsco, lco = ogr_lco) if vector else None
    labeled_writer = io.RasterWriter(os.path.join(outdir, 'bluespots.tif'), tr, crs, 0)
    labeled_vector_writer = io.VectorWriter(ogr_drv, outvector, 'bluespots', None, ogr.wkbPolygon, crs, dsco=ogr_dsco, lco = ogr_lco) if vector else None

    bluespot_tool = bluespots.BluespotTool(
        input_depths=depths_reader,
        input_flowdir=flowdir_reader,
        input_bluespot_filter_function=filter_function,
        input_accum=accum_reader,
        input_dem=dem_reader,
        output_labeled_raster=labeled_writer,
        output_labeled_vector=labeled_vector_writer,
        output_pourpoints=pourpoint_writer,
        output_watersheds_raster=watershed_writer,
        output_watersheds_vector=watershed_vector_writer
    )
    bluespot_tool.process()

    # Added by Oriane Strouk (2026).
    # ------------------------------
    # Process initial abstraction if requested
    if initial_abstraction_method.lower() != "none":
        logger.info("Calculating {} rasters from landuse".format(initial_abstraction_method))
        landuse_reader = io.RasterReader(landuse, nodatasubst=99)
        watershed_labels_reader = io.RasterReader(watershed_writer.filepath)

        watershed_initial_abstraction_coefficient_writer = io.RasterWriter(os.path.join(outdir, 'watershed_{}.tif'.format(initial_abstraction_method)), tr, crs, 0)

        hydrologic_tool = hydrology.HydrologicCoefficientTool(
            initial_abstraction_method=initial_abstraction_method,
            scenario=scenario,
            input_landuse=landuse_reader,
            input_watershed_labels=watershed_labels_reader,
            output_watershed_raster=watershed_initial_abstraction_coefficient_writer
        )
        hydrologic_tool.process()

    # Moved earlier by Oriane Strouk (2026), needed by drainage.
    # ----------------------------------------------------------
    # Hypsometry
    bluespot_reader = io.RasterReader(labeled_writer.filepath)
    pourpoints_reader = io.VectorReader(outvector, pourpoint_writer.layername)
    hyps_writer = io.VectorWriter(ogr_drv, outvector, "hypsometry", None, ogr.wkbNone, dem_reader.crs)
    hyps.bluespot_hypsometry_io(bluespot_reader, dem_reader, pourpoints_reader, zresolution, hyps_writer)

    # Added by Oriane Strouk (2026).
    # ------------------------------
    # Process drainage if requested
    drainage_reader = None
    if sumps:
        logger.info("Calculating drainage capacity from sumps")

        sumps_reader = io.VectorReader(sumps)
        sump_capacity_writer = io.RasterWriter(os.path.join(outdir, 'sump_capacity.tif'), tr, crs, 0)
        drainage_writer = io.VectorWriter(ogr_drv, outvector, "drainage", None, ogr.wkbPoint, crs, dsco=ogr_dsco, lco=ogr_lco)

        bluespot_drainage_io(
            bluespots_reader=io.RasterReader(labeled_writer.filepath),
            dem_reader=dem_reader,
            sumps_reader=sumps_reader,
            resolution=zresolution,
            pourpoints_reader=io.VectorReader(outvector, pourpoint_writer.layername),
            hyps_reader=io.VectorReader(outvector, "hypsometry"),
            drainage_writer=drainage_writer,
            output_sump_capacity_raster=sump_capacity_writer
        )
        drainage_reader = io.VectorReader(outvector, "drainage")

    # Modified by Oriane Strouk (2026) to include initial abstraction and drainage in the stream network processing.
    # --------------------------------------------------------------------------------------------------------------
    # Process pourpoints
    pourpoints_reader = io.VectorReader(outvector, pourpoint_writer.layername)
    bluespot_reader = io.RasterReader(labeled_writer.filepath)
    flowdir_reader = io.RasterReader(flowdir_writer.filepath)
    watershed_initial_abstraction_coefficient_reader = io.RasterReader(watershed_initial_abstraction_coefficient_writer.filepath) if initial_abstraction_method.lower() != "none" else None

    nodes_writer = io.VectorWriter(ogr_drv, outvector, 'nodes', None, ogr.wkbPoint, crs, dsco=ogr_dsco, lco = ogr_lco)
    streams_writer = io.VectorWriter(ogr_drv, outvector, 'streams', None, ogr.wkbLineString, crs, dsco=ogr_dsco, lco = ogr_lco)

    stream_tool = streams.StreamTool(
        input_pourpoints=pourpoints_reader,
        input_bluespots=bluespot_reader,
        input_flowdir=flowdir_reader,
        output_nodes=nodes_writer,
        output_streams=streams_writer,
        initial_abstraction_method=initial_abstraction_method,
        input_watershed_initial_abstraction_coefficient=watershed_initial_abstraction_coefficient_reader,
        input_drainage=drainage_reader
    )
    stream_tool.process()

    # Modified by Oriane Strouk (2026) to account for the effective runoff volume entering each bluespot due to initial abstraction.
    # ------------------------------------------------------------------------------------------------------------------------------
    # Calculate volumes
    nodes_reader = io.VectorReader(outvector, nodes_writer.layername)
    volumes_writer = io.VectorWriter(ogr_drv, outvector, 'initvolumes', None, ogr.wkbPoint, crs, dsco=ogr_dsco, lco = ogr_lco) 
    rain_tool = raintool.SimpleVolumeTool(nodes_reader, volumes_writer, "inputv" ,mm ,initial_abstraction_method)
    rain_tool.process()

    # Modified by Oriane Strouk (2026) to account for drainage in bluespots in the final state calculation.
    # -----------------------------------------------------------------------------------------------------
    # Process final state
    volumes_reader = io.VectorReader(outvector, volumes_writer.layername)
    events_writer = io.VectorWriter(ogr_drv, outvector, 'finalstate', None, ogr.wkbPoint, crs, dsco=ogr_dsco, lco = ogr_lco)
    calculator = network.FinalStateCalculator(volumes_reader, "inputv", events_writer, simulation_duration_s=simulation_duration_s, allow_initial_spillover=initial_spillover_bool)
    calculator.process()

    # Approximation on levels
    finalvols_reader = io.VectorReader(outvector, events_writer.layername)
    hyps_reader = io.VectorReader(outvector, hyps_writer.layername)
    levels_writer = io.VectorWriter(ogr_drv, outvector, "finallevels", None, ogr.wkbNone, dem_reader.crs)
    approx.approx_water_level_io(finalvols_reader, hyps_reader, levels_writer)    

    # Approximation on bluespots
    levels_reader = io.VectorReader(outvector, levels_writer.layername)
    final_depths_writer = io.RasterWriter(os.path.join(outdir, 'finaldepths.tif'), tr, crs)
    final_bs_writer = io.RasterWriter(os.path.join(outdir, 'finalbluespots.tif'), tr, crs, 0)
    approx.approx_bluespots_io(bluespot_reader, levels_reader, dem_reader, final_depths_writer, final_bs_writer)

    # Polygonize final bluespots
    logger.info("Polygonizing final bluespots")
    vectorize_labels_file_io(final_bs_writer.filepath, outvector, "finalbluespots", ogr_drv, ogr_dsco, ogr_lco)
    logger.info("Complete done...")