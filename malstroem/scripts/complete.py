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
from __future__ import (absolute_import, division, print_function) #, unicode_literals)
from builtins import *

import click
import click_log

from malstroem import dem as demtool, bluespots, io, streams, rain as raintool, network, hyps, approx
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
@click.option('-landuse', type=click.Path(exists=True), help='Landuse raster file containing COTQ land use codes')
@click.option('-infiltration_method',
              type=click.Choice(['none', 'manning', 'runoff_coefficient'], case_sensitive=False),
              default='none',
              help='Method used to account for infiltration. "none" means no infiltration, "manning" means using Manning roughness coefficients, and "runoff_coefficient" means using runoff coefficients C for the Rational Method.')
@click.option('-sumps', type=click.Path(exists=True), help='Sump point vector file with a flow capacity attribute')
@click.option('-simulation_duration_s', type=float, default=3600.0, show_default=True, help='Duration of the rain event in seconds.')
@click.option('-allow_initial_spillover', type=click.Choice(['YES', 'NO'], case_sensitive=False), default='YES', help='Instantly spill over water volumes exceeding the bluespot max capacity before drainage calculations.')

@click_log.simple_verbosity_option()

def process_all(dem, outdir, accum, filter, mm, zresolution, vector, landuse, infiltration_method, sumps, simulation_duration_s, allow_initial_spillover):
    """Quick option to run all processes.

    \b
    Example:
    malstroem complete -mm 20 -filter "volume > 2.5" -dem dem.tif  -zresolution 0.1 -outdir ./outdir/
    """
    # Check that outdir exists and is empty
    if not os.path.isdir(outdir) or not os.path.exists(outdir) or os.listdir(outdir):
        logger.error("outdir isn't an empty directory")
        return 1
    
    # Check that landuse is provided if infiltration method is initial abstraction watershed
    if infiltration_method.lower() != "none" and not landuse:
        logger.error("landuse must be provided when using infiltration method: {}".format(infiltration_method))
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
    logger.info('   sumps: {}'.format(sumps))
    logger.info('   infiltration_method: {}'.format(infiltration_method))
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

    if infiltration_method.lower() != "none":
        logger.info("Calculating {} rasters from landuse".format(infiltration_method))
        landuse_reader = io.RasterReader(landuse, nodatasubst=nodatasubst)
        bluespot_labels_reader = io.RasterReader(labeled_writer.filepath)
        watershed_labels_reader = io.RasterReader(watershed_writer.filepath)

        bluespot_infiltration_coefficient_writer = io.RasterWriter(os.path.join(outdir, 'bluespot_{}.tif'.format(infiltration_method)), tr, crs, 0)
        watershed_infiltration_coefficient_writer = io.RasterWriter(os.path.join(outdir, 'watershed_{}.tif'.format(infiltration_method)), tr, crs, 0)

        hydrologic_tool = bluespots.HydrologicCoefficientTool(
            infiltration_method=infiltration_method,
            input_landuse=landuse_reader,
            input_bluespot_labels=bluespot_labels_reader,
            input_watershed_labels=watershed_labels_reader,
            coefficient_map=bluespots.COTQ_landuse_runoff_map() if infiltration_method == "runoff_coefficient" else bluespots.COTQ_landuse_manning_map(),
            output_bluespot_raster=bluespot_infiltration_coefficient_writer,
            output_watershed_raster=watershed_infiltration_coefficient_writer
        )
        hydrologic_tool.process()

    # Hypsometry (moved earlier, needed by drainage)
    bluespot_reader = io.RasterReader(labeled_writer.filepath)
    pourpoints_reader = io.VectorReader(outvector, pourpoint_writer.layername)
    hyps_writer = io.VectorWriter(ogr_drv, outvector, "hypsometry", None, ogr.wkbNone, dem_reader.crs)
    hyps.bluespot_hypsometry_io(bluespot_reader, dem_reader, pourpoints_reader, zresolution, hyps_writer)

# Drainage (sumps only)
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

    # Process pourpoints
    pourpoints_reader = io.VectorReader(outvector, pourpoint_writer.layername)
    bluespot_reader = io.RasterReader(labeled_writer.filepath)
    flowdir_reader = io.RasterReader(flowdir_writer.filepath)
    watershed_infiltration_coefficient_reader = io.RasterReader(watershed_infiltration_coefficient_writer.filepath) if infiltration_method.lower() != "none" else None
    bluespot_infiltration_coefficient_reader = io.RasterReader(bluespot_infiltration_coefficient_writer.filepath) if infiltration_method.lower() != "none" else None

    nodes_writer = io.VectorWriter(ogr_drv, outvector, 'nodes', None, ogr.wkbPoint, crs, dsco=ogr_dsco, lco = ogr_lco)
    streams_writer = io.VectorWriter(ogr_drv, outvector, 'streams', None, ogr.wkbLineString, crs, dsco=ogr_dsco, lco = ogr_lco)

    stream_tool = streams.StreamTool(
        input_pourpoints=pourpoints_reader,
        input_bluespots=bluespot_reader,
        input_flowdir=flowdir_reader,
        output_nodes=nodes_writer,
        output_streams=streams_writer,
        infiltration_method=infiltration_method,
        input_watershed_infiltration_coefficient=watershed_infiltration_coefficient_reader,
        input_bluespot_infiltration_coefficient=bluespot_infiltration_coefficient_reader,
        input_drainage=drainage_reader
    )
    stream_tool.process()

    # Calculate volumes
    nodes_reader = io.VectorReader(outvector, nodes_writer.layername)
    volumes_writer = io.VectorWriter(ogr_drv, outvector, 'initvolumes', None, ogr.wkbPoint, crs, dsco=ogr_dsco, lco = ogr_lco) 
    rain_tool = raintool.SimpleVolumeTool(nodes_reader, volumes_writer, "inputv" ,mm ,infiltration_method)
    rain_tool.process()

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