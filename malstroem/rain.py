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
# Modified by Oriane Strouk (2026) to add optional rainfall-runoff transformation based on various
# initial abstraction methods.
#
# Effective rainfall volumes are computed from total rainfall by accounting for initial abstraction
# and infiltration losses. The implemented approaches include landuse-based runoff coefficients
# and the SCS Curve Number method.
# -------------------------------------------------------------------------------------------------

from __future__ import (absolute_import, division, print_function) #, unicode_literals)
from enum import Enum
from builtins import *

from .network import Network
from .algorithms import label
import logging

def curve_number_runoff(rain_mm, cn):
    """Compute effective rainfall using the SCS Curve Number method.

    Parameters
    ----------
    rain_mm : float
        Total rainfall depth [mm].
    cn : float
        Curve Number.

    Returns
    -------
    float
        Direct runoff depth [mm].
    """

    if cn <= 0 or cn > 100:
        raise ValueError(f"Curve Number must be between 0 and 100, got {cn}")

    S = 25400.0 / cn - 254.0
    Ia = 0.2 * S

    if rain_mm <= Ia:
        return 0.0

    Q = (rain_mm - Ia)**2 / (rain_mm + 0.8 * S)

    return Q

class SimpleVolumeTool(object):
    """Calculates water volume for each watershed for a simple rain event of x mm on each cell

    Parameters
    ----------
    input_nodes : vectorreader
        Nodes
    output_volumedata : vectorwriter
        Writes the output volumes data
    output_volume_attribute: str
        Name of attribute which calculated water volume is written to
    rainmm : float
        Rain (in mm).
    infiltration_method : str
        Infiltration method.

    Attributes
    ----------
    input_nodes : list
        Nodes
    output_volumedata : vectorwriter
        Writes the output volumes data
    output_volume_attribute: str
        Name of attribute which calculated water volume is written to
    """

    def __init__(self, input_nodes, output_volumedata, output_volume_attribute, rainmm, infiltration_method):
        self.input_nodes = input_nodes
        self.output_volumedata = output_volumedata
        self.output_volume_attribute = str(output_volume_attribute)
        self.rainmm = float(rainmm)
        self.infiltration_method = infiltration_method
        self.logger = logging.getLogger(__name__)

    def process(self):
        """Process

        Returns
        -------
        None
        """

        self.logger.info("Reading input nodes")
        all_nodes = self.input_nodes.read_geojson_features()
        
        self.logger.info("Calculating rain volumes")
        self.logger.info("  {}mm".format(self.rainmm))
        for node in all_nodes:
            props = node["properties"]
            area = float(props['wshed_area'])

        # Added by Oriane Strouk (2026).
        # ------------------------------
        # Compute effective rainfall based on infiltration method

            if self.infiltration_method == "none":
                effective_rain_mm = self.rainmm
            elif self.infiltration_method == "runoff_coefficient":
                runoff_coeff = props.get('wshed_ia_coeff')
                if runoff_coeff is None:
                    effective_rain_mm = self.rainmm
                else:
                    effective_rain_mm = self.rainmm * float(runoff_coeff)
            elif self.infiltration_method == "curve_number":
                curve_number = props.get('wshed_ia_coeff')
                if curve_number is None:
                    effective_rain_mm = self.rainmm
                else:
                    cn = float(curve_number)
                    effective_rain_mm = curve_number_runoff(self.rainmm, cn)

            effective_wshed_water_vol = area * effective_rain_mm * 0.001
            props[self.output_volume_attribute] = effective_wshed_water_vol  # Water vol from local catchment

        # End of added by Oriane Strouk (2026).
        # ------------------------------

        self.logger.info("Writing output")
        self.output_volumedata.write_geojson_features(all_nodes)

        self.logger.info("Done")


class Unit(Enum):
    mm = 1,
    l = 2,
    m3 = 3

class RasterVolumeTool(object):
    """Calculates water volume for each watershed defined by a raster with cells giving input water per cell

    Parameters
    ----------
    input_nodes : vectorreader
        Nodes
    input_bluespots : rasterreader
        Bluespot ids raster
    input_precip : rasterreader
        Precipitation raster
    input_precip_unit: Unit
        Unit of cell value 
    output_volumedata : vectorwriter
        Writes the output volumes data
    output_volume_attribute: str
        Name of attribute which calculated water volume is written to

    Attributes
    ----------
    input_nodes : list
        Nodes
    output_volumedata : vectorwriter
        Writes the output volumes data
    output_volume_attribute: str
        Name of attribute which calculated water volume is written to
    """

    def __init__(self, input_nodes, input_bluespots, input_precip, input_precip_unit, output_volumedata, output_volume_attribute):
        self.input_nodes = input_nodes
        self.output_volumedata = output_volumedata
        self.output_volume_attribute = str(output_volume_attribute)
        self.bluespotids = input_bluespots
        self.precipraster = input_precip
        self.precipunit = input_precip_unit
        self.logger = logging.getLogger(__name__)

    def process(self):
        """Process

        Returns
        -------
        None
        """

        self.logger.info("Reading input data")
        all_nodes = self.input_nodes.read_geojson_features()
        
        # Input rasters
        transform = self.bluespotids.transform
        cell_width = abs(transform[1])
        cell_height = abs(transform[5])
        cell_area = cell_width * cell_height
        bluespot_labels_raster = self.bluespotids.read()
        water_raster = self.precipraster.read()
        assert bluespot_labels_raster.shape == water_raster.shape, "Rasters must have same extent and resolution"

        self.logger.info("Calculating water volumes")
        water_stats = label.label_stats(water_raster, bluespot_labels_raster)
        
        # Factor to convert from raster cell unit to m3
        conversion_factor = 1
        if self.precipunit is Unit.mm:
            self.logger.info(f"Cell unit is [mm]")
            conversion_factor = 1e-3 * cell_area    
        elif self.precipunit is Unit.l:
            self.logger.info(f"Cell unit is [l]")
            conversion_factor = 1e-3
        elif self.precipunit is Unit.m3:
            self.logger.info(f"Cell unit is [m3]")
            conversion_factor = 1
        else:
            raise NotImplementedError(F"Unit {self.precipunit} is not implemented")

        self.logger.info(f"Using conversion factor {conversion_factor} to go from cell values to m3")
        water_volumes = water_stats["sum"] * conversion_factor

        max_id = len(water_volumes) - 1
        for node in all_nodes:
            props = node["properties"]
            bsid = props["nodeid"]
            wshed_water_vol = 0
            # Do we have a value for this node? (junctions do not have a bluespot id and hence no label)
            if bsid <= max_id:
                wshed_water_vol = water_volumes[bsid]
            props[self.output_volume_attribute] = wshed_water_vol  # Water vol from local catchment

        self.logger.info("Writing output")
        self.output_volumedata.write_geojson_features(all_nodes)

        self.logger.info("Done")
