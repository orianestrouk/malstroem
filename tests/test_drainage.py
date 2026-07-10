# coding=utf-8
import os
import numpy as np
import pytest
from osgeo import ogr, osr

from malstroem import bluespots, io
from malstroem.drainage import bluespot_drainage_io
from malstroem.hyps import bluespot_hypsometry_io
from malstroem.rain import SimpleVolumeTool
from malstroem.network import FinalStateCalculator
from malstroem.scripts._utils import rasterize_sumps

from data.fixtures import dtmfile, flowdirnoflatsfile, filledfile
from malstroem.io import RasterReader


class NumpyRasterReader(object):
    """Minimal in-memory raster reader for arrays not backed by a file."""
    def __init__(self, array, transform, crs=None):
        self._array = array
        self.transform = transform
        self.crs = crs
        self.shape = array.shape
        self.resolution = (abs(transform[5]), abs(transform[1]))

    def read(self):
        return self._array

def test_rasterize_sumps(tmpdir):
    """Test sump rasterization in isolation."""

    dem_reader = io.RasterReader(dtmfile)
    transform = dem_reader.transform
    rows, cols = dem_reader.shape
    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = transform[5]

    output_dir = os.path.join(os.path.dirname(__file__), 'output_drainage_test')
    os.makedirs(output_dir, exist_ok=True)

    # --- Write sump vector directly to output_dir so it persists for inspection ---
    sump_file = os.path.join(output_dir, 'sumps.gpkg')
    if os.path.isfile(sump_file):
        os.remove(sump_file)  # GPKG driver fails to overwrite an existing file

    sump_driver = ogr.GetDriverByName('GPKG')
    sump_ds = sump_driver.CreateDataSource(sump_file)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dem_reader.crs)
    sump_layer = sump_ds.CreateLayer('sumps', srs, ogr.wkbPoint)
    sump_layer.CreateField(ogr.FieldDefn('capacity', ogr.OFTReal))

    sump_pixels = [
        # Diagonal from top-left to bottom-right, with increasing capacity
        (rows // 4, cols // 4, 0.01),
        (rows // 2, cols // 2, 0.05),
        (3 * rows // 4, 3 * cols // 4, 0.10),
    ]
    for row, col, capacity in sump_pixels:
        x = x_origin + (col + 0.5) * pixel_width
        y = y_origin + (row + 0.5) * pixel_height
        feat = ogr.Feature(sump_layer.GetLayerDefn())
        feat.SetField('capacity', capacity)
        feat.SetGeometry(ogr.CreateGeometryFromWkt('POINT ({} {})'.format(x, y)))
        sump_layer.CreateFeature(feat)
    sump_ds.FlushCache()
    sump_ds = None

    sump_capacity_path = os.path.join(output_dir, 'sump_capacity_isolated.tif')
    sump_capacity_writer = io.RasterWriter(sump_capacity_path, dem_reader.transform, dem_reader.crs, 0)

    sumps_reader = io.VectorReader(sump_file, 'sumps')
    result = rasterize_sumps(
        sumps_reader,
        dem_reader.transform, dem_reader.shape, dem_reader.crs,
        sump_capacity_writer
    )

    # --- Assertions ---
    assert os.path.isfile(sump_capacity_path), "Sump capacity raster not written"
    assert os.path.isfile(sump_file), "Sump vector not written"
    assert result.shape == dem_reader.shape

    expected_capacities = [c for _, _, c in sump_pixels]
    nonzero_values = result[result > 0]
    assert len(nonzero_values) == len(sump_pixels), \
        "Expected {} non-zero cells, got {}".format(len(sump_pixels), len(nonzero_values))
    for cap in expected_capacities:
        assert any(np.isclose(nonzero_values, cap)), "Capacity {} not found in raster".format(cap)

    print("\nsump_capacity_isolated.tif saved to: {}".format(output_dir))
    print("sumps.gpkg saved to: {}".format(output_dir))