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

from osgeo import gdal

def check_filter(filter):
    # TODO: Stupid check mechanism. Could be done better
    allowed_words = ['area', 'maxdepth', 'volume', 'and', 'or']
    allowedchars = '<>=!0123456789.()'
    for w in allowed_words:
        filter = filter.replace(w, '')
    for c in allowedchars:
        filter = filter.replace(c, '')
    filter = filter.strip()
    if filter:
        raise Exception('Unsupported filter statement. Illegal parts: {}'.format(filter))


def parse_filter(filter):
    if not filter:
        filter_function = lambda stats: True
    else:
        check_filter(filter)
        # filter_function = lambda stats: stats['max'] > 0.05  # and stats['area'] > 5 and stats['volume'] > 1
        filter = filter.replace('area', 'stats["area"]')
        filter = filter.replace('maxdepth', 'stats["max"]')
        filter = filter.replace('volume', 'stats["volume"]')
        filter = 'lambda stats: {}'.format(filter)
        filter_function = eval(filter)
    return filter_function


def rasterize_sumps(input_sumps, reference_transform, reference_shape, crs,
                    output_raster):
    """Rasterize a sump (puisard) point vector layer into a capacity array.

    Parameters
    ----------
    input_sumps : VectorReader
        Point vector layer with sump locations.
    reference_transform : sequence of six numbers
        GDAL style affine transform aligned with the reference raster.
    reference_shape : tuple
        (rows, cols) aligned with the reference raster.
    crs : str
        WKT representation of the CRS.
    output_raster : RasterWriter
        Writes the sump capacity array to disk.

    Returns
    -------
    np.ndarray
        Array of shape `reference_shape` with sump capacity values, 0 elsewhere.
    """
    rows, cols = reference_shape

    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_Float64)
    target_ds.SetGeoTransform(reference_transform)
    if crs:
        target_ds.SetProjection(crs)

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.Fill(0)

    gdal.RasterizeLayer(
        target_ds, [1], input_sumps.ogr_layer,
        options=["ATTRIBUTE=capacity"]
    )

    result = band.ReadAsArray()
    target_ds = None

    output_raster.write(result)

    return result