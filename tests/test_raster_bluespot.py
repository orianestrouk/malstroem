import numpy as np

from malstroem import bluespots, io
from malstroem.algorithms import label
from osgeo import ogr
import os
from data.fixtures import flowdirnoflatsfile, dtmfile, filledfile, bspotdata, depthsdata

class NumpyRasterReader(object):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def read(self):
        return self.data


def test_bluespots(tmpdir):
    flowdir_reader = io.RasterReader(flowdirnoflatsfile)
    dem_reader = io.RasterReader(dtmfile)
    filled_reader = io.RasterReader(filledfile)
    depths_reader = NumpyRasterReader(filled_reader.read() - dem_reader.read(), dem_reader.transform)
    outdbfile = str(tmpdir.join('test.gpkg'))

    # At least 5cm deep, 5 cells wide and at least one cell-meter volume
    filter_function = lambda r: r['max'] > 0.05 and r['count'] > 5 and r['sum'] > 1

    pourpoint_writer = io.VectorWriter('gpkg', outdbfile, 'pourpoints', None, ogr.wkbPoint, dem_reader.crs)
    watershed_writer = io.RasterWriter(str(tmpdir.join('watersheds.tif')), dem_reader.transform, dem_reader.crs, 0)

    watershed_vector_writer = io.VectorWriter('gpkg', outdbfile, 'watersheds', None, ogr.wkbPolygon, dem_reader.crs)

    labeled_writer = io.RasterWriter(str(tmpdir.join('labeled.tif')), dem_reader.transform, dem_reader.crs, 0)

    labeled_vector_writer = io.VectorWriter('gpkg', outdbfile, 'bluespots', None, ogr.wkbPolygon, dem_reader.crs)

    bluespot_tool = bluespots.BluespotTool(
        input_depths=depths_reader,
        input_flowdir=flowdir_reader,
        input_bluespot_filter_function=filter_function,
        input_accum=None,
        input_dem=dem_reader,
        output_labeled_raster=labeled_writer,
        output_labeled_vector=labeled_vector_writer,
        output_pourpoints=pourpoint_writer,
        output_watersheds_raster=watershed_writer,
        output_watersheds_vector=watershed_vector_writer
    )
    bluespot_tool.process()

    assert os.path.isfile(outdbfile)
    assert os.path.isfile(watershed_writer.filepath)
    assert os.path.isfile(labeled_writer.filepath)

def test_manning_tool(tmpdir):
    # --- Prerequisite: run BluespotTool to get label rasters ---
    flowdir_reader = io.RasterReader(flowdirnoflatsfile)
    dem_reader = io.RasterReader(dtmfile)
    filled_reader = io.RasterReader(filledfile)
    depths_reader = NumpyRasterReader(filled_reader.read() - dem_reader.read(), dem_reader.transform)

    outdbfile = str(tmpdir.join('test.gpkg'))
    filter_function = lambda r: r['max'] > 0.05 and r['count'] > 5 and r['sum'] > 1

    pourpoint_writer = io.VectorWriter('gpkg', outdbfile, 'pourpoints', None, ogr.wkbPoint, dem_reader.crs)
    watershed_writer = io.RasterWriter(str(tmpdir.join('watersheds.tif')), dem_reader.transform, dem_reader.crs, 0)
    labeled_writer = io.RasterWriter(str(tmpdir.join('labeled.tif')), dem_reader.transform, dem_reader.crs, 0)

    bluespot_tool = bluespots.BluespotTool(
        input_depths=depths_reader,
        input_flowdir=flowdir_reader,
        input_bluespot_filter_function=filter_function,
        input_accum=None,
        input_dem=dem_reader,
        output_labeled_raster=labeled_writer,
        output_labeled_vector=None,
        output_pourpoints=pourpoint_writer,
        output_watersheds_raster=watershed_writer,
        output_watersheds_vector=None
    )
    bluespot_tool.process()

    # --- Synthetic landuse: grid of patches, one COTQ code per patch ---
    dem_array = dem_reader.read()
    rows, cols = dem_array.shape
    cotq_codes = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # COTQ codes

    n_patches = len(cotq_codes)
    patch_rows = int(np.ceil(np.sqrt(n_patches))) 
    patch_cols = int(np.ceil(n_patches / patch_rows))  

    patch_height = rows // patch_rows
    patch_width = cols // patch_cols

    landuse_array = np.zeros((rows, cols), dtype=np.uint8)
    for idx, code in enumerate(cotq_codes):
        pr = idx // patch_cols
        pc = idx % patch_cols
        row_start = pr * patch_height
        row_end = (pr + 1) * patch_height if pr < patch_rows - 1 else rows
        col_start = pc * patch_width
        col_end = (pc + 1) * patch_width if pc < patch_cols - 1 else cols
        landuse_array[row_start:row_end, col_start:col_end] = code

    landuse_reader = NumpyRasterReader(landuse_array, dem_reader.transform)

    # --- Save synthetic landuse for inspection ---
    output_dir = os.path.join(os.path.dirname(__file__), 'output_manning_test')
    os.makedirs(output_dir, exist_ok=True)

    landuse_out = io.RasterWriter(os.path.join(output_dir, 'landuse_synthetic.tif'), dem_reader.transform, dem_reader.crs, 0)
    landuse_out.write(landuse_array)

    # --- ManningTool ---
    bluespot_labels_reader = io.RasterReader(labeled_writer.filepath)
    watershed_labels_reader = io.RasterReader(watershed_writer.filepath)

    bluespot_manning_path = os.path.join(output_dir, 'bluespot_manning.tif')
    watershed_manning_path = os.path.join(output_dir, 'watershed_manning.tif')

    bluespot_manning_writer = io.RasterWriter(bluespot_manning_path, dem_reader.transform, dem_reader.crs, 0)
    watershed_manning_writer = io.RasterWriter(watershed_manning_path, dem_reader.transform, dem_reader.crs, 0)

    manning_tool = bluespots.ManningTool(
        input_landuse=landuse_reader,
        input_bluespot_labels=bluespot_labels_reader,
        input_watershed_labels=watershed_labels_reader,
        manning_map=bluespots.COTQ_landuse_manning_map(),
        default_value=0.0,
        output_bluespot_manning_raster=bluespot_manning_writer,
        output_watershed_manning_raster=watershed_manning_writer
    )
    manning_tool.process()

    # --- Assertions ---
    assert os.path.isfile(bluespot_manning_path)
    assert os.path.isfile(watershed_manning_path)

    bluespot_manning = io.RasterReader(bluespot_manning_path).read()
    watershed_manning = io.RasterReader(watershed_manning_path).read()

    assert bluespot_manning.shape == dem_array.shape
    assert watershed_manning.shape == dem_array.shape
    assert bluespot_manning.max() <= 0.4 + 1e-10
    assert bluespot_manning.min() >= 0.0
    assert (bluespot_manning > 0).any()
    assert (watershed_manning > 0).any()

    print(f"\nOutput rasters saved to: {output_dir}")
    print(f"  landuse_synthetic.tif  — damier de codes COTQ {cotq_codes}")
    print(f"  bluespot_manning.tif   — Manning moyen par bluespot")
    print(f"  watershed_manning.tif  — Manning moyen par watershed")


def test_filter(bspotdata, depthsdata):
    raw_bluespot_stats = label.label_stats(depthsdata, bspotdata)
    filter_function = lambda r: r['max'] > 2 and r['count'] > 5 and r['sum'] > 1
    keepers = bluespots.filterbluespots(filter_function, 1.0, raw_bluespot_stats)
    assert len(keepers) == len(raw_bluespot_stats)
    assert sum(keepers) == 19


def test_nofilter(bspotdata, depthsdata):
    raw_bluespot_stats = label.label_stats(depthsdata, bspotdata)
    filter_function = lambda r: True
    keepers = bluespots.filterbluespots(filter_function, 1.0, raw_bluespot_stats)
    assert len(keepers) == len(raw_bluespot_stats)
    assert sum(keepers) == len(raw_bluespot_stats)