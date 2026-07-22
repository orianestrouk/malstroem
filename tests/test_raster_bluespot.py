import numpy as np
import pytest

from malstroem import bluespots, io, hydrology
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

# Test added by Oriane Strouk (2026)
# ------------------------------
# Test hydrologic coefficient derivation from landuse rasters for rainfall-runoff transformation.

@pytest.mark.parametrize(
    "initial_abstraction_method ,scenario, expected_max, expected_min, output_dirname",
    [
        ("runoff_coefficient", "lower_bound", 1.0, 0.0, "output_hydrocoeff/runoff_lower"),
        ("runoff_coefficient", "default_value", 1.0, 0.0, "output_hydrocoeff/runoff_default"),
        ("runoff_coefficient", "upper_bound", 1.0, 0.0, "output_hydrocoeff/runoff_upper"),

        ("curve_number", "lower_bound", 95, 0, "output_hydrocoeff/cn_lower"),
        ("curve_number", "default_value", 98, 0, "output_hydrocoeff/cn_default"),
        ("curve_number", "upper_bound", 98, 0, "output_hydrocoeff/cn_upper"),
    ],
)

def test_hydrologic_coefficient_tool(tmpdir, initial_abstraction_method, scenario,
                                     expected_max, expected_min, output_dirname):

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

    # --- 2. Synthetic landuse raster using landuse_code ---

    dem_array = dem_reader.read()
    rows, cols = dem_array.shape


    # codes from lookup table
    landuse_codes = [
        1,   # Impervious
        2,   # Commercial centre-ville
        3,   # Commercial banlieue
        4,   # Maisons de banlieue
        5,   # Maisons detachees
        6,   # Unites jumelees
        7,   # Maisons de ville
        8,   # Blocs appartements
        9,   # Industrielle legere
        10,  # Industrielle lourde
        11,  # Parc et cimetiere
        12,  # Terrain de jeux
        13,  # Champs
    ]

    n_patches = len(landuse_codes)
    patch_rows = int(np.ceil(np.sqrt(n_patches)))
    patch_cols = int(np.ceil(n_patches / patch_rows))

    patch_height = rows // patch_rows
    patch_width = cols // patch_cols

    landuse_array = np.ones((rows, cols), dtype=np.uint8)
    for idx, code in enumerate(landuse_codes):
        pr = idx // patch_cols
        pc = idx % patch_cols
        row_start = pr * patch_height
        row_end = (pr + 1) * patch_height if pr < patch_rows - 1 else rows
        col_start = pc * patch_width
        col_end = (pc + 1) * patch_width if pc < patch_cols - 1 else cols
        landuse_array[row_start:row_end, col_start:col_end] = code

    landuse_reader = NumpyRasterReader(landuse_array, dem_reader.transform)

    # --- Save synthetic landuse for inspection ---
    output_dir = os.path.join(os.path.dirname(__file__), output_dirname)
    os.makedirs(output_dir, exist_ok=True)

    landuse_out = io.RasterWriter(os.path.join(output_dir, 'landuse_synthetic.tif'), dem_reader.transform, dem_reader.crs, 0)
    landuse_out.write(landuse_array)

    # --- HydrologicCoefficientTool ---
    watershed_labels_reader = io.RasterReader(watershed_writer.filepath)

    watershed_coeff_path = os.path.join(output_dir, f'watershed_{initial_abstraction_method}_{scenario}.tif')

    watershed_coeff_writer = io.RasterWriter(watershed_coeff_path, dem_reader.transform, dem_reader.crs, 0)

    tool = hydrology.HydrologicCoefficientTool(
        initial_abstraction_method=initial_abstraction_method,
        scenario=scenario,
        input_landuse=landuse_reader,
        input_watershed_labels=watershed_labels_reader,
        output_watershed_raster=watershed_coeff_writer
    )
    tool.process()

    # --- Assertions ---

    assert os.path.isfile(watershed_coeff_path)

    watershed_coeff = io.RasterReader(watershed_coeff_path).read()

    assert watershed_coeff.shape == dem_array.shape

    assert np.nanmax(watershed_coeff) <= expected_max
    assert np.nanmin(watershed_coeff) >= expected_min

    assert (watershed_coeff > 0).any()

    print(f"\nOutput saved: {watershed_coeff_path}")
    print(f"  landuse_synthetic.tif           — damier de landuse_code  {landuse_codes}")
    print(f"  watershed_{initial_abstraction_method}_{scenario}.tif  — coefficient moyen par watershed")

# End of test added by Oriane Strouk (2026)
# -----------------------------------------

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