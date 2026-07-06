from click.testing import CliRunner

from malstroem.bluespots import COTQ_landuse_manning_map
from malstroem.scripts.cli import cli
from malstroem import io
from data.fixtures import dtmfile, landusefile, sumpsfile, filledfile, flowdirnoflatsfile, depthsfile, labeledfile, wshedsfile, pourpointsfile, nodesfile, initvolsfile, finalvolsfile, hypsfile, precipraster_byte_file, precipraster_float_file
import numpy as np
import os
import pytest


def test_complete(tmpdir):
    output_dir = os.path.join(os.path.dirname(__file__), 'output_complete')
    os.makedirs(output_dir, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(cli, ['complete',
                                 '-mm', 100,
                                 '-zresolution', 0.1,
                                 '-filter', 'area > 20.5 and maxdepth > 0.5 or volume > 2.5',
                                 '-dem', dtmfile,
                                 '-outdir', str(tmpdir)])
    assert result.exit_code == 0, result.output

     # --- Copy outputs for comparison ---
    import shutil
    for filename in [
        'filled.tif', 'flowdir.tif', 'bs_depths.tif',
        'bluespots.tif', 'watersheds.tif',
        'finaldepths.tif', 'finalbluespots.tif',
        'malstroem.gpkg'
    ]:
        src = str(tmpdir.join(filename))
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(output_dir, filename))

    assert os.path.isfile(str(tmpdir.join('filled.tif')))

    r = io.RasterReader(str(tmpdir.join('bluespots.tif')))
    data = r.read()

    assert np.max(data) == 486, result.output

    v = io.VectorReader(str(tmpdir.join('malstroem.gpkg')), 'finalstate')
    data = v.read_geojson_features()
    assert len(data) == 544, result.output

    v = io.VectorReader(str(tmpdir.join('malstroem.gpkg')), 'finalbluespots')
    data = v.read_geojson_features()
    assert len(data) == 500, result.output

    print(f"\nOutputs saved to: {output_dir}")


def test_complete_with_landuse(tmpdir):
    output_dir = os.path.join(os.path.dirname(__file__), 'output_complete_with_inital_abstraction_watershed')
    os.makedirs(output_dir, exist_ok=True)

    runner = CliRunner()
    result = runner.invoke(cli, ['complete',
                                 '-mm', 100,
                                 '-zresolution', 0.1,
                                 '-filter', 'area > 20.5 and maxdepth > 0.5 or volume > 2.5',
                                 '-dem', dtmfile,
                                 '-landuse', landusefile,
                                 '-infiltration_method', 'initial_abstraction_watershed',
                                 '-outdir', str(tmpdir)])

    assert result.exit_code == 0, result.output

    # --- Copy outputs for comparison ---
    import shutil
    for filename in [
        'filled.tif', 'flowdir.tif', 'bs_depths.tif',
        'bluespots.tif', 'watersheds.tif',
        'bluespot_manning.tif', 'watershed_manning.tif',
        'finaldepths.tif', 'finalbluespots.tif',
        'malstroem.gpkg'
    ]:
        src = str(tmpdir.join(filename))
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(output_dir, filename))

    # --- Manning rasters should be produced ---
    assert os.path.isfile(str(tmpdir.join('bluespot_manning.tif'))), "Bluespot Manning raster not written"
    assert os.path.isfile(str(tmpdir.join('watershed_manning.tif'))), "Watershed Manning raster not written"

    bluespot_manning = io.RasterReader(str(tmpdir.join('bluespot_manning.tif'))).read()
    watershed_manning = io.RasterReader(str(tmpdir.join('watershed_manning.tif'))).read()

    max_expected = max(COTQ_landuse_manning_map().values())
    assert bluespot_manning.min() >= 0.0
    assert bluespot_manning.max() <= max_expected + 1e-10
    assert (bluespot_manning > 0).any(), "All bluespot Manning values are zero"
    assert (watershed_manning > 0).any(), "All watershed Manning values are zero"

    # --- Bluespots count unchanged (Manning affects volumes, not bluespot detection) ---
    r = io.RasterReader(str(tmpdir.join('bluespots.tif')))
    data = r.read()
    assert np.max(data) == 486, result.output

    # --- Node count unchanged ---
    v = io.VectorReader(str(tmpdir.join('malstroem.gpkg')), 'finalstate')
    data = v.read_geojson_features()
    assert len(data) == 544, result.output

    # --- Manning retention reduces water volume so fewer bluespots should be filled ---
    v = io.VectorReader(str(tmpdir.join('malstroem.gpkg')), 'finalbluespots')
    data = v.read_geojson_features()
    assert len(data) <= 500, \
        f"Expected fewer finalbluespots with Manning retention, got {len(data)}"

    print(f"\nOutputs saved to: {output_dir}")

def test_complete_with_sumps_only(tmpdir):
    output_dir = os.path.join(os.path.dirname(__file__), 'output_complete_sumps_only')
    os.makedirs(output_dir, exist_ok=True)
    
    runner = CliRunner()
    result = runner.invoke(cli, ['complete',
                                 '-mm', 100,
                                 '-zresolution', 0.1,
                                 '-filter', 'area > 20.5 and maxdepth > 0.5 or volume > 2.5',
                                 '-dem', dtmfile,
                                 '-sumps', sumpsfile,
                                 '-simulation_duration_s', 360000,
                                 '-allow_initial_spillover', 'YES',
                                 '-outdir', str(tmpdir)])
    
    assert result.exit_code == 0, result.output

    # --- Copy outputs for comparison ---
    import shutil
    for filename in [
        'filled.tif', 'flowdir.tif', 'bs_depths.tif',
        'bluespots.tif', 'watersheds.tif',
        'finaldepths.tif', 'finalbluespots.tif',
        'malstroem.gpkg'
    ]:
        src = str(tmpdir.join(filename))
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(output_dir, filename))

    # --- Base file existence assertions ---
    assert os.path.isfile(str(tmpdir.join('filled.tif')))

    # --- Raster output verifications ---
    r = io.RasterReader(str(tmpdir.join('bluespots.tif')))
    data = r.read()
    assert np.max(data) > 0, result.output 

    # --- Vector output verifications (finalstate and finalbluespots) ---
    v_state = io.VectorReader(str(tmpdir.join('malstroem.gpkg')), 'finalstate')
    data_state = v_state.read_geojson_features()
    assert len(data_state) > 0, result.output

    # =========================================================================
    # === MASS CONSERVATION VERIFICATION FOR TARGET BLUESPOTS ===
    # =========================================================================

    # --- Set of target bluespot IDs to verify mass conservation. Some with and without sumps in them. ---
    target_bs_ids = {32, 38, 92, 166, 173, 180, 236, 294, 298, 308, 363, 383, 486, 13, 56, 100, 142, 238, 386}
    
    nodes_map = {f['properties']['nodeid']: f['properties'] for f in data_state}

    print("\n--- Verification of Mass Conservation ---")
    for bs_id in target_bs_ids:
        # --- Verify that the expected node exists within the final state outputs ---
        assert bs_id in nodes_map, f"Missing Node {bs_id} in final state outputs."
        
        props = nodes_map[bs_id]
        
        # --- Extract and parse inflow volumes ---
        inputv = float(props.get('inputv', 0.0))
        upstreamv = float(props.get('upstreamv', 0.0))
        total_in = inputv + upstreamv

        # --- Extract and parse storage and outflow volumes ---
        v = float(props.get('v', 0.0))
        spillv = float(props.get('spillv', 0.0))
        
        # --- Parse and sum up the pipe-separated drainv history string (e.g., "150.23|45.10" or "0.0") ---
        drainv_str = props.get('drainv', '0.0')
        if drainv_str and drainv_str != '0.0':
            sum_drainv = sum(float(x) for x in drainv_str.split('|'))
        else:
            sum_drainv = 0.0
            
        total_out = spillv + sum_drainv + v

        # --- Log detailed metrics to terminal for debugging purposes (visible via pytest -s) ---
        print(f"BS {bs_id:3d} | In: {total_in:12.4f} (Input: {inputv:11.2f}, Upstream: {upstreamv:11.2f}) "
              f"-> Out: {total_out:12.4f} (Drain: {sum_drainv:11.2f}, Stored v: {v:11.2f}, Spill: {spillv:11.2f})")

        # --- Mathematical mass conservation assertion (Using absolute tolerance of 0.05 m3 to handle string rounding artifacts) ---
        assert np.isclose(total_in, total_out, atol=0.05), (
            f"Mass Balance Leak detected at Bluespot {bs_id}!\n"
            f"  Total Inflow  (inputv + upstreamv) = {total_in}\n"
            f"  Total Outflow (spillv + drainv + v) = {total_out}\n"
            f"  Discrepancy Volume                  = {abs(total_in - total_out)}"
        )
    print("-----------------------------------------\n"
          "All targeted bluespots match perfect mass conservation!")
    # =========================================================================

    v_bs = io.VectorReader(str(tmpdir.join('malstroem.gpkg')), 'finalbluespots')
    data_bs = v_bs.read_geojson_features()
    assert len(data_bs) > 0, result.output

    print(f"\nOutputs saved to: {output_dir}")

def test_complete_with_sumps_only_no_initial_spillover(tmpdir):
    output_dir = os.path.join(os.path.dirname(__file__), 'output_complete_sumps_only')
    os.makedirs(output_dir, exist_ok=True)
    
    runner = CliRunner()
    result = runner.invoke(cli, ['complete',
                                 '-mm', 100,
                                 '-zresolution', 0.1,
                                 '-filter', 'area > 20.5 and maxdepth > 0.5 or volume > 2.5',
                                 '-dem', dtmfile,
                                 '-sumps', sumpsfile,
                                 '-simulation_duration_s', 360000,
                                 '-allow_initial_spillover', 'NO',
                                 '-outdir', str(tmpdir)])
    
    assert result.exit_code == 0, result.output

    # --- Copy outputs for comparison ---
    import shutil
    for filename in [
        'filled.tif', 'flowdir.tif', 'bs_depths.tif',
        'bluespots.tif', 'watersheds.tif',
        'finaldepths.tif', 'finalbluespots.tif',
        'malstroem.gpkg'
    ]:
        src = str(tmpdir.join(filename))
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(output_dir, filename))

    # --- Base file existence assertions ---
    assert os.path.isfile(str(tmpdir.join('filled.tif')))

    # --- Raster output verifications ---
    r = io.RasterReader(str(tmpdir.join('bluespots.tif')))
    data = r.read()
    assert np.max(data) > 0, result.output 

    # --- Vector output verifications (finalstate and finalbluespots) ---
    v_state = io.VectorReader(str(tmpdir.join('malstroem.gpkg')), 'finalstate')
    data_state = v_state.read_geojson_features()
    assert len(data_state) > 0, result.output

    # =========================================================================
    # === MASS CONSERVATION VERIFICATION FOR TARGET BLUESPOTS ===
    # =========================================================================

    # --- Set of target bluespot IDs to verify mass conservation. Some with and without sumps in them. ---
    target_bs_ids = {32, 38, 92, 166, 173, 180, 236, 294, 298, 308, 363, 383, 486, 13, 56, 100, 142, 238, 386}
    
    nodes_map = {f['properties']['nodeid']: f['properties'] for f in data_state}

    print("\n--- Verification of Mass Conservation ---")
    for bs_id in target_bs_ids:
        # --- Verify that the expected node exists within the final state outputs ---
        assert bs_id in nodes_map, f"Missing Node {bs_id} in final state outputs."
        
        props = nodes_map[bs_id]
        
        # --- Extract and parse inflow volumes ---
        inputv = float(props.get('inputv', 0.0))
        upstreamv = float(props.get('upstreamv', 0.0))
        total_in = inputv + upstreamv

        # --- Extract and parse storage and outflow volumes ---
        v = float(props.get('v', 0.0))
        spillv = float(props.get('spillv', 0.0))
        
        # --- Parse and sum up the pipe-separated drainv history string (e.g., "150.23|45.10" or "0.0") ---
        drainv_str = props.get('drainv', '0.0')
        if drainv_str and drainv_str != '0.0':
            sum_drainv = sum(float(x) for x in drainv_str.split('|'))
        else:
            sum_drainv = 0.0
            
        total_out = spillv + sum_drainv + v

        # --- Log detailed metrics to terminal for debugging purposes (visible via pytest -s) ---
        print(f"BS {bs_id:3d} | In: {total_in:12.4f} (Input: {inputv:11.2f}, Upstream: {upstreamv:11.2f}) "
              f"-> Out: {total_out:12.4f} (Drain: {sum_drainv:11.2f}, Stored v: {v:11.2f}, Spill: {spillv:11.2f})")

        # --- Mathematical mass conservation assertion (Using absolute tolerance of 0.05 m3 to handle string rounding artifacts) ---
        assert np.isclose(total_in, total_out, atol=0.05), (
            f"Mass Balance Leak detected at Bluespot {bs_id}!\n"
            f"  Total Inflow  (inputv + upstreamv) = {total_in}\n"
            f"  Total Outflow (spillv + drainv + v) = {total_out}\n"
            f"  Discrepancy Volume                  = {abs(total_in - total_out)}"
        )
    print("-----------------------------------------\n"
          "All targeted bluespots match perfect mass conservation!")
    # =========================================================================

    v_bs = io.VectorReader(str(tmpdir.join('malstroem.gpkg')), 'finalbluespots')
    data_bs = v_bs.read_geojson_features()
    assert len(data_bs) > 0, result.output

    print(f"\nOutputs saved to: {output_dir}")

def test_complete_nofilter(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli, ['complete',
                                 '-mm', 100,
                                 '-zresolution', 0.1,
                                 '-dem', dtmfile,
                                 '-outdir', str(tmpdir)])
    assert result.exit_code == 0, result.output
    assert os.path.isfile(str(tmpdir.join('filled.tif')))
    r = io.RasterReader(str(tmpdir.join('bluespots.tif')))
    data = r.read()

    assert np.max(data) == 523

    v = io.VectorReader(str(tmpdir.join('malstroem.gpkg')), 'finalstate')
    data = v.read_geojson_features()
    assert len(data) == 587, result.output

    v = io.VectorReader(str(tmpdir.join('malstroem.gpkg')), 'finalbluespots')
    data = v.read_geojson_features()
    assert len(data) == 537, result.output


def test_filled(tmpdir):
    ff = str(tmpdir.join('filled.tif'))
    runner = CliRunner()
    result = runner.invoke(cli, ['filled',
                                 '-dem', dtmfile,
                                 '-out', ff])
    assert result.exit_code == 0
    assert result.output == ''
    assert os.path.isfile(ff)


def test_depths(tmpdir):
    df = str(tmpdir.join('depths.tif'))
    runner = CliRunner()
    result = runner.invoke(cli, ['depths',
                                 '-dem', dtmfile,
                                 '-filled', filledfile,
                                 '-out', df])
    assert result.exit_code == 0
    assert result.output == ''
    assert os.path.isfile(df)


def test_flowdir(tmpdir):
    ff = str(tmpdir.join('flowdir.tif'))
    runner = CliRunner()
    result = runner.invoke(cli, ['flowdir',
                                 '-dem', dtmfile,
                                 '-out', ff])
    assert result.output == ''
    assert result.exit_code == 0
    assert os.path.isfile(ff)


def test_accum(tmpdir):
    f = str(tmpdir.join('accum.tif'))
    runner = CliRunner()
    result = runner.invoke(cli, ['accum',
                                 '-flowdir', flowdirnoflatsfile,
                                 '-out', f])
    assert result.output == ''
    assert result.exit_code == 0
    assert os.path.isfile(f)


def test_bspot(tmpdir):
    f = str(tmpdir.join('bspots.tif'))
    runner = CliRunner()
    result = runner.invoke(cli, ['bspots',
                                 '-depths', depthsfile,
                                 '-out', f])
    assert result.output == ''
    assert result.exit_code == 0
    assert os.path.isfile(f)




def test_filtered_bspot(tmpdir):
    f = str(tmpdir.join('bspots.tif'))
    runner = CliRunner()
    result = runner.invoke(cli, ['bspots',
                                 '-filter', 'area > 20.5 and maxdepth > 0.5 or volume > 2.5',
                                 '-depths', depthsfile,
                                 '-out', f])
    assert result.exit_code == 0
    assert result.output == ''
    assert os.path.isfile(f)


def test_wsheds(tmpdir):
    f = str(tmpdir.join('wsheds.tif'))
    runner = CliRunner()
    result = runner.invoke(cli, ['wsheds',
                                 '-bluespots', labeledfile,
                                 '-flowdir', flowdirnoflatsfile,
                                 '-out', f])
    assert result.output == ''
    assert result.exit_code == 0
    assert os.path.isfile(f)


def test_pourpoints(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli, ['pourpts',
                                 '-bluespots', labeledfile,
                                 '-depths', depthsfile,
                                 '-watersheds', wshedsfile,
                                 '-dem', dtmfile,
                                 '-out', str(tmpdir)])
    assert result.output == ''
    assert result.exit_code == 0
    assert os.path.isfile(str(tmpdir.join('pourpoints.shp')))


def test_network(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli, ['network',
                                 '-bluespots', labeledfile,
                                 '-flowdir', flowdirnoflatsfile,
                                 '-pourpoints', pourpointsfile,
                                 '-out', str(tmpdir),
                                ])

    assert result.exit_code == 0, 'Output: {}'.format(result.output)
    assert os.path.isfile(str(tmpdir.join('nodes.shp')))
    assert os.path.isfile(str(tmpdir.join('streams.shp')))


def test_initvolumes_mm(tmpdir):
    runner = CliRunner()
    result = runner.invoke(cli, ['initvolumes',
                                 '-nodes', nodesfile,
                                 '-mm', 20,
                                 '-out', str(tmpdir)],)
    assert result.exit_code == 0, 'Output: {}'.format(result.output)
    assert os.path.isfile(str(tmpdir.join('initvolumes.shp')))

@pytest.mark.parametrize("precipfile", [precipraster_byte_file, precipraster_float_file])
def test_initvolumes_pr_filetype(tmpdir, precipfile):
    runner = CliRunner()
    result = runner.invoke(cli, ['initvolumes',
                                 '-nodes', nodesfile,
                                 '-pr', precipfile,
                                 '-pr_unit', "mm",
                                 '-bluespots', labeledfile,
                                 '-out', str(tmpdir)],)
    assert result.exit_code == 0, 'Output: {}'.format(result.output)
    assert os.path.isfile(str(tmpdir.join('initvolumes.shp')))    

@pytest.mark.parametrize("pr_unit", ["mm", "l", "m3"])
def test_initvolumes_pr_unit(tmpdir, pr_unit):
    runner = CliRunner()
    result = runner.invoke(cli, ['initvolumes',
                                 '-nodes', nodesfile,
                                 '-pr', precipraster_float_file,
                                 '-pr_unit', pr_unit,
                                 '-bluespots', labeledfile,
                                 '-out', str(tmpdir)],)
    assert result.exit_code == 0, 'Output: {}'.format(result.output)
    assert os.path.isfile(str(tmpdir.join('initvolumes.shp')))    


def test_finalvols(tmpdir):
    runner = CliRunner()
    out_file = str(tmpdir.join("finalvolumes.shp"))
    result = runner.invoke(cli, ['finalvolumes',
                                 '-inputvolumes', initvolsfile,
                                 '-out', str(tmpdir)])
    assert result.exit_code == 0, 'Output: {}'.format(result.output)
    assert os.path.isfile(out_file)

def test_hyps(tmpdir):
    runner = CliRunner()
    out_file = str(tmpdir.join('hyps.csv'))
    result = runner.invoke(cli, ['hyps',
                                 '-bluespots', labeledfile,
                                 '-dem', dtmfile,
                                 '-pourpoints', pourpointsfile,
                                 '-zresolution', 0.1,
                                 '-out', out_file])
    assert result.exit_code == 0, 'Output: {}'.format(result.output)
    assert os.path.isfile(out_file)
    from csv import DictReader
    from malstroem import hyps
    with open(out_file) as f:
        reader = DictReader(f)
        for row in reader:
            if int(row["bspot_id"]) != 0:
                for float_key in ["bspot_dmax", "hist_num_bins", "hist_lower_bound", "hist_upper_bound", "hist_resolution", "zmin", "zmax", "cell_area"]:
                    float(row[float_key])
                assert int(row["hist_num_bins"]) > 0

                h = hyps.hypsometrystats_from_flatdict(row)
                assert len(h.zhistogram.counts) > 0 
                hyps.assert_hypsometrystats_valid(h)    

def test_finallevels(tmpdir):
    runner = CliRunner()
    out_file = str(tmpdir.join('hyps.geojson'))
    result = runner.invoke(cli, ['finallevels',
                                 '-finalvols', finalvolsfile,
                                 '-hyps', hypsfile,
                                 '-out', out_file])
    assert result.exit_code == 0, 'Output: {}'.format(result.output)
    assert os.path.isfile(out_file)
    import json
    with open(out_file) as f:
        parsed = json.load(f)
        for row in parsed["features"]:
            for float_key in ["approx_z"]:
                float(row["properties"][float_key])


def test_chained(tmpdir):
    filled = str(tmpdir.join('filled.tif'))
    depths = str(tmpdir.join('depths.tif'))
    flowdir = str(tmpdir.join('flowdir.tif'))
    accum = str(tmpdir.join('accum.tif'))
    bspots = str(tmpdir.join('bspots.tif'))
    pourpoints = str(tmpdir.join('pourpoints.shp'))
    nodes = str(tmpdir.join('nodes.shp'))
    streams = str(tmpdir.join('streams.shp'))
    initvolumes = str(tmpdir.join('initvolumes.shp'))
    final = str(tmpdir.join('finalvolumes.shp'))

    runner = CliRunner()

    # Filled
    result = runner.invoke(cli, ['filled',
                                 '-dem', dtmfile,
                                 '-out', filled])
    assert result.exit_code == 0
    assert result.output == ''
    assert os.path.isfile(filled)

    # Depths
    result = runner.invoke(cli, ['depths',
                                 '-dem', dtmfile,
                                 '-filled', filled,
                                 '-out', depths])
    assert result.exit_code == 0
    assert result.output == ''
    assert os.path.isfile(depths)

    # Flowdir
    result = runner.invoke(cli, ['flowdir',
                                 '-dem', dtmfile,
                                 '-out', flowdir])
    assert result.output == ''
    assert result.exit_code == 0
    assert os.path.isfile(flowdir)

    # Accum
    result = runner.invoke(cli, ['accum',
                                 '-flowdir', flowdir,
                                 '-out', accum])
    assert result.output == ''
    assert result.exit_code == 0
    assert os.path.isfile(accum)

    # Bluespots
    result = runner.invoke(cli, ['bspots',
                                 '-filter', 'area > 20.5 and maxdepth > 0.5 or volume > 2.5',
                                 '-depths', depths,
                                 '-out', bspots])
    assert result.exit_code == 0
    assert result.output == ''
    assert os.path.isfile(bspots)

    # Watersheds
    wsheds = str(tmpdir.join('wsheds.tif'))
    result = runner.invoke(cli, ['wsheds',
                                 '-bluespots', bspots,
                                 '-flowdir', flowdir,
                                 '-out', wsheds])
    assert result.output == ''
    assert result.exit_code == 0
    assert os.path.isfile(wsheds)

    # Pourpoints
    result = runner.invoke(cli, ['pourpts',
                                 '-bluespots', bspots,
                                 '-depths', depths,
                                 '-watersheds', wsheds,
                                 '-dem', dtmfile,
                                 '-out', str(tmpdir)])
    assert result.output == ''
    assert result.exit_code == 0
    assert os.path.isfile(pourpoints)

    # Nodes
    result = runner.invoke(cli, ['network',
                                 '-bluespots', bspots,
                                 '-flowdir', flowdir,
                                 '-pourpoints', str(tmpdir),
                                 '-out', str(tmpdir),
                                 ])

    assert result.exit_code == 0, 'Output: {}'.format(result.output)
    assert os.path.isfile(nodes)
    assert os.path.isfile(streams)

    # Initial volumes
    result = runner.invoke(cli, ['initvolumes',
                                 '-nodes', str(tmpdir),
                                 '-mm', 20,
                                 '-out', str(tmpdir)])
    assert result.exit_code == 0, 'Output: {}'.format(result.output)
    assert os.path.isfile(initvolumes)

    # Network
    result = runner.invoke(cli, ['finalvolumes',
                                 '-inputvolumes', str(tmpdir),
                                 '-out', str(tmpdir)])
    assert result.exit_code == 0, 'Output: {}'.format(result.output)
    assert os.path.isfile(final)

    reader = io.VectorReader(str(tmpdir), 'finalvolumes')
    data = reader.read_geojson_features()
    assert len(data) == 544
