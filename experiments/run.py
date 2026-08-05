#!/usr/bin/env python3

from pathlib import Path
import argparse
import yaml
from click.testing import CliRunner
import time

from malstroem.scripts.cli import cli

CLI_OPTIONS = {
    "dem": "-dem",
    "landuse": "-landuse",
    "sumps": "-sumps",
    "rainfall_mm": "-mm",
    "zresolution": "-zresolution",
    "filter": "-filter",
    "initial_abstraction_method": "-initial_abstraction_method",
    "scenario": "-scenario",
    "simulation_duration_s": "-simulation_duration_s",
    "allow_initial_spillover": "-allow_initial_spillover",
    "outdir": "-outdir",
}


def expand_config_variables(config):
    """
    Expand variables inside YAML strings.

    Example:
        location: pat
        rainfall_mm: 100
        outdir: results/{location}/{rainfall_mm}mm

    becomes:
        outdir: results/pat/100mm
    """

    changed = True

    while changed:
        changed = False

        for key, value in config.items():

            if isinstance(value, str):

                try:
                    new_value = value.format(**config)

                    if new_value != value:
                        config[key] = new_value
                        changed = True

                except KeyError:
                    # Variable not available yet, try next iteration
                    pass

    return config


def load_config(filename):

    with open(filename, "r") as f:
        config = yaml.safe_load(f)

    return expand_config_variables(config)


def build_cli_arguments(config):

    args = ["complete"]

    for key, option in CLI_OPTIONS.items():

        if key in config:
            args.extend(
                [
                    option,
                    str(config[key])
                ]
            )

    return args

def prepare_output_directory(outdir):
    """
    Create output directory if it does not exist.
    Fail if it already exists and is not empty.
    """
    outdir = Path(outdir)

    if outdir.exists():
        if any(outdir.iterdir()):
            raise RuntimeError(
                f"Output directory is not empty: {outdir}"
            )
    else:
        outdir.mkdir(parents=True)

    return str(outdir)

def run_experiment(config):

    # Create output directory
    config["outdir"] = prepare_output_directory(
        config["outdir"]
    )

    runner = CliRunner()

    args = build_cli_arguments(config)

    print("\nRunning experiment:")
    print(config["name"])
    print("\nCommand:")
    print(" ".join(args))
    print()

    start = time.perf_counter()

    result = runner.invoke(cli, args)

    elapsed = time.perf_counter() - start

    if result.exit_code != 0:
        print(result.output)
        print("\nException:")
        print(result.exception)
        raise RuntimeError(
            "Experiment failed"
        )

    print(result.output)
    print(
        f"\nResults saved in {config['outdir']}"
    )
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60
    print(f"Execution time: {hours:02d}:{minutes:02d}:{seconds:05.2f}")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        "-c",
        help="Run a single experiment YAML file"
    )

    parser.add_argument(
        "--directory",
        "-d",
        help="Run all YAML experiments contained in a directory"
    )

    args = parser.parse_args()

    if (args.config is None) == (args.directory is None):
        parser.error("Specify exactly one of --config or --directory.")

    if args.config:

        config = load_config(args.config)
        run_experiment(config)

    else:

        config_files = sorted(Path(args.directory).glob("*.yaml"))

        if not config_files:
            raise RuntimeError(
                f"No YAML files found in {args.directory}"
            )

        print(f"\nFound {len(config_files)} experiments.\n")

        for config_file in config_files:

            print("=" * 80)
            print(config_file.name)
            print("=" * 80)

            config = load_config(config_file)

            run_experiment(config)


if __name__ == "__main__":
    main()