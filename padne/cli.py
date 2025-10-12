import argparse
import warnings
import unittest.mock
import logging
import pickle
import sys
import traceback
import functools
from contextlib import contextmanager
from pathlib import Path

import padne.kicad
import padne.solver
import padne.ui
import padne.mesh
import padne.paraview
from padne import __version__


def setup_logging(debug_mode: bool):
    """Configures basic logging for the application."""
    level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", # Added %(name)s
        handlers=[
            logging.StreamHandler()
        ]
    )


@contextmanager
def collect_warnings():
    """Context manager to collect warnings."""
    # Note: we cannot use warnings.catch_warnings here because it
    # does not print the warnings as they occur.
    warns = []
    orig_showwarning = warnings.showwarning

    def showwarning_wrapper(message, category, filename, lineno, file=None, line=None):
        msg = warnings.WarningMessage(message, category, filename, lineno, file, line)
        warns.append(msg)
        orig_showwarning(message, category, filename, lineno, file=file, line=line)

    with unittest.mock.patch("warnings.showwarning", new=showwarning_wrapper):
        yield warns


def add_mesher_args(parser):
    """Add mesher configuration arguments to a parser."""
    default_config = padne.mesh.Mesher.Config()
    parser.add_argument(
        "--mesh-angle",
        type=float,
        default=default_config.minimum_angle,
        help="Minimum angle constraint for mesh triangles (degrees)"
    )
    parser.add_argument(
        "--mesh-size",
        type=float,
        default=default_config.maximum_size,
        help="Maximum size constraint for mesh triangles"
    )
    parser.add_argument(
        "--variable-density-min-distance",
        type=float,
        default=default_config.variable_density_min_distance,
        help="Minimum distance for variable density transition"
    )
    parser.add_argument(
        "--variable-density-max-distance",
        type=float,
        default=default_config.variable_density_max_distance,
        help="Maximum distance for variable density transition"
    )
    parser.add_argument(
        "--variable-size-maximum-factor",
        type=float,
        default=default_config.variable_size_maximum_factor,
        help="Maximum size scaling factor (1.0 disables variable density)"
    )
    parser.add_argument(
        "--distance-map-quantization",
        type=float,
        default=default_config.distance_map_quantization,
        help="Quantization step for distance map"
    )


def mesher_config_from_args(args):
    """Construct a Mesher.Config from parsed arguments."""
    return padne.mesh.Mesher.Config(
        minimum_angle=args.mesh_angle,
        maximum_size=args.mesh_size,
        variable_density_min_distance=args.variable_density_min_distance,
        variable_density_max_distance=args.variable_density_max_distance,
        variable_size_maximum_factor=args.variable_size_maximum_factor,
        distance_map_quantization=args.distance_map_quantization
    )


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging output."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"padne {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_gui = subparsers.add_parser(
        "gui",
        help="Run the GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_gui.add_argument(
        "kicad_pro_file",
        type=Path,
        help="Path to the input file",
    )
    add_mesher_args(parser_gui)

    parser_show = subparsers.add_parser(
        "show",
        help="Load and display a pre-computed solution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_show.add_argument(
        "solution_file",
        type=Path,
        help="Path to the pickled solution file",
    )

    parser_solve = subparsers.add_parser(
        "solve",
        help="Solve the problem and save the solution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_solve.add_argument(
        "kicad_pro_file",
        type=Path,
        help="Path to the KiCad project file (.kicad_pro)",
    )

    parser_solve.add_argument(
        "output_file",
        type=Path,
        help="Path to save the pickled solution file",
    )
    add_mesher_args(parser_solve)

    parser_paraview = subparsers.add_parser(
        "paraview",
        help="Export solution to ParaView VTK format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser_paraview.add_argument(
        "solution_file",
        type=Path,
        help="Path to the pickled solution file",
    )
    parser_paraview.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the VTU files (one per layer)",
    )

    return parser.parse_args()


def handle_errors(func):
    """Decorator for handling errors with enhanced display."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Print the normal traceback
            traceback.print_exc()

            # Print the exception message in bold yellow
            print(f"\033[1;33m{str(e)}\033[0m")

            # Exit with error code
            sys.exit(1)
    return wrapper


@handle_errors
def do_gui(args):
    log = logging.getLogger(__name__)
    log.info(f"Loading KiCad project for GUI: {args.kicad_pro_file}")
    prob = padne.kicad.load_kicad_project(args.kicad_pro_file)
    log.info("Solving problem for GUI...")
    mesher_config = mesher_config_from_args(args)

    # Capture warnings emitted during solving
    with collect_warnings() as warns:
        solution = padne.solver.solve(prob, mesher_config=mesher_config)

    captured_warnings = [
        msg
        for msg in warns
        if issubclass(msg.category, padne.solver.SolverWarning)
    ]

    # TODO: Store the project name in the problem/solution object
    project_name = args.kicad_pro_file.name
    return padne.ui.main(solution, project_name, captured_warnings)


@handle_errors
def do_solve(args):
    log = logging.getLogger(__name__)
    log.info(f"Loading KiCad project: {args.kicad_pro_file}")
    prob = padne.kicad.load_kicad_project(args.kicad_pro_file)
    log.info("Solving problem...")
    mesher_config = mesher_config_from_args(args)
    solution = padne.solver.solve(prob, mesher_config=mesher_config)
    with open(args.output_file, "wb") as f:
        pickle.dump(solution, f)
    log.info(f"Solution saved to {args.output_file}")


def do_show(args):
    log = logging.getLogger(__name__)
    log.info(f"Loading solution from: {args.solution_file}")
    with open(args.solution_file, "rb") as f:
        solution = pickle.load(f)
    project_name = args.solution_file.name
    return padne.ui.main(solution, project_name)


@handle_errors
def do_paraview(args):
    log = logging.getLogger(__name__)
    log.info(f"Loading solution from: {args.solution_file}")
    with open(args.solution_file, "rb") as f:
        solution = pickle.load(f)
    padne.paraview.export_solution(solution, args.output_dir)
    log.info(f"ParaView export completed: {args.output_dir}")


def main():
    args = parse_args()
    setup_logging(args.debug)

    log = logging.getLogger(__name__)
    log.debug(f"Parsed arguments: {args}")

    command_func = {
        "gui": do_gui,
        "solve": do_solve,
        "show": do_show,
        "paraview": do_paraview,
    }[args.command]
    result = command_func(args)

    if isinstance(result, int):
        sys.exit(result)


if __name__ == "__main__":
    main()
