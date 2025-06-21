import argparse
import logging
import pickle
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import padne.kicad
import padne.solver
import padne.ui


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging output."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_gui = subparsers.add_parser("gui", help="Run the GUI")
    parser_gui.add_argument(
        "kicad_pro_file",
        type=Path,
        help="Path to the input file",
    )

    parser_show = subparsers.add_parser("show", help="Load and display a pre-computed solution")
    parser_show.add_argument(
        "solution_file",
        type=Path,
        help="Path to the pickled solution file",
    )

    parser_solve = subparsers.add_parser("solve", help="Solve the problem and save the solution")
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

    return parser.parse_args()


@contextmanager
def handle_errors():
    """Context manager for handling errors with enhanced display."""
    try:
        yield
    except Exception as e:
        # Print the normal traceback
        traceback.print_exc()

        # Print the exception message in bold yellow
        print(f"\033[1;33m{str(e)}\033[0m")

        # Exit with error code
        sys.exit(1)

def do_gui(args):
    with handle_errors():
        log = logging.getLogger(__name__)
        log.info(f"Loading KiCad project for GUI: {args.kicad_pro_file}")
        prob = padne.kicad.load_kicad_project(args.kicad_pro_file)
        log.info("Solving problem for GUI...")
        solution = padne.solver.solve(prob)
        # TODO: Store the project name in the problem/solution object
        project_name = args.kicad_pro_file.name
        return padne.ui.main(solution, project_name)


def do_solve(args):
    with handle_errors():
        log = logging.getLogger(__name__)
        log.info(f"Loading KiCad project: {args.kicad_pro_file}")
        prob = padne.kicad.load_kicad_project(args.kicad_pro_file)
        log.info("Solving problem...")
        solution = padne.solver.solve(prob)
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


def main():
    args = parse_args()
    setup_logging(args.debug)

    log = logging.getLogger(__name__)
    log.debug(f"Parsed arguments: {args}")

    command_func = {
        "gui": do_gui,
        "solve": do_solve,
        "show": do_show,
    }[args.command]
    result = command_func(args)

    if isinstance(result, int):
        sys.exit(result)


if __name__ == "__main__":
    main()
