
import argparse
import logging # Add this import
from pathlib import Path


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
    # TODO: Improve the UX here
    parser_gui.add_argument(
        "--just-solve",
        action="store_true",
        help="Exit immediately after producing the solution. Useful for profiling. ",
    )

    return parser.parse_args()


def do_gui(args):
    import padne.ui
    padne.ui.main(args)


def main():
    args = parse_args()
    setup_logging(args.debug)
    
    log = logging.getLogger(__name__)
    log.debug(f"Parsed arguments: {args}")

    {
        "gui": do_gui,
    }[args.command](args)


if __name__ == "__main__":
    main()
