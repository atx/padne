
import argparse

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
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
    {
        "gui": do_gui,
    }[args.command](args)


if __name__ == "__main__":
    main()
