"""Project a real CT with the same input and output path as the installed CLI."""

import argparse

from dpt.cli import project_parser, run_projection
from dpt.rendering import save_projection_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_parser(parser)
    parser.set_defaults(views=["ap", "lateral", "oblique"])
    args = parser.parse_args()
    depth, volume, geometry, _ = run_projection(args)
    save_projection_figure(volume, depth, geometry, tuple(args.views), args.output)
    print(f"Comparison figure: {(args.output / 'ct-projections.png').resolve()}")


if __name__ == "__main__":
    main()
