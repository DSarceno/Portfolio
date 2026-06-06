"""Command-line interface for :mod:`simgen`.

Subcommands
-----------
``generate``  Generate a project from a prompt or a library phenomenon.
``list``      List the offline reference phenomena.
``info``      Show resolved configuration.

User-facing output is written to ``stdout`` via :func:`_echo`; diagnostic
logging goes to ``stderr`` (see :mod:`simgen.utilities.logging_config`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from simgen.config.settings import get_settings
from simgen.llm.generator import SimulationGenerator
from simgen.simulations import library
from simgen.simulations.runner import run_spec
from simgen.utilities.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def _echo(message: str = "") -> None:
    """Write a line to stdout (the CLI's user-facing channel)."""
    sys.stdout.write(message + "\n")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="simgen",
        description="Generate math/physics simulations, plots, animations and reports from a prompt.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (INFO) logging."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate a project from a prompt or library spec.")
    gen.add_argument("prompt", nargs="?", default=None, help="Natural-language phenomenon description.")
    gen.add_argument(
        "--from-library",
        metavar="SLUG",
        default=None,
        help="Use a built-in reference spec instead of an LLM prompt.",
    )
    gen.add_argument(
        "-o", "--output", default=None, help="Output directory (default: from settings)."
    )
    gen.add_argument("--run", action="store_true", help="Run the simulation after generating.")
    gen.add_argument("--plot", action="store_true", help="Render plots after running.")
    gen.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of falling back to the offline library when the LLM is unavailable.",
    )
    gen.set_defaults(func=_cmd_generate)

    lst = sub.add_parser("list", help="List the offline reference phenomena.")
    lst.set_defaults(func=_cmd_list)

    info = sub.add_parser("info", help="Show resolved configuration.")
    info.set_defaults(func=_cmd_info)

    return parser


def _cmd_list(_: argparse.Namespace) -> int:
    """Handle the ``list`` subcommand."""
    _echo("Available offline reference phenomena:")
    for slug in library.list_specs():
        spec = library.get_spec(slug)
        _echo(f"  {slug:<28} {spec.name}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    """Handle the ``info`` subcommand."""
    settings = get_settings()
    _echo("simgen configuration:")
    _echo(f"  model           : {settings.model}")
    _echo(f"  max_tokens      : {settings.max_tokens}")
    _echo(f"  output_dir      : {settings.output_dir}")
    _echo(f"  api_key_present : {settings.has_api_key}")
    _echo(f"  project_root    : {settings.project_root}")
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    """Handle the ``generate`` subcommand."""
    if args.from_library:
        try:
            spec = library.get_spec(args.from_library)
        except KeyError as exc:
            _echo(f"error: {exc}")
            return 2
    elif args.prompt:
        generator = SimulationGenerator(allow_fallback=not args.no_fallback)
        spec = generator.spec_from_prompt(args.prompt)
    else:
        _echo("error: provide a PROMPT or --from-library SLUG")
        return 2

    output_dir = args.output or str(get_settings().output_dir)
    generator = SimulationGenerator()
    project = generator.generate(spec, output_dir)

    _echo(f"Generated project for '{spec.name}' ({spec.slug}):")
    for key, path in project.as_dict().items():
        _echo(f"  {key:<12}: {path}")

    if args.run or args.plot:
        result = run_spec(spec)
        _echo(
            f"Simulation: {result.n_steps} steps, "
            f"final state = {result.states[-1].round(4).tolist()} "
            f"({'ok' if result.success else 'FAILED: ' + result.message})"
        )
        if args.plot:
            from simgen.visualization.plotting import render_all_plots

            paths = render_all_plots(spec, result, Path(project.root))
            for path in paths:
                _echo(f"  figure      : {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Parameters
    ----------
    argv:
        Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging("INFO" if args.verbose else "WARNING", force=True)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        logger.exception("Command failed")
        _echo(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
