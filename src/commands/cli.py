"""The dispatcher shared by scripts/data.py, train.py, evaluate.py and analyze.py.

Each CLI maps a subcommand to a module in this package, or to a function for
the few that are not modules. Only the chosen module is imported, so a CPU
subcommand never pulls in torch, unsloth, trl or transformers.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping

Target = str | Callable[[list[str]], int]


def usage(prog: str, description: str, commands: Mapping[str, tuple[Target, str]]) -> str:
    width = max(len(name) for name in commands)
    lines = [f"usage: {prog} <subcommand> [options]", "", description, "", "subcommands:"]
    lines += [f"  {name:<{width}}  {help_text}" for name, (_, help_text) in commands.items()]
    lines += ["", f"Run `{prog} <subcommand> --help` for that subcommand's options."]
    return "\n".join(lines)


def run(
    prog: str,
    description: str,
    commands: Mapping[str, tuple[Target, str]],
    argv: list[str] | None = None,
) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(usage(prog, description, commands), file=sys.stdout if argv else sys.stderr)
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name not in commands:
        print(
            f"{prog}: unknown subcommand {name!r}; choose from {', '.join(commands)}",
            file=sys.stderr,
        )
        return 2

    target = commands[name][0]
    if not isinstance(target, str):
        return target(rest)
    module = importlib.import_module(f"commands.{target}")
    # argparse names the program after sys.argv[0], so help and errors read
    # "analyze.py failure-taxonomy" rather than the bare CLI name.
    sys.argv[0] = f"{prog} {name}"
    try:
        return module.main(rest)
    except getattr(module, "REFUSALS", ()) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
