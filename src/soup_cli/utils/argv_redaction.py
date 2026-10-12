"""Mask the values of credential-bearing CLI options before argv is persisted.

The audit log (``utils/audit_log.py``) records every command's arguments. Its
value-pattern layer only recognises a few token shapes (``hf_*``, ``sk-*``,
``Bearer ...``), so a credential passed as ``--tool-auth-token Zq3v...`` or
``--api-key gsk_...`` would be written as typed. This module masks by OPTION
NAME instead: any value-taking option whose long name contains ``token``,
``key``, ``secret``, ``password``, ``passwd`` or ``credential`` has its value
replaced with ``<redacted>``, unless the name is listed in
``NOT_SECRET_OPTIONS``.

The command tree is read by attribute (``commands``, ``params``, ``opts``),
never by class: Typer builds it from whichever click it was installed with,
and that is not always the ``click`` package this module could import. When
the tree yields no credential option at all, :func:`mask_args_by_name` masks
on the option names typed on the command line instead.

Stdlib only: importing this module must stay cheap for the CLI.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    import click

REDACTED = "<redacted>"

# Applied to the long option name without its leading dashes.
SECRET_NAME_RE = re.compile(r"token|key|secret|password|passwd|credential", re.IGNORECASE)

# Long option names that match SECRET_NAME_RE but never carry a credential.
# When unsure, leave a name OUT of this set: masking a harmless value costs
# nothing, keeping a credential in the audit file does.
NOT_SECRET_OPTIONS: frozenset[str] = frozenset(
    {
        "--max-tokens",  # generation length (an integer)
        "--max-new-tokens",  # generation length (an integer)
        "--num-assistant-tokens",  # speculative-decoding draft length (an integer)
        "--num-speculative-tokens",  # speculative-decoding draft length (an integer)
        "--tokenizer",  # tokenizer repo id or local path
        "--special-token",  # a vocabulary token string, e.g. "<|im_end|>"
        "--echo-trap-tokenizer-aware",  # a mode switch, not a credential
        "--show-token",  # a display switch, not a credential
        "--public-key",  # path to a PUBLIC verification key
        "--generate-key",  # output path for a newly generated key pair
    }
)


def _long_names(param: object) -> list[str]:
    """The ``--name`` strings of a parameter; an argument has none."""
    declared = (
        *(getattr(param, "opts", None) or ()),
        *(getattr(param, "secondary_opts", None) or ()),
    )
    return [o for o in declared if o.startswith("--")]


def _is_secret_option(option: object) -> bool:
    if getattr(option, "is_flag", False) or getattr(option, "count", False):
        return False
    longs = _long_names(option)
    if not any(SECRET_NAME_RE.search(name[2:]) for name in longs):
        return False
    return not any(name in NOT_SECRET_OPTIONS for name in longs)


def secret_option_strings(command: click.Command) -> frozenset[str]:
    """Every option string (long and short) of ``command``'s credential options."""
    found: set[str] = set()
    for param in getattr(command, "params", None) or ():
        if _is_secret_option(param):
            found.update(param.opts)
    return frozenset(found)


def subcommands(command: object) -> dict | None:
    """The name -> command mapping of a group, or ``None`` for a leaf command."""
    found = getattr(command, "commands", None)
    return found if isinstance(found, dict) else None


def iter_commands(command: click.Command) -> Iterable[click.Command]:
    """Yield ``command`` and every command below it."""
    yield command
    for sub in (subcommands(command) or {}).values():
        yield from iter_commands(sub)


def all_secret_option_strings(root: click.Command) -> frozenset[str]:
    """Union of :func:`secret_option_strings` over the whole command tree."""
    found: set[str] = set()
    for command in iter_commands(root):
        found.update(secret_option_strings(command))
    return frozenset(found)


def resolve_command(root: click.Command, args: Sequence[str]) -> click.Command:
    """Walk ``args`` down the group tree and return the deepest command named.

    At each group, the first token that does not start with ``-`` is looked
    up as a subcommand; when it is not one, the walk stops at that group.
    """
    current = root
    tokens = list(args)
    while (below := subcommands(current)) is not None:
        index = next(
            (i for i, tok in enumerate(tokens) if not str(tok).startswith("-")),
            None,
        )
        if index is None:
            break
        sub = below.get(str(tokens[index]))
        if sub is None:
            break
        current = sub
        tokens = tokens[index + 1:]
    return current


def _mask_short_bundle(tok: str, short_opts: set[str]) -> tuple[str, bool] | None:
    """Mask a bundled short-option token, or return ``None`` if it holds no secret.

    Click allows short options to be bundled: ``-xtVALUE`` is ``-x`` (a flag)
    followed by ``-t VALUE``, and ``-xt VALUE`` puts the value in the next
    token. Matching only ``tok[:2]`` therefore missed every secret typed behind
    another short flag.

    Returns ``(masked_token, mask_next_token)``. Everything up to and including
    the secret option letter is kept — those are flag names, not credentials,
    and they are what makes the audit line readable.
    """
    for index, char in enumerate(tok[1:], start=1):
        if f"-{char}" not in short_opts:
            continue
        if index == len(tok) - 1:
            # The secret option ends the bundle: its value is the next token.
            return tok, True
        return f"{tok[:index + 1]}{REDACTED}", False
    return None


def mask_secret_args(args: Sequence[str], secret_opts: frozenset[str]) -> tuple[str, ...]:
    """Return ``args`` with the value of every option in ``secret_opts`` masked.

    Handles ``--opt value``, ``--opt=value``, ``-o value``, ``-oVALUE`` and
    bundled short options (``-xoVALUE`` / ``-xo VALUE``, where ``-x`` is a flag
    and ``-o`` takes the value). An option with no following value is kept as
    is.
    """
    long_opts = {o for o in secret_opts if o.startswith("--")}
    short_opts = {o for o in secret_opts if not o.startswith("--") and len(o) == 2}
    out: list[str] = []
    mask_next = False
    for raw in args:
        tok = str(raw)
        if mask_next:
            out.append(REDACTED)
            mask_next = False
            continue
        if tok in secret_opts:
            out.append(tok)
            mask_next = True
            continue
        if tok.startswith("--") and "=" in tok:
            name = tok.split("=", 1)[0]
            if name in long_opts:
                out.append(f"{name}={REDACTED}")
                continue
        elif tok.startswith("-") and not tok.startswith("--") and len(tok) > 1:
            masked = _mask_short_bundle(tok, short_opts)
            if masked is not None:
                out.append(masked[0])
                mask_next = masked[1]
                continue
        out.append(tok)
    return tuple(out)


def mask_args_by_name(args: Sequence[str]) -> tuple[str, ...]:
    """Mask the value of every long option whose NAME looks like a credential.

    Used when the command tree yields no credential option at all, so that
    situation masks more rather than nothing. Reads only what was typed:
    ``--name value`` and ``--name=value`` are masked when ``name`` matches
    ``SECRET_NAME_RE`` and is not in ``NOT_SECRET_OPTIONS``. A switch with such
    a name hides the token after it, which costs nothing; short options cannot
    be told apart by name and are left to the value-pattern layer.
    """
    out: list[str] = []
    mask_next = False
    for raw in args:
        tok = str(raw)
        if mask_next:
            out.append(REDACTED)
            mask_next = False
            continue
        if tok.startswith("--"):
            name, has_value, _ = tok.partition("=")
            if SECRET_NAME_RE.search(name[2:]) and name not in NOT_SECRET_OPTIONS:
                out.append(f"{name}={REDACTED}" if has_value else tok)
                mask_next = not has_value
                continue
        out.append(tok)
    return tuple(out)
