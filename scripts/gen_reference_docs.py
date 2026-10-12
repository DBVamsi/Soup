"""Render the two generated reference pages from the code.

``docs/config-reference.md`` lists every key ``soup.yaml`` accepts, walked out of
``SoupConfig.model_fields``. ``docs/cli-reference.md`` lists every command, option and
argument, walked out of the Click tree behind the Typer app. Neither page is written
by hand: change the schema field or the option, then run this script.

    python scripts/gen_reference_docs.py            # rewrite the pages that changed
    python scripts/gen_reference_docs.py --check    # exit 1 and name each stale page

``tests/test_gen_reference_docs.py`` runs the check on every interpreter and OS in the
CI matrix, so the same text has to come out everywhere. What keeps it that way:

* Types are spelled here, from ``typing`` introspection, never from ``repr()``:
  ``Optional[int]``, ``Union[int, None]`` and ``int | None`` all read ``int | null``.
* Nothing is taken from ``--help`` output, which carries ANSI colour and wraps at the
  terminal width. Commands and parameters are read off the Click objects; Rich markup
  in help strings is removed by a regex of its own.
* Field and parameter order is the declaration order; command names are sorted.
* A default that only this machine could have produced (a path under the home or the
  working directory, a drive path, a callable, a non-container factory value) is
  printed as ``(set at runtime)``, never as its value. Environment variables are named,
  never read.
* The two completion options Typer adds change shape with shell detection, so their
  rows are fixed text.
* Pages are written as UTF-8 with ``\\n`` newlines; the check compares text, so a CRLF
  checkout (``core.autocrlf``) of an LF blob still passes.

A numeric default computed from the machine (``os.cpu_count()``) cannot be told from a
literal one. There is none today; the cross-OS check is what would catch the first.

Rendering imports the schema and the CLI, never torch: both stay on the light core.
"""

from __future__ import annotations

import argparse
import collections.abc
import enum
import json
import os
import re
import sys
import tempfile
import types
import typing
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Sequence

import click
from pydantic import BaseModel
from pydantic.fields import FieldInfo

_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
#: Where the pages are read and written. Tests point this at a scratch tree.
_REPO_ROOT = _SCRIPT_REPO_ROOT

CONFIG_PAGE = "docs/config-reference.md"
CLI_PAGE = "docs/cli-reference.md"
RUNTIME_DEFAULT = "(set at runtime)"

_REGENERATE = "python scripts/gen_reference_docs.py"
_NONE_TYPE = type(None)
_NOT_LITERAL = object()


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
#: ``&`` first, so the entities written for ``<`` are not escaped again; the backslash
#: before every escape that is itself written with one.
_PROSE_ESCAPES = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    ("\\", "\\\\"),
    ("`", "\\`"),
    ("*", "\\*"),
    ("[", "\\["),
    ("|", "\\|"),
    ("~", "\\~"),
    ("$", "\\$"),
)
#: An underscore inside a word (``snake_case``) never opens emphasis; any other one can.
_LOOSE_UNDERSCORE = re.compile(r"(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])")
#: What would turn a paragraph of its own into a heading, a quote or a list item.
_BLOCK_START = re.compile(r"^(?:#{1,6}(?=\s|$)|>|[+-](?=\s|$))")
_LIST_START = re.compile(r"^(\d+)([.)])(?=\s|$)")
_INTERNAL_PHASE = re.compile(r"\b(v\d+(?:\.\d+){1,2})\s+part\s+[a-z]\b", re.IGNORECASE)
#: Rich's own tag grammar (``rich.markup.RE_TAGS``): a ``[`` then a lowercase letter,
#: ``#``, ``/`` or ``@``, up to the next ``]``, with any backslashes in front of it.
_RICH_TAG = re.compile(r"(\\*)\[([a-z#/@][^\[\]]*)\]")
_RICH_COLOUR = re.compile(
    r"#[0-9a-f]{6}|color\(\d{1,3}\)|rgb\(\d{1,3},\d{1,3},\d{1,3}\)|gr[ae]y\d{0,3}"
    r"|(?:bright_)?(?:black|red|green|yellow|blue|magenta|cyan|white)"
)
_RICH_WORDS = frozenset(
    "bold b dim italic i underline u strike s reverse blink conceal link not on default".split()
)


def _escape_prose(text: str) -> str:
    for raw, safe in _PROSE_ESCAPES:
        text = text.replace(raw, safe)
    return _LOOSE_UNDERSCORE.sub(r"\\_", text)


def escape_cell(text: str) -> str:
    """``text`` as one table cell: a single line, no cell break, no stray markup.

    Code spans are kept as code (only their ``|`` is escaped, which a GitHub table
    undoes inside code); everything else has the characters that Markdown or HTML
    would interpret escaped, so ``<id>`` stays visible and ``*`` stays a star.
    """
    text = " ".join(text.split())
    parts: list[str] = []
    position = 0
    for match in _CODE_SPAN.finditer(text):
        parts.append(_escape_prose(text[position : match.start()]))
        parts.append(match.group(0).replace("|", "\\|"))
        position = match.end()
    parts.append(_escape_prose(text[position:]))
    return "".join(parts)


def _escape_paragraph(text: str) -> str:
    """``escape_cell`` for a paragraph of its own, which must not open a block."""
    escaped = _LIST_START.sub(r"\1\\\2", escape_cell(text))
    return f"\\{escaped}" if _BLOCK_START.match(escaped) else escaped


def _code(text: str) -> str:
    """``text`` as a code span that survives a table cell."""
    text = text.replace("\r", " ").replace("\n", " ")
    fence = "`"
    while fence in text:
        fence += "`"
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text.replace('|', chr(92) + '|')}{pad}{fence}"


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table_head(*titles: str) -> list[str]:
    return [_row(titles), "|" + "---|" * len(titles)]


def strip_internal_phase(text: str) -> str:
    """Drop the planning label from ``v0.42.0 Part B``; public docs keep the version."""
    return _INTERNAL_PHASE.sub(r"\1", text)


def _is_rich_style(tag: str) -> bool:
    body = tag[1:] if tag.startswith("/") else tag
    if not body:
        return True  # ``[/]`` closes whatever is open
    if body.startswith(("link=", "@")):
        return True
    return all(word in _RICH_WORDS or _RICH_COLOUR.fullmatch(word) for word in body.split())


def strip_rich_markup(text: str) -> str:
    """Remove Rich style tags from a help string and undo its ``\\[`` escapes.

    Only tags that are styles are removed. Rich itself also swallows ``[sigstore]`` in
    ``pip install soup-cli[sigstore]`` when the author forgot the backslash; the
    bracketed text is what was meant, so it is kept here.
    """

    def replace(match: re.Match[str]) -> str:
        slashes, tag = match.group(1), match.group(2)
        kept = "\\" * (len(slashes) // 2)
        if len(slashes) % 2 or not _is_rich_style(tag):
            return f"{kept}[{tag}]"
        return kept

    return _RICH_TAG.sub(replace, text)


def summary_line(help_text: str | None) -> str:
    """The first paragraph of a command's help, on one line."""
    text = (help_text or "").split("\f", 1)[0].strip()
    first = re.split(r"\n[ \t]*\n", text, maxsplit=1)[0]
    return " ".join(first.split())


def github_slug(heading: str) -> str:
    """The anchor GitHub gives a heading: lowercase, spaces dashed, and only letters,
    marks, decimal digits, ``_`` and ``-`` kept (so the subscript in ``CO₂`` is dropped)."""
    text = heading.replace("`", "").strip().lower()
    kept = []
    for char in text:
        category = unicodedata.category(char)
        if char == " ":
            kept.append("-")
        elif char == "-" or category[0] in "LM" or category in ("Nd", "Nl", "Pc"):
            kept.append(char)
    return "".join(kept)


class _Slugger:
    """Anchors in document order; GitHub numbers a repeated heading ``-1``, ``-2``."""

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}

    def take(self, heading: str) -> str:
        base = github_slug(heading)
        count = self._seen.get(base, 0)
        self._seen[base] = count + 1
        return base if not count else f"{base}-{count}"


# --------------------------------------------------------------------------
# Values that must read the same on every machine
# --------------------------------------------------------------------------


def _machine_markers() -> tuple[str, ...]:
    """This machine's home, working and temp directories, normalised for matching."""
    found = set()
    for raw in (os.path.expanduser("~"), os.getcwd(), tempfile.gettempdir()):
        marker = raw.replace("\\", "/").rstrip("/").casefold()
        if len(marker) > 4:  # "/", "c:" and "/tmp" would match constants
            found.add(marker)
    return tuple(sorted(found))


def _is_machine_specific(text: str) -> bool:
    """True for a string only this machine could have produced."""
    normalised = text.replace("\\", "/").casefold()
    if re.match(r"^[a-z]:/", normalised):
        return True  # a Windows drive path
    return any(
        re.search(re.escape(marker) + r"(?=/|$)", normalised) for marker in _machine_markers()
    )


def _plain_literal(value: Any) -> Any:
    """``value`` as JSON-ready data, or ``_NOT_LITERAL`` when it is not a stable literal."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, enum.Enum):
        return _plain_literal(value.value)
    if isinstance(value, PurePath):
        return _plain_literal(value.as_posix())
    if isinstance(value, str):
        return _NOT_LITERAL if _is_machine_specific(value) else value
    if isinstance(value, (set, frozenset)):
        try:
            value = sorted(value)
        except TypeError:
            return _NOT_LITERAL
    if isinstance(value, (list, tuple)):
        items = [_plain_literal(item) for item in value]
        return _NOT_LITERAL if any(item is _NOT_LITERAL for item in items) else items
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            plain = _plain_literal(item)
            if not isinstance(key, str) or plain is _NOT_LITERAL:
                return _NOT_LITERAL
            out[key] = plain
        return out
    return _NOT_LITERAL


def format_default(value: Any) -> str:
    """A default as a YAML flow value (``null``, ``true``, ``"auto"``, ``["a"]``).

    Anything that is not a plain literal, or that carries this machine's paths, is
    ``RUNTIME_DEFAULT``.
    """
    plain = _plain_literal(value)
    if plain is _NOT_LITERAL:
        return RUNTIME_DEFAULT
    return json.dumps(plain, ensure_ascii=False)


def _default_cell(text: str) -> str:
    return f"*{RUNTIME_DEFAULT}*" if text == RUNTIME_DEFAULT else _code(text)


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------

_SCALARS = {str: "str", int: "int", float: "float", bool: "bool", bytes: "bytes"}


def _is_class(annotation: Any) -> bool:
    """A real class, never a parametrised generic. ``isinstance(type[int], type)`` is
    true on Python 3.10 and false from 3.11, so asking only that would spell the same
    annotation two ways."""
    return typing.get_origin(annotation) is None and isinstance(annotation, type)


def _container_name(base: Any) -> str | None:
    """``list`` / ``dict`` / ``set`` / ``tuple`` for a container class or generic origin
    (``typing.List``, ``collections.abc.Sequence`` and ``list`` all read ``list``)."""
    if not isinstance(base, type) or issubclass(base, (str, bytes)):
        return None
    if issubclass(base, tuple):
        return "tuple"
    if issubclass(base, collections.abc.Mapping):
        return "dict"
    if issubclass(base, collections.abc.Set):
        return "set"
    if issubclass(base, collections.abc.Sequence):
        return "list"
    return None


def _is_model(annotation: Any) -> bool:
    return _is_class(annotation) and issubclass(annotation, BaseModel)


def _union_members(annotation: Any) -> list[Any]:
    """The members of a union, flattened, ``Annotated`` unwrapped, ``None`` as its type."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _union_members(typing.get_args(annotation)[0])
    if origin is typing.Union or origin is types.UnionType:
        members: list[Any] = []
        for arg in typing.get_args(annotation):
            members.extend(_union_members(arg))
        return members
    return [_NONE_TYPE if annotation is None else annotation]


def _choice_values(annotation: Any) -> list[Any] | None:
    """The values of a ``Literal`` or an ``Enum`` class, else ``None``."""
    if typing.get_origin(annotation) is typing.Literal:
        return [
            item.value if isinstance(item, enum.Enum) else item
            for item in typing.get_args(annotation)
        ]
    if _is_class(annotation) and issubclass(annotation, enum.Enum):
        return [member.value for member in annotation]
    return None


def _unique(items: Sequence[Any]) -> list[Any]:
    """Order-preserving, and ``1`` is not ``True``."""
    seen = set()
    out = []
    for item in items:
        key = (type(item), item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def literal_choices(annotation: Any) -> list[Any]:
    """The allowed values when the annotation is nothing but a choice (or ``None``).

    A choice mixed with another type (``int | "auto"``) is spelled in the type instead,
    so this is empty for it.
    """
    values: list[Any] = []
    for member in _union_members(annotation):
        if member is _NONE_TYPE:
            continue
        found = _choice_values(member)
        if found is None:
            return []
        values.extend(found)
    return _unique(values)


def _value_type(value: Any) -> str:
    return "null" if value is None else _SCALARS.get(type(value), type(value).__name__)


def _format_member(annotation: Any) -> str:
    if annotation is typing.Any:
        return "any"
    origin = typing.get_origin(annotation)
    container = _container_name(annotation if origin is None else origin)
    if container is not None:
        args = [arg for arg in typing.get_args(annotation) if arg is not Ellipsis]
        inner = ", ".join(format_type(arg) for arg in args)
        return f"{container}[{inner}]" if inner else container
    if not _is_class(annotation):
        return "any"
    if annotation in _SCALARS:
        return _SCALARS[annotation]
    if issubclass(annotation, BaseModel):
        return "section"
    if issubclass(annotation, PurePath):
        return "path"
    return annotation.__name__


def format_type(annotation: Any) -> str:
    """One spelling per type, whatever the source wrote and whatever Python runs this.

    ``null`` goes last. A pure choice reads as the type of its values (the values are
    listed as a constraint); a choice mixed with other types is spelled inline.
    """
    members = _union_members(annotation)
    rest = [member for member in members if member is not _NONE_TYPE]
    pure_choice = bool(rest) and all(_choice_values(member) is not None for member in rest)
    names: list[str] = []
    for member in rest:
        values = _choice_values(member)
        if values is None:
            names.append(_format_member(member))
        elif pure_choice:
            names.extend(_value_type(value) for value in values)
        else:
            names.extend(json.dumps(value, ensure_ascii=False) for value in values)
    if len(rest) != len(members):
        names.append("null")
    return " | ".join(dict.fromkeys(names))


# --------------------------------------------------------------------------
# docs/config-reference.md
# --------------------------------------------------------------------------

_BOUNDS = (("gt", ">"), ("ge", ">="), ("lt", "<"), ("le", "<="))
_CONSTRAINT_ATTRS = (
    "gt",
    "ge",
    "lt",
    "le",
    "multiple_of",
    "min_length",
    "max_length",
    "pattern",
    "allow_inf_nan",
)


@dataclass(frozen=True)
class ConfigRow:
    """One ``soup.yaml`` key, every cell already Markdown."""

    key: str
    type: str
    default: str
    constraints: str
    description: str


def format_constraints(info: FieldInfo) -> str:
    """The single-value rules of a field, as one Markdown cell."""
    found: dict[str, Any] = {}
    for meta in info.metadata:
        for attr in _CONSTRAINT_ATTRS:
            value = getattr(meta, attr, None)
            if value is not None:
                found[attr] = value
    bounds = [_code(f"{sign} {found[attr]!r}") for attr, sign in _BOUNDS if attr in found]
    parts = [", ".join(bounds)] if bounds else []
    if "multiple_of" in found:
        parts.append(f"multiple of {found['multiple_of']!r}")
    if "min_length" in found:
        parts.append(f"min length {found['min_length']}")
    if "max_length" in found:
        parts.append(f"max length {found['max_length']}")
    if "pattern" in found:
        pattern = found["pattern"]
        parts.append("pattern " + _code(getattr(pattern, "pattern", pattern)))
    if found.get("allow_inf_nan") is False:
        parts.append("finite")
    choices = literal_choices(info.annotation)
    if choices:
        parts.append(
            "one of " + ", ".join(_code(json.dumps(item, ensure_ascii=False)) for item in choices)
        )
    return "; ".join(parts)


def _config_default(info: FieldInfo) -> str:
    factory = info.default_factory
    if factory is not None:
        if _is_model(factory):
            return "section defaults"
        try:
            value = factory()  # type: ignore[call-arg]
        except Exception:
            return f"*{RUNTIME_DEFAULT}*"
        # A factory exists to build a fresh container. A scalar out of one was computed.
        if not isinstance(value, (list, tuple, dict, set, frozenset)):
            return f"*{RUNTIME_DEFAULT}*"
        return _default_cell(format_default(value))
    if info.is_required():
        return "**required**"
    return _default_cell(format_default(info.default))


def _nested_models(annotation: Any) -> list[tuple[str, type[BaseModel]]]:
    """``(path suffix, model)`` for every model a field holds: ``""`` for the field
    itself, ``"[]"`` for the items of a sequence, ``".*"`` for the values of a mapping."""
    found: list[tuple[str, type[BaseModel]]] = []
    for member in _union_members(annotation):
        if _is_model(member):
            found.append(("", member))
            continue
        container = _container_name(typing.get_origin(member))
        if container is None:
            continue
        args = [arg for arg in typing.get_args(member) if arg is not Ellipsis]
        if container == "dict":
            args, suffix = args[1:], ".*"
        else:
            suffix = "[]"
        for arg in args:
            found.extend((suffix + deeper, model) for deeper, model in _nested_models(arg))
    return found


def _yaml_key(name: str, info: FieldInfo) -> str:
    for alias in (info.validation_alias, info.alias):
        if isinstance(alias, str):
            return alias
    return name


def config_sections(
    model: type[BaseModel], prefix: str = "", _stack: tuple[type[BaseModel], ...] = ()
) -> list[tuple[str, list[ConfigRow]]]:
    """``(YAML path, rows)`` per section: the model's own keys, then every nested model
    depth-first in field order. A model already being rendered is not entered again."""
    stack = (*_stack, model)
    rows: list[ConfigRow] = []
    nested: list[tuple[str, type[BaseModel]]] = []
    for name, info in model.model_fields.items():
        key = _yaml_key(name, info)
        path = f"{prefix}.{key}" if prefix else key
        children = [
            (path + suffix, sub)
            for suffix, sub in _nested_models(info.annotation)
            if sub not in stack
        ]
        nested.extend(children)
        description = escape_cell(strip_internal_phase(info.description or ""))
        links = ", ".join(f"[{_code(child)}](#{github_slug(child)})" for child, _ in children)
        if links:
            if description and not description.endswith((".", "!", "?", ":")):
                description += "."
            description = f"{description} See {links}.".strip()
        rows.append(
            ConfigRow(
                key=path,
                type=_code(format_type(info.annotation)),
                default=_config_default(info),
                constraints=format_constraints(info),
                description=description,
            )
        )
    sections = [(prefix, rows)]
    for child_path, sub in nested:
        sections.extend(config_sections(sub, child_path, stack))
    return sections


def _load_config_model() -> type[BaseModel]:
    from soup_cli.config.schema import SoupConfig

    return SoupConfig


def render_config_reference(model: type[BaseModel] | None = None) -> str:
    """The text of ``docs/config-reference.md``."""
    sections = config_sections(model if model is not None else _load_config_model())
    total = sum(len(rows) for _, rows in sections)
    lines = [
        "# Config reference",
        "",
        "[← Back to the docs index](README.md)",
        "",
        "> **Generated file — do not edit by hand.**",
        "> It is rendered from `SoupConfig` in `src/soup_cli/config/schema.py` by",
        "> `scripts/gen_reference_docs.py`. To change a row, change the field in the schema and",
        f"> run `{_REGENERATE}`. The test suite runs",
        f"> `{_REGENERATE} --check`, which fails while this page is stale.",
        "",
        f"Every key `soup.yaml` accepts: {total} keys in {len(sections)} sections. A key that is",
        "not on this page **refuses the load** — see",
        "[Unknown config keys](backends-and-ops.md#unknown-config-keys) for the error and the",
        "nearest-name hint. The guides linked from the [docs index](README.md) explain what the",
        "settings do together; this page is the lookup table.",
        "",
        "How to read a table:",
        "",
        "- **Key** is the full YAML path: `training.lora.r` is `r:` under `lora:` under",
        "  `training:`.",
        "- **Type** uses `str`, `int`, `float`, `bool`, `list[...]` and `dict[...]`; `null` means",
        "  the key may be left out or set to `null`; `section` is a nested block with a table of",
        "  its own. A value in quotes (`\"auto\"`) is the one literal accepted next to the type.",
        "- **Default** is written the way YAML reads it (`null`, `true`, `\"auto\"`, `[\"a\"]`).",
        "  **required** means there is none. *(set at runtime)* marks a default computed on",
        "  the machine, which this page does not print.",
        "- **Constraints** are the rules on the value itself: bounds, lengths, the allowed",
        "  choices. Rules that span keys (a setting valid for one task or backend only) are",
        "  enforced by the schema's validators and named in the description.",
        "",
        *_table_head("Section", "Keys"),
    ]
    for path, rows in sections:
        title = _code(path) if path else "Top-level keys"
        anchor = github_slug(path) if path else "top-level-keys"
        lines.append(_row([f"[{title}](#{anchor})", str(len(rows))]))
    for path, rows in sections:
        lines += ["", f"## {_code(path)}" if path else "## Top-level keys", ""]
        lines += _table_head("Key", "Type", "Default", "Constraints", "Description")
        lines += [
            _row([_code(row.key), row.type, row.default, row.constraints, row.description])
            for row in rows
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# docs/cli-reference.md
# --------------------------------------------------------------------------

#: Typer builds these two as flags when it can detect the shell and as options that
#: take a shell name when it cannot (no ``shellingham``), with different help text.
_TYPER_COMPLETION_HELP = {
    "install_completion": "Install shell completion for the current shell.",
    "show_completion": "Print the shell completion script, to copy or customise it.",
}


@dataclass(frozen=True)
class CommandEntry:
    """One visible command and the names that lead to it (``("data", "inspect")``)."""

    path: tuple[str, ...]
    command: click.Command


def _visible_params(command: click.Command) -> list[click.Parameter]:
    return [param for param in command.params if not getattr(param, "hidden", False)]


def iter_commands(root: click.Command) -> list[CommandEntry]:
    """Every visible command below ``root``: a group before its subcommands, names sorted.

    A hidden group hides everything under it.
    """
    entries: list[CommandEntry] = []

    def walk(command: click.Command, path: tuple[str, ...]) -> None:
        if path:
            entries.append(CommandEntry(path, command))
        if isinstance(command, click.Group):
            for name in sorted(command.commands):
                sub = command.commands[name]
                if not getattr(sub, "hidden", False):
                    walk(sub, (*path, name))

    walk(root, ())
    return entries


def _is_typer_completion(param: click.Parameter) -> bool:
    return param.name in _TYPER_COMPLETION_HELP and not param.expose_value


def _range_words(param_type: Any) -> str:
    low, high = param_type.min, param_type.max
    if low is not None and high is not None and not (param_type.min_open or param_type.max_open):
        return f"{low!r} to {high!r}"
    parts = []
    if low is not None:
        parts.append(("above " if param_type.min_open else "at least ") + repr(low))
    if high is not None:
        parts.append(("below " if param_type.max_open else "at most ") + repr(high))
    return ", ".join(parts)


def _click_type_name(param_type: click.ParamType) -> str:
    if isinstance(param_type, click.Choice):
        values = [str(getattr(choice, "value", choice)) for choice in param_type.choices]
        return "one of " + ", ".join(_code(value) for value in values)
    if isinstance(param_type, (click.IntRange, click.FloatRange)):
        name = "integer" if isinstance(param_type, click.IntRange) else "float"
        return ", ".join(part for part in (name, _range_words(param_type)) if part)
    if isinstance(param_type, click.types.IntParamType):
        return "integer"
    if isinstance(param_type, click.types.FloatParamType):
        return "float"
    if isinstance(param_type, click.types.BoolParamType):
        return "boolean"
    if isinstance(param_type, click.Path):
        return "path"
    if isinstance(param_type, click.File):
        return "file"
    if isinstance(param_type, click.DateTime):
        return "datetime"
    return "text"


def format_click_type(param: click.Parameter) -> str:
    """The Type cell of a parameter, read off the Click object (never its ``repr``)."""
    if _is_typer_completion(param) or getattr(param, "is_flag", False):
        return "flag"
    if getattr(param, "count", False):
        return "counter"
    parts = [_click_type_name(param.type)]
    if param.nargs == -1:
        parts.append("any number")
    elif param.nargs != 1:
        parts.append(f"{param.nargs} values")
    if getattr(param, "multiple", False):
        parts.append("repeatable")
    return ", ".join(parts)


def format_click_default(param: click.Parameter) -> str:
    """The Default cell: the value as typed on the command line, ``on`` / ``off`` for a
    flag, empty when there is none, the placeholder when the machine computed it."""
    value = param.default
    if _is_typer_completion(param) or value is None:
        return ""
    if type(value).__module__.partition(".")[0] == "click":
        return ""  # Click's own "no default" sentinel
    if callable(value) and not isinstance(value, enum.Enum):
        return f"*{RUNTIME_DEFAULT}*"
    if getattr(param, "is_flag", False) and isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (list, tuple, set, frozenset, dict)) and not value:
        return ""
    plain = _plain_literal(value)
    if plain is _NOT_LITERAL:
        return f"*{RUNTIME_DEFAULT}*"
    if isinstance(plain, str):
        return _code(plain) if plain else _code('""')
    return _code(json.dumps(plain, ensure_ascii=False))


def _param_name(param: click.Parameter) -> str:
    if isinstance(param, click.Option):
        names = ", ".join(_code(opt) for opt in param.opts)
        if param.secondary_opts:
            names += " / " + ", ".join(_code(opt) for opt in param.secondary_opts)
        return names
    return _code(_metavar(param))


def _argument_name(param: click.Parameter) -> str:
    return getattr(param, "metavar", None) or (param.name or "").upper()


def _metavar(param: click.Parameter) -> str:
    name = _argument_name(param)
    return f"{name}..." if param.nargs == -1 else name


def _declared_argument_help(command: click.Command) -> dict[str, str]:
    """``{argument name: help}`` as declared to Typer.

    Click 8.5 gave ``Argument`` a ``help`` of its own and resets the attribute Typer
    0.20 has already set, so ``param.help`` is ``None`` there and the declared string
    on an older Click. The declaration reads the same on both.
    """
    if command.callback is None:
        return {}
    try:
        from typer.models import ArgumentInfo
        from typer.utils import get_params_from_function

        declared = get_params_from_function(command.callback)
    except Exception:
        return {}
    return {
        name: meta.default.help
        for name, meta in declared.items()
        if isinstance(meta.default, ArgumentInfo) and meta.default.help
    }


def _param_help(param: click.Parameter, declared: dict[str, str]) -> str:
    if _is_typer_completion(param):
        return _TYPER_COMPLETION_HELP[param.name or ""]
    raw = getattr(param, "help", None) or declared.get(param.name or "") or ""
    text = escape_cell(strip_internal_phase(strip_rich_markup(raw)))
    envvar = getattr(param, "envvar", None)
    if envvar:
        names = [envvar] if isinstance(envvar, str) else list(envvar)
        if text and not text.endswith((".", "!", "?", ":")):
            text += "."
        text = f"{text} Env: {', '.join(_code(name) for name in names)}.".strip()
    return text


def _param_rows(command: click.Command) -> list[str]:
    declared = _declared_argument_help(command)
    return [
        _row(
            [
                _param_name(param),
                format_click_type(param),
                format_click_default(param),
                "yes" if param.required else "",
                _param_help(param, declared),
            ]
        )
        for param in _visible_params(command)
    ]


def _usage(path: tuple[str, ...], command: click.Command) -> str:
    params = _visible_params(command)
    parts = ["soup", *path]
    if any(isinstance(param, click.Option) for param in params):
        parts.append("[OPTIONS]")
    if isinstance(command, click.Group):
        # `<command>`, not Click's `COMMAND`: the docs guard that resolves every
        # documented invocation (#1465) reads a bare capitalised word as a subcommand.
        optional = bool(path) and command.invoke_without_command
        parts.append("[<command>] [ARGS]..." if optional else "<command> [ARGS]...")
    for param in params:
        if isinstance(param, click.Argument):
            name = _argument_name(param)
            name = name if param.required else f"[{name}]"
            parts.append(f"{name}..." if param.nargs == -1 else name)
    passes_extra = (command.context_settings or {}).get("allow_extra_args")
    variadic = any(isinstance(param, click.Argument) and param.nargs == -1 for param in params)
    if passes_extra and not variadic and not isinstance(command, click.Group):
        parts.append("[ARGS]...")
    return " ".join(parts)


def _command_summary(command: click.Command) -> str:
    return _escape_paragraph(strip_internal_phase(strip_rich_markup(summary_line(command.help))))


def _command_section(entry: CommandEntry, anchors: dict[tuple[str, ...], str]) -> list[str]:
    command = entry.command
    level = "#" * min(len(entry.path) + 1, 6)
    blocks = [[f"{level} {_code('soup ' + ' '.join(entry.path))}"]]
    summary = _command_summary(command)
    if summary:
        blocks.append([summary])
    blocks.append([f"Usage: {_code(_usage(entry.path, command))}"])
    if isinstance(command, click.Group):
        subs = [
            f"[{_code(name)}](#{anchors[(*entry.path, name)]})"
            for name in sorted(command.commands)
            if (*entry.path, name) in anchors
        ]
        if subs:
            blocks.append(["Subcommands: " + ", ".join(subs)])
    rows = _param_rows(command)
    if rows:
        blocks.append([*_table_head("Name", "Type", "Default", "Required", "Help"), *rows])
    elif not isinstance(command, click.Group):
        blocks.append(["Takes no options or arguments."])
    return [line for block in blocks for line in ("", *block)]


def _load_cli_root() -> click.Command:
    import typer

    from soup_cli.cli import app

    return typer.main.get_command(app)


def render_cli_reference(root: click.Command | None = None) -> str:
    """The text of ``docs/cli-reference.md``."""
    root = root if root is not None else _load_cli_root()
    entries = iter_commands(root)
    slugger = _Slugger()
    for fixed in ("CLI reference", "Global options", "Commands"):
        slugger.take(fixed)
    anchors = {entry.path: slugger.take("soup " + " ".join(entry.path)) for entry in entries}

    groups = sum(isinstance(entry.command, click.Group) for entry in entries)
    params = [
        param
        for command in (root, *(entry.command for entry in entries))
        for param in _visible_params(command)
    ]
    options = sum(isinstance(param, click.Option) for param in params)
    lines = [
        "# CLI reference",
        "",
        "[← Back to the docs index](README.md)",
        "",
        "> **Generated file — do not edit by hand.**",
        "> It is rendered from the live Typer command tree (`soup_cli.cli:app`) by",
        "> `scripts/gen_reference_docs.py`. To change a row, change the option in the code and",
        f"> run `{_REGENERATE}`. The test suite runs",
        f"> `{_REGENERATE} --check`, which fails while this page is stale.",
        "",
        f"Every `soup` command with every option and argument: {len(entries) - groups} commands",
        f"in {groups} command groups, {options} options and {len(params) - options} arguments.",
        "For a one-line-per-task listing see the [Command reference](commands.md); for the keys",
        "of `soup.yaml` see the [Config reference](config-reference.md).",
        "",
        "How to read a table:",
        "",
        "- **Name** is an option with its short form, or an argument in capitals (`PATH`;",
        "  `PATH...` takes any number of values). `--flag` / `--no-flag` is one switch with",
        "  both spellings.",
        "- **Type**: a `flag` takes no value; `text`, `integer`, `float` and `path` take one;",
        "  a `repeatable` option may be given more than once.",
        "- **Default** is the value used when the option is left out (`on` / `off` for a flag);",
        "  an empty cell means there is none. *(set at runtime)* marks a default computed on",
        "  the machine, which this page does not print.",
        "- **Required** says `yes` when the command refuses to run without it.",
        "",
        "`--help` works on every command and is not listed. Global options go before the",
        "command name: `soup --verbose train --config soup.yaml`.",
        "",
        "## Global options",
        "",
        f"Usage: {_code(_usage((), root))}",
        "",
        *_table_head("Name", "Type", "Default", "Required", "Help"),
        *_param_rows(root),
        "",
        "## Commands",
        "",
        *_table_head("Command", "Summary"),
    ]
    for entry in entries:
        if len(entry.path) == 1:
            link = f"[{_code('soup ' + entry.path[0])}](#{anchors[entry.path]})"
            lines.append(_row([link, _command_summary(entry.command)]))
    for entry in entries:
        lines += _command_section(entry, anchors)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Check / write
# --------------------------------------------------------------------------


def render_pages() -> dict[str, str]:
    """``{repo-relative path: text}`` for both pages."""
    return {
        CONFIG_PAGE: render_config_reference(_load_config_model()),
        CLI_PAGE: render_cli_reference(_load_cli_root()),
    }


def _committed(root: Path, rel: str) -> str | None:
    """The page on disk with LF newlines, or ``None`` when there is no such file."""
    path = root / rel
    if not path.is_file():
        return None
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def stale_pages(root: Path, pages: dict[str, str]) -> list[str]:
    """The pages under ``root`` that are missing or differ from ``pages``."""
    return [rel for rel, text in pages.items() if _committed(root, rel) != text]


def _first_difference(committed: str, fresh: str) -> list[str]:
    old, new = committed.split("\n"), fresh.split("\n")
    index = next(
        (i for i, (left, right) in enumerate(zip(old, new)) if left != right),
        min(len(old), len(new)),
    )

    def shown(lines: list[str]) -> str:
        text = lines[index] if index < len(lines) else "<end of file>"
        return text[:160].encode("ascii", "backslashreplace").decode("ascii")

    return [
        f"  first difference at line {index + 1}:",
        f"    committed: {shown(old)}",
        f"    rendered:  {shown(new)}",
    ]


def _ensure_checkout_on_path() -> None:
    """Import ``soup_cli`` from this checkout when the script is run by path."""
    src = str(_SCRIPT_REPO_ROOT / "src")
    if "soup_cli" not in sys.modules and src not in sys.path:
        sys.path.insert(0, src)


def _foreign_install() -> str | None:
    """Where ``soup_cli`` was imported from, when that is not this checkout's ``src``."""
    import soup_cli

    src = os.path.normcase(os.path.realpath(_SCRIPT_REPO_ROOT / "src"))
    origin = os.path.normcase(os.path.realpath(soup_cli.__file__))
    try:
        inside = os.path.commonpath([src, origin]) == src
    except ValueError:  # different drives
        inside = False
    return None if inside else soup_cli.__file__


def main(argv: Sequence[str] | None = None) -> int:
    """Rewrite the pages, or with ``--check`` report the stale ones and exit 1."""
    parser = argparse.ArgumentParser(
        description="Render docs/config-reference.md and docs/cli-reference.md from the code."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 and name each page that differs from a fresh render",
    )
    args = parser.parse_args(argv)

    _ensure_checkout_on_path()
    foreign = _foreign_install()
    if foreign is not None:
        print(f"soup_cli is imported from {foreign}, not from {_SCRIPT_REPO_ROOT / 'src'}.")
        print("The pages would describe that copy. Install this checkout (pip install -e .)")
        print("or put its src directory first on PYTHONPATH, then run this again.")
        return 2

    pages = render_pages()
    stale = stale_pages(_REPO_ROOT, pages)
    if not stale:
        state = "up to date" if args.check else "up to date; nothing to do"
        print(f"{CONFIG_PAGE} and {CLI_PAGE} are {state}.")
        return 0
    if args.check:
        for rel in stale:
            committed = _committed(_REPO_ROOT, rel)
            if committed is None:
                print(f"{rel} is missing.")
                continue
            print(f"{rel} is stale.")
            for line in _first_difference(committed, pages[rel]):
                print(line)
        print(f"Run `{_REGENERATE}` and commit the result.")
        return 1
    for rel in stale:
        target = _REPO_ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pages[rel].encode("utf-8"))
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
