"""Credential options are found by reading the command tree, whatever built it."""

from __future__ import annotations

import pytest
import typer

from soup_cli.utils.argv_redaction import (
    REDACTED,
    all_secret_option_strings,
    iter_commands,
    mask_args_by_name,
    mask_secret_args,
    resolve_command,
    secret_option_strings,
    subcommands,
)

# ---------------------------------------------------------------------------
# A command tree whose classes share nothing with the importable click package,
# as Typer builds it when it carries its own copy.
# ---------------------------------------------------------------------------


class _Option:
    def __init__(self, *opts, secondary=(), is_flag=False, count=False):
        self.opts = list(opts)
        self.secondary_opts = list(secondary)
        self.is_flag = is_flag
        self.count = count


class _Argument:
    def __init__(self, name):
        self.opts = [name]
        self.secondary_opts = []


class _Command:
    def __init__(self, name, params=()):
        self.name = name
        self.params = list(params)


class _Group(_Command):
    def __init__(self, name, commands, params=()):
        super().__init__(name, params)
        self.commands = {command.name: command for command in commands}


@pytest.fixture
def tree():
    serve = _Command(
        "serve",
        [
            _Option("--model", "-m"),
            _Option("--api-key"),
            _Option("--tool-auth-token", "-t"),
            _Option("--use-auth-token", is_flag=True),
            _Option("--token-verbosity", "-v", count=True),
            _Option("--max-tokens"),
            _Option("--cache", secondary=["--no-key-cache"], is_flag=True),
            _Argument("token"),
        ],
    )
    generate = _Command("generate", [_Option("--api-key", "-k"), _Option("--rows")])
    data = _Group("data", [generate], [_Option("--data-secret")])
    return _Group("soup", [serve, data])


def test_every_command_of_a_foreign_tree_is_visited(tree):
    assert [command.name for command in iter_commands(tree)] == [
        "soup",
        "serve",
        "data",
        "generate",
    ]


def test_credential_options_of_a_foreign_tree_are_found(tree):
    assert all_secret_option_strings(tree) == {
        "--api-key",
        "--tool-auth-token",
        "-t",
        "-k",
        "--data-secret",
    }


def test_only_value_taking_options_of_a_command_count(tree):
    """A switch, a counter, a listed non-credential name and an argument do not."""
    serve = tree.commands["serve"]
    assert secret_option_strings(serve) == {"--api-key", "--tool-auth-token", "-t"}


def test_group_and_leaf_are_told_apart_by_their_subcommands(tree):
    data = tree.commands["data"]
    assert subcommands(tree) is tree.commands
    assert subcommands(data) is data.commands
    assert subcommands(tree.commands["serve"]) is None
    assert subcommands(_Group("empty", [])) == {}


@pytest.mark.parametrize(
    ("args", "name"),
    [
        (["serve", "--api-key", "VALUE"], "serve"),
        (["--verbose", "data", "generate", "--rows", "3"], "generate"),
        (["data"], "data"),
        (["data", "no-such-command"], "data"),
        (["no-such-command"], "soup"),
        ([], "soup"),
    ],
)
def test_arguments_resolve_down_a_foreign_tree(tree, args, name):
    assert resolve_command(tree, args).name == name


def test_values_are_masked_with_the_options_of_a_foreign_tree(tree):
    args = ["--model", "m", "--api-key", "VALUE-ONE", "--tool-auth-token=VALUE-TWO", "-t", "V3"]
    assert mask_secret_args(args, all_secret_option_strings(tree)) == (
        "--model",
        "m",
        "--api-key",
        REDACTED,
        f"--tool-auth-token={REDACTED}",
        "-t",
        REDACTED,
    )


# ---------------------------------------------------------------------------
# The real command tree
# ---------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch):
    import soup_cli.cli as cli_module

    monkeypatch.setattr(cli_module, "_ALL_SECRET_OPTION_STRINGS", None)
    return cli_module


def test_the_real_tree_is_walked_and_has_credential_options(cli):
    root = typer.main.get_command(cli.app)
    assert sum(1 for _ in iter_commands(root)) > 200
    assert {"--api-key", "--auth-token", "--tool-auth-token"} <= all_secret_option_strings(root)


def test_the_real_tree_masks_by_option_name(cli):
    args = ["--model", "m", "--api-key", "VALUE-ONE", "--tool-auth-token=VALUE-TWO"]
    assert cli._mask_audit_args("serve", args) == (
        "--model",
        "m",
        "--api-key",
        REDACTED,
        f"--tool-auth-token={REDACTED}",
    )


def test_a_group_is_masked_with_every_option_of_the_app(cli):
    """`-t` is a credential option of some command, so a bare group masks it."""
    assert cli._mask_audit_args("data", ["-t", "VALUE"]) == ("-t", REDACTED)


# ---------------------------------------------------------------------------
# When the tree yields no credential option, the names typed decide
# ---------------------------------------------------------------------------


def test_an_empty_option_set_masks_by_the_names_on_the_command_line(cli, monkeypatch):
    monkeypatch.setattr(cli, "_ALL_SECRET_OPTION_STRINGS", frozenset())
    args = [
        "--model",
        "m",
        "--api-key",
        "VALUE-ONE",
        "--tool-auth-token=VALUE-TWO",
        "--max-tokens",
        "64",
    ]
    assert cli._mask_audit_args("serve", args) == (
        "--model",
        "m",
        "--api-key",
        REDACTED,
        f"--tool-auth-token={REDACTED}",
        "--max-tokens",
        "64",
    )


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--api-key", "V"], ("--api-key", REDACTED)),
        (["--api-key=V"], (f"--api-key={REDACTED}",)),
        (["--api-key="], (f"--api-key={REDACTED}",)),
        (["--hf-token", "V", "next"], ("--hf-token", REDACTED, "next")),
        (["--password=V"], (f"--password={REDACTED}",)),
        (["--passwd", "V"], ("--passwd", REDACTED)),
        (["--client-secret", "V"], ("--client-secret", REDACTED)),
        (["--credential-file", "V"], ("--credential-file", REDACTED)),
        (["--API-KEY", "V"], ("--API-KEY", REDACTED)),
        (["--api-key", "--looks-like-an-option"], ("--api-key", REDACTED)),
        (
            ["--api-key", "V1", "--auth-token", "V2"],
            ("--api-key", REDACTED, "--auth-token", REDACTED),
        ),
        (["--api-key"], ("--api-key",)),
    ],
)
def test_names_that_look_like_a_credential_have_their_value_masked(args, expected):
    assert mask_args_by_name(args) == expected


@pytest.mark.parametrize(
    "args",
    [
        ["serve", "--model", "m", "--port", "8000"],
        ["--max-tokens", "64"],
        ["--max-new-tokens=64"],
        ["--tokenizer", "org/model"],
        ["--public-key", "key.pem"],
        ["--special-token", "<|im_end|>"],
        ["-t", "V"],
        ["--", "token"],
        ["token", "key", "secret"],
        [],
    ],
)
def test_other_arguments_are_kept_as_typed(args):
    assert mask_args_by_name(args) == tuple(args)
