"""The limited YAML loader: one size, nesting and expansion rule for every reader."""

from __future__ import annotations

import pytest
import yaml

from soup_cli.utils import yaml_limits
from soup_cli.utils.yaml_limits import (
    MAX_CONFIG_BYTES,
    MAX_YAML_NESTING,
    YamlLimitError,
    check_yaml_expanded_size,
    expanded_node_count,
    load_yaml_file_limited,
    load_yaml_limited,
)

# ---------------------------------------------------------------------------
# Documents named by the limit they sit under or cross
# ---------------------------------------------------------------------------


def _flow_sequences(depth: int) -> str:
    return "[" * depth + "]" * depth


def _flow_mappings(depth: int) -> str:
    return "{a: " * depth + "1" + "}" * depth


def _block_sequences(depth: int) -> str:
    return "- " * depth + "x\n"


def _block_mappings(depth: int) -> str:
    lines = [" " * level + "k:" for level in range(depth)]
    return "\n".join(lines) + " v\n"


def _alternating(depth: int) -> str:
    """Mappings and sequences in turn: every collection counts as a level."""
    opened = "".join("[" if level % 2 else "{a: " for level in range(depth))
    closed = "".join("]" if level % 2 else "}" for level in reversed(range(depth)))
    return opened + ("" if depth % 2 == 0 else "1") + closed


_NESTED_SPELLINGS = [
    _flow_sequences,
    _flow_mappings,
    _block_sequences,
    _block_mappings,
    _alternating,
]


def _alias_chain(lists: int) -> str:
    """``lists`` single-item lists, each holding the one before it by alias.

    The text never nests more than two levels, but the loaded object nests
    ``lists + 1``: the root mapping plus every list in the chain.
    """
    doc = "l0: &a0 [x]\n"
    for index in range(1, lists):
        doc += f"l{index}: &a{index} [*a{index - 1}]\n"
    return doc


def _alias_fan_out(levels: int) -> str:
    """Ten references per level: the node count grows tenfold with each one."""
    doc = "l0: &a0 [" + ", ".join(["x"] * 10) + "]\n"
    for index in range(1, levels + 1):
        doc += f"l{index}: &a{index} [" + ", ".join([f"*a{index - 1}"] * 10) + "]\n"
    return doc


# ---------------------------------------------------------------------------
# Valid documents load exactly as yaml.safe_load loads them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# only a comment\n",
        "a: 1\n",
        "- a\n- b\n",
        "plain scalar",
        "a: 2e-5\nb: 1e3\nc: 0x10\nd: 1_000\ne: yes\nf: ~\ng: 2026-01-01\n",
        "base: &b {r: 8, alpha: 16}\none:\n  <<: *b\n  r: 4\ntwo: *b\n",
        "text: |\n  line one\n  line two\n\nfolded: >-\n  a\n  b\n",
        'q: "a\\tb"\ns: \'it\'\'s\'\n',
        "﻿a: 1\n",
        "a: [1, [2, [3, {b: [4]}]]]\n",
        "name: café ☃\n",
    ],
)
def test_valid_document_loads_as_safe_load_does(text):
    assert load_yaml_limited(text, "doc", max_bytes=4096) == yaml.safe_load(text)


def test_bytes_are_accepted_like_text():
    assert load_yaml_limited(b"a: [1, 2]\n", "doc", max_bytes=64) == {"a": [1, 2]}


def test_anything_but_text_or_bytes_is_a_type_error():
    with pytest.raises(TypeError, match="doc must be str or bytes, got list"):
        load_yaml_limited(["a: 1"], "doc", max_bytes=64)


def test_a_syntax_error_is_still_a_yaml_error():
    with pytest.raises(yaml.YAMLError):
        load_yaml_limited("a: [1, 2\nb: }", "doc", max_bytes=64)


def test_every_refusal_is_a_value_error():
    assert issubclass(YamlLimitError, ValueError)


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


def test_text_exactly_at_the_size_limit_loads():
    text = "a: " + "x" * 61
    assert len(text.encode()) == 64
    assert load_yaml_limited(text, "doc", max_bytes=64) == {"a": "x" * 61}


def test_text_one_byte_over_the_size_limit_is_refused():
    text = "a: " + "x" * 62
    assert len(text.encode()) == 65
    with pytest.raises(YamlLimitError, match=r"^doc is larger than 64 bytes; refusing$"):
        load_yaml_limited(text, "doc", max_bytes=64)


def test_size_is_counted_in_bytes_not_characters():
    text = "a: " + "é" * 31  # 34 characters, 65 bytes
    assert len(text) == 34
    assert len(text.encode()) == 65
    with pytest.raises(YamlLimitError, match="doc is larger than 64 bytes"):
        load_yaml_limited(text, "doc", max_bytes=64)
    assert load_yaml_limited(text, "doc", max_bytes=65) == {"a": "é" * 31}


def test_size_is_checked_before_anything_is_parsed(monkeypatch):
    def _no_parse(*args, **kwargs):
        raise AssertionError("an over-size document must not reach the parser")

    monkeypatch.setattr(yaml_limits, "_parse", _no_parse)
    with pytest.raises(YamlLimitError, match="is larger than 8 bytes"):
        load_yaml_limited("a: [1, 2, 3, 4]", "doc", max_bytes=8)


def test_the_config_size_limit_is_one_mebibyte():
    assert MAX_CONFIG_BYTES == 1024 * 1024


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------


def test_the_nesting_limit_is_one_hundred_levels():
    assert MAX_YAML_NESTING == 100


@pytest.mark.parametrize("spelling", _NESTED_SPELLINGS)
def test_document_nested_exactly_to_the_limit_loads(spelling):
    text = spelling(MAX_YAML_NESTING)
    assert load_yaml_limited(text, "doc", max_bytes=10**6) == yaml.safe_load(text)


@pytest.mark.parametrize("spelling", _NESTED_SPELLINGS)
def test_document_nested_one_level_past_the_limit_is_refused(spelling):
    text = spelling(MAX_YAML_NESTING + 1)
    with pytest.raises(
        YamlLimitError,
        match=r"^doc nests collections more than 100 levels deep; refusing$",
    ):
        load_yaml_limited(text, "doc", max_bytes=10**6)


def test_sibling_collections_do_not_add_up_to_a_depth():
    text = "[" + ", ".join(["[[]]"] * 500) + "]"
    assert load_yaml_limited(text, "doc", max_bytes=8192) == [[[]]] * 500


@pytest.mark.parametrize("spelling", [_flow_sequences, _block_sequences, _flow_mappings])
def test_nesting_refusal_does_not_depend_on_the_interpreter_stack(spelling):
    """Far past where the composer's own recursion would give out."""
    text = spelling(20_000)
    with pytest.raises(YamlLimitError, match="doc nests collections more than 100 levels"):
        load_yaml_limited(text, "doc", max_bytes=1_000_000)


def test_parsing_stops_at_the_limit_rather_than_after_the_document(monkeypatch):
    """The refusal comes from the event stream, before any object is built."""

    def _no_check(*args, **kwargs):
        raise AssertionError("an over-nested document must not be built first")

    monkeypatch.setattr(yaml_limits, "check_yaml_expanded_size", _no_check)
    with pytest.raises(YamlLimitError, match="nests collections more than 100 levels"):
        load_yaml_limited(_flow_sequences(MAX_YAML_NESTING + 1), "doc", max_bytes=4096)


def test_nesting_reached_through_aliases_at_the_limit_loads():
    text = _alias_chain(MAX_YAML_NESTING - 1)  # root mapping + 99 lists
    data = load_yaml_limited(text, "doc", max_bytes=10**6)
    assert data == yaml.safe_load(text)


def test_nesting_reached_through_aliases_past_the_limit_is_refused():
    text = _alias_chain(MAX_YAML_NESTING)  # root mapping + 100 lists
    with pytest.raises(YamlLimitError, match="doc nests collections more than 100 levels"):
        load_yaml_limited(text, "doc", max_bytes=10**6)


def test_stack_exhaustion_while_parsing_is_a_refusal_not_a_recursion_error(monkeypatch):
    """With the event limit out of the way the composer's own recursion gives out."""
    monkeypatch.setattr(yaml_limits, "MAX_YAML_NESTING", 10**9)
    with pytest.raises(
        YamlLimitError,
        match=r"^doc is nested too deeply to load \(limit: 1000000000 levels\); refusing$",
    ):
        load_yaml_limited(_flow_sequences(5000), "doc", max_bytes=10**6)


def test_document_rule_refuses_an_object_nested_past_the_limit():
    data: list = []
    for _ in range(MAX_YAML_NESTING):
        data = [data]
    with pytest.raises(YamlLimitError, match="doc nests collections more than 100 levels"):
        check_yaml_expanded_size(data, "doc", source_bytes=10**6)


def test_document_rule_accepts_an_object_nested_to_the_limit():
    data: list = []
    for _ in range(MAX_YAML_NESTING - 1):
        data = [data]
    check_yaml_expanded_size(data, "doc", source_bytes=10**6)


# ---------------------------------------------------------------------------
# The counter sees depth that only exists through shared references
# ---------------------------------------------------------------------------


def test_counter_treats_a_shared_chain_past_its_depth_as_over_the_limit():
    data = yaml.safe_load(_alias_chain(1500))
    assert expanded_node_count(data, limit=10**9) == 10**9 + 1


def test_counter_counts_a_shared_chain_within_its_depth_exactly():
    data = yaml.safe_load(_alias_chain(500))
    # Root: 1 + 500 keys. List k holds list k-1, so it expands to k + 2 nodes
    # (itself, the k lists inside it, and the one scalar at the bottom).
    assert expanded_node_count(data, limit=10**9) == 1 + 500 + sum(
        index + 2 for index in range(500)
    )


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------


def test_modest_alias_reuse_loads():
    text = _alias_fan_out(2)
    assert load_yaml_limited(text, "doc", max_bytes=10**6) == yaml.safe_load(text)


def test_alias_fan_out_past_the_node_allowance_is_refused():
    text = _alias_fan_out(9)
    allowance = len(text.encode()) + yaml_limits.ALIAS_EXPANSION_SLACK
    with pytest.raises(
        YamlLimitError,
        match=(
            rf"^doc expands to more than {allowance} nodes from "
            rf"{len(text.encode())} bytes of YAML; refusing$"
        ),
    ):
        load_yaml_limited(text, "doc", max_bytes=10**6)


def test_self_referencing_alias_is_refused():
    with pytest.raises(YamlLimitError, match="doc expands to more than"):
        load_yaml_limited("a: &x [1, *x]\n", "doc", max_bytes=64)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_file_exactly_at_the_size_limit_loads(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_bytes(b"a: " + b"x" * 61)
    assert load_yaml_file_limited(path, "f.yaml", max_bytes=64) == {"a": "x" * 61}


def test_file_one_byte_over_the_size_limit_is_refused(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_bytes(b"a: " + b"x" * 62)
    with pytest.raises(YamlLimitError, match=r"^f.yaml is larger than 64 bytes; refusing$"):
        load_yaml_file_limited(path, "f.yaml", max_bytes=64)


def test_file_read_never_asks_for_more_than_one_byte_past_the_limit(tmp_path, monkeypatch):
    path = tmp_path / "f.yaml"
    path.write_bytes(b"x" * 10_000)
    requested: list = []
    real_open = open

    class _Recording:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self._handle.close()

        def read(self, size=-1):
            requested.append(size)
            return self._handle.read(size)

    def _open(file, mode="r", *args, **kwargs):
        assert mode == "rb"
        return _Recording(real_open(file, mode, *args, **kwargs))

    monkeypatch.setattr(yaml_limits, "open", _open, raising=False)
    with pytest.raises(YamlLimitError, match="is larger than 64 bytes"):
        load_yaml_file_limited(path, "f.yaml", max_bytes=64)
    assert requested == [65]


def test_file_size_limit_counts_the_bytes_on_disk(tmp_path):
    """CRLF line ends count as two bytes even though they load as one."""
    path = tmp_path / "f.yaml"
    path.write_bytes(b"a: 1\r\n" * 11)  # 66 bytes, 55 once newlines are translated
    with pytest.raises(YamlLimitError, match="is larger than 64 bytes"):
        load_yaml_file_limited(path, "f.yaml", max_bytes=64)


@pytest.mark.parametrize(
    "content",
    [
        b"a: 1\nb: [x, y]\n",
        b"a: 1\r\nb: [x, y]\r\n",
        b"\xef\xbb\xbfa: 1\nb: [x, y]\n",
        b"\xef\xbb\xbfa: 1\r\nb: [x, y]\r\n",
        b"a: 1\rb: [x, y]\r",
    ],
    ids=["lf", "crlf", "bom", "bom-crlf", "cr"],
)
def test_file_line_ends_and_bom_load_the_same_document(tmp_path, content):
    path = tmp_path / "f.yaml"
    path.write_bytes(content)
    assert load_yaml_file_limited(path, "f.yaml", max_bytes=4096) == {"a": 1, "b": ["x", "y"]}


@pytest.mark.parametrize(
    "content",
    [
        b"t: |\r\n  one\r\n  two\r\n\r\nf: >\r\n  a\r\n  b\r\n",
        b'q: "one\r\n  two"\r\n',
        b"p: one\r\n  two\r\n",
        "name: café ☃\r\n".encode(),
    ],
    ids=["block-scalars", "double-quoted", "plain-multiline", "non-ascii"],
)
def test_file_is_decoded_exactly_as_read_text_decodes_it(tmp_path, content):
    path = tmp_path / "f.yaml"
    path.write_bytes(content)
    expected = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert load_yaml_file_limited(path, "f.yaml", max_bytes=4096) == expected


def test_file_that_is_not_utf8_is_still_a_decode_error(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_bytes(b"a: \xff\xfe\n")
    with pytest.raises(UnicodeDecodeError):
        load_yaml_file_limited(path, "f.yaml", max_bytes=4096)


def test_missing_file_is_still_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml_file_limited(tmp_path / "absent.yaml", "absent.yaml", max_bytes=4096)


def test_file_over_the_nesting_limit_is_refused_by_name(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_text(_block_mappings(MAX_YAML_NESTING + 1), encoding="utf-8")
    with pytest.raises(YamlLimitError, match="f.yaml nests collections more than 100 levels"):
        load_yaml_file_limited(path, "f.yaml", max_bytes=10**6)


def test_file_over_the_node_allowance_is_refused_by_name(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_text(_alias_fan_out(9), encoding="utf-8")
    with pytest.raises(YamlLimitError, match="f.yaml expands to more than"):
        load_yaml_file_limited(path, "f.yaml", max_bytes=10**6)
