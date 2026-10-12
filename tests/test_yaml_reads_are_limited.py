"""Every YAML document read under ``src/soup_cli`` goes through the limited loader.

Two halves. The source scan keeps ``utils/yaml_limits.py`` the only module that
calls a PyYAML loading function, so a new reader cannot skip the limits by
accident. The reader table then drives each existing reader with a document
past the node allowance and one past the nesting limit, and checks that the
refusal comes out in that reader's own contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from soup_cli.utils.yaml_limits import MAX_YAML_NESTING, YamlLimitError

SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"

#: The one module allowed to call PyYAML's loading functions.
ALLOWED = {"utils/yaml_limits.py"}

#: PyYAML entry points that turn text into objects or nodes.
_LOADING_NAMES = frozenset(
    {
        "load",
        "load_all",
        "safe_load",
        "safe_load_all",
        "full_load",
        "full_load_all",
        "unsafe_load",
        "unsafe_load_all",
        "compose",
        "compose_all",
    }
)


def _yaml_loading_uses(source: str) -> list[tuple[int, str]]:
    """``(line, spelling)`` for every use of a PyYAML loading function.

    Sees ``yaml.safe_load(...)``, an aliased module (``import yaml as y``), a
    name imported from the package (``from yaml import safe_load``), and a bare
    reference that is called later (``loader = yaml.safe_load``).
    """
    tree = ast.parse(source)
    modules: set[str] = set()
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "yaml":
                    modules.add(alias.asname or "yaml")
        elif isinstance(node, ast.ImportFrom) and node.module == "yaml":
            for alias in node.names:
                if alias.name in _LOADING_NAMES:
                    imported[alias.asname or alias.name] = alias.name

    uses: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _LOADING_NAMES
            and isinstance(node.value, ast.Name)
            and node.value.id in modules
        ):
            uses.append((node.lineno, f"{node.value.id}.{node.attr}"))
        elif isinstance(node, ast.ImportFrom) and node.module == "yaml":
            uses.extend(
                (node.lineno, f"from yaml import {alias.name}")
                for alias in node.names
                if alias.name in _LOADING_NAMES
            )
    return sorted(uses)


@pytest.mark.parametrize(
    ("source", "spelling"),
    [
        ("import yaml\nx = yaml.safe_load(t)\n", "yaml.safe_load"),
        ("import yaml\nx = yaml.load(t, Loader=yaml.SafeLoader)\n", "yaml.load"),
        ("import yaml as y\nx = y.safe_load(t)\n", "y.safe_load"),
        ("import yaml\nfor d in yaml.safe_load_all(t):\n    pass\n", "yaml.safe_load_all"),
        ("import yaml\nx = yaml.full_load(t)\n", "yaml.full_load"),
        ("import yaml\nx = yaml.unsafe_load(t)\n", "yaml.unsafe_load"),
        ("import yaml\nx = yaml.compose(t)\n", "yaml.compose"),
        ("def f(t):\n    import yaml\n    return yaml.safe_load(t)\n", "yaml.safe_load"),
        ("import yaml\nloader = yaml.safe_load\nx = loader(t)\n", "yaml.safe_load"),
        ("from yaml import safe_load\nx = safe_load(t)\n", "from yaml import safe_load"),
        ("from yaml import safe_load as sl\nx = sl(t)\n", "from yaml import safe_load"),
    ],
)
def test_scan_sees_each_spelling_of_a_yaml_load(source, spelling):
    assert [found for _, found in _yaml_loading_uses(source)] == [spelling]


@pytest.mark.parametrize(
    "source",
    [
        "import yaml\ntext = yaml.safe_dump(data)\n",
        "import yaml\ntry:\n    pass\nexcept yaml.YAMLError:\n    pass\n",
        "from yaml import YAMLError, safe_dump\n",
        "import json\nx = json.load(fh)\n",
        "x = config.load(t)\n",
        '"""Uses ``yaml.safe_load`` semantics."""\n',
    ],
)
def test_scan_ignores_what_is_not_a_yaml_load(source):
    assert _yaml_loading_uses(source) == []


def test_only_the_limited_loader_calls_pyyaml_loading_functions():
    files = sorted(SRC.rglob("*.py"))
    assert len(files) > 300, "the scan did not find the package"
    offenders = []
    for path in files:
        rel = path.relative_to(SRC).as_posix()
        uses = _yaml_loading_uses(path.read_text(encoding="utf-8"))
        if uses and rel not in ALLOWED:
            offenders.extend(f"{rel}:{line} {spelling}" for line, spelling in uses)
    assert offenders == [], (
        "read YAML with soup_cli.utils.yaml_limits.load_yaml_limited / "
        "load_yaml_file_limited instead:\n" + "\n".join(offenders)
    )


def test_the_limited_loader_is_where_the_scan_expects_it():
    for rel in ALLOWED:
        assert _yaml_loading_uses((SRC / rel).read_text(encoding="utf-8")), rel


# ---------------------------------------------------------------------------
# Each reader, driven with a document past a limit
# ---------------------------------------------------------------------------


def _alias_fan_out(levels: int = 9) -> str:
    """Lists that each reference the previous one ten times."""
    text = "l0: &a0 [" + ", ".join(["x"] * 10) + "]\n"
    for index in range(1, levels + 1):
        text += f"l{index}: &a{index} [" + ", ".join([f"*a{index - 1}"] * 10) + "]\n"
    return text


def _nested_past_the_limit() -> str:
    inner = MAX_YAML_NESTING  # plus the root mapping: one level past the limit
    return "deep: " + "[" * inner + "]" * inner + "\n"


_OVER_LIMIT = {
    "node-allowance": (_alias_fan_out, "expands to more than"),
    "nesting": (
        _nested_past_the_limit,
        f"nests collections more than {MAX_YAML_NESTING} levels deep",
    ),
}


def _write(name: str, text: str) -> str:
    Path(name).write_text(text, encoding="utf-8")
    return name


def _plan(text):
    from soup_cli.commands.plan import _load_yaml_config

    return _load_yaml_config(_write("soup.yaml", text))


def _gate_suite(text):
    from soup_cli.eval.gate import load_suite

    return load_suite(_write("gate.yaml", text))


def _rubric(text):
    from soup_cli.eval.judge import load_rubric

    return load_rubric(Path(_write("rubric.yaml", text)))


def _axolotl(text):
    from soup_cli.migrate.axolotl import migrate_axolotl

    return migrate_axolotl(Path(_write("ax.yaml", text)))


def _llamafactory(text):
    from soup_cli.migrate.llamafactory import migrate_llamafactory

    return migrate_llamafactory(Path(_write("lf.yaml", text)))


def _agent_spec(text):
    from soup_cli.utils.agent_forge import load_spec_file

    return load_spec_file(_write("spec.yaml", text))


def _build(text):
    from soup_cli.utils.build_dag import parse_build_yaml

    return parse_build_yaml(text)


def _expectations(text):
    from soup_cli.utils.expectations import parse_suite_yaml

    return parse_suite_yaml(text)


def _recipe_dag(text):
    from soup_cli.utils.recipe_dag import parse_recipe_yaml

    return parse_recipe_yaml(text)


def _checklist(text):
    from soup_cli.utils.checklist_dsl import load_checklist_spec

    return load_checklist_spec(_write("checklist.yaml", text))


def _mix_recipe(text):
    from soup_cli.utils.data_mix import load_mix_recipe

    return load_mix_recipe(_write("recipe.yaml", text))


def _mix_proxy(text):
    from soup_cli.utils.mix_proxy import _render_overlay_yaml

    return _render_overlay_yaml(text, ("a.jsonl",), (1.0,))


def _ra_dit(text):
    from soup_cli.utils.ra_dit_run import _load_yaml_config

    return _load_yaml_config(_write("ra.yaml", text))


def _quant_config(text):
    from soup_cli.utils.save_formats import load_quant_config

    return load_quant_config(_write("quant.yaml", text))


def _sweep(text):
    from soup_cli.utils.sweep_config import parse_sweep_yaml

    return parse_sweep_yaml(text)


#: reader id -> (driver, how the refusal names the document)
_RAISING_READERS = {
    "commands/plan.py": (_plan, "config 'soup.yaml'"),
    "eval/gate.py": (_gate_suite, "eval-gate suite 'gate.yaml'"),
    "eval/judge.py": (_rubric, "Rubric file rubric.yaml"),
    "migrate/axolotl.py": (_axolotl, "Config file ax.yaml"),
    "migrate/llamafactory.py": (_llamafactory, "Config file lf.yaml"),
    "utils/agent_forge.py": (_agent_spec, "spec file"),
    "utils/build_dag.py": (_build, "build text"),
    "utils/expectations.py": (_expectations, "suite text"),
    "utils/recipe_dag.py": (_recipe_dag, "recipe text"),
    "utils/checklist_dsl.py": (_checklist, "spec file"),
    "utils/data_mix.py": (_mix_recipe, "recipe"),
    "utils/mix_proxy.py": (_mix_proxy, "base YAML"),
    "utils/ra_dit_run.py": (_ra_dit, "config YAML"),
    "utils/save_formats.py": (_quant_config, "quant_config"),
    "utils/sweep_config.py": (_sweep, "sweep yaml"),
}


@pytest.mark.parametrize("limit", sorted(_OVER_LIMIT))
@pytest.mark.parametrize("reader", sorted(_RAISING_READERS))
def test_reader_refuses_a_document_past_the_limit(tmp_path, monkeypatch, reader, limit):
    monkeypatch.chdir(tmp_path)
    drive, what = _RAISING_READERS[reader]
    make_text, message = _OVER_LIMIT[limit]
    with pytest.raises(YamlLimitError) as raised:
        drive(make_text())
    assert str(raised.value).startswith(f"{what} {message}"), str(raised.value)


# Readers whose contract is a fallback value rather than an exception: each is
# shown returning the real value first, so the fallback is the refusal at work.


@pytest.mark.parametrize("limit", sorted(_OVER_LIMIT))
def test_deploy_template_detection_falls_back_past_the_limit(tmp_path, monkeypatch, limit):
    from soup_cli.commands.deploy import _auto_detect_template

    monkeypatch.chdir(tmp_path)
    base = "data:\n  format: chatml\n"
    _write("soup.yaml", base)
    assert _auto_detect_template() is not None
    _write("soup.yaml", base + _OVER_LIMIT[limit][0]())
    assert _auto_detect_template() is None


@pytest.mark.parametrize("limit", sorted(_OVER_LIMIT))
def test_push_training_config_falls_back_past_the_limit(tmp_path, monkeypatch, limit):
    from soup_cli.commands.push import _load_training_config

    monkeypatch.chdir(tmp_path)
    base = "base: some/model\n"
    _write("training_config.yaml", base)
    assert _load_training_config(tmp_path) == {"base": "some/model"}
    _write("training_config.yaml", base + _OVER_LIMIT[limit][0]())
    assert _load_training_config(tmp_path) == {}


@pytest.mark.parametrize("limit", sorted(_OVER_LIMIT))
def test_license_frontmatter_falls_back_past_the_limit(tmp_path, monkeypatch, limit):
    from soup_cli.utils.license_matrix import _license_from_frontmatter

    monkeypatch.chdir(tmp_path)
    readme = str(tmp_path / "README.md")
    _write("README.md", "---\nlicense: mit\n---\nbody\n")
    assert _license_from_frontmatter(readme) == "mit"
    _write("README.md", "---\nlicense: mit\n" + _OVER_LIMIT[limit][0]() + "---\nbody\n")
    assert _license_from_frontmatter(readme) is None


def test_every_routed_reader_has_a_case():
    """The table above and the modules that import the limited loader agree."""
    importers = set()
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        if "load_yaml_limited" in text or "load_yaml_file_limited" in text:
            importers.add(rel)
    covered_elsewhere = {
        "config/loader.py",  # tests/test_config_loader_limits.py
        "cans/unpack.py",  # tests/test_yaml_expanded_limit.py
        "commands/deploy.py",
        "commands/push.py",
        "utils/license_matrix.py",
    }
    assert importers == set(_RAISING_READERS) | covered_elsewhere
