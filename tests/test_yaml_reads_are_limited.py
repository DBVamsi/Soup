"""Every YAML document read under ``src/soup_cli`` goes through the limited loader.

Two halves. The source scan keeps ``utils/yaml_limits.py`` the only module that
can parse YAML: no other module may hold PyYAML except to write YAML or to
catch its error class, and each one that does is listed here with exactly the
attributes it uses. The reader table then drives each existing reader with a
document past the node allowance and one past the nesting limit, and checks
that the refusal comes out in that reader's own contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from soup_cli.utils.yaml_limits import MAX_YAML_NESTING, YamlLimitError

SRC = Path(__file__).resolve().parents[1] / "src" / "soup_cli"

#: The one module that parses YAML.
LOADER_MODULE = "utils/yaml_limits.py"

#: What another module may take from PyYAML: writing, and the error class.
_NOT_PARSING = frozenset({"safe_dump", "dump", "YAMLError"})

#: Every other module that imports PyYAML, with the attributes it uses. A new
#: entry needs one of ``_NOT_PARSING``; reading YAML goes through LOADER_MODULE.
YAML_USERS = {
    "autopilot/generate_config.py": ("safe_dump",),
    "cans/pack.py": ("safe_dump",),
    "commands/adapters.py": ("YAMLError",),
    "commands/deploy.py": ("YAMLError",),
    "commands/push.py": ("YAMLError",),
    "commands/ship.py": ("YAMLError",),
    "commands/spectrum.py": ("safe_dump",),
    "commands/sweep.py": ("YAMLError",),
    "migrate/common.py": ("dump",),
    "ui/app.py": ("dump",),
    "utils/adapter_branch.py": ("safe_dump",),
    "utils/build_dag.py": ("YAMLError",),
    "utils/checklist_dsl.py": ("YAMLError",),
    "utils/compile_tools.py": ("safe_dump",),
    "utils/expectations.py": ("YAMLError",),
    "utils/iterative_dpo.py": ("safe_dump",),
    "utils/local_rl.py": ("safe_dump",),
    "utils/loop_stages.py": ("safe_dump",),
    "utils/mix_proxy.py": ("safe_dump",),
    "utils/onboarding.py": ("safe_dump",),
    "utils/ra_dit_run.py": ("safe_dump",),
    "utils/recipe_dag.py": ("YAMLError",),
}

_YAML_PACKAGES = ("yaml", "ruamel")


def _is_yaml_package(name: object) -> bool:
    return isinstance(name, str) and name.split(".")[0] in _YAML_PACKAGES


def _yaml_uses(source: str) -> tuple[set[str], list[tuple[int, str]]]:
    """``(attributes, problems)`` for one module's use of a YAML package.

    ``attributes`` are the ``yaml.<name>`` accesses made through a plain
    ``import yaml`` (or ``import yaml as y``). ``problems`` is every other way
    of reaching the package, each of which could hide a parse: a from-import, a
    submodule import, ``ruamel``, a dynamic import by literal name, and any use
    of the module object that is not an attribute access (``y = yaml``,
    ``getattr(yaml, ...)``, passing it on).
    """
    tree = ast.parse(source)
    names: set[str] = set()
    problems: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "yaml":
                    names.add(alias.asname or "yaml")
                elif _is_yaml_package(alias.name):
                    problems.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and _is_yaml_package(node.module):
            problems.append((node.lineno, f"from {node.module} import ..."))
        elif isinstance(node, ast.Call):
            target = node.func
            called = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            literal = node.args[0].value if node.args and isinstance(
                node.args[0], ast.Constant
            ) else None
            if called in ("import_module", "__import__") and _is_yaml_package(literal):
                problems.append((node.lineno, f"{called}({literal!r})"))

    attributes: set[str] = set()
    through_attribute: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in names
        ):
            attributes.add(node.attr)
            through_attribute.add(id(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names and id(node) not in through_attribute:
            problems.append((node.lineno, f"bare use of {node.id}"))
    return attributes, sorted(problems)


@pytest.mark.parametrize(
    ("source", "attribute"),
    [
        ("import yaml\nx = yaml.safe_load(t)\n", "safe_load"),
        ("import yaml\nx = yaml.load(t, Loader=None)\n", "load"),
        ("import yaml as y\nx = y.safe_load(t)\n", "safe_load"),
        ("import yaml\nfor d in yaml.safe_load_all(t):\n    pass\n", "safe_load_all"),
        ("import yaml\nx = yaml.full_load(t)\n", "full_load"),
        ("import yaml\nx = yaml.unsafe_load(t)\n", "unsafe_load"),
        ("import yaml\nx = yaml.compose(t)\n", "compose"),
        ("import yaml\nfor e in yaml.parse(t):\n    pass\n", "parse"),
        ("import yaml\nfor e in yaml.scan(t):\n    pass\n", "scan"),
        ("import yaml\nx = yaml.SafeLoader(t).get_single_data()\n", "SafeLoader"),
        ("def f(t):\n    import yaml\n    return yaml.safe_load(t)\n", "safe_load"),
        ("import yaml\nloader = yaml.safe_load\nx = loader(t)\n", "safe_load"),
    ],
)
def test_scan_reports_each_parsing_attribute(source, attribute):
    attributes, problems = _yaml_uses(source)
    assert attributes == {attribute}
    assert not attributes <= _NOT_PARSING
    assert problems == []


@pytest.mark.parametrize(
    ("source", "problem"),
    [
        ("from yaml import safe_load\nx = safe_load(t)\n", "from yaml import ..."),
        ("from yaml import safe_load as sl\nx = sl(t)\n", "from yaml import ..."),
        ("from yaml import *\nx = safe_load(t)\n", "from yaml import ..."),
        ("from yaml import SafeLoader\n", "from yaml import ..."),
        ("from yaml.loader import SafeLoader\n", "from yaml.loader import ..."),
        ("import yaml.loader\n", "import yaml.loader"),
        ("import yaml\ny = yaml\nx = y.safe_load(t)\n", "bare use of yaml"),
        ('import yaml\nx = getattr(yaml, "safe_load")(t)\n', "bare use of yaml"),
        ("import yaml\nx = helper(yaml, t)\n", "bare use of yaml"),
        ('import importlib\ny = importlib.import_module("yaml")\n', "import_module('yaml')"),
        (
            'from importlib import import_module\ny = import_module("yaml")\n',
            "import_module('yaml')",
        ),
        ('y = __import__("yaml")\n', "__import__('yaml')"),
        ('y = __import__("yaml.loader")\n', "__import__('yaml.loader')"),
        ("import ruamel.yaml\n", "import ruamel.yaml"),
        ("from ruamel.yaml import YAML\n", "from ruamel.yaml import ..."),
        (
            'import importlib\ny = importlib.import_module("ruamel.yaml")\n',
            "import_module('ruamel.yaml')",
        ),
    ],
)
def test_scan_reports_each_other_way_of_reaching_a_yaml_package(source, problem):
    assert [found for _, found in _yaml_uses(source)[1]] == [problem]


@pytest.mark.parametrize(
    "source",
    [
        "import yaml\ntext = yaml.safe_dump(data)\n",
        "import yaml\ntext = yaml.dump(data)\n",
        "import yaml\ntry:\n    pass\nexcept yaml.YAMLError:\n    pass\n",
        "import yaml\ntry:\n    pass\nexcept (yaml.YAMLError, OSError):\n    pass\n",
    ],
)
def test_scan_accepts_writing_yaml_and_catching_its_error(source):
    attributes, problems = _yaml_uses(source)
    assert attributes <= _NOT_PARSING
    assert problems == []


@pytest.mark.parametrize(
    "source",
    [
        "import json\nx = json.load(fh)\n",
        "x = config.load(t)\n",
        'overlay_yaml = "a: 1"\nsize = len(overlay_yaml.encode())\n',
        '"""Uses ``yaml.safe_load`` semantics."""\n',
        'y = importlib.import_module("json")\n',
        "y = importlib.import_module(name)\n",
    ],
)
def test_scan_ignores_what_is_not_a_yaml_package(source):
    assert _yaml_uses(source) == (set(), [])


def test_no_module_but_the_limited_loader_can_parse_yaml():
    files = sorted(SRC.rglob("*.py"))
    assert len(files) > 300, "the scan did not find the package"
    found: dict[str, tuple[str, ...]] = {}
    offenders: list[str] = []
    for file in files:
        rel = file.relative_to(SRC).as_posix()
        if rel == LOADER_MODULE:
            continue
        attributes, problems = _yaml_uses(file.read_text(encoding="utf-8"))
        offenders.extend(f"{rel}:{line} {problem}" for line, problem in problems)
        offenders.extend(
            f"{rel}: yaml.{name}" for name in sorted(attributes - _NOT_PARSING)
        )
        if attributes:
            found[rel] = tuple(sorted(attributes))
    assert offenders == [], (
        "read YAML with soup_cli.utils.yaml_limits.load_yaml_limited / "
        "load_yaml_file_limited instead:\n" + "\n".join(offenders)
    )
    assert found == YAML_USERS, "YAML_USERS must list each module with the attributes it uses"


def test_the_limited_loader_is_where_the_scan_expects_it():
    attributes, _ = _yaml_uses((SRC / LOADER_MODULE).read_text(encoding="utf-8"))
    assert "SafeLoader" in attributes


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
        if rel == LOADER_MODULE:
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
