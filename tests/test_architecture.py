"""Enforces the import rules in AGENTS.md §2.1. If this fails, fix the import, not the test."""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "src" / "assistive_arm"
CORE = {"geometry", "vision", "robot_iface", "motion"}
SECTIONS = {"mouth_perception", "food_perception", "grasping", "feeding", "language", "sim", "pick_place"}
CONTRACT_MODULES = {"types", "schema", "config", "vocabulary", "states"}
CORE_ALLOWED = {
    "geometry": set(),
    "vision": {"geometry"},
    "robot_iface": {"geometry", "vision"},
    "motion": {"geometry", "robot_iface", "vision"},
}
# Documented exceptions (AGENTS.md); remove them as they are fixed.
EXCEPTIONS = {("robot_iface/viam.py", "motion.safety")}


def _imports(path: Path):
    """Yield dotted assistive_arm targets imported by ``path`` (skipping TYPE_CHECKING blocks)."""
    tree = ast.parse(path.read_text())
    pkg_parts = path.relative_to(PKG).parts[:-1]
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            skip.update(id(n) for n in ast.walk(node))
    for node in ast.walk(tree):
        if id(node) in skip or not isinstance(node, ast.ImportFrom):
            continue
        if node.level == 0:
            if node.module and node.module.startswith("assistive_arm."):
                yield node.module.split(".", 1)[1]
            continue
        base = list(pkg_parts[: len(pkg_parts) - (node.level - 1)])
        mod = ".".join(base + ([node.module] if node.module else []))
        if node.module is None:
            for a in node.names:
                yield f"{mod}.{a.name}" if mod else a.name
        else:
            yield mod


def violations():
    out = []
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG)
        top = rel.parts[0].removesuffix(".py")
        for target in _imports(path):
            t_top, _, rest = target.partition(".")
            if t_top == top or t_top not in CORE | SECTIONS | {"orchestration"}:
                continue
            if (rel.as_posix(), target) in EXCEPTIONS:
                continue
            if top in CORE:
                ok = t_top in CORE_ALLOWED[top]
            elif top in SECTIONS:
                ok = t_top in CORE or rest.split(".")[0] in CONTRACT_MODULES
            else:  # orchestration and top-level modules compose everything
                ok = True
            if not ok:
                out.append(f"{rel}: imports {target}")
    return out


def test_import_rules_from_agents_md():
    assert violations() == []


def test_every_package_has_a_plan_readme():
    for d in [p for p in PKG.iterdir() if p.is_dir() and not p.name.startswith("_")]:
        assert (d / "README.md").exists(), f"{d.name} has no README.md plan"
