"""Smoke test for the example module(s).

WHY a smoke test:
    A smoke test does NOT check that the analysis is correct.
    It checks the cheapest, most important thing: does the file load at all?
    A typo, a bad import, or a syntax error stops a module from importing,
    and this test catches that, before anything else can run.


WHY it does not name the module:
    The brittle way is `from datafun import app`.
    That breaks the moment the module gets renamed.
    This test DISCOVERS the example module by
    naming convention since my example scripts end in `case`
    (app, app_co2, app_penguins, ...).

In practice, we provide unit tests to ensure each function is working
as we intended.

Just knowing about testing is a valuable skill.
Feel free to experiment with adding additional tests to this file,
or adding additional files.
Nothing regarding testing is required.

Run from the project root:

    uv run pytest
"""

import importlib
import pkgutil

import datafun


def find_example_modules() -> list[str]:
    """Find example modules in the datafun package by naming convention.

    Convention: example entry-point scripts are named so the module name
    ends in `case` (for example: app, app_co2, app_penguins).

    Returns:
        list[str]: Fully qualified module names, e.g. ["datafun.app_co2"].
    """
    found: list[str] = []

    # iter_modules walks the package folder and yields one entry per module.
    for module_info in pkgutil.iter_modules(datafun.__path__):
        if module_info.name.endswith("case"):
            found.append(f"datafun.{module_info.name}")

    return found


def test_package_imports() -> None:
    """The datafun package itself imports without error.

    This is the simplest possible check and it always runs first.
    """
    assert datafun is not None


def test_example_modules_expose_main() -> None:
    """Each example module imports cleanly and exposes a callable main().

    Importing a module runs its top-level code (imports, logger setup) but
    NOT main(), because main() is behind the `if __name__ == "__main__"`
    guard. So this is safe: it does not load data or open charts.

    If a repo has no example module yet, there is nothing to import and the
    test still passes - the workflow stays green.
    """
    module_names: list[str] = find_example_modules()

    for name in module_names:
        module = importlib.import_module(name)
        assert callable(module.main), f"{name} has no callable main()"
