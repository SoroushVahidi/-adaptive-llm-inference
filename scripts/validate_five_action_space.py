#!/usr/bin/env python3
"""Validate the canonical 5-action routing space.

This script checks whether each of the five routing actions is:

  1. Registered in configs/action_space_catalog.yaml with status=implemented.
  2. Registered in configs/five_action_space.yaml.
  3. Listed in MULTI_ACTION_ORACLE_STRATEGIES (oracle_subset_eval.py).
  4. Listed in MULTI_ACTION_ORDER (multi_action_routing.py).
  5. Has a cost-proxy entry in STRATEGY_COST_PROXY.
  6. Has a callable runner in the oracle runner registry.
  7. Has per-query outcome data columns in each of the four main routing CSVs.

Usage
-----
    # From repo root:
    python scripts/validate_five_action_space.py

    # Verbose mode (shows all feature columns too):
    python scripts/validate_five_action_space.py --verbose

Exit codes
----------
  0 — all checks pass (or all failures are expected/documented blockers)
  1 — unexpected failure found (missing implementation, missing runner, etc.)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Canonical 5-action definition (mirrors configs/five_action_space.yaml)
# ---------------------------------------------------------------------------

FIVE_ACTIONS: list[dict] = [
    {
        "id": "A0",
        "name": "reasoning_greedy",
        "cost_proxy": 1,
        "outcome_col": "reasoning_correct",
        "data_expected": True,
        "in_multi_action_order": True,
    },
    {
        "id": "A1",
        "name": "direct_plus_revise",
        "cost_proxy": 2,
        "outcome_col": "revise_correct",
        "data_expected": True,
        "in_multi_action_order": True,
    },
    {
        "id": "A2",
        "name": "reasoning_then_revise",
        "cost_proxy": 2,
        "outcome_col": "reasoning_then_revise__correct",
        "data_expected": False,  # Blocker B1: not yet generated
        "in_multi_action_order": True,
    },
    {
        "id": "A3",
        "name": "self_consistency_3",
        "cost_proxy": 3,
        "outcome_col": "self_consistency_3__correct",
        "data_expected": False,  # Blocker B1: not yet generated
        "in_multi_action_order": True,
    },
    {
        "id": "A4",
        "name": "direct_plus_critique_plus_final",
        "cost_proxy": 3,
        "outcome_col": "direct_plus_critique_plus_final__correct",
        "data_expected": False,  # Blocker B1: not yet generated
        "in_multi_action_order": True,
    },
]

# Four main regimes used across all router experiments
ROUTING_CSVS: dict[str, str] = {
    "gsm8k_random_100": "data/real_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_100": "data/real_hard_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_b2": "data/real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    "math500_100": "data/real_math500_routing_dataset_enriched.csv",
}

# ---------------------------------------------------------------------------
# Colour helpers (degrade gracefully if no TTY)
# ---------------------------------------------------------------------------

_IS_TTY = sys.stdout.isatty()


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _IS_TTY else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _IS_TTY else s


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if _IS_TTY else s


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _IS_TTY else s


PASS = _green("PASS")
FAIL = _red("FAIL")
WARN = _yellow("WARN")

# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


def check_action_catalog(action_name: str) -> tuple[bool, str]:
    """Check that action is in configs/action_space_catalog.yaml with status=implemented."""
    try:
        import yaml
    except ImportError:
        return False, "pyyaml not installed; run: pip install pyyaml"

    catalog_path = _REPO_ROOT / "configs" / "action_space_catalog.yaml"
    if not catalog_path.exists():
        return False, f"Action catalog not found: {catalog_path}"

    with catalog_path.open() as fh:
        catalog = yaml.safe_load(fh)

    strategies = catalog.get("curated_strategies", [])
    for s in strategies:
        if s.get("name") == action_name:
            status = s.get("status", "")
            if status == "implemented":
                return True, f"status={status}"
            return False, f"status={status!r} (expected 'implemented')"
    return False, f"action '{action_name}' not found in catalog"


def check_five_action_yaml(action_name: str) -> tuple[bool, str]:
    """Check that action is registered in configs/five_action_space.yaml."""
    try:
        import yaml
    except ImportError:
        return False, "pyyaml not installed"

    yaml_path = _REPO_ROOT / "configs" / "five_action_space.yaml"
    if not yaml_path.exists():
        return False, f"five_action_space.yaml not found: {yaml_path}"

    with yaml_path.open() as fh:
        doc = yaml.safe_load(fh)

    actions = doc.get("actions", [])
    for a in actions:
        if a.get("name") == action_name:
            return True, f"id={a.get('id')}"
    return False, f"action '{action_name}' not found in five_action_space.yaml"


def _read_source(rel_path: str) -> str:
    """Read a repo source file as text (no import needed)."""
    return (_REPO_ROOT / rel_path).read_text()


def _extract_string_list(source: str, var_name: str) -> list[str] | None:
    """
    Extract a list of quoted strings assigned to *var_name* from Python source.

    Handles both list and tuple literals.  Returns None if not found.
    """
    import re
    # Match `VAR_NAME: ... = (\n)* [( ... )]`
    pattern = rf"{re.escape(var_name)}\s*[^=]*=\s*[\(\[]([^\)\]]*?)[\)\]]"
    m = re.search(pattern, source, re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    return re.findall(r'"([^"]+)"|\'([^\']+)\'', body)


def check_multi_action_order(action_name: str) -> tuple[bool, str]:
    """Check that action is in MULTI_ACTION_ORDER (multi_action_routing.py).

    Reads source directly to avoid importing heavy optional dependencies.
    """
    src_path = "src/evaluation/multi_action_routing.py"
    try:
        source = _read_source(src_path)
    except FileNotFoundError:
        return False, f"source file not found: {src_path}"

    matches = _extract_string_list(source, "MULTI_ACTION_ORDER")
    if matches is None:
        return False, f"could not parse MULTI_ACTION_ORDER in {src_path}"
    names = [a or b for a, b in matches]
    if action_name in names:
        return True, f"index={names.index(action_name)}"
    return False, f"'{action_name}' not in MULTI_ACTION_ORDER {names}"


def check_multi_action_oracle_strategies(action_name: str) -> tuple[bool, str]:
    """Check that action is in MULTI_ACTION_ORACLE_STRATEGIES (oracle_subset_eval.py).

    Reads source directly to avoid importing heavy optional dependencies.
    """
    src_path = "src/evaluation/oracle_subset_eval.py"
    try:
        source = _read_source(src_path)
    except FileNotFoundError:
        return False, f"source file not found: {src_path}"

    matches = _extract_string_list(source, "MULTI_ACTION_ORACLE_STRATEGIES")
    if matches is None:
        return False, f"could not parse MULTI_ACTION_ORACLE_STRATEGIES in {src_path}"
    names = [a or b for a, b in matches]
    if action_name in names:
        return True, f"index={names.index(action_name)}"
    return False, f"'{action_name}' not in MULTI_ACTION_ORACLE_STRATEGIES {names}"


def check_cost_proxy(action_name: str, expected_cost: int) -> tuple[bool, str]:
    """Check that action has a cost-proxy entry in STRATEGY_COST_PROXY.

    Reads source directly to avoid importing heavy optional dependencies.
    """
    import re
    src_path = "src/evaluation/oracle_subset_eval.py"
    try:
        source = _read_source(src_path)
    except FileNotFoundError:
        return False, f"source file not found: {src_path}"

    # Match entries like: "action_name": 3,
    pattern = rf'["\']({re.escape(action_name)})["\']:\s*(\d+)'
    m = re.search(pattern, source)
    if not m:
        return False, f"'{action_name}' not found in STRATEGY_COST_PROXY"
    cost = int(m.group(2))
    if cost != expected_cost:
        return False, f"cost_proxy={cost}, expected {expected_cost}"
    return True, f"cost_proxy={cost}"


def check_runner(action_name: str) -> tuple[bool, str]:
    """Check that action has a non-None runner entry in _ORACLE_RUNNERS.

    Reads source directly to avoid importing heavy optional dependencies.
    Handles both inline dict entries and late-assignment patterns like
    `_ORACLE_RUNNERS["name"] = run_something`.
    """
    import re
    src_path = "src/evaluation/oracle_subset_eval.py"
    try:
        source = _read_source(src_path)
    except FileNotFoundError:
        return False, f"source file not found: {src_path}"

    # Pattern: late assignment  _ORACLE_RUNNERS["action_name"] = run_something
    late = re.search(
        rf'_ORACLE_RUNNERS\["({re.escape(action_name)})"\]\s*=\s*(\w+)',
        source,
    )
    if late:
        runner_ref = late.group(2)
        if runner_ref == "None":
            return False, f"runner for '{action_name}' is None"
        return True, f"runner={runner_ref} (late-assigned)"

    # Pattern: inline dict entry within _ORACLE_RUNNERS block
    # Extract the _ORACLE_RUNNERS = { ... } block first
    block_m = re.search(
        r"_ORACLE_RUNNERS\s*:\s*dict[^=]*=\s*\{([^}]+)\}", source, re.DOTALL
    )
    search_area = block_m.group(1) if block_m else source

    inline = re.search(
        rf'["\']({re.escape(action_name)})["\']:\s*(\w+)',
        search_area,
    )
    if inline:
        runner_ref = inline.group(2)
        if runner_ref == "None":
            return False, f"runner for '{action_name}' is None (no late assignment found)"
        return True, f"runner={runner_ref}"

    return False, f"'{action_name}' not found in _ORACLE_RUNNERS"


def _csv_columns(csv_path: Path) -> set[str]:
    """Return the set of column names from a CSV header row."""
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return set()
    return set(header)


def check_outcome_data(
    action_name: str,
    outcome_col: str,
    data_expected: bool,
) -> list[tuple[str, bool, str]]:
    """
    Check each routing CSV for the outcome column.

    Returns a list of (regime, ok, message) tuples.
    ``data_expected=False`` means the absence of the column is a known
    blocker (Blocker B1), not an unexpected failure.
    """
    results = []
    for regime, rel_path in ROUTING_CSVS.items():
        csv_path = _REPO_ROOT / rel_path
        if not csv_path.exists():
            results.append((regime, False, f"CSV not found: {rel_path}"))
            continue
        cols = _csv_columns(csv_path)
        if outcome_col in cols:
            results.append((regime, True, f"column '{outcome_col}' present"))
        else:
            if data_expected:
                results.append((
                    regime, False,
                    f"column '{outcome_col}' MISSING from {csv_path.name}",
                ))
            else:
                results.append((
                    regime, None,  # type: ignore[arg-type]
                    f"column '{outcome_col}' not yet generated (Blocker B1 — expected)",
                ))
    return results


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------


def run_validation(verbose: bool = False) -> int:
    """Run all checks; return exit code (0 = pass, 1 = unexpected failure)."""
    unexpected_failures: list[str] = []
    expected_blockers: list[str] = []

    print(_bold("\n=== 5-Action Routing Space Validation ===\n"))

    for action in FIVE_ACTIONS:
        name = action["name"]
        aid = action["id"]
        print(_bold(f"--- {aid}: {name} ---"))

        # 1. Action catalog
        ok, msg = check_action_catalog(name)
        label = PASS if ok else FAIL
        print(f"  [1] Action catalog (implemented):      {label}  {msg}")
        if not ok:
            unexpected_failures.append(f"{aid}/{name}: catalog check failed: {msg}")

        # 2. five_action_space.yaml
        ok, msg = check_five_action_yaml(name)
        label = PASS if ok else FAIL
        print(f"  [2] five_action_space.yaml registry:   {label}  {msg}")
        if not ok:
            unexpected_failures.append(f"{aid}/{name}: five_action_space.yaml check failed: {msg}")

        # 3. MULTI_ACTION_ORDER
        ok, msg = check_multi_action_order(name)
        label = PASS if ok else FAIL
        print(f"  [3] MULTI_ACTION_ORDER:                {label}  {msg}")
        if not ok:
            unexpected_failures.append(f"{aid}/{name}: MULTI_ACTION_ORDER check failed: {msg}")

        # 4. MULTI_ACTION_ORACLE_STRATEGIES
        ok, msg = check_multi_action_oracle_strategies(name)
        label = PASS if ok else FAIL
        print(f"  [4] MULTI_ACTION_ORACLE_STRATEGIES:    {label}  {msg}")
        if not ok:
            unexpected_failures.append(
                f"{aid}/{name}: MULTI_ACTION_ORACLE_STRATEGIES check failed: {msg}"
            )

        # 5. Cost proxy
        ok, msg = check_cost_proxy(name, action["cost_proxy"])
        label = PASS if ok else FAIL
        print(f"  [5] STRATEGY_COST_PROXY (={action['cost_proxy']}):        {label}  {msg}")
        if not ok:
            unexpected_failures.append(f"{aid}/{name}: cost proxy check failed: {msg}")

        # 6. Runner callable
        ok, msg = check_runner(name)
        label = PASS if ok else FAIL
        print(f"  [6] Oracle runner callable:            {label}  {msg}")
        if not ok:
            unexpected_failures.append(f"{aid}/{name}: runner check failed: {msg}")

        # 7. Outcome data in routing CSVs
        data_results = check_outcome_data(
            name, action["outcome_col"], action["data_expected"]
        )
        print(f"  [7] Outcome data ({action['outcome_col']}):")
        for regime, ok, msg in data_results:
            if ok is True:
                label = PASS
            elif ok is None:
                label = WARN  # expected blocker
                expected_blockers.append(
                    f"{aid}/{name}: {regime}: {msg}"
                )
            else:
                label = FAIL
                unexpected_failures.append(
                    f"{aid}/{name}: outcome data check failed for {regime}: {msg}"
                )
            print(f"       {label}  [{regime}] {msg}")

        print()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(_bold("=== Summary ===\n"))

    if expected_blockers:
        print(_yellow(f"Known / expected blockers ({len(expected_blockers)}):"))
        for b in expected_blockers:
            print(f"  {WARN} {b}")
        print()
        print("  These are Blocker B1 items (outcome data not yet generated).")
        print("  Run scripts/run_build_multi_action_dataset.py on all regimes to resolve.")
        print()

    if unexpected_failures:
        print(_red(f"Unexpected failures ({len(unexpected_failures)}):"))
        for f in unexpected_failures:
            print(f"  {FAIL} {f}")
        print()
        return 1
    else:
        print(_green("No unexpected failures."))
        if expected_blockers:
            print(_yellow("All remaining issues are documented Blocker B1 items (data generation)."))
        else:
            print(_green("All 5 actions are fully ready for 5-way router training."))
        print()
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the canonical 5-action routing space. "
            "Checks catalog status, runner availability, cost proxies, and "
            "whether per-action outcome data exists in the routing datasets."
        )
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional detail (feature columns, etc.).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    exit_code = run_validation(verbose=args.verbose)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
