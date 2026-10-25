"""Plain-toolset scenarios (read/write/edit/bash/glob/grep + subtask),
modelled on the dominant lane-worker archetypes observed in this team's
session history and PR patterns: red-first bugfix, feature + test + commit,
mechanical refactor, and delegated investigation.
"""

from __future__ import annotations

from aibench.agentic.scenarios.base import Scenario

BUGFIX_FILES = {
    "stats.py": '''"""Tiny stats helpers (has a bug)."""


def median(xs):
    if not xs:
        raise ValueError("median of empty")
    xs = sorted(xs)
    n = len(xs)
    if n % 2 == 1:
        return xs[n // 2]
    return xs[n // 2]
''',
    "test_stats.py": '''import unittest

from stats import median


class TestMedian(unittest.TestCase):
    def test_odd(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_even(self):
        self.assertEqual(median([1, 2, 3, 4]), 2.5)
        self.assertEqual(median([10, 0]), 5.0)

    def test_single(self):
        self.assertEqual(median([7]), 7)


if __name__ == "__main__":
    unittest.main()
''',
    "run_tests.sh": "#!/usr/bin/env bash\nset -e\npython3 -m unittest test_stats -v\n",
}


def _setup_bugfix(bundle) -> None:
    for path, content in BUGFIX_FILES.items():
        bundle.env.write_file(path, content)


async def _check_tests_pass(ctx) -> tuple[bool, str]:
    out, code = await ctx.env.bash("bash run_tests.sh")
    return code == 0, f"exit={code} tail={out[-200:]!r}"


async def _check_median_even(ctx) -> tuple[bool, str]:
    out, code = await ctx.env.bash(
        "python3 -c \"from stats import median; assert median([1,2,3,4])==2.5; assert median([10,0])==5.0\""
    )
    return code == 0, out[-200:]


def _check_no_weakening(ctx) -> tuple[bool, str]:
    test = ctx.env.read_file("test_stats.py", max_chars=100_000)
    ok = "2.5" in test and "sleep" not in test.lower()
    return ok, (
        "assertions intact" if ok
        else "test weakened (missing 2.5 assertion or sleep added)"
    )


def _check_fix_in_source(ctx) -> tuple[bool, str]:
    src = ctx.env.read_file("stats.py", max_chars=100_000)
    return "n // 2 - 1" in src, "even-path averages two middle values"


plain_bugfix_flake = Scenario(
    name="plain-bugfix-median",
    title="Red-first bugfix: median fails on even-length input",
    toolset="plain",
    task=(
        "The test suite in this workspace is failing. Diagnose the failure, "
        "fix the implementation properly (no weakening or editing of the test "
        "assertions, no sleeps), and finish only once `bash run_tests.sh` exits 0."
    ),
    setup=_setup_bugfix,
    checks=[
        ("tests_pass", _check_tests_pass),
        ("median_even_correct", _check_median_even),
        ("tests_not_weakened", _check_no_weakening),
        ("fix_in_source", _check_fix_in_source),
    ],
    max_turns=24,
    max_tool_calls=60,
    max_seconds=420,
)

# ---------------------------------------------------------------- feature

FEATURE_FILES = {
    "ledger.py": '''class Ledger:
    """Records money movements and reports a balance."""

    def __init__(self):
        self._entries = []

    def add(self, amount, label="deposit"):
        self._entries.append((label, amount))

    def balance(self):
        return sum(a for _, a in self._entries)
''',
    "test_ledger.py": '''import unittest

from ledger import Ledger


class TestLedger(unittest.TestCase):
    def test_add_balance(self):
        lg = Ledger()
        lg.add(100, "open")
        lg.add(20, "topup")
        self.assertEqual(lg.balance(), 120)


if __name__ == "__main__":
    unittest.main()
''',
    "run_tests.sh": "#!/usr/bin/env bash\nset -e\npython3 -m unittest test_ledger -v\n",
}


async def _setup_feature(bundle) -> None:
    for path, content in FEATURE_FILES.items():
        bundle.env.write_file(path, content)
    bundle.env.write_file(".gitignore", "results/\n")
    await bundle.env.bash("git init -q && git add -A && git commit -qm seed")


async def _check_feature_behavior(ctx) -> tuple[bool, str]:
    out, code = await ctx.env.bash(
        "python3 - <<'EOF'\n"
        "from ledger import Ledger\n"
        "lg = Ledger()\n"
        "lg.add(100, 'open')\n"
        "lg.subtract(30, 'buy')\n"
        "assert lg.balance() == 70, lg.balance()\n"
        "h = lg.history()\n"
        "assert h[1][0] == 'buy' and h[1][1] == -30, h\n"
        "assert h[1][2] == 70, h\n"
        "print('ok')\n"
        "EOF"
    )
    return code == 0 and "ok" in out, out[-200:]


def _check_feature_tested(ctx) -> tuple[bool, str]:
    try:
        test = ctx.env.read_file("test_ledger.py", max_chars=200_000)
    except Exception:
        rows = ctx.env.grep("subtract", include="test_*.py")
        return bool(rows), "grep fallback: " + "; ".join(rows[:3])
    ok = "subtract" in test and "history" in test
    return ok, "test covers subtract+history" if ok else "tests missing coverage"


async def _check_feature_committed(ctx) -> tuple[bool, str]:
    out, code = await ctx.env.bash(
        "git log --oneline | wc -l && git status --porcelain | wc -l"
    )
    lines = [ln.strip() for ln in out.strip().splitlines()]
    try:
        commits, dirty = int(lines[0]), int(lines[1])
    except (ValueError, IndexError):
        return False, f"unexpected git output: {out!r}"
    return commits >= 2 and dirty == 0, f"commits={commits} dirty_files={dirty}"


plain_feature_add = Scenario(
    name="plain-feature-ledger",
    title="Feature: add subtract() + history() with tests, commit cleanly",
    toolset="plain",
    task=(
        "Extend the Ledger class in ledger.py with:\n"
        "1. subtract(amount, label) — records a negative movement\n"
        "2. history() — returns a list of (label, delta, balance_after) tuples\n"
        "Add unittest coverage for both, keep `bash run_tests.sh` green, then "
        "commit everything to git (clean working tree afterwards)."
    ),
    setup=_setup_feature,
    checks=[
        ("behavior_subtract_history", _check_feature_behavior),
        ("tests_cover_new_api", _check_feature_tested),
        ("tests_still_pass", _check_tests_pass),
        ("git_committed_clean", _check_feature_committed),
    ],
    max_turns=30,
    max_tool_calls=80,
    max_seconds=480,
)

# ---------------------------------------------------------------- refactor

REFACTOR_FILES = {
    "users.py": '''def get_user_data(user_id):
    """Load a user profile from the local store."""
    return {"id": user_id, "name": f"user-{user_id}", "active": True}


def render_profile(user_id):
    data = get_user_data(user_id)
    return f"<profile>{data['name']}</profile>"
''',
    "api.py": '''from users import get_user_data


def handle_get(request):
    user = get_user_data(request["user_id"])
    return {"ok": True, "user": user}
''',
    "runner.py": '''from users import get_user_data, render_profile


def main(user_id):
    print(render_profile(user_id))
    return get_user_data(user_id)
''',
    "test_users.py": '''import unittest

from users import get_user_data


class TestUsers(unittest.TestCase):
    def test_load(self):
        u = get_user_data(42)
        self.assertEqual(u["id"], 42)
        self.assertTrue(u["active"])


if __name__ == "__main__":
    unittest.main()
''',
    "run_tests.sh": "#!/usr/bin/env bash\nset -e\npython3 -m unittest test_users -v\n",
}


def _setup_refactor(bundle) -> None:
    for path, content in REFACTOR_FILES.items():
        bundle.env.write_file(path, content)


def _check_old_name_gone(ctx) -> tuple[bool, str]:
    rows = ctx.env.grep("get_user_data", include="*.py")
    return not rows, "; ".join(rows[:5]) or "old name fully removed"


def _check_new_name_present(ctx) -> tuple[bool, str]:
    for f in ("users.py", "api.py", "runner.py", "test_users.py"):
        try:
            if "fetch_user_profile" not in ctx.env.read_file(f, max_chars=200_000):
                return False, f"{f} missing fetch_user_profile"
        except Exception as e:
            return False, f"cannot read {f}: {e}"
    return True, "renamed everywhere"


plain_refactor_rename = Scenario(
    name="plain-refactor-rename",
    title="Mechanical refactor: rename get_user_data -> fetch_user_profile",
    toolset="plain",
    task=(
        "Rename the function get_user_data to fetch_user_profile across the "
        "entire codebase (definition, all call sites, tests). Nothing else "
        "should change; `bash run_tests.sh` must stay green."
    ),
    setup=_setup_refactor,
    checks=[
        ("old_name_gone", _check_old_name_gone),
        ("new_name_everywhere", _check_new_name_present),
        ("tests_pass", _check_tests_pass),
    ],
    max_turns=24,
    max_tool_calls=60,
    max_seconds=420,
)

# ---------------------------------------------------------------- delegate

DELEGATE_FILES = {
    "router.py": '''ROUTES = []


def register(pattern, handler):
    ROUTES.append((pattern, handler))


def dispatch(path):
    """Return the handler registered for exactly this path, else None."""
    for pattern, handler in ROUTES:
        if pattern == path[: len(pattern)]:  # suspicious prefix match
            return handler
    return None
''',
    "test_router.py": '''import unittest

import router


class TestRouter(unittest.TestCase):
    def setUp(self):
        router.ROUTES.clear()

    def test_exact(self):
        router.register("/users", lambda: "users")
        self.assertEqual(router.dispatch("/users")(), "users")

    def test_prefix_must_not_match(self):
        router.register("/users", lambda: "users")
        router.register("/users/archive", lambda: "archive")
        self.assertEqual(router.dispatch("/users/archive")(), "archive")

    def test_unknown(self):
        self.assertIsNone(router.dispatch("/nope"))


if __name__ == "__main__":
    unittest.main()
''',
    "middleware.py": (
        "# request middleware; unrelated to routing decisions\n\n"
        "class Logging:\n    def wrap(self, req, inner):\n        print('req', req)\n"
        "        return inner(req)\n"
    ),
    "config.py": "PORT = 8080\nDEBUG = False\n",
    "app.py": (
        "import router\nfrom middleware import Logging\n\n\ndef build():\n"
        "    router.register('/health', lambda: 'ok')\n    return router\n"
    ),
    "run_tests.sh": "#!/usr/bin/env bash\nset -e\npython3 -m unittest test_router -v\n",
}


def _setup_delegate(bundle) -> None:
    for path, content in DELEGATE_FILES.items():
        bundle.env.write_file(path, content)


async def _check_dispatch_exact(ctx) -> tuple[bool, str]:
    out, code = await ctx.env.bash(
        "python3 - <<'EOF'\n"
        "import router\n"
        "router.ROUTES.clear()\n"
        "router.register('/users', lambda: 'users')\n"
        "router.register('/users/archive', lambda: 'archive')\n"
        "assert router.dispatch('/users')() == 'users'\n"
        "assert router.dispatch('/users/archive')() == 'archive'\n"
        "assert router.dispatch('/nope') is None\n"
        "print('ok')\n"
        "EOF"
    )
    return code == 0 and "ok" in out, out[-200:]


def _check_delegated(ctx) -> tuple[bool, str]:
    calls = [c for c in ctx.run_ctx.env.call_log if c.name == "subtask"]
    return len(calls) >= 1, f"{len(calls)} subtask call(s)"


async def _check_tests_pass_router(ctx) -> tuple[bool, str]:
    out, code = await ctx.env.bash("bash run_tests.sh")
    return code == 0, f"exit={code} tail={out[-200:]!r}"


plain_delegate_investigate = Scenario(
    name="plain-delegate-router",
    title="Delegated investigation: subtask explores, agent fixes router",
    toolset="plain",
    task=(
        "test_router.py fails. Protocol: FIRST delegate a read-only investigation "
        "to the subtask tool (ask it to locate the routing decision logic and "
        "explain exactly why the prefix test fails, citing file:line). Then fix "
        "the bug yourself so `bash run_tests.sh` passes. Do not skip the "
        "delegation step."
    ),
    setup=_setup_delegate,
    checks=[
        ("subtask_used", _check_delegated),
        ("dispatch_exact_match", _check_dispatch_exact),
        ("tests_pass", _check_tests_pass_router),
    ],
    max_turns=28,
    max_tool_calls=90,
    max_seconds=540,
)

PLAIN_SCENARIOS = [
    plain_bugfix_flake,
    plain_feature_add,
    plain_refactor_rename,
    plain_delegate_investigate,
]
