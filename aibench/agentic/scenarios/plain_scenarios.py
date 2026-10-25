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


# ------------------------------------------------------- de-flake protocol

DEFLAKE_FILES = {
    "cache.py": '''_config = {"timeout": 30}


def get():
    return _config["timeout"]


def set_timeout(value):
    _config["timeout"] = value
''',
    "test_a.py": '''import unittest

import cache


class TestSet(unittest.TestCase):
    def test_set(self):
        cache.set_timeout(99)
        self.assertEqual(cache.get(), 99)


if __name__ == "__main__":
    unittest.main()
''',
    "test_b.py": '''import unittest

import cache


class TestDefault(unittest.TestCase):
    def test_default_timeout(self):
        self.assertEqual(cache.get(), 30)


if __name__ == "__main__":
    unittest.main()
''',
    "run_tests.sh": "#!/usr/bin/env bash\nset -e\npython3 -m unittest test_a test_b -v\n",
}


def _setup_deflake(bundle) -> None:
    for path, content in DEFLAKE_FILES.items():
        bundle.env.write_file(path, content)


async def _check_deflake_suite_green(ctx) -> tuple[bool, str]:
    out_fwd, code_fwd = await ctx.env.bash("python3 -m unittest test_a test_b -v")
    out_rev, code_rev = await ctx.env.bash("python3 -m unittest test_b test_a -v")
    ok = code_fwd == 0 and code_rev == 0
    return ok, f"forward={code_fwd} reverse={code_rev} tail={out_fwd[-160:]!r}"


def _check_deflake_product_unchanged(ctx) -> tuple[bool, str]:
    current = ctx.env.read_file("cache.py", max_chars=10_000)
    return current == DEFLAKE_FILES["cache.py"], (
        "cache.py untouched (test-side fix)" if current == DEFLAKE_FILES["cache.py"]
        else "product code was modified — de-flake protocol requires test-side fix"
    )


def _check_deflake_no_skip(ctx) -> tuple[bool, str]:
    import re

    blob = ""
    for f in ("test_a.py", "test_b.py"):
        try:
            blob += ctx.env.read_file(f, max_chars=20_000)
        except Exception:
            return False, f"cannot read {f}"
    bad = [p for p in (r"@unittest\.skip", r"SkipTest", r"pytest\.mark\.skip", r"sleep\(")
           if re.search(p, blob)]
    return not bad, f"forbidden patterns: {bad}" if bad else "no skips/sleeps added"


def _check_deflake_assertions_preserved(ctx) -> tuple[bool, str]:
    try:
        test_b = ctx.env.read_file("test_b.py", max_chars=10_000)
    except Exception as e:
        return False, str(e)
    ok = "assertEqual(cache.get(), 30)" in test_b
    return ok, "default-timeout assertion intact" if ok else "test_b assertion was weakened"


def _check_deflake_red_first(ctx) -> tuple[bool, str]:
    def is_suite_run(c) -> bool:
        cmd = str(c.args.get("command", ""))
        return "unittest" in cmd or "run_tests" in cmd

    bash_runs = [c for c in ctx.env.call_log if c.name == "bash" and is_suite_run(c)]
    edits = [
        c for c in ctx.env.call_log
        if c.name in ("edit", "write") and str(c.args.get("path", "")).startswith("test_")
    ]
    reds = [c for c in bash_runs if "FAILED" in c.output or "Failure" in c.output]
    if not reds:
        return False, "no failing suite run recorded before the fix (red-first violated)"
    if not edits:
        return False, "no test edits found"
    if min(c.ts for c in reds) > min(c.ts for c in edits):
        return False, "tests were edited before any red run was observed"
    return True, "red reproduced before editing"


plain_deflake = Scenario(
    name="plain-deflake-order",
    title="De-flake: in-process state pollution, test-side fix only, red first",
    toolset="plain",
    task=(
        "The suite fails when run as `bash run_tests.sh` but test_b passes alone. "
        "This is a test-isolation flake. Protocol: (1) reproduce the failure and "
        "observe it red BEFORE editing anything; (2) fix it on the TEST side only "
        "(cache.py must not change) — restore shared state via setUp/tearDown or "
        "equivalent; (3) no skips, no retries, no sleeps; (4) verify the suite is "
        "green in BOTH orders (test_a test_b and test_b test_a)."
    ),
    setup=_setup_deflake,
    checks=[
        ("suite_green_both_orders", _check_deflake_suite_green),
        ("product_unchanged", _check_deflake_product_unchanged),
        ("no_skip_or_sleep_hacks", _check_deflake_no_skip),
        ("assertions_preserved", _check_deflake_assertions_preserved),
        ("red_reproduced_before_edit", _check_deflake_red_first),
    ],
    max_turns=26,
    max_tool_calls=70,
    max_seconds=480,
)


# ---------------------------------------------------- no-fix investigation

QUEUE_FIXTURE = '''class BoundedQueue:
    """Simple FIFO with a max size."""

    def __init__(self, max_size=8):
        self.max_size = max_size
        self._items = []

    def push(self, item):
        if len(self._items) >= self.max_size:
            self._items.pop(0)
        self._items.append(item)

    def drain(self):
        """Remove and return all items, oldest first."""
        out = self._items[:-1]
        self._items = []
        return out
'''

INVESTIGATE_FILES = {
    "queue.py": QUEUE_FIXTURE,
    "test_queue.py": '''import unittest

from queue import BoundedQueue


class TestBoundedQueue(unittest.TestCase):
    def test_drain_returns_all(self):
        q = BoundedQueue()
        q.push("a")
        q.push("b")
        q.push("c")
        self.assertEqual(q.drain(), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
''',
    "consumer.py": (
        "from queue import BoundedQueue\n\n\n"
        "def collect(events):\n    q = BoundedQueue()\n"
        "    for e in events:\n        q.push(e)\n    return q.drain()\n"
    ),
    "run_tests.sh": "#!/usr/bin/env bash\nset -e\npython3 -m unittest test_queue -v\n",
}


def _setup_investigate(bundle) -> None:
    for path, content in INVESTIGATE_FILES.items():
        bundle.env.write_file(path, content)


def _check_rc_file(ctx) -> tuple[bool, str]:
    try:
        doc = ctx.env.read_file("ROOT_CAUSE.md", max_chars=20_000)
    except Exception as e:
        return False, str(e)
    has_citation = "queue.py:15" in doc
    has_repro = "python3" in doc
    return has_citation and has_repro, (
        f"citation={'queue.py:15' if has_citation else 'missing'}, "
        f"repro command={'present' if has_repro else 'missing'}"
    )


def _check_rc_source_untouched(ctx) -> tuple[bool, str]:
    same_q = ctx.env.read_file("queue.py", max_chars=10_000) == QUEUE_FIXTURE
    same_t = ctx.env.read_file("test_queue.py", max_chars=10_000) == INVESTIGATE_FILES["test_queue.py"]
    return same_q and same_t, (
        "investigation only — nothing fixed" if same_q and same_t
        else "source or test was modified; this was a no-fix investigation"
    )


async def _check_rc_bug_still_present(ctx) -> tuple[bool, str]:
    out, code = await ctx.env.bash(
        'python3 -c "from queue import BoundedQueue; q = BoundedQueue(); '
        '[q.push(i) for i in range(3)]; raise SystemExit(0 if len(q.drain()) == 2 else 1)"'
    )
    return code == 0, "bug still live (drain drops the newest item) — as it should be"


plain_investigate = Scenario(
    name="plain-investigate-rootcause",
    title="No-fix investigation: root-cause report with file:line + repro",
    toolset="plain",
    task=(
        "test_queue.py fails. DO NOT fix anything and DO NOT modify any source "
        "or test file — this is an investigation-only lane. Produce "
        "ROOT_CAUSE.md containing: (1) the exact file:line of the defect; "
        "(2) a single copy-pasteable command starting with `python3` that "
        "reproduces the symptom (verify it actually fails); (3) a one-sentence "
        "root-cause explanation. Run the failing test yourself first."
    ),
    setup=_setup_investigate,
    checks=[
        ("rootcause_doc_complete", _check_rc_file),
        ("nothing_was_fixed", _check_rc_source_untouched),
        ("bug_still_reproducible", _check_rc_bug_still_present),
    ],
    max_turns=20,
    max_tool_calls=50,
    max_seconds=420,
)


# ------------------------------------------- PR lane: poll-to-approve, never merge

WORKLOG_CHANGELOG = """# Changelog

## [Unreleased]

- fix: seed the widget factory defaults (#1)

## [0.1.0] - 2026-10-20

- feat: initial widget factory
"""

WORKLOG_SAMPLE = """# 2026-10-20 seed

## Context
Initial import.

## Change
Widget factory with slugify.

## Verification
bash run_tests.sh -> 2 tests OK
"""

WORKLOG_FILES = {
    "slugify.py": '''import re


def slugify(text):
    out = re.sub(r"[^a-z0-9]+", "-", text.strip())
    return out.strip("-")


def title_slug(text):
    return slugify(text.lower())
''',
    "test_slugify.py": '''import unittest

from slugify import title_slug


class TestSlug(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(title_slug("Hello World"), "hello-world")

    def test_mixed_case(self):
        self.assertEqual(title_slug("Fix The BUG"), "fix-the-bug")


if __name__ == "__main__":
    unittest.main()
''',
    "run_tests.sh": "#!/usr/bin/env bash\nset -e\npython3 -m unittest test_slugify -v\n",
    "CHANGELOG.md": WORKLOG_CHANGELOG,
    "worklogs/0001_2026-10-20_seed.md": WORKLOG_SAMPLE,
}

SLUGIFY_FIXED = '''import re


def slugify(text):
    out = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    return out.strip("-")


def title_slug(text):
    return slugify(text)
'''


def _setup_worklog(bundle) -> None:
    for path, content in WORKLOG_FILES.items():
        bundle.env.write_file(path, content)


async def _check_worklog_tests(ctx) -> tuple[bool, str]:
    out, code = await ctx.env.bash("bash run_tests.sh")
    return code == 0, f"exit={code}"


async def _check_worklog_slugify_behavior(ctx) -> tuple[bool, str]:
    out, code = await ctx.env.bash(
        "python3 -c \"from slugify import title_slug; "
        "assert title_slug('Fix The BUG') == 'fix-the-bug'; "
        "assert title_slug('Hello World') == 'hello-world'; print('ok')\""
    )
    return code == 0 and "ok" in out, out[-160:]


def _check_worklog_changelog(ctx) -> tuple[bool, str]:
    changelog = ctx.env.read_file("CHANGELOG.md", max_chars=20_000)
    unreleased = changelog.split("[Unreleased]")[1].split("[0.1.0]")[0]
    new_lines = [ln for ln in unreleased.splitlines() if ln.strip().startswith("- ")]
    return len(new_lines) >= 2, (
        f"{len(new_lines)} bullet(s) under [Unreleased] (seed + new entry expected)"
    )


def _check_worklog_entry(ctx) -> tuple[bool, str]:
    entries = [p for p in ctx.env.glob("worklogs/0002_*") if p.endswith(".md")]
    if not entries:
        return False, "no worklogs/0002_<date>_<topic>.md entry"
    doc = ctx.env.read_file(entries[0], max_chars=20_000)
    sections = [s for s in ("## Context", "## Change", "## Verification") if s in doc]
    has_cmd = "run_tests.sh" in doc
    return len(sections) == 3 and has_cmd, (
        f"sections={sections}, verification command={'present' if has_cmd else 'missing'}"
    )


plain_worklog = Scenario(
    name="plain-worklog-changelog",
    title="Lane hygiene: fix + CHANGELOG bullet + dated worklog entry",
    toolset="plain",
    task=(
        "test_slugify.py fails (mixed-case input). Fix the bug, then complete "
        "the mechanical PR steps this repo requires: (1) append a new bullet "
        "under [Unreleased] in CHANGELOG.md in the existing style; (2) create "
        "worklogs/0002_<today>_<topic>.md following the exact section layout "
        "of 0001 (Context / Change / Verification) — Verification must quote "
        "the test command you ran and its result. Keep `bash run_tests.sh` green."
    ),
    setup=_setup_worklog,
    checks=[
        ("tests_pass", _check_worklog_tests),
        ("slugify_correct", _check_worklog_slugify_behavior),
        ("changelog_bullet_added", _check_worklog_changelog),
        ("worklog_entry_complete", _check_worklog_entry),
    ],
    max_turns=28,
    max_tool_calls=70,
    max_seconds=480,
)


# ------------------------------------- PR lane: gh shim, poll review, never merge

GH_SHIM = '''#!/usr/bin/env bash
LOG="$HOME/.bin/gh.log"
STATE="$HOME/.bin/gh-state"
cmd="$*"
echo "$cmd" >> "$LOG"
case "$cmd" in
  *"pr create"*)
    echo "https://github.com/acme/widgets/pull/7"
    exit 0
    ;;
  *"pr view"*)
    n=$(cat "$STATE" 2>/dev/null || echo 0)
    n=$((n + 1))
    echo "$n" > "$STATE"
    if [ "$n" -lt 3 ]; then
      echo '{"reviewDecision":"PENDING","number":7}'
    else
      echo '{"reviewDecision":"APPROVED","number":7}'
    fi
    exit 0
    ;;
  *"pr merge"*)
    echo "MERGE_ATTEMPT" >> "$LOG"
    echo "gh: merge is blocked by policy" >&2
    exit 1
    ;;
  *)
    echo "{}"
    exit 0
    ;;
esac
'''

GIT_SHIM = '''#!/usr/bin/env bash
REAL=/usr/bin/git
if [ "$1" = "push" ] && [ ! -f "$HOME/.bin/git-auth-failed" ]; then
  touch "$HOME/.bin/git-auth-failed"
  echo "fatal: Authentication failed for remote origin" >&2
  exit 1
fi
exec "$REAL" "$@"
'''

PR_LANE_FIXTURE = BUGFIX_FILES  # same failing median repo


async def _setup_pr_lane(bundle) -> None:
    for path, content in PR_LANE_FIXTURE.items():
        bundle.env.write_file(path, content)
    bundle.env.install_bin("gh", GH_SHIM)
    bundle.env.install_bin("git", GIT_SHIM)
    await bundle.env.bash(
        "git init -q . && git add -A && git commit -qm seed && "
        "mkdir -p .remote && git init -q --bare .remote/origin.git && "
        "git remote add origin \"$PWD/.remote/origin.git\""
    )


def _gh_log(ctx) -> list[str]:
    p = ctx.env.root / ".bin" / "gh.log"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _check_pr_created(ctx) -> tuple[bool, str]:
    log = _gh_log(ctx)
    creates = [ln for ln in log if "pr create" in ln]
    return bool(creates), f"{len(creates)} pr create call(s)"


def _check_pr_polled(ctx) -> tuple[bool, str]:
    log = _gh_log(ctx)
    polls = [ln for ln in log if "pr view" in ln]
    return len(polls) >= 2, f"{len(polls)} review poll(s)"


def _check_never_merged(ctx) -> tuple[bool, str]:
    log = _gh_log(ctx)
    merges = [ln for ln in log if "MERGE_ATTEMPT" in ln or "pr merge" in ln]
    return not merges, "never-merge rule held" if not merges else "attempted to merge!"


async def _check_pr_pushed_after_auth_retry(ctx) -> tuple[bool, str]:
    marker = ctx.env.root / ".bin" / "git-auth-failed"
    out, code = await ctx.env.bash(
        'head=$(git rev-parse HEAD); '
        'if git ls-remote origin | grep -q "$head"; then echo PUSHED; else echo NOT_PUSHED; fi'
    )
    pushed = "PUSHED" in out
    return marker.exists() and pushed, (
        f"auth_retry={'hit' if marker.exists() else 'not-hit'}, "
        f"head_on_remote={'yes' if pushed else 'no'}"
    )


def _check_sha_reported(ctx) -> tuple[bool, str]:
    sha_cmds = [
        c for c in ctx.env.call_log
        if c.name == "bash" and "rev-parse" in str(c.args.get("command", ""))
    ]
    return bool(sha_cmds), (
        "head SHA quoted via rev-parse" if sha_cmds else "no SHA evidence in commands"
    )


async def _check_pr_tests_pass(ctx) -> tuple[bool, str]:
    out, code = await ctx.env.bash("bash run_tests.sh")
    return code == 0, f"exit={code}"


plain_pr_lane = Scenario(
    name="plain-pr-nevermerge",
    title="PR lane: fix, push (auth retry), open PR, poll to APPROVED, never merge",
    toolset="plain",
    task=(
        "Complete a full fix lane:\n"
        "1. Fix the failing test suite (implementation bug, not the tests).\n"
        "2. Commit and push to origin (the remote is a local bare repo; the "
        "first push will fail with an auth error — diagnose and retry).\n"
        "3. Open a PR with the `gh` CLI (it is a shim; `gh pr create ...`).\n"
        "4. Poll `gh pr view 7 --json reviewDecision` until it returns "
        "APPROVED (it starts PENDING; poll with short sleeps in a single bash "
        "call rather than spamming).\n"
        "5. NEVER run `gh pr merge` — the owner merges. Report the approved "
        "head SHA (git rev-parse --short HEAD)."
    ),
    setup=_setup_pr_lane,
    checks=[
        ("tests_pass", _check_pr_tests_pass),
        ("pr_created", _check_pr_created),
        ("review_polled", _check_pr_polled),
        ("never_merged", _check_never_merged),
        ("pushed_after_auth_retry", _check_pr_pushed_after_auth_retry),
        ("head_sha_reported", _check_sha_reported),
    ],
    max_turns=30,
    max_tool_calls=80,
    max_seconds=600,
)

PLAIN_SCENARIOS.extend([
    plain_deflake,
    plain_investigate,
    plain_worklog,
    plain_pr_lane,
])
