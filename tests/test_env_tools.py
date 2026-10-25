import pytest

from aibench.agentic.env import Env, EnvError


@pytest.fixture()
def env():
    e = Env()
    yield e
    e.cleanup()


async def test_write_read_edit(env):
    env.write_file("a/b.txt", "one\ntwo\nthree\n")
    assert "two" in env.read_file("a/b.txt")
    env.edit_file("a/b.txt", "two", "TWO")
    assert "TWO" in env.read_file("a/b.txt")
    with pytest.raises(EnvError):
        env.edit_file("a/b.txt", "two", "x")  # gone now


async def test_edit_ambiguous_requires_replace_all(env):
    env.write_file("dup.txt", "x y x y")
    with pytest.raises(EnvError):
        env.edit_file("dup.txt", "x", "z")
    env.edit_file("dup.txt", "x", "z", replace_all=True)
    assert env.read_file("dup.txt") == "z y z y"


async def test_path_escape_blocked(env):
    with pytest.raises(EnvError):
        env.write_file("../outside.txt", "no")
    with pytest.raises(EnvError):
        env.read_file("/etc/passwd")


async def test_bash_runs_and_reports_exit(env):
    out, code = await env.bash("echo hello; false")
    assert "hello" in out
    assert code == 1


async def test_bash_timeout(env):
    out, code = await env.bash("sleep 30", timeout_s=1)
    assert code == 124
    assert "timed out" in out


async def test_install_bin_overrides_path(env):
    env.install_bin("fortune", "#!/usr/bin/env bash\necho custom-fortune\n")
    out, code = await env.bash("fortune")
    assert code == 0 and "custom-fortune" in out


async def test_git_works_out_of_box(env):
    env.write_file("f.txt", "hi")
    out, code = await env.bash("git init -q && git add -A && git commit -qm test && git log --oneline")
    assert code == 0 and "test" in out


async def test_glob_and_grep(env):
    env.write_file("src/alpha.py", "target = 1\n")
    env.write_file("src/beta.py", "other = 2\n")
    assert "src/alpha.py" in env.glob("**/*.py")
    rows = env.glob("*.py")
    assert rows == []  # nothing at root
    hits = env.grep("target", include="*.py")
    assert len(hits) == 1 and "alpha.py" in hits[0]


async def test_read_missing(env):
    with pytest.raises(EnvError):
        env.read_file("nope.txt")
