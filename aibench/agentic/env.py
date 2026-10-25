"""Sandboxed environment for agentic scenarios: a temp workspace, confined
bash execution, PATH-shimmed binaries, and a complete tool-call log.

Scenarios seed files via :meth:`Env.write_file` and may install fake CLIs
(e.g. a ``gh`` shim) with :meth:`Env.install_bin` — the PATH-stub pattern.
"""

from __future__ import annotations

import asyncio
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Any


class EnvError(Exception):
    pass


class CallRecord:
    __slots__ = ("depth", "name", "args", "ok", "output_chars", "duration_s", "ts", "output")

    def __init__(self, depth: int, name: str, args: dict, ok: bool, output: str, duration_s: float):
        self.depth = depth
        self.name = name
        self.args = args
        self.ok = ok
        self.output = output
        self.output_chars = len(output)
        self.duration_s = duration_s
        self.ts = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "name": self.name,
            "args": self.args,
            "ok": self.ok,
            "output_chars": self.output_chars,
            "duration_s": round(self.duration_s, 4),
            "output": self.output[:4000],
        }


MAX_OUTPUT_CHARS = 12_000
DEFAULT_BASH_TIMEOUT = 60


class Env:
    def __init__(self, root: Path | None = None, max_output_chars: int = MAX_OUTPUT_CHARS):
        self.root = (root or Path(tempfile.mkdtemp(prefix="aibench-env-"))).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.bin_dir = self.root / ".bin"
        self.bin_dir.mkdir(exist_ok=True)
        self.max_output_chars = max_output_chars
        self.call_log: list[CallRecord] = []
        # git works out of the box inside the sandbox
        (self.root / ".gitconfig").write_text(
            "[user]\n\tname = bench-agent\n\temail = bench@aibench.local\n"
            "[init]\n\tdefaultBranch = main\n[safe]\n\tdirectory = *\n"
        )

    # ---------- paths ----------
    def resolve(self, rel_path: str) -> Path:
        p = Path(rel_path)
        if not p.is_absolute():
            p = self.root / p
        resolved = p.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise EnvError(f"path escapes sandbox: {rel_path}")
        return resolved

    # ---------- file ops ----------
    def write_file(self, rel_path: str, content: str) -> Path:
        p = self.resolve(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def read_file(self, rel_path: str, max_chars: int | None = None) -> str:
        p = self.resolve(rel_path)
        if not p.exists():
            raise EnvError(f"file not found: {rel_path}")
        if p.is_dir():
            raise EnvError(f"is a directory: {rel_path}")
        content = p.read_text(errors="replace")
        limit = max_chars if max_chars is not None else self.max_output_chars
        if len(content) > limit:
            content = content[:limit] + f"\n... [truncated, {len(content)} chars total]"
        return content

    def edit_file(
        self, rel_path: str, old: str, new: str, replace_all: bool = False
    ) -> str:
        p = self.resolve(rel_path)
        if not p.exists():
            raise EnvError(f"file not found: {rel_path}")
        content = p.read_text()
        count = content.count(old)
        if count == 0:
            raise EnvError("old_string not found in content")
        if count > 1 and not replace_all:
            raise EnvError(
                f"old_string found {count} times; provide more context or set replace_all"
            )
        if replace_all:
            content = content.replace(old, new)
        else:
            content = content.replace(old, new, 1)
        p.write_text(content)
        return f"replaced {count if replace_all else 1} occurrence(s) in {rel_path}"

    def glob(self, pattern: str) -> list[str]:
        matches = sorted(self.root.glob(pattern))
        out = []
        for m in matches:
            rel = m.relative_to(self.root).as_posix()
            if rel.startswith(".bin/") or rel == ".gitconfig":
                continue
            out.append(rel + ("/" if m.is_dir() else ""))
        return out[:200]

    def grep(self, pattern: str, include: str | None = None) -> list[str]:
        import re

        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise EnvError(f"invalid regex: {e}") from e
        results: list[str] = []
        if include and "/" not in include:
            include = f"**/{include}"
        files = sorted(set(self.root.glob(include or "**/*")))
        for f in files:
            if not f.is_file() or any(
                part.startswith(".") for part in f.relative_to(self.root).parts
            ):
                continue
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    results.append(
                        f"{f.relative_to(self.root).as_posix()}:{i}: {line.strip()[:240]}"
                    )
                    if len(results) >= 200:
                        return results
        return results

    # ---------- bash ----------
    async def bash(self, command: str, timeout_s: int = DEFAULT_BASH_TIMEOUT) -> tuple[str, int]:
        env = {
            "PATH": f"{self.bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": str(self.root),
            "GIT_CONFIG_GLOBAL": str(self.root / ".gitconfig"),
            "LANG": "C.UTF-8",
            "TERM": "dumb",
        }
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=self.root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return f"error: command timed out after {timeout_s}s", 124
        text = out.decode(errors="replace")
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + f"\n... [truncated, {len(text)} chars total]"
        return text, proc.returncode or 0

    def install_bin(self, name: str, script: str) -> Path:
        """Install an executable shim into the sandbox PATH (overrides real CLIs)."""
        p = self.bin_dir / name
        p.write_text(script)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return p

    # ---------- lifecycle ----------
    def cleanup(self) -> None:
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)
