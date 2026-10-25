"""Plain coding toolset (opencode-flavoured): read, write, edit, bash,
glob, grep. The subtask tool lives in :mod:`aibench.agentic.tools.subtask`.
"""

from __future__ import annotations

import json
from typing import Any

from aibench.agentic.env import EnvError
from aibench.agentic.tools.base import Tool


def _fmt_env_error(e: EnvError) -> str:
    return json.dumps({"error": str(e)})


class ReadTool(Tool):
    name = "read"
    description = (
        "Read a file from the workspace. Returns text content "
        "(truncated for very large files)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to workspace root"},
            "offset": {"type": "integer", "description": "1-indexed line to start from"},
            "limit": {"type": "integer", "description": "Max lines to return"},
        },
        "required": ["path"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        try:
            content = ctx.env.read_file(args["path"])
        except EnvError as e:
            return _fmt_env_error(e)
        offset = args.get("offset")
        limit = args.get("limit")
        if offset or limit:
            lines = content.splitlines(keepends=True)
            start = max(0, (offset or 1) - 1)
            end = start + limit if limit else len(lines)
            content = "".join(lines[start:end])
        return content or "(empty file)"


class WriteTool(Tool):
    name = "write"
    description = "Create or overwrite a file with the given content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        ctx.env.write_file(args["path"], args.get("content", ""))
        return f"wrote {args['path']}"


class EditTool(Tool):
    name = "edit"
    description = (
        "Replace an exact string in a file. old_string must match exactly "
        "(including whitespace) and be unique unless replace_all is true."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        try:
            return ctx.env.edit_file(
                args["path"],
                args["old_string"],
                args["new_string"],
                replace_all=bool(args.get("replace_all")),
            )
        except EnvError as e:
            return _fmt_env_error(e)


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a bash command in the workspace root. stdout+stderr are returned; "
        "non-zero exit codes are reported. Commands have a timeout (default 60s)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "integer"},
        },
        "required": ["command"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        output, code = await ctx.env.bash(
            args["command"], timeout_s=int(args.get("timeout_s", 60))
        )
        if code != 0:
            return output + f"\n[exit code {code}]"
        return output or "(no output)"


class GlobTool(Tool):
    name = "glob"
    description = "List files matching a glob pattern (e.g. '**/*.py')."
    parameters = {
        "type": "object",
        "properties": {"pattern": {"type": "string"}},
        "required": ["pattern"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        matches = ctx.env.glob(args["pattern"])
        return "\n".join(matches) or "(no matches)"


class GrepTool(Tool):
    name = "grep"
    description = "Regex search across workspace files; returns path:line: text rows."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "include": {"type": "string", "description": "Glob filter, e.g. '*.py'"},
        },
        "required": ["pattern"],
    }

    async def run(self, args: dict[str, Any], ctx) -> str:
        try:
            rows = ctx.env.grep(args["pattern"], args.get("include"))
        except EnvError as e:
            return _fmt_env_error(e)
        return "\n".join(rows) or "(no matches)"


def plain_tools() -> list[Tool]:
    return [ReadTool(), WriteTool(), EditTool(), BashTool(), GlobTool(), GrepTool()]
