"""aibench: LLM inference benchmarking harness.

Covers three families of measurement:

- **Performance**: decode tokens/sec at context levels, prefill speed (TTFT),
  prefix-cache hit/miss behaviour with deliberately varied contexts.
- **Hardware**: capture the GPU/CPU/RAM configuration a run executed against.
- **Agentic**: tool-loop scenarios over plain coding tools (read/write/edit/
  bash/subtask) and a mocked llmsafespaces board/session/task platform.
"""

__version__ = "0.1.0"
