"""Prompt construction: token estimation and non-cachable filler text.

Every prefix-cache-aware benchmark needs prompts that genuinely differ in
their token prefix. We build filler from a random word stream (optionally
salted with a UUID) so no two "miss" requests share a token prefix, while
"hit" requests share one byte-identical prefix.
"""

from __future__ import annotations

import math
import random
import uuid

# ~256 common english words; enough entropy that random sequences of these
# never collapse into a shared token prefix across requests.
_WORDS = (  # noqa: SIM905
    """
    river lantern orbit copper meadow signal granite thunder parcel velvet cinder
    harbor beacon marsh quiver tundra falcon orchard prism socket ladder ember
    willow cavern ripple bramble socket forge aster cascade hollow juniper slate
    mosaic pebble kindle vantage wicker zephyr anchor drift meadow lattice spur
    cobalt fathom glisten hinge ivoire jasper kettle lumen mantle nugget opal
    plume quill russet saffron thistle umber verdant whorl xenon yonder zodiac
    amber brine clove dune etch flint glade haven iris jade kelp loam mote nook
    oasis prow quest reed shale tidal urn vane warp yew apex bloom crest dale
    eld fern grove heath inlet jamb knoll loch mull nil ort pier quay ramp scon
    tarn vale wane yarn zeal atlas birch chalk dross ergo fern glee helm itch
    jolt keel lark moss numb onyx pivot quip rove sash trek urge vext wisp yoga
    zest arch bail cope doze eave fray gum hoard ink jam knot lob mew nip oar
    pact rim sob tap vow wry yak zone ale bid cog dew elm fig hop ice jet lux
    mig nod oak pad rig sap tux vow wax yam zen
    """
).split()


class TokenEstimator:
    """chars-per-token heuristic (default ~4.0 for english prose).

    If ``tiktoken`` is installed it is used for exact counts.  The estimate
    only sizes filler targets; real counts come back from server usage data.
    """

    def __init__(self, chars_per_token: float = 4.0, use_tiktoken: bool = True):
        self.chars_per_token = chars_per_token
        self._enc = None
        if use_tiktoken:
            try:
                import tiktoken  # type: ignore

                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._enc = None

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text))
        return max(1, math.ceil(len(text) / self.chars_per_token))

    def filler(self, target_tokens: int, seed: int | None = None, unique: bool = False) -> str:
        """Random word filler of approximately ``target_tokens`` tokens.

        With ``unique=True`` a UUID sentence is prepended so the token prefix
        is guaranteed distinct from every other unique filler (defeats prefix
        caching). With a fixed ``seed`` the text is reproducible and stable —
        exactly what the cache-hit phase wants.
        """
        rng = random.Random(seed if seed is not None else uuid.uuid4().int)
        prefix = ""
        if unique:
            prefix = f"[document-{uuid.uuid4().hex}] "
        # calibrate word count against the actual estimator (heuristic or tiktoken)
        words: list[str] = []
        approx_words = max(1, int(target_tokens * 0.65))
        for _ in range(3):  # converge from below; overshooting slightly is fine
            words = [rng.choice(_WORDS) for _ in range(approx_words)]
            text = prefix + " ".join(words)
            got = self.count(text)
            if got >= target_tokens:
                return text
            approx_words = int(approx_words * (target_tokens / max(1, got)) * 1.02) + 1
        return prefix + " ".join(words)

    def context_prompt(
        self,
        target_tokens: int,
        *,
        instruction: str,
        seed: int | None = None,
        unique: bool = False,
    ) -> str:
        """A user message of ~``target_tokens`` tokens ending in an instruction.

        The filler comes first (it is the prefix); the instruction lands at
        the end where it can't be confused with padding.
        """
        overhead = self.count(instruction) + 40
        filler = self.filler(max(1, target_tokens - overhead), seed=seed, unique=unique)
        return (
            "Background material follows; it is reference-only context.\n\n"
            f"{filler}\n\n"
            f"{instruction}"
        )


DECODE_INSTRUCTION = (
    "Now count upward from 1, one number per line, without stopping, "
    "until you reach 200."
)

SUMMARIZE_INSTRUCTION = (
    "In one short sentence, state how many paragraphs of background material "
    "were provided above. Answer immediately."
)

CACHE_PROBE_INSTRUCTION = (
    "What is the first word of the background material above? Answer with "
    "just that word."
)
