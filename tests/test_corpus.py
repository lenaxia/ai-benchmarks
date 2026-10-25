from aibench.bench.corpus import TokenEstimator


def test_estimate_and_filler_size():
    est = TokenEstimator(use_tiktoken=False)
    text = est.filler(1000)
    assert 700 <= est.count(text) <= 1400  # rough sizing is fine


def test_unique_fillers_differ_in_prefix():
    est = TokenEstimator(use_tiktoken=False)
    a = est.filler(500, unique=True)
    b = est.filler(500, unique=True)
    assert a[:64] != b[:64]
    seeded_a = est.filler(500, seed=42)
    seeded_b = est.filler(500, seed=42)
    assert seeded_a == seeded_b  # reproducible -> shareable prefix


def test_context_prompt_ends_with_instruction():
    est = TokenEstimator(use_tiktoken=False)
    instruction = "Say OK."
    p = est.context_prompt(600, instruction=instruction)
    assert p.rstrip().endswith(instruction)
    assert est.count(p) >= 550
