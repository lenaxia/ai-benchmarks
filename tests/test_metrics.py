import time

from aibench.metrics import StreamTracker, decode_tps, percentile, prefill_tps


def test_tracker_derived_stats(monkeypatch):
    t = StreamTracker()
    real = time.perf_counter
    fake = {"t": real()}
    monkeypatch.setattr(time, "perf_counter", lambda: fake["t"])
    t.request_sent = fake["t"]
    fake["t"] += 0.5  # ttft 500ms
    t.on_token()
    for gap in (0.01, 0.01, 0.02, 0.01):
        fake["t"] += gap
        t.on_token()
    fake["t"] += 0.001
    snap = t.snapshot()
    assert snap.ttft_ms is not None and 499 < snap.ttft_ms < 502
    assert snap.n_chunks == 5
    # gaps [10, 10, 20, 10] ms (approx)
    assert snap.itl_mean_ms is not None and 12 < snap.itl_mean_ms < 13
    assert snap.itl_p95_ms is not None and 17 < snap.itl_p95_ms < 21
    assert snap.itl_max_ms is not None and 19 < snap.itl_max_ms < 21
    assert snap.total_ms > 540


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([5.0], 50) == 5.0
    vals = [1.0, 2.0, 3.0, 4.0]
    assert percentile(vals, 50) == 2.5
    assert percentile(vals, 100) == 4.0


def test_decode_tps_math():
    class T:
        first_token = 10.0
        last_token = 11.0

    assert decode_tps(T(), 101) == 100.0
    assert decode_tps(T(), 1) is None
    assert decode_tps(T(), None) is None


def test_prefill_tps_math():
    class T:
        ttft_ms = 1000.0

    assert prefill_tps(T(), 5000) == 5000.0  # 5000 non-cached tokens / 1s
    assert prefill_tps(T(), 5000, cached_tokens=4000) == 1000.0
    assert prefill_tps(T(), None) is None
