import time 
from google.genai import errors as genai_errors
from rag import client
# ---- Rate limiting -------------------------------------------------
# Free tier allows 15 generate_content requests/min. Each case makes up to 3
# calls (rerank, answer, judge), so throttle every call that goes through the
# shared client. Time spent waiting is tracked so it can be excluded from latency.
RPM_LIMIT = 14  # stay one under the 15/min cap
_call_times: list[float] = []
_waited_s = 0.0
_orig_generate_content = client.models.generate_content


def get_waited_s() -> float:
    """Total seconds spent sleeping for rate limits so far."""
    return _waited_s


def _throttled_generate_content(*args, **kwargs):
    global _waited_s
    for attempt in range(6):
        now = time.monotonic()
        _call_times[:] = [t for t in _call_times if now - t < 60]
        if len(_call_times) >= RPM_LIMIT:
            wait = 60 - (now - _call_times[0]) + 0.5
            print(f"  [throttle] rate limit window full, sleeping {wait:.1f}s")
            time.sleep(wait)
            _waited_s += wait
            continue
        _call_times.append(time.monotonic())
        try:
            return _orig_generate_content(*args, **kwargs)
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) != 429:
                raise
            wait = 15.0
            for d in (getattr(e, "details", None) or {}).get("error", {}).get("details", []):
                if d.get("@type", "").endswith("RetryInfo"):
                    wait = float(str(d.get("retryDelay", "15s")).rstrip("s")) + 1
            print(f"  [429] quota hit, retrying in {wait:.1f}s")
            time.sleep(wait)
            _waited_s += wait
    raise RuntimeError("generate_content: still rate limited after retries")

