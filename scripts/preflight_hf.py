"""Pre-flight: do every Hugging Face id the lab depends on still resolve?

The changelog records one dataset disappearing (401) after the lab shipped, and
a Colab session is expensive to lose. NB1-NB4 download all of these, so check
the ids and the configs the notebooks actually index before spending GPU time.

    python scripts/preflight_hf.py          # exit 1 if anything 404s/401s
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

# (kind, repo id, what needs it)
DEPS = [
    ("model", "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", "NB1-NB4 base (T4 tier)"),
    ("model", "unsloth/Qwen3-8B-unsloth-bnb-4bit", "base (BIGGPU tier)"),
    ("model", "Skywork/Skywork-Reward-V2-Qwen3-4B", "NB4 judge panel"),
    ("model", "Skywork/Skywork-Reward-V2-Llama-3.2-3B", "NB4 judge panel"),
    ("dataset", "saillab/alpaca-vietnamese-cleaned", "NB1 SFT data"),
    ("dataset", "sailor2/sea-ultrafeedback-onpolicy", "NB2 preference data"),
    ("dataset", "vuongtsc/vi-gsm8k-agentic", "NB7 GRPO (bonus)"),
]


def head(kind: str, repo_id: str) -> tuple[int, str]:
    """Return (http status, note) for a repo's metadata endpoint."""
    url = f"https://huggingface.co/api/{'models' if kind == 'model' else 'datasets'}/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "lab22-preflight"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            meta = json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.reason
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, f"network: {exc}"
    gated = " (gated: needs HF_TOKEN)" if meta.get("gated") not in (False, None) else ""
    return 200, f"private={meta.get('private', False)}{gated}"


def main() -> int:
    bad = []
    for kind, repo_id, why in DEPS:
        status, note = head(kind, repo_id)
        ok = status == 200
        print(f"{'OK  ' if ok else 'FAIL'} {kind:8s} {repo_id:55s} [{status}] {note}  <- {why}")
        if not ok:
            bad.append(repo_id)
    print()
    if bad:
        print(f"{len(bad)} id(s) unavailable: {bad}")
        print("Fix lab22/config.py (BASE_MODEL / *_DATASET / JUDGE_RM_MODELS) before running on Colab.")
        return 1
    print("All ids resolve: safe to spend Colab GPU time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())