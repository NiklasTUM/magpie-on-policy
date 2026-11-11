#!/usr/bin/env python
"""Regenerate the responses of a Magpie file with a model served through OpenRouter.

Used for the fixed-instruction capability experiment: the instructions (and their
instruction-side tags: control_task, task_category, input_quality) stay exactly as they
are, only the response is rewritten by another model. The original response is kept as
`reference_response` so the two can be compared pair by pair.

Matches how exp/gen_res.py produced the Qwen3.6 responses: the instruction is sent as a
single user turn with no system prompt, thinking disabled, 4096 max tokens. Sampling is
the response model's own recommended setting (here Kimi K3.1 Pro: temperature 1.0,
top_p 0.95), as Qwen used its model-card preset.

Cost note: usage cost covers the final attempt only; resamples are billed too.

Resumable: every finished record is appended to OUT.jsonl; a rerun skips ids already
there. When all ids are done, OUT.json is written in the usual Magpie list format.

  python scripts/openrouter_gen_res.py IN.json OUT_PREFIX [--limit 30] [--concurrency 16]

Needs network (run on a login node) and the key in ~/.openrouter_api_key.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

URL = "https://openrouter.ai/api/v1/chat/completions"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tamia"))
from filter_pairs import REASONING_LEAK  # noqa: E402  (same rule the filter applies)


def call(body, key, retries=8):
    """POST with backoff on 429/5xx/network errors. Returns (json, None) or (None, err)."""
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(URL, data=data, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=600))
            if "error" in r:  # OpenRouter can return 200 with an error body
                err = str(r["error"])[:300]
                code = (r["error"] or {}).get("code") if isinstance(r["error"], dict) else None
                if code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(min(2 ** attempt + random.random(), 120))
                    continue
                return None, err
            return r, None
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(min(2 ** attempt + random.random(), 120))
                continue
            return None, err
        except Exception as e:  # timeouts, connection resets
            err = repr(e)[:300]
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt + random.random(), 120))
                continue
            return None, err
    return None, "retries exhausted"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_file")
    ap.add_argument("out_prefix", help="writes OUT_PREFIX.jsonl (progress) and OUT_PREFIX.json")
    ap.add_argument("--model", default="moonshotai/kimi-k3.1-pro")
    ap.add_argument("--provider", default="moonshotai",
                    help="Pinned OpenRouter provider; fallbacks are disabled so every "
                         "response comes from the same host and weights.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--leak_retries", type=int, default=3,
                    help="With thinking off, Kimi K3.1 sometimes writes its reasoning into "
                         "the answer ('I need respond to user...') and never answers (2/30 "
                         "in the pilot). Resample up to this many times; a response that "
                         "still leaks is kept but flagged, and filter_pairs.py drops it.")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N records (pilot).")
    args = ap.parse_args()

    key = open(os.path.expanduser("~/.openrouter_api_key")).read().strip()
    data = json.load(open(args.input_file, encoding="utf-8"))
    if args.limit:
        data = data[:args.limit]
    progress = args.out_prefix + ".jsonl"
    done = {}
    if os.path.exists(progress):
        for line in open(progress, encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
    todo = [r for r in data if r["id"] not in done]
    print(f"[openrouter] {len(data)} records, {len(done)} already done, {len(todo)} to go; "
          f"model={args.model} provider={args.provider} temp={args.temperature} "
          f"top_p={args.top_p} max_tokens={args.max_tokens} thinking=off", flush=True)

    config = {"model": args.model, "provider": args.provider, "temperature": args.temperature,
              "top_p": args.top_p, "max_tokens": args.max_tokens, "thinking": False,
              "system_prompt": None, "engine": "openrouter"}
    lock = threading.Lock()
    stats = {"ok": 0, "err": 0, "cost": 0.0, "out_tok": 0, "length": 0, "reasoning_tok": 0,
             "resampled": 0, "leak_kept": 0}
    t0 = time.time()

    def work(rec):
        body = {"model": args.model,
                "messages": [{"role": "user", "content": rec["instruction"]}],
                "temperature": args.temperature, "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "reasoning": {"enabled": False},
                "usage": {"include": True},
                "provider": {"order": [args.provider], "allow_fallbacks": False}}
        attempts = 0
        while True:
            attempts += 1
            r, err = call(body, key)
            text = "" if err else (r["choices"][0]["message"].get("content") or "")
            leak = bool(not err and REASONING_LEAK.search(text[:300]))
            if not leak or attempts > args.leak_retries:
                break
        out = dict(rec)
        out["reference_response"] = rec.get("response")
        out["reference_generator"] = (rec.get("gen_response_configs") or {}).get("output_generator")
        cfg = dict(config)
        if err:
            out["response"] = None
            cfg["error"] = err
        else:
            ch = r["choices"][0]
            u = r.get("usage") or {}
            out["response"] = ch["message"].get("content") or ""
            out["finish_reason"] = ch.get("finish_reason")
            cfg.update({"output_generator": args.model, "served_by": r.get("provider"),
                        "usage": u, "attempts": attempts, "reasoning_leak": leak})
        out["gen_response_configs"] = cfg
        return out, err, (r or {}).get("usage") or {}

    with open(progress, "a", encoding="utf-8") as fout, \
            cf.ThreadPoolExecutor(args.concurrency) as ex:
        for i, fut in enumerate(cf.as_completed([ex.submit(work, r) for r in todo]), 1):
            out, err, u = fut.result()
            with lock:
                if err:
                    stats["err"] += 1
                    print(f"[openrouter] id={out['id']} ERROR {err}", flush=True)
                    continue  # not written: a rerun retries it
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                stats["ok"] += 1
                stats["cost"] += float(u.get("cost") or 0)
                stats["out_tok"] += int(u.get("completion_tokens") or 0)
                stats["reasoning_tok"] += int((u.get("completion_tokens_details") or {})
                                              .get("reasoning_tokens") or 0)
                stats["length"] += out.get("finish_reason") == "length"
                stats["resampled"] += out["gen_response_configs"]["attempts"] > 1
                stats["leak_kept"] += out["gen_response_configs"]["reasoning_leak"]
                if i % 100 == 0 or i == len(todo):
                    rate = i / max(time.time() - t0, 1)
                    print(f"[openrouter] {i}/{len(todo)} ok={stats['ok']} err={stats['err']} "
                          f"cost=${stats['cost']:.2f} out_tok={stats['out_tok']} "
                          f"reasoning_tok={stats['reasoning_tok']} length={stats['length']} "
                          f"resampled={stats['resampled']} leak_kept={stats['leak_kept']} "
                          f"{rate:.2f}/s", flush=True)

    done = {}
    for line in open(progress, encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            done[rec["id"]] = rec
    missing = [r["id"] for r in data if r["id"] not in done]
    n = max(stats["ok"], 1)
    print(f"[openrouter] this run: ok={stats['ok']} err={stats['err']} cost=${stats['cost']:.2f} "
          f"(${stats['cost'] / n:.4f}/response) mean_out_tok={stats['out_tok'] / n:.0f} "
          f"reasoning_tok={stats['reasoning_tok']} hit_max_tokens={stats['length']} "
          f"resampled_for_leak={stats['resampled']} still_leaking={stats['leak_kept']}")
    if missing:
        print(f"[openrouter] {len(missing)} records still missing; rerun the same command.")
        return 1
    final = [done[r["id"]] for r in data]
    json.dump(final, open(args.out_prefix + ".json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"[openrouter] wrote {args.out_prefix}.json ({len(final)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
