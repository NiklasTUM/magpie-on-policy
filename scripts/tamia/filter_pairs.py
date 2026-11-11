#!/usr/bin/env python
"""Deterministic quality filter for a Magpie instruction/response file.

Removes the defects that survive gen_ins.py's sanitizer. These are all detectable by
rule, so they are handled here rather than by an LLM judge -- notably "dangling
context", which a judge is unreliable at (asking a model to notice ABSENT material) but
which the response itself announces in plain language.

Reject reasons, measured on the 12k pilot (job 418152):
  dangling_context  ~6.5%  instruction refers to a snippet/image/table never provided,
                           so the response is the model asking for it. Worse than
                           useless as SFT data: it teaches the model to request input.
  repetition         6.3%  a phrase echoed close to itself, three or more times
  token_leakage      rare  literal <|im_end|>/<|user|> text in the instruction
  short / unterminated / empty_response / truncated_response

Re-applies the CURRENT sanitizer to raw_instruction, so a file generated before a
sanitizer fix still benefits from it.

  python scripts/tamia/filter_pairs.py IN.json [-o OUT.json] [--distance_file D.jsonl]
  python scripts/tamia/filter_pairs.py IN.json --sweep     # choose a threshold, write nothing
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "exp"))
import str_utils  # noqa: E402

# The response is where a dangling-context instruction reveals itself.
MISSING_CONTEXT = re.compile(
    r"didn'?t come through|didn'?t appear|please paste|please provide the|"
    r"I don'?t see (the|any)|you (mentioned|referenced).{0,40}(but|however).{0,60}(not|didn'?t)|"
    r"appears? to be (missing|empty)|seems? (to be )?(missing|empty)|"
    r"could you (please )?(share|paste|provide)|"
    # Added after a recall check on the pilot: the first pass left 62 cases (0.70%) that
    # used these wordings instead. Found by probing the FILTERED set with deliberately
    # different phrasings -- re-running the same regex only ever confirms itself.
    r"you have not provided|you haven'?t provided|have not been provided|"
    r"you did not (provide|include|attach)|I need the (actual|specific|real)|"
    r"no (text|code|data|image|table|passage) (was |has been )?provided|"
    r"please (supply|share) the (text|code|data|passage)", re.I)
# A response that is the model's reasoning instead of an answer. Seen from Kimi K3.1 Pro
# with thinking disabled (2/30 in the pilot): "I need respond to user ... Should think
# what best." -- several thousand characters that never reach an answer.
REASONING_LEAK = re.compile(
    r"^\s*(I need (to )?respond to|I need to (answer|produce|write) |We need to (answer|respond)|"
    r"The user (wants|asks|is asking)|User (wants|asks)|Should think|Let me think)", re.I)
LEAK_MARKERS = ("<think>", "</think>", "<|im_", "<|user|>", "<|assistant|>")
# Only the opening of the response matters: a long answer may legitimately say
# "please provide more detail" at the end without the instruction being broken.
RESPONSE_HEAD = 600


def echo_score(text, k=3, window=25):
    """Largest number of times any k-gram recurs within `window` words of itself.

    Targets the actual defect -- a phrase echoed close to itself, e.g. "local skin
    management, skin care advice, and local skin management" -- rather than global
    n-gram diversity. A global duplicated-shingle ratio was tried first and does not
    work here: at 5-grams it scored that exact example 0.000 (the echoed phrase is too
    short to form a duplicate 5-gram), and at bigrams it flagged 42.7% of the corpus,
    because long legitimate instructions naturally reuse bigrams.

    Calibrated on the 12k pilot: score 0 for 68.9%, 2 for 24.8%, >=3 for 6.3%.
    Score >=3 is reliably defective. Score 2 is genuinely MIXED -- a random sample of
    ten was roughly a third good, a third bad, a third borderline -- so it is
    deliberately left in and handed to the LLM quality pass instead of being guessed at
    here. See --max_echo.
    """
    words = text.lower().split()
    if len(words) < k * 2:
        return 0
    seen, worst = {}, 0
    for i in range(len(words) - k + 1):
        gram = " ".join(words[i:i + k])
        if gram in seen:
            near = [p for p in seen[gram] if i - p <= window]
            if near:
                worst = max(worst, len(near) + 1)
        seen.setdefault(gram, []).append(i)
    return worst


def classify(rec, args):
    """Return None if the pair is good, else the reject reason."""
    ins = (rec.get("instruction") or "").strip()
    res = (rec.get("response") or "").strip()
    if not ins:
        return "empty_instruction"
    if not res:
        return "empty_response"
    if any(m in ins for m in LEAK_MARKERS):
        return "token_leakage"
    if len(ins) < args.min_chars:
        return "short"
    if ins[-1] not in str_utils.TERMINAL_PUNCTUATION:
        return "unterminated"
    if echo_score(ins) >= args.max_echo:
        return "repetition"
    if REASONING_LEAK.search(res[:RESPONSE_HEAD]):
        return "reasoning_leak"
    if MISSING_CONTEXT.search(res[:RESPONSE_HEAD]):
        return "dangling_context"
    if args.drop_truncated and rec.get("finish_reason") == "length":
        return "truncated_response"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_file")
    ap.add_argument("-o", "--output_file")
    ap.add_argument("--max_echo", type=int, default=3,
                    help="Reject when a phrase echoes itself this many times. 3 is "
                         "calibrated: >=3 is reliably defective, 2 is a mixed bag best "
                         "left to the quality judge.")
    ap.add_argument("--min_chars", type=int, default=25)
    ap.add_argument("--drop_truncated", action="store_true",
                    help="Also drop responses cut off by max_tokens.")
    ap.add_argument("--distance_file",
                    help="A *_distance.jsonl from exp/gen_dis.py; drops near-duplicates.")
    ap.add_argument("--min_distance", type=float, default=0.05,
                    help="Minimum neighbour distance to keep (upstream's repetition cut).")
    ap.add_argument("--no_resanitize", action="store_true",
                    help="Do not re-apply the current sanitizer to raw_instruction.")
    ap.add_argument("--sweep", action="store_true",
                    help="Print a --max_repetition sweep and exit without writing.")
    args = ap.parse_args()

    data = json.load(open(args.input_file, encoding="utf-8"))
    total = len(data)

    # Re-sanitize from raw so this file picks up any sanitizer fix made since it was
    # generated. Records whose raw text is unavailable keep their existing instruction.
    resanitized = 0
    if not args.no_resanitize:
        for rec in data:
            raw = rec.get("raw_instruction")
            if not raw:
                continue
            # Sanitize with the rules for the model that generated the record; files
            # written before the field existed came from Qwen3.6.
            generator = ((rec.get("gen_input_configs") or {}).get("input_generator")
                         or "Qwen/Qwen3.6-27B")
            fixed, _ = str_utils.instruction_post_process(raw, generator,
                                                          rec.get("finish_reason"))
            if fixed.strip() and fixed != rec.get("instruction"):
                rec["instruction"] = fixed
                resanitized += 1

    if args.sweep:
        print(f"{total} records; --max_echo sweep (other filters held at defaults)\n")
        print(f"  {'max_echo':>8s} {'dropped':>8s} {'share':>7s}")
        for t in (2, 3, 4, 5):
            args.max_echo = t
            k = sum(1 for r in data if classify(r, args) == "repetition")
            print(f"  {t:8d} {k:8d} {k / total:7.1%}")
        return 0

    keep_ids = None
    if args.distance_file:
        keep_ids = set()
        dropped_dup = 0
        for line in open(args.distance_file, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("min_neighbor_distance", 1.0) >= args.min_distance:
                keep_ids.add(row.get("id"))
            else:
                dropped_dup += 1
        print(f"[filter] distance file: {dropped_dup} below {args.min_distance}")

    kept, reasons = [], Counter()
    for rec in data:
        reason = classify(rec, args)
        if reason is None and keep_ids is not None and rec.get("id") not in keep_ids:
            reason = "near_duplicate"
        if reason:
            reasons[reason] += 1
        else:
            kept.append(rec)

    out = args.output_file or args.input_file.replace(".json", "_filtered.json")
    json.dump(kept, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print(f"[filter] {args.input_file}")
    if resanitized:
        print(f"[filter] re-sanitized with the current rules: {resanitized}")
    print(f"[filter] kept {len(kept)}/{total} ({len(kept) / total:.1%})")
    for reason, count in reasons.most_common():
        print(f"[filter]   dropped {reason:20s} {count:6d} ({count / total:5.1%})")
    print(f"[filter] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
