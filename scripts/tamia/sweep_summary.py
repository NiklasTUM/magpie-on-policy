#!/usr/bin/env python
"""Compare instruction-sampling variants from magpie_glm53_sweep.sbatch.

Takes the per-variant *_ins.json files plus the Gemma-judged merge of their usable
records, and prints one row per variant:

  usable   instruction survives the sanitizer (non-empty)
  length   generation hit max_tokens (a runaway, not a user turn that ended)
  junk     regex for the failures seen in smoke job 680693: assistant-side reasoning
           ("The user wants...", "Let me..."), <tool_call> text, CJK code-switching
  good+    Gemma input_quality in {good, excellent}, as a share of ALL generations,
           so a variant cannot look good by having the sanitizer drop most of it

  python scripts/tamia/sweep_summary.py SWEEP_DIR
"""
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

JUNK = re.compile(r"^\s*(The user|Let me|Okay,|Hmm|We need to|I need to)|<tool_call>|"
                  r"[一-鿿]")
LEVELS = ["very poor", "poor", "average", "good", "excellent"]


def main(sweep_dir):
    gens = {}
    for f in sorted(glob.glob(os.path.join(sweep_dir, "*", "*_ins.json"))):
        variant = os.path.basename(os.path.dirname(f))
        gens[variant] = json.load(open(f, encoding="utf-8"))

    judged = defaultdict(Counter)
    qfile = os.path.join(sweep_dir, "sweep_usable_quality.json")
    if os.path.exists(qfile):
        for r in json.load(open(qfile, encoding="utf-8")):
            judged[r["sweep_variant"]][r.get("input_quality")] += 1
    else:
        print(f"(no {qfile}; Gemma columns empty)")

    print(f"{'variant':34s} {'n':>4s} {'usable':>7s} {'length':>7s} {'junk':>6s} "
          f"{'good+':>6s}   Gemma very poor/poor/average/good/excellent")
    for variant, recs in gens.items():
        n = len(recs)
        usable = [r for r in recs if r["instruction"].strip()]
        length = sum(r.get("finish_reason") == "length" for r in recs)
        junk = sum(bool(JUNK.search(r.get("raw_instruction") or r["instruction"]))
                   for r in recs)
        q = judged[variant]
        goodp = q["good"] + q["excellent"]
        dist = "/".join(str(q[l]) for l in LEVELS)
        print(f"{variant:34s} {n:4d} {len(usable) / n:7.1%} {length / n:7.1%} "
              f"{junk / n:6.1%} {goodp / n:6.1%}   {dist}")

    for variant, recs in gens.items():
        good = [r for r in recs if r["instruction"].strip()]
        print(f"\n=== {variant}: first 3 usable instructions")
        for r in good[:3]:
            print(f"  [{r['control_task']}] {r['instruction'][:240]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
