#!/usr/bin/env python
"""Report whether a Magpie run actually produced diverse instructions.

This is the check the whole category-rotation machinery exists to satisfy. Magpie's
diversity comes entirely from sampling: gen_ins.py feeds one fixed pre-query template
and draws n completions, so an un-nudged run just reproduces the model's prior over
first user messages -- which for a coding-focused model collapses onto code.

Reads the output of `unitag.py --tag_mission classification` and prints:

  1. the task_category histogram over the whole run;
  2. a control_task x task_category cross-tab, i.e. did each steered shard land in the
     category it was steered toward;
  3. the pass/fail verdict against the acceptance bar.

The 'free' shard is the control: it is the bare unsteered pre_query_template, sampled
in the same job, so its histogram is what the run would have looked like without any
steering at all.

  python scripts/tamia/diversity_report.py <..._ins_res_category.json>
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

# gen_ins.py control_task slug -> the primary_tag that unitag.py's classification
# prompt can emit for it (the 12 tags hardcoded in exp/str_utils.py).
SLUG_TO_TAG = {
    "information_seeking": "Information seeking",
    "reasoning": "Reasoning",
    "planning": "Planning",
    "editing": "Editing",
    "code": "Coding & Debugging",
    "math": "Math",
    "role_playing": "Role playing",
    "data_analysis": "Data analysis",
    "creative_writing": "Creative writing",
    "advice_seeking": "Advice seeking",
    "brainstorming": "Brainstorming",
}
CODE_TAG = "Coding & Debugging"
UNSTEERED = "free"

# Acceptance bar, fixed in advance so a run is not graded after the fact.
MAX_CODE_SHARE = 0.20
MIN_CATEGORIES = 8
MIN_CATEGORY_SHARE = 0.03
# Mean fraction of a steered shard's instructions that the tagger puts in the category
# the shard was steered toward. Chance is 1/11 ~= 9%, so this is well above chance but
# forgiving of tagger noise and of genuinely overlapping categories (reasoning vs math).
# Without this check, instructions spread evenly across categories for reasons unrelated
# to the nudge would satisfy the two share-based checks above and look like success.
MIN_ON_TARGET = 0.40
# Well-formedness. Diversity is not quality: an earlier run passed all three checks above
# on instructions that were mechanically truncated 16-token fragments, 83% of them cut
# mid-sentence and one cut mid-word. The tagger still assigned them plausible categories,
# so the category-based checks alone cannot catch it.
MIN_WELL_FORMED = 0.95
MIN_MEDIAN_CHARS = 40
TERMINAL_PUNCTUATION = '.?!。？！'


def bar(fraction, width=28):
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_file", help="A *_category.json produced by unitag.py")
    ap.add_argument("--max_code_share", type=float, default=MAX_CODE_SHARE)
    ap.add_argument("--min_categories", type=int, default=MIN_CATEGORIES)
    ap.add_argument("--min_category_share", type=float, default=MIN_CATEGORY_SHARE)
    ap.add_argument("--min_on_target", type=float, default=MIN_ON_TARGET)
    ap.add_argument("--min_well_formed", type=float, default=MIN_WELL_FORMED)
    ap.add_argument("--min_median_chars", type=int, default=MIN_MEDIAN_CHARS)
    args = ap.parse_args()

    data = json.load(open(args.input_file, encoding="utf-8"))
    total = len(data)
    if not total:
        sys.exit(f"{args.input_file} is empty.")

    tags = Counter(item.get("task_category") or "(untagged)" for item in data)
    by_shard = defaultdict(Counter)
    for item in data:
        by_shard[item.get("control_task") or "(unknown)"][
            item.get("task_category") or "(untagged)"] += 1

    empty_responses = sum(1 for item in data if not (item.get("response") or "").strip())
    untagged = tags.get("(untagged)", 0)

    print(f"{args.input_file}")
    print(f"{total} records | {untagged} untagged ({untagged / total:.1%}) | "
          f"{empty_responses} empty responses ({empty_responses / total:.1%})")

    print("\n=== task_category over the whole run ===")
    for tag, count in tags.most_common():
        share = count / total
        print(f"  {tag:22s} {count:7d}  {share:6.1%}  {bar(share)}")

    print("\n=== control_task -> task_category (did the nudge take?) ===")
    print(f"  {'shard':22s} {'n':>7s}  {'on-target':>9s}   top categories")
    on_target_rates = {}
    for shard in sorted(by_shard, key=lambda s: (s == UNSTEERED, s)):
        counts = by_shard[shard]
        n = sum(counts.values())
        want = SLUG_TO_TAG.get(shard)
        hit = counts.get(want, 0) if want else None
        rate = (hit / n) if (want and n) else None
        if rate is not None:
            on_target_rates[shard] = rate
        top = ", ".join(f"{t} {c / n:.0%}" for t, c in counts.most_common(3))
        rate_s = f"{rate:8.1%}" if rate is not None else "       -"
        print(f"  {shard:22s} {n:7d}  {rate_s}   {top}")

    # The comparison that answers the original complaint.
    if UNSTEERED in by_shard:
        free = by_shard[UNSTEERED]
        free_n = sum(free.values())
        free_code = free.get(CODE_TAG, 0) / free_n if free_n else 0.0
        steered = Counter()
        for shard, counts in by_shard.items():
            if shard != UNSTEERED:
                steered.update(counts)
        steered_n = sum(steered.values())
        steered_code = steered.get(CODE_TAG, 0) / steered_n if steered_n else 0.0
        print(f"\n=== steered vs unsteered ({CODE_TAG} share) ===")
        print(f"  unsteered ('{UNSTEERED}') {free_code:6.1%}   (n={free_n})")
        print(f"  steered shards          {steered_code:6.1%}   (n={steered_n})")
        if on_target_rates:
            mean_rate = sum(on_target_rates.values()) / len(on_target_rates)
            print(f"  mean on-target rate across steered shards: {mean_rate:.1%} "
                  f"(chance is {1.0 / len(SLUG_TO_TAG):.0%})")
    instructions = [(item.get("instruction") or "").strip() for item in data]
    non_empty = [s for s in instructions if s]
    well_formed = [s for s in non_empty if s[-1] in TERMINAL_PUNCTUATION]
    lengths = sorted(len(s) for s in non_empty) or [0]
    median_chars = lengths[len(lengths) // 2]
    leaked = [s for s in non_empty if "<think>" in s or "</think>" in s or "<|im_" in s]
    print("\n=== instruction well-formedness ===")
    print(f"  non-empty                 {len(non_empty)}/{total} ({len(non_empty) / total:.1%})")
    print(f"  ends on sentence boundary {len(well_formed)}/{max(len(non_empty), 1)} "
          f"({len(well_formed) / max(len(non_empty), 1):.1%})")
    print(f"  chat/think token leakage  {len(leaked)}")
    print(f"  length chars: p10={lengths[len(lengths) // 10]} median={median_chars} "
          f"p90={lengths[9 * len(lengths) // 10]}")

    print("\n=== acceptance bar ===")
    code_share = tags.get(CODE_TAG, 0) / total
    well_represented = [t for t, c in tags.items()
                        if t != "(untagged)" and c / total >= args.min_category_share]
    wf_rate = len(well_formed) / max(len(non_empty), 1)
    checks = [
        # Well-formedness first: if the instructions are fragments, the category numbers
        # below are describing junk and nothing else on this list means anything.
        (f"instructions ending on a sentence boundary >= {args.min_well_formed:.0%}",
         wf_rate >= args.min_well_formed, f"{wf_rate:.1%}"),
        (f"median instruction length >= {args.min_median_chars} chars",
         median_chars >= args.min_median_chars, f"{median_chars}"),
        ("no chat/think token leakage", not leaked, f"{len(leaked)}"),
        (f"{CODE_TAG} <= {args.max_code_share:.0%}",
         code_share <= args.max_code_share, f"{code_share:.1%}"),
        (f">= {args.min_categories} categories at >= {args.min_category_share:.0%}",
         len(well_represented) >= args.min_categories, f"{len(well_represented)}"),
    ]
    if on_target_rates:
        # Without this, an even spread produced for reasons unrelated to the nudge
        # would satisfy the two checks above and read as success.
        mean_rate = sum(on_target_rates.values()) / len(on_target_rates)
        chance = 1.0 / len(SLUG_TO_TAG)
        checks.append((
            f"mean on-target rate >= {args.min_on_target:.0%} (chance {chance:.0%})",
            mean_rate >= args.min_on_target, f"{mean_rate:.1%}"))
    else:
        print("  [WARN] no steered shards found -- was --mix_control_tasks used? "
              "The steering checks below cannot run.")
    failed = 0
    for label, passed, actual in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label:44s} actual: {actual}")
        failed += not passed

    # Non-zero exit so this can gate a pipeline, but the report above is always printed.
    if failed:
        print(f"\n{failed} check(s) failed. See the escalation options in the plan before "
              f"scaling this run up.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
