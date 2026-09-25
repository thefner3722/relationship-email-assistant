"""Contact-level dev/test split. Run once; do not re-run after tuning starts.

Usage (from repo root):
    python eval/data/split.py

Reads  eval/data/processed/contacts.jsonl
Writes eval/data/splits.json   {"seed": 5980, "dev": [...], "test": [...]}

Method (per data card): stratify by internal/external, then by
message-count tertile (low/medium/high, computed from the actual
distribution of n_messages across all contacts) — 6 strata total.
Split each stratum 70/30 separately, then combine.
"""
import json, random
from collections import defaultdict
from pathlib import Path

SEED = 5980
TEST_FRAC = 0.30
SRC = Path("eval/data/processed/contacts.jsonl")
OUT = Path("eval/data/splits.json")


def tertile_bounds(values):
    """Two cut points splitting sorted values into three equal-size groups."""
    s = sorted(values)
    n = len(s)
    lo_cut = s[n // 3]
    hi_cut = s[(2 * n) // 3]
    return lo_cut, hi_cut


def volume_bucket(n, lo_cut, hi_cut):
    return "low" if n <= lo_cut else "high" if n > hi_cut else "medium"


def main():
    contacts = [json.loads(l) for l in open(SRC)]
    lo_cut, hi_cut = tertile_bounds([c["n_messages"] for c in contacts])

    strata = defaultdict(list)
    for c in contacts:
        internal = "internal" if c["internal"] else "external"
        vol = volume_bucket(c["n_messages"], lo_cut, hi_cut)
        strata[(internal, vol)].append(c["address"])

    rng = random.Random(SEED)
    dev, test = [], []
    for key in sorted(strata):
        addrs = sorted(strata[key])
        rng.shuffle(addrs)
        k = round(len(addrs) * TEST_FRAC)
        test += addrs[:k]
        dev += addrs[k:]

    OUT.write_text(json.dumps({"seed": SEED, "dev": sorted(dev), "test": sorted(test)}, indent=2))
    print(f"tertile cuts: low <= {lo_cut}, medium <= {hi_cut}, high > {hi_cut}")
    print(f"dev {len(dev)}  test {len(test)}")
    for key in sorted(strata):
        addrs = strata[key]
        n = len(addrs)
        t = sum(a in test for a in addrs)
        print(f"  {key[0]:>8}/{key[1]:<6}: {n:3d} contacts -> dev {n-t:3d}, test {t:3d}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
