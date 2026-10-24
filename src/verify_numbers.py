"""
Number verification pass.

Every numeric claim in the manuscript is matched back to the result file that produced
it. The point is not to confirm a number exists somewhere in the results -- the error
this check was built for (1.25 where 1.17 belonged) was a number that existed in the
right CSV, in the wrong row. So the report distinguishes three outcomes:

  UNIQUE     the value appears in exactly one place in the results. Transcription is
             almost certainly right.
  AMBIGUOUS  it appears in several places. The value is real but the manuscript could be
             quoting the wrong one, which is exactly how 1.25 survived three sentences.
             These need a human to confirm the quoted row is the intended row.
  UNSOURCED  no result file contains it. Either it is derived arithmetic, a dimension, a
             count, or an error.

Usage:  python src/verify_numbers.py [paper/openset | paper]
"""
import os, re, sys, glob
import pandas as pd, numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
# values that are structural rather than measured
# Structural constants rather than measured results: window lengths, layer widths,
# sample counts, percentile settings. "20" is the window length and "19" the number of
# timesteps two adjacent stride-1 windows therefore share -- the same design parameter
# stated in the protocol section, not a figure quoted from a result table.
EXEMPT = {"0.5", "1.0", "0.0", "2.0", "5.0", "13", "19", "20", "64", "128", "256",
          "3412", "5031", "40", "30", "100", "95", "2000", "50", "5", "1", "2", "3", "4"}


def load_results():
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, "*.csv"))):
        if os.path.basename(f) == "verify_numbers.csv":
            continue                      # never check a number against this report
        try:
            out[os.path.basename(f)] = pd.read_csv(f)
        except Exception:
            pass
    return out


def find(val, text, tabs):
    """Where does this value appear in the result files, to the printed precision?"""
    dec = len(text.split(".")[1]) if "." in text else 0
    hits = []
    for name, df in tabs.items():
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.isna().all():
                continue
            m = np.isclose(np.round(s.values.astype(float), dec), val,
                           atol=10 ** (-dec) / 2, equal_nan=False)
            # percentages are stored as fractions
            m2 = np.isclose(np.round(s.values.astype(float) * 100, dec), val,
                            atol=10 ** (-dec) / 2, equal_nan=False)
            for i in np.where(m | m2)[0]:
                key = " ".join(str(df.iloc[i][c]) for c in df.columns[:3]
                               if not pd.api.types.is_numeric_dtype(df[c]))
                hits.append(f"{name}[{key.strip()[:46]}].{col}")
    return hits


if __name__ == "__main__":
    where = sys.argv[1] if len(sys.argv) > 1 else "paper/openset"
    tabs = load_results()
    files = sorted(glob.glob(os.path.join(ROOT, where, "*.tex")))
    # the union of every source this manuscript declares; narrative files (abstract,
    # discussion, conclusion) quote other sections and are scoped to that union rather
    # than to the whole results directory
    srcpat0 = re.compile(r"%\s*source:.*?((?:results/)?\w+\.csv)")
    OWN = set()
    for f in files:
        OWN |= {os.path.basename(m) for m in srcpat0.findall(open(f).read())}
    # only math-mode literals: this is how every measured value is written in these
    # files, and it excludes column widths, line-number citations and \ref targets.
    # An optional leading "+" is accepted, because every positive delta in these papers
    # is written that way, and \mathbf{} is stripped first, because the emphasised cells
    # in a table are exactly the values a reader checks. Omitting either leaves a large
    # and non-random set of numbers unexamined.
    pat = re.compile(r"\$([+-]?\d+(?:\.\d+)?)(?:\\%)?\$")
    mathbf = re.compile(r"\\mathbf\{\s*([^{}]*?)\s*\}")
    # Confidence intervals are written as a single math expression, $[a,b]$, so neither
    # bound is a bare literal and both would otherwise go unchecked. Rewrite each pair
    # into two separate literals before scanning, since the bounds are transcribed from
    # a result file like any other number and can be mistyped like any other number.
    interval = re.compile(r"\$\[\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\]\$")
    srcpat = re.compile(r"%\s*source:.*?((?:results/)?\w+\.csv)")
    rows, nuniq, namb, nuns = [], 0, 0, 0
    print("=" * 100)
    print(f"NUMBER VERIFICATION — {where}   against {len(tabs)} result files")
    print("=" * 100)
    for f in files:
        base = os.path.basename(f)
        if base.startswith("_"):
            continue
        body = open(f).read()
        # the CSVs this section declares as its sources; fall back to all of them
        cited = {os.path.basename(m) for m in srcpat.findall(body)}
        use = cited or OWN
        scope = {k: v for k, v in tabs.items() if k in use} or tabs
        for ln, raw in enumerate(body.splitlines(), 1):
            line = interval.sub(lambda m: f"${m.group(1)}$ ${m.group(2)}$", raw)
            line = mathbf.sub(r"\1", line)
            if line.lstrip().startswith("%"):
                continue
            for m in pat.finditer(line):
                t = m.group(1).lstrip("+")
                if t in EXEMPT or t.lstrip("-").replace(".", "").strip("0") == "":
                    continue
                try:
                    v = float(t)
                except ValueError:
                    continue
                hits = find(v, t, scope)
                st = "UNIQUE" if len(hits) == 1 else ("UNSOURCED" if not hits else "AMBIGUOUS")
                nuniq += st == "UNIQUE"; namb += st == "AMBIGUOUS"; nuns += st == "UNSOURCED"
                rows.append(dict(file=base, line=ln, value=t, status=st,
                                 scope=("declared" if cited else "manuscript-wide"),
                                 n_matches=len(hits), where="; ".join(hits[:3])))
    df = pd.DataFrame(rows)
    for st in ("UNSOURCED", "AMBIGUOUS", "UNIQUE"):
        sub = df[df.status == st]
        print(f"\n--- {st}  ({len(sub)}) ---")
        if st == "UNIQUE":
            print("  (transcription confirmed against a single result row)")
            continue
        for _, r in sub.iterrows():
            print(f"  {r['file']}:{r['line']:<4} {r['value']:<10} "
                  f"{r['n_matches']} match(es)  {r['where'][:70]}")
    out = os.path.join(RES, "verify_numbers.csv")
    df.to_csv(out, index=False)
    print(f"\n  UNIQUE {nuniq}   AMBIGUOUS {namb}   UNSOURCED {nuns}")
    print(f"-> {out}")
