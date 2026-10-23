# Open-set extension — section drafts

Each file is a `\subsection` ready to `\input`, in outline order. They assume the
closed-set paper's macros (`\macf`) and labels (`sec:res-gap`, `sec:res-mech`,
`sec:res-feat`, `sec:method-tiers`, `tab:bnchannel`).

| file | outline § | source script | result file |
|---|---|---|---|
| `00_deploy.tex` | deploy box | exp26/exp27 | `exp26_unknown_scores.csv` |
| `99_frontmatter.tex` | title, abstract, conclusion | — | — |
| `01_setup.tex` | §1 | — | — |
| `02_penalty.tex` | §2 | `exp24_openset_gap.py` | `exp24_openset_gap.csv` |
| `04_methods_trap.tex` | §3+§4 merged | `exp25_openset_tta.py` | `exp25_openset_tta.csv` |
| `05_detectors.tex` | §5 | `exp26_unknown_scores.py`, `exp28_auroc_ci.py` | `exp28_auroc_ci.csv` |
| `06_mahalanobis.tex` | §6 | `exp27_proto_depth.py` | `exp27_proto_depth.csv` |
| `07_threshold.tex` | §7 | `exp27_proto_depth.py` | `exp27_proto_depth.csv` |
| `08_depth.tex` | §8 | `exp27_proto_depth.py`, `exp29_feature_limit.py` | `exp27_proto_depth.csv`, `exp29_feature_limit.csv` |
| `08b_temporal.tex` | §8b | `exp30_temporal_detector.py` | `exp30_temporal_detector.csv` |
| `lead_violations.tex` | §9 | — | `ispc-lab/LEAD` at HEAD |
| `10_discussion.tex` | §10 | — | — |

§1 is drafted standalone: it restates the 0.6/21.3 asymmetry and the full Tier-1
definition without relying on `../main.tex` for context, so the section can be read on its
own or split into a separate paper.

Every number in these files was read from the result CSVs, not transcribed from a run log.
Diagnostic figures that use target labels (the matched-rejection-rate table, the oracle-tau
bound) are labelled as such in the captions.

## Standing as its own paper

`99_frontmatter.tex` carries a standalone title, abstract and conclusion. The closed-set
study is cited as prior work (`\cite{kbaier_closedset}`), not assumed, and §1 was written
to stand without it. Section order for the standalone build:

```
99_frontmatter (title + abstract)  →  01_setup  →  00_deploy  →  02_penalty
→ 04_methods_trap → 05_detectors → 06_mahalanobis → 07_threshold
→ 08_depth → 08b_temporal → lead_violations → 10_discussion
→ 99_frontmatter (conclusion)
```

`00_deploy` stays between §1 and §2 by decision: the recommendation is stated before the
negative results that justify it.

## Compile check

All twelve files were compiled together against `cas-dc` with the main paper's preamble
(`natbib`, `amsmath`, `booktabs`, `multirow`, and the `\macf` macro). Eight pages as a standalone paper, zero undefined references. The driver used for the check is not kept; recreate it by
`\input`-ing the files in the order of the table above and defining `\macf` plus
`\phantomsection\label{}` stubs for the five labels owned by `../main.tex`:
`sec:res-gap`, `sec:res-mech`, `sec:res-feat`, `sec:method-tiers`, `tab:bnchannel`.

## Conventions

Any figure computed with target labels is marked DIAGNOSTIC in its caption and in the
generating script: the matched-rejection-rate table (§6), the supervised reference (§8),
the oracle-threshold bound (§5 source), and the [ph] variants in §8b. The convention is
stated in §1 so a reader meets it before the first instance.
