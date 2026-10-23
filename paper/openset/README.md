# Open-set extension — section drafts

Each file is a `\subsection` ready to `\input`, in outline order. They assume the
closed-set paper's macros (`\macf`) and labels (`sec:res-gap`, `sec:res-mech`,
`sec:res-feat`, `sec:method-tiers`, `tab:bnchannel`).

| file | outline § | source script | result file |
|---|---|---|---|
| `00_deploy.tex` | deploy box | exp26/exp27 | `exp26_unknown_scores.csv` |
| `02_penalty.tex` | §2 | `exp24_openset_gap.py` | `exp24_openset_gap.csv` |
| `04_methods_trap.tex` | §3+§4 merged | `exp25_openset_tta.py` | `exp25_openset_tta.csv` |
| `05_detectors.tex` | §5 | `exp26_unknown_scores.py`, `exp28_auroc_ci.py` | `exp28_auroc_ci.csv` |
| `06_mahalanobis.tex` | §6 | `exp27_proto_depth.py` | `exp27_proto_depth.csv` |
| `07_threshold.tex` | §7 | `exp27_proto_depth.py` | `exp27_proto_depth.csv` |
| `08_depth.tex` | §8 | `exp27_proto_depth.py` | `exp27_proto_depth.csv` |
| `lead_violations.tex` | §9 | — | `ispc-lab/LEAD` at HEAD |
| `10_discussion.tex` | §10 | — | — |

§1 (motivation and protocol) is not drafted: it recaps the closed-set paper and states the
Tier-1 definition, both of which already exist in `../main.tex`.

Every number in these files was read from the result CSVs, not transcribed from a run log.
Diagnostic figures that use target labels (the matched-rejection-rate table, the oracle-tau
bound) are labelled as such in the captions.
