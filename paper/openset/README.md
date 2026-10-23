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

## Number verification log

Run `python src/verify_numbers.py paper/openset` before submission. Every math-mode
literal in these files is matched back to the result CSVs the section declares as its
source (narrative files — abstract, discussion, conclusion — are scoped to the union of
all declared sources). Output goes to `results/verify_numbers.csv`.

The check exists because of a specific failure. The abstract once reported a matched-rate
multiple of `1.25` where `1.17` belonged: `1.25` was a real number, in the right file, in
the wrong row — the 256-d variant rather than the 64-d one the deploy configuration uses.
Existence-checking would have passed it. So the report distinguishes:

| status | meaning | action |
|---|---|---|
| UNIQUE | appears in exactly one place in the declared sources | transcription confirmed |
| AMBIGUOUS | appears in several | confirm the quoted row is the intended row |
| UNSOURCED | in no result file | must be derived arithmetic, a count, or an error |

Last full pass:

| | count | adjudication |
|---|---|---|
| UNIQUE | 156 | confirmed |
| AMBIGUOUS | 86 | benign multiplicity — the same value repeated across `q` rows for one configuration |
| UNSOURCED | 39 | all accounted for, see below |

The unsourced values fall into four groups, none of them errors:

- **chance multiples** (`1.58`, `1.17`, `1.56`, `1.22`, `1.49`, `1.23`, `1.25`, `1.02`,
  `0.99`) — UCDR divided by the 0.2 chance rate, arithmetic on a stored number
- **H differences** (`10.9`, `20.1`) — differences of two stored H-scores
- **the chance rate itself** (`0.2000`) and the **cleaning class count** (`113`), which is
  a property of the dataset rather than a result
- **cross-section quotations** — values a section quotes from another section's source,
  which fall outside its own declared scope (`0.6616` in §8, `0.5176` and `1.6` in §8b)

Two gaps this pass found and closed: `exp30` printed the LOCO per-dimension weights and
the bootstrap point deltas without saving them, so `1.58`/`1.96` and `0.064`/`0.136` were
unverifiable from any file. Both are now written to
`results/exp30_temporal_detector.csv`. This is the third time a number reached a draft
from a run log rather than a stored artefact — the earlier two were Table 7 of the
closed-set paper and the matched-rate rows in `exp27`. Any new experiment should persist
every number it prints.

### Closed-set manuscript pass

`python src/verify_numbers.py paper` covers `../main.tex`. It is noisier — one file, more
prose-embedded values — but it found three things the open-set pass could not:

1. **A stale value.** The inline table in §6.8 still carried occupancy source-only MCC
   `0.6583` after the occupancy experiments were rerun to `0.6628`, so the same quantity
   appeared twice in one paper with two values. Corrected.
2. **A table sourced to a temp file.** `tab:diga` declared
   `% source: /tmp/diga_sim.py`, which no longer existed. Regenerated by
   `src/exp31_petta_diga_tables.py`; every value reproduces exactly.
3. **A labelling error.** `tab:petta` reported `median d`, `median gamma` and
   `median alpha_t`, but the values were means — the summed statistic averages `71.8`
   while its median is `48.28`. The numbers were right and the column headers were wrong.
   Headers corrected, and `exp31` now writes both statistics so the label cannot drift
   again.

Table 7 (`tab:digaimpl`), flagged earlier in this project as having no stored source, is
**closed**: all four macro-F1 values and all four confidence intervals match
`results/exp19_diga_impl.csv` to the printed precision.
