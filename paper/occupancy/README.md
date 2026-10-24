# Occupancy section drafts

Each file is a `\subsection` ready to `\input`. They assume the closed-set macros
(`\macf`) and labels (`sec:res-leak`, `sec:res-feat`, `sec:res-gap`, `sec:res-mech`,
`sec:os-methods`).

| file | content | script | result file |
|---|---|---|---|
| `occ_01_contrast.tex` | two extracts, opposite answers | `exp34_occ_version_asymmetry.py` | `exp34_occ_version_asymmetry.csv` |
| `occ_02_features.tex` | sign-reversing features; penalty | `exp32_...py`, `exp33_...py` | `exp32_...csv`, `exp33_...csv` |
| `occ_03_methods.tex` | methods, traps, calibration | `exp35_occ_tta_raw.py` | `exp35_occ_tta_raw.csv` |

Raw extracts are the reference throughout. Anonymised-extract numbers appear only in the
final block of `tab:occmethods`, labelled as the prior extract for the
Section~\ref{sec:occ-contrast} contrast.
