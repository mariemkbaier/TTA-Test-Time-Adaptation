# Cross-reference rewrites (assembly point 1)

Ten `\ref`s pointed at labels owned by the closed-set manuscript. A standalone paper
cannot reference into another, so each became prose plus `\cite{kbaier_closedset}`.
These are the only places where assembly changed the text rather than moving it.

| file | was | now |
|---|---|---|
| `02_penalty.tex` | "the measurement of Section~\ref{sec:res-gap}" | "the measurement of the closed-set study \cite{kbaier_closedset}" |
| `02_penalty.tex` | "Closed-set rows reproduce Section~\ref{sec:res-gap}" | "Closed-set rows reproduce \cite{kbaier_closedset}" |
| `04_methods_trap.tex` | "the four adaptation methods of Section~\ref{sec:res-mech}" | "the four adaptation methods of the closed-set study \cite{kbaier_closedset}" |
| `04_methods_trap.tex` | "the ordering is the one Section~\ref{sec:res-mech} predicts" | "the ordering is the one the closed-set study predicts" |
| `04_methods_trap.tex` | "as Table~\ref{tab:bnchannel} shows" | "as the closed-set study shows" |
| `05_detectors.tex` | "the per-sensor feature result of Section~\ref{sec:res-feat}" | "the per-sensor feature result of \cite{kbaier_closedset}" |
| `07_threshold.tex` | "harmful in Section~\ref{sec:res-mech}" | "two separate findings in the closed-set study \cite{kbaier_closedset}: ... batch-norm ... and ... per-sensor features" — the two findings are named as distinct |
| `07_threshold.tex` | "anti-informative in Section~\ref{sec:res-feat}" | "anti-informative in the closed-set study \cite{kbaier_closedset}" |
| `10_discussion.tex` | "a consistent reading alongside Section~\ref{sec:res-mech}" | "a consistent reading alongside the closed-set study \cite{kbaier_closedset}" |
| `lead_violations.tex` | "the Tier-1 definition of Section~\ref{sec:method-tiers}" | "the Tier-1 definition of Section~\ref{sec:os-setup}" (now an internal ref) |

The last one is different in kind: the Tier-1 definition is restated in full in §1 of this
paper, so that reference became internal rather than a citation.

The `07_threshold.tex` entry was initially the only one where a clause was dropped rather
than reworded, which let the sentence imply the batch-norm result and the per-sensor
result were the same finding. It has been rewritten to name them as two findings sharing
one cause, with the citation kept.
