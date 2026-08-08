# Investigation notebooks

Each notebook in this directory is one dated, self-contained investigation
against a specific kernel commit -- a sweep over configuration, an analysis
of behavior, or a comparison against a prior baseline. Notebooks are never
edited in place once committed; a follow-up investigation (e.g. against a
changed kernel) gets its own new `<date>-<slug>.ipynb` file so the series
stays a readable history of how the simulation's behavior has evolved,
without digging through git blame on a single mutated file.

Each notebook's own reproducibility header (its first code cells) records
the exact git commit, branch, and dirty state it was run against -- that's
the authoritative version record. The table below is just an index for
finding and skimming prior investigations; add a row whenever a new notebook
lands.

| Date | Notebook | Kernel commit | Finding |
|---|---|---|---|
| 2026-08-08 | [baseline-behavior-sweep](2026-08-08-baseline-behavior-sweep.ipynb) | `8d8e5bf` | With any truth-grounded channel present, the kernel converges to truth-aligned consensus in 100% of 290 runs across a broad OFAT + interaction grid; social parameters (bounded confidence, dynamic trust) are mechanically active but outcome-dormant in that regime, and only produce persistent disagreement once the truth-grounded channel is removed entirely. |
