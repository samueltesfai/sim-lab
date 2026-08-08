# Investigation notebooks

Each notebook in this directory is one dated, self-contained investigation
against a specific kernel commit -- a sweep over configuration, an analysis
of behavior, or a comparison against a prior baseline. Notebooks are never
edited in place once committed; a follow-up investigation (e.g. against a
changed kernel) gets its own new `<date>-<slug>.ipynb` file so the series
stays a readable history of how the simulation's behavior has evolved,
without digging through git blame on a single mutated file.

## Starting a new investigation

Copy `_template.ipynb` to `<date>-<slug>.ipynb`, resolve every `[[FILL: ...]]`
marker (code cells use `# EDIT:` comments for the same purpose), run it end-
to-end, then add a row to the index below. The static text around those
markers is intentionally the same in every notebook in this series -- it's
generated from one shared source, not hand-copied, so it can't drift between
investigations. `_template.ipynb` itself stays blank (no outputs, no
`[[FILL:]]` markers resolved) so it's always a clean starting point; a dated
notebook may also add its own extra sections beyond the template (e.g. an
interaction grid or mechanism-activation diagnostics) where the investigation
calls for them.

## The `kernel_commit_on_main` rule

Every notebook's reproducibility header checks whether the most recent
commit touching `src/simlab` is reachable from `main`, and prints a loud
warning if not. **Only commit a dated notebook to this directory once that
check prints `True`.** A notebook attributed to kernel code still sitting on
an unmerged branch references code that can be rebased, squashed, or
force-pushed away -- silently invalidating what the notebook claims to have
measured. The check deliberately looks at the kernel code's own last commit,
not `HEAD` (which also includes the notebook's own not-yet-merged commit and
would make the check impossible to satisfy before its own PR merges).

## Index

Each notebook's own reproducibility header (its first code cells) records
the exact git commit, branch, and dirty state it was run against -- that's
the authoritative version record. The table below is just an index for
finding and skimming prior investigations; add a row whenever a new notebook
lands.

| Date | Notebook | Kernel commit | Finding |
|---|---|---|---|
| 2026-08-08 | [baseline-behavior-sweep](2026-08-08-baseline-behavior-sweep.ipynb) | `8d8e5bf` | With any truth-grounded channel present, the kernel converges to truth-aligned consensus in 100% of 290 runs across a broad OFAT + interaction grid; social parameters (bounded confidence, dynamic trust) are mechanically active but outcome-dormant in that regime, and only produce persistent disagreement once the truth-grounded channel is removed entirely. |
