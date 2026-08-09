# Investigation notebooks

Each notebook in this directory is one dated, self-contained investigation against a specific kernel commit -- a sweep over configuration, an analysis of behavior, or a comparison against a prior baseline. Notebooks are never edited in place once committed; a follow-up investigation (e.g. against a changed kernel) gets its own new `<date>-<slug>.ipynb` file so the series stays a readable history of how the simulation's behavior has evolved, without digging through git blame on a single mutated file.

## Starting a new investigation

Each notebook answers its own question and is free to structure itself however that requires -- which diagnostics to run, in what order, with what commentary. What's shared across investigations is the mechanical, non-narrative code: `sweep_utils.py` (right here in this directory, so `import sweep_utils` works directly -- Jupyter's kernel cwd is the notebook's own directory) holds it, kept separate from the simulation kernel in `src/simlab` since none of it is used by the CLI or by a single `execute_run()` call -- it exists purely to orchestrate and compare many runs for notebook analysis, sparing every investigation from reimplementing (or silently redoing slightly wrong) the same logic.

- `reproducibility_header(experiment_name)` -- call this first; see the rule below.
- `expand_ofat_scenarios(baseline_cfg, axes)` / `run_sweep(scenarios, seeds, steps)` -- build and execute a sweep.
- `between_scenario_variance_share(df, metric)` -- scenario-vs-seed variance decomposition.
- `predictability_probe(df, feature_cols, regression_targets, binary_target)` -- grouped-CV linear/random-forest probe. Requires the `notebook` uv dependency group (`uv run --group notebook ...`).
- `plot_variance_shares` / `plot_parameter_response` / `plot_predictability` -- matching plots.

See `2026-08-08-baseline-behavior-sweep.ipynb` for a worked example, or `sweep_utils.py` for full function docs. If a new investigation needs a pattern that isn't in there yet (e.g. Latin-hypercube sampling instead of OFAT), write it in the notebook first -- only promote it into `sweep_utils.py` once a second notebook actually needs the same thing, so the module doesn't accumulate speculative generality ahead of real reuse.

## The `kernel_commit_on_main` rule

Every notebook's reproducibility header checks whether the most recent commit touching `src/simlab` is reachable from `main`, and that `src/simlab` itself has no uncommitted changes, printing a loud warning if either fails. **Only commit a dated notebook to this directory once `kernel_commit_on_main` prints `True`.** A notebook attributed to kernel code still sitting on an unmerged branch references code that can be rebased, squashed, or force-pushed away -- silently invalidating what the notebook claims to have measured; an uncommitted edit under `src/simlab` is the same problem one step earlier, since `git log` only sees committed history and wouldn't reflect it at all. The check deliberately looks at the kernel code's own last commit, not `HEAD` (which also includes the notebook's own not-yet-merged commit and would make the check impossible to satisfy before its own PR merges).

## Index

Each notebook's own reproducibility header (its first code cells) records the exact git commit, branch, and dirty state it was run against -- that's the authoritative version record. The table below is just an index for finding and skimming prior investigations; add a row whenever a new notebook lands.

| Date | Notebook | Kernel commit | Finding |
|---|---|---|---|
| 2026-08-08 | [baseline-behavior-sweep](2026-08-08-baseline-behavior-sweep.ipynb) | `8d8e5bf` | With any truth-grounded channel present, the kernel converges to truth-aligned consensus in 100% of 240 runs across a broad OFAT + interaction grid; social parameters (bounded confidence, dynamic trust) are mechanically active but outcome-dormant in that regime, and only produce persistent disagreement once the truth-grounded channel is removed entirely. |
