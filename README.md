# OWMR paper reproducibility supplement

This supplement compiles the audited results available on the revised submission. It contains 312 main-manifest rows, 39 sensitivity rows, 25 nonstationary rows, and 63 selected paired comparisons (21 method-case groups). The main manifest has 307 validated cells and 5 N/A entries.

## Find a result

`index/reported_cells.csv` maps each manifest cell to its table grouping, selected checkpoint or policy, result source, and JSON config. It covers the main experiments and the supplemental sensitivity and nonstationary results. The manifest's K0--K4 correspond to Kaynov-1--Kaynov-5, while K5 and K6 correspond to PB-1 and PB-2. The index uses `tab:literature_results` for the main-text group and `tab:nonstationary_results` for the historical five-SKU table; the current appendix case-level literature table is `tab:literature_case_results`, and the current selected-SKU table is `tab:industrial_selected_skus`.

`configs/reported/` stores the manifest row verbatim, plus any independently saved evaluation protocol or Ray `params.json` that matches that checkpoint. `index/packaged_checkpoints.csv` maps 345 selected checkpoint references and SHA-256 hashes to 340 distinct model files. The archive contains selected models and their configuration records. For `configuration_evidence=manifest/checkpoint only`, a null action-bound or seed field denotes an unrecorded value.

Across the reported-cell index, evidence is: {'Ray params.json': 24, 'checkpoint-matched archived protocol': 86, 'formal saved evaluation protocol': 27, 'heuristic parameters/method in manifest': 90, 'manifest/checkpoint only': 144, 'not applicable': 5}.

The `action_bounds` object gives saved overrides or discrete branch counts; `resolved_action_bounds` gives effective target ranges from matching formal protocols. Source PPO branch counts describe discrete action dimensions; physical order quantities follow unit scaling and capacity/request caps. The source PPO `ray_params` records training and evaluation allocation settings.

## Reproduce a quick result

Check the integrity of the packaged files with Python 3:

```bash
python scripts/verify_bundle.py
```

The 2-1 actor/environment example uses the native dependencies recorded in `environment_versions.json`:

```bash
python scripts/quickstart_2_1.py
```

This evaluates the packaged RA-SAC seed-0 actor on 10 common-demand episodes. Formal results use 1,000 episodes. The source-PPO 2-1 example uses the separate Python 3.9/Ray environment and a portable checkpoint:

```bash
cd source_snapshot
python evaluate_source_ppo_checkpoint_common1000_20260813.py --specs examples/source_ppo_2_1_specs.json --output-dir examples/source_output --job source_ppo_2_1_smoke --episodes 10
```

## Results and statistical definitions

`training_budget_steps` records the configured budget, `observed_final_training_step` records the final step in trial progress or the native recorder, and `training_step` identifies the selected checkpoint.

`results/paired_path_differences.csv` contains precomputed comparisons against RA-SAC on the five representative cases. Each row records the comparator-minus-RA-SAC mean, median, paired-path SE, and fraction of paths on which RA-SAC has lower cost, using 1,000 common-demand paths identified by their demand-bank SHA-256 hash.

The index uses the paper's `Seq-SAC` name for the method labeled `SA-SAC` in the original manifest and historical table fragment. It displays the original `RA-SAC ineq.` label as `RA-SAC-Ineq`. Source PPO is a distinct benchmark.

The nonstationary manifest distinguishes retained/controllable cost from `full_mean_cost`; its original q0.9 and cold q0.7 protocols are separate rows. The paper's selected-SKU table uses the original cold q0.9 initialization, labels the five SKUs as Cases 1--5, and compares RA-SAC with Rolling-DA + proportional and Rolling-DA + Reallocation. `DA rolling` in the index denotes the proportional variant. The cold q0.7 and RA-SAC-Ineq rows are supplementary; the Reallocation variant has its own selected-SKU results. Sensitivity entries use the sensitivity-analysis grouping. Nonstationary costs cover 13 weeks; stationary benchmark means cover 1,000 periods. Industrial forecast inputs reside in the original private input-data directory.

## Source and scope

`source_snapshot/native/` contains the native environment, agents, training entry points, and evaluator. `source_snapshot/scm_rl-master/` contains source-PPO runtime code, dependency locks, reported Ray parameter files, and available scenario YAMLs; its formal evaluator is at the source-snapshot root. The native Penalty-AN-SAC and Lag-AN-SAC implementation follows Appendix D: their critics use original actor outputs, and penalties enter critic targets without being added directly to actor objectives. The packaged Penalty/Lag checkpoints were trained with `source_snapshot/as_trained/AgentSAC.py`. The Appendix D implementation and the historical checkpoint implementation are both included. AN-SAC uses the executed-action critic. The native PyTorch test for this distinction is:

```bash
cd source_snapshot/native
python -m unittest discover -s tests -p test_an_sac_appendix_protocol.py
```

The archived multi-run launchers retain their original absolute interpreter paths. Selected native actors are under `models/native/`; selected source-PPO checkpoints, metadata, and `params.json` are under short `portable_trials/` paths. Replay buffers and unselected training checkpoints are outside this selected-model package. `ENVIRONMENT.md` and `SHA256SUMS.txt` identify the packaged snapshot and dependencies. Historical LaTeX table fragments are in `results/paper_tables/`.
