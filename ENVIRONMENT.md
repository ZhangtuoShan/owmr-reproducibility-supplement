# Environment and source provenance

This file snapshot was assembled on 2026-10-01. `SHA256SUMS.txt` identifies every included file by content hash.

`source_snapshot/native/elegantrl/agents/AgentSAC.py` implements the Appendix D Penalty-AN-SAC and Lag-AN-SAC original-action critic and critic-target penalty protocol. The packaged Penalty/Lag checkpoints were trained with `source_snapshot/as_trained/AgentSAC.py`. Both implementations are included.

The native project and Kaynov-source PPO use separate interpreters. `environment_versions.json` records the interpreter and installed package versions observed on the packaging host. The source-PPO `pyproject.toml` and `poetry.lock` record its dependency specifications. The working Python 3.9 environment uses Ray 1.x, protobuf 3.19.6, and tensorboardX 2.6.5.

The quick example evaluates a selected checkpoint. The package includes selected Ray checkpoints, configuration records, and evaluation programs. Training histories and other checkpoints remain in the full project archive.
