# Graph-only vector storage benchmark criteria

## Hypothesis

A built-in no-op vector storage lets LightRAG complete graph-first ingestion without invoking the embedding function, while preserving the existing storage contract and allowing vector indexes to be rebuilt later with `lightrag-rebuild-vdb`.

## Frozen benchmark

- Python: current development environment.
- Input: three vector namespaces (`entities`, `relationships`, `chunks`).
- Records per namespace: `256`.
- Embedding batch size: `32`.
- Simulated embedding latency: `50 ms` per batch.
- Repeats: `5`.
- Baseline: `NanoVectorDBStorage` from upstream `main`.
- Treatment: proposed built-in `NoopVectorDBStorage`.

## Hard correctness gates

- Baseline materializes all `768` records.
- Baseline embeds all `768` records.
- Treatment calls the embedding function `0` times.
- Treatment accepts upsert, delete, lifecycle, and inspection calls without persistence.
- Treatment `query` raises an actionable error that tells the user to rebuild vector indexes.
- Relevant offline unit tests pass.
- Ruff and pre-commit checks pass for changed files.

## Performance gate

- Treatment median ingestion time is at most `10%` of the same-run baseline median under the frozen simulated embedding latency.
- Repeated baseline median differs by at most `20%` from the original baseline, or the report explains the host-load difference.

## Promotion rule

Promote the implementation to the pull-request branch only when every correctness gate passes and the treatment meets the performance gate.
