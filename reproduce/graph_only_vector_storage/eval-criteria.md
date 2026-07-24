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

## Large-graph backfill benchmark

### Motivation

The main target is an initial backfill where common entities and relationships
are updated by many documents. A normal vector backend persists intermediate
versions after each document, while graph-only ingestion can embed only the
final graph state during one rebuild.

### Frozen workload

- Use the production `LightRAG.ainsert_custom_kg` storage path.
- Use `NetworkXStorage`, JSON KV stores, and `NanoVectorDBStorage`.
- Each document contains shared entities/relationships plus document-unique
  entities/relationships and one unique chunk.
- Flush after every document, matching the normal durable document commit.
- Baseline: persistent vector storage during every document insertion.
- Treatment: `NoopVectorDBStorage` during insertion, followed by one rebuild
  into `NanoVectorDBStorage`.
- Run the same generated documents and embedding delay for both modes.

### Hard correctness gates

- Final graph node and edge counts match between baseline and treatment.
- Final vector counts equal final graph nodes, graph edges, and text chunks.
- Treatment performs zero embedding calls during graph ingestion.
- Rebuild reports no failed batches or lost records.

### Quality and performance metrics

- End-to-end wall time: ingestion plus rebuild.
- Ingestion-only wall time.
- Embedding calls and embedded texts.
- Vector write amplification:
  `embedded texts / final vector records`.
- Final node, edge, and chunk counts.

### Pass criteria

- On the high-overlap workload, treatment reduces embedded texts by at least
  40% versus baseline.
- Treatment end-to-end time is no more than 80% of baseline under the frozen
  simulated embedding latency.
- No correctness gate fails.
