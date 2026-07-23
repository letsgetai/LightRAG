# Graph-only vector storage evaluation report

## Result

`NoopVectorDBStorage` passed every correctness and performance gate defined before implementation. With three vector namespaces and 256 records per namespace, the final same-environment NanoVectorDB baseline took a median `0.0850594286 s` and invoked the simulated embedding function 24 times per repeat. The no-op backend took a median `0.0000554230 s` and invoked the embedding function zero times.

## Frozen setup

| Setting | Value |
|---|---:|
| Namespaces | 3 (`entities`, `relationships`, `chunks`) |
| Records per namespace | 256 |
| Total accepted records | 768 |
| Embedding batch size | 32 |
| Simulated embedding latency | 50 ms per batch |
| Repeats | 5 |
| Final implementation commit | `a56b4dc` |

Command:

```bash
uv run python reproduce/graph_only_vector_storage/benchmark.py \
  --mode <nano|noop> \
  --records 256 \
  --batch-size 32 \
  --embedding-delay-ms 50 \
  --repeats 5 \
  --output <result.json>
```

## Performance comparison

| Metric | NanoVectorDB | NoopVectorDBStorage | Result |
|---|---:|---:|---:|
| Median ingestion time | `0.0850594286 s` | `0.0000554230 s` | `1534.73×` lower simulated vector-phase time |
| Embedding calls per repeat | 24 | 0 | 24 calls removed |
| Embedded texts per repeat | 768 | 0 | 768 embeddings removed |
| Materialized vector records | 768 | 0 | Expected graph-only behavior |
| Noop/baseline ratio | — | `0.00065158` | Pass, threshold `≤ 0.10` |

The benchmark isolates vector storage work with deterministic simulated latency. The speedup value describes this isolated phase; an end-to-end ingestion run will also include parsing, LLM extraction, graph writes, KV writes, and scheduling overhead.

## Correctness checks

| Check | Result |
|---|---|
| Storage is registered through the normal vector backend registry | Pass |
| Upsert, delete, lifecycle, and inspection methods call embedding zero times | Pass |
| Query raises an actionable `lightrag-rebuild-vdb` error | Pass |
| `ainsert_custom_kg` persists two graph nodes and one edge | Pass |
| LightRAG initializes with `embedding_func=None` in graph-only mode | Pass |
| Focused no-op tests | `4 passed` |
| Graph insertion, keyed-lock, and rebuild regression subset | `62 passed, 4 skipped, 2 deselected` |
| Ruff check | Pass |
| Ruff format check | Pass |

Two PostgreSQL ordering tests were excluded from the clean regression run because the standard `test` extra does not install the optional `asyncpg` and `pgvector` dependencies. An earlier unfiltered run produced `62 passed, 4 skipped, 2 failed`, with both failures occurring at `import asyncpg` before the tested code executed.

## Pre-commit status

`pre-commit` could not finish initialization because this host timed out while cloning `https://github.com/pre-commit/pre-commit-hooks/`. The repository-equivalent Ruff checks and `git diff --check` passed. Pre-commit remains a pull-request gate and must be retried before marking the PR ready for maintainer review.

## Promotion decision

Promote the code and tests to the pull-request branch. Keep the benchmark harness and raw experiment results on the experiment branch so the upstream PR remains focused on the storage backend, tests, and user documentation.
