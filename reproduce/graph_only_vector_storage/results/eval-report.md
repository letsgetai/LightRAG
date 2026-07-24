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

## Large-graph initial backfill

The original microbenchmark proves that a no-op backend removes the vector
phase, but it does not model the main large-scale benefit: common entities and
relationships are embedded again after every durable document commit. The
large-graph benchmark therefore inserts multiple documents with overlapping
entities and relationships through `LightRAG.ainsert_custom_kg`.

Baseline uses `NanoVectorDBStorage` for every document. Treatment uses
`NoopVectorDBStorage` during graph construction, then rebuilds the final entity,
relationship, and chunk indexes once with `NanoVectorDBStorage`.

### Results

| Workload | Final graph | Baseline total | Deferred total | Time reduction | Embedded texts reduction | Write amplification |
|---|---:|---:|---:|---:|---:|---:|
| 20 documents | 1,100 nodes + 1,100 edges | 1.233 s | 1.093 s | 11.3% | 63.1% | 2.71× → 1.00× |
| 100 documents, median of 3 | 10,200 nodes + 10,200 edges | 42.646 s | 34.380 s | 19.4% | 65.9% | 2.93× → 1.00× |
| 200 documents | 20,400 nodes + 20,400 edges | 169.020 s | 131.930 s | 21.9% | 79.5% | 4.88× → 1.00× |

The 200-document workload embedded `200,200` texts in the baseline and
`41,000` texts in the deferred workflow. Embedding calls fell from `3,400` to
`650`. Final graph and vector counts matched, and all rebuild batches completed
without errors.

The 100-document repeated run was stable:

- Baseline total seconds: `43.128`, `42.646`, `42.448`.
- Deferred total seconds: `34.294`, `34.568`, `34.380`.
- Deferred rebuild median: `1.737 s`.
- Every run produced 10,200 nodes, 10,200 edges, and 100 chunks.

### Scalability boundary

Deferred rebuild used more peak memory because the current rebuild functions
materialize graph records and prepared payloads before batched vector writes.

| Workload | Baseline peak RSS | Deferred peak RSS |
|---|---:|---:|
| 100 documents, median of 3 | 275 MiB | 359 MiB |
| 200 documents | 303 MiB | 471 MiB |

The PR therefore demonstrates a meaningful large-backfill time and write-load
benefit, especially when entities and relationships repeat across documents.
It should not claim unbounded large-graph scalability until rebuild sources and
payload preparation can be streamed or paginated.
