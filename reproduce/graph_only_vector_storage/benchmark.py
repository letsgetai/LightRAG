#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc


class DelayedEmbedding:
    def __init__(self, dimension: int, delay_seconds: float) -> None:
        self.dimension = dimension
        self.delay_seconds = delay_seconds
        self.call_count = 0
        self.text_count = 0

    async def __call__(self, texts: list[str], **_: object) -> np.ndarray:
        self.call_count += 1
        self.text_count += len(texts)
        await asyncio.sleep(self.delay_seconds)
        return np.ones((len(texts), self.dimension), dtype=np.float32)


def build_payload(prefix: str, record_count: int) -> dict[str, dict[str, str]]:
    return {
        f"{prefix}-{index}": {"content": f"{prefix} content {index}"}
        for index in range(record_count)
    }


def load_storage_class(mode: str):
    if mode == "nano":
        return NanoVectorDBStorage
    if mode == "noop":
        from lightrag.kg.noop_vector_db_impl import NoopVectorDBStorage

        return NoopVectorDBStorage
    raise ValueError(f"Unsupported mode: {mode}")


async def run_once(
    mode: str,
    record_count: int,
    batch_size: int,
    embedding_delay_seconds: float,
) -> dict[str, float | int | str]:
    finalize_share_data()
    initialize_share_data()
    embedding = DelayedEmbedding(dimension=8, delay_seconds=embedding_delay_seconds)
    storage_class = load_storage_class(mode)

    with tempfile.TemporaryDirectory(prefix=f"lightrag-{mode}-") as temp_dir:
        global_config = {
            "working_dir": temp_dir,
            "embedding_batch_num": batch_size,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.2
            },
        }
        embedding_func = EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=embedding,
        )
        storages = [
            storage_class(
                namespace=namespace,
                workspace="benchmark",
                global_config=global_config,
                embedding_func=embedding_func,
                meta_fields={"content"},
            )
            for namespace in ("entities", "relationships", "chunks")
        ]
        await asyncio.gather(*(storage.initialize() for storage in storages))
        payloads = [
            build_payload(storage.namespace, record_count) for storage in storages
        ]

        started = time.perf_counter()
        await asyncio.gather(
            *(storage.upsert(payload) for storage, payload in zip(storages, payloads))
        )
        await asyncio.gather(
            *(storage.index_done_callback() for storage in storages)
        )
        elapsed_seconds = time.perf_counter() - started

        materialized_records = 0
        if mode == "nano":
            for storage, payload in zip(storages, payloads):
                records = await storage.get_by_ids(list(payload))
                materialized_records += len(records)

        await asyncio.gather(*(storage.finalize() for storage in storages))

    finalize_share_data()
    return {
        "mode": mode,
        "elapsed_seconds": elapsed_seconds,
        "embedding_calls": embedding.call_count,
        "embedded_texts": embedding.text_count,
        "materialized_records": materialized_records,
    }


async def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    samples = [
        await run_once(
            mode=args.mode,
            record_count=args.records,
            batch_size=args.batch_size,
            embedding_delay_seconds=args.embedding_delay_ms / 1000,
        )
        for _ in range(args.repeats)
    ]
    elapsed = [float(sample["elapsed_seconds"]) for sample in samples]
    return {
        "mode": args.mode,
        "records_per_namespace": args.records,
        "namespaces": 3,
        "batch_size": args.batch_size,
        "embedding_delay_ms": args.embedding_delay_ms,
        "repeats": args.repeats,
        "median_seconds": statistics.median(elapsed),
        "min_seconds": min(elapsed),
        "max_seconds": max(elapsed),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("nano", "noop"), required=True)
    parser.add_argument("--records", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-delay-ms", type=float, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_benchmark(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
