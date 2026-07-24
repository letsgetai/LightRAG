#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from lightrag import LightRAG
from lightrag.tools.rebuild_vdb import (
    enumerate_kv_keys,
    rebuild_chunks_vdb,
    rebuild_entities_vdb,
    rebuild_relationships_vdb,
)
from lightrag.utils import EmbeddingFunc


class CountingEmbedding:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.call_count = 0
        self.text_count = 0

    async def __call__(self, texts: list[str], **_: object) -> np.ndarray:
        self.call_count += 1
        self.text_count += len(texts)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return np.ones((len(texts), 8), dtype=np.float32)


async def dummy_llm(*_: object, **__: object) -> str:
    return ""


def make_document(
    document_index: int,
    shared_entities: int,
    unique_entities: int,
) -> dict[str, list[dict[str, object]]]:
    source_id = f"source-{document_index:05d}"
    file_path = f"document-{document_index:05d}.txt"
    shared_names = [f"SHARED_{index:05d}" for index in range(shared_entities)]
    unique_names = [
        f"DOC_{document_index:05d}_ENTITY_{index:05d}"
        for index in range(unique_entities)
    ]
    entity_names = shared_names + unique_names

    entities = [
        {
            "entity_name": name,
            "entity_type": "CONCEPT",
            "description": f"{name} observed in document {document_index}",
            "source_id": source_id,
            "file_path": file_path,
        }
        for name in entity_names
    ]

    relationships = []
    if len(shared_names) > 1:
        relationships.extend(
            {
                "src_id": name,
                "tgt_id": shared_names[(index + 1) % len(shared_names)],
                "description": f"Shared relation updated by document {document_index}",
                "keywords": "shared relation",
                "weight": 1.0,
                "source_id": source_id,
                "file_path": file_path,
            }
            for index, name in enumerate(shared_names)
        )
    relationships.extend(
        {
            "src_id": name,
            "tgt_id": shared_names[index % len(shared_names)]
            if shared_names
            else unique_names[(index + 1) % len(unique_names)],
            "description": f"Unique relation from document {document_index}",
            "keywords": "unique relation",
            "weight": 1.0,
            "source_id": source_id,
            "file_path": file_path,
        }
        for index, name in enumerate(unique_names)
        if len(entity_names) > 1
    )

    return {
        "chunks": [
            {
                "content": f"Synthetic content for document {document_index}",
                "source_id": source_id,
                "file_path": file_path,
            }
        ],
        "entities": entities,
        "relationships": relationships,
    }


def build_rag(
    working_dir: str,
    vector_storage: str,
    embedding: CountingEmbedding,
    embedding_batch_size: int,
) -> LightRAG:
    return LightRAG(
        working_dir=working_dir,
        vector_storage=vector_storage,
        llm_model_func=dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=embedding,
        ),
        embedding_batch_num=embedding_batch_size,
    )


async def graph_counts(rag: LightRAG) -> dict[str, int]:
    return {
        "nodes": len(await rag.chunk_entity_relation_graph.get_all_nodes()),
        "edges": len(await rag.chunk_entity_relation_graph.get_all_edges()),
        "chunks": len(await enumerate_kv_keys(rag.text_chunks)),
    }


async def run_once(args: argparse.Namespace) -> dict[str, object]:
    embedding = CountingEmbedding(args.embedding_delay_ms / 1000)
    with tempfile.TemporaryDirectory(prefix=f"lightrag-{args.mode}-") as temp_dir:
        vector_storage = (
            "NanoVectorDBStorage" if args.mode == "baseline" else "NoopVectorDBStorage"
        )
        rag = build_rag(
            temp_dir,
            vector_storage,
            embedding,
            args.embedding_batch_size,
        )
        await rag.initialize_storages()

        ingest_started = time.perf_counter()
        for document_index in range(args.documents):
            await rag.ainsert_custom_kg(
                make_document(
                    document_index,
                    args.shared_entities,
                    args.unique_entities,
                )
            )
        ingest_seconds = time.perf_counter() - ingest_started
        counts = await graph_counts(rag)
        ingestion_embedding_calls = embedding.call_count
        ingestion_embedded_texts = embedding.text_count
        await rag.finalize_storages()

        rebuild_seconds = 0.0
        rebuild_stats: list[dict[str, object]] = []
        if args.mode == "deferred":
            indexed_rag = build_rag(
                temp_dir,
                "NanoVectorDBStorage",
                embedding,
                args.embedding_batch_size,
            )
            await indexed_rag.initialize_storages()
            rebuild_started = time.perf_counter()
            rebuild_stats = [
                await rebuild_entities_vdb(
                    indexed_rag.chunk_entity_relation_graph,
                    indexed_rag.entities_vdb,
                    indexed_rag._build_global_config(),
                    batch_size=args.rebuild_batch_size,
                ),
                await rebuild_relationships_vdb(
                    indexed_rag.chunk_entity_relation_graph,
                    indexed_rag.relationships_vdb,
                    indexed_rag._build_global_config(),
                    batch_size=args.rebuild_batch_size,
                ),
                await rebuild_chunks_vdb(
                    indexed_rag.text_chunks,
                    indexed_rag.chunks_vdb,
                    batch_size=args.rebuild_batch_size,
                ),
            ]
            rebuild_seconds = time.perf_counter() - rebuild_started
            rebuilt_counts = await graph_counts(indexed_rag)
            assert rebuilt_counts == counts
            assert all(not stats["errors"] for stats in rebuild_stats)
            assert [stats["rebuilt"] for stats in rebuild_stats] == [
                counts["nodes"],
                counts["edges"],
                counts["chunks"],
            ]
            await indexed_rag.finalize_storages()

        final_vector_records = sum(counts.values())
        return {
            "mode": args.mode,
            "documents": args.documents,
            "shared_entities": args.shared_entities,
            "unique_entities_per_document": args.unique_entities,
            "embedding_batch_size": args.embedding_batch_size,
            "embedding_delay_ms": args.embedding_delay_ms,
            "graph": counts,
            "ingest_seconds": ingest_seconds,
            "rebuild_seconds": rebuild_seconds,
            "total_seconds": ingest_seconds + rebuild_seconds,
            "ingestion_embedding_calls": ingestion_embedding_calls,
            "ingestion_embedded_texts": ingestion_embedded_texts,
            "total_embedding_calls": embedding.call_count,
            "total_embedded_texts": embedding.text_count,
            "final_vector_records": final_vector_records,
            "vector_write_amplification": (
                embedding.text_count / final_vector_records
                if final_vector_records
                else 0.0
            ),
            "rebuild_stats": rebuild_stats,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "deferred"), required=True)
    parser.add_argument("--documents", type=int, default=20)
    parser.add_argument("--shared-entities", type=int, default=100)
    parser.add_argument("--unique-entities", type=int, default=50)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--rebuild-batch-size", type=int, default=500)
    parser.add_argument("--embedding-delay-ms", type=float, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_once(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
