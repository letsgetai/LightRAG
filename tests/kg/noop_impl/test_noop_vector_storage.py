from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg import STORAGE_IMPLEMENTATIONS, STORAGES
from lightrag.kg.factory import get_storage_class
from lightrag.kg.noop_vector_db_impl import NoopVectorDBStorage
from lightrag.utils import EmbeddingFunc


class FailingEmbedding:
    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(self, texts: list[str], **_: object) -> np.ndarray:
        self.call_count += 1
        raise AssertionError("NoopVectorDBStorage must not call embedding_func")


def make_storage(embedding: FailingEmbedding) -> NoopVectorDBStorage:
    return NoopVectorDBStorage(
        namespace="test_vectors",
        workspace="test_workspace",
        global_config={},
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=embedding,
        ),
        meta_fields={"content"},
    )


def test_noop_vector_storage_is_registered() -> None:
    assert "NoopVectorDBStorage" in STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"][
        "implementations"
    ]
    assert STORAGES["NoopVectorDBStorage"] == ".kg.noop_vector_db_impl"
    assert get_storage_class("NoopVectorDBStorage") is NoopVectorDBStorage


@pytest.mark.offline
@pytest.mark.asyncio
async def test_noop_vector_storage_contract_never_embeds() -> None:
    embedding = FailingEmbedding()
    storage = make_storage(embedding)

    await storage.initialize()
    await storage.upsert({"id-1": {"content": "alpha"}})
    await storage.delete(["id-1"])
    await storage.delete_entity("Entity")
    await storage.delete_entity_relation("Entity")
    await storage.index_done_callback()
    await storage.drop_pending_index_ops()

    assert await storage.get_by_id("id-1") is None
    assert await storage.get_by_ids(["id-1", "id-2"]) == [None, None]
    assert await storage.get_vectors_by_ids(["id-1"]) == {}
    assert await storage.drop() == {
        "status": "success",
        "message": "Noop vector storage contains no data",
    }
    assert embedding.call_count == 0

    with pytest.raises(RuntimeError, match="lightrag-rebuild-vdb"):
        await storage.query("question", top_k=5)

    await storage.finalize()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_graph_only_custom_kg_insertion_preserves_graph(tmp_path) -> None:
    embedding = FailingEmbedding()
    rag = LightRAG(
        working_dir=str(tmp_path),
        vector_storage="NoopVectorDBStorage",
        llm_model_func=AsyncMock(return_value=""),
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=embedding,
        ),
    )
    await rag.initialize_storages()

    await rag.ainsert_custom_kg(
        {
            "chunks": [
                {
                    "content": "Alice knows Bob.",
                    "source_id": "source-1",
                    "file_path": "example.txt",
                }
            ],
            "entities": [
                {
                    "entity_name": "Alice",
                    "entity_type": "PERSON",
                    "description": "A person",
                    "source_id": "source-1",
                    "file_path": "example.txt",
                },
                {
                    "entity_name": "Bob",
                    "entity_type": "PERSON",
                    "description": "Another person",
                    "source_id": "source-1",
                    "file_path": "example.txt",
                },
            ],
            "relationships": [
                {
                    "src_id": "Alice",
                    "tgt_id": "Bob",
                    "description": "Alice knows Bob",
                    "keywords": "knows",
                    "weight": 1.0,
                    "source_id": "source-1",
                    "file_path": "example.txt",
                }
            ],
        }
    )

    assert await rag.chunk_entity_relation_graph.has_node("ALICE")
    assert await rag.chunk_entity_relation_graph.has_node("BOB")
    assert await rag.chunk_entity_relation_graph.has_edge("ALICE", "BOB")
    assert embedding.call_count == 0

    await rag.finalize_storages()
