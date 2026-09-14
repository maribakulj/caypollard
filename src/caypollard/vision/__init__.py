"""Vision encoder and embedding-store utilities."""

from .store import EmbeddingTable, load_embedding_table, save_embedding_table

__all__ = ["EmbeddingTable", "load_embedding_table", "save_embedding_table"]
