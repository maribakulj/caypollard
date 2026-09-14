"""Generic embedding storage shared by visual and graph experiments."""

from .store import EmbeddingTable, l2_normalize, load_embedding_table, save_embedding_table

__all__ = ["EmbeddingTable", "l2_normalize", "load_embedding_table", "save_embedding_table"]
