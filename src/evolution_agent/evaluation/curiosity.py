"""Curiosity-driven exploration via behavioral embeddings.

Maintains a replay buffer of behavioral embeddings (per-instance fitness
vectors) for all evaluated individuals. Computes a curiosity bonus based
on how novel a new individual is compared to the buffer.

Uses GPU (PyTorch) for batch cosine similarity when available.

The curiosity bonus encourages the LLM to produce structurally novel
solutions rather than minor variants of the current best.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class BehaviorEntry:
    """One entry in the experience replay buffer."""

    code_hash: str
    embedding: list[float]  # behavioral embedding (per-instance fitness)
    fitness: float
    generation: int
    mutation_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CuriosityModule:
    """Computes curiosity bonus from behavioral embeddings.

    The curiosity bonus for a new individual is based on how far its
    behavioral embedding is from the nearest neighbor in the replay buffer.

    curiosity_bonus = 1 - max_cosine_similarity(new, buffer)

    Range: [0, 1] where 1 = maximally novel, 0 = identical to something seen.

    Combined fitness: fitness + lambda * curiosity_bonus
    """

    def __init__(
        self,
        curiosity_weight: float = 0.1,
        buffer_max_size: int = 500,
        use_gpu: bool = True,
    ) -> None:
        self._weight = curiosity_weight
        self._max_size = buffer_max_size
        self._buffer: list[BehaviorEntry] = []
        self._use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self._device = torch.device("cuda") if self._use_gpu else (
            torch.device("cpu") if TORCH_AVAILABLE else None
        )

        # Cached embedding matrix for GPU batch operations
        self._embedding_matrix: torch.Tensor | None = None
        self._matrix_dirty = True

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def curiosity_weight(self) -> float:
        return self._weight

    @curiosity_weight.setter
    def curiosity_weight(self, value: float) -> None:
        self._weight = value

    def add(self, entry: BehaviorEntry) -> None:
        """Add an individual to the replay buffer."""
        # Deduplicate by code hash
        for existing in self._buffer:
            if existing.code_hash == entry.code_hash:
                return

        self._buffer.append(entry)
        self._matrix_dirty = True

        # Evict oldest if buffer is full
        if len(self._buffer) > self._max_size:
            self._buffer.pop(0)

    def compute_curiosity(self, embedding: list[float]) -> float:
        """Compute curiosity bonus for a new embedding.

        Returns a value in [0, 1] where:
        - 1.0 = maximally novel (nothing similar in buffer)
        - 0.0 = identical to something already seen
        """
        if not self._buffer:
            return 1.0  # Everything is novel when buffer is empty

        if len(embedding) == 0:
            return 0.0

        if TORCH_AVAILABLE:
            return self._curiosity_torch(embedding)
        else:
            return self._curiosity_numpy(embedding)

    def _curiosity_torch(self, embedding: list[float]) -> float:
        """GPU/CPU accelerated curiosity via PyTorch."""
        # Rebuild embedding matrix if dirty
        if self._matrix_dirty or self._embedding_matrix is None:
            embeddings = [e.embedding for e in self._buffer]
            self._embedding_matrix = torch.tensor(
                embeddings, dtype=torch.float32, device=self._device,
            )
            # Normalize rows for cosine similarity
            norms = self._embedding_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self._embedding_matrix = self._embedding_matrix / norms
            self._matrix_dirty = False

        # Compute cosine similarity with entire buffer
        new_vec = torch.tensor(
            [embedding], dtype=torch.float32, device=self._device,
        )
        new_norm = new_vec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        new_vec = new_vec / new_norm

        # (1, d) @ (d, n) -> (1, n)
        similarities = (new_vec @ self._embedding_matrix.T).squeeze(0)

        # Curiosity = 1 - max similarity (most similar neighbor)
        max_sim = similarities.max().item()
        return max(0.0, 1.0 - max_sim)

    def _curiosity_numpy(self, embedding: list[float]) -> float:
        """Pure Python fallback for curiosity computation."""
        # Normalize new vector
        norm_new = math.sqrt(sum(x * x for x in embedding)) or 1e-8
        new_normed = [x / norm_new for x in embedding]

        max_sim = -1.0
        for entry in self._buffer:
            # Cosine similarity
            norm_buf = math.sqrt(sum(x * x for x in entry.embedding)) or 1e-8
            dot = sum(a * b / norm_buf for a, b in zip(new_normed, entry.embedding))
            if dot > max_sim:
                max_sim = dot

        return max(0.0, 1.0 - max_sim)

    def adjusted_fitness(
        self,
        fitness: float,
        embedding: list[float],
        direction: str = "maximize",
    ) -> float:
        """Compute fitness + curiosity bonus.

        For maximize: fitness + weight * curiosity
        For minimize: fitness - weight * curiosity
        """
        curiosity = self.compute_curiosity(embedding)

        if direction == "maximize":
            return fitness + self._weight * curiosity
        else:
            return fitness - self._weight * curiosity

    def get_stats(self) -> dict[str, Any]:
        """Return buffer statistics."""
        if not self._buffer:
            return {"buffer_size": 0}

        fitnesses = [e.fitness for e in self._buffer]
        return {
            "buffer_size": len(self._buffer),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "best_fitness": max(fitnesses),
            "embedding_dim": len(self._buffer[0].embedding) if self._buffer else 0,
            "gpu": self._use_gpu,
        }

    def get_diversity_matrix(self) -> list[list[float]]:
        """Compute pairwise cosine similarity matrix (for analysis).

        Returns NxN matrix where N = buffer size.
        GPU-accelerated when available.
        """
        if not self._buffer or not TORCH_AVAILABLE:
            return []

        if self._matrix_dirty or self._embedding_matrix is None:
            embeddings = [e.embedding for e in self._buffer]
            self._embedding_matrix = torch.tensor(
                embeddings, dtype=torch.float32, device=self._device,
            )
            norms = self._embedding_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self._embedding_matrix = self._embedding_matrix / norms
            self._matrix_dirty = False

        # (N, d) @ (d, N) -> (N, N)
        sim_matrix = (self._embedding_matrix @ self._embedding_matrix.T)
        return sim_matrix.cpu().tolist()
