from __future__ import annotations

import faiss
import numpy as np

from rank_bm25 import (
    BM25Okapi,
)

from sentence_transformers import (
    SentenceTransformer,
)

from .models import (
    KnowledgeUnit,
)


class HybridRetriever:

    def __init__(
        self,
        embedding_config: dict,
        retrieval_config: dict,
    ):

        self.embedding_config = (
            embedding_config
        )

        self.config = (
            retrieval_config
        )

        model_name = (
            embedding_config[
                "model"
            ]
        )

        self.encoder = (
            SentenceTransformer(
                model_name
            )
        )

        self.units = []

        self.faiss_index = None

        self.bm25 = None

    def index(
        self,
        units: list[
            KnowledgeUnit
        ],
    ):

        self.units = (
            units
        )

        if not units:
            return

        texts = [
            (
                unit.title
                + "\n"
                + unit.text
            )
            for unit
            in units
        ]

        tokenized = [
            text.lower().split()
            for text
            in texts
        ]

        self.bm25 = (
            BM25Okapi(
                tokenized
            )
        )

        embeddings = (
            self.encoder.encode(
                texts,

                normalize_embeddings=(
                    self
                    .embedding_config
                    .get(
                        "normalize_embeddings",
                        True,
                    )
                ),
            )
        )

        embeddings = (
            np.asarray(
                embeddings,
                dtype="float32",
            )
        )

        dimension = (
            embeddings.shape[1]
        )

        self.faiss_index = (
            faiss.IndexFlatIP(
                dimension
            )
        )

        self.faiss_index.add(
            embeddings
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ):

        if not self.units:
            return []

        top_k = (
            top_k
            or self.config.get(
                "top_k",
                8,
            )
        )

        multiplier = (
            self.config.get(
                "candidate_multiplier",
                4,
            )
        )

        query_embedding = (
            self.encoder.encode(
                [query],
                normalize_embeddings=(
                    self
                    .embedding_config
                    .get(
                        "normalize_embeddings",
                        True,
                    )
                ),
            )
        )

        query_embedding = (
            np.asarray(
                query_embedding,
                dtype="float32",
            )
        )

        dense_scores = np.zeros(
            len(self.units),
            dtype=float,
        )

        candidate_count = min(
            len(self.units),
            max(
                top_k * multiplier,
                top_k,
            ),
        )

        scores, indexes = (
            self.faiss_index.search(
                query_embedding,
                candidate_count,
            )
        )

        for score, index in zip(
            scores[0],
            indexes[0],
        ):

            if index >= 0:

                dense_scores[
                    index
                ] = float(
                    score
                )

        sparse_scores = (
            self.bm25
            .get_scores(
                query
                .lower()
                .split()
            )
        )

        if (
            len(sparse_scores)
            and sparse_scores.max()
            > 0
        ):

            sparse_scores = (
                sparse_scores
                / (
                    sparse_scores.max()
                    + 1e-9
                )
            )

        vector_weight = (
            self.config.get(
                "vector_weight",
                0.45,
            )
        )

        keyword_weight = (
            self.config.get(
                "keyword_weight",
                0.35,
            )
        )

        authority_weight = (
            self.config.get(
                "authority_weight",
                0.20,
            )
        )

        results = []

        for index, unit in enumerate(
            self.units
        ):

            authority_score = (
                unit.authority
                / 100.0
            )

            final_score = (
                vector_weight
                * dense_scores[index]

                + keyword_weight
                * float(
                    sparse_scores[
                        index
                    ]
                )

                + authority_weight
                * authority_score
            )

            results.append(
                (
                    final_score,
                    unit,
                )
            )

        results.sort(
            key=lambda item:
                item[0],
            reverse=True,
        )

        return (
            results[:top_k]
        )
