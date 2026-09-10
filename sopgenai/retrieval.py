from __future__ import annotations

import faiss
import numpy as np

from rank_bm25 import (
    BM25Okapi,
)

from sentence_transformers import (
    SentenceTransformer,
)


class HybridRetriever:

    def __init__(
        self,
        embedding_config,
        retrieval_config,
    ):

        self.embedding_config = (
            embedding_config
        )

        self.config = (
            retrieval_config
        )

        self.encoder = (
            SentenceTransformer(
                embedding_config[
                    "model"
                ]
            )
        )

        self.units = []

        self.faiss_index = None

        self.bm25 = None

    def index(
        self,
        units,
    ):

        self.units = units

        if not units:

            print(
                "[Retrieval] "
                "No knowledge units to index."
            )

            return

        print(
            "[Retrieval] "
            f"Indexing {len(units)} "
            "knowledge units..."
        )

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

            text
            .lower()
            .split()

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
                    self.embedding_config
                    .get(
                        "normalize_embeddings",
                        True,
                    )
                ),

                show_progress_bar=True,
            )
        )

        embeddings = (
            np.asarray(
                embeddings,
                dtype="float32",
            )
        )

        self.faiss_index = (
            faiss.IndexFlatIP(
                embeddings.shape[1]
            )
        )

        self.faiss_index.add(
            embeddings
        )

        print(
            "[Retrieval] "
            "Index ready."
        )

    def search(
        self,
        query,
        top_k=None,
    ):

        if not self.units:

            return []

        top_k = (
            top_k
            or self.config.get(
                "top_k",
                6,
            )
        )

        multiplier = (
            self.config.get(
                "candidate_multiplier",
                3,
            )
        )

        query_embedding = (
            self.encoder.encode(

                [query],

                normalize_embeddings=(
                    self.embedding_config
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
                top_k
                * multiplier,
                top_k,
            ),
        )

        scores, indexes = (
            self.faiss_index
            .search(
                query_embedding,
                candidate_count,
            )
        )

        for (
            score,
            index,
        ) in zip(
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

        results = []

        for (
            index,
            unit,
        ) in enumerate(
            self.units
        ):

            authority = (
                unit.authority
                / 100.0
            )

            score = (

                self.config.get(
                    "vector_weight",
                    0.45,
                )
                * dense_scores[
                    index
                ]

                +

                self.config.get(
                    "keyword_weight",
                    0.35,
                )
                * float(
                    sparse_scores[
                        index
                    ]
                )

                +

                self.config.get(
                    "authority_weight",
                    0.20,
                )
                * authority
            )

            results.append(
                (
                    score,
                    unit,
                )
            )

        return sorted(
            results,

            key=lambda item:
                item[0],

            reverse=True,

        )[:top_k]
