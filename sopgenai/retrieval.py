from __future__ import annotations

import numpy as np
import faiss

from rank_bm25 import BM25Okapi

from sentence_transformers import (
    SentenceTransformer,
)

from .models import KnowledgeUnit


class HybridRetriever:

    def __init__(
        self,
        embedding_model=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    ):

        self.model = (
            SentenceTransformer(
                embedding_model
            )
        )

        self.units = []

        self.index = None

        self.embeddings = None

        self.bm25 = None

    def index_documents(
        self,
        units: list[KnowledgeUnit],
    ):

        self.units = units

        texts = [
            f"{unit.title} {unit.text}"
            for unit in units
        ]

        embeddings = (
            self.model.encode(
                texts,
                normalize_embeddings=True,
            )
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        self.embeddings = embeddings

        dimension = (
            embeddings.shape[1]
        )

        self.index = (
            faiss.IndexFlatIP(
                dimension
            )
        )

        self.index.add(
            embeddings
        )

        tokenized = [
            text.lower().split()
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            tokenized
        )

    def search(
        self,
        query,
        top_k=10,
    ):

        query_embedding = (
            self.model.encode(
                [query],
                normalize_embeddings=True,
            )
        )

        query_embedding = (
            np.asarray(
                query_embedding,
                dtype="float32",
            )
        )

        vector_scores, vector_ids = (
            self.index.search(
                query_embedding,
                min(
                    len(self.units),
                    top_k * 3,
                ),
            )
        )

        bm25_scores = (
            self.bm25.get_scores(
                query.lower().split()
            )
        )

        scores = {}

        for score, idx in zip(
            vector_scores[0],
            vector_ids[0],
        ):

            if idx < 0:
                continue

            scores[idx] = (
                scores.get(idx, 0)
                + float(score) * 0.55
            )

        max_bm25 = max(
            bm25_scores
        ) if len(bm25_scores) else 1

        for idx, score in enumerate(
            bm25_scores
        ):

            normalized = (
                score / max_bm25
                if max_bm25
                else 0
            )

            scores[idx] = (
                scores.get(idx, 0)
                + normalized * 0.30
            )

        for idx in scores:

            authority = (
                self.units[idx]
                .authority_level
                / 100
            )

            scores[idx] += (
                authority * 0.15
            )

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            self.units[idx]
            for idx, _
            in ranked[:top_k]
        ]
