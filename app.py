"""Resume classification and semantic search API."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_NAME = os.getenv("HF_DATASET_NAME", "C0ldSmi1eresume-dataset")
DATASET_SPLIT = os.getenv("HF_DATASET_SPLIT", "train")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_COLUMN = os.getenv("TEXT_COLUMN")
LABEL_COLUMN = os.getenv("LABEL_COLUMN")
DATABASE_URL = os.getenv("DATABASE_URL")
MAX_DATASET_ROWS = int(os.getenv("MAX_DATASET_ROWS", "0"))
MAX_TEXT_CHARS = 50_000
COLLECTION_NAME = "resume_embeddings"


class ServiceInitializationError(RuntimeError):
    """Raised when model, dataset, or index initialization fails."""


@dataclass
class ResumeService:
    """Holds the trained classifier and the in-memory Qdrant index."""

    model: SentenceTransformer
    classifier: LogisticRegression
    qdrant: QdrantClient
    collection_name: str

    @classmethod
    def from_hugging_face(cls) -> "ResumeService":
        token = os.getenv("HF_TOKEN")
        try:
            dataset = load_dataset(
                DATASET_NAME,
                **({"token": token} if token else {}),
            )
        except Exception as exc:
            raise ServiceInitializationError(
                f"Could not load Hugging Face dataset '{DATASET_NAME}' "
                f"(split '{DATASET_SPLIT}'). Check the dataset ID, access, "
                "HF_TOKEN, and HF_DATASET_SPLIT."
            ) from exc

        if isinstance(dataset, DatasetDict):
            if DATASET_SPLIT in dataset:
                dataset = dataset[DATASET_SPLIT]
            else:
                raise ServiceInitializationError(
                    f"Split '{DATASET_SPLIT}' is not available in the dataset. "
                    f"Available splits: {list(dataset.keys())}."
                )
        if not isinstance(dataset, Dataset):
            raise ServiceInitializationError(
                "The selected Hugging Face split did not return a tabular dataset."
            )

        text_column, label_column = cls._resolve_columns(dataset)
        texts: list[str] = []
        labels: list[str] = []
        label_feature = dataset.features.get(label_column)

        for row_number, row in enumerate(dataset):
            if MAX_DATASET_ROWS and row_number >= MAX_DATASET_ROWS:
                break
            raw_text = row.get(text_column)
            raw_label = row.get(label_column)
            if raw_text is None or raw_label is None:
                continue
            text = str(raw_text).strip()
            label = cls._label_name(raw_label, label_feature)
            if not text:
                continue
            if not label:
                continue
            texts.append(text)
            labels.append(label)

        if not texts:
            raise ServiceInitializationError(
                f"No usable resume text and labels were found in columns "
                f"'{text_column}' and '{label_column}'."
            )
        if len(set(labels)) < 2:
            raise ServiceInitializationError(
                "LogisticRegression needs at least two categories; the selected "
                "dataset split contains fewer than two."
            )

        logger.info(
            "Loading embedding model %s and encoding %d dataset rows",
            MODEL_NAME,
            len(texts),
        )
        model = SentenceTransformer(MODEL_NAME)
        vectors = model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        classifier.fit(vectors, labels)

        qdrant = QdrantClient(location=":memory:")
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=int(vectors.shape[1]),
                distance=models.Distance.COSINE,
            ),
        )
        for start in range(0, len(texts), 256):
            end = min(start + 256, len(texts))
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=index,
                        vector=vectors[index].tolist(),
                        payload={"text": texts[index], "category": labels[index]},
                    )
                    for index in range(start, end)
                ],
                wait=True,
            )

        logger.info(
            "Resume service ready: %d indexed rows, %d categories",
            len(texts),
            len(classifier.classes_),
        )
        return cls(model, classifier, qdrant, COLLECTION_NAME)

    @staticmethod
    def _resolve_columns(dataset: Dataset) -> tuple[str, str]:
        columns = dataset.column_names
        text_column = TEXT_COLUMN or next(
            (
                candidate
                for candidate in ("resume_text", "resume", "text", "content")
                if candidate in columns
            ),
            None,
        )
        label_column = LABEL_COLUMN or next(
            (
                candidate
                for candidate in ("category", "label", "job_category", "class")
                if candidate in columns
            ),
            None,
        )
        if text_column not in columns or label_column not in columns:
            raise ServiceInitializationError(
                f"Could not identify text and category columns in {columns}. "
                "Set TEXT_COLUMN and LABEL_COLUMN to the correct column names."
            )
        return text_column, label_column

    @staticmethod
    def _label_name(value: Any, feature: Any) -> str:
        names = getattr(feature, "names", None)
        if names is not None and isinstance(value, (int, np.integer)):
            if 0 <= int(value) < len(names):
                return str(names[int(value)])
        return str(value).strip()

    def classify(self, text: str) -> dict[str, Any]:
        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        probabilities = self.classifier.predict_proba(vector)[0]
        ranked = np.argsort(probabilities)[::-1]
        best = int(ranked[0])
        confidence = float(probabilities[best])
        alternatives = [
            {
                "category": str(self.classifier.classes_[index]),
                "confidence": float(probabilities[index]),
            }
            for index in ranked[:3]
        ]
        return {
            "category": str(self.classifier.classes_[best]),
            "confidence": confidence,
            "alternatives": alternatives,
        }

    def search(self, text: str, limit: int) -> list[dict[str, Any]]:
        vector = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            limit=limit,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "category": payload.get("category"),
                    "score": float(point.score),
                    "excerpt": str(payload.get("text", ""))[:500],
                }
            )
        return results


class PredictionStore:
    """Stores prediction metadata without retaining submitted resume text."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_predictions (
                    id BIGSERIAL PRIMARY KEY,
                    category TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    def save(self, category: str, confidence: float) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                """
                INSERT INTO resume_predictions (category, confidence)
                VALUES (%s, %s)
                """,
                (category, confidence),
            )


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1_000_000
    service: ResumeService | None = None
    service_error: Exception | None = None
    service_lock = threading.Lock()

    store: PredictionStore | None = None
    if DATABASE_URL:
        try:
            store = PredictionStore(DATABASE_URL)
        except psycopg.Error:
            logger.exception("Could not initialize PostgreSQL prediction storage")
            raise
    else:
        logger.warning(
            "DATABASE_URL is not set; PostgreSQL prediction logging is disabled."
        )

    def get_service() -> ResumeService:
        nonlocal service, service_error
        if service is not None:
            return service
        with service_lock:
            if service is not None:
                return service
            if service_error is not None:
                raise ServiceInitializationError(str(service_error)) from service_error
            try:
                service = ResumeService.from_hugging_face()
            except Exception as exc:
                logger.exception("Could not initialize resume classification service")
                service_error = exc
                raise ServiceInitializationError(str(exc)) from exc
        return service

    def read_text_payload() -> tuple[dict[str, Any] | None, tuple[Any, int] | None]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return None, (jsonify(error="Request body must be a JSON object."), 400)
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return None, (jsonify(error="'text' must be a non-empty string."), 400)
        if len(text) > MAX_TEXT_CHARS:
            return None, (
                jsonify(error=f"'text' must be at most {MAX_TEXT_CHARS} characters."),
                413,
            )
        payload["text"] = text.strip()
        return payload, None

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/health")
    def health() -> Any:
        return jsonify(
            status="ok",
            model_loaded=service is not None,
            dataset=DATASET_NAME,
            postgres_enabled=store is not None,
        )

    @app.post("/api/classify-resume")
    def classify_resume() -> Any:
        payload, error = read_text_payload()
        if error:
            return error
        try:
            result = get_service().classify(payload["text"])
            if store is not None:
                store.save(result["category"], result["confidence"])
            return jsonify(result)
        except ServiceInitializationError as exc:
            return jsonify(error=str(exc)), 503

    @app.post("/api/search")
    def semantic_search() -> Any:
        payload, error = read_text_payload()
        if error:
            return error
        raw_limit = payload.get("limit", 5)
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
            return jsonify(error="'limit' must be an integer between 1 and 20."), 400
        if not 1 <= raw_limit <= 20:
            return jsonify(error="'limit' must be an integer between 1 and 20."), 400
        try:
            results = get_service().search(payload["text"], raw_limit)
            return jsonify(results=results)
        except ServiceInitializationError as exc:
            return jsonify(error=str(exc)), 503

    @app.errorhandler(413)
    def request_too_large(_: Any) -> Any:
        return jsonify(error="Request body is too large."), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
        threaded=True,
    )
