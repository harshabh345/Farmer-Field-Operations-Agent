#!/usr/bin/env python3

"""
Final CMD-Based Agricultural AI System

Combines:
1. Farmer and plot information
2. Latitude and longitude
3. Open-Meteo live weather
4. ChromaDB agricultural knowledge base
5. BM25 lexical retrieval
6. Dense vector retrieval
7. Metadata-aware fusion
8. Light cross-encoder reranking
9. Diverse evidence selection
10. Two trained XGBoost models
11. Groq agricultural scientist reasoning
12. Evidence-based farmer-friendly final report

Excluded:
- Bhoonidhi
- VEDAS
- Market analysis
- Web application
"""

from __future__ import annotations

import os
import re
import sys
import json
import math
import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from tkintermapview import TkinterMapView

import joblib
import numpy as np
import requests
import chromadb
import pandas as pd

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path("/mnt/d/EdgeAi_resistor_Dl_model/RAG")
# ============================================================
# MAP LOCATION STATE
# ============================================================

selected_latitude = None
selected_longitude = None
map_widget = None
map_window = None

load_dotenv(PROJECT_ROOT / ".env")

ML_ROOT = PROJECT_ROOT / "ml_training"
MODEL_DIR = ML_ROOT / "models"

DATA_ROOT = PROJECT_ROOT / "Data"
EMBEDDING_DIR = DATA_ROOT / "embeddings"
VECTOR_STORE_DIR = DATA_ROOT / "vector_store"

EMBEDDING_FILE = EMBEDDING_DIR / "agri_embeddings.npy"
RECORDS_FILE = EMBEDDING_DIR / "agri_embedding_records.jsonl"

PRODUCTION_MODEL_FILE = MODEL_DIR / "agriculture_production_xgb.pkl"
CROP_RECOMMENDATION_MODEL_FILE = MODEL_DIR / "crop_recommendation_xgb.pkl"

CHROMA_COLLECTION_NAME = "agri_documents"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Optional reranker. It is loaded only if available.
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

use_reranker=False,


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_text(value: Any) -> str:
    """
    Convert any value into clean text.
    """
    if value is None:
        return ""

    if isinstance(value, dict):
        return " ".join(
            f"{key} {safe_text(item)}"
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple, set)):
        return " ".join(safe_text(item) for item in value)

    return str(value).strip()


def clean_text(text: str) -> str:
    """
    Normalize whitespace but preserve useful content.
    """
    text = safe_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """
    BM25 tokenizer.
    """
    text = clean_text(text).lower()
    return re.findall(r"[a-zA-Z0-9]+", text)


def scalar_metadata(value: Any) -> Any:
    """
    ChromaDB accepts scalar metadata values.
    Convert lists, dictionaries, numpy values, and None safely.
    """
    if value is None:
        return ""

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple, set)):
        return ", ".join(clean_text(item) for item in value)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    return clean_text(value)


def deterministic_id(record: Dict[str, Any], index: int) -> str:
    """
    Create a stable ID for ChromaDB.
    """
    metadata = record.get("metadata", {})
    content = record.get("content", "")

    source_file = metadata.get("source_file", "")
    page = metadata.get("page", "")
    chunk_id = metadata.get("chunk_id", "")

    raw = f"{chunk_id}|{source_file}|{page}|{content}|{index}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()

    return f"agri_{digest}"


def flatten_dictionary(data: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """
    Flatten nested farmer data into searchable key-value text.
    """
    result = {}

    if not isinstance(data, dict):
        return result

    for key, value in data.items():
        current_key = f"{prefix}_{key}" if prefix else str(key)

        if isinstance(value, dict):
            result.update(flatten_dictionary(value, current_key))
        else:
            result[current_key] = safe_text(value)

    return result


# ============================================================
# CROP AND REGION ALIASES
# ============================================================

CROP_ALIASES = {
    "sugarcane": [
        "sugarcane",
        "sugar cane",
        "cane",
        "ganna",
        "saccharum"
    ],
    "rice": [
        "rice",
        "paddy",
        "dhan"
    ],
    "maize": [
        "maize",
        "corn",
        "makka"
    ],
    "pigeonpea": [
        "pigeonpea",
        "pigeon pea",
        "tur",
        "arhar",
        "red gram"
    ],
    "chickpea": [
        "chickpea",
        "gram",
        "bengal gram",
        "chana"
    ],
    "blackgram": [
        "blackgram",
        "black gram",
        "urad"
    ],
    "greengram": [
        "greengram",
        "green gram",
        "mungbean",
        "moong"
    ],
    "groundnut": [
        "groundnut",
        "peanut"
    ],
    "cotton": [
        "cotton",
        "kapas"
    ],
    "soybean": [
        "soybean",
        "soyabean"
    ],
    "wheat": [
        "wheat",
        "gehun"
    ],
}

STATE_ALIASES = {
    "maharashtra": [
        "maharashtra",
        "mh",
        "maharashtra state"
    ],
    "karnataka": [
        "karnataka",
        "ka",
        "karnataka state"
    ],
    "madhya pradesh": [
        "madhya pradesh",
        "mp"
    ],
    "gujarat": [
        "gujarat",
        "gj"
    ],
    "telangana": [
        "telangana",
        "ts"
    ],
    "andhra pradesh": [
        "andhra pradesh",
        "ap"
    ],
    "uttar pradesh": [
        "uttar pradesh",
        "up"
    ],
    "punjab": [
        "punjab",
        "pb"
    ],
    "haryana": [
        "haryana",
        "hr"
    ],
}


TOPIC_TERMS = {
    "weather": [
        "weather",
        "rainfall",
        "rain",
        "temperature",
        "humidity",
        "wind",
        "cloud",
        "forecast",
        "drought",
        "dry spell",
        "heavy rain"
    ],
    "soil": [
        "soil",
        "black soil",
        "red soil",
        "loam",
        "clay",
        "sandy",
        "soil moisture",
        "soil fertility",
        "soil test",
        "ph"
    ],
    "irrigation": [
        "irrigation",
        "water",
        "borewell",
        "drip",
        "sprinkler",
        "moisture",
        "drainage"
    ],
    "crop_management": [
        "planting",
        "sowing",
        "seed",
        "setts",
        "spacing",
        "land preparation",
        "intercropping",
        "earthing",
        "mulching",
        "weeding"
    ],
    "fertilizer": [
        "fertilizer",
        "fertiliser",
        "nitrogen",
        "phosphorus",
        "potassium",
        "urea",
        "fym",
        "manure",
        "nutrient"
    ],
    "disease_pest": [
        "disease",
        "pest",
        "borer",
        "fungus",
        "fungal",
        "virus",
        "insect",
        "infection",
        "spray",
        "pathogen"
    ],
    "yield": [
        "yield",
        "production",
        "productivity",
        "tonne",
        "harvest",
        "output"
    ],
    "crop_suitability": [
        "suitability",
        "recommended crop",
        "crop selection",
        "climate suitability",
        "adaptation",
        "alternative crop"
    ],
}


# ============================================================
# RECORD NORMALIZATION
# ============================================================

def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize different possible JSONL record formats.

    Supported examples:
    {
        "page_content": "...",
        "metadata": {...}
    }

    or:

    {
        "content": "...",
        "source_file": "...",
        "page": 10
    }
    """

    metadata = raw.get("metadata", {})

    if not isinstance(metadata, dict):
        metadata = {}

    content = (
        raw.get("page_content")
        or raw.get("content")
        or raw.get("text")
        or ""
    )

    content = clean_text(content)

    # If metadata is stored flat, move useful fields into metadata.
    for key, value in raw.items():
        if key not in {
            "page_content",
            "content",
            "text",
            "metadata",
            "embedding"
        } and key not in metadata:
            metadata[key] = value

    metadata = {
        str(key): scalar_metadata(value)
        for key, value in metadata.items()
    }

    # Page numbers from PDF loaders are often zero-based.
    page = metadata.get("page", "")

    try:
        page_number = int(page) + 1
    except Exception:
        page_number = page

    metadata["page_display"] = scalar_metadata(page_number)

    return {
        "content": content,
        "metadata": metadata
    }


def load_embedding_records() -> List[Dict[str, Any]]:
    """
    Load exactly the records corresponding to the saved embeddings.
    This avoids the previous 4012-chunk versus 2575-embedding mismatch.
    """

    if not RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"Embedding records not found: {RECORDS_FILE}"
        )

    records = []

    with RECORDS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw = json.loads(line)
                record = normalize_record(raw)

                if record["content"]:
                    records.append(record)

            except Exception as error:
                print(
                    f"Warning: could not read JSONL line "
                    f"{line_number}: {error}"
                )

    return records


# ============================================================
# OPEN-METEO WEATHER
# ============================================================

def fetch_open_meteo(
    latitude: float,
    longitude: float,
    forecast_days: int = 16
) -> Dict[str, Any]:
    """
    Fetch comprehensive weather data for the farmer's exact location.
    """

    hourly_fields = ",".join([
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation_probability",
        "precipitation",
        "rain",
        "showers",
        "weather_code",
        "cloud_cover",
        "wind_speed_10m",
        "wind_gusts_10m",
        "et0_fao_evapotranspiration",
        "vapour_pressure_deficit",
        "soil_temperature_0cm",
        "soil_temperature_6cm",
        "soil_temperature_18cm",
        "soil_temperature_54cm",
        "soil_moisture_0_to_1cm",
        "soil_moisture_1_to_3cm",
        "soil_moisture_3_to_9cm",
        "soil_moisture_9_to_27cm",
        "soil_moisture_27_to_81cm",
    ])

    daily_fields = ",".join([
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "precipitation_probability_max",
        "wind_speed_10m_max",
        "et0_fao_evapotranspiration",
    ])

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
            "wind_gusts_10m",
            "cloud_cover",
        ]),
        "hourly": hourly_fields,
        "daily": daily_fields,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        weather = response.json()

        weather["_status"] = "success"
        weather["_latitude"] = latitude
        weather["_longitude"] = longitude

        return weather

    except Exception as error:
        return {
            "_status": "failed",
            "_error": str(error),
            "_latitude": latitude,
            "_longitude": longitude,
        }


def numeric_values(values: Any) -> List[float]:
    """
    Extract finite numeric values.
    """
    if not isinstance(values, list):
        return []

    result = []

    for value in values:
        try:
            number = float(value)

            if math.isfinite(number):
                result.append(number)

        except Exception:
            continue

    return result


def weather_statistics(weather: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a compact but useful weather summary.
    Raw weather remains available separately.
    """

    if weather.get("_status") != "success":
        return weather

    hourly = weather.get("hourly", {})
    daily = weather.get("daily", {})
    current = weather.get("current", {})

    temperature = numeric_values(hourly.get("temperature_2m"))
    humidity = numeric_values(hourly.get("relative_humidity_2m"))
    precipitation = numeric_values(hourly.get("precipitation"))
    rain = numeric_values(hourly.get("rain"))
    rain_probability = numeric_values(
        hourly.get("precipitation_probability")
    )
    wind = numeric_values(hourly.get("wind_speed_10m"))
    gusts = numeric_values(hourly.get("wind_gusts_10m"))
    et0 = numeric_values(
        hourly.get("et0_fao_evapotranspiration")
    )
    vpd = numeric_values(
        hourly.get("vapour_pressure_deficit")
    )

    def mean(values: List[float]) -> Optional[float]:
        return round(sum(values) / len(values), 3) if values else None

    def maximum(values: List[float]) -> Optional[float]:
        return round(max(values), 3) if values else None

    def minimum(values: List[float]) -> Optional[float]:
        return round(min(values), 3) if values else None

    summary = {
        "status": "success",
        "timezone": weather.get("timezone"),
        "latitude": weather.get("latitude"),
        "longitude": weather.get("longitude"),
        "current": current,
        "daily_forecast": daily,
        "hourly_statistics": {
            "temperature_min": minimum(temperature),
            "temperature_max": maximum(temperature),
            "temperature_mean": mean(temperature),
            "humidity_mean": mean(humidity),
            "precipitation_total_mm": round(
                sum(precipitation), 3
            ) if precipitation else 0,
            "rain_total_mm": round(
                sum(rain), 3
            ) if rain else 0,
            "maximum_rain_probability": maximum(rain_probability),
            "wind_speed_mean": mean(wind),
            "maximum_wind_speed": maximum(wind),
            "maximum_wind_gust": maximum(gusts),
            "et0_total": round(sum(et0), 3) if et0 else 0,
            "vpd_mean": mean(vpd),
        },
    }

    # Include all available daily fields.
    # This keeps the future Groq prompt informed without dumping
    # every hourly record into the main textual context.
    return summary


# ============================================================
# HYBRID AGRICULTURAL RETRIEVER
# ============================================================

class AgriculturalRetriever:
    def __init__(
        self,
        use_reranker: bool = True,
        dense_k: int = 100,
        bm25_k: int = 100,
        metadata_k: int = 100,
        final_context_chunks: int = 35,
        max_context_chars: int = 70000,
    ):
        print("\n" + "=" * 70)
        print("INITIALIZING FINAL AGRICULTURAL RETRIEVER")
        print("=" * 70)

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.metadata_k = metadata_k
        self.final_context_chunks = final_context_chunks
        self.max_context_chars = max_context_chars

        print("\nLoading embeddings...")
        self.embeddings = np.load(EMBEDDING_FILE).astype(
            np.float32
        )

        print(f"Loaded embeddings: {self.embeddings.shape}")

        print("\nLoading embedding records...")
        self.records = load_embedding_records()

        print(f"Loaded records: {len(self.records)}")

        if len(self.records) != len(self.embeddings):
            raise ValueError(
                "Embedding and record count mismatch: "
                f"{len(self.embeddings)} embeddings versus "
                f"{len(self.records)} records"
            )

        self.ids = [
            deterministic_id(record, index)
            for index, record in enumerate(self.records)
        ]

        self.id_to_index = {
            record_id: index
            for index, record_id in enumerate(self.ids)
        }

        print("\nPreparing BM25 search index...")

        self.search_texts = [
            self.record_search_text(record)
            for record in self.records
        ]

        self.bm25 = BM25Okapi([
            tokenize(text)
            for text in self.search_texts
        ])

        print("BM25 index created successfully.")

        print("\nLoading embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print("Embedding model loaded.")

        self.reranker = None

        if use_reranker:
            try:
                print("\nLoading optional reranker...")

                from sentence_transformers import CrossEncoder

                self.reranker = CrossEncoder(
                    RERANKER_MODEL_NAME,
                    max_length=512
                )

                print("Reranker loaded successfully.")

            except Exception as error:
                print(
                    "Reranker unavailable. Continuing with light "
                    f"fusion ranking. Reason: {error}"
                )

        print("\nInitializing ChromaDB...")

        VECTOR_STORE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.chroma_client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_DIR)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        print(
            f"ChromaDB collection: "
            f"{self.collection.name}"
        )

        existing_count = self.collection.count()

        print(
            f"Existing ChromaDB records: "
            f"{existing_count}"
        )

        # Upsert exact records and exact saved embeddings.
        # This guarantees ChromaDB alignment.
        self.collection.upsert(
            ids=self.ids,
            embeddings=self.embeddings.tolist(),
            documents=[
                record["content"]
                for record in self.records
            ],
            metadatas=[
                record["metadata"]
                for record in self.records
            ]
        )

        print(
            f"ChromaDB records after upsert: "
            f"{self.collection.count()}"
        )

        print("\nRetriever initialized successfully.")
        print(f"Records loaded: {len(self.records)}")
        print(f"Embedding shape: {self.embeddings.shape}")
        print(f"Embedding model: {EMBEDDING_MODEL_NAME}")

    def record_search_text(
        self,
        record: Dict[str, Any]
    ) -> str:
        """
        Combine content and important metadata for BM25.
        """

        content = record.get("content", "")
        metadata = record.get("metadata", {})

        metadata_parts = []

        for key in [
            "source_file",
            "page",
            "page_display",
            "crop",
            "crops",
            "state",
            "district",
            "region",
            "topics",
            "topic",
            "source_authority",
            "document_type",
            "season",
        ]:
            if key in metadata:
                metadata_parts.append(
                    f"{key} {safe_text(metadata[key])}"
                )

        return clean_text(
            content + " " + " ".join(metadata_parts)
        )

    def build_metadata_terms(
        self,
        query: str,
        farmer_context: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Extract crop, state, topic, soil, and region terms.
        No primary filtering by year or date is performed.
        """

        flat_context = flatten_dictionary(farmer_context)
        combined = clean_text(
            query + " " + " ".join(flat_context.values())
        ).lower()

        detected_crops = []
        detected_states = []
        detected_topics = []

        for canonical, aliases in CROP_ALIASES.items():
            if any(alias.lower() in combined for alias in aliases):
                detected_crops.append(canonical)

        for canonical, aliases in STATE_ALIASES.items():
            if any(alias.lower() in combined for alias in aliases):
                detected_states.append(canonical)

        for topic, terms in TOPIC_TERMS.items():
            if any(term.lower() in combined for term in terms):
                detected_topics.append(topic)

        return {
            "crops": detected_crops,
            "states": detected_states,
            "topics": detected_topics,
            "soil_terms": [
                term
                for term in [
                    "black soil",
                    "red soil",
                    "loam",
                    "clay",
                    "sandy",
                    "soil moisture",
                    "soil fertility",
                    "drainage",
                    "soil test",
                    "ph"
                ]
                if term in combined
            ],
            "irrigation_terms": [
                term
                for term in [
                    "borewell",
                    "drip",
                    "sprinkler",
                    "irrigation",
                    "rainfed",
                    "water source"
                ]
                if term in combined
            ],
        }

    def expand_query(
        self,
        query: str,
        farmer_context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, List[str]]]:
        """
        Create an agriculture-aware expanded query.
        """

        terms = self.build_metadata_terms(
            query,
            farmer_context
        )

        flat_context = flatten_dictionary(farmer_context)

        context_text = " ".join(
            f"{key} {value}"
            for key, value in flat_context.items()
            if value
        )

        alias_text = []

        for crop in terms["crops"]:
            alias_text.extend(
                CROP_ALIASES.get(crop, [])
            )

        for state in terms["states"]:
            alias_text.extend(
                STATE_ALIASES.get(state, [])
            )

        expanded = clean_text(
            query
            + " "
            + context_text
            + " "
            + " ".join(alias_text)
            + " "
            + " ".join(terms["topics"])
            + " "
            + " ".join(terms["soil_terms"])
            + " "
            + " ".join(terms["irrigation_terms"])
        )

        return expanded, terms

    def dense_search(
        self,
        expanded_query: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Dense vector search using the existing ChromaDB collection.
        """

        query_text = (
            "Represent this sentence for searching relevant passages: "
            + expanded_query
        )

        query_embedding = self.embedding_model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        result = self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=min(
                self.dense_k,
                self.collection.count()
            ),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        candidates = {}

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for rank, record_id in enumerate(ids, start=1):
            index = self.id_to_index.get(record_id)

            if index is None:
                continue

            distance = (
                float(distances[rank - 1])
                if rank - 1 < len(distances)
                else 1.0
            )

            candidates[record_id] = {
                "id": record_id,
                "index": index,
                "content": documents[rank - 1],
                "metadata": metadatas[rank - 1],
                "dense_rank": rank,
                "dense_distance": distance,
                "bm25_rank": None,
                "metadata_rank": None,
                "rrf_score": 0.0,
            }

        return candidates

    def bm25_search(
        self,
        expanded_query: str
    ) -> Dict[str, int]:
        """
        BM25 search over content plus metadata.
        """

        scores = self.bm25.get_scores(
            tokenize(expanded_query)
        )

        ranked_indices = np.argsort(
            scores
        )[::-1][:self.bm25_k]

        result = {}

        for rank, index in enumerate(ranked_indices, start=1):
            record_id = self.ids[int(index)]
            result[record_id] = rank

        return result

    def metadata_search(
        self,
        terms: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """
        Metadata-aware search.

        This is deliberately a soft ranking signal instead of a hard
        Chroma filter because metadata fields may be stored as strings
        containing multiple crops, topics, or regions.
        """

        scored = []

        for index, record in enumerate(self.records):
            metadata = record.get("metadata", {})

            metadata_text = " ".join(
                safe_text(value).lower()
                for value in metadata.values()
            )

            content_text = record.get(
                "content",
                ""
            ).lower()

            searchable = metadata_text + " " + content_text

            score = 0.0

            for crop in terms.get("crops", []):
                aliases = CROP_ALIASES.get(crop, [crop])

                if any(alias.lower() in searchable for alias in aliases):
                    score += 5.0

            for state in terms.get("states", []):
                aliases = STATE_ALIASES.get(state, [state])

                if any(alias.lower() in searchable for alias in aliases):
                    score += 3.0

            for topic in terms.get("topics", []):
                topic_terms = TOPIC_TERMS.get(topic, [topic])

                if any(
                    topic_term.lower() in searchable
                    for topic_term in topic_terms
                ):
                    score += 1.5

            for soil_term in terms.get("soil_terms", []):
                if soil_term.lower() in searchable:
                    score += 1.5

            for irrigation_term in terms.get(
                "irrigation_terms",
                []
            ):
                if irrigation_term.lower() in searchable:
                    score += 1.0

            if score > 0:
                scored.append(
                    (index, score)
                )

        scored.sort(
            key=lambda item: item[1],
            reverse=True
        )

        result = {}

        for rank, (index, _) in enumerate(
            scored[:self.metadata_k],
            start=1
        ):
            result[self.ids[index]] = rank

        return result

    def reciprocal_rank_fusion(
        self,
        dense_candidates: Dict[str, Dict[str, Any]],
        bm25_ranks: Dict[str, int],
        metadata_ranks: Dict[str, int]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fuse dense, BM25, and metadata ranking using RRF.
        """

        fused = dict(dense_candidates)

        for record_id in bm25_ranks:
            if record_id not in fused:
                index = self.id_to_index[record_id]

                fused[record_id] = {
                    "id": record_id,
                    "index": index,
                    "content": self.records[index]["content"],
                    "metadata": self.records[index]["metadata"],
                    "dense_rank": None,
                    "dense_distance": None,
                    "bm25_rank": None,
                    "metadata_rank": None,
                    "rrf_score": 0.0,
                }

        for record_id in metadata_ranks:
            if record_id not in fused:
                index = self.id_to_index[record_id]

                fused[record_id] = {
                    "id": record_id,
                    "index": index,
                    "content": self.records[index]["content"],
                    "metadata": self.records[index]["metadata"],
                    "dense_rank": None,
                    "dense_distance": None,
                    "bm25_rank": None,
                    "metadata_rank": None,
                    "rrf_score": 0.0,
                }

        for record_id, item in fused.items():
            dense_rank = item.get("dense_rank")
            bm25_rank = bm25_ranks.get(record_id)
            metadata_rank = metadata_ranks.get(record_id)

            score = 0.0

            if dense_rank is not None:
                score += 1.0 / (60.0 + dense_rank)

            if bm25_rank is not None:
                score += 1.0 / (60.0 + bm25_rank)

            if metadata_rank is not None:
                score += 1.0 / (60.0 + metadata_rank)

            item["bm25_rank"] = bm25_rank
            item["metadata_rank"] = metadata_rank
            item["rrf_score"] = score

        return dict(
            sorted(
                fused.items(),
                key=lambda item: item[1]["rrf_score"],
                reverse=True
            )
        )

    def rerank_candidates(
        self,
        expanded_query: str,
        fused_candidates: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Light reranking.

        Only the top 40 fused candidates are reranked.
        This keeps ranking limited while preserving a large context.
        """

        candidates = list(
            fused_candidates.values()
        )

        rerank_pool = candidates[:40]

        if not rerank_pool:
            return []

        if self.reranker is None:
            for item in rerank_pool:
                item["final_score"] = item["rrf_score"]

            return sorted(
                rerank_pool,
                key=lambda item: item["final_score"],
                reverse=True
            )

        pairs = [
            [
                expanded_query,
                item["content"]
            ]
            for item in rerank_pool
        ]

        try:
            rerank_scores = self.reranker.predict(
                pairs,
                show_progress_bar=False
            )

            rerank_scores = [
                float(score)
                for score in rerank_scores
            ]

            min_score = min(rerank_scores)
            max_score = max(rerank_scores)

            score_range = max_score - min_score

            for item, score in zip(
                rerank_pool,
                rerank_scores
            ):
                if score_range > 0:
                    normalized = (
                        score - min_score
                    ) / score_range
                else:
                    normalized = 1.0

                item["reranker_score"] = score

                # Give more importance to the reranker,
                # but retain the hybrid retrieval signal.
                item["final_score"] = (
                    0.70 * normalized
                    + 0.30 * item["rrf_score"]
                )

            return sorted(
                rerank_pool,
                key=lambda item: item["final_score"],
                reverse=True
            )

        except Exception as error:
            print(
                f"Reranking failed. Using RRF ranking: {error}"
            )

            for item in rerank_pool:
                item["final_score"] = item["rrf_score"]

            return sorted(
                rerank_pool,
                key=lambda item: item["final_score"],
                reverse=True
            )

    def cosine_similarity(
        self,
        vector_a: np.ndarray,
        vector_b: np.ndarray
    ) -> float:
        """
        Cosine similarity for normalized vectors.
        """

        denominator = (
            np.linalg.norm(vector_a)
            * np.linalg.norm(vector_b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(vector_a, vector_b)
            / denominator
        )

    def select_diverse_context(
        self,
        ranked_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select a large, diverse final context.

        Diversity rules:
        - Maximum 5 chunks from one PDF
        - Maximum 2 chunks from one page
        - Prefer different source documents
        - Prefer different topics and crops
        - Stop at chunk count or character budget
        """

        selected = []

        source_counts = {}
        page_counts = {}

        for candidate in ranked_candidates:
            metadata = candidate.get("metadata", {})

            source_file = safe_text(
                metadata.get(
                    "source_file",
                    "unknown_source"
                )
            )

            page = safe_text(
                metadata.get(
                    "page_display",
                    metadata.get("page", "unknown_page")
                )
            )

            source_key = source_file
            page_key = f"{source_file}|{page}"

            if source_counts.get(source_key, 0) >= 5:
                continue

            if page_counts.get(page_key, 0) >= 2:
                continue

            selected.append(candidate)

            source_counts[source_key] = (
                source_counts.get(source_key, 0) + 1
            )

            page_counts[page_key] = (
                page_counts.get(page_key, 0) + 1
            )

            current_chars = sum(
                len(item.get("content", ""))
                for item in selected
            )

            if (
                len(selected) >= self.final_context_chunks
                or current_chars >= self.max_context_chars
            ):
                break

        return selected

    def citation_for(
        self,
        item: Dict[str, Any]
    ) -> str:
        """
        Create a reliable PDF citation label.
        """

        metadata = item.get("metadata", {})

        source_file = safe_text(
            metadata.get(
                "source_file",
                "unknown_source"
            )
        )

        page = safe_text(
            metadata.get(
                "page_display",
                metadata.get("page", "unknown_page")
            )
        )

        authority = safe_text(
            metadata.get(
                "source_authority",
                "Unknown"
            )
        )

        document_type = safe_text(
            metadata.get(
                "document_type",
                "agricultural_document"
            )
        )

        return (
            f"{source_file}, page {page}, "
            f"{authority}, {document_type}"
        )

    def build_context_text(
        self,
        selected: List[Dict[str, Any]]
    ) -> str:
        """
        Build the evidence context passed to Groq.
        """

        blocks = []

        for number, item in enumerate(
            selected,
            start=1
        ):
            citation = self.citation_for(item)

            metadata = item.get("metadata", {})

            crop = safe_text(
                metadata.get("crop", "")
            )

            state = safe_text(
                metadata.get("state", "")
            )

            topics = safe_text(
                metadata.get("topics", "")
            )

            block = (
                f"[EVIDENCE {number}]\n"
                f"Source: {citation}\n"
                f"Crop metadata: {crop}\n"
                f"State/region metadata: {state}\n"
                f"Topics: {topics}\n"
                f"Content:\n{item.get('content', '')}\n"
            )

            blocks.append(block)

        return "\n\n".join(blocks)

    def retrieve(
        self,
        query: str,
        farmer_context: Dict[str, Any],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Complete retrieval operation.
        """

        print("\n" + "=" * 70)
        print("HYBRID AGRICULTURAL RETRIEVAL")
        print("=" * 70)

        expanded_query, terms = self.expand_query(
            query,
            farmer_context
        )

        print(f"\nOriginal query: {query}")
        print(f"\nExpanded query: {expanded_query}")

        weather_raw = {}

        if latitude is not None and longitude is not None:
            print("\nFetching Open-Meteo weather...")

            weather_raw = fetch_open_meteo(
                latitude=latitude,
                longitude=longitude,
                forecast_days=16
            )

            if weather_raw.get("_status") == "success":
                print("Open-Meteo weather loaded successfully.")
            else:
                print(
                    "Open-Meteo failed. Continuing without live "
                    f"weather: {weather_raw.get('_error')}"
                )

        print("\nRunning dense vector search...")

        dense_candidates = self.dense_search(
            expanded_query
        )

        print(
            f"Vector candidates: "
            f"{len(dense_candidates)}"
        )

        print("\nRunning BM25 search...")

        bm25_ranks = self.bm25_search(
            expanded_query
        )

        print(
            f"BM25 candidates: "
            f"{len(bm25_ranks)}"
        )

        print("\nRunning metadata-aware search...")

        metadata_ranks = self.metadata_search(
            terms
        )

        print(
            f"Metadata candidates: "
            f"{len(metadata_ranks)}"
        )

        print("\nFusing retrieval results...")

        fused = self.reciprocal_rank_fusion(
            dense_candidates=dense_candidates,
            bm25_ranks=bm25_ranks,
            metadata_ranks=metadata_ranks
        )

        print(
            f"Fused candidates: "
            f"{len(fused)}"
        )

        print("\nApplying limited reranking...")

        reranked = self.rerank_candidates(
            expanded_query=expanded_query,
            fused_candidates=fused
        )

        print(
            f"Reranked candidates: "
            f"{len(reranked)}"
        )

        print("\nSelecting diverse large context...")

        selected = self.select_diverse_context(
            reranked
        )

        context_text = self.build_context_text(
            selected
        )

        citations = [
            self.citation_for(item)
            for item in selected
        ]

        weather_summary = weather_statistics(
            weather_raw
        )

        unique_sources = sorted(
            set(
                safe_text(
                    item.get("metadata", {}).get(
                        "source_file",
                        "unknown_source"
                    )
                )
                for item in selected
            )
        )

        print("\nRetrieval completed.")
        print(f"Final context chunks: {len(selected)}")
        print(f"Context characters: {len(context_text)}")
        print(f"Unique source PDFs: {len(unique_sources)}")

        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "detected_terms": terms,
            "farmer_context": farmer_context,
            "weather_raw": weather_raw,
            "weather_summary": weather_summary,
            "selected_results": selected,
            "context_text": context_text,
            "citations": citations,
            "unique_sources": unique_sources,
        }


# ============================================================
# ML MODEL WRAPPER
# ============================================================

class AgriculturalModels:
    def __init__(self):
        print("\n" + "=" * 70)
        print("LOADING TRAINED AGRICULTURAL MODELS")
        print("=" * 70)

        self.production_model = None
        self.crop_model = None

        if PRODUCTION_MODEL_FILE.exists():
            self.production_model = joblib.load(
                PRODUCTION_MODEL_FILE
            )
            print(
                "Loaded production/yield model:"
                f" {PRODUCTION_MODEL_FILE.name}"
            )
        else:
            print(
                "Production model not found:"
                f" {PRODUCTION_MODEL_FILE}"
            )

        if CROP_RECOMMENDATION_MODEL_FILE.exists():
            self.crop_model = joblib.load(
                CROP_RECOMMENDATION_MODEL_FILE
            )
            print(
                "Loaded crop recommendation model:"
                f" {CROP_RECOMMENDATION_MODEL_FILE.name}"
            )
        else:
            print(
                "Crop recommendation model not found:"
                f" {CROP_RECOMMENDATION_MODEL_FILE}"
            )

    def safe_prediction(
        self,
        model: Any,
        features: Any
    ) -> Any:
        """
        Run prediction safely.
        Tries DataFrame first, then falls back to list format.
        """

        if model is None:
            return None

        try:
            prediction = model.predict(features)

            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()

            return prediction

        except Exception as first_error:
            try:
                if isinstance(features, pd.DataFrame):
                    prediction = model.predict(
                        features.to_numpy()
                    )

                    if isinstance(prediction, np.ndarray):
                        prediction = prediction.tolist()

                    return prediction

            except Exception as second_error:
                return {
                    "error": (
                        f"DataFrame prediction failed: {first_error}; "
                        f"array prediction failed: {second_error}"
                    )
                }

            return {
                "error": str(first_error)
            }

    def predict_production(
        self,
        farmer_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Production model trained on:
        state_name, district_name, crop_year, season,
        crop, area

        Important:
        This is a historical production/yield model.
        It is not a complete agronomic suitability model.
        """

        if self.production_model is None:
            return {
                "status": "unavailable",
                "reason": "Production model file not found."
            }

        flat = flatten_dictionary(
            farmer_context
        )

        state = (
            flat.get("state")
            or flat.get("state_name")
            or ""
        )

        district = (
            flat.get("district")
            or flat.get("district_name")
            or ""
        )

        season = flat.get("season", "")
        crop = (
            flat.get("crop")
            or flat.get("crop_name")
            or ""
        )

        area = (
            flat.get("area")
            or flat.get("land_area")
            or 0
        )

        crop_year = flat.get(
            "crop_year",
            2026
        )

        try:
            area = float(area)
        except Exception:
            area = 0.0

        try:
            crop_year = int(float(crop_year))
        except Exception:
            crop_year = 2026

        features = pd.DataFrame([{
          "state_name": state,
          "district_name": district,
          "crop_year": crop_year,
          "season": season,
          "crop": crop,
          "area": area,
         }])

        prediction = self.safe_prediction(
           self.production_model,
           features
         ) 

        return {
            "status": "success",
            "model": PRODUCTION_MODEL_FILE.name,
            "features": {
                "state_name": state,
                "district_name": district,
                "crop_year": crop_year,
                "season": season,
                "crop": crop,
                "area": area,
            },
            "prediction": prediction,
            "interpretation": (
                "Historical production/yield estimate. "
                "Use with soil, weather, irrigation, and RAG evidence."
            ),
        }

    def predict_crop_recommendation(
        self,
        farmer_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Crop recommendation model trained on:
        n, p, k, temperature, humidity, ph, rainfall
        """

        if self.crop_model is None:
            return {
                "status": "unavailable",
                "reason": "Crop recommendation model file not found."
            }

        flat = flatten_dictionary(
            farmer_context
        )

        def get_number(*keys, default=0.0):
            for key in keys:
                if key in flat:
                    try:
                        return float(flat[key])
                    except Exception:
                        pass

            return default

        nitrogen = get_number(
            "nitrogen",
            "n",
            "soil_nitrogen",
            default=0.0
        )

        phosphorus = get_number(
            "phosphorus",
            "p",
            "soil_phosphorus",
            default=0.0
        )

        potassium = get_number(
            "potassium",
            "k",
            "soil_potassium",
            default=0.0
        )

        temperature = get_number(
            "temperature",
            "temperature_2m",
            "current_temperature",
            default=0.0
        )

        humidity = get_number(
            "humidity",
            "relative_humidity",
            "relative_humidity_2m",
            default=0.0
        )

        ph = get_number(
            "ph",
            "soil_ph",
            default=0.0
        )

        rainfall = get_number(
            "rainfall",
            "precipitation",
            "rainfall_mm",
            default=0.0
        )

        features = pd.DataFrame([{
         "n": nitrogen,
         "p": phosphorus,
         "k": potassium,
         "temperature": temperature,
         "humidity": humidity,
         "ph": ph,
         "rainfall": rainfall,
         }])

        prediction = self.safe_prediction(
          self.crop_model,
          features
        )

        probabilities = None

        try:
           probabilities = self.crop_model.predict_proba(
           features
              ).tolist()
        except Exception:
            probabilities = None

        return {
            "status": "success",
            "model": CROP_RECOMMENDATION_MODEL_FILE.name,
            "features": {
                "n": nitrogen,
                "p": phosphorus,
                "k": potassium,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall,
            },
            "prediction": prediction,
            "probabilities": probabilities,
            "important_limitation": (
                "This model only knows the crop classes present in "
                "its training dataset. It does not directly predict "
                "every Indian crop, including crops absent from training."
            ),
        }

    def run_all(
        self,
        farmer_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "production_model": self.predict_production(
                farmer_context
            ),
            "crop_recommendation_model": (
                self.predict_crop_recommendation(
                    farmer_context
                )
            ),
        }


# ============================================================
# GROQ AGRICULTURAL SCIENTIST
# ============================================================

def create_groq_prompt(
    farmer_context: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    ml_result: Dict[str, Any]
) -> str:
    """
    Strong evidence-grounded agricultural scientist prompt.
    """

    farmer_json = json.dumps(
        farmer_context,
        ensure_ascii=False,
        indent=2
    )

    weather_json = json.dumps(
        retrieval_result.get("weather_summary", {}),
        ensure_ascii=False,
        indent=2
    )

    ml_json = json.dumps(
        ml_result,
        ensure_ascii=False,
        indent=2
    )

    evidence_context = retrieval_result.get(
        "context_text",
        ""
    )

    return f"""
You are an experienced Indian agricultural scientist with around
30 years of practical knowledge in crop planning, soil science,
agrometeorology, irrigation, crop physiology, plant pathology,
farm economics, and region-specific Indian agriculture.

Your task is to prepare a practical and scientifically reasoned
"what should the farmer plan now?" report.

You must reason using the following priority order:

1. Farmer and plot details
2. Exact latitude and longitude
3. Live Open-Meteo weather
4. Retrieved ICAR, KCC, agricultural advisory, and weather evidence
5. Trained ML model outputs
6. General Indian agricultural scientific reasoning

The RAG evidence is the main decision-support knowledge.
The ML models are supporting signals, not absolute truth.
Never blindly follow an ML prediction when it conflicts with:
- Soil conditions
- Water availability
- Weather
- Crop season
- Regional suitability
- Farmer's actual crop stage
- Reliable retrieved agricultural evidence

Do not use Bhoonidhi.
Do not use VEDAS.
Do not invent market prices, disease outbreaks, pesticide doses,
fertilizer doses, crop varieties, or yield values.

If the retrieved evidence does not support a specific recommendation,
clearly state:
"Evidence is insufficient for this specific recommendation."

For pesticide or chemical recommendations:
- Prefer prevention, monitoring, sanitation, drainage, and cultural control.
- Do not provide a chemical recommendation merely because a disease
  is theoretically possible.
- If a chemical is mentioned in the evidence, state that the farmer
  must verify the current label, crop registration, local advisory,
  and dosage with the nearest KVK or agriculture officer.
- Never recommend unsafe mixing or unverified pesticide combinations.

You must:
1. Identify the farmer's immediate planning objective.
2. Analyze soil, rainfall, temperature, humidity, irrigation,
   drainage, crop season, crop stage, and regional conditions.
3. Compare the ML outputs with the RAG evidence.
4. Select a short list of suitable crops or crop options.
5. Explain why each selected crop is suitable.
6. Explain why some crops should not be selected now.
7. Give immediate starting guidance for the selected crop or crops.
8. Give practical first-stage farming recommendations.
9. Mention water and drainage requirements.
10. Mention soil preparation and soil-testing needs.
11. Mention crop monitoring and disease/pest warning signs.
12. Mention uncertainty and missing data.
13. Cite every important evidence-based claim using the exact
    evidence citation format supplied in the context.

Use simple language that an Indian farmer can understand.
Do not write like a research paper.
Do not make the answer vague.
Do not give a huge list of crops without ranking them.

Return the report in this structure:

A. Farmer situation summary
B. Immediate recommendation
C. Ranked crop options
D. Why the selected crops were chosen
E. Why other options were rejected or kept lower
F. First 7-15 days action plan
G. Soil and irrigation guidance
H. Weather-related precautions
I. Disease and pest monitoring
J. ML model interpretation
K. Evidence and citations
L. Important limitations and safety note

FARMER DETAILS:
{farmer_json}

LIVE WEATHER SUMMARY:
{weather_json}

ML MODEL OUTPUTS:
{ml_json}

RETRIEVED AGRICULTURAL EVIDENCE:
{evidence_context}
"""


def call_groq(
    prompt: str,
    model_name: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Call Groq using GROQ_API_KEY from .env or environment.
    """

    try:
        from groq import Groq
    except Exception as error:
        return (
            "Groq package is not installed. "
            f"Install it using pip. Details: {error}"
        )

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "GROQ_API_KEY was not found. "
            "Add it to the .env file or export it in Linux."
        )

    try:
        client = Groq(
            api_key=api_key
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful, evidence-grounded "
                        "Indian agricultural scientist."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_tokens=7000,
        )

        return response.choices[0].message.content

    except Exception as error:
        return f"Groq request failed: {error}"


# ============================================================
# FARMER INPUT
# ============================================================

def ask_float(
    message: str,
    default: Optional[float] = None
) -> Optional[float]:
    while True:
        value = input(message).strip()

        if not value and default is not None:
            return default

        try:
            return float(value)
        except Exception:
            print("Please enter a valid number.")


def ask_int(
    message: str,
    default: Optional[int] = None
) -> Optional[int]:
    while True:
        value = input(message).strip()

        if not value and default is not None:
            return default

        try:
            return int(value)
        except Exception:
            print("Please enter a valid whole number.")




# ============================================================
# INTERACTIVE FARM LOCATION MAP
# ============================================================

def open_map_window():
    """
    Open an interactive map.

    The farmer clicks the exact farm location.
    Latitude and longitude are extracted automatically.
    """

    global selected_latitude
    global selected_longitude
    global map_widget
    global map_window

    if map_window is not None and map_window.winfo_exists():
        map_window.lift()
        map_window.focus_force()
        return

    map_window = tk.Toplevel(root)
    map_window.title("Select Farm Location")
    map_window.geometry("1050x720")
    map_window.minsize(800, 600)

    instruction = ttk.Label(
        map_window,
        text=(
            "Zoom into your farm and click the exact location. "
            "The latitude and longitude will be selected automatically."
        ),
        font=("Arial", 11)
    )
    instruction.pack(pady=8)

    map_widget = TkinterMapView(
        map_window,
        width=1000,
        height=600,
        corner_radius=0
    )
    map_widget.pack(
        fill="both",
        expand=True,
        padx=8,
        pady=5
    )

    # Default center: Karnataka / central India
    map_widget.set_position(
        15.3173,
        75.7139
    )
    map_widget.set_zoom(6)

    def select_location(coords):
        global selected_latitude
        global selected_longitude

        selected_latitude = float(coords[0])
        selected_longitude = float(coords[1])

        latitude_text.set(
            f"{selected_latitude:.7f}"
        )

        longitude_text.set(
            f"{selected_longitude:.7f}"
        )

        # Remove previous marker and place a new marker
        map_widget.delete_all_marker()

        map_widget.set_marker(
            selected_latitude,
            selected_longitude,
            text="Selected Farm Location"
        )

        location_status_text.set(
            "Farm location selected successfully"
        )

        messagebox.showinfo(
            "Location Selected",
            (
                f"Latitude: {selected_latitude:.7f}\n"
                f"Longitude: {selected_longitude:.7f}"
            ),
            parent=map_window
        )

    map_widget.add_left_click_map_command(
        select_location
    )

    def close_map():
        global map_window
        map_window.destroy()
        map_window = None

    button_frame = ttk.Frame(map_window)
    button_frame.pack(pady=8)

    ttk.Button(
        button_frame,
        text="Use Selected Location",
        command=close_map
    ).grid(
        row=0,
        column=0,
        padx=8
    )

    ttk.Button(
        button_frame,
        text="Close Map",
        command=close_map
    ).grid(
        row=0,
        column=1,
        padx=8
    )

    map_window.protocol(
        "WM_DELETE_WINDOW",
        close_map
    )




# ============================================================
# FARMER DATA COLLECTION
# ============================================================

def ask_required(prompt_text):
    while True:
        value = simpledialog.askstring(
            "Farmer Details",
            prompt_text,
            parent=root
        )

        if value is None:
            raise KeyboardInterrupt

        value = value.strip()

        if value:
            return value

        messagebox.showwarning(
            "Required Field",
            "Please enter a value.",
            parent=root
        )


def ask_optional(prompt_text):
    value = simpledialog.askstring(
        "Farmer Details",
        prompt_text,
        parent=root
    )

    if value is None:
        return ""

    return value.strip()


def collect_farmer_details():
    """
    Collect farmer and farm details.

    Latitude and longitude are taken automatically
    from the interactive map.
    """

    global selected_latitude
    global selected_longitude

    try:
        if (
            selected_latitude is None
            or selected_longitude is None
        ):
            messagebox.showwarning(
                "Location Required",
                (
                    "Please open the map and select the exact "
                    "farm location before starting the analysis."
                ),
                parent=root
            )
            return None

        farmer = {
            "name": ask_required(
                "Enter farmer name:"
            ),

            "phone": ask_optional(
                "Enter phone number, optional:"
            ),

            "email": ask_optional(
                "Enter email, optional:"
            ),

            "farm_name": ask_required(
                "Enter farm name:"
            ),

            "state": ask_required(
                "Enter state:"
            ),

            "district": ask_required(
                "Enter district:"
            ),

            "taluk": ask_optional(
                "Enter taluk or tehsil, optional:"
            ),

            "village": ask_optional(
                "Enter village, optional:"
            ),

            "land_area_acres": float(
                ask_required(
                    "Enter total farm area in acres:"
                )
            ),

            "plot_area_acres": float(
                ask_required(
                    "Enter plot area in acres:"
                )
            ),

            "soil_type": ask_optional(
                (
                    "Enter known soil type, for example "
                    "black, red, sandy, clay:"
                )
            ),

            "soil_colour": ask_optional(
                "Enter soil colour:"
            ),

            "soil_ph": ask_optional(
                "Enter soil pH if known:"
            ),

            "nitrogen": ask_optional(
                "Enter soil nitrogen value if known:"
            ),

            "phosphorus": ask_optional(
                "Enter soil phosphorus value if known:"
            ),

            "potassium": ask_optional(
                "Enter soil potassium value if known:"
            ),

            "water_availability": ask_optional(
                (
                    "Enter water availability: "
                    "low, medium, or high:"
                )
            ),

            "water_source": ask_optional(
                (
                    "Enter water source: "
                    "borewell, canal, rainfed, tank, etc.:"
                )
            ),

            "irrigation_method": ask_optional(
                (
                    "Enter irrigation method: "
                    "drip, sprinkler, flood, rainfed, etc.:"
                )
            ),

            "previous_crops": ask_optional(
                "Enter crops grown previously:"
            ),

            "previous_yield": ask_optional(
                "Enter previous crop yield if known:"
            ),

            "previous_problems": ask_optional(
                "Enter previous crop problems, optional:"
            ),

            "rainfall_pattern": ask_optional(
                "Average rainfall pattern: low, medium, or high:"
            ),

            "temperature_pattern": ask_optional(
                "Average temperature pattern: low, medium, or high:"
            ),

            "humidity_pattern": ask_optional(
                "Average humidity pattern: low, medium, or high:"
            ),

            "sunlight_pattern": ask_optional(
                "Average sunlight pattern: low, medium, or high:"
            ),

            "farmer_question": ask_optional(
                (
                    "What do you want to know now? "
                    "Press Enter for the default planning question:"
                )
            )
        }

        # Automatically attach map coordinates
        farmer["latitude"] = selected_latitude
        farmer["longitude"] = selected_longitude

        if not farmer["farmer_question"]:
            farmer["farmer_question"] = (
                "What should I plan now considering "
                "soil, weather, irrigation, crop suitability, "
                "disease risk, crop management, and expected yield?"
            )

        return farmer

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Area values must be valid numbers.",
            parent=root
        )
        return None

    except KeyboardInterrupt:
        return None


# ============================================================
# SAVE FINAL REPORT
# ============================================================

def save_report(
    report: str,
    retrieval_result: Dict[str, Any],
    ml_result: Dict[str, Any]
) -> Path:
    """
    Save the final report and supporting machine-readable context.
    """

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    from datetime import datetime

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_file = output_dir / (
        f"agri_final_report_{timestamp}.txt"
    )

    context_file = output_dir / (
        f"agri_final_context_{timestamp}.json"
    )

    report_file.write_text(
        report,
        encoding="utf-8"
    )

    serializable_retrieval = {
        "original_query": retrieval_result.get(
            "original_query"
        ),
        "expanded_query": retrieval_result.get(
            "expanded_query"
        ),
        "detected_terms": retrieval_result.get(
            "detected_terms"
        ),
        "weather_summary": retrieval_result.get(
            "weather_summary"
        ),
        "citations": retrieval_result.get(
            "citations"
        ),
        "unique_sources": retrieval_result.get(
            "unique_sources"
        ),
        "selected_results": [
            {
                "content": item.get("content"),
                "metadata": item.get("metadata"),
                "rrf_score": item.get("rrf_score"),
                "final_score": item.get("final_score"),
                "dense_rank": item.get("dense_rank"),
                "bm25_rank": item.get("bm25_rank"),
                "metadata_rank": item.get("metadata_rank"),
            }
            for item in retrieval_result.get(
                "selected_results",
                []
            )
        ],
    }

    context_payload = {
        "retrieval": serializable_retrieval,
        "ml_result": ml_result,
    }

    context_file.write_text(
        json.dumps(
            context_payload,
            ensure_ascii=False,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    return report_file


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("FINAL CMD-BASED SMART AGRICULTURAL AI SYSTEM")
    print("=" * 70)
    print("Bhoonidhi: Disabled")
    print("VEDAS: Disabled")
    print("Mode: Farmer details + Weather + RAG + ML + Groq")
    print("=" * 70)

    try:
        farmer_context, latitude, longitude = (
            collect_farmer_details()
        )

        query = farmer_context.get(
            "farmer_question",
            ""
        )

        if not query:
            query = (
                "What should I plan now considering soil, "
                "weather, irrigation, crop suitability, "
                "disease risk, and expected yield?"
            )

        retriever = AgriculturalRetriever(
            use_reranker=True,
            dense_k=100,
            bm25_k=100,
            metadata_k=100,
            final_context_chunks=35,
            max_context_chars=70000,
        )

        retrieval_result = retriever.retrieve(
            query=query,
            farmer_context=farmer_context,
            latitude=latitude,
            longitude=longitude,
        )

        # Add weather-derived values to ML input only when the
        # farmer did not provide them.
        weather_summary = retrieval_result.get(
            "weather_summary",
            {}
        )

        current_weather = weather_summary.get(
            "current",
            {}
        )

        farmer_context.setdefault(
            "weather",
            {}
        )

        farmer_context["weather"].update({
            "temperature": current_weather.get(
                "temperature_2m",
                farmer_context["weather"].get(
                    "temperature",
                    0.0
                )
            ),
            "humidity": current_weather.get(
                "relative_humidity_2m",
                farmer_context["weather"].get(
                    "humidity",
                    0.0
                )
            ),
            "precipitation": current_weather.get(
                "precipitation",
                farmer_context["weather"].get(
                    "precipitation",
                    0.0
                )
            ),
            "wind_speed": current_weather.get(
                "wind_speed_10m",
                farmer_context["weather"].get(
                    "wind_speed",
                    0.0
                )
            ),
        })

        models = AgriculturalModels()

        ml_result = models.run_all(
            farmer_context
        )

        prompt = create_groq_prompt(
            farmer_context=farmer_context,
            retrieval_result=retrieval_result,
            ml_result=ml_result
        )

        print("\n" + "=" * 70)
        print("CALLING GROQ AGRICULTURAL SCIENTIST")
        print("=" * 70)

        final_report = call_groq(
            prompt=prompt,
            model_name="llama-3.3-70b-versatile"
        )

        print("\n" + "=" * 70)
        print("FINAL AGRICULTURAL REPORT")
        print("=" * 70)
        print(final_report)

        report_file = save_report(
            report=final_report,
            retrieval_result=retrieval_result,
            ml_result=ml_result
        )

        print("\n" + "=" * 70)
        print("REPORT SAVED")
        print("=" * 70)
        print(report_file)

    except KeyboardInterrupt:
        print("\n\nProcess stopped by user.")

    except Exception as error:
        print("\n" + "=" * 70)
        print("SYSTEM ERROR")
        print("=" * 70)
        print(str(error))
        print("\nCheck:")
        print("1. Correct project path")
        print("2. Virtual environment")
        print("3. Required packages")
        print("4. GROQ_API_KEY")
        print("5. Embedding and model files")
        print("6. Internet connection for Open-Meteo and Groq")


if __name__ == "__main__":
    main()