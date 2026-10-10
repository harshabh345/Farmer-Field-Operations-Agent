#!/usr/bin/env python3

"""
Final CMD-Based Smart Agricultural AI System

Features:
- Interactive map-based latitude and longitude selection
- Farmer and plot data collection
- Open-Meteo weather data
- ChromaDB vector retrieval
- BM25 retrieval
- Metadata-aware retrieval
- Reciprocal Rank Fusion
- Optional cross-encoder reranking
- Diverse evidence selection
- Two trained XGBoost models
- Groq agricultural scientist reasoning
- Evidence-based farmer-friendly report
- JSON and TXT report saving

Excluded:
- Bhoonidhi
- VEDAS
- Market analysis
"""

from __future__ import annotations

import os
import re
import json
import math
import hashlib
import warnings
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import chromadb

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import tkinter as tk
from tkinter import ttk, messagebox
from tkintermapview import TkinterMapView

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path("/mnt/d/EdgeAi_resistor_Dl_model/RAG")

load_dotenv(PROJECT_ROOT / ".env")

DATA_ROOT = PROJECT_ROOT / "Data"
EMBEDDING_DIR = DATA_ROOT / "embeddings"
VECTOR_STORE_DIR = DATA_ROOT / "vector_store"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR = PROJECT_ROOT / "ml_training" / "models"

EMBEDDING_FILE = EMBEDDING_DIR / "agri_embeddings.npy"
RECORDS_FILE = EMBEDDING_DIR / "agri_embedding_records.jsonl"

PRODUCTION_MODEL_FILE = (
    MODEL_DIR / "agriculture_production_xgb.pkl"
)

CROP_MODEL_FILE = (
    MODEL_DIR / "crop_recommendation_xgb.pkl"
)

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

EMBEDDING_MODEL_NAME = (
    "BAAI/bge-small-en-v1.5"
)

RERANKER_MODEL_NAME = (
    "BAAI/bge-reranker-base"
)

CHROMA_COLLECTION_NAME = (
    "agri_documents_final_v1"
)

GROQ_MODEL_NAME = (
    "openai/gpt-oss-120b"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()


# ============================================================
# GLOBAL GUI STATE
# ============================================================

root: Optional[tk.Tk] = None
map_window: Optional[tk.Toplevel] = None
map_widget: Optional[TkinterMapView] = None

selected_latitude: Optional[float] = None
selected_longitude: Optional[float] = None

latitude_text: Optional[tk.StringVar] = None
longitude_text: Optional[tk.StringVar] = None
location_status_text: Optional[tk.StringVar] = None

result_text: Optional[tk.Text] = None
status_text: Optional[tk.StringVar] = None

analysis_button: Optional[ttk.Button] = None


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, dict):
        return " ".join(
            f"{key} {safe_text(item)}"
            for key, item in value.items()
        )

    if isinstance(value, (list, tuple, set)):
        return " ".join(
            safe_text(item)
            for item in value
        )

    return str(value).strip()


def clean_text(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        safe_text(value)
    ).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(
        r"[a-zA-Z0-9]+",
        clean_text(text).lower()
    )


def scalar_metadata(value: Any) -> Any:
    if value is None:
        return ""

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple, set)):
        return ", ".join(
            clean_text(item)
            for item in value
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False
        )

    return clean_text(value)


def flatten_dictionary(
    data: Dict[str, Any],
    prefix: str = ""
) -> Dict[str, str]:
    result = {}

    if not isinstance(data, dict):
        return result

    for key, value in data.items():
        current_key = (
            f"{prefix}_{key}"
            if prefix
            else str(key)
        )

        if isinstance(value, dict):
            result.update(
                flatten_dictionary(
                    value,
                    current_key
                )
            )
        else:
            result[current_key] = safe_text(value)

    return result


def deterministic_id(
    record: Dict[str, Any],
    index: int
) -> str:
    metadata = record.get("metadata", {})
    content = record.get("content", "")

    raw = "|".join([
        safe_text(metadata.get("chunk_id")),
        safe_text(metadata.get("source_file")),
        safe_text(metadata.get("page")),
        content,
        str(index)
    ])

    digest = hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()

    return f"agri_{digest}"


def finite_float(
    value: Any,
    default: float = 0.0
) -> float:
    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except Exception:
        pass

    return default


def finite_int(
    value: Any,
    default: int = 0
) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


# ============================================================
# AGRICULTURAL ALIASES
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
    "millet": [
        "millet",
        "bajra",
        "jowar",
        "sorghum"
    ],
    "onion": [
        "onion",
        "kanda"
    ],
    "potato": [
        "potato",
        "aloo"
    ],
    "tomato": [
        "tomato"
    ]
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
    "rajasthan": [
        "rajasthan",
        "rj"
    ],
    "tamil nadu": [
        "tamil nadu",
        "tn"
    ],
    "kerala": [
        "kerala",
        "kl"
    ],
    "odisha": [
        "odisha",
        "orissa",
        "od"
    ],
    "west bengal": [
        "west bengal",
        "wb"
    ],
    "bihar": [
        "bihar",
        "br"
    ]
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
    ]
}


# ============================================================
# RECORD LOADING
# ============================================================

def normalize_record(
    raw: Dict[str, Any]
) -> Dict[str, Any]:

    metadata = raw.get(
        "metadata",
        {}
    )

    if not isinstance(metadata, dict):
        metadata = {}

    content = (
        raw.get("page_content")
        or raw.get("content")
        or raw.get("text")
        or ""
    )

    content = clean_text(content)

    for key, value in raw.items():
        if key not in {
            "page_content",
            "content",
            "text",
            "metadata",
            "embedding"
        }:
            metadata.setdefault(
                key,
                value
            )

    metadata = {
        str(key): scalar_metadata(value)
        for key, value in metadata.items()
    }

    page = metadata.get("page", "")

    try:
        page_display = int(page) + 1
    except Exception:
        page_display = page

    metadata["page_display"] = scalar_metadata(
        page_display
    )

    return {
        "content": content,
        "metadata": metadata
    }


def load_embedding_records() -> List[Dict[str, Any]]:
    if not RECORDS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {RECORDS_FILE}"
        )

    records = []

    with RECORDS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

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
                    f"Warning: JSONL line "
                    f"{line_number} skipped: {error}"
                )

    if not records:
        raise ValueError(
            "No valid embedding records found."
        )

    return records


# ============================================================
# OPEN-METEO
# ============================================================

def fetch_open_meteo(
    latitude: float,
    longitude: float,
    forecast_days: int = 16
) -> Dict[str, Any]:

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
        "soil_moisture_27_to_81cm"
    ])

    daily_fields = ",".join([
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "precipitation_probability_max",
        "wind_speed_10m_max",
        "et0_fao_evapotranspiration"
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
            "cloud_cover"
        ]),
        "hourly": hourly_fields,
        "daily": daily_fields,
        "forecast_days": forecast_days,
        "timezone": "auto"
    }

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=40
        )

        response.raise_for_status()

        result = response.json()

        result["_status"] = "success"
        result["_latitude"] = latitude
        result["_longitude"] = longitude

        return result

    except Exception as error:
        return {
            "_status": "failed",
            "_error": str(error),
            "_latitude": latitude,
            "_longitude": longitude
        }


def numeric_values(value: Any) -> List[float]:
    if not isinstance(value, list):
        return []

    result = []

    for item in value:
        try:
            number = float(item)

            if math.isfinite(number):
                result.append(number)

        except Exception:
            continue

    return result


def weather_statistics(
    weather: Dict[str, Any]
) -> Dict[str, Any]:

    if weather.get("_status") != "success":
        return weather

    hourly = weather.get(
        "hourly",
        {}
    )

    daily = weather.get(
        "daily",
        {}
    )

    current = weather.get(
        "current",
        {}
    )

    def mean(values):
        return (
            round(sum(values) / len(values), 3)
            if values
            else None
        )

    def minimum(values):
        return (
            round(min(values), 3)
            if values
            else None
        )

    def maximum(values):
        return (
            round(max(values), 3)
            if values
            else None
        )

    temperature = numeric_values(
        hourly.get("temperature_2m")
    )

    humidity = numeric_values(
        hourly.get("relative_humidity_2m")
    )

    precipitation = numeric_values(
        hourly.get("precipitation")
    )

    rain = numeric_values(
        hourly.get("rain")
    )

    rain_probability = numeric_values(
        hourly.get("precipitation_probability")
    )

    wind = numeric_values(
        hourly.get("wind_speed_10m")
    )

    gusts = numeric_values(
        hourly.get("wind_gusts_10m")
    )

    et0 = numeric_values(
        hourly.get("et0_fao_evapotranspiration")
    )

    vpd = numeric_values(
        hourly.get("vapour_pressure_deficit")
    )

    return {
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
                sum(precipitation),
                3
            ) if precipitation else 0,
            "rain_total_mm": round(
                sum(rain),
                3
            ) if rain else 0,
            "maximum_rain_probability": maximum(
                rain_probability
            ),
            "wind_speed_mean": mean(wind),
            "maximum_wind_speed": maximum(wind),
            "maximum_wind_gust": maximum(gusts),
            "et0_total": round(
                sum(et0),
                3
            ) if et0 else 0,
            "vpd_mean": mean(vpd)
        }
    }


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class AgriculturalRetriever:

    def __init__(
        self,
        use_reranker: bool = True,
        dense_k: int = 100,
        bm25_k: int = 100,
        metadata_k: int = 100,
        final_context_chunks: int = 35,
        max_context_chars: int = 70000
    ):

        print("\n" + "=" * 75)
        print("INITIALIZING AGRICULTURAL HYBRID RETRIEVER")
        print("=" * 75)

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.metadata_k = metadata_k
        self.final_context_chunks = final_context_chunks
        self.max_context_chars = max_context_chars

        if not EMBEDDING_FILE.exists():
            raise FileNotFoundError(
                f"Missing embeddings: {EMBEDDING_FILE}"
            )

        print("\nLoading saved embeddings...")

        self.embeddings = np.load(
            EMBEDDING_FILE
        ).astype(np.float32)

        print(
            f"Embedding shape: {self.embeddings.shape}"
        )

        print("\nLoading embedding records...")

        self.records = load_embedding_records()

        print(
            f"Records loaded: {len(self.records)}"
        )

        if len(self.records) != len(self.embeddings):
            raise ValueError(
                "Embedding-record mismatch: "
                f"{len(self.embeddings)} embeddings, "
                f"{len(self.records)} records."
            )

        self.ids = [
            deterministic_id(
                record,
                index
            )
            for index, record in enumerate(
                self.records
            )
        ]

        self.id_to_index = {
            record_id: index
            for index, record_id in enumerate(
                self.ids
            )
        }

        print("\nPreparing BM25 index...")

        self.search_texts = [
            self.record_search_text(record)
            for record in self.records
        ]

        self.bm25 = BM25Okapi([
            tokenize(text)
            for text in self.search_texts
        ])

        print("BM25 index ready.")

        print("\nLoading embedding model...")

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        print("Embedding model ready.")

        self.reranker = None

        if use_reranker:
            try:
                print("\nLoading optional reranker...")

                from sentence_transformers import CrossEncoder

                self.reranker = CrossEncoder(
                    RERANKER_MODEL_NAME,
                    max_length=512
                )

                print("Reranker ready.")

            except Exception as error:
                print(
                    "Reranker unavailable. "
                    "Continuing with hybrid ranking."
                )
                print(str(error))

        print("\nInitializing ChromaDB...")

        VECTOR_STORE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.chroma_client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_DIR)
        )

        self.collection = (
            self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine"
                }
            )
        )

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
            "ChromaDB records:",
            self.collection.count()
        )

        print("\nRetriever initialized successfully.")

    def record_search_text(
        self,
        record: Dict[str, Any]
    ) -> str:

        content = record.get(
            "content",
            ""
        )

        metadata = record.get(
            "metadata",
            {}
        )

        metadata_parts = []

        for key in [
            "source_file",
            "page",
            "page_display",
            "chunk_id",
            "crop",
            "crops",
            "state",
            "district",
            "region",
            "topics",
            "topic",
            "source_authority",
            "document_type",
            "season"
        ]:

            if key in metadata:
                metadata_parts.append(
                    f"{key} {safe_text(metadata[key])}"
                )

        return clean_text(
            content
            + " "
            + " ".join(metadata_parts)
        )

    def build_metadata_terms(
        self,
        query: str,
        farmer_context: Dict[str, Any]
    ) -> Dict[str, List[str]]:

        flat = flatten_dictionary(
            farmer_context
        )

        combined = clean_text(
            query
            + " "
            + " ".join(flat.values())
        ).lower()

        crops = []
        states = []
        topics = []

        for canonical, aliases in CROP_ALIASES.items():
            if any(
                alias.lower() in combined
                for alias in aliases
            ):
                crops.append(canonical)

        for canonical, aliases in STATE_ALIASES.items():
            if any(
                alias.lower() in combined
                for alias in aliases
            ):
                states.append(canonical)

        for topic, terms in TOPIC_TERMS.items():
            if any(
                term.lower() in combined
                for term in terms
            ):
                topics.append(topic)

        soil_terms = [
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
        ]

        irrigation_terms = [
            term
            for term in [
                "borewell",
                "drip",
                "sprinkler",
                "irrigation",
                "rainfed",
                "water source",
                "drainage"
            ]
            if term in combined
        ]

        return {
            "crops": crops,
            "states": states,
            "topics": topics,
            "soil_terms": soil_terms,
            "irrigation_terms": irrigation_terms
        }

    def expand_query(
        self,
        query: str,
        farmer_context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, List[str]]]:

        terms = self.build_metadata_terms(
            query,
            farmer_context
        )

        flat = flatten_dictionary(
            farmer_context
        )

        context_text = " ".join(
            f"{key} {value}"
            for key, value in flat.items()
            if value
        )

        aliases = []

        for crop in terms["crops"]:
            aliases.extend(
                CROP_ALIASES.get(
                    crop,
                    []
                )
            )

        for state in terms["states"]:
            aliases.extend(
                STATE_ALIASES.get(
                    state,
                    []
                )
            )

        expanded = clean_text(
            query
            + " "
            + context_text
            + " "
            + " ".join(aliases)
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

        query_text = (
            "Represent this sentence for searching "
            "relevant passages: "
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

        ids = result.get(
            "ids",
            [[]]
        )[0]

        documents = result.get(
            "documents",
            [[]]
        )[0]

        metadatas = result.get(
            "metadatas",
            [[]]
        )[0]

        distances = result.get(
            "distances",
            [[]]
        )[0]

        for rank, record_id in enumerate(
            ids,
            start=1
        ):

            index = self.id_to_index.get(
                record_id
            )

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
                "rrf_score": 0.0
            }

        return candidates

    def bm25_search(
        self,
        expanded_query: str
    ) -> Dict[str, int]:

        scores = self.bm25.get_scores(
            tokenize(expanded_query)
        )

        ranked_indices = np.argsort(
            scores
        )[::-1][:self.bm25_k]

        result = {}

        for rank, index in enumerate(
            ranked_indices,
            start=1
        ):

            record_id = self.ids[
                int(index)
            ]

            result[record_id] = rank

        return result

    def metadata_search(
        self,
        terms: Dict[str, List[str]]
    ) -> Dict[str, int]:

        scored = []

        for index, record in enumerate(
            self.records
        ):

            metadata = record.get(
                "metadata",
                {}
            )

            metadata_text = " ".join(
                safe_text(value).lower()
                for value in metadata.values()
            )

            content_text = record.get(
                "content",
                ""
            ).lower()

            searchable = (
                metadata_text
                + " "
                + content_text
            )

            score = 0.0

            for crop in terms.get(
                "crops",
                []
            ):

                aliases = CROP_ALIASES.get(
                    crop,
                    [crop]
                )

                if any(
                    alias.lower() in searchable
                    for alias in aliases
                ):
                    score += 5.0

            for state in terms.get(
                "states",
                []
            ):

                aliases = STATE_ALIASES.get(
                    state,
                    [state]
                )

                if any(
                    alias.lower() in searchable
                    for alias in aliases
                ):
                    score += 3.0

            for topic in terms.get(
                "topics",
                []
            ):

                topic_terms = TOPIC_TERMS.get(
                    topic,
                    [topic]
                )

                if any(
                    item.lower() in searchable
                    for item in topic_terms
                ):
                    score += 1.5

            for term in terms.get(
                "soil_terms",
                []
            ):

                if term.lower() in searchable:
                    score += 1.5

            for term in terms.get(
                "irrigation_terms",
                []
            ):

                if term.lower() in searchable:
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

        for rank, item in enumerate(
            scored[:self.metadata_k],
            start=1
        ):

            index = item[0]

            result[
                self.ids[index]
            ] = rank

        return result

    def reciprocal_rank_fusion(
        self,
        dense_candidates: Dict[str, Dict[str, Any]],
        bm25_ranks: Dict[str, int],
        metadata_ranks: Dict[str, int]
    ) -> Dict[str, Dict[str, Any]]:

        fused = dict(
            dense_candidates
        )

        for record_id in list(
            bm25_ranks.keys()
        ) + list(
            metadata_ranks.keys()
        ):

            if record_id not in fused:

                index = self.id_to_index[
                    record_id
                ]

                fused[record_id] = {
                    "id": record_id,
                    "index": index,
                    "content": self.records[
                        index
                    ]["content"],
                    "metadata": self.records[
                        index
                    ]["metadata"],
                    "dense_rank": None,
                    "dense_distance": None,
                    "bm25_rank": None,
                    "metadata_rank": None,
                    "rrf_score": 0.0
                }

        for record_id, item in fused.items():

            dense_rank = item.get(
                "dense_rank"
            )

            bm25_rank = bm25_ranks.get(
                record_id
            )

            metadata_rank = metadata_ranks.get(
                record_id
            )

            score = 0.0

            if dense_rank is not None:
                score += 1.0 / (
                    60.0 + dense_rank
                )

            if bm25_rank is not None:
                score += 1.0 / (
                    60.0 + bm25_rank
                )

            if metadata_rank is not None:
                score += 1.0 / (
                    60.0 + metadata_rank
                )

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

        candidates = list(
            fused_candidates.values()
        )

        rerank_pool = candidates[:40]

        if not rerank_pool:
            return []

        if self.reranker is None:

            for item in rerank_pool:
                item["final_score"] = item[
                    "rrf_score"
                ]

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
            scores = self.reranker.predict(
                pairs,
                show_progress_bar=False
            )

            scores = [
                float(score)
                for score in scores
            ]

            minimum = min(scores)
            maximum = max(scores)
            score_range = maximum - minimum

            for item, score in zip(
                rerank_pool,
                scores
            ):

                normalized = (
                    (score - minimum) / score_range
                    if score_range > 0
                    else 1.0
                )

                item["reranker_score"] = score

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
                "Reranking failed. "
                "Using RRF ranking."
            )
            print(str(error))

            for item in rerank_pool:
                item["final_score"] = item[
                    "rrf_score"
                ]

            return sorted(
                rerank_pool,
                key=lambda item: item["final_score"],
                reverse=True
            )

    def select_diverse_context(
        self,
        ranked_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        selected = []

        source_counts = {}
        page_counts = {}

        for candidate in ranked_candidates:

            metadata = candidate.get(
                "metadata",
                {}
            )

            source = safe_text(
                metadata.get(
                    "source_file",
                    "unknown_source"
                )
            )

            page = safe_text(
                metadata.get(
                    "page_display",
                    metadata.get(
                        "page",
                        "unknown_page"
                    )
                )
            )

            source_key = source
            page_key = f"{source}|{page}"

            if source_counts.get(
                source_key,
                0
            ) >= 5:
                continue

            if page_counts.get(
                page_key,
                0
            ) >= 2:
                continue

            selected.append(candidate)

            source_counts[source_key] = (
                source_counts.get(
                    source_key,
                    0
                ) + 1
            )

            page_counts[page_key] = (
                page_counts.get(
                    page_key,
                    0
                ) + 1
            )

            current_chars = sum(
                len(item.get("content", ""))
                for item in selected
            )

            if len(selected) >= (
                self.final_context_chunks
            ):
                break

            if current_chars >= (
                self.max_context_chars
            ):
                break

        return selected

    def citation_for(
        self,
        item: Dict[str, Any]
    ) -> str:

        metadata = item.get(
            "metadata",
            {}
        )

        source_file = safe_text(
            metadata.get(
                "source_file",
                "unknown_source"
            )
        )

        page = safe_text(
            metadata.get(
                "page_display",
                metadata.get(
                    "page",
                    "unknown_page"
                )
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

        blocks = []

        for number, item in enumerate(
            selected,
            start=1
        ):

            metadata = item.get(
                "metadata",
                {}
            )

            citation = self.citation_for(
                item
            )

            block = (
                f"[EVIDENCE {number}]\n"
                f"Source: {citation}\n"
                f"Crop: {safe_text(metadata.get('crop'))}\n"
                f"State/Region: {safe_text(metadata.get('state'))}\n"
                f"Topics: {safe_text(metadata.get('topics'))}\n"
                f"Content:\n{item.get('content', '')}"
            )

            blocks.append(block)

        return "\n\n".join(blocks)

    def retrieve(
        self,
        query: str,
        farmer_context: Dict[str, Any],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ) -> Dict[str, Any]:

        print("\n" + "=" * 75)
        print("HYBRID AGRICULTURAL RETRIEVAL")
        print("=" * 75)

        expanded_query, terms = self.expand_query(
            query,
            farmer_context
        )

        print("\nRunning Open-Meteo...")

        weather_raw = {}

        if (
            latitude is not None
            and longitude is not None
        ):

            weather_raw = fetch_open_meteo(
                latitude=latitude,
                longitude=longitude,
                forecast_days=16
            )

            if weather_raw.get(
                "_status"
            ) == "success":

                print(
                    "Open-Meteo weather loaded."
                )

            else:

                print(
                    "Weather unavailable:",
                    weather_raw.get("_error")
                )

        print("Running dense vector search...")

        dense_candidates = self.dense_search(
            expanded_query
        )

        print(
            "Dense candidates:",
            len(dense_candidates)
        )

        print("Running BM25 search...")

        bm25_ranks = self.bm25_search(
            expanded_query
        )

        print(
            "BM25 candidates:",
            len(bm25_ranks)
        )

        print("Running metadata search...")

        metadata_ranks = self.metadata_search(
            terms
        )

        print(
            "Metadata candidates:",
            len(metadata_ranks)
        )

        print("Fusing retrieval results...")

        fused = self.reciprocal_rank_fusion(
            dense_candidates,
            bm25_ranks,
            metadata_ranks
        )

        print(
            "Fused candidates:",
            len(fused)
        )

        print("Applying limited reranking...")

        reranked = self.rerank_candidates(
            expanded_query,
            fused
        )

        print(
            "Reranked candidates:",
            len(reranked)
        )

        print("Selecting diverse context...")

        selected = self.select_diverse_context(
            reranked
        )

        context_text = self.build_context_text(
            selected
        )

        weather_summary = weather_statistics(
            weather_raw
        )

        citations = [
            self.citation_for(item)
            for item in selected
        ]

        unique_sources = sorted(
            set(
                safe_text(
                    item.get(
                        "metadata",
                        {}
                    ).get(
                        "source_file",
                        "unknown_source"
                    )
                )
                for item in selected
            )
        )

        print(
            "Final context chunks:",
            len(selected)
        )

        print(
            "Context characters:",
            len(context_text)
        )

        print(
            "Unique source PDFs:",
            len(unique_sources)
        )

        return {
            "original_query": query,
            "expanded_query": expanded_query,
            "detected_terms": terms,
            "weather_raw": weather_raw,
            "weather_summary": weather_summary,
            "selected_results": selected,
            "context_text": context_text,
            "citations": citations,
            "unique_sources": unique_sources
        }


# ============================================================
# ML MODEL WRAPPER
# ============================================================

class AgriculturalModels:

    def __init__(self):

        print("\n" + "=" * 75)
        print("LOADING TRAINED ML MODELS")
        print("=" * 75)

        self.production_model = None
        self.crop_model = None

        if PRODUCTION_MODEL_FILE.exists():

            self.production_model = joblib.load(
                PRODUCTION_MODEL_FILE
            )

            print(
                "Production model loaded."
            )

        else:

            print(
                "Production model not found:",
                PRODUCTION_MODEL_FILE
            )

        if CROP_MODEL_FILE.exists():

            self.crop_model = joblib.load(
                CROP_MODEL_FILE
            )

            print(
                "Crop recommendation model loaded."
            )

        else:

            print(
                "Crop model not found:",
                CROP_MODEL_FILE
            )

    def safe_prediction(
        self,
        model: Any,
        features: pd.DataFrame
    ) -> Any:

        if model is None:
            return None

        try:

            prediction = model.predict(
                features
            )

            if isinstance(
                prediction,
                np.ndarray
            ):
                prediction = prediction.tolist()

            return prediction

        except Exception as first_error:

            try:

                prediction = model.predict(
                    features.to_numpy()
                )

                if isinstance(
                    prediction,
                    np.ndarray
                ):
                    prediction = prediction.tolist()

                return prediction

            except Exception as second_error:

                return {
                    "error": (
                        f"Named-column prediction failed: "
                        f"{first_error}; "
                        f"array prediction failed: "
                        f"{second_error}"
                    )
                }

    def predict_production(
        self,
        farmer: Dict[str, Any]
    ) -> Dict[str, Any]:

        if self.production_model is None:
            return {
                "status": "unavailable",
                "reason": "Production model not found."
            }

        state = safe_text(
            farmer.get("state")
        )

        district = safe_text(
            farmer.get("district")
        )

        season = safe_text(
            farmer.get("season")
        )

        crop = (
            safe_text(
                farmer.get("crop")
            )
            or safe_text(
                farmer.get("crop_name")
            )
            or safe_text(
                farmer.get("planned_crop")
            )
        )

        area = finite_float(
            farmer.get(
                "plot_area_acres",
                farmer.get(
                    "area",
                    0
                )
            )
        )

        crop_year = finite_int(
            farmer.get(
                "crop_year",
                2026
            ),
            2026
        )

        features = pd.DataFrame([{
            "state_name": state,
            "district_name": district,
            "crop_year": crop_year,
            "season": season,
            "crop": crop,
            "area": area
        }])

        prediction = self.safe_prediction(
            self.production_model,
            features
        )

        return {
            "status": "success",
            "model": PRODUCTION_MODEL_FILE.name,
            "features": features.to_dict(
                orient="records"
            )[0],
            "prediction": prediction,
            "interpretation": (
                "Historical production/yield estimate. "
                "It is not a complete crop suitability model."
            )
        }

    def predict_crop_recommendation(
        self,
        farmer: Dict[str, Any]
    ) -> Dict[str, Any]:

        if self.crop_model is None:
            return {
                "status": "unavailable",
                "reason": "Crop recommendation model not found."
            }

        weather = farmer.get(
            "weather",
            {}
        )

        def get_number(
            *keys,
            default=0.0
        ):

            for key in keys:

                if key in farmer:
                    value = finite_float(
                        farmer.get(key),
                        default
                    )

                    if value != default:
                        return value

                if key in weather:
                    value = finite_float(
                        weather.get(key),
                        default
                    )

                    if value != default:
                        return value

            return default

        nitrogen = get_number(
            "nitrogen",
            "n",
            "soil_nitrogen"
        )

        phosphorus = get_number(
            "phosphorus",
            "p",
            "soil_phosphorus"
        )

        potassium = get_number(
            "potassium",
            "k",
            "soil_potassium"
        )

        temperature = get_number(
            "temperature",
            "temperature_2m",
            default=25.0
        )

        humidity = get_number(
            "humidity",
            "relative_humidity_2m",
            default=60.0
        )

        ph = get_number(
            "ph",
            "soil_ph",
            default=6.5
        )

        rainfall = get_number(
            "rainfall",
            "precipitation",
            "rainfall_mm"
        )

        features = pd.DataFrame([{
            "n": nitrogen,
            "p": phosphorus,
            "k": potassium,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
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
            "model": CROP_MODEL_FILE.name,
            "features": features.to_dict(
                orient="records"
            )[0],
            "prediction": prediction,
            "probabilities": probabilities,
            "important_limitation": (
                "This model only knows crop classes "
                "present in its training dataset. "
                "It does not directly predict every "
                "Indian crop."
            )
        }

    def run_all(
        self,
        farmer: Dict[str, Any]
    ) -> Dict[str, Any]:

        return {
            "production_model": self.predict_production(
                farmer
            ),
            "crop_recommendation_model": (
                self.predict_crop_recommendation(
                    farmer
                )
            )
        }


# ============================================================
# GROQ PROMPT
# ============================================================

def create_groq_prompt(
    farmer: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    ml_result: Dict[str, Any]
) -> str:

    farmer_json = json.dumps(
        json_safe(farmer),
        ensure_ascii=False,
        indent=2
    )

    weather_json = json.dumps(
        json_safe(
            retrieval_result.get(
                "weather_summary",
                {}
            )
        ),
        ensure_ascii=False,
        indent=2
    )

    ml_json = json.dumps(
        json_safe(ml_result),
        ensure_ascii=False,
        indent=2
    )

    evidence_context = retrieval_result.get(
        "context_text",
        ""
    )

    return f"""
You are an experienced Indian agricultural scientist
with approximately 30 years of practical experience in:

- Indian crop planning
- Soil science
- Agrometeorology
- Irrigation management
- Crop physiology
- Plant pathology
- Crop nutrition
- Region-specific agriculture
- Crop yield improvement
- Farmer-friendly advisory services

Your task is to prepare a practical, evidence-grounded
agricultural planning report for the farmer.

IMPORTANT RULES:

1. Use the farmer's exact details as the main input.
2. Use the selected latitude and longitude.
3. Use the Open-Meteo weather information.
4. Use the retrieved ICAR/KCC agricultural evidence.
5. Use the two ML model outputs carefully.
6. Do not blindly trust any single ML prediction.
7. Do not invent missing soil values.
8. Clearly distinguish measured farmer data,
   weather-model data, ML predictions, and evidence.
9. Do not make market-price recommendations.
10. Do not use Bhoonidhi or VEDAS.
11. Do not give unsupported pesticide or chemical
    recommendations.
12. If a pesticide or spray is mentioned in evidence,
    advise the farmer to verify the product, dose,
    label, local agricultural officer, or KVK.
13. Do not claim that a crop is suitable only because
    the ML model predicted it.
14. Sugarcane and other crops absent from the crop
    recommendation model may still be discussed only
    when supported by retrieved agricultural evidence
    and farmer conditions.
15. Mention uncertainty wherever data is missing.
16. Do not use years or dates as the main retrieval logic.
17. Explain why each selected crop is suitable or unsuitable.
18. Give immediate starting guidance that a farmer can
    understand and follow.
19. Use proper evidence citations exactly in this format:
    [Source filename, page number]
20. Never create fake citations.

FARMER DETAILS:
{farmer_json}

OPEN-METEO WEATHER SUMMARY:
{weather_json}

ML MODEL RESULTS:
{ml_json}

RETRIEVED AGRICULTURAL EVIDENCE:
{evidence_context}

Prepare the final report using this structure:

1. FARM AND LOCATION SUMMARY

2. CURRENT WEATHER AND AGRICULTURAL MEANING

3. MAIN RECOMMENDED CROPS
   - Crop name
   - Suitability level
   - Why it is selected
   - Soil reasoning
   - Weather and climate reasoning
   - Water and irrigation reasoning
   - Expected management difficulty
   - Evidence citations

4. CROPS TO AVOID OR DELAY
   - Crop name
   - Reason
   - Missing condition or risk
   - Evidence citations

5. IMMEDIATE FARM STARTING PLAN
   - Land preparation
   - Seed or planting material
   - Sowing or planting timing
   - Spacing
   - Irrigation
   - Soil and nutrient checks
   - Weed management
   - Disease and pest monitoring

6. CROP-SPECIFIC GUIDANCE
   Explain practical early-stage actions for every
   recommended crop.

7. ML MODEL INTERPRETATION
   Explain what the models predicted and their limits.

8. RISK AND UNCERTAINTY

9. FINAL FARMER DECISION
   Give a clear ranked recommendation and explain
   the reasoning in simple language.

Use clear headings, short paragraphs, tables where useful,
and farmer-friendly language. Do not exaggerate certainty.
"""


def call_groq(
    prompt: str
) -> str:

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing from the .env file."
        )

    try:

        from groq import Groq

        client = Groq(
            api_key=GROQ_API_KEY
        )

        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful, evidence-grounded "
                        "Indian agricultural scientist."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=7000
        )

        return response.choices[0].message.content

    except Exception as error:
        raise RuntimeError(
            f"Groq request failed: {error}"
        )


# ============================================================
# REPORT SAVING
# ============================================================

def save_report(
    report: str,
    farmer: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    ml_result: Dict[str, Any]
) -> Tuple[Path, Path]:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = (
        pd.Timestamp.now()
        .strftime("%Y%m%d_%H%M%S")
    )

    txt_file = REPORT_DIR / (
        f"agri_report_{timestamp}.txt"
    )

    json_file = REPORT_DIR / (
        f"agri_report_{timestamp}.json"
    )

    txt_file.write_text(
        report,
        encoding="utf-8"
    )

    json_payload = {
        "farmer": farmer,
        "weather": retrieval_result.get(
            "weather_summary",
            {}
        ),
        "citations": retrieval_result.get(
            "citations",
            []
        ),
        "ml_results": ml_result,
        "report": report
    }

    json_file.write_text(
        json.dumps(
            json_safe(json_payload),
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return txt_file, json_file


# ============================================================
# MAP INTERFACE
# ============================================================

def open_map_window():

    global map_window
    global map_widget
    global selected_latitude
    global selected_longitude

    if map_window is not None:

        try:

            if map_window.winfo_exists():
                map_window.lift()
                map_window.focus_force()
                return

        except Exception:
            pass

    map_window = tk.Toplevel(
        root
    )

    map_window.title(
        "Select Exact Farm Location"
    )

    map_window.geometry(
        "1050x720"
    )

    map_window.minsize(
        800,
        600
    )

    ttk.Label(
        map_window,
        text=(
            "Zoom into your farm and click the exact "
            "farm location on the map."
        ),
        font=("Arial", 11)
    ).pack(
        pady=8
    )

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

    map_widget.set_position(
        15.3173,
        75.7139
    )

    map_widget.set_zoom(
        6
    )

    def select_location(coords):

        global selected_latitude
        global selected_longitude

        selected_latitude = float(
            coords[0]
        )

        selected_longitude = float(
            coords[1]
        )

        latitude_text.set(
            f"{selected_latitude:.7f}"
        )

        longitude_text.set(
            f"{selected_longitude:.7f}"
        )

        location_status_text.set(
            "Farm location selected successfully."
        )

        map_widget.delete_all_marker()

        map_widget.set_marker(
            selected_latitude,
            selected_longitude,
            text="Selected Farm"
        )

    map_widget.add_left_click_map_command(
        select_location
    )

    button_frame = ttk.Frame(
        map_window
    )

    button_frame.pack(
        pady=8
    )

    def close_map():

        global map_window

        try:
            map_window.destroy()
        except Exception:
            pass

        map_window = None

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
# FARMER INPUT HELPERS
# ============================================================

def ask_required(
    title: str,
    prompt: str
) -> str:

    from tkinter import simpledialog

    while True:

        value = simpledialog.askstring(
            title,
            prompt,
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


def ask_optional(
    title: str,
    prompt: str
) -> str:

    from tkinter import simpledialog

    value = simpledialog.askstring(
        title,
        prompt,
        parent=root
    )

    if value is None:
        return ""

    return value.strip()


def collect_farmer_details() -> Optional[Dict[str, Any]]:

    global selected_latitude
    global selected_longitude

    if (
        selected_latitude is None
        or selected_longitude is None
    ):

        messagebox.showwarning(
            "Location Required",
            (
                "Please open the map and select "
                "the farm location first."
            ),
            parent=root
        )

        return None

    try:

        farmer = {
            "name": ask_required(
                "Farmer Details",
                "Enter farmer name:"
            ),

            "state": ask_required(
                "Farmer Details",
                "Enter state:"
            ),

            "district": ask_required(
                "Farmer Details",
                "Enter district:"
            ),

            "taluk": ask_optional(
                "Farmer Details",
                "Enter taluk or tehsil, optional:"
            ),

            "village": ask_optional(
                "Farmer Details",
                "Enter village, optional:"
            ),

            "farm_name": ask_optional(
                "Farmer Details",
                "Enter farm name, optional:"
            ),

            "land_area_acres": finite_float(
                ask_required(
                    "Farmer Details",
                    "Enter total farm area in acres:"
                )
            ),

            "plot_area_acres": finite_float(
                ask_required(
                    "Farmer Details",
                    "Enter plot area in acres:"
                )
            ),

            "season": ask_optional(
                "Farmer Details",
                "Enter season, for example Kharif, Rabi, Summer:"
            ),

            "crop": ask_optional(
                "Farmer Details",
                "Enter currently planned crop, optional:"
            ),

            "soil_type": ask_optional(
                "Farmer Details",
                "Enter soil type, for example black, red, loamy:"
            ),

            "soil_colour": ask_optional(
                "Farmer Details",
                "Enter soil colour:"
            ),

            "ph": finite_float(
                ask_optional(
                    "Farmer Details",
                    "Enter soil pH if known, otherwise press Enter:"
                ),
                0.0
            ),

            "nitrogen": finite_float(
                ask_optional(
                    "Farmer Details",
                    "Enter soil nitrogen value if known:"
                ),
                0.0
            ),

            "phosphorus": finite_float(
                ask_optional(
                    "Farmer Details",
                    "Enter soil phosphorus value if known:"
                ),
                0.0
            ),

            "potassium": finite_float(
                ask_optional(
                    "Farmer Details",
                    "Enter soil potassium value if known:"
                ),
                0.0
            ),

            "water_availability": ask_optional(
                "Farmer Details",
                "Water availability: low, medium, or high:"
            ),

            "water_source": ask_optional(
                "Farmer Details",
                "Water source: borewell, canal, rainfed, tank, etc.:"
            ),

            "irrigation_method": ask_optional(
                "Farmer Details",
                "Irrigation method: drip, sprinkler, flood, rainfed:"
            ),

            "previous_crops": ask_optional(
                "Farmer Details",
                "Crops grown previously:"
            ),

            "previous_problems": ask_optional(
                "Farmer Details",
                "Previous crop problems:"
            ),

            "growth_stage": ask_optional(
                "Farmer Details",
                "Current growth stage, if applicable:"
            ),

            "farmer_question": ask_optional(
                "Farmer Details",
                "Enter your question, or press Enter for default:"
            ),

            "latitude": selected_latitude,
            "longitude": selected_longitude
        }

        if not farmer["farmer_question"]:

            farmer["farmer_question"] = (
                "What should I plan now considering "
                "soil, weather, irrigation, crop suitability, "
                "disease risk, crop management, and expected yield?"
            )

        return farmer

    except KeyboardInterrupt:
        return None


# ============================================================
# COMPLETE ANALYSIS PIPELINE
# ============================================================

def run_analysis():

    global analysis_button

    try:

        if analysis_button is not None:
            analysis_button.config(
                state="disabled"
            )

        status_text.set(
            "Collecting farmer details..."
        )

        farmer = collect_farmer_details()

        if farmer is None:
            return

        latitude = farmer["latitude"]
        longitude = farmer["longitude"]

        query = farmer[
            "farmer_question"
        ]

        result_text.delete(
            "1.0",
            tk.END
        )

        result_text.insert(
            tk.END,
            "Initializing agricultural AI system...\n"
        )

        status_text.set(
            "Loading retriever and knowledge base..."
        )

        retriever = AgriculturalRetriever(
            use_reranker=True,
            dense_k=100,
            bm25_k=100,
            metadata_k=100,
            final_context_chunks=35,
            max_context_chars=70000
        )

        status_text.set(
            "Retrieving agricultural evidence and weather..."
        )

        retrieval_result = retriever.retrieve(
            query=query,
            farmer_context=farmer,
            latitude=latitude,
            longitude=longitude
        )

        weather_summary = retrieval_result.get(
            "weather_summary",
            {}
        )

        current_weather = weather_summary.get(
            "current",
            {}
        )

        farmer["weather"] = {
            "temperature": current_weather.get(
                "temperature_2m",
                25.0
            ),
            "humidity": current_weather.get(
                "relative_humidity_2m",
                60.0
            ),
            "precipitation": current_weather.get(
                "precipitation",
                0.0
            ),
            "wind_speed": current_weather.get(
                "wind_speed_10m",
                0.0
            )
        }

        status_text.set(
            "Running trained ML models..."
        )

        models = AgriculturalModels()

        ml_result = models.run_all(
            farmer
        )

        status_text.set(
            "Generating evidence-based Groq report..."
        )

        prompt = create_groq_prompt(
            farmer=farmer,
            retrieval_result=retrieval_result,
            ml_result=ml_result
        )

        final_report = call_groq(
            prompt
        )

        txt_file, json_file = save_report(
            report=final_report,
            farmer=farmer,
            retrieval_result=retrieval_result,
            ml_result=ml_result
        )

        result_text.delete(
            "1.0",
            tk.END
        )

        result_text.insert(
            tk.END,
            final_report
        )

        result_text.insert(
            tk.END,
            "\n\n"
            + "=" * 70
            + "\n"
            + "REPORT FILES\n"
            + "=" * 70
            + "\n"
            + str(txt_file)
            + "\n"
            + str(json_file)
        )

        status_text.set(
            "Analysis completed successfully."
        )

        messagebox.showinfo(
            "Analysis Completed",
            (
                "Agricultural report generated successfully.\n\n"
                f"Text report:\n{txt_file}\n\n"
                f"JSON report:\n{json_file}"
            ),
            parent=root
        )

    except Exception as error:

        error_details = (
            f"{error}\n\n"
            f"{traceback.format_exc()}"
        )

        print("\n" + "=" * 75)
        print("SYSTEM ERROR")
        print("=" * 75)
        print(error_details)

        status_text.set(
            "Analysis failed. Check the terminal."
        )

        result_text.delete(
            "1.0",
            tk.END
        )

        result_text.insert(
            tk.END,
            "SYSTEM ERROR\n\n"
            + error_details
        )

        messagebox.showerror(
            "System Error",
            str(error),
            parent=root
        )

    finally:

        if analysis_button is not None:
            analysis_button.config(
                state="normal"
            )


# ============================================================
# MAIN GUI
# ============================================================

def build_gui():

    global root
    global latitude_text
    global longitude_text
    global location_status_text
    global result_text
    global status_text
    global analysis_button

    root = tk.Tk()

    root.title(
        "Smart Agricultural AI Agent"
    )

    root.geometry(
        "1100x780"
    )

    root.minsize(
        900,
        650
    )

    latitude_text = tk.StringVar(
        value="Not selected"
    )

    longitude_text = tk.StringVar(
        value="Not selected"
    )

    location_status_text = tk.StringVar(
        value=(
            "Open the map and select the exact "
            "farm location."
        )
    )

    status_text = tk.StringVar(
        value="Ready."
    )

    title_label = ttk.Label(
        root,
        text="Smart Agricultural AI Agent",
        font=("Arial", 18, "bold")
    )

    title_label.pack(
        pady=12
    )

    subtitle_label = ttk.Label(
        root,
        text=(
            "RAG + BM25 + Vector Search + ML + "
            "Open-Meteo + Groq"
        ),
        font=("Arial", 10)
    )

    subtitle_label.pack(
        pady=3
    )

    location_frame = ttk.LabelFrame(
        root,
        text="Farm Location"
    )

    location_frame.pack(
        fill="x",
        padx=20,
        pady=10
    )

    ttk.Label(
        location_frame,
        text="Latitude:"
    ).grid(
        row=0,
        column=0,
        padx=10,
        pady=6,
        sticky="w"
    )

    ttk.Label(
        location_frame,
        textvariable=latitude_text
    ).grid(
        row=0,
        column=1,
        padx=10,
        pady=6,
        sticky="w"
    )

    ttk.Label(
        location_frame,
        text="Longitude:"
    ).grid(
        row=1,
        column=0,
        padx=10,
        pady=6,
        sticky="w"
    )

    ttk.Label(
        location_frame,
        textvariable=longitude_text
    ).grid(
        row=1,
        column=1,
        padx=10,
        pady=6,
        sticky="w"
    )

    ttk.Label(
        location_frame,
        textvariable=location_status_text
    ).grid(
        row=2,
        column=0,
        columnspan=2,
        padx=10,
        pady=6,
        sticky="w"
    )

    ttk.Button(
        location_frame,
        text="Open Farm Map",
        command=open_map_window
    ).grid(
        row=0,
        column=2,
        rowspan=2,
        padx=20,
        pady=8
    )

    control_frame = ttk.Frame(
        root
    )

    control_frame.pack(
        pady=10
    )

    analysis_button = ttk.Button(
        control_frame,
        text="Start Agricultural Analysis",
        command=lambda: threading.Thread(
            target=run_analysis,
            daemon=True
        ).start()
    )

    analysis_button.grid(
        row=0,
        column=0,
        padx=8
    )

    ttk.Button(
        control_frame,
        text="Clear Report",
        command=lambda: result_text.delete(
            "1.0",
            tk.END
        )
    ).grid(
        row=0,
        column=1,
        padx=8
    )

    ttk.Button(
        control_frame,
        text="Exit",
        command=root.destroy
    ).grid(
        row=0,
        column=2,
        padx=8
    )

    status_frame = ttk.LabelFrame(
        root,
        text="System Status"
    )

    status_frame.pack(
        fill="x",
        padx=20,
        pady=5
    )

    ttk.Label(
        status_frame,
        textvariable=status_text
    ).pack(
        anchor="w",
        padx=10,
        pady=6
    )

    result_frame = ttk.LabelFrame(
        root,
        text="Final Agricultural Report"
    )

    result_frame.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=10
    )

    result_text = tk.Text(
        result_frame,
        wrap="word",
        font=("Arial", 10)
    )

    result_text.pack(
        side="left",
        fill="both",
        expand=True,
        padx=5,
        pady=5
    )

    scrollbar = ttk.Scrollbar(
        result_frame,
        orient="vertical",
        command=result_text.yview
    )

    scrollbar.pack(
        side="right",
        fill="y"
    )

    result_text.configure(
        yscrollcommand=scrollbar.set
    )

    root.mainloop()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("=" * 75)
    print("SMART AGRICULTURAL AI AGENT")
    print("=" * 75)
    print("Project root:", PROJECT_ROOT)
    print("Map-based coordinates enabled.")
    print("Bhoonidhi and VEDAS disabled.")
    print("=" * 75)

    build_gui()