"""
agri_core.py
============
Pure-Python core functions extracted from final_agri_cmd.py.

This module is importable without any Tkinter / GUI dependency.
It exposes:
  - All helper functions (safe_text, clean_text, tokenize, …)
  - fetch_open_meteo, weather_statistics
  - AgriculturalRetriever
  - AgriculturalModels
  - create_groq_prompt
  - call_groq

The LangGraph graph (agri_graph.py) and Streamlit UI (streamlit_app.py)
import from here rather than from the monolithic final_agri_cmd.py.
"""

from __future__ import annotations

import os
import re
import json
import math
import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import numpy as np
import requests
import chromadb
import pandas as pd

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ── Project paths ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
load_dotenv(PROJECT_ROOT / ".env")

ML_ROOT   = PROJECT_ROOT / "ml_training"
MODEL_DIR = ML_ROOT / "models"

DATA_ROOT     = PROJECT_ROOT / "Data"
EMBEDDING_DIR = DATA_ROOT / "embeddings"
VECTOR_STORE_DIR = DATA_ROOT / "vector_store"

EMBEDDING_FILE              = EMBEDDING_DIR / "agri_embeddings.npy"
RECORDS_FILE                = EMBEDDING_DIR / "agri_embedding_records.jsonl"
PRODUCTION_MODEL_FILE       = MODEL_DIR / "agriculture_production_xgb.pkl"
CROP_RECOMMENDATION_MODEL_FILE = MODEL_DIR / "crop_recommendation_xgb.pkl"

CHROMA_COLLECTION_NAME = "agri_documents"
OPEN_METEO_URL         = "https://api.open-meteo.com/v1/forecast"
EMBEDDING_MODEL_NAME   = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_NAME    = "BAAI/bge-reranker-base"


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {safe_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(safe_text(i) for i in value)
    return str(value).strip()


def clean_text(text: str) -> str:
    text = safe_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    text = clean_text(text).lower()
    return re.findall(r"[a-zA-Z0-9]+", text)


def scalar_metadata(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(clean_text(i) for i in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return clean_text(value)


def deterministic_id(record: Dict[str, Any], index: int) -> str:
    metadata  = record.get("metadata", {})
    content   = record.get("content", "")
    source_file = metadata.get("source_file", "")
    page      = metadata.get("page", "")
    chunk_id  = metadata.get("chunk_id", "")
    raw       = f"{chunk_id}|{source_file}|{page}|{content}|{index}"
    digest    = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"agri_{digest}"


def flatten_dictionary(data: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
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
# CROP / STATE / TOPIC ALIASES  (unchanged from original)
# ============================================================

CROP_ALIASES = {
    "sugarcane":  ["sugarcane", "sugar cane", "cane", "ganna", "saccharum"],
    "rice":       ["rice", "paddy", "dhan"],
    "maize":      ["maize", "corn", "makka"],
    "pigeonpea":  ["pigeonpea", "pigeon pea", "tur", "arhar", "red gram"],
    "chickpea":   ["chickpea", "gram", "bengal gram", "chana"],
    "blackgram":  ["blackgram", "black gram", "urad"],
    "greengram":  ["greengram", "green gram", "mungbean", "moong"],
    "groundnut":  ["groundnut", "peanut"],
    "cotton":     ["cotton", "kapas"],
    "soybean":    ["soybean", "soyabean"],
    "wheat":      ["wheat", "gehun"],
}

STATE_ALIASES = {
    "maharashtra":    ["maharashtra", "mh", "maharashtra state"],
    "karnataka":      ["karnataka", "ka", "karnataka state"],
    "madhya pradesh": ["madhya pradesh", "mp"],
    "gujarat":        ["gujarat", "gj"],
    "telangana":      ["telangana", "ts"],
    "andhra pradesh": ["andhra pradesh", "ap"],
    "uttar pradesh":  ["uttar pradesh", "up"],
    "punjab":         ["punjab", "pb"],
    "haryana":        ["haryana", "hr"],
}

TOPIC_TERMS = {
    "weather":          ["weather", "rainfall", "rain", "temperature", "humidity",
                         "wind", "cloud", "forecast", "drought", "dry spell", "heavy rain"],
    "soil":             ["soil", "black soil", "red soil", "loam", "clay", "sandy",
                         "soil moisture", "soil fertility", "soil test", "ph"],
    "irrigation":       ["irrigation", "water", "borewell", "drip", "sprinkler",
                         "moisture", "drainage"],
    "crop_management":  ["planting", "sowing", "seed", "setts", "spacing",
                         "land preparation", "intercropping", "earthing", "mulching", "weeding"],
    "fertilizer":       ["fertilizer", "fertiliser", "nitrogen", "phosphorus", "potassium",
                         "urea", "fym", "manure", "nutrient"],
    "disease_pest":     ["disease", "pest", "borer", "fungus", "fungal", "virus",
                         "insect", "infection", "spray", "pathogen"],
    "yield":            ["yield", "production", "productivity", "tonne", "harvest", "output"],
    "crop_suitability": ["suitability", "recommended crop", "crop selection",
                         "climate suitability", "adaptation", "alternative crop"],
}


# ============================================================
# RECORD NORMALIZATION
# ============================================================

def normalize_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    content = (
        raw.get("page_content") or raw.get("content") or raw.get("text") or ""
    )
    content = clean_text(content)
    for key, value in raw.items():
        if key not in {"page_content", "content", "text", "metadata", "embedding"} \
                and key not in metadata:
            metadata[key] = value
    metadata = {str(k): scalar_metadata(v) for k, v in metadata.items()}
    page = metadata.get("page", "")
    try:
        page_number = int(page) + 1
    except Exception:
        page_number = page
    metadata["page_display"] = scalar_metadata(page_number)
    return {"content": content, "metadata": metadata}


def load_embedding_records() -> List[Dict[str, Any]]:
    if not RECORDS_FILE.exists():
        raise FileNotFoundError(f"Embedding records not found: {RECORDS_FILE}")
    records = []
    with RECORDS_FILE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                record = normalize_record(raw)
                if record["content"]:
                    records.append(record)
            except Exception as err:
                print(f"Warning: JSONL line {line_no}: {err}")
    return records


# ============================================================
# OPEN-METEO WEATHER
# ============================================================

def fetch_open_meteo(latitude: float, longitude: float, forecast_days: int = 16) -> Dict[str, Any]:
    hourly_fields = ",".join([
        "temperature_2m", "relative_humidity_2m", "precipitation_probability",
        "precipitation", "rain", "showers", "weather_code", "cloud_cover",
        "wind_speed_10m", "wind_gusts_10m", "et0_fao_evapotranspiration",
        "vapour_pressure_deficit", "soil_temperature_0cm", "soil_temperature_6cm",
        "soil_temperature_18cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm",
        "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm",
        "soil_moisture_27_to_81cm",
    ])
    daily_fields = ",".join([
        "weather_code", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "precipitation_probability_max",
        "wind_speed_10m_max", "et0_fao_evapotranspiration",
    ])
    params = {
        "latitude":  latitude,
        "longitude": longitude,
        "current":   ",".join([
            "temperature_2m", "relative_humidity_2m", "precipitation",
            "weather_code", "wind_speed_10m", "wind_gusts_10m", "cloud_cover",
        ]),
        "hourly":       hourly_fields,
        "daily":        daily_fields,
        "forecast_days": forecast_days,
        "timezone":     "auto",
    }
    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        response.raise_for_status()
        weather = response.json()
        weather["_status"]    = "success"
        weather["_latitude"]  = latitude
        weather["_longitude"] = longitude
        return weather
    except Exception as error:
        return {"_status": "failed", "_error": str(error),
                "_latitude": latitude, "_longitude": longitude}


def numeric_values(values: Any) -> List[float]:
    if not isinstance(values, list):
        return []
    result = []
    for v in values:
        try:
            n = float(v)
            if math.isfinite(n):
                result.append(n)
        except Exception:
            continue
    return result


def weather_statistics(weather: Dict[str, Any]) -> Dict[str, Any]:
    if weather.get("_status") != "success":
        return weather
    hourly  = weather.get("hourly", {})
    daily   = weather.get("daily", {})
    current = weather.get("current", {})
    temperature   = numeric_values(hourly.get("temperature_2m"))
    humidity      = numeric_values(hourly.get("relative_humidity_2m"))
    precipitation = numeric_values(hourly.get("precipitation"))
    rain          = numeric_values(hourly.get("rain"))
    rain_prob     = numeric_values(hourly.get("precipitation_probability"))
    wind          = numeric_values(hourly.get("wind_speed_10m"))
    gusts         = numeric_values(hourly.get("wind_gusts_10m"))
    et0           = numeric_values(hourly.get("et0_fao_evapotranspiration"))
    vpd           = numeric_values(hourly.get("vapour_pressure_deficit"))

    def mean(v): return round(sum(v) / len(v), 3) if v else None
    def maximum(v): return round(max(v), 3) if v else None
    def minimum(v): return round(min(v), 3) if v else None

    return {
        "status":  "success",
        "timezone": weather.get("timezone"),
        "latitude": weather.get("latitude"),
        "longitude": weather.get("longitude"),
        "current":  current,
        "daily_forecast": daily,
        "hourly_statistics": {
            "temperature_min":         minimum(temperature),
            "temperature_max":         maximum(temperature),
            "temperature_mean":        mean(temperature),
            "humidity_mean":           mean(humidity),
            "precipitation_total_mm":  round(sum(precipitation), 3) if precipitation else 0,
            "rain_total_mm":           round(sum(rain), 3) if rain else 0,
            "maximum_rain_probability": maximum(rain_prob),
            "wind_speed_mean":         mean(wind),
            "maximum_wind_speed":      maximum(wind),
            "maximum_wind_gust":       maximum(gusts),
            "et0_total":               round(sum(et0), 3) if et0 else 0,
            "vpd_mean":                mean(vpd),
        },
    }


# ============================================================
# HYBRID AGRICULTURAL RETRIEVER
# ============================================================

class AgriculturalRetriever:
    def __init__(
        self,
        use_reranker: bool = False,
        dense_k: int = 100,
        bm25_k: int = 100,
        metadata_k: int = 100,
        final_context_chunks: int = 35,
        max_context_chars: int = 70_000,
    ):
        print("\n" + "=" * 70)
        print("INITIALIZING AGRICULTURAL RETRIEVER")
        print("=" * 70)

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.metadata_k = metadata_k
        self.final_context_chunks = final_context_chunks
        self.max_context_chars = max_context_chars

        print("Loading embeddings…")
        self.embeddings = np.load(EMBEDDING_FILE).astype(np.float32)
        print(f"Embeddings shape: {self.embeddings.shape}")

        print("Loading embedding records…")
        self.records = load_embedding_records()
        print(f"Records: {len(self.records)}")

        if len(self.records) != len(self.embeddings):
            raise ValueError(
                f"Mismatch: {len(self.embeddings)} embeddings vs {len(self.records)} records"
            )

        self.ids = [deterministic_id(r, i) for i, r in enumerate(self.records)]
        self.id_to_index = {rid: i for i, rid in enumerate(self.ids)}

        print("Building BM25 index…")
        self.search_texts = [self._record_search_text(r) for r in self.records]
        self.bm25 = BM25Okapi([tokenize(t) for t in self.search_texts])
        print("BM25 ready.")

        print("Loading embedding model…")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model ready.")

        self.reranker = None
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)
                print("Reranker loaded.")
            except Exception as e:
                print(f"Reranker unavailable: {e}")

        # ── ChromaDB Init (robust, auto-recovers from corrupt index) ──
        print("Initializing ChromaDB...")
        self._init_chromadb()
        print("Retriever ready.\n")

    def _init_chromadb(self, retry: bool = True):
        """
        Initialise ChromaDB PersistentClient.

        If the on-disk HNSW index is corrupt (e.g. version mismatch after
        upgrading chromadb), the entire vector_store directory is wiped and
        rebuilt from the in-memory embeddings automatically.
        """
        import shutil
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(VECTOR_STORE_DIR)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            # Upsert in batches of 500 to avoid memory / RPC limits
            batch_size = 500
            total = len(self.ids)
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                self.collection.upsert(
                    ids=self.ids[start:end],
                    embeddings=self.embeddings[start:end].tolist(),
                    documents=[r["content"] for r in self.records[start:end]],
                    metadatas=[r["metadata"] for r in self.records[start:end]],
                )
            print(f"ChromaDB records: {self.collection.count()}")

        except Exception as exc:
            print(f"ChromaDB error: {exc}")
            if retry:
                print("Wiping corrupt vector store and rebuilding...")
                try:
                    # Close client before deleting
                    try:
                        del self.chroma_client
                    except Exception:
                        pass
                    shutil.rmtree(str(VECTOR_STORE_DIR), ignore_errors=True)
                    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
                    self._init_chromadb(retry=False)   # one retry
                except Exception as rebuild_exc:
                    print(f"ChromaDB rebuild also failed: {rebuild_exc}")
                    print("Falling back to BM25-only retrieval.")
                    self.chroma_client = None
                    self.collection = None
            else:
                print("Falling back to BM25-only retrieval.")
                self.chroma_client = None
                self.collection = None



    def _record_search_text(self, record: Dict[str, Any]) -> str:
        content  = record.get("content", "")
        metadata = record.get("metadata", {})
        parts = []
        for key in ["source_file", "page", "page_display", "crop", "crops",
                    "state", "district", "region", "topics", "topic",
                    "source_authority", "document_type", "season"]:
            if key in metadata:
                parts.append(f"{key} {safe_text(metadata[key])}")
        return clean_text(content + " " + " ".join(parts))

    def _build_metadata_terms(self, query: str, farmer_context: Dict[str, Any]) -> Dict[str, List[str]]:
        flat = flatten_dictionary(farmer_context)
        combined = clean_text(query + " " + " ".join(flat.values())).lower()
        crops  = [c for c, aliases in CROP_ALIASES.items()
                  if any(a.lower() in combined for a in aliases)]
        states = [s for s, aliases in STATE_ALIASES.items()
                  if any(a.lower() in combined for a in aliases)]
        topics = [t for t, terms in TOPIC_TERMS.items()
                  if any(term.lower() in combined for term in terms)]
        soil_terms = [t for t in [
            "black soil", "red soil", "loam", "clay", "sandy",
            "soil moisture", "soil fertility", "drainage", "soil test", "ph",
        ] if t in combined]
        irr_terms = [t for t in [
            "borewell", "drip", "sprinkler", "irrigation", "rainfed", "water source",
        ] if t in combined]
        return {"crops": crops, "states": states, "topics": topics,
                "soil_terms": soil_terms, "irrigation_terms": irr_terms}

    def _expand_query(self, query: str, farmer_context: Dict[str, Any]) -> Tuple[str, Dict]:
        terms = self._build_metadata_terms(query, farmer_context)
        flat  = flatten_dictionary(farmer_context)
        ctx   = " ".join(f"{k} {v}" for k, v in flat.items() if v)
        aliases = []
        for crop in terms["crops"]:
            aliases.extend(CROP_ALIASES.get(crop, []))
        for state in terms["states"]:
            aliases.extend(STATE_ALIASES.get(state, []))
        expanded = clean_text(
            query + " " + ctx + " " + " ".join(aliases)
            + " " + " ".join(terms["topics"])
            + " " + " ".join(terms["soil_terms"])
            + " " + " ".join(terms["irrigation_terms"])
        )
        return expanded, terms

    def _dense_search(self, expanded_query: str) -> Dict[str, Dict[str, Any]]:
        # If ChromaDB failed to init (BM25-only mode), return empty
        if self.collection is None:
            return {}
        q  = ("Represent this sentence for searching relevant passages: " + expanded_query)
        qe = self.embedding_model.encode(q, normalize_embeddings=True,
                                         convert_to_numpy=True).astype(np.float32)
        try:
            result = self.collection.query(
                query_embeddings=[qe.tolist()],
                n_results=min(self.dense_k, max(1, self.collection.count())),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            print(f"ChromaDB query failed: {exc}. Falling back to BM25 only.")
            return {}
        candidates = {}
        ids_  = result.get("ids", [[]])[0]
        docs  = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for rank, rid in enumerate(ids_, start=1):
            idx = self.id_to_index.get(rid)
            if idx is None:
                continue
            dist = float(dists[rank - 1]) if rank - 1 < len(dists) else 1.0
            candidates[rid] = {
                "id": rid, "index": idx,
                "content":  docs[rank - 1],
                "metadata": metas[rank - 1],
                "dense_rank": rank, "dense_distance": dist,
                "bm25_rank": None, "metadata_rank": None, "rrf_score": 0.0,
            }
        return candidates


    def _bm25_search(self, expanded_query: str) -> Dict[str, int]:
        scores = self.bm25.get_scores(tokenize(expanded_query))
        ranked = np.argsort(scores)[::-1][:self.bm25_k]
        return {self.ids[int(i)]: rank for rank, i in enumerate(ranked, start=1)}

    def _metadata_search(self, terms: Dict[str, List[str]]) -> Dict[str, int]:
        scored = []
        for idx, record in enumerate(self.records):
            meta_text = " ".join(safe_text(v).lower() for v in record.get("metadata", {}).values())
            content   = record.get("content", "").lower()
            searchable = meta_text + " " + content
            score = 0.0
            for crop in terms.get("crops", []):
                if any(a.lower() in searchable for a in CROP_ALIASES.get(crop, [crop])):
                    score += 5.0
            for state in terms.get("states", []):
                if any(a.lower() in searchable for a in STATE_ALIASES.get(state, [state])):
                    score += 3.0
            for topic in terms.get("topics", []):
                if any(t.lower() in searchable for t in TOPIC_TERMS.get(topic, [topic])):
                    score += 1.5
            for st in terms.get("soil_terms", []):
                if st.lower() in searchable:
                    score += 1.5
            for it in terms.get("irrigation_terms", []):
                if it.lower() in searchable:
                    score += 1.0
            if score > 0:
                scored.append((idx, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return {self.ids[i]: rank for rank, (i, _) in enumerate(scored[:self.metadata_k], start=1)}

    def _rrf(self, dense_candidates, bm25_ranks, metadata_ranks):
        fused = dict(dense_candidates)
        for rid in list(bm25_ranks) + list(metadata_ranks):
            if rid not in fused:
                idx = self.id_to_index[rid]
                fused[rid] = {
                    "id": rid, "index": idx,
                    "content":  self.records[idx]["content"],
                    "metadata": self.records[idx]["metadata"],
                    "dense_rank": None, "dense_distance": None,
                    "bm25_rank": None, "metadata_rank": None, "rrf_score": 0.0,
                }
        for rid, item in fused.items():
            score = 0.0
            if item.get("dense_rank") is not None:
                score += 1.0 / (60.0 + item["dense_rank"])
            if rid in bm25_ranks:
                score += 1.0 / (60.0 + bm25_ranks[rid])
            if rid in metadata_ranks:
                score += 1.0 / (60.0 + metadata_ranks[rid])
            item["bm25_rank"]     = bm25_ranks.get(rid)
            item["metadata_rank"] = metadata_ranks.get(rid)
            item["rrf_score"]     = score
        return dict(sorted(fused.items(), key=lambda x: x[1]["rrf_score"], reverse=True))

    def _rerank(self, expanded_query, fused_candidates):
        pool = list(fused_candidates.values())[:40]
        if not pool:
            return []
        if self.reranker is None:
            for item in pool:
                item["final_score"] = item["rrf_score"]
            return sorted(pool, key=lambda x: x["final_score"], reverse=True)
        pairs = [[expanded_query, item["content"]] for item in pool]
        try:
            scores = [float(s) for s in self.reranker.predict(pairs, show_progress_bar=False)]
            mn, mx = min(scores), max(scores)
            rng    = mx - mn
            for item, score in zip(pool, scores):
                norm = (score - mn) / rng if rng > 0 else 1.0
                item["reranker_score"] = score
                item["final_score"]    = 0.70 * norm + 0.30 * item["rrf_score"]
            return sorted(pool, key=lambda x: x["final_score"], reverse=True)
        except Exception:
            for item in pool:
                item["final_score"] = item["rrf_score"]
            return sorted(pool, key=lambda x: x["final_score"], reverse=True)

    def _select_diverse(self, ranked):
        selected = []
        source_counts = {}
        page_counts   = {}
        for c in ranked:
            meta        = c.get("metadata", {})
            source_file = safe_text(meta.get("source_file", "unknown"))
            page        = safe_text(meta.get("page_display", meta.get("page", "?")))
            sk, pk      = source_file, f"{source_file}|{page}"
            if source_counts.get(sk, 0) >= 5:
                continue
            if page_counts.get(pk, 0) >= 2:
                continue
            selected.append(c)
            source_counts[sk] = source_counts.get(sk, 0) + 1
            page_counts[pk]   = page_counts.get(pk, 0) + 1
            current_chars = sum(len(i.get("content", "")) for i in selected)
            if len(selected) >= self.final_context_chunks or current_chars >= self.max_context_chars:
                break
        return selected

    def _citation_for(self, item) -> str:
        meta = item.get("metadata", {})
        return (
            f"{safe_text(meta.get('source_file', 'unknown'))}, "
            f"page {safe_text(meta.get('page_display', meta.get('page', '?')))}, "
            f"{safe_text(meta.get('source_authority', 'Unknown'))}, "
            f"{safe_text(meta.get('document_type', 'agricultural_document'))}"
        )

    def _build_context_text(self, selected) -> str:
        blocks = []
        for n, item in enumerate(selected, start=1):
            meta   = item.get("metadata", {})
            blocks.append(
                f"[EVIDENCE {n}]\n"
                f"Source: {self._citation_for(item)}\n"
                f"Crop metadata: {safe_text(meta.get('crop', ''))}\n"
                f"State/region metadata: {safe_text(meta.get('state', ''))}\n"
                f"Topics: {safe_text(meta.get('topics', ''))}\n"
                f"Content:\n{item.get('content', '')}\n"
            )
        return "\n\n".join(blocks)

    def retrieve(
        self,
        query: str,
        farmer_context: Dict[str, Any],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> Dict[str, Any]:
        print("\n" + "=" * 70)
        print("HYBRID AGRICULTURAL RETRIEVAL")
        print("=" * 70)

        expanded, terms = self._expand_query(query, farmer_context)
        print(f"Original query : {query}")
        print(f"Expanded query : {expanded}")

        weather_raw = {}
        if latitude is not None and longitude is not None:
            print("Fetching weather…")
            weather_raw = fetch_open_meteo(latitude, longitude, forecast_days=16)
            print(f"Weather status: {weather_raw.get('_status')}")

        dense_candidates = self._dense_search(expanded)
        bm25_ranks       = self._bm25_search(expanded)
        metadata_ranks   = self._metadata_search(terms)
        fused            = self._rrf(dense_candidates, bm25_ranks, metadata_ranks)
        reranked         = self._rerank(expanded, fused)
        selected         = self._select_diverse(reranked)
        context_text     = self._build_context_text(selected)
        citations        = [self._citation_for(i) for i in selected]
        weather_summary  = weather_statistics(weather_raw)
        unique_sources   = sorted(set(
            safe_text(i.get("metadata", {}).get("source_file", "unknown"))
            for i in selected
        ))

        print(f"Final context chunks : {len(selected)}")
        print(f"Context characters   : {len(context_text)}")
        print(f"Unique source PDFs   : {len(unique_sources)}")

        return {
            "original_query":  query,
            "expanded_query":  expanded,
            "detected_terms":  terms,
            "farmer_context":  farmer_context,
            "weather_raw":     weather_raw,
            "weather_summary": weather_summary,
            "selected_results": selected,
            "context_text":    context_text,
            "citations":       citations,
            "unique_sources":  unique_sources,
        }


# ============================================================
# ML MODEL WRAPPER
# ============================================================

class AgriculturalModels:
    def __init__(self):
        print("\n" + "=" * 70)
        print("LOADING AGRICULTURAL ML MODELS")
        print("=" * 70)
        self.production_model = None
        self.crop_model       = None
        if PRODUCTION_MODEL_FILE.exists():
            self.production_model = joblib.load(PRODUCTION_MODEL_FILE)
            print(f"Loaded: {PRODUCTION_MODEL_FILE.name}")
        else:
            print(f"Production model not found: {PRODUCTION_MODEL_FILE}")
        if CROP_RECOMMENDATION_MODEL_FILE.exists():
            self.crop_model = joblib.load(CROP_RECOMMENDATION_MODEL_FILE)
            print(f"Loaded: {CROP_RECOMMENDATION_MODEL_FILE.name}")
        else:
            print(f"Crop recommendation model not found: {CROP_RECOMMENDATION_MODEL_FILE}")

    def _safe_predict(self, model, features):
        if model is None:
            return None
        try:
            pred = model.predict(features)
            return pred.tolist() if isinstance(pred, np.ndarray) else pred
        except Exception as e1:
            try:
                pred = model.predict(features.to_numpy())
                return pred.tolist() if isinstance(pred, np.ndarray) else pred
            except Exception as e2:
                return {"error": f"{e1}; {e2}"}

    def predict_production(self, farmer_context: Dict[str, Any]) -> Dict[str, Any]:
        if self.production_model is None:
            return {"status": "unavailable", "reason": "Model file not found."}
        flat   = flatten_dictionary(farmer_context)
        state  = flat.get("state") or flat.get("state_name") or ""
        district = flat.get("district") or flat.get("district_name") or ""
        season = flat.get("season", "")
        crop   = flat.get("crop") or flat.get("crop_name") or ""
        area   = float(flat.get("area") or flat.get("land_area") or 0)
        try:
            crop_year = int(float(flat.get("crop_year", 2026)))
        except Exception:
            crop_year = 2026
        features = pd.DataFrame([{
            "state_name": state, "district_name": district,
            "crop_year": crop_year, "season": season, "crop": crop, "area": area,
        }])
        return {
            "status": "success",
            "model":  PRODUCTION_MODEL_FILE.name,
            "features": {"state_name": state, "district_name": district,
                         "crop_year": crop_year, "season": season, "crop": crop, "area": area},
            "prediction": self._safe_predict(self.production_model, features),
            "interpretation": "Historical production/yield estimate.",
        }

    def predict_crop_recommendation(self, farmer_context: Dict[str, Any]) -> Dict[str, Any]:
        if self.crop_model is None:
            return {"status": "unavailable", "reason": "Model file not found."}
        flat = flatten_dictionary(farmer_context)

        def g(*keys, default=0.0):
            for k in keys:
                if k in flat:
                    try: return float(flat[k])
                    except Exception: pass
            return default

        n, p, k = g("nitrogen", "n", "soil_nitrogen"), g("phosphorus", "p", "soil_phosphorus"), g("potassium", "k", "soil_potassium")
        temp    = g("temperature", "temperature_2m", "current_temperature")
        hum     = g("humidity", "relative_humidity", "relative_humidity_2m")
        ph      = g("ph", "soil_ph")
        rain    = g("rainfall", "precipitation", "rainfall_mm")

        features = pd.DataFrame([{
            "n": n, "p": p, "k": k,
            "temperature": temp, "humidity": hum, "ph": ph, "rainfall": rain,
        }])
        probabilities = None
        try:
            probabilities = self.crop_model.predict_proba(features).tolist()
        except Exception:
            pass
        return {
            "status": "success",
            "model":  CROP_RECOMMENDATION_MODEL_FILE.name,
            "features": {"n": n, "p": p, "k": k, "temperature": temp,
                         "humidity": hum, "ph": ph, "rainfall": rain},
            "prediction":    self._safe_predict(self.crop_model, features),
            "probabilities": probabilities,
        }

    def run_all(self, farmer_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "production_model":           self.predict_production(farmer_context),
            "crop_recommendation_model":  self.predict_crop_recommendation(farmer_context),
        }


# ============================================================
# GROQ PROMPT BUILDER
# ============================================================

def create_groq_prompt(
    farmer_context: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    ml_result: Dict[str, Any],
) -> str:
    farmer_json  = json.dumps(farmer_context,  ensure_ascii=False, indent=2)
    weather_json = json.dumps(retrieval_result.get("weather_summary", {}), ensure_ascii=False, indent=2)
    ml_json      = json.dumps(ml_result,       ensure_ascii=False, indent=2)
    evidence     = retrieval_result.get("context_text", "")

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
Never blindly follow an ML prediction when it conflicts with soil, weather,
irrigation, crop season, regional suitability, or reliable retrieved evidence.

Do not use Bhoonidhi. Do not use VEDAS.
Do not invent market prices, disease outbreaks, pesticide doses, fertilizer
doses, crop varieties, or yield values.

If the retrieved evidence does not support a specific recommendation, state:
"Evidence is insufficient for this specific recommendation."

For pesticide or chemical recommendations:
- Prefer prevention, monitoring, sanitation, drainage, and cultural control.
- Do not recommend chemicals unless mentioned in the evidence.
- Always tell the farmer to verify with the nearest KVK or agriculture officer.

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
{evidence}
"""


# ============================================================
# GROQ LLM CALLER
# ============================================================

# Best available models on this account (update if your plan changes)
GROQ_PRIMARY_MODEL  = "qwen/qwen3.8-27b"
GROQ_FALLBACK_MODEL = "openai/gpt-oss-20b"


def call_groq(prompt: str, model_name: str = None) -> str:
    """Call Groq LLM. Auto-falls back to secondary model on 404."""
    if model_name is None:
        model_name = GROQ_PRIMARY_MODEL
    try:
        from groq import Groq
    except Exception as e:
        return f"Groq package not installed. pip install groq. Details: {e}"

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not found. Add it to .env or environment."

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a careful, evidence-grounded Indian agricultural scientist."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        return response.choices[0].message.content
    except Exception as e:
        err = str(e)
        # Auto-fallback: if model not found, retry with fallback model
        if ("model_not_found" in err or "404" in err or "does not exist" in err):
            if model_name != GROQ_FALLBACK_MODEL:
                print(f"Model '{model_name}' not available. Retrying with '{GROQ_FALLBACK_MODEL}'...")
                return call_groq(prompt, model_name=GROQ_FALLBACK_MODEL)
        return f"Groq request failed: {e}"

