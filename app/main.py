import logging
from logging.config import dictConfig
import os
import sqlite3
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel

# ---------------- Logging (unchanged) ----------------
os.makedirs("logs", exist_ok=True)
log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s - %(message)s",
        },
        "file": {
            "format": "%(levelname)s: %(asctime)s - %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "formatter": "console",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "file": {
            "formatter": "file",
            "class": "logging.FileHandler",
            "filename": "logs/hrm_agent.log",
            "mode": "a",
            "encoding": "utf-8"
        },
    },
    "loggers": {
        "hrm_agent": {
            "handlers": ["console", "file"],
            "level": "INFO",
        },
    },
}
dictConfig(log_config)
logger = logging.getLogger("hrm_agent")

# ---------------- App ----------------
app = FastAPI(
    title="HRM Agent API",
    openapi_url="/api/v1/openapi.json",
)

DB_FILE = os.getenv("DB_FILE", "places.db")  # keep your db in same folder

# ---------------- Optional deps (semantic + fuzzy) ----------------
EMBEDDINGS_AVAILABLE = False
FUZZY_AVAILABLE = False
DATEPARSER_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _embed_model: Optional[SentenceTransformer] = None
    EMBEDDINGS_AVAILABLE = True
except Exception as e:
    logger.warning(f"sentence-transformers not available: {e}")
    _embed_model = None

try:
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except Exception as e:
    logger.warning(f"rapidfuzz not available: {e}")

try:
    import dateparser
    DATEPARSER_AVAILABLE = True
except Exception as e:
    logger.warning(f"dateparser not available: {e}")

# ---------------- Data structures ----------------
@dataclass
class PlaceRow:
    place_id: int
    title: str

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    debug: bool = False

# ---------------- Helpers ----------------
def connect_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def fetch_places() -> List[PlaceRow]:
    with connect_db() as conn:
        rows = conn.execute("SELECT place_id, title FROM places").fetchall()
    return [PlaceRow(place_id=row["place_id"], title=row["title"]) for row in rows]

_PLACES_CACHE: List[PlaceRow] = []
_TITLE_EMB: Optional[Any] = None  # numpy array if embeddings are available

def ensure_model():
    global _embed_model
    if EMBEDDINGS_AVAILABLE and _embed_model is None:
        # Small, fast, general-purpose model
        _embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts: List[str]):
    ensure_model()
    if EMBEDDINGS_AVAILABLE and _embed_model is not None:
        return _embed_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return None  # fallback path uses fuzzy matching

def cosine_top_k(query: str, corpus_texts: List[str], k: int = 5) -> List[Tuple[int, float]]:
    """
    Return [(index, score)] sorted by score desc using embeddings if available,
    else fuzzy ratio fallback.
    """
    if EMBEDDINGS_AVAILABLE and _embed_model is not None:
        q = _embed_model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        c = embed_texts(corpus_texts) if isinstance(_TITLE_EMB, type(None)) else _TITLE_EMB
        sims = st_util.cos_sim(q, c)[0].tolist()
        ranked = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)[:k]
        return ranked
    # Fallback: fuzzy token sort ratio
    if FUZZY_AVAILABLE:
        scores = [(i, fuzz.token_sort_ratio(query, t) / 100.0) for i, t in enumerate(corpus_texts)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    # Last resort: naive substring presence
    scores = []
    ql = query.lower()
    for i, t in enumerate(corpus_texts):
        tl = t.lower()
        s = 1.0 if ql in tl or tl in ql else 0.0
        scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]

def init_place_cache():
    global _PLACES_CACHE, _TITLE_EMB
    _PLACES_CACHE = fetch_places()
    titles = [p.title for p in _PLACES_CACHE]
    if EMBEDDINGS_AVAILABLE:
        _TITLE_EMB = embed_texts(titles)

def query_db(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    with connect_db() as conn:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
    return [dict(r) for r in rows]

# ---------------- Time parsing ----------------
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAYS_LOWER = [d.lower() for d in DAYS]

def parse_day_time(q: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract day_of_week and hour (0-23) from free text.
    Supports 'now', explicit day names, HH, HH am/pm.
    """
    ql = q.lower()

    # 'now' shortcut using IST
    if "now" in ql:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        return (now.strftime("%A"), now.hour)

    # day of week
    day_of_week = None
    for i, d in enumerate(DAYS_LOWER):
        if d in ql:
            day_of_week = DAYS[i]
            break

    # time extraction via dateparser if available (more robust)
    hour = None
    if DATEPARSER_AVAILABLE:
        # We don't want a full date in future/past; we just need time if present
        dt = dateparser.parse(q, settings={"PREFER_DATES_FROM": "future"})
        if dt is not None:
            hour = dt.hour
    else:
        # regex 10, 10am, 10 pm, 22:00 etc.
        m = re.search(r"\b(\d{1,2})(?:\s*[:.]\s*(\d{2}))?\s*(am|pm)?\b", ql)
        if m:
            hh = int(m.group(1))
            ampm = m.group(3)
            if ampm == "pm" and hh != 12:
                hh += 12
            if ampm == "am" and hh == 12:
                hh = 0
            if 0 <= hh <= 23:
                hour = hh

    return (day_of_week, hour)

# ---------------- Intent definitions ----------------
INTENT_DESCRIPTIONS = [
    ("quiet_places", "Find quiet, best, peaceful places with the lowest average crowd."),
    ("crowded_places", "Find busy, popular, most crowded places with highest average crowd."),
    ("search_by_title", "Find places whose names or themes match the user text."),
    ("occupancy_at_time", "Find places ranked by occupancy at a specific day of week and hour."),
    ("best_time_for_place", "Find the least crowded time (day, hour) to visit a specific place."),
]

def detect_intent(q: str, place_titles: List[str]) -> Tuple[str, Dict[str, Any], Dict[str, float]]:
    # default fallback
    intent = "unknown_intent"
    slots = {"day_of_week": None, "hour": None, "place": None, "place_id": None}
    intent_scores = {}

    try:
        day, hour = parse_day_time(q)

        title_top1 = cosine_top_k(q, place_titles, k=1)
        place_idx = title_top1[0][0] if title_top1 else None
        place_score = title_top1[0][1] if title_top1 else 0.0
        target_place = _PLACES_CACHE[place_idx] if place_idx is not None and place_score >= 0.45 else None

        # Semantic scoring
        intent_texts = [desc for _, desc in INTENT_DESCRIPTIONS]
        scored = cosine_top_k(q, intent_texts, k=len(intent_texts))  # [(idx, score), ...]
        idx_to_intent = {i: INTENT_DESCRIPTIONS[i][0] for i in range(len(INTENT_DESCRIPTIONS))}
        intent_scores = {idx_to_intent[idx]: score for idx, score in scored}

        # Decide intent heuristically
        if day and (hour is not None):
            intent = "occupancy_at_time"
        elif target_place and re.search(r"\b(best time|when|less crowded|least crowded|avoid crowd)\b", q, re.I):
            intent = "best_time_for_place"
        elif intent_scores:
            intent = max(intent_scores.items(), key=lambda x: x[1])[0]

        slots = {
            "day_of_week": day,
            "hour": hour,
            "place": target_place.title if target_place else None,
            "place_id": target_place.place_id if target_place else None,
        }
    except Exception as e:
        # log exception if needed
        print("detect_intent error:", e)

    return intent, slots, intent_scores

# ---------------- SQL Builders ----------------
def sql_quiet_places(limit: int):
    return ("""
        SELECT p.place_id, p.title, ROUND(AVG(pt.occupancy_percent), 2) AS avg_occupancy
        FROM places p
        JOIN popular_times pt ON p.place_id = pt.place_id
        GROUP BY p.place_id
        HAVING COUNT(pt.id) > 0
        ORDER BY avg_occupancy ASC
        LIMIT ?
    """, (limit,))

def sql_crowded_places(limit: int):
    return ("""
        SELECT p.place_id, p.title, ROUND(AVG(pt.occupancy_percent), 2) AS avg_occupancy
        FROM places p
        JOIN popular_times pt ON p.place_id = pt.place_id
        GROUP BY p.place_id
        HAVING COUNT(pt.id) > 0
        ORDER BY avg_occupancy DESC
        LIMIT ?
    """, (limit,))

def sql_search_by_title_like(q: str, limit: int):
    # semantic shortlist -> LIKE
    # Expand a few water-related hints to help titles that include beach/coast/etc.
    HINTS = ["sea", "ocean", "beach", "coast", "bay", "waterfront", "seaside", "shore", "marina", "harbor", "lagoon"]
    terms = [t for t in re.split(r"[^a-zA-Z0-9]+", q) if t]
    like_terms = terms[:]
    if any(h in q.lower() for h in HINTS):
        like_terms.extend(HINTS)
    # Build OR LIKE clauses safely
    like_sql = " OR ".join(["LOWER(p.title) LIKE ?"] * len(like_terms))
    params = tuple([f"%{t.lower()}%" for t in like_terms])
    if not like_sql:
        like_sql = "1=1"
        params = tuple()
    sql = f"""
        SELECT p.place_id, p.title, ROUND(AVG(pt.occupancy_percent), 2) AS avg_occupancy
        FROM places p
        LEFT JOIN popular_times pt ON p.place_id = pt.place_id
        WHERE {like_sql}
        GROUP BY p.place_id
        ORDER BY COALESCE(avg_occupancy, 9999) ASC
        LIMIT ?
    """
    return (sql, params + (limit,))

def sql_occupancy_at_time(day: str, hour: int, limit: int):
    return ("""
        SELECT p.place_id, p.title, pt.day_of_week, pt.hour, pt.occupancy_percent
        FROM places p
        JOIN popular_times pt ON p.place_id = pt.place_id
        WHERE UPPER(pt.day_of_week) = UPPER(?) AND pt.hour = ?
        ORDER BY pt.occupancy_percent DESC
        LIMIT ?
    """, (day, hour, limit))

def sql_best_time_for_place(place_id: int, limit: int):
    # lowest occupancy slots for the given place
    return ("""
        SELECT pt.day_of_week, pt.hour, pt.occupancy_percent
        FROM popular_times pt
        WHERE pt.place_id = ?
        ORDER BY pt.occupancy_percent ASC
        LIMIT ?
    """, (place_id, limit))

# ---------------- Core dispatcher ----------------
def answer_query(q: str, top_k: int = 5, debug: bool = False) -> Dict[str, Any]:
    if not _PLACES_CACHE:
        init_place_cache()

    titles = [p.title for p in _PLACES_CACHE]
    intent, slots, intent_scores = detect_intent(q, titles)

    sql, params = None, ()
    explanation = ""

    # --- QUICK FIX: If time is detected in query, always use occupancy_at_time ---
    if slots.get("day_of_week") and slots.get("hour") is not None:
        sql, params = sql_occupancy_at_time(slots["day_of_week"], int(slots["hour"]), top_k)
        explanation = f"Places ranked by occupancy for {slots['day_of_week']} at {int(slots['hour']):02d}:00."
        intent = "occupancy_at_time"

    # --- Other intents ---
    elif intent == "quiet_places":
        sql, params = sql_quiet_places(top_k)
        explanation = "Least crowded places overall (by average occupancy)."

    elif intent == "crowded_places":
        sql, params = sql_crowded_places(top_k)
        explanation = "Most crowded places overall (by average occupancy)."

    elif intent == "search_by_title":
        shortlist = cosine_top_k(q, titles, k=max(10, top_k))
        ids = [_PLACES_CACHE[i].place_id for i, s in shortlist if s >= 0.35]
        if ids:
            in_clause = ",".join("?" * len(ids))
            sql = f"""
                SELECT p.place_id, p.title, ROUND(AVG(pt.occupancy_percent), 2) AS avg_occupancy
                FROM places p
                LEFT JOIN popular_times pt ON p.place_id = pt.place_id
                WHERE p.place_id IN ({in_clause})
                GROUP BY p.place_id
                ORDER BY COALESCE(avg_occupancy, 9999) ASC
                LIMIT ?
            """
            params = tuple(ids) + (top_k,)
            explanation = "Places semantically similar to your query, ranked by lower average crowd."
        else:
            sql, params = sql_search_by_title_like(q, top_k)
            explanation = "Places matching your text (title search), ranked by lower average crowd."

    elif intent == "best_time_for_place" and slots.get("place_id"):
        sql, params = sql_best_time_for_place(int(slots["place_id"]), max(10, top_k))
        explanation = f"Least crowded times for {slots['place']}."

    else:
        # Fallback semantic search
        shortlist = cosine_top_k(q, titles, k=max(10, top_k))
        ids = [_PLACES_CACHE[i].place_id for i, s in shortlist if s >= 0.30]
        if ids:
            in_clause = ",".join("?" * len(ids))
            sql = f"""
                SELECT p.place_id,pt.hour,pt.day_of_week, p.title, ROUND(AVG(pt.occupancy_percent), 2) AS avg_occupancy
                FROM places p
                LEFT JOIN popular_times pt ON p.place_id = pt.place_id
                WHERE p.place_id IN ({in_clause})
                GROUP BY p.place_id
                ORDER BY COALESCE(avg_occupancy, 9999) ASC
                LIMIT ?
            """
            params = tuple(ids) + (top_k,)
            explanation = "Closest matching places based on your text."
        else:
            return {
                "query": q,
                "intent": intent,
                "slots": slots,
                "results": [],
                "message": "I couldn't map your question to the available data (place names / crowd times). Try including a place name or a day/time.",
                "debug": {"intent_scores": intent_scores} if debug else None
            }

    results = query_db(sql, params)

    return {
        "query": q,
        "intent": intent,
        "slots": slots,
        "explanation": explanation,
        "results": results,
        "debug": {
            "intent_scores": intent_scores,
            "sql": sql,
            "params": params
        } if debug else None
    }


# ---------------- Routes & Exception handlers ----------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"An unexpected error occurred: {exc}")
    return JSONResponse(status_code=500, content={"detail": "An internal server error occurred."})

@app.get("/")
async def init():
    return {"message": "Application started successfully"}

@app.post("/chat")
async def chat(payload: ChatRequest = Body(...)):
    logger.info(f"User query: {payload.query}")
    ans = answer_query(payload.query, top_k=payload.top_k, debug=payload.debug)
    return ans
