# backend/server.py
# FastAPI backend for the Trajectory-Features AI Detector
import os
import re
import json
import time
import math
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Reduce noisy HF/Tokenizers logs for web usage
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------------------
# Config & artifacts
# --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

# Where your artifacts live (set this env var before starting uvicorn)
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", str(ROOT_DIR / "exp_v4")))

ARTIFACTS_JSON = ARTIFACTS_DIR / "artifacts.json"
MODEL_PKL = ARTIFACTS_DIR / "clf.pkl"
TOP_PC_BASIS = ARTIFACTS_DIR / "top_pc_basis.npy"  # exists only if you trained with --remove_top_pcs>0

if not ARTIFACTS_JSON.exists() or not MODEL_PKL.exists():
    raise RuntimeError(
        f"Missing artifacts. Expected at least:\n  - {ARTIFACTS_JSON}\n  - {MODEL_PKL}\n\n"
        "Set ARTIFACTS_DIR to the folder created by your training run (e.g., exp_v4)."
    )

with open(ARTIFACTS_JSON, "r", encoding="utf-8") as f:
    ART = json.load(f)

# Pull what we need
MODEL_NAME: str = ART.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
CLF_FEATURES: List[str] = ART.get("clf_features") or [
    "n_sents_log", "log_path_len", "log_mean_step", "log_p90_step", "log_avg_nn_dist",
    "straightness", "dir_persistence", "turn_mean_deg", "step_cv", "burstiness", "frac_backtrack",
]
DECISION_THRESHOLD: float = float(ART.get("decision_threshold", 0.5))
CALIBRATED: bool = bool(ART.get("calibrated", False))
CALIBRATION_METHOD: str = ART.get("calibration_method", "isotonic") if CALIBRATED else "none"
REMOVE_TOP_PCS: int = int(ART.get("remove_top_pcs", 0))

# Optional PCA basis (only present if you trained with --remove_top_pcs > 0)
U_BASIS: Optional[np.ndarray] = None
if REMOVE_TOP_PCS > 0 and TOP_PC_BASIS.exists():
    try:
        U_BASIS = np.load(str(TOP_PC_BASIS))
    except Exception:
        U_BASIS = None

# Load classifier (Pipeline or CalibratedClassifierCV wrapping Pipeline)
with open(MODEL_PKL, "rb") as f:
    ESTIMATOR = pickle.load(f)

# Lazy-loaded embedder (instantiate on first request)
_EMBEDDER = None
def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(MODEL_NAME, device="cpu")
    return _EMBEDDER

# --------------------------------------------------------------------------------------
# Light, deployment-friendly preprocessing
# --------------------------------------------------------------------------------------
_ASCII_MAP = str.maketrans({
    "“": "\"", "”": "\"", "„": "\"", "‟": "\"", "«": "\"", "»": "\"",
    "‘": "'",  "’": "'",  "‚": "'",  "‛": "'",
    "—": "-",  "–": "-",  "‐": "-",
    "…": "...",
})

def sanitize_text(t: str) -> str:
    """Normalize curly quotes/dashes; strip zero-width chars; squeeze spaces."""
    t = (t or "").replace("\xa0", " ").translate(_ASCII_MAP)
    t = t.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    # Trim super long runs of whitespace
    t = " ".join(t.split())
    return t.strip()

# Minimal sentence splitter
_SENT_SPLIT_REGEX = r"(?<=[.!?])\s+(?=[A-Z0-9\"'])"
def sent_tokenize(text: str) -> List[str]:
    parts = re.split(_SENT_SPLIT_REGEX, text.strip())
    # Clean fragments
    sents = []
    for s in parts:
        s = s.strip(" \t\r\n\"'“”‘’")
        if len(s) >= 15:
            sents.append(s)
    # Cap extremely long sentences
    out = []
    for s in sents:
        words = s.split()
        if len(words) > 120:
            out.append(" ".join(words[:120]))
        else:
            out.append(s)
    return out

# --------------------------------------------------------------------------------------
# Geometry & features (match training)
# --------------------------------------------------------------------------------------
def embed_sentences(sents: List[str]) -> np.ndarray:
    if not sents:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_embedder()
    E = model.encode(
        sents,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    # exact L2-normalize
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms
    # Optional: remove top common components if basis is provided
    if U_BASIS is not None and U_BASIS.size:
        P = E.copy()
        for ui in U_BASIS:
            coeff = P @ ui.reshape(-1, 1)
            P = P - coeff * ui.reshape(1, -1)
        norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        E = (P / norms).astype(np.float32)
    return E

def step_lengths(E: np.ndarray) -> np.ndarray:
    if E.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    diffs = E[1:] - E[:-1]
    return np.linalg.norm(diffs, axis=1)

def end_to_end(E: np.ndarray) -> float:
    if E.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(E[-1] - E[0]))

def straightness(E: np.ndarray) -> float:
    s = step_lengths(E)
    L = float(s.sum()) if s.size else 0.0
    if L <= 1e-12:
        return 0.0
    return end_to_end(E) / L

def _dir_cosines(E: np.ndarray) -> np.ndarray:
    if E.shape[0] < 3:
        return np.zeros((0,), dtype=np.float32)
    V = E[1:] - E[:-1]
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    D = V / norms
    cos = (D[1:] * D[:-1]).sum(axis=1)
    return np.clip(cos, -1.0, 1.0)

def directional_persistence(E: np.ndarray) -> float:
    cs = _dir_cosines(E)
    return float(cs.mean()) if cs.size else 0.0

def mean_turn_angle_deg(E: np.ndarray) -> float:
    cs = _dir_cosines(E)
    if cs.size == 0:
        return 0.0
    ang = np.degrees(np.arccos(cs))
    return float(np.mean(np.abs(ang)))

def frac_backtrack(E: np.ndarray) -> float:
    cs = _dir_cosines(E)
    if cs.size == 0:
        return 0.0
    return float((cs < 0.0).mean())

def avg_nn_distance(E: np.ndarray) -> float:
    n = E.shape[0]
    if n <= 1:
        return 0.0
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    max_sim = S.max(axis=1)
    dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * max_sim))
    return float(dists.mean())

def p_quantile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))

def features_from_doc_embeddings(E: np.ndarray) -> Dict[str, float]:
    s = step_lengths(E)
    path_len = float(s.sum()) if s.size else 0.0
    mean_step = float(s.mean()) if s.size else 0.0
    p90_step = p_quantile(s, 0.90)
    p95_step = p_quantile(s, 0.95)
    step_std = float(s.std()) if s.size else 0.0
    step_cv = (step_std / (mean_step + 1e-12)) if s.size else 0.0
    burstiness = (p95_step / (mean_step + 1e-12)) if s.size else 0.0
    return {
        "n_sents": int(E.shape[0]),
        "path_len": path_len,
        "mean_step": mean_step,
        "p90_step": p90_step,
        "p95_step": p95_step,
        "step_cv": step_cv,
        "burstiness": burstiness,
        "straightness": straightness(E),
        "dir_persistence": directional_persistence(E),
        "turn_mean_deg": mean_turn_angle_deg(E),
        "frac_backtrack": frac_backtrack(E),
        "avg_nn_dist": avg_nn_distance(E),
    }

def to_clf_vector(feats: Dict[str, float]) -> np.ndarray:
    # Build derived/log features exactly like training did
    f = dict(feats)
    f["n_sents_log"] = math.log1p(f.get("n_sents", 0.0))
    f["log_path_len"] = math.log1p(f.get("path_len", 0.0))
    f["log_mean_step"] = math.log1p(f.get("mean_step", 0.0))
    f["log_p90_step"] = math.log1p(f.get("p90_step", 0.0))
    f["log_avg_nn_dist"] = math.log1p(f.get("avg_nn_dist", 0.0))
    vec = [float(f.get(name, 0.0)) for name in CLF_FEATURES]
    return np.asarray(vec, dtype=np.float32).reshape(1, -1)

# --------------------------------------------------------------------------------------
# Invite-code gate, total usage limit, word cap
# --------------------------------------------------------------------------------------
INVITE_CODE = os.getenv("INVITE_CODE", "DEMO-ACCESS-2025")
RATE_LIMIT_TOTAL = int(os.getenv("RATE_LIMIT_TOTAL", "10"))  # total detections per code
MAX_WORDS = int(os.getenv("MAX_WORDS", "1500"))

USAGE: Dict[str, int] = {}

def count_words(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])

def check_gate(request: Request):
    code = request.headers.get("X-Invite-Code", "").strip()
    if not code or code != INVITE_CODE:
        raise HTTPException(status_code=401, detail="Invalid or missing invite code.")
    used = USAGE.get(code, 0)
    if used >= RATE_LIMIT_TOTAL:
        raise HTTPException(status_code=429, detail="Rate limit reached for this invite code.")
    USAGE[code] = used + 1

# --------------------------------------------------------------------------------------
# FastAPI app & routes
# --------------------------------------------------------------------------------------
app = FastAPI(title="AI Detector · Trajectory Features", version="1.0.0")
api = APIRouter(prefix="/api")

@api.get("/healthz")
def healthz():
    return {"ok": True}

class DetectIn(BaseModel):
    text: str

@api.post("/detect")
def detect(payload: DetectIn, request: Request):
    # gate + rate limit + word cap
    check_gate(request)
    raw_text = (payload.text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=422, detail="Empty text.")
    if count_words(raw_text) > MAX_WORDS:
        raise HTTPException(status_code=413, detail=f"Input too long. Max {MAX_WORDS} words.")

    t0 = time.time()

    # 1) Preprocess
    text = sanitize_text(raw_text)
    sents = sent_tokenize(text)
    if len(sents) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 complete sentences to analyze.")

    # 2) Embed
    E = embed_sentences(sents)

    # 3) Features -> vector
    feats = features_from_doc_embeddings(E)
    X = to_clf_vector(feats)

    # 4) Predict proba -> threshold
    try:
        prob_ai = float(ESTIMATOR.predict_proba(X)[:, 1][0])
    except Exception as e:
        try:
            prob_ai = float(ESTIMATOR.base_estimator_.predict_proba(X)[:, 1][0])  # type: ignore[attr-defined]
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Model error: {e2}") from e

    label = "ai" if prob_ai >= DECISION_THRESHOLD else "human"
    latency_ms = round((time.time() - t0) * 1000.0, 1)

    preview = {
        "n_sents": feats["n_sents"],
        "path_len": round(feats["path_len"], 4),
        "mean_step": round(feats["mean_step"], 4),
        "p90_step": round(feats["p90_step"], 4),
        "straightness": round(feats["straightness"], 4),
        "dir_persistence": round(feats["dir_persistence"], 4),
        "avg_nn_dist": round(feats["avg_nn_dist"], 4),
    }

    return JSONResponse({
        "label": label,
        "prob_ai": prob_ai,
        "threshold": DECISION_THRESHOLD,
        "n_sents": feats["n_sents"],
        "model_name": MODEL_NAME,
        "calibrated": CALIBRATED,
        "calibration_method": CALIBRATION_METHOD,
        "latency_ms": latency_ms,
        "debug": {
            "features_preview": preview,
        },
    })

@api.get("/meta")
def meta():
    return {
        "artifacts_dir": str(ARTIFACTS_DIR),
        "model_name": MODEL_NAME,
        "threshold": DECISION_THRESHOLD,
        "calibrated": CALIBRATED,
        "calibration_method": CALIBRATION_METHOD,
        "clf_features": CLF_FEATURES,
        "remove_top_pcs": REMOVE_TOP_PCS,
    }

# Include API router
app.include_router(api)

# Landing and app routes (served before the static mount)
@app.get("/", response_class=HTMLResponse)
def landing():
    lp = FRONTEND_DIR / "landing.html"
    if lp.exists():
        return FileResponse(str(lp))
    # Fallback: if landing missing, go straight to app
    ix = FRONTEND_DIR / "index.html"
    if ix.exists():
        return FileResponse(str(ix))
    return HTMLResponse("<h1>Frontend missing</h1>", status_code=200)

@app.get("/app", response_class=HTMLResponse)
def app_page():
    ix = FRONTEND_DIR / "index.html"
    if ix.exists():
        return FileResponse(str(ix))
    return HTMLResponse("<h1>index.html missing</h1>", status_code=200)

# Finally, mount the static frontend (so direct asset links work)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

