# backend/server.py
# FastAPI backend for the Trajectory-Features AI Detector

import os
import re
import json
import time
import math
import pickle
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", str(ROOT_DIR / "exp_v4")))

INVITE_CODE = os.environ.get("INVITE_CODE", "").strip()
RATE_LIMIT_TOTAL = int(os.environ.get("RATE_LIMIT_TOTAL", "10"))
MAX_WORDS = int(os.environ.get("MAX_WORDS", "1500"))

# Prefer local model dir (preloaded at build); otherwise use model name from artifacts
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "").strip()

ARTIFACTS_JSON = ARTIFACTS_DIR / "artifacts.json"
MODEL_PKL = ARTIFACTS_DIR / "clf.pkl"
TOP_PC_BASIS = ARTIFACTS_DIR / "top_pc_basis.npy"

if not ARTIFACTS_JSON.exists() or not MODEL_PKL.exists():
    raise RuntimeError(
        f"Missing artifacts. Need:\n  - {ARTIFACTS_JSON}\n  - {MODEL_PKL}\n"
        "Set ARTIFACTS_DIR to your exp folder (e.g., exp_v4)."
    )

with open(ARTIFACTS_JSON, "r", encoding="utf-8") as f:
    ART = json.load(f)

MODEL_NAME: str = ART.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
CLF_FEATURES = ART.get("clf_features") or [
    "n_sents_log", "log_path_len", "log_mean_step", "log_p90_step", "log_avg_nn_dist",
    "straightness", "dir_persistence", "turn_mean_deg", "step_cv", "burstiness", "frac_backtrack",
]
DECISION_THRESHOLD: float = float(ART.get("decision_threshold", 0.5))
CALIBRATED: bool = bool(ART.get("calibrated", False))
CALIBRATION_METHOD: str = ART.get("calibration_method", "isotonic") if CALIBRATED else "none"
REMOVE_TOP_PCS: int = int(ART.get("remove_top_pcs", 0))

U_BASIS: Optional[np.ndarray] = None
if REMOVE_TOP_PCS > 0 and TOP_PC_BASIS.exists():
    try:
        U_BASIS = np.load(str(TOP_PC_BASIS))
    except Exception:
        U_BASIS = None

with open(MODEL_PKL, "rb") as f:
    ESTIMATOR = pickle.load(f)

USAGE = defaultdict(int)

_EMBEDDER = None
def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        load_from = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR else MODEL_NAME
        print(f"[embedder] loading from: {load_from}")
        _EMBEDDER = SentenceTransformer(load_from, device="cpu")
    return _EMBEDDER

_ASCII_MAP = str.maketrans({
    "“": "\"", "”": "\"", "„": "\"", "‟": "\"", "«": "\"", "»": "\"",
    "‘": "'",  "’": "'",  "‚": "'",  "‛": "'",
    "—": "-",  "–": "-",  "‐": "-",
    "…": "...",
})
def sanitize_text(t: str) -> str:
    t = (t or "").replace("\xa0", " ").translate(_ASCII_MAP)
    t = t.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    t = " ".join(t.split())
    return t.strip()

_SENT_SPLIT_REGEX = r"(?<=[.!?])\s+(?=[A-Z0-9\"'])"
def sent_tokenize(text: str) -> List[str]:
    parts = re.split(_SENT_SPLIT_REGEX, text.strip())
    sents = []
    for s in parts:
        s = s.strip(" \t\r\n\"'“”‘’")
        if len(s) >= 15:
            sents.append(s)
    out = []
    for s in sents:
        out.append(" ".join(s.split()[:120]) if len(s.split()) > 120 else s)
    return out

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
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms
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
    import numpy as _np
    ang = _np.degrees(_np.arccos(cs))
    return float(_np.mean(_np.abs(ang)))

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
    f = dict(feats)
    f["n_sents_log"]     = math.log1p(f.get("n_sents", 0.0))
    f["log_path_len"]    = math.log1p(f.get("path_len", 0.0))
    f["log_mean_step"]   = math.log1p(f.get("mean_step", 0.0))
    f["log_p90_step"]    = math.log1p(f.get("p90_step", 0.0))
    f["log_avg_nn_dist"] = math.log1p(f.get("avg_nn_dist", 0.0))
    vec = [float(f.get(name, 0.0)) for name in CLF_FEATURES]
    return np.asarray(vec, dtype=np.float32).reshape(1, -1)

app = FastAPI(title="AI Detector · Trajectory Features", version="1.2.0")

@app.on_event("startup")
def _startup():
    try:
        model = get_embedder()
        _ = model.encode(["Hello world."], convert_to_numpy=True, normalize_embeddings=True)
        print("[startup] Embedder ready.")
    except Exception as e:
        print("[startup] Embedder init failed:", e)
        traceback.print_exc()

@app.get("/", response_class=FileResponse)
def landing():
    f = FRONTEND_DIR / "landing.html"
    if not f.exists():
        return PlainTextResponse("landing.html missing", status_code=500)
    return FileResponse(str(f))

@app.get("/app", response_class=FileResponse)
def app_page():
    f = FRONTEND_DIR / "index.html"
    if not f.exists():
        return PlainTextResponse("index.html missing", status_code=500)
    return FileResponse(str(f))

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=False), name="static")

api = APIRouter(prefix="/api")

@api.get("/healthz")
def healthz():
    ok = _EMBEDDER is not None
    return {"ok": True, "embedder_loaded": ok, "model": (LOCAL_MODEL_DIR or MODEL_NAME)}

class DetectIn(BaseModel):
    text: str

def _get_invite(req: Request) -> str:
    code = req.headers.get("X-Invite-Code", "").strip()
    if not code:
        qp = req.query_params.get("code", "")
        code = (qp or "").strip()
    return code

def _count_words(s: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", s or ""))

@api.post("/detect")
async def detect(req: Request, payload: DetectIn):
    t0 = time.time()

    if INVITE_CODE:
        code = _get_invite(req)
        if not code:
            raise HTTPException(status_code=401, detail="Missing invite code.")
        if code != INVITE_CODE:
            raise HTTPException(status_code=403, detail="Invalid invite code.")
        if USAGE[code] >= RATE_LIMIT_TOTAL:
            raise HTTPException(status_code=429, detail="Usage limit reached for this invite code.")
        USAGE[code] += 1

    raw_text = (payload.text or "").trim()
    if not raw_text:
        raise HTTPException(status_code=422, detail="Empty text.")
    wc = _count_words(raw_text)
    if wc > MAX_WORDS:
        raise HTTPException(status_code=413, detail=f"Input too long: {wc} words. Max is {MAX_WORDS}.")

    text = sanitize_text(raw_text)
    sents = sent_tokenize(text)
    if len(sents) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 complete sentences to analyze.")

    try:
        E = embed_sentences(sents)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    feats = features_from_doc_embeddings(E)
    X = to_clf_vector(feats)

    try:
        prob_ai = float(ESTIMATOR.predict_proba(X)[:, 1][0])
    except Exception as e:
        try:
            prob_ai = float(ESTIMATOR.base_estimator_.predict_proba(X)[:, 1][0])  # type: ignore[attr-defined]
        except Exception as e2:
            traceback.print_exc()
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
        "model_name": (LOCAL_MODEL_DIR or MODEL_NAME),
        "calibrated": CALIBRATED,
        "calibration_method": CALIBRATION_METHOD,
        "latency_ms": latency_ms,
        "debug": {"features_preview": preview},
        "remaining": (RATE_LIMIT_TOTAL - USAGE[code]) if INVITE_CODE else None
    })

app.include_router(api)

