#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 13:09:42 2025

@author: domenico
"""

# scripts/preload_model.py
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TARGET = os.environ.get("LOCAL_MODEL_DIR", "./.models/minilm")

Path(TARGET).mkdir(parents=True, exist_ok=True)
print(f"[preload] downloading '{MODEL_ID}' → {TARGET}")
model = SentenceTransformer(MODEL_ID, device="cpu")
model.save(TARGET)
print("[preload] done")
