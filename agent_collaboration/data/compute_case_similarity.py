"""
Compute pairwise similarity across every case entry defined in data/case.json.
The script follows the embedding workflow demonstrated in the reference snippet:
  - serialize each case into a single paragraph-style text
  - encode texts with SentenceTransformer (moka-ai/m3e-base)
  - build a cosine-similarity matrix (via normalized dot product)
  - rescale it to a standard-normal CDF (mean 0.5) and save as a NumPy matrix
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


CASE_PATH = Path("data/case.json")
OUTPUT_MATRIX = Path("data/case_similarity_matrix.npy")
MODEL_NAME = "moka-ai/m3e-base"
EMB_BATCH_SIZE = 64


def _objectives_text(case_obj: Dict[str, Any]) -> str:
    """Flatten objectives (either objextives or objectives_by_scene) into text."""
    blocks = case_obj.get("objectives_by_scene") or case_obj.get("objextives") or []
    if not isinstance(blocks, list):
        return ""

    sections: List[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for title, content in block.items():
            sections.append(str(title))
            if isinstance(content, dict):
                section_lines = [f"{k}：{v}" for k, v in content.items()]
                sections.append("\n".join(section_lines))
            else:
                sections.append(str(content))

    return "\n\n".join(section for section in sections if section)


def build_case_text(case_id: str, case_obj: Dict[str, Any]) -> str:
    """Serialize contextualized_case + objectives + optional original_case into one string."""
    parts: List[str] = [case_id]

    context = case_obj.get("contextualized_case")
    if isinstance(context, list):
        parts.append("\n".join(str(seg) for seg in context if seg))

    objectives = _objectives_text(case_obj)
    if objectives:
        parts.append(objectives)

    original_case = case_obj.get("original_case")
    if isinstance(original_case, str):
        parts.append(original_case)

    return "\n\n".join(part for part in parts if part)


def load_cases() -> List[str]:
    """Load case.json and return all cases serialized to text."""
    with CASE_PATH.open("r", encoding="utf-8") as f:
        cases: Dict[str, Dict[str, Any]] = json.load(f)

    case_ids = sorted(cases.keys())
    return [build_case_text(case_id, cases[case_id]) for case_id in case_ids]


def encode_cases(texts: List[str]) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            # normalize_embeddings=True ensures dot product equals cosine similarity
            embeddings = model.encode(
                texts,
                batch_size=EMB_BATCH_SIZE,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
    return embeddings


def main() -> None:
    texts = load_cases()
    embeddings = encode_cases(texts)

    similarity = embeddings @ embeddings.T
    sim_mean = similarity.mean()
    sim_std = similarity.std(unbiased=False)
    if sim_std == 0:
        normalized = torch.full_like(similarity, 0.5)
    else:
        z = (similarity - sim_mean) / (sim_std + 1e-12)
        normalized = 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))

    similarity_np = normalized.to(dtype=torch.float32).cpu().numpy()

    OUTPUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_MATRIX, similarity_np)

    print(f"Similarity matrix saved to {OUTPUT_MATRIX}")


if __name__ == "__main__":
    main()
