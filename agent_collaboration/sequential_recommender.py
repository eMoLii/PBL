"""
序列自适应效用模型（Sequential Adaptive Utility Model）推荐工具。

本文件实现了提示中描述的序列感知推荐流程：
    1) 针对历史中的每个案例，比较候选案例与之的相似度并计算“假效用”；
    2) 使用随时间衰减的权重（Recency）对假效用进行加权；
    3) 聚合得到候选案例相对于历史 H、成绩 S 的整体效用 U(j | H, S)，按得分排序并推荐。

末尾包含两个示例：
    - run_toy_demo(): 手写的 4×4 相似度矩阵；
    - run_real_demo(): 若存在 data/case_similarity_matrix.npy，将用真实案例跑一次示例。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


SCORE_MIN = 0.0
SCORE_MAX = 5.0


def _clip_score(score: float) -> float:
    if np.isnan(score):
        raise ValueError("Score contains NaN.")
    return float(min(max(score, SCORE_MIN), SCORE_MAX))


@dataclass
class Recommendation:
    case_id: str
    utility: float


class SequentialAdaptiveRecommender:
    """
    基于序列自适应效用模型计算推荐结果。

    参数
    ----------
    sim_matrix: np.ndarray
        预先计算好的案例相似度矩阵（值域 0~1）。
    case_ids: Sequence[str]
        与矩阵索引一一对应的案例 ID；为了和 compute_case_similarity.py 对齐，这里也用排序后的 ID。
    """

    def __init__(self, sim_matrix: np.ndarray, case_ids: Sequence[str]):
        if sim_matrix.ndim != 2 or sim_matrix.shape[0] != sim_matrix.shape[1]:
            raise ValueError("Similarity matrix must be square.")
        if sim_matrix.shape[0] != len(case_ids):
            raise ValueError("case_ids length must match sim_matrix dimension.")

        self.sim_matrix = sim_matrix.astype(np.float32, copy=False)
        self.case_ids = list(case_ids)
        self.id_to_idx = {cid: i for i, cid in enumerate(self.case_ids)}

    @staticmethod
    def _sim_target(score: float, tau: float) -> float:
        """
        目标相似度 (公式 1)：随历史得分调节的理想相似度。

        score: 学生在历史案例上的得分（0~5）。
        tau: 达到最高分时的相似度下限（默认 0.5）。
        """
        tau = float(np.clip(tau, 0.0, 1.0))
        normalized = _clip_score(score) / SCORE_MAX
        return 1.0 - (1.0 - tau) * normalized

    @staticmethod
    def _partial_utility(sim_val: float, target: float, sigma: float) -> float:
        """
        假效用 U(j | i_k, s_k) (公式 2)：越接近目标相似度，值越高。
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        diff = float(sim_val) - float(target)
        return float(np.exp(-(diff * diff) / (2.0 * sigma * sigma)))

    @staticmethod
    def _normalized_importance_weights(length: int, gamma: float) -> np.ndarray:
        """
        计算 γ 衰减权重 (公式 3)，最后归一化为 1。
        """
        if length <= 0:
            raise ValueError("History must contain at least one item.")
        gamma = float(np.clip(gamma, 0.0, 1.0))
        indices = np.arange(length, dtype=np.float32)
        # history is chronological -> w_k = gamma^(n - k) with k starting at 1
        raw = gamma ** (length - indices - 1)
        total = raw.sum()
        if total == 0:
            # occurs if gamma = 0 and length > 1; fall back to giving all mass
            # to the most recent item.
            raw = np.zeros_like(raw)
            raw[-1] = 1.0
            total = 1.0
        return raw / total

    def recommend(
        self,
        history: Sequence[str],
        scores: Sequence[float],
        *,
        top_k: int = 5,
        gamma: float = 0.5,
        tau: float = 0.2,
        sigma: float = 0.1,
        candidate_pool: Iterable[str] | None = None,
    ) -> List[Recommendation]:
        """
        推荐 top_k 个不在历史中的案例。

        history: 按时间顺序排列的历史案例 ID。
        scores: 与 history 对齐的得分（0~5）。
        candidate_pool: 待选案例子集；为 None 时默认为全量案例。
        """
        if len(history) != len(scores):
            raise ValueError("history and scores lengths must match.")
        if not history:
            raise ValueError("history must not be empty.")

        hist_indices = []
        for cid in history:
            if cid not in self.id_to_idx:
                raise KeyError(f"Unknown case id in history: {cid}")
            hist_indices.append(self.id_to_idx[cid])

        weights = self._normalized_importance_weights(len(history), gamma)
        targets = [self._sim_target(score, tau) for score in scores]
        history_set = set(history)

        if candidate_pool is None:
            candidate_iter = self.case_ids
        else:
            candidate_iter = list(candidate_pool)

        recommendations: List[Recommendation] = []
        for candidate_id in candidate_iter:
            if candidate_id in history_set:
                continue
            idx = self.id_to_idx.get(candidate_id)
            if idx is None:
                continue

            partials = []
            for hist_idx, target in zip(hist_indices, targets):
                sim_val = self.sim_matrix[idx, hist_idx]
                partials.append(self._partial_utility(sim_val, target, sigma))
            utility = float(np.dot(weights, partials))
            recommendations.append(Recommendation(candidate_id, utility))

        recommendations.sort(key=lambda rec: rec.utility, reverse=True)
        return recommendations[:top_k]


def load_case_ids(case_path: Path = Path("data/case.json")) -> List[str]:
    """加载并返回排序后的案例 ID 列表，与相似度矩阵保持同序。"""
    with case_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    return sorted(cases.keys())


# # ------------------------- Demo helpers -------------------------

# def run_toy_demo() -> None:
#     """
#     手写 4×4 相似度矩阵的示例，包含两个不同的历史序列。
#     """
#     case_ids = ["case_A", "case_B", "case_C", "case_D"]
#     sim_matrix = np.array(
#         [
#             [1.0, 0.9, 0.2, 0.1],
#             [0.9, 1.0, 0.3, 0.2],
#             [0.2, 0.3, 1.0, 0.8],
#             [0.1, 0.2, 0.8, 1.0],
#         ],
#         dtype=np.float32,
#     )
#     recommender = SequentialAdaptiveRecommender(sim_matrix, case_ids)

#     history = ["case_A", "case_B"]
#     scores = [4.5, 2.0]
#     recs = recommender.recommend(history, scores, top_k=5)
#     print("Toy demo #1 recommendations:", recs)

#     history = ["case_C"]
#     scores = [1.0]
#     recs = recommender.recommend(history, scores, top_k=5)
#     print("Toy demo #2 recommendations:", recs)


# def run_real_demo() -> None:
#     """
#     若 data/case_similarity_matrix.npy 存在，就用真实 500×500 矩阵跑一次示例。
#     """
#     matrix_path = Path("data/case_similarity_matrix.npy")
#     case_path = Path("data/case.json")

#     if not (matrix_path.exists() and case_path.exists()):
#         print("Skipping real-case demo (matrix or case.json missing).")
#         return

#     sim_matrix = np.load(matrix_path)
#     case_ids = load_case_ids(case_path)

#     recommender = SequentialAdaptiveRecommender(sim_matrix, case_ids)

#     sample_history = case_ids[:3]  # 简单地取前 3 个案例当做历史
#     sample_scores = [4.0, 3.5, 2.0]
#     recs = recommender.recommend(sample_history, sample_scores, top_k=5)
#     print("Real-case demo recommendations:")
#     for rec in recs:
#         print(f"  {rec.case_id}: utility={rec.utility:.4f}")


# if __name__ == "__main__":
#     run_toy_demo()
#     run_real_demo()
