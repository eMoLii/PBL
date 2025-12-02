"""
PBL 多agent（LangGraph）— 多场景/多触发点 & 场景级能力矩阵版
----------------------------------------------------------------
新增能力：
1) 多场景：每个场景有独立的病例段落 + 学习目标组；按场景顺序运行。
2) 学生能力矩阵：ability_mean_matrix[scene][objective]，逐场景为学生采样能力。
3) 记忆清空开关：clear_memory_between_scenes=True 时，每个场景结束后重置 messages/last_spoke 等。
4) 其余保持：<OBJ: 目标> 细粒度调度、按目标能力参与发言打分、教师评估/总结流程。

注：本文件在你现有的 agent 结构基础上扩展（参考基线实现的打分/能力/路由细节）。
"""

from __future__ import annotations
import os, json, math, random, re
from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import queue
import time

from openai import OpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage


# ------------------------- 载入配置与提示词 -------------------------

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
CASE_PATH = BASE_DIR / "data" / "case.json"
PROMPTS_PATH = BASE_DIR / "prompts.json"

def _init_activeness(cfg: Dict[str, Any], students: List[str]) -> Dict[str, float]:
    """
    Initialize activeness values with gentle diversity. When not explicitly provided,
    the roster spans from a higher-engagement student down to a quieter peer.
    """
    raw = cfg.get("activeness")
    default_val = float(cfg.get("activeness_default", 0.5))

    if isinstance(raw, dict):
        # Respect per-student overrides and allow a "default" fallback.
        fallback = float(raw.get("default", default_val))
        return {student: float(raw.get(student, fallback)) for student in students}

    if isinstance(raw, list):
        return {
            student: float(raw[i]) if i < len(raw) else default_val
            for i, student in enumerate(students)
        }

    if not students:
        return {}

    high = float(cfg.get("activeness_high", 0.8))
    low = float(cfg.get("activeness_low", 0.4))
    high = max(0.0, min(1.0, high))
    low = max(0.0, min(1.0, low))

    if len(students) == 1:
        return {students[0]: max(low, min(high, default_val))}

    step = (high - low) / max(len(students) - 1, 1)
    distribution = {
        student: round(high - step * idx, 3)
        for idx, student in enumerate(students)
    }
    return {student: max(0.0, min(1.0, val)) for student, val in distribution.items()}

def _prepare_students(cfg: Dict[str, Any]) -> List[str]:
    """Ensure cfg contains a student roster generated from configurable count."""
    if "students_count" in cfg:
        count = int(cfg["students_count"])
        if count <= 0:
            raise ValueError("students_count must be a positive integer.")
        students = [f"Student{i}" for i in range(1, count + 1)]
        cfg["students"] = students
    elif "students" in cfg:
        students = [str(name) for name in cfg["students"]]
    else:
        raise KeyError("Configuration must include 'students_count' or 'students'.")

    cfg["activeness"] = _init_activeness(cfg, students)
    return students

def load_case(cfg: Dict[str, Any], case_path: str = CASE_PATH) -> Dict[str, Any]:
    """Read case description from case.json using a configurable case_id."""
    case_id = cfg.get("case_id")
    if not case_id:
        raise KeyError("Configuration must include 'case_id' to select a scenario.")

    with open(case_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if case_id not in cases:
        raise KeyError(f"Case '{case_id}' not found in {case_path}.")

    case_raw = cases[case_id]
    scenes = case_raw.get("contextualized_case") or []
    if not isinstance(scenes, list) or len(scenes) == 0:
        raise ValueError(f"Case '{case_id}' is missing 'contextualized_case' scenes.")

    objectives = case_raw.get("objectives_by_scene")
    if objectives is None:
        # Older datasets used a misspelled key; keep compatibility.
        objectives = case_raw.get("objextives")
    if not isinstance(objectives, list) or len(objectives) == 0:
        raise ValueError(f"Case '{case_id}' is missing objective sets.")

    case = {
        "case_id": case_id,
        "contextualized_case": scenes,
        "objectives_by_scene": objectives,
    }
    if "original_case" in case_raw:
        case["original_case"] = case_raw["original_case"]
    return case

def init_ability_mean_matrix(cfg: Dict[str, Any], objectives_all: List[Dict[str, Any]]) -> List[List[float]]:
    """
    Ensure cfg holds an ability_mean_matrix; generate random means when absent.
    """
    existing = cfg.get("ability_mean_matrix")
    if isinstance(existing, list) and all(isinstance(row, list) for row in existing):
        return existing

    low = float(cfg.get("ability_mean_low", 1.8))
    high = float(cfg.get("ability_mean_high", 3.2))
    if low > high:
        low, high = high, low

    matrix: List[List[float]] = []
    for obj_scene in objectives_all:
        num_obj = len(obj_scene)
        if num_obj == 0:
            matrix.append([])
            continue
        # Sample each objective's mean ability within the configured band.
        means = [round(random.uniform(low, high), 3) for _ in range(num_obj)]
        matrix.append(means)

    cfg["ability_mean_matrix"] = matrix
    return matrix


# ------------------------- 工具函数 -------------------------

def format_objectives_text(objectives: Dict[str, Dict[str, str]]) -> str:
    lines = []
    for obj, items in objectives.items():
        lines.append(f"- {obj}")
        for k, v in items.items():
            lines.append(f"  * {k}：{v}")
    return "\n".join(lines)

def ability_value_to_text(val: float, prompts: Dict[str, Any]) -> str:
    """将 0~5 的能力值映射到五点制中文描述（临床推理成熟度）。"""
    levels = prompts.get("ability_levels", {})
    if val < 1.5:
        return levels.get("very_low", "几乎完全不了解")
    elif val < 2.5:
        return levels.get("low", "理解非常薄弱")
    elif val < 3.5:
        return levels.get("medium", "基础一般")
    elif val < 4.5:
        return levels.get("good", "基本掌握")
    else:
        return levels.get("excellent", "非常熟练")

def build_ability_profile_text(desc_for_student: Dict[str, str]) -> str:
    return "\n".join([f"{obj}：{desc}" for obj, desc in desc_for_student.items()])

def sample_student_abilities_for_scene(
    objectives_in_scene: Dict[str, Dict[str, str]],
    mean_vector_for_scene: List[float],
    sigma: float | List[float],
    students: List[str],
    prompts: Dict[str, Any]
):
    """
    针对“单个场景”的学习目标，按 ability_mean_matrix[scene] 的均值为每个学生采样能力。
    """
    obj_keys = list(objectives_in_scene.keys())
    assert len(obj_keys) == len(mean_vector_for_scene), (
        f"ability_mean_matrix 的该场景维度({len(mean_vector_for_scene)})与目标数({len(obj_keys)})不一致"
    )
    # sigma 可为标量或逐目标列表
    if isinstance(sigma, list):
        assert len(sigma) == len(obj_keys), "ability_sigma 若为列表，长度需与该场景目标数一致"
        sigmas = sigma
    else:
        sigmas = [float(sigma)] * len(obj_keys)

    abilities_raw: Dict[str, Dict[str, float]] = {}
    abilities_desc: Dict[str, Dict[str, str]] = {}
    competence_avg: Dict[str, float] = {}

    for stu in students:
        vals, descs, per = {}, {}, []
        for i, obj in enumerate(obj_keys):
            val = random.gauss(mean_vector_for_scene[i], sigmas[i])
            val = max(0.0, min(5.0, val))
            vals[obj] = val
            descs[obj] = ability_value_to_text(val, prompts)
            per.append(val)
        abilities_raw[stu] = vals
        abilities_desc[stu] = descs
        competence_avg[stu] = sum(per) / len(per) if per else 0.0
    return abilities_raw, abilities_desc, competence_avg


# ------------------------- 知识图谱 RAG -------------------------

class KnowledgeGraphRAG:
    DISPLAY_FIELDS = [
        ("desc", "概述"),
        ("symptom", "常见症状"),
        ("cause", "可能病因"),
        ("prevent", "预防"),
        ("cure_way", "治疗方式"),
        ("cure_lasttime", "治疗周期"),
        ("cured_prob", "治愈率"),
        ("easy_get", "易感人群"),
        ("get_prob", "患病概率"),
    ]

    TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+")

    @staticmethod
    def _safe_load_json_list(path: str) -> List[Any]:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("[") and not content.endswith("]"):
                content = content + "]"
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return []

    def __init__(
        self,
        entities_path: str,
        relations_path: str,
        default_top_k: int = 3,
        max_relation_items: int = 4,
    ) -> None:
        self.entities_path = entities_path
        self.relations_path = relations_path
        self.default_top_k = max(1, int(default_top_k or 1))
        self.max_relation_items = max(1, int(max_relation_items or 1))
        self.entities = self._load_entities()
        self.relation_map = self._load_relations()

    def _load_entities(self) -> List[Dict[str, Any]]:
        raw_entities = self._safe_load_json_list(self.entities_path)
        if not raw_entities:
            return []

        entities = []
        for item in raw_entities or []:
            payload = item.get("name", item)
            if not isinstance(payload, dict):
                payload = {"name": payload}

            name = self._extract_name(payload)
            summary_lines = self._build_summary(payload)
            flat_text = self._flatten_text(payload)
            entities.append({
                "label": item.get("label", "Entity"),
                "name": name or "",
                "summary": summary_lines,
                "search_blob": flat_text,
            })
        return entities

    def _load_relations(self) -> Dict[str, List[str]]:
        rel_map: Dict[str, List[str]] = defaultdict(list)
        raw_rels = self._safe_load_json_list(self.relations_path)
        if not raw_rels:
            return rel_map

        for rel_block in raw_rels or []:
            rel_label = rel_block.get("rel_name") or rel_block.get("rel_type") or "关联"
            end_type = rel_block.get("end_entity_type", "Entity")
            for rel in rel_block.get("rels", []):
                start = rel.get("start_entity_name")
                end = rel.get("end_entity_name")
                if not start or not end:
                    continue
                rel_text = f"{rel_label} -> [{end_type}] {end}"
                rel_map[start].append(rel_text)
        return rel_map

    @staticmethod
    def _extract_name(payload: Dict[str, Any]) -> str:
        for key in ("name", "Name", "title", "Title"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    @classmethod
    def _build_summary(cls, payload: Dict[str, Any]) -> List[str]:
        lines = []
        for key, label in cls.DISPLAY_FIELDS:
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                cleaned = re.sub(r"\s+", " ", val.strip())
                lines.append(f"{label}：{cleaned}")
        # 回退：若关键字段为空，则提供裁剪后的全部文本
        if not lines:
            text = re.sub(r"\s+", " ", cls._flatten_text(payload)).strip()
            if text:
                lines.append(text[:160])
        return lines

    @classmethod
    def _flatten_text(cls, data: Any) -> str:
        if isinstance(data, dict):
            return " ".join([cls._flatten_text(v) for v in data.values()])
        if isinstance(data, list):
            return " ".join([cls._flatten_text(v) for v in data])
        if data is None:
            return ""
        return str(data)

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        tokens: List[str] = []
        for tok in self.TOKEN_PATTERN.findall(text):
            chunk = tok.strip()
            if not chunk:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
                parts = re.split(r"[的和及与、，。/\s]+", chunk)
                tokens.extend([p for p in parts if p])
            else:
                tokens.append(chunk)
        # 去重但保持顺序
        seen = set()
        ordered = []
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            ordered.append(tok)
        return ordered

    @staticmethod
    def _score_text(text: str, keywords: List[str]) -> float:
        if not text:
            return 0.0
        lowered = text.lower()
        score = 0.0
        for kw in keywords:
            key = kw.lower()
            if not key:
                continue
            if key in lowered:
                score += max(1.0, len(key) * 0.3)
        return score

    def search(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if not query:
            return []
        keywords = self._tokenize(query)
        if not keywords:
            return []
        seen = set()
        scored: List[tuple[float, Dict[str, Any]]] = []
        for entity in self.entities:
            blob_score = self._score_text(entity.get("search_blob", ""), keywords)
            name = entity.get("name", "")
            if name:
                blob_score += 2.0 * self._score_text(name, keywords)
            if blob_score <= 0:
                continue
            if name in seen:
                continue
            seen.add(name)
            scored.append((blob_score, entity))

        scored.sort(key=lambda item: item[0], reverse=True)
        limit = top_k or self.default_top_k
        results = []
        for score, entity in scored[:limit]:
            relations = self.relation_map.get(entity.get("name", ""), [])
            results.append({
                "name": entity.get("name", "未知实体"),
                "label": entity.get("label", "Entity"),
                "summary": entity.get("summary", [])[:3],
                "relations": relations[: self.max_relation_items],
                "score": score,
            })
        return results

    def render_matches(self, matches: List[Dict[str, Any]], start_index: int = 1) -> str:
        if not matches:
            return ""
        lines: List[str] = []
        for offset, match in enumerate(matches):
            header = f"{start_index + offset}. {match['name']}（{match['label']}）"
            lines.append(header)
            for snippet in match.get("summary", []) or []:
                lines.append(f"   - {snippet}")
            for rel in match.get("relations", []) or []:
                lines.append(f"   - 关联：{rel}")
        return "\n".join(lines)

    def build_context(self, query: str, top_k: int | None = None) -> str:
        matches = self.search(query, top_k=top_k)
        return self.render_matches(matches)

    def keywords_from_text(self, text: str) -> List[str]:
        """Expose tokenizer for external consumers (e.g., evaluator)."""
        return self._tokenize(text)


# ------------------------- 评估辅助 -------------------------

def discretize_score(value: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = 0.0
    val = max(0.0, min(5.0, val))
    return round(val * 2) / 2.0


def compose_case_summary(case: Dict[str, Any]) -> str:
    scenes = case.get("contextualized_case", [])
    objectives = case.get("objectives_by_scene", [])
    lines: List[str] = ["病例场景："]
    for idx, text in enumerate(scenes, 1):
        lines.append(f"场景{idx}：{text}")
    lines.append("学习目标：")
    for idx, obj_scene in enumerate(objectives, 1):
        lines.append(f"场景{idx} 目标：")
        for key, detail in obj_scene.items():
            if isinstance(detail, dict):
                detail_text = "; ".join([f"{k}:{v}" for k, v in detail.items()])
            else:
                detail_text = str(detail)
            lines.append(f"- {key}：{detail_text}")
    return "\n".join(lines)


class EvaluationAgent:
    def __init__(
        self,
        llm: QwenLLM,
        knowledge: KnowledgeGraphRAG | None,
        case_summary: str,
        max_context_chars: int = 1600,
        prompt_bundle: Dict[str, str] | None = None,
        activeness_sigmoid_a: float = 3.0,
        dimension_defs: Sequence[Dict[str, Any]] | None = None,
    ):
        self.llm = llm
        self.knowledge = knowledge
        self.case_summary = case_summary
        self.max_context_chars = max(200, max_context_chars)
        prompt_bundle = prompt_bundle or {}
        system_prompt = prompt_bundle.get("system")
        user_prompt = prompt_bundle.get("user")
        if not system_prompt or not user_prompt:
            raise ValueError("evaluation prompts 缺少 system 或 user 模板，请在 prompts.json 中配置。")
        self.system_prompt_template = system_prompt
        self.user_prompt_template = user_prompt
        self.activeness_sigmoid_a = float(activeness_sigmoid_a or 3.0)
        if not dimension_defs:
            raise ValueError("evaluation prompts 未提供维度定义，请在 prompts.json.evaluation.dimensions 中配置。")
        self.dimension_defs = self._normalize_dimension_defs(dimension_defs)

    @staticmethod
    def _normalize_dimension_defs(defs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in defs or []:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            title = item.get("title") or key
            if not key or not title:
                continue
            normalized.append({
                "key": str(key),
                "title": str(title),
                "use_rag": bool(item.get("use_rag", False)),
                "default_instruction": item.get("default_instruction", ""),
            })
        if not normalized:
            raise ValueError("未找到有效的维度定义，请检查 prompts.json.evaluation.dimensions。")
        return normalized

    def _format_transcripts(self, log: List[Dict[str, Any]], student: str) -> tuple[str, str]:
        full_lines: List[str] = []
        student_lines: List[str] = []
        for entry in log:
            line = (
                f"[Scene {entry.get('scene_index', 0)+1}]"
                f"[{entry.get('stage','discussion')}] {entry.get('speaker')}: {entry.get('content')}"
            )
            full_lines.append(line)
            if entry.get("speaker") == student:
                student_lines.append(line)
        return "\n".join(full_lines), "\n".join(student_lines)

    def _knowledge_context(
        self,
        dimension_title: str,
        student_transcript: str,
        full_transcript: str,
    ) -> str:
        if not self.knowledge:
            return "（知识图谱不可用，改用讨论记录与病例摘要进行评估）"
        context_seed = student_transcript or full_transcript or self.case_summary
        seed_tail = context_seed[-1500:]
        query = f"维度：{dimension_title}\n病例摘要：{self.case_summary}\n学生相关发言：{seed_tail}"
        raw_matches = self.knowledge.search(
            query,
            top_k=self.knowledge.default_top_k * 3,
        )
        if not raw_matches:
            return "（知识图谱未检索到结果）"

        keyword_source = f"{self.case_summary}\n{seed_tail}"
        case_tokens = set(self.knowledge.keywords_from_text(keyword_source))

        def relevance(match: Dict[str, Any]) -> tuple[int, float]:
            text = " ".join(match.get("summary", [])) + " " + match.get("name", "")
            tokens = set(self.knowledge.keywords_from_text(text))
            overlap = len(case_tokens & tokens)
            return overlap, float(match.get("score", 0.0))

        sorted_matches = sorted(
            raw_matches,
            key=lambda m: relevance(m),
            reverse=True,
        )
        top_matches = sorted_matches[: self.knowledge.default_top_k]
        if not top_matches:
            return "（知识图谱未检索到结果）"

        assembled_blocks: List[str] = []
        current_index = 1
        used_chars = 0
        for match in top_matches:
            block = self.knowledge.render_matches([match], start_index=current_index)
            block_len = len(block)
            # 始终保留首个匹配，即使超限
            if assembled_blocks and used_chars + block_len > self.max_context_chars:
                break
            assembled_blocks.append(block)
            used_chars += block_len + 1  # +1 for newline separator
            current_index += 1

        context = "\n".join(assembled_blocks)
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars] + "\n...（知识图谱内容已截断）"
        return context or "（知识图谱未检索到结果）"

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raw = self.llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format="json",
        )
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return {"score": 0, "justification": "评分失败：模型未返回合法JSON。"}

    def _evaluate_dimension(
        self,
        student: str,
        dimension: Dict[str, Any],
        full_transcript: str,
        student_transcript: str,
    ) -> Dict[str, Any]:
        title = dimension["title"]
        use_rag = dimension["use_rag"]
        instructions_raw = dimension.get("default_instruction", "")
        instructions_text = (
            "评估要点：\n" + instructions_raw.strip()
            if instructions_raw else "评估要点：请结合讨论记录自主评估。"
        )
        knowledge_block = (
            self._knowledge_context(title, student_transcript, full_transcript)
            if use_rag else "（本维度无需知识图谱）"
        )
        system_prompt = self.system_prompt_template
        user_prompt = self.user_prompt_template.format(
            student=student,
            title=title,
            instructions=instructions_text,
            knowledge_block=knowledge_block,
            student_transcript=student_transcript or "（无发言）",
            full_transcript=full_transcript or "（无记录）",
        )
        data = self._call_llm(system_prompt, user_prompt)
        score = discretize_score(data.get("score", 0))
        return {
            "title": title,
            "score": score,
            "justification": data.get("justification", ""),
        }

    def _compute_activeness(
        self,
        student: str,
        student_stats: Dict[str, Dict[str, float]],
        students: List[str],
    ) -> Dict[str, Any]:
        counts = [student_stats.get(s, {}).get("message_count", 0.0) for s in students]
        chars = [student_stats.get(s, {}).get("char_count", 0.0) for s in students]
        avg_count = sum(counts) / len(counts) if counts else 0.0
        avg_char = sum(chars) / len(chars) if chars else 0.0
        stu_count = student_stats.get(student, {}).get("message_count", 0.0)
        stu_char = student_stats.get(student, {}).get("char_count", 0.0)
        ratio_count = stu_count / avg_count if avg_count > 0 else 1.0
        ratio_char = stu_char / avg_char if avg_char > 0 else 1.0
        eps = 1e-6
        inv_sum = (1.0 / max(ratio_count, eps)) + (1.0 / max(ratio_char, eps))
        combined_ratio = 2.0 / inv_sum if inv_sum > 0 else 0.0

        a = self.activeness_sigmoid_a
        sigmoid = 1.0 / (1.0 + math.exp(-a * (combined_ratio - 1.0)))
        score_cont = 5.0 * sigmoid
        score = discretize_score(score_cont)
        justification = (
            f"发言{int(stu_count)}次（平均{avg_count:.1f}），字数约{int(stu_char)}（平均{avg_char:.1f}）")
        return {"title": "参与度", "score": score, "justification": justification}

    def evaluate(
        self,
        student: str,
        log: List[Dict[str, Any]],
        student_stats: Dict[str, Dict[str, float]],
        students: List[str],
    ) -> Dict[str, Any]:
        full_transcript, student_transcript = self._format_transcripts(log, student)
        results: Dict[str, Any] = {}
        for dimension in self.dimension_defs:
            results[dimension["key"]] = self._evaluate_dimension(
                student, dimension, full_transcript, student_transcript
            )
        results["activeness"] = self._compute_activeness(student, student_stats, students)
        return {
            "student": student,
            "dimensions": results,
            "stats": student_stats.get(student, {}),
        }


ADVISOR_SYSTEM_PROMPT_DEFAULT = (
    "你是一名医学教育顾问，负责在评估结束后向学生提供简洁、可执行的学习建议。"
    "结合评估结果和完整对话，先概括整体表现，再给出重点资源与短小精悍的改进行动。"
    "避免冗长段落，突出最具价值的提醒。"
)

class AdvisorAgent:
    """根据评估结果 + 完整对话，输出整体与细化建议。"""

    def __init__(
        self,
        llm: QwenLLM,
        *,
        prompt_bundle: Dict[str, str] | None = None,
        max_transcript_chars: int = 10000,
    ):
        self.llm = llm
        bundle = prompt_bundle or {}
        self.system_prompt = bundle.get("system", ADVISOR_SYSTEM_PROMPT_DEFAULT)
        user_prompt = bundle.get("user")
        if not user_prompt:
            raise ValueError("advisor 提示词缺少 user 模板，请在 prompts.json 中配置。")
        self.user_prompt_template = user_prompt
        self.max_transcript_chars = max(int(max_transcript_chars or 0), 0)

    def _build_transcript(self, log: List[Dict[str, Any]]) -> str:
        if not log:
            return "（无讨论记录）"
        lines = []
        for entry in log:
            scene_idx = entry.get("scene_index")
            scene_text = f"场景{int(scene_idx)+1}" if isinstance(scene_idx, int) else "场景?"
            stage = entry.get("stage", "")
            speaker = entry.get("speaker", "Unknown")
            content = entry.get("content", "")
            lines.append(f"[{scene_text}][{stage}] {speaker}: {content}")
        transcript = "\n".join(lines)
        if self.max_transcript_chars and len(transcript) > self.max_transcript_chars:
            return (
                transcript[-self.max_transcript_chars :]
                + "\n...（对话已截断，仅保留最近的片段）"
            )
        return transcript

    @staticmethod
    def _score_overview(evaluation_result: Dict[str, Any]) -> str:
        dims = evaluation_result.get("dimensions") or {}
        parts = []
        for key, payload in dims.items():
            if not isinstance(payload, dict):
                continue
            title = payload.get("title") or key
            score = payload.get("score")
            if score is None:
                continue
            parts.append(f"{title}:{score}")
        return "，".join(parts) if parts else "（无维度得分）"

    @staticmethod
    def _weak_points_summary(evaluation_result: Dict[str, Any], keep: int = 3) -> str:
        dims = evaluation_result.get("dimensions") or {}
        scored: List[tuple[float, str, Dict[str, Any]]] = []
        for key, payload in dims.items():
            if not isinstance(payload, dict):
                continue
            try:
                score = float(payload.get("score"))
            except (TypeError, ValueError):
                continue
            scored.append((score, key, payload))
        if not scored:
            return "（无可用得分）"
        scored.sort(key=lambda item: item[0])
        lines = []
        for score, key, payload in scored[:keep]:
            title = payload.get("title") or key
            justification = payload.get("justification", "")
            lines.append(f"{title}：得分{score}，{justification}".strip())
        return "\n".join(lines)

    def advise(
        self,
        student: str,
        evaluation_result: Dict[str, Any],
        log: List[Dict[str, Any]],
        test_scores: Dict[str, Any] | None = None,
        test_report: str | None = None,
    ) -> Dict[str, Any]:
        transcript = self._build_transcript(log)
        pre_score = None
        post_score = None
        if test_scores:
            pre_score = test_scores.get("pre_score")
            post_score = test_scores.get("post_score")
        def _fmt(score: Any) -> str:
            if score is None:
                return "未提供"
            try:
                return f"{float(score):.1f}"
            except (TypeError, ValueError):
                return str(score)
        pre_text = _fmt(pre_score)
        post_text = _fmt(post_score)
        delta_text = "未提供"
        if pre_score is not None and post_score is not None:
            try:
                delta_text = f"{float(post_score) - float(pre_score):+.1f}"
            except (TypeError, ValueError):
                delta_text = "未提供"
        report_text = test_report or "（未提供测验题目表现）"
        print(
            "[AdvisorAgent] test_report:\n"
            f"{report_text}",
            flush=True,
        )
        user_prompt = self.user_prompt_template.format(
            student=student,
            score_overview=self._score_overview(evaluation_result),
            weak_points=self._weak_points_summary(evaluation_result),
            evaluation_json=json.dumps(evaluation_result, ensure_ascii=False, indent=2),
            transcript=transcript,
            pre_score_text=pre_text,
            post_score_text=post_text,
            score_delta_text=delta_text,
            test_report=report_text,
        )
        print(
            "[AdvisorAgent] user_prompt:\n"
            f"{user_prompt}",
            flush=True,
        )
        raw = self.llm.invoke(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format="json",
        )
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        fallback_text = raw.strip() if isinstance(raw, str) else ""
        return {
            "general_advice": {
                "summary": fallback_text or "生成建议失败，请依据评估结果自行制定学习计划。",
                "recommended_resources": [],
            },
            "detailed_advice": [],
        }


# ------------------------- 状态类型 -------------------------

class PBLState(TypedDict):
    # LangGraph 消息
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    stage: str                           # 'scenario' | 'discussion' | 'summary' | 'end'
    round_idx: int
    last_spoke: Dict[str, int]

    # 配置/提示词
    cfg: Dict[str, Any]
    prompts: Dict[str, Any]

    # 场景级数据（本场景）
    scene_index: int
    scene_total: int
    case_text: str
    objectives: Dict[str, Dict[str, str]]

    # 能力（本场景）
    abilities_raw: Dict[str, Dict[str, float]]     # [student][objective] -> 0..5
    abilities_desc: Dict[str, Dict[str, str]]
    competence_avg: Dict[str, float]               # 平均能力（备用）

    # 当前讨论目标
    current_objective: str | None
    queued_next: str | None


# ------------------------- LLM 包装 -------------------------

class QwenLLM:
    def __init__(self, model: str, api_key: str, base_url: str, temperature: float = 0.7):
        # 兼容阿里 Qwen 的 OpenAI兼容模式
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def invoke(self, messages: List[Dict[str, str]], response_format: str = "text") -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=False,
            response_format=({"type": "json_object"} if response_format == "json" else None),
            frequency_penalty=0.05,
            max_tokens=2048,
            extra_body={"enable_thinking": False},
        )
        return resp.choices[0].message.content


# ------------------------- 调度打分（细粒度：按目标能力） -------------------------

def recency_decay(k: float, tau_d: float) -> float:
    if k == float("inf"):
        return 0.0
    return math.exp(-k / max(tau_d, 1e-6))

def gumbel() -> float:
    u = random.random()
    return -math.log(-math.log(u + 1e-12) + 1e-12)

def _competence_for_obj(state: PBLState, student: str) -> float:
    """返回该学生在当前目标上的能力(0~1)。若未设目标，则退化为平均能力。"""
    cur = state["current_objective"]
    if cur and (student in state["abilities_raw"]) and (cur in state["abilities_raw"][student]):
        return state["abilities_raw"][student][cur] / 5.0
    return state["competence_avg"].get(student, 0.0)

def score_student(name: str, t: int, state: PBLState) -> float:
    cfg = state["cfg"]
    alpha = cfg.get("alpha", 2.0)      # activeness 权重
    beta  = cfg.get("beta", 2.0)       # 能力权重（细粒度）
    gamma = cfg.get("gamma", 3.0)      # 冷却惩罚权重
    delta = cfg.get("delta", 1.0)      # 随机扰动权重
    tau_d = cfg.get("tau_d", 1.0)

    a = float(cfg.get("activeness", {}).get(name, 0.5))
    c = float(_competence_for_obj(state, name))  # 0~1

    t_last = state["last_spoke"].get(name, -10**9)
    k = (t - t_last) if t_last > -10**9 else float("inf")
    r = recency_decay(k, tau_d)

    return alpha*a + beta*c - gamma*r + delta*gumbel()

def softmax_pick(scores: Dict[str, float], tau: float) -> str:
    keys = list(scores.keys())
    vals = [scores[k] / max(tau, 1e-6) for k in keys]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    Z = sum(exps) + 1e-12
    probs = [e/Z for e in exps]
    r = random.random(); s = 0.0
    for k, p in zip(keys, probs):
        s += p
        if r <= s:
            return k
    return keys[-1]

def pick_next_student(students: List[str], state: PBLState) -> str:
    t = state["round_idx"]
    tau = state["cfg"].get("tau", 1.0)
    scores = {s: score_student(s, t, state) for s in students}
    return softmax_pick(scores, tau)


# ------------------------- 学生代理 -------------------------

class StudentAgent:
    def __init__(
        self,
        name: str,
        llm: QwenLLM,
        user_queue: "queue.Queue[str]" | None = None,
        turn_event: threading.Event | None = None,
    ):
        self.name = name
        self.llm = llm
        self.user_queue = user_queue
        self.turn_event = turn_event

    def _student_sys_prompt(
        self, stage: str, prompts: Dict[str, Any],
        objectives: Dict[str, Dict[str, str]] | None,
        abilities_desc: Dict[str, Dict[str, str]] | None
    ) -> str:
        tmpl = prompts["student"][stage]
        objectives_text = format_objectives_text(objectives) if objectives else ""
        ability_text = build_ability_profile_text(abilities_desc.get(self.name, {})) if abilities_desc else ""
        return tmpl.format(
            name=self.name,
            objectives_text=objectives_text,
            ability_profile_text=ability_text
        )

    def __call__(self, state: PBLState) -> Dict[str, Any]:
        stage = state["stage"]
        if stage not in ("discussion", "summary"):
            return {"next": "teacher"}
        waiting_cb = state["cfg"].get("user_waiting_callback")
        turn_event = state["cfg"].get("user_turn_event")
        start_cb = state["cfg"].get("agent_turn_start_callback")
        end_cb = state["cfg"].get("agent_turn_end_callback")
        is_user_agent = self.user_queue is not None
        if not is_user_agent and start_cb:
            start_cb(self.name)
        try:
            if self.user_queue is None and turn_event is not None:
                while turn_event.is_set():
                    time.sleep(0.2)

            if self.user_queue:
                if turn_event is None or not turn_event.is_set():
                    return {"next": "teacher"}
                if waiting_cb:
                    waiting_cb(True)
                while True:
                    try:
                        user_content = self.user_queue.get(timeout=0.2)
                        break
                    except queue.Empty:
                        if turn_event is None or not turn_event.is_set():
                            if waiting_cb:
                                waiting_cb(False)
                            return {"next": "teacher"}
                if waiting_cb:
                    waiting_cb(False)
                turn_event.clear()
                msg = HumanMessage(content=user_content, name=self.name)
                return {"messages": [msg], "next": "teacher"}

            sys_prompt = self._student_sys_prompt(
                stage=stage,
                prompts=state["prompts"],
                objectives=state["objectives"] if stage == "discussion" else None,
                abilities_desc=state["abilities_desc"] if stage == "discussion" else None,
            )
            chat = [{"role": "system", "content": sys_prompt}]
            for m in state["messages"]:
                speaker = getattr(m, "name", None)
                content = getattr(m, "content", "")
                if speaker == self.name:
                    chat.append({"role": "assistant", "content": content})
                else:
                    tagged_content = f"{speaker}: {content}" if speaker else content
                    chat.append({"role": "user", "content": tagged_content})

            reply = self.llm.invoke(chat, response_format="text")
            if self.user_queue is None and turn_event is not None and turn_event.is_set():
                return {"next": "teacher"}
            msg = HumanMessage(content=f"{reply}", name=self.name)
            return {"messages": [msg], "next": "teacher"}
        finally:
            if not is_user_agent and end_cb:
                end_cb(self.name)


class UserPauseNode:
    def __call__(self, state: PBLState) -> Dict[str, Any]:
        cfg = state["cfg"]
        fallback = cfg.get("user_student_name", "Student1")
        candidate = state.get("queued_next") or fallback
        turn_event = cfg.get("user_turn_event")
        grace = float(cfg.get("user_turn_grace_period", cfg.get("interval", 0.0)))
        next_target = candidate
        if candidate != fallback and turn_event is not None:
            if turn_event.is_set():
                next_target = fallback
            elif grace > 0 and turn_event.wait(grace):
                next_target = fallback
        elif candidate == fallback and turn_event is not None and turn_event.is_set():
            next_target = fallback
        return {
            "next": next_target,
            "stage": state["stage"],
            "round_idx": state["round_idx"],
            "last_spoke": state["last_spoke"],
            "current_objective": state["current_objective"],
            "queued_next": None,
        }


# ------------------------- 教师代理 -------------------------

OBJ_TAG_RE = re.compile(r"<\s*OBJ\s*:\s*(.+?)\s*>")

class TeacherAgent:
    def __init__(self, llm: QwenLLM, students: List[str], knowledge: KnowledgeGraphRAG | None = None):
        self.llm = llm
        self.students = students
        self.knowledge = knowledge
        self.pause_node_name = "user_pause"

    def _wrap_with_pause(
        self,
        state: PBLState,
        payload: Dict[str, Any],
        candidate: str,
        fallback: str,
        *,
        force_user_event: bool = False,
    ) -> Dict[str, Any]:
        cfg = state["cfg"]
        turn_event = cfg.get("user_turn_event")
        waiting_cb = cfg.get("user_waiting_callback")

        if candidate == fallback:
            payload["next"] = fallback
            payload["queued_next"] = None
            if force_user_event and turn_event is not None:
                turn_event.set()
            if force_user_event and waiting_cb:
                waiting_cb(True)
            return payload
        payload["next"] = self.pause_node_name
        payload["queued_next"] = candidate
        return payload

    def _present_scenario(
        self,
        prompts_teacher,
        case_text: str,
        objectives,
        scene_number: int,
        scene_total: int,
    ) -> str:
        system_prompt = (prompts_teacher or {}).get("present_scenario_system")
        user_template = (prompts_teacher or {}).get("present_scenario_user")
        if not (system_prompt and user_template):
            return (
                f"场景{scene_number}/{scene_total}：请同学们结合病例资料，自主阐述关键线索，"
                f"并围绕学习目标开展讨论。"
            )
        sys_msg = {"role": "system", "content": system_prompt}
        user_msg = {
            "role": "user",
            "content": user_template.format(
                scene_number=scene_number,
                scene_total=scene_total,
                scenario_text=case_text,
                objectives_text=format_objectives_text(objectives),
            ),
        }
        return self.llm.invoke([sys_msg, user_msg], response_format="text")

    @staticmethod
    def _extract_objective_tag(text: str, valid_keys: List[str]) -> str | None:
        m = OBJ_TAG_RE.search(text or "")
        if not m: return None
        raw = m.group(1).strip()
        if raw in valid_keys: return raw
        for k in valid_keys:
            if raw in k or k in raw:
                return k
        return None

    def _gather(self, messages: List[BaseMessage]) -> str:
        return "\n".join([f"{getattr(m,'name',m.type)}: {m.content}" for m in messages])

    def _eval_discussion(self, prompts_teacher, messages, objectives):
        system_prompt = (prompts_teacher or {}).get("evaluate_discussion_system")
        user_template = (prompts_teacher or {}).get("evaluate_discussion_user")
        if not (system_prompt and user_template):
            return {
                "decision": "continue",
                "feedback": "提示：请继续围绕学习目标展开讨论。",
            }
        sys_msg = {"role": "system", "content": system_prompt}
        user_msg = {
            "role": "user",
            "content": user_template.format(
                conversation=self._gather(messages),
                objectives_text=format_objectives_text(objectives),
            ),
        }
        raw = self.llm.invoke([sys_msg, user_msg], response_format="json")
        try:
            data = json.loads(raw)
            if "decision" not in data: raise ValueError
            return data
        except Exception:
            return {"decision":"continue", "feedback":"评估异常，请继续深入该目标的讨论。"}

    def _eval_summary(self, prompts_teacher, messages):
        system_prompt = (prompts_teacher or {}).get("evaluate_summary_system")
        user_template = (prompts_teacher or {}).get("evaluate_summary_user")
        if not (system_prompt and user_template):
            return {
                "decision": "continue",
                "feedback": "提示：请补充总结要点并说明反思。",
                "has_question": False,
                "question": "",
            }
        sys_msg = {"role": "system", "content": system_prompt}
        user_msg = {
            "role": "user",
            "content": user_template.format(conversation=self._gather(messages)),
        }
        raw = self.llm.invoke([sys_msg, user_msg], response_format="json")
        try:
            data = json.loads(raw)
            if "decision" not in data: raise ValueError
            return data
        except Exception:
            return {"decision":"continue", "feedback":"评估异常，请继续完善总结。"}

    def _answer_summary_question(self, prompts_teacher, question: str, feedback: str, messages) -> str:
        if not question.strip():
            return ""
        system_prompt = (prompts_teacher or {}).get("answer_summary_question_system")
        user_template = (prompts_teacher or {}).get("answer_summary_question_user")
        if not (system_prompt and user_template):
            return ""
        knowledge_context = ""
        if self.knowledge:
            knowledge_context = self.knowledge.build_context(question, top_k=self.knowledge.default_top_k)
        if not knowledge_context:
            knowledge_context = "（知识图谱未检索到与该问题直接相关的条目，请结合已有讨论推理。）"
        sys_msg = {"role": "system", "content": system_prompt}
        user_msg = {"role": "user", "content": user_template.format(
            question=question,
            evaluation_feedback=feedback,
            conversation=self._gather(messages),
            knowledge_context=knowledge_context,
        )}
        return self.llm.invoke([sys_msg, user_msg], response_format="text")

    def __call__(self, state: PBLState) -> Dict[str, Any]:
        turn_event = state["cfg"].get("user_turn_event")
        students = state["cfg"]["students"]
        user_student_name = state["cfg"].get("user_student_name", "Student1")
        fallback_target = (
            user_student_name if user_student_name in students
            else (students[0] if students else "END")
        )
        if turn_event is not None and turn_event.is_set():
            time.sleep(0.2)
            return {"next": fallback_target, "queued_next": None}
        msgs = list(state["messages"])
        stage = state["stage"]
        prompts_all = state.get("prompts") or {}
        prompts_t = prompts_all.get("teacher")
        if not prompts_t:
            return {"next": fallback_target, "queued_next": None}
        summary_lead = students[0] if students else None
        last_spoke = dict(state["last_spoke"])
        round_idx = state["round_idx"]
        objectives = state["objectives"]
        obj_keys = list(objectives.keys())
        current_obj = state["current_objective"]
        scene_number = state["scene_index"] + 1
        scene_total = state.get("scene_total", scene_number)

        # --- 硬停止：每场景最大轮数 ---
        max_rounds = state["cfg"].get("max_rounds_per_scene", 60)
        if round_idx >= max_rounds and stage != "summary":
            stop_msg = HumanMessage(
                content="本场景已达到最大讨论轮数，请抓紧做一个整体总结后再请我点评。",
                name="Teacher"
            )
            pick_state = {**state, "current_objective": current_obj, "last_spoke": last_spoke}
            nxt = summary_lead or pick_next_student(students, pick_state)
            payload = {
                "messages": [stop_msg],
                "stage": "summary",
                "round_idx": round_idx + 1,
                "last_spoke": last_spoke,
                "current_objective": current_obj
            }
            return self._wrap_with_pause(state, payload, nxt, fallback_target)

        # 初次：呈现“本场景” -> 进入讨论 -> 抽第一位
        if len(msgs) == 0 and stage == "scenario":
            intro = self._present_scenario(
                prompts_teacher=prompts_t,
                case_text=state["case_text"],
                objectives=objectives,
                scene_number=scene_number,
                scene_total=scene_total,
            )
            tmsg = HumanMessage(content=intro, name="Teacher")
            first = pick_next_student(students, state)
            payload = {
                "messages":[tmsg], "stage":"discussion",
                "round_idx": round_idx+1,
                "last_spoke": last_spoke, "current_objective": current_obj
            }
            return self._wrap_with_pause(state, payload, first, fallback_target)

        # 若上一条是学生，更新 last_spoke
        last = msgs[-1] if msgs else None
        if last and hasattr(last, "name") and last.name in students:
            last_spoke[last.name] = round_idx

        # 解析学生是否声明/切换了目标
        if last:
            tag = self._extract_objective_tag(last.content, obj_keys)
            if tag:
                current_obj = tag  # 更新当前讨论目标

        asked_teacher = bool(last and "<call_teacher>" in (last.content or "").lower())

        if stage == "discussion":
            if asked_teacher:
                res = self._eval_discussion(prompts_t, msgs, objectives)
                decision_flag = str(res.get("decision") or "").strip().lower()
                if decision_flag != "finish":
                    tmsg = HumanMessage(content=res.get("feedback","继续。"), name="Teacher")
                    pick_state = {**state, "current_objective": current_obj, "last_spoke": last_spoke}
                    nxt = pick_next_student(students, pick_state)
                    payload = {
                        "messages":[tmsg], "stage":"discussion",
                        "round_idx": round_idx+1, "last_spoke": last_spoke, "current_objective": current_obj
                    }
                    return self._wrap_with_pause(state, payload, nxt, fallback_target)
                else:
                    tmsg = HumanMessage(content=res.get("feedback","进入总结。"), name="Teacher")
                    nxt = fallback_target
                    payload = {
                        "messages":[tmsg], "stage":"summary",
                        "round_idx": round_idx+1, "last_spoke": last_spoke, "current_objective": current_obj
                    }
                    return self._wrap_with_pause(state, payload, nxt, fallback_target, force_user_event=True)

            # 未请示老师 → 继续按当前目标调度
            pick_state = {**state, "current_objective": current_obj, "last_spoke": last_spoke}
            nxt = pick_next_student(students, pick_state)
            payload = {
                "round_idx": round_idx+1,
                "last_spoke": last_spoke, "current_objective": current_obj
            }
            return self._wrap_with_pause(state, payload, nxt, fallback_target)

        if stage == "summary":
            if asked_teacher:
                res = self._eval_summary(prompts_t, msgs)
                feedback_text = res.get("feedback","继续完善。")
                teacher_msgs: List[HumanMessage] = []
                has_question = str(res.get("has_question",""))
                has_question = has_question.lower() if isinstance(has_question, str) else str(has_question).lower()
                need_answer = has_question in {"true", "1", "yes"}
                question_text = res.get("question") or res.get("question_text") or ""
                if need_answer and question_text.strip():
                    answer_text = self._answer_summary_question(prompts_t, question_text, feedback_text, msgs)
                    if answer_text and answer_text.strip():
                        teacher_msgs.append(HumanMessage(content=answer_text, name="Teacher"))
                if not teacher_msgs:
                    teacher_msgs.append(HumanMessage(content=feedback_text, name="Teacher"))

                if need_answer and teacher_msgs:
                    return {
                        "messages": teacher_msgs,
                        "stage": "end",
                        "next": "END",
                        "round_idx": round_idx + 1,
                        "last_spoke": last_spoke,
                        "current_objective": current_obj,
                        "queued_next": None,
                    }

                decision_flag = str(res.get("decision") or "").strip().lower()
                if decision_flag == "finish":
                    return {
                        "messages":teacher_msgs, "stage":"end", "next":"END",
                        "round_idx": round_idx+1, "last_spoke": last_spoke, "current_objective": current_obj,
                        "queued_next": None,
                    }
                else:
                    pick_state = {**state, "current_objective": current_obj, "last_spoke": last_spoke}
                    nxt = summary_lead or pick_next_student(students, pick_state)
                    payload = {
                        "messages":teacher_msgs, "stage":"summary",
                        "round_idx": round_idx+1, "last_spoke": last_spoke, "current_objective": current_obj
                    }
                    return self._wrap_with_pause(state, payload, nxt, fallback_target)

            # 未请老师 → 继续总结
            pick_state = {**state, "current_objective": current_obj, "last_spoke": last_spoke}
            nxt = summary_lead or pick_next_student(students, pick_state)
            payload = {
                "round_idx": round_idx+1,
                "last_spoke": last_spoke, "current_objective": current_obj
            }
            return self._wrap_with_pause(state, payload, nxt, fallback_target)

        if stage == "end":
            return {"next":"END", "queued_next": None}

        return {"next":"END", "queued_next": None}


# ------------------------- 构图（单场景） -------------------------

def build_graph(cfg, prompts, students, knowledge):
    llm = QwenLLM(
        model=cfg.get("model_name","qwen-flash"),
        api_key=cfg.get("api_key",""),
        base_url=cfg.get("base_url","https://dashscope.aliyuncs.com/compatible-mode/v1"),
        temperature=cfg.get("temperature",0.7),
    )
    teacher = TeacherAgent(llm, students, knowledge)
    pause_node = UserPauseNode()
    user_student = cfg.get("user_student_name", "Student1")
    user_queue = cfg.get("user_input_queue")
    user_turn_event = cfg.get("user_turn_event")
    students_map = {
        s: StudentAgent(
            s,
            llm,
            user_queue=user_queue if s == user_student else None,
            turn_event=user_turn_event if s == user_student else None,
        )
        for s in students
    }

    workflow = StateGraph(PBLState)
    workflow.add_node("teacher", teacher)
    workflow.add_node("user_pause", pause_node)

    for s, agent in students_map.items():
        def mk(a=agent):
            def _f(st: PBLState): return a(st)
            return _f
        workflow.add_node(s, mk())
        workflow.add_edge(s, "teacher")

    cond_teacher = {s: s for s in students}
    cond_teacher["END"] = END
    cond_teacher["teacher"] = "teacher"
    cond_teacher["user_pause"] = "user_pause"
    workflow.add_conditional_edges("teacher", lambda st: st["next"], cond_teacher)

    cond_pause = {s: s for s in students}
    cond_pause["END"] = END
    cond_pause["teacher"] = "teacher"
    workflow.add_conditional_edges("user_pause", lambda st: st["next"], cond_pause)
    workflow.add_edge(START, "teacher")
    return workflow.compile()


# ------------------------- 运行一个场景 -------------------------

def run_one_scene(scene_index: int,
                  scene_total: int,
                  case_text: str,
                  objectives_scene: Dict[str, Dict[str, str]],
                  cfg: Dict[str, Any],
                  prompts: Dict[str, Any],
                  knowledge: KnowledgeGraphRAG | None,
                  carry_messages: List[BaseMessage] | None,
                  carry_last_spoke: Dict[str,int] | None,
                  global_log: List[Dict[str, Any]],
                  student_stats: Dict[str, Dict[str, float]]):
    """运行单个场景，返回本场景消息列表与 last_spoke 用于可能的跨场景延续。"""
    students = cfg["students"]

    # 采样本场景能力
    mean_mat = cfg.get("ability_mean_matrix", [])
    assert scene_index < len(mean_mat), f"ability_mean_matrix 未提供第 {scene_index+1} 场景的均值向量"
    sigma = cfg.get("ability_sigma", 1.0)  # 可为标量或列表（逐目标）
    abilities_raw, abilities_desc, competence_avg = sample_student_abilities_for_scene(
        objectives_scene, mean_mat[scene_index], sigma, students, prompts
    )
    if cfg.get("debug_print_abilities"):
        print(f"\n[Scene {scene_index + 1}/{scene_total}] 学生能力采样：")
        for stu in students:
            raw = abilities_raw.get(stu, {})
            desc = abilities_desc.get(stu, {})
            summary_bits = [
                f"{obj}={raw.get(obj, 0):.2f}"
                for obj in objectives_scene.keys()
            ]
            print(
                f"  - {stu}: "
                f"{'; '.join(summary_bits)} | "
                f"平均 {competence_avg.get(stu, 0):.2f}"
            )
            if desc:
                for obj, text in desc.items():
                    print(f"      · {obj}：{text}")

    # 编译本场景图
    graph = build_graph(cfg, prompts, students, knowledge)

    # 初始状态（是否清空记忆）
    clear_mem = cfg.get("clear_memory_between_scenes", True)
    init_messages = [] if (carry_messages is None or clear_mem) else carry_messages
    init_last_spoke = {s: -10**9 for s in students} if (carry_last_spoke is None or clear_mem) else carry_last_spoke

    init_state: Dict[str, Any] = {
        "messages": init_messages,
        "next": "teacher",
        "stage": "scenario",
        "round_idx": 0 if clear_mem else 0,  # 每场景从 0 开始计数
        "last_spoke": init_last_spoke,
        "cfg": cfg,
        "prompts": prompts,

        # 本场景数据
        "scene_index": scene_index,
        "case_text": case_text,
        "objectives": objectives_scene,
        "scene_total": scene_total,

        # 能力（本场景）
        "abilities_raw": abilities_raw,
        "abilities_desc": abilities_desc,
        "competence_avg": competence_avg,

        "current_objective": None,
        "queued_next": None,
    }

    # 流式运行
    print(f"\n=== 场景 {scene_index+1} 开始 ===\n")
    last_messages = init_messages[:]
    last_spoke = dict(init_last_spoke)

    current_stage = init_state["stage"]
    message_hook = cfg.get("message_hook")
    for update in graph.stream(init_state, config={"recursion_limit": cfg.get("recursion_limit", 400)}):
        for node, val in update.items():
            if isinstance(val, dict):
                if "stage" in val:
                    current_stage = val["stage"]
                for msg in (val.get("messages",[]) or []):
                    speaker = getattr(msg,'name',msg.type)
                    if cfg.get("show_console", True):
                        print(f"【{speaker}】:\n{msg.content}\n")
                    last_messages.append(msg)
                    global_log.append({
                        "scene_index": scene_index,
                        "stage": current_stage,
                        "speaker": speaker,
                        "content": msg.content,
                    })
                    if message_hook:
                        payload = {
                            "scene_index": scene_index,
                            "stage": current_stage,
                            "speaker": speaker,
                            "content": msg.content,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        message_hook(payload)
                    if speaker in student_stats:
                        student_stats[speaker]["message_count"] += 1
                        student_stats[speaker]["char_count"] += len(msg.content or "")
                if "last_spoke" in val:
                    last_spoke = val["last_spoke"]
                if val.get("stage")=="end" or val.get("next")=="END":
                    if cfg.get("show_console", True):
                        print(f"=== 场景 {scene_index+1} 结束 ===")
                    return last_messages, last_spoke

    # 正常不会到这里
    if cfg.get("show_console", True):
        print(f"=== 场景 {scene_index+1} 结束（非常规） ===")
    return last_messages, last_spoke


# ------------------------- 示例数据与主入口 -------------------------

def ini_env(cfg):
    if cfg.get("api_key"):
        os.environ["DASHSCOPE_API_KEY"] = cfg["api_key"]


def load_knowledge_graph(cfg: Dict[str, Any]) -> KnowledgeGraphRAG | None:
    kg_cfg = cfg.get("knowledge_graph") or {}
    if kg_cfg.get("enabled") is False:
        return None

    candidates: List[tuple[str, str, str]] = []
    manual_dir = kg_cfg.get("dir")
    manual_entities = kg_cfg.get("entities_path")
    manual_relations = kg_cfg.get("relations_path")
    if manual_entities and manual_relations:
        candidates.append(("manual", manual_entities, manual_relations))
    elif manual_dir:
        candidates.append((
            manual_dir,
            os.path.join(manual_dir, "entities.json"),
            os.path.join(manual_dir, "relations.json"),
        ))
    else:
        # Common fallbacks, keep向后兼容旧目录名（含拼写错误）
        for base_dir in [
            os.path.join("data", "knoledge_graph"),
            os.path.join("data", "knowledge_graph"),
            "knoledge_graph",
            "knowledge_graph",
        ]:
            candidates.append((
                base_dir,
                os.path.join(base_dir, "entities.json"),
                os.path.join(base_dir, "relations.json"),
            ))

    entities_path = relations_path = None
    for _, ent_path, rel_path in candidates:
        if os.path.exists(ent_path) and os.path.exists(rel_path):
            entities_path, relations_path = ent_path, rel_path
            break

    if not entities_path:
        return None

    top_k = kg_cfg.get("top_k", 3)
    max_rel = kg_cfg.get("max_relation_items", 4)
    return KnowledgeGraphRAG(
        entities_path=entities_path,
        relations_path=relations_path,
        default_top_k=top_k,
        max_relation_items=max_rel,
    )


def initialize_system(
    config_path: str = CONFIG_PATH,
    prompts_path: str = PROMPTS_PATH,
    case_path: str = CASE_PATH,
    config_overrides: Dict[str, Any] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], KnowledgeGraphRAG | None]:
    """
    Aggregate the setup steps so entry points can bootstrap with a single call.
    Loads configuration, prompts, students, environment variables, case data,
    and the ability matrix derived from the selected case.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if config_overrides:
        cfg.update(config_overrides)
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    _prepare_students(cfg)
    ini_env(cfg)
    knowledge = load_knowledge_graph(cfg)
    case = load_case(cfg, case_path)
    init_ability_mean_matrix(cfg, case["objectives_by_scene"])

    return cfg, prompts, case, knowledge


def run_pbl_multi_scene(case, cfg, prompts, knowledge):
    scenes = case["contextualized_case"]
    objectives_all = case["objectives_by_scene"]
    assert len(scenes) == len(objectives_all), "contextualized_case 与 objectives_by_scene 场景数不一致"

    carry_messages = None
    carry_last_spoke = None
    total_scenes = len(scenes)
    global_log: List[Dict[str, Any]] = []
    student_stats: Dict[str, Dict[str, float]] = {
        s: {"message_count": 0.0, "char_count": 0.0}
        for s in cfg["students"]
    }

    for i, (scene_text, obj_scene) in enumerate(zip(scenes, objectives_all)):
        carry_messages, carry_last_spoke = run_one_scene(
            scene_index=i,
            scene_total=total_scenes,
            case_text=scene_text,
            objectives_scene=obj_scene,
            cfg=cfg,
            prompts=prompts,
            knowledge=knowledge,
            carry_messages=carry_messages,
            carry_last_spoke=carry_last_spoke,
            global_log=global_log,
            student_stats=student_stats,
        )

    return global_log, student_stats


def run_case_evaluation(
    cfg: Dict[str, Any],
    case: Dict[str, Any],
    knowledge: KnowledgeGraphRAG | None,
    log: List[Dict[str, Any]],
    student_stats: Dict[str, Dict[str, float]],
    prompts: Dict[str, Any],
    *,
    display: bool = True,
):
    target = cfg.get("evaluation_student")
    if not target:
        return None
    if knowledge is None:
        print("\n[评估提示] 知识图谱不可用，将在无RAG的情况下直接评估学生发言。")
    students = cfg.get("students", [])
    if target not in students:
        raise ValueError(f"未找到需要评估的学生：{target}")

    llm = QwenLLM(
        model=cfg.get("model_name","qwen-flash"),
        api_key=cfg.get("api_key",""),
        base_url=cfg.get("base_url","https://dashscope.aliyuncs.com/compatible-mode/v1"),
        temperature=cfg.get("temperature",0.7),
    )
    case_summary = compose_case_summary(case)
    eval_prompts = (prompts or {}).get("evaluation") if prompts else None
    if not eval_prompts:
        raise ValueError("prompts.json 中缺少 evaluation 提示配置。")
    dimension_defs = eval_prompts.get("dimensions")
    if not dimension_defs:
        raise ValueError("prompts.json.evaluation 中缺少 dimensions 列表。")
    evaluator = EvaluationAgent(
        llm,
        knowledge,
        case_summary,
        max_context_chars=int(cfg.get("evaluation_max_context_chars", 1600)),
        prompt_bundle=eval_prompts,
        activeness_sigmoid_a=float(cfg.get("activeness_sigmoid_a", 3.0)),
        dimension_defs=dimension_defs,
    )
    result = evaluator.evaluate(target, log, student_stats, students)
    if display:
        print("\n=== 学生评估 ===")
        print(render_evaluation_for_display(result))
    return result


def run_learning_advisor(
    cfg: Dict[str, Any],
    prompts: Dict[str, Any],
    log: List[Dict[str, Any]],
    evaluation_result: Dict[str, Any] | None,
    *,
    display: bool = True,
    test_scores: Dict[str, Any] | None = None,
    test_report: str | None = None,
):
    if not evaluation_result:
        return None

    llm = QwenLLM(
        model=cfg.get("model_name", "qwen-flash"),
        api_key=cfg.get("api_key", ""),
        base_url=cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        temperature=cfg.get("temperature", 0.7),
    )
    advisor_prompts = (prompts or {}).get("advisor") if prompts else None
    agent = AdvisorAgent(
        llm,
        prompt_bundle=advisor_prompts,
        max_transcript_chars=int(cfg.get("advisor_max_transcript_chars", 12000)),
    )
    student = evaluation_result.get("student") or cfg.get("evaluation_student", "")
    advice = agent.advise(
        student,
        evaluation_result,
        log,
        test_scores=test_scores,
        test_report=test_report,
    )
    if display:
        print("\n=== 学习建议 ===")
        print(render_advice_for_display(advice))
    return advice


def render_evaluation_for_display(result: Dict[str, Any] | None) -> str:
    if not result:
        return "暂无评估结果。"
    student = result.get("student", "未知学生")
    dims = result.get("dimensions") or {}
    lines = [f"学生：{student}"]
    for key, payload in dims.items():
        if not isinstance(payload, dict):
            continue
        title = payload.get("title") or key
        score = payload.get("score")
        justification = payload.get("justification", "")
        score_text = f"{score}分" if score is not None else "未评分"
        snippet = f"- {title}：{score_text}"
        if justification:
            snippet += f"（{justification}）"
        lines.append(snippet)
    stats = result.get("stats") or {}
    if stats:
        lines.append(
            f"- 统计：发言{int(stats.get('message_count', 0))}次，字数{int(stats.get('char_count', 0))}。"
        )
    return "\n".join(lines)


ADVICE_DIMENSION_LABELS = {
    "clinical_reasoning": "临床推理能力",
    "knowledge_accuracy": "知识准确性",
    "communication": "沟通与协作能力",
    "reflection": "总结与反思能力",
    "activeness": "参与度",
}


def _translate_dimension_label(label: str | None) -> str:
    if not label:
        return "未命名维度"
    key = str(label).strip()
    normalized = key.lower()
    return ADVICE_DIMENSION_LABELS.get(normalized, key)


def render_advice_for_display(advice: Dict[str, Any] | None) -> str:
    if not advice:
        return "暂无学习建议。"
    general = advice.get("general_advice") or {}
    lines: List[str] = []
    summary = general.get("summary")
    if summary:
        lines.append(f"总体建议：{summary}")
    resources = general.get("recommended_resources") or []
    if resources:
        lines.append("推荐资源：")
        for item in resources:
            lines.append(f"  - {item}")
    details = advice.get("detailed_advice") or []
    if details:
        lines.append("细化建议：")
        for idx, item in enumerate(details, 1):
            dimension = _translate_dimension_label(item.get("dimension", f"建议{idx}"))
            issue = str(item.get("issue", "")).strip()
            suggestion = str(item.get("suggestion", "")).strip()
            components = [f"{idx}. [{dimension}]"]
            if issue:
                components.append(issue)
            if suggestion:
                components.append(suggestion)
            text = " ".join(components)
            lines.append(text)
    return "\n".join(lines) if lines else "暂无学习建议。"


def run_pbl_workflow(
    case_id: str | None = None,
    *,
    config_path: str = CONFIG_PATH,
    prompts_path: str = PROMPTS_PATH,
    case_path: str = CASE_PATH,
    config_overrides: Dict[str, Any] | None = None,
    display: bool = False,
) -> Dict[str, Any]:
    overrides = dict(config_overrides or {})
    if case_id:
        overrides["case_id"] = case_id
    cfg, prompts, case, knowledge = initialize_system(
        config_path=config_path,
        prompts_path=prompts_path,
        case_path=case_path,
        config_overrides=overrides,
    )
    if not display:
        cfg["show_console"] = False
    full_log, student_stats = run_pbl_multi_scene(case, cfg, prompts, knowledge)
    evaluation_result = run_case_evaluation(
        cfg, case, knowledge, full_log, student_stats, prompts, display=display
    )
    return {
        "cfg": cfg,
        "case": case,
        "prompts": prompts,
        "log": full_log,
        "student_stats": student_stats,
        "evaluation": evaluation_result,
        "advice": None,
    }


if __name__ == "__main__":
    run_pbl_workflow(display=True)
