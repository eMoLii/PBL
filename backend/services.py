"""Utility helpers used by the Streamlit frontend."""
from __future__ import annotations

import json
import random
import threading
import time
import uuid
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence
import queue

import numpy as np

from agent_collaboration import pbl_engine
from agent_collaboration.sequential_recommender import (
    SequentialAdaptiveRecommender,
    load_case_ids,
)

BASE_DIR = Path(__file__).resolve().parents[1]
CASE_PATH = BASE_DIR / "agent_collaboration" / "data" / "case.json"
MATRIX_PATH = BASE_DIR / "agent_collaboration" / "data" / "case_similarity_matrix.npy"
CONFIG_PATH = BASE_DIR / "agent_collaboration" / "config.json"


@dataclass
class CaseBrief:
    case_id: str
    title: str
    summary: str
    objectives: List[str]


class CaseService:
    def __init__(self) -> None:
        with CASE_PATH.open("r", encoding="utf-8") as f:
            self.case_raw = json.load(f)
        self.case_ids = sorted(self.case_raw.keys())
        self.recommender: SequentialAdaptiveRecommender | None = None
        if MATRIX_PATH.exists():
            matrix = np.load(MATRIX_PATH)
            case_ids_sorted = load_case_ids(CASE_PATH)
            # 仅当矩阵与case_id长度匹配时启用推荐
            if matrix.shape[0] == len(case_ids_sorted):
                self.recommender = SequentialAdaptiveRecommender(matrix, case_ids_sorted)

    def list_cases(self) -> List[CaseBrief]:
        return [self.to_brief(case_id) for case_id in self.case_ids]

    def to_brief(self, case_id: str) -> CaseBrief:
        case = self.case_raw.get(case_id, {})
        original = case.get("original_case") or {}
        title = next((v for v in original.values() if isinstance(v, str) and v.strip()), case_id)
        summary = (case.get("contextualized_case") or [""])[0]
        objectives = []
        raw_objs = case.get("objectives_by_scene") or case.get("objextives") or []
        for obj_scene in raw_objs:
            for key in obj_scene.keys():
                objectives.append(str(key))
        if not objectives:
            objectives.append("综合诊断与管理")
        return CaseBrief(case_id=case_id, title=title.strip(), summary=summary.strip(), objectives=objectives)

    def recommend(self, history: Sequence[Dict[str, Any]], top_k: int = 3) -> List[CaseBrief]:
        if not history or not self.recommender:
            return [self.to_brief(cid) for cid in self.case_ids[:top_k]]
        hist_ids = []
        scores = []
        for item in history:
            cid = item.get("case_id")
            if not cid or cid not in self.case_ids:
                continue
            hist_ids.append(cid)
            score = item.get("post_score") or item.get("pre_score") or 60.0
            scores.append(max(0.0, min(5.0, float(score) / 20.0)))
        if not hist_ids:
            return [self.to_brief(cid) for cid in self.case_ids[:top_k]]
        try:
            recs = self.recommender.recommend(hist_ids, scores, top_k=top_k)
            return [self.to_brief(rec.case_id) for rec in recs]
        except Exception:
            return [self.to_brief(cid) for cid in self.case_ids[:top_k]]

    def remaining_cases(self, exclude: Sequence[str]) -> List[CaseBrief]:
        excl = set(exclude)
        return [self.to_brief(cid) for cid in self.case_ids if cid not in excl]

    def fetch_tests(self, case_id: str) -> Dict[str, List[Dict[str, Any]]]:
        case = self.case_raw.get(case_id, {})
        exams = case.get("exams") or {}
        pre = exams.get("pre-test") or exams.get("pre_test") or []
        post = exams.get("post-test") or exams.get("post_test") or []
        return {
            "pre": [dict(item) for item in pre],
            "post": [dict(item) for item in post],
        }

    def build_teacher_intro(self, case_id: str) -> str:
        case = self.case_raw.get(case_id, {})
        scenes = case.get("contextualized_case") or []
        first_scene = scenes[0] if scenes else ""
        objectives = case.get("objectives_by_scene") or case.get("objextives") or []
        obj_lines: List[str] = []
        for idx, scene_obj in enumerate(objectives, 1):
            keys = list(scene_obj.keys()) if isinstance(scene_obj, dict) else []
            if keys:
                obj_lines.append(f"目标{idx}:{'、'.join(keys)}")
        objectives_text = "; ".join(obj_lines) if obj_lines else "聚焦临床推理与诊疗计划"
        if first_scene:
            intro = f"同学们好，今天我们启动案例 {case_id} 的学习。第一幕情境概述：{first_scene}"
        else:
            intro = f"同学们好，今天我们启动案例 {case_id} 的学习。"
        intro += f" 请依次围绕这些学习目标展开：{objectives_text}。"
        return intro

    def scene_count(self, case_id: str) -> int:
        case = self.case_raw.get(case_id, {})
        scenes = case.get("contextualized_case") or []
        if isinstance(scenes, list) and scenes:
            return len(scenes)
        return 1


case_service = CaseService()


class PBLInteractiveSession:
    def __init__(
        self,
        case_id: str,
        interval: float = 12.0,
        user_student: str = "Student1",
        ability_window: tuple[float, float] | None = None,
        students_count: int | None = None,
    ):
        self.id = uuid.uuid4().hex
        self.case_id = case_id
        self.base_interval = max(0.0, float(interval))
        self.interval = self.base_interval
        self.speed_factor = 1.0
        self.ability_window = ability_window
        self.students_count = students_count
        self.user_student = user_student
        self.queue: "queue.Queue[str]" = queue.Queue()
        self.user_turn_event = threading.Event()
        self.log: List[Dict[str, Any]] = []
        self.status = "init"
        self.waiting_for_user = False
        self.error = ""
        self.evaluation: Dict[str, Any] | None = None
        self.advice: Dict[str, Any] | None = None
        self.student_stats: Dict[str, Any] | None = None
        self._thread: threading.Thread | None = None
        self.last_emit_time = 0.0
        self.state_lock = threading.Lock()
        self.suppress_non_user_messages = False
        self.freeze_log_index: int | None = None
        self.cfg_snapshot: Dict[str, Any] | None = None
        self.prompts_snapshot: Dict[str, Any] | None = None

    def start(self) -> None:
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            result = pbl_engine.run_pbl_workflow(
                case_id=self.case_id,
                config_overrides={
                    "user_input_queue": self.queue,
                    "user_turn_event": self.user_turn_event,
                    "message_hook": self._on_message,
                    "user_waiting_callback": self._on_waiting_flag,
                    "user_student_name": self.user_student,
                    "user_turn_grace_period": self.interval,
                    "show_console": False,
                    **(
                        {
                            "ability_mean_low": self.ability_window[0],
                            "ability_mean_high": self.ability_window[1],
                        }
                        if self.ability_window
                        else {}
                    ),
                    **(
                        {"students_count": int(self.students_count)}
                        if self.students_count is not None
                        else {}
                    ),
                },
                display=False,
            )
            self.evaluation = result.get("evaluation")
            self.advice = result.get("advice")
            self.student_stats = result.get("student_stats")
            self.cfg_snapshot = result.get("cfg")
            self.prompts_snapshot = result.get("prompts")
            if self.evaluation is not None:
                print(
                    "[PBL] EvaluationAgent output:\n"
                    f"{json.dumps(self.evaluation, ensure_ascii=False, indent=2)}",
                    flush=True,
                )
            if self.advice is not None:
                print(
                    "[PBL] AdvisorAgent output:\n"
                    f"{json.dumps(self.advice, ensure_ascii=False, indent=2)}",
                    flush=True,
                )
            self.status = "completed"
        except Exception as exc:
            self.error = str(exc)
            self.status = "error"
        finally:
            self.waiting_for_user = False

    def refresh_advice_with_tests(
        self,
        pre_score: float,
        post_score: float,
        test_report: str | None = None,
    ) -> Dict[str, Any] | None:
        with self.state_lock:
            if not self.evaluation:
                raise ValueError("evaluation not ready")
            cfg = dict(self.cfg_snapshot or {})
            prompts = dict(self.prompts_snapshot or {})
            log_copy = list(self.log)
        test_scores = {"pre_score": pre_score, "post_score": post_score}
        advice = pbl_engine.run_learning_advisor(
            cfg,
            prompts,
            log_copy,
            self.evaluation,
            display=False,
            test_scores=test_scores,
            test_report=test_report,
        )
        with self.state_lock:
            self.advice = advice
        if advice is not None:
            print(
                "[PBL] AdvisorAgent (with tests) output:\n"
                f"{json.dumps(advice, ensure_ascii=False, indent=2)}",
                flush=True,
            )
        return advice

    def _on_message(self, payload: Dict[str, Any]) -> None:
        speaker = payload.get("speaker")
        with self.state_lock:
            drop = False
            freeze = self.freeze_log_index
            is_teacher = bool(speaker and str(speaker).lower() == "teacher")
            if (
                speaker
                and speaker != self.user_student
                and not is_teacher
                and freeze is not None
                and len(self.log) >= freeze
            ):
                drop = True
            elif (
                speaker
                and speaker != self.user_student
                and not is_teacher
                and self.suppress_non_user_messages
            ):
                drop = True
            if speaker == self.user_student:
                self.suppress_non_user_messages = False
                self.freeze_log_index = None
            log_len = len(self.log)
        print(
            f"[hook] {time.strftime('%H:%M:%S')} speaker={speaker} "
            f"drop={drop} len_before={log_len} freeze={freeze} suppress={self.suppress_non_user_messages}",
            flush=True,
        )
        if speaker:
            try:
                content = payload.get("content") or ""
                print(f"[dialogue] {speaker}: {content}\n\n", flush=True)
            except Exception:
                pass
        if drop:
            return

        if speaker:
            now = time.time()
            if speaker == self.user_student:
                self.last_emit_time = now
            else:
                wait = self.interval - (now - self.last_emit_time)
                if wait > 0:
                    time.sleep(wait)
                self.last_emit_time = time.time()
        self.log.append(payload)

    def _on_waiting_flag(self, flag: bool) -> None:
        with self.state_lock:
            self.waiting_for_user = flag or self.user_turn_event.is_set()

    def set_speed_factor(self, factor: float) -> None:
        factor = float(factor)
        if factor <= 0:
            raise ValueError("speed factor must be positive")
        with self.state_lock:
            self.speed_factor = factor
            # 防止出现过小的等待时间导致线程忙等
            adjusted = self.base_interval * factor
            self.interval = max(0.05, adjusted)

    def _activate_user_turn_locked(self) -> None:
        self.user_turn_event.set()
        self.waiting_for_user = True

    def submit_user_message(self, text: str) -> None:
        cleaned = text.strip() or "（用户暂未发言）"
        self.queue.put(cleaned)
        with self.state_lock:
            self.user_turn_event.clear()
            self.waiting_for_user = False
            self.suppress_non_user_messages = False
            self.freeze_log_index = None

    def request_turn(self) -> None:
        with self.state_lock:
            self.suppress_non_user_messages = True
            self.freeze_log_index = len(self.log)
            self._activate_user_turn_locked()

    def resume_discussion(self) -> None:
        with self.state_lock:
            self.user_turn_event.clear()
            self.waiting_for_user = False
            self.suppress_non_user_messages = False
            self.freeze_log_index = None

    def serialize(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "log": list(self.log),
            "waiting_for_user": self.waiting_for_user,
            "error": self.error,
            "evaluation": self.evaluation,
            "advice": self.advice,
            "student_stats": self.student_stats,
        }


class PBLSessionManager:
    def __init__(self, pause_interval: float = 12.0):
        self.sessions: Dict[str, PBLInteractiveSession] = {}
        self.lock = threading.Lock()
        self.pause_interval = pause_interval

    def create_session(
        self,
        case_id: str,
        speed_factor: float = 1.0,
        ability_window: tuple[float, float] | None = None,
        students_count: int | None = None,
    ) -> str:
        session = PBLInteractiveSession(
            case_id,
            interval=self.pause_interval,
            ability_window=ability_window,
            students_count=students_count,
        )
        if speed_factor != 1.0:
            session.set_speed_factor(speed_factor)
        with self.lock:
            self.sessions[session.id] = session
        session.start()
        return session.id

    def get_state(self, session_id: str) -> Dict[str, Any]:
        session = self.sessions.get(session_id)
        if not session:
            return {"status": "not_found"}
        return session.serialize()

    def submit_message(self, session_id: str, text: str) -> None:
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError("session not found")
        session.submit_user_message(text)

    def resume(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError("session not found")
        session.resume_discussion()

    def set_speed_factor(self, session_id: str, factor: float) -> None:
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError("session not found")
        session.set_speed_factor(factor)

    def refresh_advice(
        self,
        session_id: str,
        pre_score: float,
        post_score: float,
        test_report: str | None = None,
    ) -> Dict[str, Any] | None:
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError("session not found")
        return session.refresh_advice_with_tests(pre_score, post_score, test_report)


def _load_pause_interval() -> float:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return float(cfg.get("user_turn_grace_period", 12.0))
    except Exception:
        return 12.0


_SESSION_MANAGER = PBLSessionManager(pause_interval=_load_pause_interval())


def run_agent_workflow(case_id: str, cfg_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    overrides = dict(cfg_overrides or {})
    overrides["case_id"] = case_id
    overrides.setdefault("show_console", False)
    return pbl_engine.run_pbl_workflow(config_overrides=overrides, display=False)


def start_interactive_session(
    case_id: str,
    speed_factor: float = 1.0,
    ability_window: tuple[float, float] | None = None,
    students_count: int | None = None,
) -> str:
    return _SESSION_MANAGER.create_session(
        case_id,
        speed_factor=speed_factor,
        ability_window=ability_window,
        students_count=students_count,
    )


def get_interactive_session_state(session_id: str) -> Dict[str, Any]:
    return _SESSION_MANAGER.get_state(session_id)


def submit_user_message(session_id: str, text: str) -> None:
    _SESSION_MANAGER.submit_message(session_id, text)


def request_user_turn(session_id: str) -> None:
    session = _SESSION_MANAGER.sessions.get(session_id)
    if not session:
        raise KeyError("session not found")
    session.request_turn()


def resume_user_turn(session_id: str) -> None:
    _SESSION_MANAGER.resume(session_id)


def set_session_speed(session_id: str, speed_factor: float) -> None:
    _SESSION_MANAGER.set_speed_factor(session_id, speed_factor)


def refresh_advice_with_tests(
    session_id: str,
    pre_score: float,
    post_score: float,
    test_report: str | None = None,
) -> Dict[str, Any] | None:
    return _SESSION_MANAGER.refresh_advice(session_id, pre_score, post_score, test_report)


PRE_TEST_QUESTIONS = [
    "病例核心症状有哪些？",
    "急需排查的危险信号是什么？",
    "你初步考虑的诊断路径？",
]

POST_TEST_QUESTIONS = [
    "概括该病例最关键的鉴别点？",
    "给出你的最终诊断及理由？",
    "下一步管理措施计划？",
]

# 参考答案占位符，若案例数据未提供新题目时使用
PRE_TEST_ANSWERS = {question: "（参考答案待补充）" for question in PRE_TEST_QUESTIONS}
POST_TEST_ANSWERS = {question: "（参考答案待补充）" for question in POST_TEST_QUESTIONS}


def default_test_items(kind: str) -> List[Dict[str, Any]]:
    if kind == "pre":
        source = PRE_TEST_QUESTIONS
        answers = PRE_TEST_ANSWERS
    else:
        source = POST_TEST_QUESTIONS
        answers = POST_TEST_ANSWERS
    return [
        {
            "question": q,
            "answer": answers.get(q, "（参考答案待补充）"),
            "question_type": "问答题",
            "option": {},
        }
        for q in source
    ]


def _normalize_key_list(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    letters = re.findall(r"[A-Z]", value.upper())
    seen = []
    for letter in letters:
        if letter not in seen:
            seen.append(letter)
    return seen


def score_test_items(test_items: List[Dict[str, Any]], answers: Dict[str, Any]) -> float:
    if not test_items:
        return 0.0
    correct = 0
    total = 0
    for item in test_items:
        qid = item.get("qid") or item.get("id")
        if not qid:
            continue
        expected_raw = item.get("answer", "")
        expected = _normalize_key_list(expected_raw)
        resp = answers.get(qid)
        if isinstance(resp, list):
            user = sorted(resp)
        else:
            if resp is None:
                user = []
            else:
                user = _normalize_key_list(str(resp))
        total += 1
        if user and expected and sorted(expected) == user:
            correct += 1
    score = (correct / total) * 100 if total else 0.0
    return round(score, 1)


def mock_evaluation(
    pre_score: float,
    post_score: float,
    user_messages: int,
    total_messages: int,
    case_id: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    gain = max(0.0, post_score - pre_score)
    reasoning = min(5.0, 2.0 + (post_score / 100.0) * 3.0)
    knowledge = min(5.0, 1.5 + (pre_score + post_score) / 80.0 * 3.5)
    activeness_ratio = user_messages / max(total_messages, 1)
    activeness = max(1.0, min(5.0, activeness_ratio * 5.0))

    evaluation = {
        "student": "Student1",
        "dimensions": {
            "clinical_reasoning": {"title": "临床推理能力", "score": round(reasoning, 1), "justification": f"后测{post_score:.0f}分，推理链条更完整。"},
            "knowledge_accuracy": {"title": "知识准确性", "score": round(knowledge, 1), "justification": "回答覆盖核心病理概念。"},
            "communication": {"title": "沟通与协作能力", "score": 4.0, "justification": "能顺接同伴观点并补充信息。"},
            "reflection": {"title": "总结与反思能力", "score": 3.8, "justification": "能指出薄弱点并制定复盘计划。"},
            "activeness": {"title": "参与度", "score": round(activeness, 1), "justification": f"发言{user_messages}次/总{total_messages}次。"},
        },
        "stats": {"pre_score": pre_score, "post_score": post_score, "gain": gain},
    }

    improvements = "明显提升" if gain >= 20 else ("小幅提升" if gain > 5 else "保持稳定")
    summary = (
        f"你在{case_id}案例训练中{improvements}，推理与知识准确性同步进步，"
        "继续保持主动提问与条理化总结。"
    )
    advice = {
        "general_advice": {
            "summary": summary,
            "recommended_resources": [
                "复习《内科学》心血管章节：胸痛评估与风险分层",
                "查阅病例相关指南，构建下一步诊疗流程图",
            ],
        },
        "detailed_advice": [
            {
                "dimension": "参与度",
                "issue": "发言节奏略慢",
                "suggestion": "每轮至少提出1个澄清问题",
            },
            {
                "dimension": "知识准确性",
                "issue": "诊断理由需更量化",
                "suggestion": "引用指南阈值支撑判断",
            },
        ],
    }
    return evaluation, advice
