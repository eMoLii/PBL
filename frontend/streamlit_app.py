from __future__ import annotations

import json
import sys
import threading
import time
import uuid
import re
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit.components.v1 as components
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
CONFIG_PATH = REPO_ROOT / "agent_collaboration" / "config.json"

from backend.database import (  # noqa: E402
    fetch_user_history,
    init_db,
    record_feedback,
    record_study_session,
    record_survey_response,
    seed_users,
    verify_user,
    create_session_token,
    fetch_user_by_session_token,
    delete_session_token,
    get_user_profile,
    upsert_user_profile,
    get_user_settings,
    upsert_user_settings,
)
from backend.services import (  # noqa: E402
    POST_TEST_ANSWERS,
    POST_TEST_QUESTIONS,
    PRE_TEST_ANSWERS,
    PRE_TEST_QUESTIONS,
    case_service,
    default_test_items,
    start_interactive_session,
    get_interactive_session_state,
    submit_user_message,
    request_user_turn,
    resume_user_turn,
    set_session_speed,
    refresh_advice_with_tests,
    score_test_items,
    save_active_session_for_user,
    load_active_session_for_user,
    clear_active_session_for_user,
)

OBJ_TAG_PATTERN = re.compile(r"<\s*OBJ\s*:\s*(.+?)>")

SPEECH_SPEED_FACTORS = {
    1: 3.0,
    2: 2.0,
    3: 1.5,
    4: 1.0,
    5: 1 / 1.5,
    6: 0.5,
    7: 1 / 3.0,
}

SPEECH_SPEED_DESCRIPTIONS = {
    1: "第1档 · 超慢",
    2: "第2档 · 慢速",
    3: "第3档 · 稍慢",
    4: "第4档 · 默认",
    5: "第5档 · 稍快",
    6: "第6档 · 极速",
    7: "第7档 · 超极速",
}

SPEECH_SPEED_DESCRIPTIONS_EN = {
    1: "Level 1 · Ultra slow",
    2: "Level 2 · Slow",
    3: "Level 3 · Slightly slow",
    4: "Level 4 · Default",
    5: "Level 5 · Slightly fast",
    6: "Level 6 · Very fast",
    7: "Level 7 · Ultra fast",
}

SESSION_COOKIE_NAME = "pbl_session_token"
SESSION_COOKIE_DAYS = 30
ADVANCED_OBJECTIVE_THRESHOLD = 0.5

def _load_app_language() -> str:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        lang = str(cfg.get("language", "zh")).lower()
    except Exception:
        lang = "zh"
    return "en" if lang.startswith("en") else "zh"


def _tr(zh: str, en: str) -> str:
    lang = st.session_state.get("app_language") or _load_app_language()
    st.session_state["app_language"] = lang
    return en if lang == "en" else zh


def _speed_description(level: int) -> str:
    lang = st.session_state.get("app_language") or _load_app_language()
    if lang == "en":
        return SPEECH_SPEED_DESCRIPTIONS_EN.get(level, "Default speed")
    return SPEECH_SPEED_DESCRIPTIONS.get(level, "默认速度")

SURVEY_QUESTIONS = [
    "我对本次使用该 PBL 讨论系统的整体体验感到满意。",
    "我认为该系统有助于我理解并掌握本病例的临床推理过程。",
    "我在讨论中始终保持投入，并积极参与推理与决策。",
    "我在讨论中能够按自己期望的方式表达观点并主导推理。",
    "虚拟学生的发言促使我更深入地思考并参与讨论。",
    "教师的引导/监督让我感到被支持，并能推动讨论朝正确方向。",
    "当我困惑或偏离主题时，系统能及时提供有效提示或纠偏。",
    "通过本次讨论，我对自己独立分析病例并形成诊断思路更有信心。",
    "我有信心将本次学到的推理方法迁移到相似病例中。",
    "在该系统中完成我的角色任务需要投入较大脑力负荷。",
    "在讨论过程中，我经常感到信息量过大或步骤过于复杂。",
    "系统界面/提示/虚拟学生的互动让我分心，影响我聚焦病例关键线索。",
]

SURVEY_CHOICES = ["非常同意", "比较同意", "不确定", "比较不同意", "非常不同意"]
SURVEY_QUESTIONS_EN = [
    "I am satisfied with the overall experience of using this PBL discussion system.",
    "The system helped me understand and master the clinical reasoning of this case.",
    "I stayed engaged and actively participated in reasoning and decision-making.",
    "I could express my views and steer my reasoning the way I wanted.",
    "The virtual students prompted me to think deeper and participate more.",
    "The teacher’s guidance/supervision made me feel supported and steered the discussion correctly.",
    "When I was confused or off-topic, the system provided timely and effective guidance or correction.",
    "After this discussion, I feel more confident in independently analyzing cases and forming diagnostic plans.",
    "I am confident I can transfer the reasoning approach learned here to similar cases.",
    "Completing my role in this system requires significant mental effort.",
    "During discussion, I often felt the information load or steps were too complex to keep up.",
    "The interface/prompts/virtual student interactions distracted me from key clinical cues.",
]
SURVEY_CHOICES_EN = ["Strongly agree", "Somewhat agree", "Neutral", "Somewhat disagree", "Strongly disagree"]


def _current_speed_level() -> int:
    level = int(st.session_state.get("speech_speed_level", 4))
    if level not in SPEECH_SPEED_FACTORS:
        level = 4
    return level


def _current_speed_factor() -> float:
    return float(SPEECH_SPEED_FACTORS.get(_current_speed_level(), 1.0))


def _load_default_students_count() -> int:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("students_count", 5))
    except Exception:
        return 5


def _apply_user_preferences(user_id: int) -> None:
    try:
        prefs = get_user_settings(user_id) or {}
    except Exception as exc:
        st.warning(_tr(f"加载个性化设置失败：{exc}", f"Failed to load preferences: {exc}"))
        return
    speed = prefs.get("speed_level")
    if isinstance(speed, int) and 1 <= speed <= 7:
        st.session_state["speech_speed_level"] = speed
        st.session_state["persisted_speed_level"] = speed
    else:
        st.session_state["persisted_speed_level"] = st.session_state.get("speech_speed_level", 4)
    count = prefs.get("student_count")
    if isinstance(count, int) and 4 <= count <= 8:
        st.session_state["desired_students_count"] = count
        st.session_state["persisted_student_count"] = count
    else:
        st.session_state["persisted_student_count"] = st.session_state.get(
            "desired_students_count", _load_default_students_count()
        )


def _persist_user_preferences(
    *,
    speed_level: Optional[int] = None,
    student_count: Optional[int] = None,
) -> bool:
    user = st.session_state.get("user")
    if not user or (speed_level is None and student_count is None):
        return False
    try:
        upsert_user_settings(
            user_id=user["id"],
            speed_level=speed_level,
            student_count=student_count,
        )
        return True
    except Exception as exc:
        st.warning(_tr(f"保存个性化设置失败：{exc}", f"Failed to save preferences: {exc}"))
        return False


def _ability_window_from_pre_score(pre_score: float | None) -> tuple[float, float] | None:
    if pre_score is None:
        return None
    normalized = max(0.0, min(5.0, float(pre_score) / 20.0))
    if normalized >= 4.0:
        return (3.0, 5.0)
    if normalized <= 1.0:
        return (0.0, 2.0)
    low = max(0.0, normalized - 1.0)
    high = min(5.0, normalized + 1.0)
    return (round(low, 3), round(high, 3))


def _normalize_choice_answer(value: str) -> List[str]:
    if not isinstance(value, str):
        return []
    letters = re.findall(r"[A-Za-z]", value.upper())
    seen: List[str] = []
    for letter in letters:
        if letter not in seen:
            seen.append(letter)
    return seen


def _evaluate_question_correct(item: Dict[str, Any], response: Any) -> bool:
    expected = _normalize_choice_answer(item.get("answer", ""))
    if isinstance(response, list):
        user = sorted([str(opt).upper() for opt in response])
        return bool(expected and user and user == sorted(expected))
    if response is None:
        return False
    user = _normalize_choice_answer(str(response))
    return bool(expected and user and sorted(user) == sorted(expected))


def _compute_scene_advanced_mask(correctness: List[bool]) -> List[bool]:
    layout = st.session_state.get("scene_objective_keys") or []
    mask: List[bool] = []
    idx = 0
    for keys in layout:
        total = 0
        passed = 0
        for _ in keys:
            if idx < len(correctness):
                total += 1
                if correctness[idx]:
                    passed += 1
            idx += 1
        if total == 0:
            mask.append(True)
        else:
            mask.append((passed / total) >= ADVANCED_OBJECTIVE_THRESHOLD)
    return mask


def _compute_student_stats(log: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for entry in log:
        speaker = entry.get("speaker")
        if not speaker:
            continue
        content = entry.get("content") or ""
        slot = stats.setdefault(speaker, {"message_count": 0.0, "char_count": 0.0})
        slot["message_count"] += 1.0
        slot["char_count"] += float(len(str(content)))
    return stats


def _build_active_session_metadata(next_scene_index: Optional[int] = None) -> Dict[str, Any]:
    log = st.session_state.get("pbl_log", [])
    if next_scene_index is None:
        next_scene_index = st.session_state.get("next_scene_index", 0)
    return {
        "log": log,
        "pre_answers": st.session_state.get("pre_answers", {}),
        "pre_score": st.session_state.get("pre_score"),
        "pre_question_correctness": st.session_state.get("pre_question_correctness", []),
        "scene_objective_keys": st.session_state.get("scene_objective_keys"),
        "advanced_scene_mask": st.session_state.get("advanced_scene_mask"),
        "desired_students_count": st.session_state.get("desired_students_count"),
        "next_scene_index": int(next_scene_index or 0),
        "total_scenes": st.session_state.get("total_scenes"),
        "student_stats": _compute_student_stats(log),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _save_active_session_state(force: bool = False, next_scene_index: Optional[int] = None) -> None:
    user = st.session_state.get("user")
    session_id = st.session_state.get("pbl_session_id")
    case_id = st.session_state.get("selected_case_id")
    if not (user and session_id and case_id):
        return
    total_scenes = st.session_state.get("total_scenes") or len(st.session_state.get("scene_objective_keys") or [])
    if next_scene_index is None:
        next_scene_index = st.session_state.get("next_scene_index", 0)
    if total_scenes and next_scene_index >= total_scenes:
        return
    log = st.session_state.get("pbl_log", [])
    last_len = st.session_state.get("last_saved_log_len", 0)
    if not force and len(log) == last_len:
        return
    metadata = _build_active_session_metadata(next_scene_index=next_scene_index)
    try:
        save_active_session_for_user(user["id"], case_id, session_id, metadata)
        st.session_state["last_saved_log_len"] = len(log)
    except Exception as exc:
        if force:
            st.warning(_tr(f"保存讨论进度失败：{exc}", f"Failed to save progress: {exc}"))


def _clear_active_session_record() -> None:
    user = st.session_state.get("user")
    if user:
        try:
            clear_active_session_for_user(user["id"])
        except Exception:
            pass
    st.session_state["pending_active_session"] = None
    st.session_state["last_saved_log_len"] = 0
    st.session_state["last_saved_scene_index"] = -1
    st.session_state["next_scene_index"] = 0
    st.session_state["session_lost"] = False
    st.session_state["pending_tab_click"] = None


def _load_active_session_record() -> None:
    user = st.session_state.get("user")
    if not user or st.session_state.get("active_session_checked"):
        return
    st.session_state["active_session_checked"] = True
    try:
        data = load_active_session_for_user(user["id"])
    except Exception as exc:
        st.warning(_tr(f"加载未完成讨论失败：{exc}", f"Failed to load unfinished discussion: {exc}"))
        data = None
    if data:
        metadata = data.get("metadata") or {}
        total = metadata.get("total_scenes")
        next_idx = metadata.get("next_scene_index")
        if isinstance(total, int) and isinstance(next_idx, int) and next_idx >= total:
            try:
                clear_active_session_for_user(user["id"])
            except Exception:
                pass
            data = None
    st.session_state["pending_active_session"] = data


def _resume_saved_session(entry: Dict[str, Any]) -> None:
    case_id = entry.get("case_id")
    metadata = entry.get("metadata") or {}
    if not case_id:
        st.warning(_tr("保存的讨论记录已失效。", "Saved discussion is invalid."))
        _clear_active_session_record()
        return
    next_scene_index = int(metadata.get("next_scene_index", 0))
    _clear_active_session_record()
    st.session_state["selected_case_id"] = case_id
    st.session_state["case_brief"] = case_service.to_brief(case_id)
    tests = case_service.fetch_tests(case_id)
    st.session_state["pre_test_items"] = _prepare_test_items(tests["pre"], "pre", "pre")
    st.session_state["post_test_items"] = _prepare_test_items(tests["post"], "post", "post")
    st.session_state["total_scenes"] = case_service.scene_count(case_id)
    st.session_state["scene_objective_keys"] = metadata.get("scene_objective_keys") or case_service.scene_objective_layout(case_id)
    st.session_state["advanced_scene_mask"] = metadata.get("advanced_scene_mask")
    st.session_state["pre_answers"] = metadata.get("pre_answers", {})
    st.session_state["pre_score"] = metadata.get("pre_score")
    st.session_state["pre_question_correctness"] = metadata.get("pre_question_correctness", [])
    st.session_state["desired_students_count"] = metadata.get("desired_students_count", st.session_state.get("desired_students_count"))
    prefill_log = metadata.get("log", [])
    st.session_state["pbl_log"] = list(prefill_log)
    st.session_state["last_force_speak_total"] = sum(
        1 for entry in prefill_log if _is_student_speaker(entry.get("speaker"))
    )
    total_scenes = st.session_state.get("total_scenes") or len(st.session_state.get("scene_objective_keys") or [])
    if total_scenes and next_scene_index >= total_scenes:
        st.info(_tr("该讨论已完成，无法继续。", "This discussion is already finished and cannot be resumed."))
        _clear_active_session_record()
        st.session_state["page"] = "case_selection"
        st.rerun()
        st.stop()
    ability_window = _ability_window_from_pre_score(st.session_state.get("pre_score"))
    speed_factor = _current_speed_factor()
    students_count = metadata.get("desired_students_count", st.session_state.get("desired_students_count"))
    current_user = st.session_state.get("user") or {}
    session_id = start_interactive_session(
        case_id,
        speed_factor=speed_factor,
        owner_user_id=current_user.get("id"),
        owner_username=current_user.get("username"),
        ability_window=ability_window,
        students_count=int(students_count) if students_count else None,
        advanced_mask=metadata.get("advanced_scene_mask"),
        start_scene_index=next_scene_index,
        prefill_log=prefill_log,
        prefill_stats=metadata.get("student_stats"),
    )
    st.session_state["pbl_session_id"] = session_id
    st.session_state["speed_sync"] = {
        "session_id": session_id,
        "level": _current_speed_level(),
    }
    st.session_state["pending_active_session"] = None
    st.session_state["last_saved_log_len"] = len(prefill_log)
    st.session_state["last_saved_scene_index"] = next_scene_index - 1
    st.session_state["next_scene_index"] = next_scene_index
    st.session_state["page"] = "pbl_training"
    st.session_state["session_lost"] = False
    st.session_state["scroll_to_top_on_train"] = True
    total_scenes = st.session_state.get("total_scenes") or len(st.session_state.get("scene_objective_keys") or [])
    target_index = min(max(next_scene_index, 0), max(total_scenes - 1, 0))
    st.session_state["pending_tab_click"] = target_index
    st.session_state["session_saved"] = False
    st.session_state["teacher_reply_pending"] = False
    st.session_state["awaiting_teacher_after_user"] = False
    st.session_state["allow_user_input"] = False
    st.session_state["manual_pause_active"] = True
    st.session_state["scene_transition_ready_at"] = 0.0
    st.session_state["scene_transition_waiting"] = False
    st.session_state["last_transition_scene"] = next_scene_index - 1
    _save_active_session_state(force=True, next_scene_index=next_scene_index)
    st.rerun()
    st.stop()


def _set_session_cookie(token: str, expires_at: datetime) -> None:
    expires_str = expires_at.isoformat()
    script = f"""
    <script>
    const expires = new Date('{expires_str}');
    document.cookie = '{SESSION_COOKIE_NAME}=' + encodeURIComponent('{token}') + '; expires=' + expires.toUTCString() + '; path=/';
    </script>
    """
    components.html(script, height=0)


def _clear_session_cookie() -> None:
    script = f"""
    <script>
    document.cookie = '{SESSION_COOKIE_NAME}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/';
    </script>
    """
    components.html(script, height=0)


def _get_session_cookie() -> Optional[str]:
    try:
        headers = getattr(st.context, "headers", {}) or {}
    except Exception:
        headers = {}
    cookie_header = headers.get("Cookie") or headers.get("cookie")
    if not cookie_header:
        return None
    parts = cookie_header.split(";")
    for part in parts:
        if not part.strip():
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        if k.strip() == SESSION_COOKIE_NAME:
            return v.strip()
    return None


def _restore_user_from_cookie() -> None:
    if st.session_state.get("user"):
        return
    token = _get_session_cookie()
    if not token:
        return
    user = fetch_user_by_session_token(token)
    if user:
        st.session_state["user"] = user
        st.session_state["session_token"] = token
        _apply_user_preferences(user["id"])
    else:
        _clear_session_cookie()


def _render_speed_slider() -> None:
    default_level = _current_speed_level()
    level = st.slider(
        _tr("小组发言速度", "Group speaking speed"),
        min_value=1,
        max_value=7,
        value=default_level,
        step=1,
        key="speech_speed_slider",
    )
    st.session_state["speech_speed_level"] = level
    user = st.session_state.get("user")
    if user:
        if st.session_state.get("persisted_speed_level") != level:
            if _persist_user_preferences(speed_level=level):
                st.session_state["persisted_speed_level"] = level
    else:
        st.session_state["persisted_speed_level"] = level
    desc = _speed_description(level)
    st.caption(desc)


def _clear_page_container(key: str) -> None:
    placeholder = st.session_state.pop(key, None)
    if placeholder is not None:
        placeholder.empty()


def _sync_speed_to_session(session_id: Optional[str]) -> None:
    if not session_id:
        st.session_state.pop("speed_sync", None)
        return
    level = _current_speed_level()
    last = st.session_state.get("speed_sync") or {}
    if last.get("session_id") == session_id and last.get("level") == level:
        return
    factor = _current_speed_factor()
    try:
        set_session_speed(session_id, factor)
    except Exception as exc:
        st.warning(_tr(f"同步发言速度失败：{exc}", f"Failed to sync speaking speed: {exc}"))
    else:
        st.session_state["speed_sync"] = {"session_id": session_id, "level": level}


def initialize_app_state() -> None:
    if "page" not in st.session_state:
        st.session_state["page"] = "login"
    if "pbl_log" not in st.session_state:
        st.session_state["pbl_log"] = []
    for key in ["agent_evaluation", "agent_advice", "agent_stats", "discussion_ran"]:
        if key not in st.session_state:
            st.session_state[key] = None
    for key, default in (
        ("pbl_session_id", None),
        ("pending_user_messages", []),
        ("student1_log_seen", 0),
        ("max_seen_scene", -1),
        ("total_scenes", 1),
        ("pbl_turn_requested", False),
        ("scene_transition_ready_at", 0.0),
        ("scene_transition_waiting", False),
        ("last_transition_scene", -1),
        ("last_auto_pause_scene", -1),
        ("allow_user_input", False),
        ("manual_pause_active", False),
        ("summary_invite_scene", -1),
        ("advice_enhanced", False),
        ("teacher_reply_pending", False),
        ("awaiting_teacher_after_user", False),
        ("last_teacher_seen_count", 0),
        ("desired_students_count", None),
        ("session_token", None),
        ("user_profile_cache", None),
        ("scene_objective_keys", []),
        ("advanced_scene_mask", None),
        ("pre_question_correctness", []),
        ("pending_active_session", None),
        ("active_session_checked", False),
        ("last_saved_log_len", 0),
        ("next_scene_index", 0),
        ("last_saved_scene_index", -1),
        ("session_lost", False),
        ("scroll_to_top_on_train", False),
        ("pending_tab_click", None),
        ("last_force_speak_total", 0),
        ("survey_answers", {}),
        ("persisted_speed_level", None),
        ("persisted_student_count", None),
    ):
        if key not in st.session_state:
            st.session_state[key] = default
    if st.session_state.get("desired_students_count") is None:
        st.session_state["desired_students_count"] = _load_default_students_count()
    if st.session_state.get("persisted_student_count") is None:
        st.session_state["persisted_student_count"] = st.session_state["desired_students_count"]
    if "speech_speed_level" not in st.session_state:
        st.session_state["speech_speed_level"] = 4
    if st.session_state.get("persisted_speed_level") is None:
        st.session_state["persisted_speed_level"] = st.session_state["speech_speed_level"]
    if "speed_sync" not in st.session_state:
        st.session_state["speed_sync"] = None


def reset_case_state() -> None:
    keys = [
        "selected_case_id",
        "case_brief",
        "session_start",
        "pre_answers",
        "pre_score",
        "post_answers",
        "post_score",
        "pre_test_items",
        "post_test_items",
        "pbl_log",
        "agent_evaluation",
        "agent_advice",
        "agent_stats",
        "discussion_ran",
        "pbl_session_id",
        "session_saved",
        "pending_user_messages",
        "pbl_turn_requested",
        "student1_log_seen",
        "max_seen_scene",
        "total_scenes",
        "scene_transition_ready_at",
        "scene_transition_waiting",
        "last_transition_scene",
        "last_auto_pause_scene",
        "allow_user_input",
        "manual_pause_active",
        "summary_invite_scene",
        "advice_enhanced",
        "teacher_reply_pending",
        "awaiting_teacher_after_user",
        "last_teacher_seen_count",
        "scene_objective_keys",
        "advanced_scene_mask",
        "pre_question_correctness",
        "pending_active_session",
        "active_session_checked",
        "last_saved_log_len",
        "next_scene_index",
        "last_saved_scene_index",
        "session_lost",
        "scroll_to_top_on_train",
        "pending_tab_click",
        "last_force_speak_total",
    ]
    for key in keys:
        st.session_state.pop(key, None)
    preferred = st.session_state.get("persisted_student_count")
    if isinstance(preferred, int) and 4 <= preferred <= 8:
        st.session_state["desired_students_count"] = preferred
    else:
        default_count = _load_default_students_count()
        st.session_state["desired_students_count"] = default_count
        st.session_state["persisted_student_count"] = default_count
    st.session_state["next_scene_index"] = 0
    st.session_state["last_saved_scene_index"] = -1
    st.session_state["last_saved_log_len"] = 0


def _handle_logout() -> None:
    token = st.session_state.pop("session_token", None)
    if token:
        try:
            delete_session_token(token)
        except Exception:
            pass
    _clear_session_cookie()
    st.session_state.pop("user", None)
    st.session_state["speech_speed_level"] = 4
    st.session_state["persisted_speed_level"] = 4
    default_count = _load_default_students_count()
    st.session_state["desired_students_count"] = default_count
    st.session_state["persisted_student_count"] = default_count
    reset_case_state()
    st.session_state["page"] = "login"


def _prepare_test_items(raw_items: List[Dict[str, Any]], fallback_kind: str, prefix: str) -> List[Dict[str, Any]]:
    items = raw_items or default_test_items(fallback_kind)
    prepared: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        obj = dict(item)
        obj["qid"] = f"{prefix}_{idx}"
        prepared.append(obj)
    return prepared


def _sanitize_content(text: str) -> str:
    if not text:
        return ""
    text = text.replace("<CALL_TEACHER>", "").replace("<call_teacher>", "")

    def repl(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        return f"<{inner}>"

    cleaned = OBJ_TAG_PATTERN.sub(repl, text)
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"(<[^>]+>)", r"\1\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    def _should_drop_tail_line(line: str) -> bool:
        core = line.strip().strip("。．.!！")
        if not core:
            return True
        normalized = re.sub(r"\s+", "", core)
        if len(normalized) > 15:
            return False
        keywords = ("当前", "目标", "讨论", "完整", "完毕", "学习目标")
        hit_count = sum(1 for word in keywords if word in normalized)
        return hit_count >= 3

    lines = cleaned.strip().splitlines()
    while lines and _should_drop_tail_line(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _format_timestamp(value: str | None) -> str:
    if not value:
        return ""
    raw = value.strip()
    if not raw:
        return ""
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    beijing = timezone(timedelta(hours=8))
    local_dt = dt.astimezone(beijing)
    return f"{local_dt.strftime('%Y-%m-%d %H:%M:%S')}"


def _chat_role_and_avatar(speaker: str | None) -> tuple[str, Optional[str]]:
    """Map speaker labels to Streamlit chat roles and friendly avatars."""
    if not speaker:
        return "assistant", "💬"
    if speaker == "Student1":
        return "user", "🧒"
    if speaker.lower() == "teacher":
        return "assistant", "👩‍🏫"
    if speaker.lower().startswith("student"):
        return "assistant", "👥"
    return "assistant", "💬"


def _is_student_speaker(label: Optional[str]) -> bool:
    if not label:
        return False
    return str(label).lower().startswith("student")


def start_case(case_id: str) -> None:
    st.session_state["selected_case_id"] = case_id
    st.session_state["case_brief"] = case_service.to_brief(case_id)
    st.session_state["session_start"] = datetime.now(timezone.utc).isoformat()
    st.session_state["session_saved"] = False
    st.session_state["pre_answers"] = {}
    st.session_state["post_answers"] = {}
    st.session_state["pre_score"] = None
    st.session_state["post_score"] = None
    tests = case_service.fetch_tests(case_id)
    st.session_state["pre_test_items"] = _prepare_test_items(tests["pre"], "pre", "pre")
    st.session_state["post_test_items"] = _prepare_test_items(tests["post"], "post", "post")
    st.session_state["pbl_log"] = []
    st.session_state["agent_evaluation"] = None
    st.session_state["agent_advice"] = None
    st.session_state["agent_stats"] = None
    st.session_state["discussion_ran"] = None
    st.session_state["pbl_session_id"] = None
    st.session_state["pending_user_messages"] = []
    st.session_state["pbl_turn_requested"] = False
    st.session_state["student1_log_seen"] = 0
    st.session_state["max_seen_scene"] = -1
    st.session_state["total_scenes"] = case_service.scene_count(case_id)
    st.session_state["scene_transition_ready_at"] = 0.0
    st.session_state["scene_transition_waiting"] = False
    st.session_state["last_transition_scene"] = -1
    st.session_state["last_auto_pause_scene"] = -1
    st.session_state["allow_user_input"] = False
    st.session_state["manual_pause_active"] = False
    st.session_state["summary_invite_scene"] = -1
    st.session_state["advice_enhanced"] = False
    st.session_state["page"] = "pre_test"
    st.session_state["scene_objective_keys"] = case_service.scene_objective_layout(case_id)
    st.session_state["advanced_scene_mask"] = None
    st.session_state["pre_question_correctness"] = []
    st.session_state["last_saved_log_len"] = 0
    st.session_state["last_force_speak_total"] = 0


def _fetch_session_state() -> Dict[str, Any] | None:
    session_id = st.session_state.get("pbl_session_id")
    if not session_id:
        return None
    data = get_interactive_session_state(session_id)
    if data.get("status") == "not_found":
        st.session_state["pbl_session_id"] = None
        st.session_state["session_lost"] = True
        user = st.session_state.get("user")
        if user:
            try:
                record = load_active_session_for_user(user["id"])
            except Exception:
                record = None
            st.session_state["pending_active_session"] = record
        return None
    return data


def _trigger_user_turn(session_id: str | None, *, allow_input: bool = False) -> None:
    if not session_id:
        return
    request_user_turn(session_id)
    st.session_state["pbl_turn_requested"] = True
    st.session_state["allow_user_input"] = allow_input
    st.session_state["manual_pause_active"] = not allow_input


def _resume_discussion(session_id: str | None) -> None:
    if not session_id:
        return
    resume_user_turn(session_id)
    st.session_state["pbl_turn_requested"] = False
    st.session_state["scene_transition_ready_at"] = 0.0
    st.session_state["scene_transition_waiting"] = False
    st.session_state["allow_user_input"] = False
    st.session_state["manual_pause_active"] = False


def render_login() -> None:
    st.title(_tr("PBL 学习平台 - 登录", "PBL Learning Platform - Sign In"))
    username = st.text_input(_tr("账号", "Username"), key="login_username")
    password = st.text_input(_tr("密码", "Password"), type="password", key="login_password")
    if st.button(_tr("登录", "Sign In")):
        user = verify_user(username.strip(), password.strip()) if username and password else None
        if user:
            expires_at = datetime.now(timezone.utc) + timedelta(days=SESSION_COOKIE_DAYS)
            token = uuid.uuid4().hex
            try:
                create_session_token(user["id"], token, expires_at.isoformat())
                st.session_state["session_token"] = token
                _set_session_cookie(token, expires_at)
            except Exception as exc:
                st.warning(_tr(f"创建会话失败：{exc}", f"Failed to create session: {exc}"))
            st.session_state["user"] = user
            _apply_user_preferences(user["id"])
            st.session_state["page"] = "case_selection"
            st.success(_tr("登录成功！", "Signed in successfully!"))
            st.rerun()
        else:
            st.error(_tr("账号或密码错误，请重试。", "Invalid credentials, please try again."))


def render_case_selection() -> None:
    container = st.session_state.get("case_selection_container")
    if container is None:
        container = st.empty()
        st.session_state["case_selection_container"] = container
    with container.container():
        st.title(_tr("选择案例开始学习", "Choose a case to start"))
        user = st.session_state.get("user")
        if not user:
            st.warning(_tr("请先登录。", "Please sign in first."))
            st.session_state["page"] = "login"
            return

        saved_session = st.session_state.get("pending_active_session")
        if saved_session:
            metadata = saved_session.get("metadata") or {}
            next_idx_raw = metadata.get("next_scene_index", 0)
            try:
                next_idx = max(0, int(next_idx_raw))
            except Exception:
                next_idx = 0
            total_scenes_saved = metadata.get("total_scenes")
            if isinstance(total_scenes_saved, int) and total_scenes_saved > 0:
                display_scene = min(next_idx + 1, total_scenes_saved)
                scene_hint = _tr(
                    f"（已讨论至场景 {display_scene}/{total_scenes_saved}）",
                    f"(Progress: scene {display_scene}/{total_scenes_saved})",
                )
            else:
                display_scene = next_idx + 1
                scene_hint = _tr(
                    f"（已讨论至场景 {display_scene}）",
                    f"(Progress: scene {display_scene})",
                ) if next_idx else ""
            st.info(
                _tr(
                    f"检测到未完成的讨论（案例 {saved_session.get('case_id')}）{scene_hint}，可继续或放弃。",
                    f"Found an unfinished discussion (case {saved_session.get('case_id')}) {scene_hint}. You may resume or discard it.",
                )
            )
            resume_col, discard_col = st.columns(2)
            with resume_col:
                if st.button(_tr("继续上次讨论", "Resume discussion"), key="resume_saved_session"):
                    _resume_saved_session(saved_session)
            with discard_col:
                if st.button(_tr("放弃该讨论", "Discard"), key="discard_saved_session"):
                    _clear_active_session_record()
                    st.success(_tr("已放弃保存的讨论。", "Saved discussion discarded."))
                    st.rerun()
                    st.stop()

        history = fetch_user_history(user_id=user["id"])
        department_options = sorted({dept for dept in case_service.departments.values()})
        dept_placeholder = "department_choice"
        preferred_dept = st.session_state.get(dept_placeholder)
        if not history:
            selected_dept = st.selectbox(
                _tr("请选择你感兴趣的科室以开始学习", "Choose a department to start"),
                ["不限科室"] + department_options,
                index=0 if not preferred_dept else (["不限科室"] + department_options).index(preferred_dept),
                key="department_select_box",
            )
            preferred_dept = None if selected_dept == "不限科室" else selected_dept
            st.session_state[dept_placeholder] = preferred_dept
        else:
            preferred_dept = None
        recommended = case_service.recommend(history, top_k=3, department_filter=preferred_dept)
        st.subheader(_tr("推荐案例", "Recommended cases"))
        rec_cols = st.columns(len(recommended) or 1)
        for col, brief in zip(rec_cols, recommended):
            with col:
                st.markdown(f"**{brief.case_id}**")
                st.caption(brief.title[:80])
                st.write(brief.summary[:120] + ("..." if len(brief.summary) > 120 else ""))
                if st.button(_tr("选择此案例", "Select this case"), key=f"rec_{brief.case_id}"):
                    start_case(brief.case_id)
                    st.rerun()

        st.subheader(_tr("全部案例列表", "All cases"))
        others = case_service.remaining_cases([b.case_id for b in recommended])
        options = [b.case_id for b in others]
        case_map = {b.case_id: b for b in others}
        selected = st.selectbox(
            _tr("或从列表中选择其他案例", "Or choose another case from the list"),
            options,
            format_func=lambda cid: f"{cid} - {case_map[cid].title[:40]}",
            key="manual_case_select",
        ) if options else None
        if selected and st.button(_tr("选择此案例", "Start this case"), key="start_manual_case"):
            start_case(selected)
            st.rerun()

        st.subheader(_tr("最近学习记录", "Recent sessions"))
        if history:
            history_rows = [
                {
                    _tr("案例", "Case"): item["case_id"],
                    _tr("开始", "Start"): _format_timestamp(item["started_at"]),
                    _tr("结束", "End"): _format_timestamp(item["ended_at"]),
                    _tr("前测", "Pre-test"): item["pre_score"],
                    _tr("后测", "Post-test"): item["post_score"],
                    _tr("综合", "Overall"): item.get("composite_score"),
                }
                for item in history
            ]
            st.table(history_rows)
        else:
            st.info(_tr("暂无历史记录，选择任意案例开始吧！", "No history yet. Pick any case to begin!"))


def render_test_page(kind: str) -> None:
    brief = st.session_state.get("case_brief")
    if not brief:
        st.session_state["page"] = "case_selection"
        st.rerun()
        return
    st.title(_tr(f"{'前' if kind == 'pre' else '后'}测验 - {brief.case_id}", f"{'Pre' if kind == 'pre' else 'Post'} test - {brief.case_id}"))
    test_key = "pre_test_items" if kind == "pre" else "post_test_items"
    tests = st.session_state.get(test_key) or default_test_items(kind)
    answers_key = "pre_answers" if kind == "pre" else "post_answers"
    answers = st.session_state.get(answers_key, {})
    for idx, item in enumerate(tests, start=1):
        qid = item.get("qid") or f"{kind}_{idx}"
        question_text = item.get("question", f"题目{idx}")
        st.markdown(_tr(f"**题目{idx}：{question_text}**", f"**Question {idx}: {question_text}**"))
        st.caption(_tr(f"{item.get('question_type', '题型未知')} | {item.get('exam_subject', '')}", f"{item.get('question_type', 'Unknown type')} | {item.get('exam_subject', '')}"))
        options = item.get("option")
        question_type = str(item.get("question_type", "")).lower()
        is_multi = "多" in question_type
        if isinstance(options, dict) and options:
            option_keys = list(options.keys())
            format_func = lambda opt: f"{opt}. {options[opt]}"
            if is_multi:
                existing = answers.get(qid, [])
                selected_set = set(existing if isinstance(existing, list) else [])
                new_selected: List[str] = []
                for opt in option_keys:
                    checked = st.checkbox(
                        format_func(opt),
                        value=opt in selected_set,
                        key=f"{qid}_{opt}_checkbox",
                    )
                    if checked:
                        new_selected.append(opt)
                answers[qid] = new_selected
            else:
                default_val = answers.get(qid)
                option_entries = [_tr("未选择", "Not selected")] + option_keys
                if default_val in option_keys:
                    default_entry = option_entries.index(default_val)
                else:
                    default_entry = 0
                selection = st.radio(
                    _tr(f"选择题目{idx}", f"Choose option for Q{idx}"),
                    option_entries,
                    index=default_entry,
                    format_func=lambda opt: format_func(opt) if opt in option_keys else opt,
                    key=f"{qid}_radio",
                )
                answers[qid] = selection if selection in option_keys else ""
        else:
            response = st.text_input(
                _tr(f"你的答案（题目{idx}）", f"Your answer (Q{idx})"),
                value=answers.get(qid, ""),
                key=f"{qid}_response",
            )
            answers[qid] = response
    st.session_state[answers_key] = answers
    if st.button(_tr("提交测验", "Submit"), key=f"submit_{kind}_test"):
        tests_to_grade = st.session_state.get(test_key) or []
        missing = []
        for idx, item in enumerate(tests_to_grade, start=1):
            qid = item.get("qid") or f"{kind}_{idx}"
            options = item.get("option")
            question_type = str(item.get("question_type", "")).lower()
            is_multi = isinstance(options, dict) and "多" in question_type
            if isinstance(options, dict) and options:
                value = answers.get(qid)
                if is_multi:
                    if not value or not isinstance(value, list) or not [opt for opt in value if opt in options]:
                        missing.append(idx)
                else:
                    if value not in options:
                        missing.append(idx)
        if missing:
            st.warning(_tr(f"还有 {len(missing)} 道选择题未作答（题号：{', '.join(map(str, missing))}），请补充后再提交。", f"{len(missing)} multiple-choice questions are unanswered (Q: {', '.join(map(str, missing))}). Please complete them."))
            return
        score = score_test_items(tests_to_grade, answers)
        if kind == "pre":
            st.session_state["pre_score"] = score
            correctness = [
                _evaluate_question_correct(item, answers.get(item.get("qid") or f"{kind}_{idx}"))
                for idx, item in enumerate(tests_to_grade, start=1)
            ]
            st.session_state["pre_question_correctness"] = correctness
            if st.session_state.get("scene_objective_keys"):
                st.session_state["advanced_scene_mask"] = _compute_scene_advanced_mask(correctness)
            st.session_state["page"] = "pbl_training"
        else:
            st.session_state["post_score"] = score
            st.session_state["page"] = "evaluation"
        st.rerun()


def render_survey_page() -> None:
    lang = st.session_state.get("app_language") or _load_app_language()
    st.title(_tr("PBL 讨论体验调查问卷", "PBL Discussion Experience Survey"))
    st.write(
        _tr(
            "请结合你在本系统中的学习体验，使用 5 分量表为以下陈述打分。结果仅用于改进产品，不会影响你的成绩。",
            "Rate the statements below on a 5-point scale based on your experience. This is for improvement only and will not affect your score.",
        )
    )
    st.info(_tr("量表含义：非常同意 ＞ 比较同意 ＞ 不确定 ＞ 比较不同意 ＞ 非常不同意。", "Scale: Strongly agree > Somewhat agree > Neutral > Somewhat disagree > Strongly disagree."))
    responses = st.session_state.setdefault("survey_answers", {})
    questions = SURVEY_QUESTIONS_EN if lang == "en" else SURVEY_QUESTIONS
    choices = SURVEY_CHOICES_EN if lang == "en" else SURVEY_CHOICES
    for idx, question in enumerate(questions, start=1):
        qid = f"survey_q_{idx}"
        default_choice = responses.get(qid, choices[2])
        st.markdown(f"**Q{idx}. {question}**")
        choice = st.radio(
            "",
            choices,
            index=choices.index(default_choice),
            key=f"survey_radio_{idx}",
            horizontal=True,
            label_visibility="collapsed",
        )
        responses[qid] = choice
        if idx < len(questions):
            st.divider()
    user = st.session_state.get("user")
    case_id = st.session_state.get("selected_case_id")
    if st.button(_tr("提交问卷", "Submit survey"), key="submit_survey", use_container_width=True):
        try:
            record_survey_response(
                user_id=user["id"] if user else None,
                case_id=case_id,
                answers=responses,
            )
            st.success(_tr("感谢填写，问卷已成功提交！", "Thank you! Survey submitted."))
        except Exception as exc:
            st.error(_tr(f"提交问卷失败：{exc}", f"Failed to submit survey: {exc}"))
    if st.button(_tr("返回推荐页面", "Back to recommendation page"), use_container_width=True):
        st.session_state["page"] = "case_selection"
        st.rerun()


def render_pbl_training() -> None:
    container = st.session_state.get("pbl_training_container")
    if container is None:
        container = st.empty()
        st.session_state["pbl_training_container"] = container
    with container.container():
        _render_pbl_training_inner()


def _render_pbl_training_inner() -> None:
    brief = st.session_state.get("case_brief")
    if not brief:
        st.session_state["page"] = "case_selection"
        st.rerun()
        return
    session_id = st.session_state.get("pbl_session_id")
    _sync_speed_to_session(session_id)
    session_state_data = _fetch_session_state()
    status = session_state_data.get("status") if session_state_data else "idle"
    waiting_for_user = session_state_data.get("waiting_for_user") if session_state_data else False
    session_lost = st.session_state.get("session_lost", False)
    st.title(_tr("PBL 讨论训练", "PBL Discussion"))
    st.markdown(
        _tr(
            "点击下方按钮即可与虚拟学习小组实时讨论。系统会按顺序推送同伴发言，你可以随时发表观点！每个场景结束后系统会自动保存当前进度。",
            "Use the buttons below to start real-time discussion with virtual peers. Messages will flow in order; you can speak anytime. Progress is auto-saved after each scene.",
        )
    )
    if st.session_state.pop("scroll_to_top_on_train", False):
        components.html("""
            <script>
            setTimeout(function() {
                window.scrollTo({top: 0, left: 0, behavior: 'auto'});
            }, 150);
            </script>
        """, height=0)
    if session_lost and not session_state_data:
        st.warning(_tr("检测到实时会话已断开，保存的讨论进度仍保留在右侧案例选项中，可返回案例选择页继续。", "Live session lost. Saved progress is available on the case selection page to resume."))
        go_back = st.button(_tr("返回案例选择并恢复讨论", "Back to cases and resume"), key="return_to_cases_from_loss")
        if go_back:
            st.session_state["page"] = "case_selection"
            st.rerun()
    discussion_ran = bool(st.session_state.get("discussion_ran"))
    max_seen_scene = st.session_state.get("max_seen_scene", -1)
    total_scenes_configured = st.session_state.get("total_scenes") or 1
    can_skip_to_post = bool(session_id) and (
        discussion_ran or max_seen_scene >= max(total_scenes_configured - 1, 0)
    )
    controls_col, skip_col = st.columns([1, 1], gap="small")
    with controls_col:
        start_disabled = status == "running" or bool(session_id)
        button_label = _tr("讨论进行中...", "Discussion running...") if session_id else _tr("开始 PBL 讨论", "Start PBL discussion")
        if st.button(button_label, key="run_pbl_agents", disabled=start_disabled):
            case_id = st.session_state.get("selected_case_id")
            if not case_id:
                st.error(_tr("请先选择案例。", "Please choose a case first."))
            else:
                speed_factor = _current_speed_factor()
                ability_window = _ability_window_from_pre_score(st.session_state.get("pre_score"))
                _clear_active_session_record()
                current_user = st.session_state.get("user") or {}
                session_id = start_interactive_session(
                    case_id,
                    speed_factor=speed_factor,
                    owner_user_id=current_user.get("id"),
                    owner_username=current_user.get("username"),
                    ability_window=ability_window,
                    students_count=int(st.session_state.get("desired_students_count", 5)),
                    advanced_mask=st.session_state.get("advanced_scene_mask"),
                )
                st.session_state["pbl_session_id"] = session_id
                st.session_state["speed_sync"] = {
                    "session_id": session_id,
                    "level": _current_speed_level(),
                }
                st.session_state["session_lost"] = False
                st.session_state["scroll_to_top_on_train"] = True
                total_scenes = st.session_state.get("total_scenes") or len(st.session_state.get("scene_objective_keys") or [])
                st.session_state["pending_tab_click"] = 0 if total_scenes <= 1 else 0
                st.session_state["scene_transition_ready_at"] = 0.0
                st.session_state["scene_transition_waiting"] = False
                st.session_state["last_transition_scene"] = -1
                st.session_state["discussion_ran"] = False
                st.session_state["agent_evaluation"] = None
                st.session_state["agent_advice"] = None
                st.session_state["agent_stats"] = None
                st.rerun()
    with skip_col:
        skip_disabled = not can_skip_to_post
        if st.button(
            _tr("结束 PBL 训练，进入后测验", "End discussion and go to post-test"),
            key="to_post_test",
            disabled=skip_disabled,
            use_container_width=True,
        ):
            st.session_state["page"] = "post_test"
            st.rerun()

    if status == "running":
        st.info(_tr("学习小组正在协作，请稍候...", "The study group is working, please wait..."))
    if status == "error":
        st.error(session_state_data.get("error", _tr("讨论失败", "Discussion failed")) if session_state_data else _tr("讨论失败", "Discussion failed"))

    if session_state_data:
        st.session_state["pbl_log"] = session_state_data.get("log", [])
        if status == "completed" and not discussion_ran:
            st.session_state["discussion_ran"] = True
            st.session_state["agent_evaluation"] = session_state_data.get("evaluation")
            st.session_state["agent_advice"] = session_state_data.get("advice")
            st.session_state["agent_stats"] = session_state_data.get("student_stats")
            _clear_active_session_record()

    log = st.session_state.get("pbl_log", [])
    total_student_messages = sum(
        1 for entry in log if _is_student_speaker(entry.get("speaker"))
    )
    teacher_message_count = sum(1 for entry in log if entry.get("speaker") == "Teacher")
    last_teacher_seen = st.session_state.get("last_teacher_seen_count", 0)
    if teacher_message_count > last_teacher_seen:
        st.session_state["teacher_reply_pending"] = False
        st.session_state["awaiting_teacher_after_user"] = False
        st.session_state["last_teacher_seen_count"] = teacher_message_count
    elif not waiting_for_user:
        st.session_state["teacher_reply_pending"] = False
    scene_groups: Dict[int, List[Dict[str, Any]]] = {}
    raw_max_seen = -1
    for entry in log:
        idx = entry.get("scene_index")
        if not isinstance(idx, int):
            idx = 0
        scene_groups.setdefault(idx, []).append(entry)
        if idx > raw_max_seen:
            raw_max_seen = idx
    prev_max_seen = st.session_state.get("max_seen_scene", -1)
    if raw_max_seen > prev_max_seen:
        st.session_state["max_seen_scene"] = raw_max_seen
        auto_scene = raw_max_seen
        last_auto_pause_scene = st.session_state.get("last_auto_pause_scene", -1)
        if auto_scene > last_auto_pause_scene and prev_max_seen >= 0:
            _trigger_user_turn(session_id)
            st.session_state["last_auto_pause_scene"] = auto_scene
    effective_max_seen = max(raw_max_seen, 0)
    pending_queue = list(st.session_state.get("pending_user_messages", []))
    prev_seen = st.session_state.get("student1_log_seen", 0)
    current_seen = sum(1 for entry in log if entry.get("speaker") == "Student1")
    new_seen = max(0, current_seen - prev_seen)
    while new_seen > 0 and pending_queue:
        pending_queue.pop(0)
        new_seen -= 1
    st.session_state["student1_log_seen"] = current_seen
    students_total_raw = st.session_state.get("desired_students_count")
    try:
        students_total = int(students_total_raw)
    except (TypeError, ValueError):
        students_total = _load_default_students_count()
    students_total = max(1, students_total)
    last_forced_total = st.session_state.get("last_force_speak_total", 0)
    need_force = (
        session_id
        and total_student_messages > (2 * students_total)
        and current_seen < (total_student_messages / (2.0 * students_total))
        and not waiting_for_user
    )
    if need_force and total_student_messages > last_forced_total:
        _trigger_user_turn(session_id, allow_input=True)
        st.session_state["last_force_speak_total"] = total_student_messages
    if not session_id:
        st.session_state["pbl_turn_requested"] = False
    elif waiting_for_user:
        st.session_state["pbl_turn_requested"] = False
        changed = False
        for entry in pending_queue:
            if not entry.get("delivered"):
                submit_user_message(session_id, entry["content"])
                entry["delivered"] = True
                changed = True
        if changed:
            st.session_state["pending_user_messages"] = pending_queue
    else:
        st.session_state["pending_user_messages"] = pending_queue
    summary_invite = False
    current_stage_label: str | None = None
    current_stage_scene = None
    if log:
        last_entry = log[-1]
        current_stage_label = last_entry.get("stage")
        current_stage_scene = last_entry.get("scene_index")
        if last_entry.get("speaker") == "Teacher":
            raw_text = last_entry.get("content") or ""
            if "Student1" in raw_text and ("总结" in raw_text or "反思" in raw_text):
                summary_invite = True

    if summary_invite and session_id and current_stage_scene is not None:
        tracked_scene = st.session_state.get("summary_invite_scene", -1)
        if current_stage_scene != tracked_scene:
            _trigger_user_turn(session_id, allow_input=True)
            st.session_state["summary_invite_scene"] = current_stage_scene

    total_scenes = st.session_state.get("total_scenes") or max(effective_max_seen + 1, 1)
    latest_completed_scene = -1
    for idx, entries in scene_groups.items():
        if not entries:
            continue
        last_entry = entries[-1]
        if last_entry.get("stage") == "end" and (last_entry.get("speaker") or "").lower() == "teacher":
            latest_completed_scene = max(latest_completed_scene, idx)
    next_scene_idx = min(max(latest_completed_scene + 1, 0), total_scenes)
    st.session_state["next_scene_index"] = next_scene_idx
    prev_saved_scene = st.session_state.get("last_saved_scene_index", -1)
    if latest_completed_scene >= 0 and latest_completed_scene > prev_saved_scene:
        st.session_state["last_saved_scene_index"] = latest_completed_scene
        if next_scene_idx < total_scenes:
            _save_active_session_state(force=True, next_scene_index=next_scene_idx)
        else:
            _clear_active_session_record()
    last_transition_scene = st.session_state.get("last_transition_scene", -1)
    transition_ready_at = st.session_state.get("scene_transition_ready_at", 0.0)
    transition_waiting = st.session_state.get("scene_transition_waiting", False)
    now_ts = time.time()
    if (
        latest_completed_scene > last_transition_scene
        and latest_completed_scene >= 0
        and latest_completed_scene < total_scenes - 1
    ):
        transition_ready_at = now_ts + 9.0
        st.session_state["scene_transition_ready_at"] = transition_ready_at
        st.session_state["scene_transition_waiting"] = True
        st.session_state["last_transition_scene"] = latest_completed_scene
        transition_waiting = True
    waiting_next_scene = transition_waiting
    if transition_waiting and now_ts >= transition_ready_at:
        waiting_next_scene = False
        transition_waiting = False
        st.session_state["scene_transition_waiting"] = False
    cooldown_remaining = max(0.0, transition_ready_at - now_ts) if waiting_next_scene else 0.0
    last_scene_entries = scene_groups.get(total_scenes - 1, [])
    all_scenes_completed = bool(
        last_scene_entries
        and last_scene_entries[-1].get("stage") == "end"
        and (last_scene_entries[-1].get("speaker") or "").lower() == "teacher"
    )

    teacher_started = any(entry.get("speaker") == "Teacher" for entry in log)
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
            display: flex !important;
            flex-wrap: nowrap;
            width: 100%;
            max-width: calc(100% - 3rem);
            margin: 0 auto;
            padding: 0 1.5rem;
            justify-content: space-evenly;
            gap: 1.5rem;
            box-sizing: border-box;
        }
        div[data-testid="stTabs"] div[role="tab"] {
            flex: 1 1 0;
            justify-content: center;
            max-width: none;
            font-size: 1.1rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    tab_labels = [_tr(f"场景 {idx + 1}", f"Scene {idx + 1}") for idx in range(total_scenes)]
    tabs = st.tabs(tab_labels)
    for idx, tab in enumerate(tabs):
        with tab:
            scene_entries = list(scene_groups.get(idx, []))
            if scene_entries:
                st.subheader(_tr("讨论记录", "Discussion log"))
                for item in scene_entries:
                    speaker = item.get("speaker", "Unknown")
                    label = _tr("Student1（你）", "Student1 (You)") if speaker == "Student1" else (speaker or "Unknown")
                    content = _sanitize_content(item.get("content", "")) or _tr("（无内容）", "(No content)")
                    timestamp = item.get("timestamp")
                    role, avatar = _chat_role_and_avatar(speaker)
                    chat_kwargs = {"avatar": avatar} if avatar else {}
                    with st.chat_message(role, **chat_kwargs):
                        caption_label = label
                        if speaker and speaker.lower().startswith("student"):
                            caption_label = f"**{label}**"
                        elif speaker and speaker.lower() == "teacher":
                            caption_label = f"**{label}**"
                        formatted_ts = _format_timestamp(timestamp)
                        if formatted_ts:
                            st.caption(f"{caption_label} · {formatted_ts}")
                        else:
                            st.caption(caption_label)
                        st.markdown(content)
            else:
                st.info(_tr("尚未生成讨论记录。", "No discussion log yet."))

    pending_tab = st.session_state.pop("pending_tab_click", None)
    if pending_tab is not None and 0 <= pending_tab < len(tab_labels):
        components.html(
            f"""
            <script>
            setTimeout(function() {{
                const tabs = window.parent.document.querySelectorAll('div[data-testid="stTabs"] div[role="tab"]');
                if (tabs[{pending_tab}]) {{
                    tabs[{pending_tab}].click();
                    tabs[{pending_tab}].scrollIntoView({{behavior: 'auto', block: 'start'}});
                }}
            }}, 200);
            </script>
            """,
            height=0,
        )

    chat_placeholder: str
    allow_input = st.session_state.get("allow_user_input", False)
    manual_pause_active = st.session_state.get("manual_pause_active", False)
    teacher_reply_pending_flag = st.session_state.get("teacher_reply_pending", False)
    awaiting_teacher_flag = st.session_state.get("awaiting_teacher_after_user", False)
    if (
        session_id
        and waiting_for_user
        and not manual_pause_active
        and not allow_input
    ):
        # 自动响应 <CALL_TEACHER> 场景，立即允许用户发言，避免出现“暂停”提示
        allow_input = True
        st.session_state["allow_user_input"] = True
        last_speaker = log[-1].get("speaker") if log else None
        if last_speaker != "Teacher":
            st.session_state["teacher_reply_pending"] = True
    if session_id and waiting_for_user:
        if waiting_next_scene or manual_pause_active or not allow_input:
            chat_placeholder = _tr("讨论已暂停，请点击“继续讨论”后再发言", "Discussion paused. Click 'Resume discussion' to speak.")
        elif teacher_reply_pending_flag or awaiting_teacher_flag:
            chat_placeholder = _tr("教师评估讨论中，请稍等~", "Teacher is reviewing. Please wait.")
        else:
            chat_placeholder = _tr("输入你的观点、总结或疑问…", "Enter your opinion, summary, or question…")
    elif session_id:
        chat_placeholder = _tr("请点击“我要发言”按钮开始发言！", "Click 'I want to speak' to start talking!")
    else:
        chat_placeholder = _tr("开始讨论后即可发言", "You can speak after starting the discussion")

    resume_button_show = False
    resume_button_disabled = False
    if not session_id:
        st.info(_tr("尚未开始 PBL 讨论。", "PBL discussion not started yet."))
    elif all_scenes_completed:
        st.success(_tr("全部场景讨论已结束，请完成后测验。", "All scenes finished. Please complete the post-test."))
    elif waiting_next_scene:
        current_scene_number = (latest_completed_scene + 1) if latest_completed_scene >= 0 else 1
        next_scene_number = min(current_scene_number + 1, total_scenes)
        st.info(
            _tr(
                f"场景{current_scene_number}结束，请切换至场景{next_scene_number}继续讨论！",
                f"Scene {current_scene_number} ended. Switch to scene {next_scene_number} to continue.",
            )
        )
        resume_button_disabled = cooldown_remaining > 0
        resume_button_show = True
    elif waiting_for_user:
        if manual_pause_active or (not allow_input):
            st.info(_tr("讨论暂停中，请点击“继续讨论”恢复。", "Discussion paused. Click 'Resume discussion' to continue."))
            resume_button_show = True
            resume_button_disabled = False
        else:
            if teacher_reply_pending_flag or awaiting_teacher_flag:
                st.info(_tr("教师评估讨论中，请稍等~", "Teacher is reviewing. Please wait."))
            else:
                st.success(_tr("轮到你发言了，请通过下方输入框发送观点。", "It's your turn. Share your thoughts below."))
    else:
        if summary_invite:
            st.info(_tr("老师正在等你进行总结反思，请点击“我要发言”开始总结。", "Teacher is waiting for your summary. Click 'I want to speak' to start."))
        pending_controls = teacher_reply_pending_flag or awaiting_teacher_flag
        disable_request = (status == "running" and not teacher_started) or pending_controls
        if pending_controls:
            st.info(_tr("教师评估讨论中，请稍等~", "Teacher is reviewing. Please wait."))
        else:
            speak_col, pause_col = st.columns([1, 1], gap="small")
            with speak_col:
                if st.button(
                    _tr("我要发言", "I want to speak"),
                    key="request_speak",
                    disabled=disable_request,
                    use_container_width=True,
                ):
                    _trigger_user_turn(session_id, allow_input=True)
                    st.rerun()
            with pause_col:
                if st.button(
                    _tr("暂停讨论", "Pause discussion"),
                    key="pause_discussion",
                    disabled=disable_request,
                    use_container_width=True,
                ):
                    _trigger_user_turn(session_id, allow_input=False)
                    st.session_state["summary_invite_scene"] = -1
                    st.rerun()

    resume_button_placeholder = st.empty()
    chat_disabled = not (
        session_id
        and waiting_for_user
        and allow_input
        and not waiting_next_scene
        and not manual_pause_active
    )
    if not chat_disabled and (teacher_reply_pending_flag or awaiting_teacher_flag):
        chat_disabled = True
    user_chat_input = st.chat_input(chat_placeholder, key="pbl_chat_input", disabled=chat_disabled)
    if (
        session_id
        and waiting_for_user
        and allow_input
        and not waiting_next_scene
        and not manual_pause_active
        and user_chat_input is not None
    ):
        message_text = user_chat_input.strip() or _tr("（学生暂不发言）", "(No input)")
        normalized = message_text.rstrip()
        lower_norm = normalized.lower()
        if current_stage_label == "summary" and "<call_teacher>" not in lower_norm:
            normalized = f"{normalized}\n\n<CALL_TEACHER>"
        entries = st.session_state.get("pending_user_messages", [])
        entries.append({
            "content": normalized,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        st.session_state["pending_user_messages"] = entries
        st.session_state["allow_user_input"] = False
        st.session_state["manual_pause_active"] = False
        should_wait_teacher = "<call_teacher>" in normalized.lower()
        st.session_state["awaiting_teacher_after_user"] = should_wait_teacher
        submit_user_message(session_id, normalized)
        st.rerun()
    elif resume_button_show:
        with resume_button_placeholder:
            if st.button(
                _tr("继续讨论", "Resume discussion"),
                key="resume_discussion",
                use_container_width=True,
                disabled=resume_button_disabled,
            ):
                _resume_discussion(session_id)
                st.rerun()
    if session_id and (
        (status == "running" and not waiting_for_user)
        or waiting_next_scene
        or waiting_for_user
    ):
        forced_teacher_reply = (
            waiting_for_user
            and not waiting_next_scene
            and not st.session_state.get("manual_pause_active", False)
            and not st.session_state.get("allow_user_input", False)
        )
        if status == "running" and not waiting_for_user:
            interval = 0.5
        elif waiting_next_scene:
            interval = 3.0
        else:
            interval = 0.5 if (waiting_for_user or forced_teacher_reply) else 1.5
        time.sleep(interval)
        st.rerun()


def render_evaluation_page() -> None:
    case_id = st.session_state.get("selected_case_id")
    if not case_id:
        st.session_state["page"] = "case_selection"
        st.rerun()
        return
    pre_score = st.session_state.get("pre_score") or 0.0
    post_score = st.session_state.get("post_score") or 0.0
    evaluation = st.session_state.get("agent_evaluation")
    advice = st.session_state.get("agent_advice")
    log = st.session_state.get("pbl_log", [])
    session_id = st.session_state.get("pbl_session_id")
    if (not evaluation or not advice) and session_id:
        data = get_interactive_session_state(session_id)
        if data and data.get("status") == "completed":
            eval_payload = data.get("evaluation")
            adv_payload = data.get("advice")
            if eval_payload:
                st.session_state["agent_evaluation"] = eval_payload
                evaluation = eval_payload
            if adv_payload:
                st.session_state["agent_advice"] = adv_payload
                advice = adv_payload
    st.session_state["evaluation_result"] = evaluation
    st.session_state["advice_result"] = advice

    st.title(_tr("自动化评估", "Automated Evaluation"))
    st.subheader(_tr("测验成绩概览", "Test Score Overview"))
    delta_score = post_score - pre_score
    col1, col2, col3 = st.columns(3)
    col1.metric(_tr("前测得分", "Pre-test"), f"{pre_score:.0f}")
    col2.metric(_tr("后测得分", "Post-test"), f"{post_score:.0f}", f"{delta_score:+.0f}")
    col3.metric(_tr("分数变化", "Score change"), f"{delta_score:+.0f}")

    def _normalize_choice_codes(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = [str(v).strip().upper() for v in value if str(v).strip()]
            return sorted(dict.fromkeys(items))
        letters = re.findall(r"[A-Z]", str(value).upper())
        return sorted(dict.fromkeys(letters))

    def _is_answer_correct(correct_val: Any, user_raw: Any) -> Optional[bool]:
        expected = _normalize_choice_codes(correct_val)
        if not expected:
            return None
        user_codes = _normalize_choice_codes(user_raw)
        if not user_codes:
            return False
        return user_codes == expected

    st.subheader(_tr("测验参考答案", "Test answers"))
    pre_tests = st.session_state.get("pre_test_items") or default_test_items("pre")
    post_tests = st.session_state.get("post_test_items") or default_test_items("post")
    pre_answers = st.session_state.get("pre_answers", {})
    post_answers = st.session_state.get("post_answers", {})
    pre_report_lines: List[str] = []
    post_report_lines: List[str] = []
    with st.expander(_tr("前测答疑", "Pre-test review"), expanded=False):
        if not pre_tests:
            st.write(_tr("暂无前测题目。", "No pre-test questions."))
            pre_report_lines.append(_tr("前测：无题目。", "Pre-test: no questions."))
        else:
            pre_report_lines.append(_tr("前测：", "Pre-test:"))
            for idx, item in enumerate(pre_tests, start=1):
                correct = item.get("answer", _tr("（参考答案待补充）", "(Reference answer TBD)"))
                qid = item.get("qid") or f"pre_{idx}"
                raw_val = pre_answers.get(qid)
                if isinstance(raw_val, list):
                    user_val = "、".join(raw_val) if raw_val else _tr("（未作答）", "(Not answered)")
                else:
                    user_val = raw_val or _tr("（未作答）", "(Not answered)")
                correct_flag = _is_answer_correct(correct, raw_val)
                prefix_icon = "✅" if correct_flag else ("❌" if correct_flag is False else "ℹ️")
                status_text = _tr("正确", "Correct") if correct_flag else (_tr("错误", "Incorrect") if correct_flag is False else _tr("未判定", "Not judged"))
                st.markdown(_tr(f"{prefix_icon} **题目{idx}：{item.get('question', '')}**", f"{prefix_icon} **Q{idx}: {item.get('question', '')}**"))
                options = item.get("option") or {}
                if isinstance(options, dict) and options:
                    for key in sorted(options.keys()):
                        st.markdown(f"- {key}. {options[key]}")
                st.markdown(_tr(f"- 你的答案：{user_val}", f"- Your answer: {user_val}"))
                st.markdown(_tr(f"- 参考答案：{correct}", f"- Correct answer: {correct}"))
                pre_report_lines.append(
                    f"  - 题目{idx}（{status_text}）：{item.get('question','')} | 你的答案：{user_val} | 参考：{correct}"
                )
    with st.expander(_tr("后测答疑", "Post-test review"), expanded=False):
        if not post_tests:
            st.write(_tr("暂无后测题目。", "No post-test questions."))
            post_report_lines.append(_tr("后测：无题目。", "Post-test: no questions."))
        else:
            post_report_lines.append(_tr("后测：", "Post-test:"))
            for idx, item in enumerate(post_tests, start=1):
                correct = item.get("answer", _tr("（参考答案待补充）", "(Reference answer TBD)"))
                qid = item.get("qid") or f"post_{idx}"
                raw_val = post_answers.get(qid)
                if isinstance(raw_val, list):
                    user_val = "、".join(raw_val) if raw_val else _tr("（未作答）", "(Not answered)")
                else:
                    user_val = raw_val or _tr("（未作答）", "(Not answered)")
                correct_flag = _is_answer_correct(correct, raw_val)
                prefix_icon = "✅" if correct_flag else ("❌" if correct_flag is False else "ℹ️")
                status_text = _tr("正确", "Correct") if correct_flag else (_tr("错误", "Incorrect") if correct_flag is False else _tr("未判定", "Not judged"))
                st.markdown(_tr(f"{prefix_icon} **题目{idx}：{item.get('question', '')}**", f"{prefix_icon} **Q{idx}: {item.get('question', '')}**"))
                options = item.get("option") or {}
                if isinstance(options, dict) and options:
                    for key in sorted(options.keys()):
                        st.markdown(f"- {key}. {options[key]}")
                st.markdown(_tr(f"- 你的答案：{user_val}", f"- Your answer: {user_val}"))
                st.markdown(_tr(f"- 参考答案：{correct}", f"- Correct answer: {correct}"))
                post_report_lines.append(
                    f"  - 题目{idx}（{status_text}）：{item.get('question','')} | 你的答案：{user_val} | 参考：{correct}"
                )
    if not pre_report_lines:
        pre_report_lines.append(_tr("前测：无数据。", "Pre-test: no data."))
    if not post_report_lines:
        post_report_lines.append(_tr("后测：无数据。", "Post-test: no data."))
    test_report_text = "\n".join(pre_report_lines + [""] + post_report_lines)
    advice_ready = advice is not None
    if (
        not advice_ready
        and not st.session_state.get("advice_enhanced", False)
        and session_id
        and evaluation
        and (st.session_state.get("pre_score") is not None)
        and (st.session_state.get("post_score") is not None)
    ):
        try:
            refreshed = refresh_advice_with_tests(
                session_id,
                float(st.session_state["pre_score"]),
                float(st.session_state["post_score"]),
                test_report_text,
            )
            if refreshed:
                advice = refreshed
                st.session_state["agent_advice"] = refreshed
                st.session_state["advice_result"] = refreshed
                st.session_state["advice_enhanced"] = True
                advice_ready = True
        except Exception as exc:
            st.warning(_tr(f"更新学习建议失败：{exc}", f"Failed to refresh advice: {exc}"))
    if not evaluation or not isinstance(evaluation.get("dimensions"), dict):
        st.info(_tr("讨论评估尚未完成，请稍候…", "Evaluation not ready yet, please wait…"))
        time.sleep(2.0)
        st.rerun()
        return

    st.subheader(_tr("PBL讨论各维度得分", "PBL Dimension Scores"))
    for dim in evaluation["dimensions"].values():
        st.markdown(_tr(f"- **{dim['title']}**：{dim['score']} 分（{dim['justification']}）", f"- **{dim['title']}**: {dim['score']} ({dim['justification']})"))

    st.subheader(_tr("学习建议", "Learning Advice"))
    if not advice_ready:
        st.info(_tr("学习建议生成中，请稍候…", "Generating advice, please wait…"))
        time.sleep(2.0)
        st.rerun()
        return
    st.write(advice["general_advice"].get("summary", ""))
    st.markdown(_tr("**推荐资源：**", "**Recommended resources:**"))
    for res in advice["general_advice"].get("recommended_resources", []):
        st.markdown(f"- {res}")
    st.markdown(_tr("**细化建议：**", "**Detailed advice:**"))
    for item in advice.get("detailed_advice", []):
        st.markdown(f"- [{item['dimension']}] {item['issue']} | {item['suggestion']}")

    if not st.session_state.get("session_saved"):
        started_at = (
            st.session_state.get("session_start")
            or datetime.now(timezone.utc).isoformat()
        )
        ended_at = datetime.now(timezone.utc).isoformat()
        user = st.session_state.get("user")
        if user:
            record_study_session(
                user_id=user["id"],
                case_id=case_id,
                started_at=started_at,
                ended_at=ended_at,
                pre_score=pre_score,
                post_score=post_score,
                evaluation=evaluation,
                advice=advice,
            )
            st.session_state["session_saved"] = True

    if st.button(_tr("返回案例选择，开始下一个案例", "Back to case selection for next case"), key="back_to_cases"):
        reset_case_state()
        st.session_state["page"] = "case_selection"
        st.rerun()


def main() -> None:
    st.set_page_config(page_title=_tr("PBL 学习系统", "PBL Learning System"), layout="wide")
    init_db()
    seed_users(None)
    initialize_app_state()
    _restore_user_from_cookie()
    if st.session_state.get("user") and st.session_state.get("page") == "login":
        st.session_state["page"] = "case_selection"
    _load_active_session_record()

    with st.sidebar:
        user = st.session_state.get("user")
        if user:
            st.caption(_tr(f"当前用户：{user['username']}", f"User: {user['username']}"))
            if st.button(_tr("退出登录", "Sign out")):
                _handle_logout()
                st.rerun()
        else:
            st.caption(_tr("请先登录以体验 PBL 训练。", "Please sign in to use the PBL trainer."))
        st.markdown("---")
        _render_speed_slider()
        page_state = st.session_state.get("page", "login")
        students_locked = (page_state == "pbl_training")
        current_count = int(st.session_state.get("desired_students_count", 5))
        new_count = st.slider(
            _tr("PBL 小组人数", "PBL group size"),
            min_value=4,
            max_value=8,
            value=current_count,
            step=1,
            disabled=students_locked,
            key="students_count_slider",
        )
        if not students_locked and new_count != current_count:
            st.session_state["desired_students_count"] = new_count
            user = st.session_state.get("user")
            if user:
                if st.session_state.get("persisted_student_count") != new_count:
                    if _persist_user_preferences(student_count=new_count):
                        st.session_state["persisted_student_count"] = new_count
            else:
                st.session_state["persisted_student_count"] = new_count
        if students_locked:
            st.caption(_tr("讨论进行中，人数设置暂不可调整。", "Group size is locked during discussion."))

        if user:
            profile_cache = st.session_state.get("user_profile_cache")
            profile_reset = False
            if not profile_cache or profile_cache.get("user_id") != user["id"]:
                try:
                    data = get_user_profile(user["id"]) or {}
                except Exception as exc:
                    st.warning(_tr(f"加载基本信息失败：{exc}", f"Failed to load profile: {exc}"))
                    data = {}
                profile_cache = {"user_id": user["id"], **data}
                st.session_state["user_profile_cache"] = profile_cache
                profile_reset = True
            st.markdown(_tr("### 基本信息", "### Basic info"))
            gender_options = [_tr("未填写", "Not set"), _tr("男", "Male"), _tr("女", "Female")]
            stored_gender = (profile_cache.get("gender") or "").strip()
            if stored_gender in ("男", "Male"):
                default_gender = gender_options[1]
            elif stored_gender in ("女", "Female"):
                default_gender = gender_options[2]
            else:
                default_gender = gender_options[0]
            gender_value = st.selectbox(
                _tr("性别", "Gender"),
                gender_options,
                index=gender_options.index(default_gender),
                key="profile_gender",
            )
            age_value = st.number_input(
                _tr("年龄", "Age"),
                min_value=0,
                max_value=120,
                step=1,
                value=int(profile_cache.get("age") or 0),
                key="profile_age",
            )
            if st.button(_tr("保存基本信息", "Save profile"), use_container_width=True):
                try:
                    upsert_user_profile(
                        user_id=user["id"],
                        gender=gender_value if gender_value != _tr("未填写", "Not set") else None,
                        age=int(age_value) if age_value > 0 else None,
                    )
                    st.success(_tr("已保存基本信息。", "Profile saved."))
                    st.session_state["user_profile_cache"] = {
                        "user_id": user["id"],
                        "gender": gender_value if gender_value != _tr("未填写", "Not set") else None,
                        "age": int(age_value) if age_value > 0 else None,
                    }
                except Exception as exc:
                    st.error(_tr(f"保存失败：{exc}", f"Failed to save: {exc}"))

        st.markdown("---")
        allow_survey = page_state in {"case_selection", "evaluation"}
        if st.button(
            _tr("填写调查问卷", "Fill survey"),
            key="sidebar_survey_button",
            use_container_width=True,
            disabled=not allow_survey,
        ):
            st.session_state["page"] = "survey"
            st.rerun()
        if not allow_survey:
            st.caption(_tr("问卷仅可在推荐或评估页面填写。", "Survey available only on recommendation or evaluation pages."))

        st.markdown(_tr("### 意见与建议", "### Feedback"))
        suggestion_disabled = user is None
        if st.session_state.pop("sidebar_feedback_reset", False):
            st.session_state["sidebar_feedback_text"] = ""
        suggestion = st.text_area(
            _tr("欢迎随时提出改进建议", "Share your feedback anytime"),
            value=st.session_state.get("sidebar_feedback_text", ""),
            key="sidebar_feedback_text",
            height=120,
            disabled=suggestion_disabled,
        )
        if st.button(
            _tr("提交反馈", "Submit feedback"),
            key="sidebar_submit_feedback",
            use_container_width=True,
            disabled=suggestion_disabled,
        ):
            content = (suggestion or "").strip()
            if not content:
                st.warning(_tr("请先填写建议内容。", "Please enter your feedback first."))
            else:
                try:
                    if user:
                        record_feedback(user["id"], content)
                        st.success(_tr("感谢反馈！", "Thanks for your feedback!"))
                        st.session_state["sidebar_feedback_reset"] = True
                        st.rerun()
                    else:
                        st.warning(_tr("请先登录后再提交建议。", "Please sign in before submitting feedback."))
                except Exception as exc:
                    st.error(_tr(f"提交失败：{exc}", f"Submission failed: {exc}"))

    page = st.session_state.get("page", "login")
    if not st.session_state.get("user") and page != "login":
        st.session_state["page"] = "login"
        page = "login"

    if page != "case_selection":
        _clear_page_container("case_selection_container")
    if page != "pbl_training":
        _clear_page_container("pbl_training_container")

    if page == "login":
        render_login()
    elif page == "case_selection":
        render_case_selection()
    elif page == "pre_test":
        render_test_page("pre")
    elif page == "pbl_training":
        render_pbl_training()
    elif page == "post_test":
        render_test_page("post")
    elif page == "survey":
        render_survey_page()
    elif page == "evaluation":
        render_evaluation_page()
    else:
        st.session_state["page"] = "login"
        render_login()


if __name__ == "__main__":
    main()
