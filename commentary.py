import asyncio
import base64
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import httpx
from google.genai import types
from PIL import Image

from gemini_client import create_gemini_client, normalize_gemini_base_url
from gemini_pool import GeminiKeyPool
from main import detect_scenes, download_youtube_video, transcribe_video
from resource_limits import resolve_thread_count
from saasshorts import generate_voiceover


class CommentaryCancelled(Exception):
    pass


_COMMENTARY_RUNTIME = threading.local()
_COMMENTARY_PROCESS_LOCK = threading.RLock()
_COMMENTARY_PROCESSES: Dict[str, List[subprocess.Popen]] = {}


def _current_commentary_cancel_event():
    return getattr(_COMMENTARY_RUNTIME, "cancel_event", None)


def _current_commentary_job_id() -> Optional[str]:
    return getattr(_COMMENTARY_RUNTIME, "job_id", None)


def _terminate_process(process: subprocess.Popen) -> None:
    if not process or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=3)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=3)
        except Exception:
            pass


def _raise_if_commentary_cancelled(process: Optional[subprocess.Popen] = None) -> None:
    cancel_event = _current_commentary_cancel_event()
    if cancel_event is not None and cancel_event.is_set():
        if process is not None:
            _terminate_process(process)
        raise CommentaryCancelled("Commentary job was cancelled.")


def _register_commentary_process(process: subprocess.Popen) -> None:
    job_id = _current_commentary_job_id()
    if not job_id or not process:
        return
    with _COMMENTARY_PROCESS_LOCK:
        _COMMENTARY_PROCESSES.setdefault(job_id, []).append(process)


def _unregister_commentary_process(process: subprocess.Popen) -> None:
    job_id = _current_commentary_job_id()
    if not job_id or not process:
        return
    with _COMMENTARY_PROCESS_LOCK:
        processes = _COMMENTARY_PROCESSES.get(job_id)
        if not processes:
            return
        _COMMENTARY_PROCESSES[job_id] = [item for item in processes if item is not process]
        if not _COMMENTARY_PROCESSES[job_id]:
            _COMMENTARY_PROCESSES.pop(job_id, None)


def terminate_commentary_job_processes(job_id: str) -> None:
    with _COMMENTARY_PROCESS_LOCK:
        processes = list(_COMMENTARY_PROCESSES.pop(job_id, []))
    for process in processes:
        _terminate_process(process)


@contextmanager
def commentary_job_context(job_id: str, cancel_event=None):
    previous_job_id = getattr(_COMMENTARY_RUNTIME, "job_id", None)
    previous_cancel_event = getattr(_COMMENTARY_RUNTIME, "cancel_event", None)
    _COMMENTARY_RUNTIME.job_id = job_id
    _COMMENTARY_RUNTIME.cancel_event = cancel_event
    try:
        yield
    finally:
        _COMMENTARY_RUNTIME.job_id = previous_job_id
        _COMMENTARY_RUNTIME.cancel_event = previous_cancel_event


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
SUPPORTED_ANALYSIS_MODES = {"current", "video", "openai"}
DEFAULT_ANALYSIS_MODE = "video"
COMMENTARY_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
COMMENTARY_MUSIC_DIR = os.path.join(COMMENTARY_ASSET_DIR, "music")
DEFAULT_BACKGROUND_MUSIC_VOLUME = float(os.environ.get("OPENSHORTS_COMMENTARY_BGM_VOLUME", "0.16"))
BACKGROUND_MUSIC_TRACKS = {
    "aodebiao_caravan": {
        "id": "aodebiao_caravan",
        "label": "默认 奥德彪专属音乐",
        "title": "Caravan",
        "artist": "a_hisa",
        "filename": "a_hisa_caravan.mp3",
        "source": "https://www.gequhai.com/play/1843430",
        "notes": "Commonly referenced as the 奥德彪拉香蕉/奥德彪专属 BGM.",
    },
}
OPENAI_FRAME_INTERVAL_SECONDS = max(1.0, float(os.environ.get("OPENSHORTS_OPENAI_FRAME_INTERVAL_SECONDS", "3")))
OPENAI_MAX_FRAMES = max(1, int(os.environ.get("OPENSHORTS_OPENAI_MAX_FRAMES", "1800")))
OPENAI_BATCH_SIZE_LIMIT = 128
OPENAI_VISUAL_CONCURRENCY_LIMIT = 8
COMMENTARY_BLOCK_CONCURRENCY_LIMIT = 8
OPENAI_BATCH_SIZE = max(1, min(OPENAI_BATCH_SIZE_LIMIT, int(os.environ.get("OPENSHORTS_OPENAI_BATCH_SIZE", "46"))))
OPENAI_VISUAL_CONCURRENCY = max(1, min(OPENAI_VISUAL_CONCURRENCY_LIMIT, int(os.environ.get("OPENSHORTS_OPENAI_VISUAL_CONCURRENCY", "2"))))
COMMENTARY_BLOCK_CONCURRENCY = max(1, min(COMMENTARY_BLOCK_CONCURRENCY_LIMIT, int(os.environ.get("OPENSHORTS_COMMENTARY_BLOCK_CONCURRENCY", "2"))))
OPENAI_SCENE_AWARE_SAMPLING = os.environ.get(
    "OPENSHORTS_OPENAI_SCENE_AWARE_SAMPLING",
    "true",
).strip().lower() not in {"0", "false", "no", "off"}
OPENAI_SCENE_AWARE_MAX_DURATION_SECONDS = max(
    0,
    float(os.environ.get("OPENSHORTS_OPENAI_SCENE_AWARE_MAX_DURATION_SECONDS", "900")),
)
OPENAI_MAX_FRAME_GAP_SECONDS = max(
    OPENAI_FRAME_INTERVAL_SECONDS * 2.5,
    float(os.environ.get("OPENSHORTS_OPENAI_MAX_FRAME_GAP_SECONDS", "10")),
)
OPENAI_SCENE_MIN_SECONDS = max(0.5, float(os.environ.get("OPENSHORTS_OPENAI_SCENE_MIN_SECONDS", "1.0")))
OPENAI_SCENE_MAX_KEYFRAMES = max(1, int(os.environ.get("OPENSHORTS_OPENAI_SCENE_MAX_KEYFRAMES", "60")))
OPENAI_FRAME_INTERVAL_MIN_SECONDS = 1.0
OPENAI_FRAME_INTERVAL_MAX_SECONDS = 60.0
OPENAI_MAX_FRAMES_LIMIT = 2000
OPENAI_SCENE_MAX_KEYFRAMES_LIMIT = 600
OPENAI_SCENE_STATIC_MOTION_THRESHOLD = max(
    0.0,
    float(os.environ.get("OPENSHORTS_OPENAI_SCENE_STATIC_MOTION_THRESHOLD", "0.06")),
)
OPENAI_SCENE_DYNAMIC_MOTION_THRESHOLD = max(
    OPENAI_SCENE_STATIC_MOTION_THRESHOLD,
    float(os.environ.get("OPENSHORTS_OPENAI_SCENE_DYNAMIC_MOTION_THRESHOLD", "0.16")),
)
OPENAI_FRAME_HEIGHT = max(144, int(os.environ.get("OPENSHORTS_OPENAI_FRAME_HEIGHT", "360")))
OPENAI_IMAGE_DETAIL = os.environ.get("OPENSHORTS_OPENAI_IMAGE_DETAIL", "low")
OPENAI_REQUEST_TIMEOUT_SECONDS = max(30, int(os.environ.get("OPENSHORTS_OPENAI_REQUEST_TIMEOUT_SECONDS", "180")))
OPENAI_SCRIPT_REQUEST_TIMEOUT_SECONDS = max(
    OPENAI_REQUEST_TIMEOUT_SECONDS,
    int(os.environ.get("OPENSHORTS_OPENAI_SCRIPT_REQUEST_TIMEOUT_SECONDS", "900")),
)
OPENAI_REQUEST_RETRIES = max(1, int(os.environ.get("OPENSHORTS_OPENAI_REQUEST_RETRIES", "3")))
OPENAI_VISUAL_MAX_TOKENS = max(1000, int(os.environ.get("OPENSHORTS_OPENAI_VISUAL_MAX_TOKENS", "6000")))
OPENAI_SCRIPT_MAX_TOKENS = max(2000, int(os.environ.get("OPENSHORTS_OPENAI_SCRIPT_MAX_TOKENS", "64000")))
OPENAI_AUDIO_PROBE_SECONDS = max(2.0, float(os.environ.get("OPENSHORTS_OPENAI_AUDIO_PROBE_SECONDS", "8")))
OPENAI_AUDIO_ANALYSIS_MAX_SECONDS = max(30.0, float(os.environ.get("OPENSHORTS_OPENAI_AUDIO_ANALYSIS_MAX_SECONDS", "1800")))
OPENAI_AUDIO_ANALYSIS_TIMEOUT_SECONDS = max(
    OPENAI_REQUEST_TIMEOUT_SECONDS,
    int(os.environ.get("OPENSHORTS_OPENAI_AUDIO_ANALYSIS_TIMEOUT_SECONDS", "300")),
)
OPENAI_ANALYSIS_FRAMES_MANIFEST = "openai_analysis_frames_manifest.json"
OPENAI_VISUAL_ANALYSIS_CACHE = "openai_visual_analysis.json"
OPENAI_VISUAL_PROMPT_MAX_CHARS = max(10000, int(os.environ.get("OPENSHORTS_OPENAI_VISUAL_PROMPT_MAX_CHARS", "45000")))
OPENAI_TEMPERATURE = float(os.environ.get("OPENSHORTS_OPENAI_TEMPERATURE", "0.7"))
OPENAI_STRICT_SCRIPT_SCHEMA = os.environ.get(
    "OPENSHORTS_OPENAI_STRICT_SCRIPT_SCHEMA",
    "true",
).strip().lower() not in {"0", "false", "no", "off"}
OPENAI_LOCK_CANDIDATE_EDIT_PLAN = os.environ.get(
    "OPENSHORTS_OPENAI_LOCK_CANDIDATE_EDIT_PLAN",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}
OPENAI_TWO_STAGE_EDIT_THEN_COMMENTARY = os.environ.get(
    "OPENSHORTS_OPENAI_TWO_STAGE_EDIT_THEN_COMMENTARY",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}
GEMINI_FILES_API_HARD_MAX_BYTES = 2 * 1024 * 1024 * 1024
GEMINI_ANALYSIS_TARGET_MAX_BYTES = min(
    GEMINI_FILES_API_HARD_MAX_BYTES,
    max(128 * 1024 * 1024, int(os.environ.get("OPENSHORTS_GEMINI_ANALYSIS_TARGET_MAX_BYTES", str(1900 * 1024 * 1024)))),
)
GEMINI_ANALYSIS_HEIGHT = max(144, int(os.environ.get("OPENSHORTS_GEMINI_ANALYSIS_HEIGHT", "360")))
GEMINI_ANALYSIS_CRF = int(os.environ.get("OPENSHORTS_GEMINI_ANALYSIS_CRF", "32"))
GEMINI_ANALYSIS_AUDIO_BITRATE = os.environ.get("OPENSHORTS_GEMINI_ANALYSIS_AUDIO_BITRATE", "48k")
FFMPEG_THREADS = resolve_thread_count("OPENSHORTS_FFMPEG_THREADS", default_cap=8, reserve_cores=6)
GEMINI_FILE_UPLOAD_RETRIES = max(1, int(os.environ.get("OPENSHORTS_GEMINI_FILE_UPLOAD_RETRIES", "3")))
GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS = max(60, int(os.environ.get("OPENSHORTS_GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS", "3600")))
GEMINI_FILE_PROCESSING_MAX_TIMEOUT_SECONDS = max(
    GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS,
    int(os.environ.get("OPENSHORTS_GEMINI_FILE_PROCESSING_MAX_TIMEOUT_SECONDS", "7200")),
)
GEMINI_FILE_PROCESSING_POLL_SECONDS = max(1, int(os.environ.get("OPENSHORTS_GEMINI_FILE_PROCESSING_POLL_SECONDS", "2")))
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_EDGE_VOICE = "zh-CN-YunyangNeural"
FULL_MODE_FULL_SOURCE_UNDER_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_FULL_SOURCE_UNDER_SECONDS", "600"))
FULL_MODE_COMPACT_SOURCE_UNDER_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_COMPACT_SOURCE_UNDER_SECONDS", "1200"))
FULL_MODE_COMPACT_SOURCE_FRACTION = float(os.environ.get("OPENSHORTS_FULL_MODE_COMPACT_SOURCE_FRACTION", "0.42"))
FULL_MODE_MIN_VISUAL_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_MIN_VISUAL_SECONDS", "240"))
FULL_MODE_LONG_MIN_VISUAL_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_LONG_MIN_VISUAL_SECONDS", "480"))
FULL_MODE_MAX_VISUAL_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_VISUAL_SECONDS", "900"))
FULL_MODE_SOURCE_FRACTION = float(os.environ.get("OPENSHORTS_FULL_MODE_SOURCE_FRACTION", "0.16"))
FULL_MODE_MAX_SOURCE_RETENTION_FRACTION = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_SOURCE_RETENTION_FRACTION", "0.45"))
FULL_MODE_MAX_NARRATION_CHARS_ZH = int(os.environ.get("OPENSHORTS_FULL_MODE_MAX_NARRATION_CHARS_ZH", "22000"))
FULL_MODE_MAX_NARRATION_CHARS_OTHER = int(os.environ.get("OPENSHORTS_FULL_MODE_MAX_NARRATION_CHARS_OTHER", "32000"))
FULL_MODE_MAX_VOICEOVER_DURATION_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_VOICEOVER_DURATION_RATIO", "1.15"))
FULL_MODE_FULL_PROCESS_MIN_PLAYABLE_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_FULL_PROCESS_MIN_PLAYABLE_RATIO", "0.90"))
FULL_MODE_MIN_PLAYABLE_TARGET_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_MIN_PLAYABLE_TARGET_RATIO", "0.90"))
FULL_MODE_MAX_PAUSE_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_PAUSE_RATIO", "0.35"))
FULL_MODE_MAX_PAUSE_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_PAUSE_SECONDS", "18"))
FULL_MODE_MAX_CONSECUTIVE_PAUSE_BLOCKS = int(os.environ.get("OPENSHORTS_FULL_MODE_MAX_CONSECUTIVE_PAUSE_BLOCKS", "1"))
FULL_MODE_VALIDATION_EPSILON_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_VALIDATION_EPSILON_SECONDS", "0.25"))
FULL_MODE_VALIDATION_EPSILON_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_VALIDATION_EPSILON_RATIO", "0.005"))
FULL_MODE_VISUAL_BUDGET_TOLERANCE_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_VISUAL_BUDGET_TOLERANCE_SECONDS", "1.0"))
FULL_MODE_VISUAL_BUDGET_TOLERANCE_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_VISUAL_BUDGET_TOLERANCE_RATIO", "0.002"))
FULL_MODE_MAX_NARRATION_SILENCE_TAIL_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_NARRATION_SILENCE_TAIL_SECONDS", "1.5"))
FULL_MODE_RENDER_SYNC_MAX_VIDEO_SPEED = float(os.environ.get("OPENSHORTS_FULL_MODE_RENDER_SYNC_MAX_VIDEO_SPEED", "4.0"))
FULL_MODE_RENDER_SYNC_MAX_AUDIO_SPEED = float(os.environ.get("OPENSHORTS_FULL_MODE_RENDER_SYNC_MAX_AUDIO_SPEED", "2.0"))
FULL_MODE_RENDER_SYNC_TTS_REWRITE_ATTEMPTS = int(os.environ.get("OPENSHORTS_FULL_MODE_RENDER_SYNC_TTS_REWRITE_ATTEMPTS", "1"))
FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION = float(os.environ.get("OPENSHORTS_FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION", "0.85"))
FULL_MODE_MIN_SCENE_MATCHED_NARRATION_CHARS_ZH = int(os.environ.get("OPENSHORTS_FULL_MODE_MIN_SCENE_MATCHED_NARRATION_CHARS_ZH", "36"))
FULL_MODE_MIN_SCENE_MATCHED_NARRATION_CHARS_OTHER = int(os.environ.get("OPENSHORTS_FULL_MODE_MIN_SCENE_MATCHED_NARRATION_CHARS_OTHER", "24"))
FULL_MODE_NARRATION_DENSITY_MIN_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_NARRATION_DENSITY_MIN_RATIO", "0.80"))
FULL_MODE_MIN_NARRATED_BLOCK_VOICEOVER_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_MIN_NARRATED_BLOCK_VOICEOVER_RATIO", "0.55"))
FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS", "12"))
FULL_MODE_MAX_PLAYABLE_TARGET_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_PLAYABLE_TARGET_RATIO", "1.10"))
FULL_MODE_OPENAI_PLAN_MAX_BLOCK_PLAYABLE_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_OPENAI_PLAN_MAX_BLOCK_PLAYABLE_SECONDS", "10.5"))
NON_FULL_TARGET_DURATION_TOLERANCE_SECONDS = float(os.environ.get("OPENSHORTS_NON_FULL_TARGET_DURATION_TOLERANCE_SECONDS", "3.0"))
NON_FULL_TARGET_DURATIONS = {
    "short": {"min_seconds": 60.0, "max_seconds": 90.0, "label": "60-90 second"},
    "two_to_four": {"min_seconds": 120.0, "max_seconds": 240.0, "label": "2-4 minute"},
    "medium": {"min_seconds": 180.0, "max_seconds": 300.0, "label": "3-5 minute"},
}
GEMINI_SAFE_INPUT_TOKEN_BUDGET = int(os.environ.get("OPENSHORTS_GEMINI_SAFE_INPUT_TOKEN_BUDGET", "180000"))
GEMINI_LOW_RES_TOKENS_PER_SECOND = 100.0
GEMINI_SCRIPT_VALIDATION_ATTEMPTS = max(2, int(os.environ.get("OPENSHORTS_GEMINI_SCRIPT_VALIDATION_ATTEMPTS", "12")))
COMMENTARY_BANNED_PHRASES = (
    "画面汇总",
    "前面的动作和结果已经交代完整",
    "这段解说也自然收住",
    "自然收住",
    "解说到这里",
    "旁白到这里",
    "这段旁白",
    "这段口播",
)
PUBLISH_BANNED_META_PHRASES = (
    "这段素材记录了",
    "本视频展示了",
    "这段视频记录了",
    "该视频展示了",
    "视频中展示了",
    "这期视频带你看",
)
COMMENTARY_NARRATION_BANNED_PATTERNS = (
    re.compile(r"镜头", re.I),
    re.compile(r"画面里|画面中|视频里|视频中"),
    re.compile(r"当前画面|当前可见|画面显示|画面展示|视频显示|视频展示|可以看到|能看到"),
    re.compile(r"前面的动作.*交代完整|动作和结果.*交代完整"),
    re.compile(r"这段解说|解说词|旁白稿|旁白文案|脚本说明|写作说明|编辑说明|收束提示"),
    re.compile(r"自然收住"),
    re.compile(r"(?:这|本|此|上|前|后)?(?:一)?(?:段|部分|条)?(?:解说|旁白|口播|文案|脚本|稿子|叙述|讲述)[^。！？!?]{0,24}(?:收住|收束|结束|收尾|告一段落|闭合|落点|讲完|说完)"),
    re.compile(r"(?:到这里|到此|讲到这里|说到这里|看到这里|进行到这里|这一段|这部分)[^。！？!?]{0,36}(?:交代|说明|讲清|说清|讲完|说完|结束|收束|收住|告一段落|完整|闭合)"),
    re.compile(r"(?:前面|前面的|之前|上文|前文|前段|上一段)[^。！？!?]{0,28}(?:动作|过程|结果|内容|信息|铺垫|线索|情节)[^。！？!?]{0,28}(?:交代|说明|讲清|说清|讲完|说完|补齐|完整)"),
    re.compile(r"(?:该说的|能说的|要讲的|该交代的)[^。！？!?]{0,20}(?:说完|讲完|交代完|讲清|说清)"),
    re.compile(r"(?:不再展开|无需赘述|不用多说|不多讲|不再多讲|不再解释|不必再讲)"),
    re.compile(r"(?:作为结尾|作为收尾|用来收尾|拿来收尾|收尾一句|结尾一句|结束语|收束语)"),
    re.compile(r"看到这里[^。！？!?]{0,32}(?:应该|已经|基本|就能|可以)[^。！？!?]{0,24}(?:明白|看懂|知道|清楚)"),
    re.compile(r"\b(?:this|the)\s+(?:narration|voiceover|script|commentary|segment)\b.{0,80}\b(?:wraps?\s+up|closes?|ends?|comes?\s+to\s+an\s+end|is\s+complete)\b", re.I),
    re.compile(r"\b(?:at\s+this\s+point|by\s+now|from\s+here)\b.{0,80}\b(?:previous|everything|action|result|story)\b.{0,80}\b(?:explained|covered|clear|complete)\b", re.I),
)
COMMENTARY_REPEAT_LIMIT_PHRASES = (
    "没事的没事的",
    "兄弟们",
    "稳住",
    "直接盘它",
    "撤了撤了",
    "爽了爽了",
)
COMMENTARY_REPEATED_SENTENCE_MIN_CHARS = 16
COMMENTARY_AUTO_FILLED_PLACEHOLDER_PHRASES = (
    "这一段把前后工序之间的衔接补上",
    "这一段补上后段收尾",
    "自动补齐的作业流程过渡",
    "自动补齐的后段作业流程收尾",
    "This section fills the transition",
    "This section fills in the late workflow",
    "auto-filled process transition",
    "auto-filled late process wrap-up",
)
COMMENTARY_AUTO_FILLED_BRIDGE_VISUAL = "原片环境声衔接段"
ASS_SUBTITLE_MAX_LINE_UNITS = 34
ASS_SUBTITLE_MAX_VISIBLE_LINES = 2
ASS_SUBTITLE_DEFAULT_WIDTH = 1080
ASS_SUBTITLE_DEFAULT_HEIGHT = 1920
ASS_SUBTITLE_FONT_HEIGHT_RATIO = 0.045
ASS_SUBTITLE_MARGIN_X_RATIO = 0.06
ASS_SUBTITLE_MARGIN_Y_RATIO = 0.073
ASS_SUBTITLE_MIN_FONT_SIZE = 28
ASS_SUBTITLE_MAX_FONT_SIZE = 104


EDGE_VOICES_BY_LANGUAGE = {
    "zh": "zh-CN-YunyangNeural",
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "ja": "ja-JP-NanamiNeural",
}


def _resolve_ffmpeg_binary() -> str:
    configured = os.environ.get("OPENSHORTS_FFMPEG_BINARY")
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    project_root = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(project_root, "render-service", "node_modules", "@remotion", "compositor-win32-x64-msvc", "ffmpeg.exe"),
        os.path.join(project_root, "remotion", "node_modules", "@remotion", "compositor-win32-x64-msvc", "ffmpeg.exe"),
    ):
        if os.path.exists(candidate):
            return candidate
    return "ffmpeg"


FFMPEG_BINARY = _resolve_ffmpeg_binary()


def _resolve_ffprobe_binary() -> str:
    configured = os.environ.get("OPENSHORTS_FFPROBE_BINARY")
    if configured:
        return configured
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg_dir = os.path.dirname(FFMPEG_BINARY) if FFMPEG_BINARY else ""
    if ffmpeg_dir:
        candidate = os.path.join(ffmpeg_dir, "ffprobe.exe" if FFMPEG_BINARY.lower().endswith(".exe") else "ffprobe")
        if os.path.exists(candidate):
            return candidate
    return "ffprobe"


FFPROBE_BINARY = _resolve_ffprobe_binary()


def _is_ffmpeg_command(command: str) -> bool:
    return os.path.basename(str(command)).lower() in {"ffmpeg", "ffmpeg.exe"}


def _windows_path_from_wsl(path: str) -> str:
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", path)
    if not match:
        return path
    drive, rest = match.groups()
    windows_rest = rest.replace("/", "\\")
    return f"{drive.upper()}:\\{windows_rest}"


def _normalize_ffmpeg_command_args(cmd: List[str]) -> List[str]:
    if not cmd or not _is_ffmpeg_command(cmd[0]):
        return cmd
    binary = FFMPEG_BINARY or cmd[0]
    if os.path.basename(str(binary)).lower() != "ffmpeg.exe":
        return [binary, *cmd[1:]]
    return [binary, *(_windows_path_from_wsl(arg) if isinstance(arg, str) and arg.startswith("/mnt/") else arg for arg in cmd[1:])]


def _limit_ffmpeg_threads(cmd: List[str]) -> List[str]:
    cmd = _normalize_ffmpeg_command_args(cmd)
    if not cmd or not _is_ffmpeg_command(cmd[0]):
        return cmd
    if "-threads" in cmd:
        return cmd
    return [cmd[0], "-threads", str(FFMPEG_THREADS), *cmd[1:]]


def _ffmpeg_output_format_args(output_path: str) -> List[str]:
    if str(output_path or "").lower().endswith(".m4a"):
        return ["-f", "mp4"]
    return []


def _run_capture_command(cmd: List[str], cwd: Optional[str] = None):
    _raise_if_commentary_cancelled()
    process = subprocess.Popen(
        _limit_ffmpeg_threads(cmd),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _register_commentary_process(process)
    try:
        while True:
            _raise_if_commentary_cancelled(process)
            try:
                stdout, stderr = process.communicate(timeout=0.25)
                return process.returncode, stdout, stderr
            except subprocess.TimeoutExpired:
                continue
    finally:
        _unregister_commentary_process(process)


def _run_command(cmd: List[str], cwd: Optional[str] = None) -> None:
    try:
        returncode, stdout, stderr = _run_capture_command(cmd, cwd=cwd)
    except FileNotFoundError as exc:
        raise Exception(
            f"FFmpeg executable not found: {FFMPEG_BINARY}. Install ffmpeg or set OPENSHORTS_FFMPEG_BINARY to a valid binary."
        ) from exc
    if returncode != 0:
        raise Exception(stderr or stdout or f"Command failed: {' '.join(cmd)}")


def _clean_publish_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    for phrase in PUBLISH_BANNED_META_PHRASES:
        text = text.replace(phrase, "")
    text = re.sub(r"^[，。；：、\s]+", "", text).strip()
    return text


def _limit_text_chars(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip("，。；：、,.!?！？ ")


def _normalize_hashtag(tag: str) -> str:
    clean = re.sub(r"\s+", "", str(tag or "").strip())
    if not clean:
        return ""
    return clean if clean.startswith("#") else f"#{clean}"


def _build_douyin_publish_fields(script: Dict) -> Dict[str, str]:
    title = _clean_publish_text(script.get("title") or "")
    summary = _clean_publish_text(script.get("summary") or "")
    visual_parts = []
    for block in script.get("narration_blocks") or []:
        if isinstance(block, dict) and block.get("visual"):
            visual_parts.append(_clean_publish_text(block.get("visual")))
    visual_context = "，".join(part for part in visual_parts if part)

    publish_title_source = title or summary or visual_context or "二创解说全过程"
    publish_title = _limit_text_chars(publish_title_source, 30)

    description_body = summary or visual_context or title or "完整二创解说，带你看清关键过程和最终结果。"
    if title and not (summary or visual_context) and not description_body.startswith(title[:8]):
        description_body = f"{title}：{description_body}"
    description_body = _clean_publish_text(description_body)

    hashtags = []
    for raw in script.get("hashtags") or []:
        tag = _normalize_hashtag(raw)
        if tag and tag not in hashtags:
            hashtags.append(tag)
    if not hashtags:
        hashtags = ["#二创解说", "#工厂实拍", "#涨知识"]
    hashtag_text = " ".join(hashtags[:8])
    publish_description = _limit_text_chars(f"{description_body}\n\n{hashtag_text}", 1000)
    return {
        "publish_title": publish_title,
        "publish_description": publish_description,
    }


def _youtube_video_id(url: str) -> Optional[str]:
    parsed = urlparse(str(url or "").strip())
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate or None
    if "youtube.com" in host:
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if query_id:
            return query_id
        path_parts = [part for part in parsed.path.split("/") if part]
        for marker in ("shorts", "embed", "live"):
            if marker in path_parts:
                index = path_parts.index(marker)
                if index + 1 < len(path_parts):
                    return path_parts[index + 1]
    return None


def _download_youtube_thumbnail_base(source_url: str, output_dir: str, slug: str) -> Optional[str]:
    video_id = _youtube_video_id(source_url)
    if not video_id:
        return None
    output_path = os.path.join(output_dir, f"{_safe_slug(slug)}_youtube_thumbnail.jpg")
    candidates = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    ]
    for url in candidates:
        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                response = client.get(url)
            if response.status_code != 200 or not response.content:
                continue
            with open(output_path, "wb") as f:
                f.write(response.content)
            with Image.open(output_path) as image:
                width, height = image.size
            if width <= 160 or height <= 90:
                continue
            return output_path
        except Exception:
            continue
    return None


def _extract_cover_frame(video_path: str, output_dir: str, slug: str, duration: float) -> Optional[str]:
    timestamp = max(0.0, min(float(duration or 0) * 0.35, max(0.0, float(duration or 0) - 0.1)))
    output_path = os.path.join(output_dir, f"{_safe_slug(slug)}_cover_base.jpg")
    try:
        _run_command([
            FFMPEG_BINARY,
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            output_path,
        ])
    except Exception:
        return None
    if not os.path.exists(output_path):
        return None
    try:
        with Image.open(output_path) as image:
            image.verify()
    except Exception:
        try:
            os.remove(output_path)
        except Exception:
            pass
        return None
    return output_path


def _compose_cover_image(base_path: str, output_path: str, ratio: float, size: Tuple[int, int]) -> None:
    with Image.open(base_path) as image:
        image = image.convert("RGB")
        src_w, src_h = image.size
        src_ratio = src_w / max(1, src_h)
        if src_ratio >= ratio:
            crop_h = src_h
            crop_w = int(crop_h * ratio)
        else:
            crop_w = src_w
            crop_h = int(crop_w / ratio)
        left = max(0, (src_w - crop_w) // 2)
        top = max(0, (src_h - crop_h) // 2)
        image = image.crop((left, top, left + crop_w, top + crop_h)).resize(size, Image.Resampling.LANCZOS)
        image.save(output_path, quality=92)


def _generate_commentary_covers(
    cover_video_path: str,
    output_dir: str,
    slug: str,
    duration: float,
    cover_title: str = "",
    source_type: str = "file",
    source_url: Optional[str] = None,
) -> Dict[str, object]:
    timestamp = max(0.0, min(float(duration or 0) * 0.35, max(0.0, float(duration or 0) - 0.1)))
    job_id = os.path.basename(os.path.abspath(output_dir))
    safe_slug = _safe_slug(slug or "commentary")
    cover_specs = [
        {
            "key": "landscape",
            "filename": f"{safe_slug}_cover_4x3.jpg",
            "ratio": 4 / 3,
            "size": (1200, 900),
            "url_key": "cover_landscape_url",
        },
        {
            "key": "portrait",
            "filename": f"{safe_slug}_cover_3x4.jpg",
            "ratio": 3 / 4,
            "size": (900, 1200),
            "url_key": "cover_portrait_url",
        },
    ]
    base_image_path = None
    if source_type == "url" and source_url:
        base_image_path = _download_youtube_thumbnail_base(source_url, output_dir, safe_slug)
    if not base_image_path:
        base_image_path = _extract_cover_frame(cover_video_path, output_dir, safe_slug, duration)

    result = {"covers": []}
    for spec in cover_specs:
        output_path = os.path.join(output_dir, spec["filename"])
        if base_image_path:
            _compose_cover_image(base_image_path, output_path, spec["ratio"], spec["size"])
        else:
            ratio_text = "4/3" if spec["key"] == "landscape" else "3/4"
            inverse_ratio = "3/4" if spec["key"] == "landscape" else "4/3"
            scale = f"{spec['size'][0]}:{spec['size'][1]}"
            vf = (
                f"crop='if(gte(iw/ih,{ratio_text}),ih*{ratio_text},iw)':"
                f"'if(gte(iw/ih,{ratio_text}),ih,iw*{inverse_ratio})',scale={scale}"
            )
            try:
                _run_command([
                    FFMPEG_BINARY,
                    "-y",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    cover_video_path,
                    "-frames:v",
                    "1",
                    "-vf",
                    vf,
                    "-q:v",
                    "2",
                    output_path,
                ])
            except Exception as exc:
                print(f"⚠️ [Commentary] Failed to generate {spec['key']} cover: {exc}")
                continue
            if not os.path.exists(output_path):
                continue
        url = f"/videos/{job_id}/{spec['filename']}"
        item = {
            "type": spec["key"],
            "url": url,
            "filename": spec["filename"],
            "base": "youtube_thumbnail" if source_type == "url" and base_image_path else "clean_video_frame",
        }
        result["covers"].append(item)
        result[spec["url_key"]] = url
    return result


def _parse_ffmpeg_progress_seconds(key: str, value: str) -> Optional[float]:
    try:
        if key in {"out_time_ms", "out_time_us"}:
            return max(0.0, float(value) / 1_000_000.0)
        if key == "out_time":
            hours, minutes, seconds = value.split(":")
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return None
    return None


def _run_ffmpeg_with_progress(
    cmd: List[str],
    duration: float = 0.0,
    progress: Optional[Callable[[str], None]] = None,
    label: str = "FFmpeg processing",
    cwd: Optional[str] = None,
) -> None:
    if not progress or duration <= 0:
        _run_command(cmd, cwd=cwd)
        return

    run_cmd = _limit_ffmpeg_threads([cmd[0], "-nostats", "-progress", "pipe:1", *cmd[1:]])
    _raise_if_commentary_cancelled()
    process = subprocess.Popen(
        run_cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    _register_commentary_process(process)
    output_lines = []
    last_percent = -1
    try:
        if process.stdout:
            for raw_line in process.stdout:
                _raise_if_commentary_cancelled(process)
                line = raw_line.strip()
                if line:
                    output_lines.append(line)
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                seconds = _parse_ffmpeg_progress_seconds(key, value)
                if seconds is None:
                    continue
                percent = int(max(0, min(99, (seconds / duration) * 100)))
                if percent >= last_percent + 5:
                    last_percent = percent
                    progress(f"{label}: {percent}%")
        while process.poll() is None:
            _raise_if_commentary_cancelled(process)
            time.sleep(0.1)
        return_code = process.returncode
    finally:
        _unregister_commentary_process(process)
    if return_code != 0:
        raise Exception("\n".join(output_lines[-80:]) or f"Command failed: {' '.join(cmd)}")
    progress(f"{label}: 100%")


async def _generate_edge_voiceover_async(
    text: str,
    output_path: str,
    voice: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise Exception("Edge TTS dependency is missing. Please install edge-tts.") from exc
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(output_path)


def generate_edge_voiceover(
    text: str,
    output_path: str,
    voice: str = DEFAULT_EDGE_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> str:
    asyncio.run(_generate_edge_voiceover_async(text, output_path, voice, rate=rate, pitch=pitch))
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise Exception("Edge TTS did not generate a valid audio file")
    return output_path


def generate_commentary_voiceover(
    text: str,
    output_path: str,
    tts_provider: str = "edge",
    language: str = "zh",
    elevenlabs_key: Optional[str] = None,
    voice_id: str = DEFAULT_VOICE_ID,
    edge_voice: Optional[str] = None,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> str:
    provider = (tts_provider or "edge").lower()
    if provider == "elevenlabs":
        if not elevenlabs_key:
            raise Exception("Missing ElevenLabs API Key")
        return generate_voiceover(text, elevenlabs_key, output_path, voice_id)
    if provider == "edge":
        voice = edge_voice or EDGE_VOICES_BY_LANGUAGE.get(language, DEFAULT_EDGE_VOICE)
        return generate_edge_voiceover(text, output_path, voice, rate=rate, pitch=pitch)
    raise Exception(f"Unsupported TTS provider: {tts_provider}")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value or "commentary").strip("_")
    return slug[:80] or "commentary"


def _clean_json_text(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```json"):
        value = value[7:]
    if value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    value = value.strip()
    start_idx = value.find("{")
    end_idx = value.rfind("}")
    if start_idx != -1 and end_idx != -1:
        value = value[start_idx:end_idx + 1]
    return value


def _gemini_response_text(response) -> str:
    try:
        text = getattr(response, "text", "") or ""
    except Exception:
        text = ""
    if str(text).strip():
        return str(text)
    parts = []
    for candidate in (getattr(response, "candidates", None) or []):
        content = getattr(candidate, "content", None)
        for part in (getattr(content, "parts", None) or []):
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(str(part_text))
    return "".join(parts)


def _ensure_gemini_response_has_text(response, context: str = "request") -> str:
    text = _gemini_response_text(response)
    if not text.strip():
        raise RuntimeError(f"Gemini returned empty response text during {context}")
    return text


def _normalize_analysis_mode(analysis_mode: Optional[str]) -> str:
    mode = (analysis_mode or DEFAULT_ANALYSIS_MODE).strip().lower()
    if mode not in SUPPORTED_ANALYSIS_MODES:
        raise ValueError(f"Unsupported commentary analysis mode: {analysis_mode}")
    return mode


def _clamp_float(value, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(min_value, min(max_value, parsed))


def _clamp_int(value, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def resolve_commentary_block_concurrency(value: Optional[int] = None) -> int:
    return _clamp_int(value, COMMENTARY_BLOCK_CONCURRENCY, 1, COMMENTARY_BLOCK_CONCURRENCY_LIMIT)


def resolve_openai_sampling_options(
    frame_interval_seconds: Optional[float] = None,
    max_frames: Optional[int] = None,
    scene_max_keyframes: Optional[int] = None,
    batch_size: Optional[int] = None,
    visual_concurrency: Optional[int] = None,
) -> Dict:
    resolved_max_frames = _clamp_int(max_frames, OPENAI_MAX_FRAMES, 1, OPENAI_MAX_FRAMES_LIMIT)
    resolved_scene_max_keyframes = _clamp_int(
        scene_max_keyframes,
        OPENAI_SCENE_MAX_KEYFRAMES,
        1,
        min(OPENAI_SCENE_MAX_KEYFRAMES_LIMIT, resolved_max_frames),
    )
    return {
        "frame_interval_seconds": _clamp_float(
            frame_interval_seconds,
            OPENAI_FRAME_INTERVAL_SECONDS,
            OPENAI_FRAME_INTERVAL_MIN_SECONDS,
            OPENAI_FRAME_INTERVAL_MAX_SECONDS,
        ),
        "max_frames": resolved_max_frames,
        "scene_max_keyframes": resolved_scene_max_keyframes,
        "batch_size": _clamp_int(batch_size, OPENAI_BATCH_SIZE, 1, OPENAI_BATCH_SIZE_LIMIT),
        "visual_concurrency": _clamp_int(visual_concurrency, OPENAI_VISUAL_CONCURRENCY, 1, OPENAI_VISUAL_CONCURRENCY_LIMIT),
    }


def _get_video_info(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps else 0.0
    return {"duration": duration, "width": width, "height": height, "fps": fps or 0}


def _get_video_duration(video_path: str) -> float:
    return float(_get_video_info(video_path).get("duration") or 0)


def _has_audio_stream(video_path: str) -> bool:
    if not video_path or not os.path.exists(video_path):
        return False
    ffprobe_name = os.path.basename(str(FFPROBE_BINARY)).lower()
    probe_path = _windows_path_from_wsl(video_path) if ffprobe_name == "ffprobe.exe" else video_path
    try:
        returncode, stdout, _stderr = _run_capture_command([
            FFPROBE_BINARY, "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            probe_path,
        ])
    except Exception:
        return False
    return returncode == 0 and bool((stdout or "").strip())


def _extract_keyframes(video_path: str, output_dir: str, duration: float, count: int = 8) -> List[str]:
    frames_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(frames_dir, exist_ok=True)
    if duration <= 0:
        return []

    timestamps = []
    usable_count = max(1, min(count, 12))
    for i in range(usable_count):
        timestamps.append((duration * (i + 1)) / (usable_count + 1))

    paths = []
    for index, ts in enumerate(timestamps, start=1):
        frame_path = os.path.join(frames_dir, f"frame_{index:02d}.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{ts:.2f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "3",
            frame_path,
        ]
        try:
            _run_command(cmd)
            if os.path.exists(frame_path):
                paths.append(frame_path)
        except Exception:
            continue
    return paths


def _sample_transcript_segments(transcript: Dict, max_segments: int = 80) -> List[Dict]:
    segments = transcript.get("segments", []) or []
    if len(segments) <= max_segments:
        return [{"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")} for s in segments]

    step = max(1, math.ceil(len(segments) / max_segments))
    sampled = segments[::step][:max_segments]
    return [{"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")} for s in sampled]


def _transcript_spoken_segments(transcript: Optional[Dict]) -> List[Dict]:
    if not isinstance(transcript, dict):
        return []
    segments = []
    for raw in transcript.get("segments") or []:
        if not isinstance(raw, dict):
            continue
        text = re.sub(r"\s+", " ", str(raw.get("text") or "")).strip()
        if len(text) < 2:
            continue
        try:
            start = float(raw.get("start"))
            end = float(raw.get("end"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        segments.append({"start": start, "end": end, "text": text})
    return segments


def _transcript_has_source_commentary(transcript: Optional[Dict]) -> bool:
    segments = _transcript_spoken_segments(transcript)
    if not segments:
        return False
    total_chars = sum(len(re.sub(r"\s+", "", item["text"])) for item in segments)
    spoken_seconds = sum(max(0.0, item["end"] - item["start"]) for item in segments)
    if total_chars >= 180 and len(segments) >= 5:
        return True
    if spoken_seconds >= 25.0 and len(segments) >= 4:
        return True
    return False


def _format_source_commentary_timeline(
    transcript: Optional[Dict],
    max_segments: int = 120,
    max_chars: int = 16000,
    start: Optional[float] = None,
    end: Optional[float] = None,
    margin: float = 0.0,
) -> str:
    segments = _transcript_spoken_segments(transcript)
    if not segments:
        return ""
    if start is not None or end is not None:
        low = float(start if start is not None else 0.0) - max(0.0, float(margin or 0.0))
        high = float(end if end is not None else max((item["end"] for item in segments), default=0.0)) + max(0.0, float(margin or 0.0))
        segments = [
            item
            for item in segments
            if item["end"] >= low and item["start"] <= high
        ]
        if not segments:
            return ""
    if len(segments) > max_segments:
        step = max(1, math.ceil(len(segments) / max_segments))
        segments = segments[::step][:max_segments]
    lines = []
    used = 0
    for item in segments:
        text = _limit_text_chars(item["text"], 260)
        line = f"{item['start']:.2f}-{item['end']:.2f}: {text}"
        next_used = used + len(line) + 1
        if next_used > max_chars:
            break
        lines.append(line)
        used = next_used
    return "\n".join(lines)


def _should_check_source_commentary_for_audio_muting(
    original_audio_volume: float,
    pause_original_audio_volume: float,
    target_duration: str,
) -> bool:
    if max(float(original_audio_volume or 0.0), float(pause_original_audio_volume or 0.0)) > 0:
        return True
    return target_duration == "full"


def _extract_audio_clip_for_openai(
    video_path: str,
    output_path: str,
    start: float,
    duration: float,
) -> str:
    clip_duration = max(0.1, float(duration or 0.0))
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, float(start or 0.0)):.3f}",
        "-t", f"{clip_duration:.3f}",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        output_path,
    ]
    _run_command(cmd)
    return output_path


def _openai_audio_content_parts(audio_path: str, mode: str = "input_audio") -> List[Dict]:
    with open(audio_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    if mode == "audio_url":
        return [{
            "type": "audio_url",
            "audio_url": {"url": f"data:audio/wav;base64,{encoded}"},
        }]
    return [{
        "type": "input_audio",
        "input_audio": {"data": encoded, "format": "wav"},
    }]


def _audio_probe_reference_segment(transcript: Optional[Dict], duration: float) -> Optional[Dict]:
    spoken = _transcript_spoken_segments(transcript)
    if not spoken:
        return None
    candidates = []
    for item in spoken:
        text = re.sub(r"\s+", " ", str(item.get("text") or "").strip())
        if len(text) < 12:
            continue
        start = max(0.0, float(item["start"]) - 0.25)
        end = min(float(duration or item["end"]), float(item["end"]) + 0.25)
        if end <= start:
            continue
        candidates.append({
            "start": start,
            "duration": min(12.0, max(2.0, end - start)),
            "text": text,
        })
    if not candidates:
        return None
    return max(candidates, key=lambda item: min(len(item["text"]), 120))


def _audio_probe_tokens(value: str) -> List[str]:
    text = str(value or "").lower()
    latin = [item for item in re.findall(r"[a-z0-9]+", text) if len(item) >= 3]
    cjk = re.findall(r"[\u3400-\u9fff]", text)
    return latin + cjk


def _audio_probe_overlap_score(expected: str, actual: str) -> float:
    expected_tokens = _audio_probe_tokens(expected)
    actual_tokens = set(_audio_probe_tokens(actual))
    if not expected_tokens or not actual_tokens:
        return 0.0
    matched = sum(1 for token in expected_tokens if token in actual_tokens)
    return matched / max(1, min(len(expected_tokens), 20))


def _probe_openai_audio_analysis_support(
    api_key: str,
    base_url: str,
    model: str,
    video_path: str,
    output_dir: str,
    duration: float,
    transcript: Optional[Dict] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict:
    if not api_key or not base_url or not model or not video_path:
        return {"supported": False, "reason": "missing OpenAI-compatible audio probe configuration"}
    reference = _audio_probe_reference_segment(transcript, duration)
    probe_duration = (
        float(reference["duration"])
        if reference
        else min(OPENAI_AUDIO_PROBE_SECONDS, max(0.1, float(duration or OPENAI_AUDIO_PROBE_SECONDS)))
    )
    probe_start = float(reference["start"]) if reference else 0.0
    if not reference and duration and duration > probe_duration * 3:
        probe_start = min(max(0.0, duration * 0.08), max(0.0, duration - probe_duration))
    probe_path = os.path.join(output_dir, "openai_audio_probe.wav")
    try:
        _extract_audio_clip_for_openai(video_path, probe_path, probe_start, probe_duration)
    except Exception as exc:
        return {"supported": False, "reason": f"could not extract probe audio: {str(exc)[:300]}"}

    if reference:
        probe_prompt = (
            "Transcribe only the spoken words you can hear in the attached WAV clip. "
            "If you cannot inspect the audio, reply exactly NO_AUDIO_SUPPORT. Do not guess from context."
        )
    else:
        probe_prompt = (
            "You are testing whether this OpenAI-compatible chat endpoint can actually inspect attached audio. "
            "Listen to the attached WAV. If you can access the audio content, reply AUDIO_SUPPORTED and a short phrase about what is audible. "
            "If you cannot inspect the audio, reply exactly NO_AUDIO_SUPPORT."
        )
    for audio_mode in ("input_audio", "audio_url"):
        try:
            text = _call_openai_compatible_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": probe_prompt}] + _openai_audio_content_parts(probe_path, audio_mode),
                }],
                max_tokens=80,
                timeout_seconds=OPENAI_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            if progress:
                progress(f"OpenAI-compatible audio probe with {audio_mode} failed: {str(exc)[:300]}")
            continue
        normalized = re.sub(r"\s+", " ", text or "").strip().upper()
        if reference:
            overlap = _audio_probe_overlap_score(reference["text"], text)
            if normalized.startswith("NO_AUDIO_SUPPORT"):
                overlap = 0.0
            if overlap >= 0.35:
                return {
                    "supported": True,
                    "mode": audio_mode,
                    "reason": f"audio probe transcript overlap {overlap:.2f}",
                    "probe_start": round(probe_start, 3),
                    "probe_duration": round(probe_duration, 3),
                }
            if progress:
                progress(
                    f"OpenAI-compatible audio probe with {audio_mode} failed transcript verification "
                    f"(overlap {overlap:.2f}); using fallback if no audio mode verifies."
                )
            continue
        if normalized.startswith("AUDIO_SUPPORTED"):
            return {"supported": True, "mode": audio_mode, "reason": text[:300]}
        if progress:
            progress(f"OpenAI-compatible audio probe with {audio_mode} did not confirm audio support: {text[:300]}")
    return {"supported": False, "reason": "endpoint did not confirm audio support"}


def _analyze_openai_source_audio(
    api_key: str,
    base_url: str,
    model: str,
    video_path: str,
    output_dir: str,
    duration: float,
    audio_mode: str,
    transcript: Optional[Dict],
    progress: Optional[Callable[[str], None]] = None,
) -> Optional[Dict]:
    if not audio_mode:
        return None
    analysis_duration = min(OPENAI_AUDIO_ANALYSIS_MAX_SECONDS, max(0.1, float(duration or 0.0)))
    audio_path = os.path.join(output_dir, "openai_source_audio_analysis.wav")
    try:
        _extract_audio_clip_for_openai(video_path, audio_path, 0.0, analysis_duration)
    except Exception as exc:
        if progress:
            progress(f"OpenAI-compatible audio analysis skipped; audio extraction failed: {str(exc)[:300]}")
        return None
    transcript_timeline = _format_source_commentary_timeline(
        transcript,
        max_segments=160,
        max_chars=18000,
        start=0.0,
        end=analysis_duration,
        margin=0.0,
    )
    prompt = f"""Analyze the original source video's spoken audio for a commentary remix.

Return valid JSON only. Focus on what the original narrator says, the audible events, and the timestamped meaning that should be combined with visual analysis. Do not write the final new narration.

Audio clip range: 0.0-{analysis_duration:.1f} seconds from the source video.

Fallback transcript timeline, if available:
{transcript_timeline or "No local transcript timeline was available."}

JSON FORMAT:
{{
  "source_audio_contains_spoken_commentary": true,
  "language": "detected language",
  "summary": "brief summary of the original spoken commentary and relevant audible events",
  "timeline": [
    {{"start": 0.0, "end": 8.0, "audio_context": "what the narrator or source audio communicates here"}}
  ]
}}
"""
    try:
        text = _call_openai_compatible_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + _openai_audio_content_parts(audio_path, audio_mode),
            }],
            max_tokens=2500,
            timeout_seconds=OPENAI_AUDIO_ANALYSIS_TIMEOUT_SECONDS,
            response_format={"type": "json_object"},
        )
        parsed = _parse_openai_json(text)
        if isinstance(parsed, dict):
            parsed["provider"] = "openai_compatible_audio"
            parsed["model"] = model
            parsed["analyzed_seconds"] = analysis_duration
            return parsed
    except Exception as exc:
        if progress:
            progress(f"OpenAI-compatible audio analysis failed; using transcript fallback: {str(exc)[:300]}")
    return None


def _openai_source_audio_analysis_prompt_text(analysis: Optional[Dict]) -> str:
    if not isinstance(analysis, dict):
        return ""
    compact = {
        key: analysis.get(key)
        for key in (
            "provider",
            "model",
            "analyzed_seconds",
            "source_audio_contains_spoken_commentary",
            "language",
            "summary",
            "timeline",
        )
        if analysis.get(key) not in (None, "", [], {})
    }
    text = json.dumps(compact, ensure_ascii=False)
    return _limit_text_chars(text, 12000)


def _source_audio_analysis_has_spoken_commentary(analysis: Optional[Dict]) -> bool:
    if not isinstance(analysis, dict):
        return False
    explicit = analysis.get("source_audio_contains_spoken_commentary")
    if isinstance(explicit, bool):
        return explicit
    if isinstance(explicit, str):
        normalized = explicit.strip().lower()
        if normalized in {"true", "yes", "spoken", "commentary", "narration"}:
            return True
        if normalized in {"false", "no", "none", "music_only", "ambient_only"}:
            return False
    timeline = analysis.get("timeline") or []
    if isinstance(timeline, list) and len(timeline) >= 3:
        text = " ".join(str(item.get("audio_context") or item.get("text") or "") for item in timeline if isinstance(item, dict))
        return len(re.sub(r"\s+", "", text)) >= 80
    summary = str(analysis.get("summary") or "")
    return len(re.sub(r"\s+", "", summary)) >= 120


def _format_source_audio_analysis_timeline(
    analysis: Optional[Dict],
    max_items: int = 80,
    max_chars: int = 9000,
    start: Optional[float] = None,
    end: Optional[float] = None,
    margin: float = 0.0,
) -> str:
    if not isinstance(analysis, dict):
        return ""
    items = []
    for raw in analysis.get("timeline") or []:
        if not isinstance(raw, dict):
            continue
        text = re.sub(
            r"\s+",
            " ",
            str(raw.get("audio_context") or raw.get("text") or raw.get("summary") or "").strip(),
        )
        if not text:
            continue
        try:
            item_start = float(raw.get("start"))
            item_end = float(raw.get("end"))
        except (TypeError, ValueError):
            continue
        if item_end <= item_start:
            continue
        items.append({"start": item_start, "end": item_end, "text": text})
    if not items:
        summary = re.sub(r"\s+", " ", str(analysis.get("summary") or "").strip())
        return _limit_text_chars(summary, max_chars) if summary else ""
    if start is not None or end is not None:
        low = float(start if start is not None else 0.0) - max(0.0, float(margin or 0.0))
        high = float(end if end is not None else max((item["end"] for item in items), default=0.0)) + max(0.0, float(margin or 0.0))
        items = [item for item in items if item["end"] >= low and item["start"] <= high]
        if not items:
            return ""
    if len(items) > max_items:
        step = max(1, math.ceil(len(items) / max_items))
        items = items[::step][:max_items]
    lines = []
    used = 0
    for item in items:
        line = f"{item['start']:.2f}-{item['end']:.2f}: {_limit_text_chars(item['text'], 240)}"
        next_used = used + len(line) + 1
        if next_used > max_chars:
            break
        lines.append(line)
        used = next_used
    return "\n".join(lines)


def _non_full_target_duration_config(target_duration: str) -> Optional[Dict]:
    return NON_FULL_TARGET_DURATIONS.get(str(target_duration or "").strip())


def _is_non_full_target_duration(target_duration: str) -> bool:
    return _non_full_target_duration_config(target_duration) is not None


def _non_full_target_duration_label(target_duration: str) -> str:
    config = _non_full_target_duration_config(target_duration)
    return str((config or {}).get("label") or "compact")


def _target_duration_hint(mode: str, source_duration: float, target_seconds: Optional[float] = None) -> str:
    non_full_config = _non_full_target_duration_config(mode)
    if non_full_config:
        min_seconds, max_seconds = _target_duration_window_seconds(source_duration, mode)
        label = str(non_full_config["label"])
        importance = "only the most important" if mode == "short" else "enough important"
        return (
            f"AI must select {importance} visual moments for a {label} commentary edit; "
            f"narration_blocks/edit_segments playable time after video_speed must not exceed {int(max_seconds)} seconds. "
            f"Aim for {int(min_seconds)}-{int(max_seconds)} seconds only when the source has enough useful non-repetitive material; "
            "backend will reject oversized results instead of inventing fallback kept ranges."
        )
    full_target = float(target_seconds) if target_seconds and target_seconds > 0 else _target_visual_duration_seconds(source_duration, "full")
    if _full_mode_preserves_source_process(source_duration, full_target):
        return (
            "Create a comprehensive full-process commentary edit. For this source length, preserve the complete visible workflow in chronological order, "
            "remove only clearly useless dead time, duplicate waiting, setup, walking, camera drift, or failed/irrelevant footage, and use video_speed for visibly slow or repetitive ranges instead of cutting away important process steps. "
            "AI must decide the kept source ranges and splice order from the visual evidence; backend will validate and render those ranges, not choose replacements. "
            "Narration should be as detailed as needed to make each timestamped visual section clear; do not intentionally over-compress the explanation."
        )
    return (
        "Create a comprehensive long-form commentary edit with an explicit editing strategy, not a raw full-length copy of the source. "
        f"For this source, select about {int(full_target)} seconds of useful original footage across the whole timeline, preserving the complete process arc while removing repetitive, slow, duplicated, waiting, setup, walking, camera drift, and low-value filler time. "
        "Do not preserve the entire source unless the source itself is already shorter than the target, but do not intentionally make the explanation terse. AI must decide the kept source ranges, skipped ranges, splice order, and video_speed from the timestamped visual evidence; backend will validate and render those ranges, not choose replacements. Use video_speed for slow-but-useful ranges; narration must clearly explain what is happening in each kept timestamp range."
    )


def _style_grounding_instruction(style: str, language: str) -> str:
    normalized = (style or "").strip().lower()
    is_zh = (language or "").lower().startswith("zh")
    if normalized in {"documentary", "纪录片解说", "纪录片", "documentary_commentary"}:
        if is_zh:
            return (
                "纪录片解说风格要求：使用冷静、沉稳、客观的旁白语气，像纪录片解说员在解释现场过程。"
                "每个段落先交代当前画面的环境、主体、动作和变化，再补充基于画面证据的背景解释或意义判断。"
                "避免网络梗、夸张口头禅、第一人称代入和无证据煽情；节奏可以有铺垫，但事实必须来自当前时间戳画面或转写。"
            )
        return (
            "Documentary style: use a calm, observant narrator voice. First describe the environment, subjects, actions, and visible changes in the timestamp range, "
            "then add grounded context or significance. Avoid memes, first-person roleplay, unsupported emotion, and claims not supported by the current evidence."
        )
    if normalized in {"news", "newscast", "news_reading", "新闻解读", "新闻播读", "新闻播报"}:
        if is_zh:
            return (
                "新闻播读风格要求：使用清晰、克制、信息密度高的新闻口播语气。"
                "每个段落按“当前画面事实 -> 进展/影响 -> 下一步看点”的顺序组织，句子利落，不玩梗，不第一人称表演。"
                "不得编造地点、机构、数字、伤亡、原因、结论或采访信息；没有证据的内容只能写成画面可见情况。"
            )
        return (
            "News reading style: use a clear, restrained, information-dense broadcast voice. Organize each block as visible fact, development or implication, then next point of interest. "
            "Do not invent locations, organizations, numbers, injuries, causes, conclusions, or interview details."
        )
    if normalized in {"storytelling", "story", "故事化旁白", "故事旁白"}:
        if is_zh:
            return (
                "故事化旁白风格要求：把视频按时间线写成有起承转合的现场故事，但只能讲画面和转写能支持的内容。"
                "每个段落要先说清楚当前画面发生了什么，再用悬念、转折、铺垫或结果推进叙事。"
                "可以有情绪和节奏，但不能虚构人物身份、心理活动、前因后果、结局或画面外剧情。"
            )
        return (
            "Storytelling style: shape the timeline into a clear beginning, development, turn, and payoff, while only narrating what the frames or transcript support. "
            "First state the visible action in each block, then use suspense, contrast, or payoff to advance the story without inventing motives or off-screen events."
        )
    if normalized in {"educational", "explainer", "knowledge", "知识科普", "科普解说"}:
        if is_zh:
            return (
                "知识科普风格要求：用通俗、准确的解释型口吻，把当前画面里的工具、材料、步骤、原理、风险或结果讲清楚。"
                "每个段落先描述时间戳内可见动作，再解释这个动作可能对应的工序、用途或注意点；解释必须受画面证据约束。"
                "不确定的专业名称、数据、因果或效果不要断言，可以用“看起来”“可能是”“从画面能确认的是”。"
            )
        return (
            "Educational explainer style: use a clear teaching voice. First describe the visible action, tools, materials, steps, risks, or result in the timestamp range, "
            "then explain the likely process or purpose while marking uncertainty instead of asserting unsupported technical details."
        )
    if normalized in {"first_person_hustle", "first-person-hustle", "整活第一视角", "第一视角整活"}:
        if is_zh:
            return (
                "整活第一视角风格要求：把解说写成正在亲自参与画面动作的第一人称口播，像边干活边碎碎念，"
                "节奏紧、短句多、反应真实。口头禅只能点到为止，同一句口头禅全片最多出现两次，禁止反复写“没事的没事的”。"
                "每个 narration_blocks 段落必须先抓住当前可见的具体动作、位置、风险、工具、材料或结果，"
                "再用略怂但硬上的语气做即时反应；所有夸张、吐槽和心理活动都必须绑定当前可见内容，不要编造看不见的剧情、身份、收益或危险。"
            )
        return (
            "First-person hustle style: write the narration as if the speaker is personally doing the visible task in real time. "
            "Use short, energetic first-person reactions, nervous confidence, and immediate observations. Avoid repeating the same catchphrase more than twice, "
            "but ground every joke, exaggeration, risk, tool, material, and result in the current visible timestamp range."
        )
    if normalized in {"hustle", "fun_hustle", "整活解说", "整活"}:
        if is_zh:
            return (
                "整活解说风格要求：保持第三人称或旁观者口播，不要装成正在参与动作的人。"
                "先说清楚当前画面正在发生什么，再用短促、有梗、带反差的方式吐槽或强化看点。"
                "梗必须来自当前可见的动作、表情、风险、工具、材料、环境或结果；口头禅全片最多点两次，禁止同一句反复刷屏。"
            )
        return (
            "Hustle commentary style: use an energetic observer voice, not first person. First describe the visible action, then add a short joke or punchy reaction grounded in the same timestamp range. Avoid repeating the same catchphrase."
        )
    if normalized in {"funny", "roast", "吐槽", "轻松吐槽"}:
        if is_zh:
            return (
                "轻松吐槽风格要求：每个 narration_blocks 段落先描述这个时间段正在发生的具体画面，再基于同一画面做轻松吐槽。"
                "可以做国际工厂/海外回收流程与中国工厂/中国回收效率的对比，但对比必须围绕当前画面，围绕当前可见的材料、设备、人工动作、工序节奏或安全细节；"
                "不要写脱离画面的国际形势、宏大政治、地域刻板印象或没有画面证据的段子。"
            )
        return (
            "Funny/roast style: each narration block must first describe the visible action in that exact range, then make a light joke grounded in the same visible material, equipment, worker action, process rhythm, or safety detail. Avoid unrelated world affairs, politics, stereotypes, or claims not supported by the frame."
        )
    return "Keep the selected style grounded in the visible action of each timestamped range."


def _custom_style_instruction(custom_style_prompt: Optional[str]) -> str:
    prompt = re.sub(r"\s+", " ", (custom_style_prompt or "").strip())
    if not prompt:
        return ""
    prompt = prompt[:2000]
    return (
        "\n- Custom user style instruction is a binding production requirement, not a loose tone hint. "
        "Apply it to every narrated block's wording, pacing, point of view, and explanatory depth while preserving visual grounding, timeline sync, factuality, safety, and JSON schema rules. "
        "If the custom style asks for process, operation, purpose, why, logic, or industrial/mechanical explanation, each ordinary process block must first name the visible action/tool/material, then add a concise purpose, operation logic, or visible result for that same timestamp range. "
        "Do not return action-only labels when the custom style asks for explanation. "
        "Use only current block visual_facts, evidence_timestamps, visible frame evidence, or transcript evidence for any purpose/result claim. Custom prompt: "
        f"{prompt}"
    )


def _normalize_edit_segments(raw_segments: List[Dict], duration: float) -> List[Dict]:
    normalized = []
    for item in raw_segments or []:
        try:
            start = float(item.get("start", 0))
            end = float(item.get("end", 0))
        except (TypeError, ValueError):
            continue
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end - start < 1.0:
            continue
        speed = _safe_video_speed(item.get("video_speed") or item.get("speed") or item.get("suggested_speed"))
        normalized.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "reason": str(item.get("reason") or item.get("title") or "selected visual segment"),
            "video_speed": speed,
            "speed_reason": str(item.get("speed_reason") or "").strip(),
        })
    normalized.sort(key=lambda segment: segment["start"])
    merged = []
    for segment in normalized:
        if (
            merged
            and segment["start"] <= merged[-1]["end"] + 0.3
            and abs(float(segment.get("video_speed") or 1.0) - float(merged[-1].get("video_speed") or 1.0)) < 0.001
        ):
            merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
            if segment.get("reason") and segment["reason"] not in merged[-1].get("reason", ""):
                merged[-1]["reason"] = f"{merged[-1]['reason']}; {segment['reason']}"
            if segment.get("speed_reason") and segment["speed_reason"] not in merged[-1].get("speed_reason", ""):
                merged[-1]["speed_reason"] = f"{merged[-1].get('speed_reason') or ''}; {segment['speed_reason']}".strip("; ")
        else:
            merged.append(segment)
    return merged


def _target_visual_duration_seconds(source_duration: float, target_duration: str) -> float:
    duration = max(0.0, float(source_duration or 0.0))
    non_full_config = _non_full_target_duration_config(target_duration)
    if non_full_config:
        return min(duration, float(non_full_config["max_seconds"]))
    if duration <= 0:
        return 0.0
    if duration <= FULL_MODE_FULL_SOURCE_UNDER_SECONDS:
        return duration
    if duration <= FULL_MODE_COMPACT_SOURCE_UNDER_SECONDS:
        return min(duration, max(FULL_MODE_MIN_VISUAL_SECONDS, duration * FULL_MODE_COMPACT_SOURCE_FRACTION))
    fractional_target = min(duration * FULL_MODE_SOURCE_FRACTION, FULL_MODE_MAX_VISUAL_SECONDS)
    return min(
        duration,
        max(
            FULL_MODE_LONG_MIN_VISUAL_SECONDS,
            fractional_target,
        ),
    )


def _target_duration_window_seconds(source_duration: float, target_duration: str) -> Tuple[float, float]:
    duration = max(0.0, float(source_duration or 0.0))
    if duration <= 0:
        return (0.0, 0.0)
    non_full_config = _non_full_target_duration_config(target_duration)
    if non_full_config:
        return (
            min(duration, float(non_full_config["min_seconds"])),
            min(duration, float(non_full_config["max_seconds"])),
        )
    target = _target_visual_duration_seconds(duration, target_duration)
    return (target, target)


def _normalized_visual_candidate_segments(
    visual_analysis: Optional[Dict],
    duration: float,
    merge_gap_seconds: float = 0.5,
) -> List[Tuple[float, float]]:
    if not visual_analysis:
        return []
    source_duration = max(0.0, float(duration or 0.0))
    segments = []
    for item in visual_analysis.get("candidate_segments") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("edit_value") or "").strip().lower() == "skippable":
            continue
        if visual_analysis.get("provider") == "openai_compatible" and _candidate_segment_importance(item, visual_analysis) < 2.5:
            continue
        try:
            start = max(0.0, min(source_duration, float(item.get("start"))))
            end = max(0.0, min(source_duration, float(item.get("end"))))
        except (TypeError, ValueError):
            continue
        if end - start >= 0.75:
            segments.append((start, end))
    segments.sort()
    merged = []
    for start, end in segments:
        if merged and start <= merged[-1][1] + merge_gap_seconds:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _visual_candidate_duration_seconds(visual_analysis: Optional[Dict], duration: float) -> float:
    return sum(end - start for start, end in _normalized_visual_candidate_segments(visual_analysis, duration))


def _visual_candidate_playable_seconds(visual_analysis: Optional[Dict], duration: float) -> float:
    if not visual_analysis:
        return 0.0
    source_duration = max(0.0, float(duration or 0.0))
    total = 0.0
    for item in visual_analysis.get("candidate_segments") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("edit_value") or "").strip().lower() == "skippable":
            continue
        if visual_analysis.get("provider") == "openai_compatible" and _candidate_segment_importance(item, visual_analysis) < 2.5:
            continue
        try:
            start = max(0.0, min(source_duration, float(item.get("start"))))
            end = max(0.0, min(source_duration, float(item.get("end"))))
        except (TypeError, ValueError):
            continue
        source_seconds = end - start
        if source_seconds < 0.75:
            continue
        total += source_seconds / _candidate_segment_speed(item)
    return total


def _visual_candidate_timeline_bucket_count(visual_analysis: Optional[Dict], duration: float) -> int:
    source_duration = max(0.0, float(duration or 0.0))
    if source_duration <= 0:
        return 0
    buckets = set()
    for start, end in _normalized_visual_candidate_segments(visual_analysis, source_duration):
        midpoint = (start + end) / 2.0
        buckets.add(min(2, int((midpoint / source_duration) * 3.0)))
    return len(buckets)


def _visual_analysis_keep_candidate_counts(visual_analysis: Optional[Dict]) -> Tuple[int, int]:
    if not visual_analysis:
        return (0, 0)
    keep_count = 0
    high_importance_count = 0
    for item in visual_analysis.get("observations") or []:
        if not isinstance(item, dict) or not bool(item.get("keep_candidate")):
            continue
        keep_count += 1
        try:
            importance = int(item.get("importance") or 0)
        except (TypeError, ValueError):
            importance = 0
        if importance >= 4:
            high_importance_count += 1
    return keep_count, high_importance_count


def _visual_analysis_observation_text_for_range(
    visual_analysis: Optional[Dict],
    start: float,
    end: float,
    limit: int = 4,
) -> List[str]:
    if not visual_analysis:
        return []
    observations = []
    for item in visual_analysis.get("observations") or []:
        if not isinstance(item, dict):
            continue
        try:
            timestamp = float(item.get("timestamp"))
        except (TypeError, ValueError):
            continue
        if not (start <= timestamp <= end):
            continue
        visual = re.sub(r"\s+", " ", str(item.get("visual") or "").strip())
        reason = re.sub(r"\s+", " ", str(item.get("reason") or "").strip())
        text = visual or reason
        if text:
            text = f"{timestamp:.3f}s: {text}"
        if text:
            observations.append(text)
        if len(observations) >= limit:
            break
    return observations


def _coerce_visual_score(value, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(5.0, score))


def _segment_overlap_seconds(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _candidate_segment_observations(segment: Dict, visual_analysis: Optional[Dict]) -> List[Dict]:
    try:
        start = float(segment.get("start"))
        end = float(segment.get("end"))
    except (TypeError, ValueError):
        return []
    observations = []
    for item in (visual_analysis or {}).get("observations") or []:
        if not isinstance(item, dict):
            continue
        try:
            timestamp = float(item.get("timestamp"))
        except (TypeError, ValueError):
            continue
        if start <= timestamp <= end:
            observations.append(item)
    return observations


def _candidate_segment_importance(segment: Dict, visual_analysis: Optional[Dict]) -> float:
    try:
        start = float(segment.get("start"))
        end = float(segment.get("end"))
    except (TypeError, ValueError):
        return 0.0
    score = 0.0
    direct_importance = max(
        _coerce_visual_score(segment.get("importance")),
        _coerce_visual_score(segment.get("importance_score")),
        _coerce_visual_score(segment.get("visual_importance")),
    )
    direct_interest = max(
        _coerce_visual_score(segment.get("interest_score")),
        _coerce_visual_score(segment.get("viewer_interest")),
        _coerce_visual_score(segment.get("watch_value")),
    )
    score += direct_importance
    score += direct_interest * 0.8
    if bool(segment.get("keep_candidate")):
        score += 1.0
    edit_value = str(segment.get("edit_value") or "").lower()
    if edit_value == "must_keep":
        score += 2.5
    elif edit_value == "useful":
        score += 1.0
    elif edit_value == "skippable":
        score -= 3.0
    observations = _candidate_segment_observations(segment, visual_analysis)
    if observations:
        importances = []
        interests = []
        for item in observations:
            try:
                importances.append(float(item.get("importance") or 0.0))
            except (TypeError, ValueError):
                pass
            for key in ("interest_score", "viewer_interest", "watch_value"):
                try:
                    interests.append(float(item.get(key) or 0.0))
                except (TypeError, ValueError):
                    pass
            if bool(item.get("keep_candidate")):
                score += 0.35
            edit_value = str(item.get("edit_value") or "").lower()
            if edit_value == "must_keep":
                score += 1.0
            elif edit_value == "useful":
                score += 0.45
            elif edit_value == "skippable":
                score -= 0.5
        if importances:
            score += max(importances) / 5.0
        if interests:
            score += max(interests) / 6.0
    return score


def _candidate_segment_priority(
    segment: Dict,
    source_duration: float,
    selected: Optional[List[Dict]] = None,
) -> Tuple[float, float, float, float, float]:
    start = float(segment.get("start") or 0.0)
    end = float(segment.get("end") or start)
    midpoint = (start + end) / 2.0
    duration = max(1.0, float(source_duration or 0.0))
    overlap_penalty = 0.0
    for existing in selected or []:
        existing_start = float(existing.get("start") or 0.0)
        existing_end = float(existing.get("end") or existing_start)
        overlap_penalty += _segment_overlap_seconds(start, end, existing_start, existing_end)
    return (
        float(segment.get("importance") or 0.0) - overlap_penalty * 0.2,
        -overlap_penalty,
        midpoint / duration,
        -float(segment.get("playable_seconds") or 0.0),
        -start,
    )


def _candidate_segment_frame_timestamps(
    segment: Dict,
    visual_analysis: Optional[Dict],
    limit: int = 4,
) -> List[float]:
    if not visual_analysis:
        return []
    try:
        start = float(segment.get("start"))
        end = float(segment.get("end"))
    except (TypeError, ValueError):
        return []
    candidates = []
    for value in segment.get("evidence_timestamps") or []:
        try:
            timestamp = round(float(value), 3)
        except (TypeError, ValueError):
            continue
        if start - 0.35 <= timestamp <= end + 0.35:
            candidates.append(timestamp)
    for item in _candidate_segment_observations(segment, visual_analysis):
        try:
            candidates.append(round(float(item.get("timestamp")), 3))
        except (TypeError, ValueError):
            continue
    frame_timestamps = _visual_analysis_frame_timestamps(visual_analysis)
    if frame_timestamps:
        for timestamp in frame_timestamps:
            if start - 0.35 <= timestamp <= end + 0.35:
                candidates.append(round(float(timestamp), 3))
    unique = []
    for timestamp in candidates:
        if timestamp not in unique:
            unique.append(timestamp)
        if len(unique) >= limit:
            break
    return unique


def _candidate_segment_bucket(segment: Dict, source_duration: float) -> int:
    duration = max(1.0, float(source_duration or 0.0))
    start = float(segment.get("start") or 0.0)
    end = float(segment.get("end") or start)
    midpoint = (start + end) / 2.0
    return min(2, max(0, int((midpoint / duration) * 3.0)))


def _candidate_segment_speed(segment: Dict) -> float:
    return _safe_video_speed(
        segment.get("suggested_speed", segment.get("video_speed", segment.get("speed", segment.get("recommended_speed"))))
    )


def _openai_plan_segment_id(segment: Dict) -> int:
    try:
        return int(segment.get("candidate_index"))
    except (TypeError, ValueError):
        return id(segment)


def _openai_plan_latest_end(segments: List[Dict]) -> float:
    return max((float(segment.get("end") or 0.0) for segment in segments or []), default=0.0)


def _openai_plan_total_playable_seconds(segments: List[Dict]) -> float:
    return sum(max(0.0, float(segment.get("playable_seconds") or 0.0)) for segment in segments or [])


def _openai_plan_late_candidate_priority(segment: Dict, source_duration: float) -> Tuple[float, float, float, float]:
    duration = max(1.0, float(source_duration or 0.0))
    return (
        float(segment.get("importance") or 0.0),
        float(segment.get("interest_score") or 0.0),
        float(segment.get("end") or 0.0) / duration,
        -float(segment.get("playable_seconds") or 0.0),
    )


def _trim_openai_candidate_selection_to_window(
    selected: List[Dict],
    total: float,
    min_seconds: float,
    max_seconds: float,
    required_latest_end: float,
) -> Tuple[List[Dict], float]:
    while total > max_seconds + FULL_MODE_VALIDATION_EPSILON_SECONDS and selected:
        removable_indexes = []
        for idx, segment in enumerate(selected):
            if len(selected) > 2 and idx in {0, len(selected) - 1}:
                continue
            candidate_total = total - float(segment.get("playable_seconds") or 0.0)
            if candidate_total < min_seconds - FULL_MODE_VALIDATION_EPSILON_SECONDS:
                continue
            remaining = selected[:idx] + selected[idx + 1:]
            if required_latest_end > 0 and _openai_plan_latest_end(remaining) < required_latest_end:
                continue
            removable_indexes.append(idx)
        if not removable_indexes:
            break
        best_index = min(
            removable_indexes,
            key=lambda idx: (
                float(selected[idx].get("importance") or 0.0),
                -float(selected[idx].get("playable_seconds") or 0.0),
            ),
        )
        total -= float(selected[best_index].get("playable_seconds") or 0.0)
        selected.pop(best_index)
    return selected, total


def _split_openai_candidate_segments_for_sync(segments: List[Dict]) -> List[Dict]:
    max_playable = max(6.0, float(FULL_MODE_OPENAI_PLAN_MAX_BLOCK_PLAYABLE_SECONDS or 18.0))
    split_segments = []
    for segment in segments or []:
        playable = max(0.0, float(segment.get("playable_seconds") or 0.0))
        source_seconds = max(0.0, float(segment.get("source_seconds") or 0.0))
        if playable <= max_playable + FULL_MODE_VALIDATION_EPSILON_SECONDS or source_seconds <= 1.0:
            split_segments.append(segment)
            continue
        part_count = max(2, int(math.ceil(playable / max_playable)))
        source_start = float(segment.get("start") or 0.0)
        source_end = float(segment.get("end") or source_start)
        source_step = (source_end - source_start) / part_count
        for part_index in range(part_count):
            part_start = source_start + source_step * part_index
            part_end = source_end if part_index == part_count - 1 else source_start + source_step * (part_index + 1)
            part_source_seconds = max(0.0, part_end - part_start)
            if part_source_seconds < 0.75:
                continue
            part = dict(segment)
            part["start"] = part_start
            part["end"] = part_end
            part["source_seconds"] = part_source_seconds
            part["playable_seconds"] = part_source_seconds / _safe_video_speed(segment.get("video_speed"))
            part["split_part_index"] = part_index + 1
            part["split_part_count"] = part_count
            split_segments.append(part)
    return split_segments


def _build_openai_candidate_edit_plan(
    visual_analysis: Optional[Dict],
    duration: float,
    target_duration: str,
    language: str,
) -> Optional[Dict]:
    if (target_duration != "full" and not _is_non_full_target_duration(target_duration)) or not visual_analysis:
        return None
    source_duration = max(0.0, float(duration or 0.0))
    target_seconds = _target_visual_duration_seconds_for_analysis(source_duration, target_duration, visual_analysis)
    if target_seconds <= 0 or _full_mode_preserves_source_process(source_duration, target_seconds):
        return None
    if _is_non_full_target_duration(target_duration):
        min_seconds, max_seconds = _target_duration_window_seconds(source_duration, target_duration)
    else:
        min_seconds = _full_mode_min_playable_visual_seconds(source_duration, target_seconds)
        max_seconds = _full_mode_max_playable_visual_seconds(source_duration, target_seconds)
    raw_segments = []
    candidate_index = 0
    for item in visual_analysis.get("candidate_segments") or []:
        if not isinstance(item, dict):
            continue
        try:
            start = max(0.0, min(source_duration, float(item.get("start"))))
            end = max(0.0, min(source_duration, float(item.get("end"))))
        except (TypeError, ValueError):
            continue
        if end - start < 1.0:
            continue
        speed = _candidate_segment_speed(item)
        candidate_index += 1
        evidence_timestamps = _candidate_segment_frame_timestamps(item, visual_analysis)
        raw_segments.append({
            "candidate_index": candidate_index,
            "start": start,
            "end": end,
            "source_seconds": end - start,
            "video_speed": speed,
            "playable_seconds": (end - start) / speed,
            "reason": str(item.get("reason") or "AI visual candidate").strip(),
            "speed_reason": str(item.get("speed_reason") or "").strip(),
            "importance": _candidate_segment_importance(item, visual_analysis),
            "edit_value": str(item.get("edit_value") or "").strip(),
            "evidence_timestamps": evidence_timestamps,
        })
    if len(raw_segments) < 4:
        return None
    raw_segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    if _is_non_full_target_duration(target_duration):
        preferred_seconds = min(max_seconds, max(min_seconds, target_seconds))
    else:
        preferred_seconds = min(max_seconds - 4.0, max(min_seconds + 24.0, target_seconds))
    if preferred_seconds < min_seconds:
        preferred_seconds = min_seconds
    selected = []
    total = 0.0
    min_candidate_importance = 2.5
    for bucket in (0, 1, 2):
        bucket_segments = [segment for segment in raw_segments if _candidate_segment_bucket(segment, source_duration) == bucket]
        if not bucket_segments:
            continue
        segment = max(bucket_segments, key=lambda item: _candidate_segment_priority(item, source_duration, selected))
        if float(segment.get("importance") or 0.0) < min_candidate_importance:
            continue
        if _openai_plan_segment_id(segment) in {_openai_plan_segment_id(item) for item in selected}:
            continue
        selected.append(segment)
        total += float(segment.get("playable_seconds") or 0.0)

    while total < preferred_seconds:
        selected_ids = {_openai_plan_segment_id(segment) for segment in selected}
        remaining = [
            segment
            for segment in raw_segments
            if _openai_plan_segment_id(segment) not in selected_ids
        ]
        if not remaining:
            break
        segment = max(remaining, key=lambda item: _candidate_segment_priority(item, source_duration, selected))
        if float(segment.get("importance") or 0.0) < min_candidate_importance:
            if total >= min_seconds:
                break
            return None
        selected.append(segment)
        total += float(segment.get("playable_seconds") or 0.0)
    if total < min_seconds:
        return None
    required_latest_end = 0.0
    if not _full_mode_preserves_source_process(source_duration, target_seconds) and source_duration > target_seconds * 1.6:
        required_latest_end = source_duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION
    if required_latest_end > 0 and _openai_plan_latest_end(selected) < required_latest_end:
        selected_ids = {_openai_plan_segment_id(segment) for segment in selected}
        late_candidates = [
            segment
            for segment in raw_segments
            if _openai_plan_segment_id(segment) not in selected_ids and float(segment.get("end") or 0.0) >= required_latest_end
        ]
        if not late_candidates:
            return None
        late_segment = max(
            late_candidates,
            key=lambda segment: _openai_plan_late_candidate_priority(segment, source_duration),
        )
        selected.append(late_segment)
        total += float(late_segment.get("playable_seconds") or 0.0)
    selected, total = _trim_openai_candidate_selection_to_window(
        selected,
        total,
        min_seconds,
        max_seconds,
        required_latest_end,
    )
    selected.sort(key=lambda segment: (float(segment.get("start") or 0.0), float(segment.get("end") or 0.0)))
    if not (min_seconds <= total <= max_seconds):
        return None
    if required_latest_end > 0 and _openai_plan_latest_end(selected) < required_latest_end:
        return None
    selected_blocks = _split_openai_candidate_segments_for_sync(selected)
    block_total = _openai_plan_total_playable_seconds(selected_blocks)
    blocks = []
    for index, segment in enumerate(selected_blocks, start=1):
        observations = _visual_analysis_observation_text_for_range(
            visual_analysis,
            segment["start"],
            segment["end"],
        )
        split_note = ""
        if int(segment.get("split_part_count") or 0) > 1:
            split_note = (
                f"AI-selected subrange {int(segment.get('split_part_index') or 1)}/"
                f"{int(segment.get('split_part_count') or 1)} of this useful visual candidate"
            )
        visual_parts = [segment["reason"], split_note] + observations
        visual = _limit_text_chars(
            re.sub(r"\s+", " ", " | ".join(part for part in visual_parts if part)).strip(),
            280,
        )
        speed_reason = segment["speed_reason"]
        if segment["video_speed"] > 1.0001 and not speed_reason:
            speed_reason = "AI visual analysis marked this exact range as slow or repetitive enough for acceleration"
        evidence_timestamps = _candidate_segment_frame_timestamps(segment, visual_analysis)
        if not evidence_timestamps:
            evidence_timestamps = [round((segment["start"] + segment["end"]) / 2.0, 3)]
        blocks.append({
            "index": index,
            "start": round(segment["start"], 3),
            "end": round(segment["end"], 3),
            "visual": visual or "AI-selected useful visual range",
            "visual_facts": observations[:3] or [segment["reason"] or "AI-selected useful visual range"],
            "evidence_timestamps": evidence_timestamps,
            "pause": False,
            "video_speed": segment["video_speed"],
            "speed_reason": speed_reason,
            "playable_seconds": round(segment["playable_seconds"], 3),
            "min_narration_chars": _minimum_sync_narration_chars_for_visual_duration(segment["playable_seconds"], language),
        })
    return {
        "target_seconds": target_seconds,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "playable_seconds": block_total,
        "source_seconds": sum(segment["source_seconds"] for segment in selected),
        "blocks": blocks,
    }


def _target_visual_duration_seconds_for_analysis(
    source_duration: float,
    target_duration: str,
    visual_analysis: Optional[Dict] = None,
) -> float:
    base_target = _target_visual_duration_seconds(source_duration, target_duration)
    if target_duration != "full" or not visual_analysis:
        return base_target
    duration = max(0.0, float(source_duration or 0.0))
    if duration <= 0 or _full_mode_preserves_source_process(duration, base_target):
        return base_target

    candidate_seconds = _visual_candidate_duration_seconds(visual_analysis, duration)
    if candidate_seconds < 30.0:
        return base_target

    bucket_count = _visual_candidate_timeline_bucket_count(visual_analysis, duration)
    keep_count, high_importance_count = _visual_analysis_keep_candidate_counts(visual_analysis)
    multiplier = 1.02
    if bucket_count >= 3:
        multiplier += 0.06
    elif bucket_count >= 2:
        multiplier += 0.03
    if keep_count >= 45 and high_importance_count >= 12:
        multiplier += 0.03

    content_target = candidate_seconds * multiplier
    floor = FULL_MODE_LONG_MIN_VISUAL_SECONDS if duration > FULL_MODE_COMPACT_SOURCE_UNDER_SECONDS else FULL_MODE_MIN_VISUAL_SECONDS
    content_target = max(floor, content_target)
    candidate_playable_seconds = _visual_candidate_playable_seconds(visual_analysis, duration)
    if (
        visual_analysis.get("provider") == "openai_compatible"
        and candidate_playable_seconds >= 30.0
        and not _full_mode_preserves_source_process(duration, base_target)
    ):
        candidate_supported_target = (
            max(0.0, candidate_playable_seconds - FULL_MODE_VALIDATION_EPSILON_SECONDS)
            / max(0.01, FULL_MODE_MIN_PLAYABLE_TARGET_RATIO)
        )
        content_target = min(content_target, candidate_supported_target)
    return round(min(base_target, duration, content_target), 3)


def _full_mode_preserves_source_process(duration: float, target_seconds: float) -> bool:
    duration = max(0.0, float(duration or 0.0))
    target_seconds = max(0.0, float(target_seconds or 0.0))
    return duration > 0 and target_seconds >= duration * 0.9


def _full_mode_min_playable_visual_seconds(duration: float, target_seconds: float) -> float:
    if target_seconds <= 0:
        return 0.0
    if _full_mode_preserves_source_process(duration, target_seconds):
        return target_seconds * FULL_MODE_FULL_PROCESS_MIN_PLAYABLE_RATIO
    return target_seconds * FULL_MODE_MIN_PLAYABLE_TARGET_RATIO


def _full_mode_max_playable_visual_seconds(duration: float, target_seconds: float) -> float:
    if target_seconds <= 0:
        return 0.0
    if _full_mode_preserves_source_process(duration, target_seconds):
        return target_seconds * 1.6
    return target_seconds * FULL_MODE_MAX_PLAYABLE_TARGET_RATIO


def _full_mode_visual_budget_tolerance_seconds(target_seconds: float) -> float:
    target_seconds = max(0.0, float(target_seconds or 0.0))
    scaled_tolerance = target_seconds * max(0.0, FULL_MODE_VISUAL_BUDGET_TOLERANCE_RATIO)
    configured_tolerance = max(0.0, FULL_MODE_VISUAL_BUDGET_TOLERANCE_SECONDS)
    return max(FULL_MODE_VALIDATION_EPSILON_SECONDS, min(configured_tolerance, scaled_tolerance))


def _full_mode_timeline_rules(duration: float, target_seconds: float) -> str:
    if _full_mode_preserves_source_process(duration, target_seconds):
        return f"""
- For TARGET DURATION full, first analyze the complete source visual timeline from 0.0 seconds through {duration:.1f} seconds in detail; do not stop after a short highlight scan, and do not summarize only the first few minutes.
- For TARGET DURATION full, this source is short enough to preserve the whole useful workflow in chronological order; remove only clearly useless dead time, duplicated waiting, setup, walking, camera drift, failed footage, or irrelevant ranges.
- For TARGET DURATION full, do not shrink this {int(duration)} second source into a much shorter highlights reel. Keep important process steps, transitions, preparation, action, result, and ending visible.
- For TARGET DURATION full, write clear scene-matched commentary over the preserved workflow. The visual analysis should be detailed, and the spoken narration should explain the timestamped on-screen actions without intentionally compressing them.
""".strip()
    return f"""
- For TARGET DURATION full, first analyze the complete source visual timeline from 0.0 seconds through {duration:.1f} seconds; do not stop after a short highlight scan, and do not summarize only the first few minutes.
- For TARGET DURATION full, produce a real edit decision list: select about {int(target_seconds)} seconds of useful visual ranges from the complete source timeline, and intentionally skip redundant or low-value ranges.
- For TARGET DURATION full, do not output one continuous 0-to-{duration:.1f} timeline. For this source, the selected playable visual duration should be near {int(target_seconds)} seconds, not {int(duration)} seconds.
- For TARGET DURATION full, preserve the complete process arc by keeping the beginning, middle, and ending payoff, but compress repeated manual actions, repeated tool operations, waiting, setup, walking, camera drift, and redundant close-ups with cuts and video_speed.
- For TARGET DURATION full, for example, compress repeated hammering, repeated climbing/setup motions, and other slow-but-useful process ranges with cuts or video_speed when the visual evidence supports it.
- For TARGET DURATION full, if the visual evidence shows the useful process is naturally tighter than the raw source, keep the edit tight instead of padding with weak ranges; the detailed analysis can be much longer than the spoken/kept edit.
- For TARGET DURATION full, write scene-matched commentary that covers the chosen visual ranges from start to finish across the full source timeline. Do not return a 60-second summary over a long source; explain the selected timestamp ranges clearly.
""".strip()


def _full_mode_regeneration_timeline_rules(duration: float, target_seconds: float) -> str:
    if _full_mode_preserves_source_process(duration, target_seconds):
        return (
            "- Keep chronological edit_segments that preserve the useful source workflow; remove only clearly useless dead time, duplicate waiting, setup, walking, camera drift, failed footage, or irrelevant ranges.\n"
            "- The final narration_blocks must include the beginning, middle, and ending portions of the source; for this source length, do not collapse the video into a much shorter highlights reel.\n"
            "- Let the AI decide video_speed from the visible action. Use video_speed above 1.0 for visibly slow or repetitive ranges instead of deleting meaningful process footage, and explain that decision in speed_reason.\n"
            "- Keep ordinary narrated blocks short enough for spoken commentary to cover them, usually 8-20s after video_speed. Split longer useful process ranges into multiple narrated blocks so the commentary can clearly explain each visible step."
        )
    return (
        f"- Keep chronological edit_segments that cover every major visible process stage across the source timeline while cutting repetitive, slow, duplicated, waiting, setup, walking, camera drift, and low-value filler ranges; selected playable visual time should be near {int(target_seconds)} seconds.\n"
        f"- The final narration_blocks must include selected source ranges from the beginning, middle, and later ending portion of the source; at least one block must end after {int(duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION)} seconds.\n"
        f"- Do not return a continuous near-full-source timeline; select about {int(target_seconds)} seconds of useful visuals, not {int(duration)} seconds. If the useful visual evidence is naturally shorter than the raw source, do not pad the edit with weak ranges.\n"
        "- Keep ordinary narrated blocks short enough for spoken commentary to cover them, usually 8-20s after video_speed. Split longer useful process ranges into multiple narrated blocks so the commentary can clearly explain each visible step."
    )


def _segments_total_duration(segments: List[Dict]) -> float:
    total = 0.0
    for segment in segments or []:
        try:
            total += max(0.0, float(segment["end"]) - float(segment["start"]))
        except (KeyError, TypeError, ValueError):
            continue
    return total


def _max_selected_source_seconds_for_real_cuts(duration: float, target_seconds: float) -> float:
    static_retention_limit = duration * FULL_MODE_MAX_SOURCE_RETENTION_FRACTION
    target_sized_limit = target_seconds * 1.60 if target_seconds > 0 else 0.0
    return min(duration * 0.9, max(static_retention_limit, target_sized_limit))


def _segments_have_real_cuts(segments: List[Dict], duration: float, target_seconds: float) -> bool:
    if duration <= 0 or target_seconds >= duration * 0.9:
        return True
    normalized = _normalize_edit_segments(segments, duration)
    total = _segments_total_duration(normalized)
    if total > _max_selected_source_seconds_for_real_cuts(duration, target_seconds):
        return False
    if len(normalized) <= 1 and total >= duration * 0.8:
        return False
    previous_end = 0.0
    removed_seconds = 0.0
    for segment in normalized:
        removed_seconds += max(0.0, float(segment["start"]) - previous_end)
        previous_end = max(previous_end, float(segment["end"]))
    removed_seconds += max(0.0, duration - previous_end)
    return removed_seconds >= max(30.0, duration - target_seconds * 1.6)


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _safe_edge_percent(value: str, minimum: int, maximum: int) -> str:
    match = re.match(r"^\s*([+-]?)(\d+(?:\.\d+)?)\s*%\s*$", str(value or ""))
    if not match:
        return "+0%"
    amount = int(round(float(match.group(2))))
    if match.group(1) == "-":
        amount *= -1
    amount = max(minimum, min(maximum, amount))
    return f"{amount:+d}%"


def _safe_edge_rate(value: str) -> str:
    return _safe_edge_percent(value, -30, 30)


def _safe_edge_pitch(value: str) -> str:
    match = re.match(r"^\s*([+-]?)(\d+(?:\.\d+)?)\s*Hz\s*$", str(value or ""), re.IGNORECASE)
    if not match:
        return "+0Hz"
    amount = int(round(float(match.group(2))))
    if match.group(1) == "-":
        amount *= -1
    amount = max(-15, min(15, amount))
    return f"{amount:+d}Hz"


def _safe_video_speed(value) -> float:
    if isinstance(value, bool):
        return 1.5 if value else 1.0
    if isinstance(value, str) and value.strip().lower() in {"true", "yes", "y", "false", "no", "n"}:
        return 1.0
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return round(max(1.0, min(2.5, speed)), 3)


def _safe_render_video_speed(value) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return round(max(1.0, min(FULL_MODE_RENDER_SYNC_MAX_VIDEO_SPEED, speed)), 3)


def _block_source_duration(block: Dict) -> float:
    return max(0.0, float(block.get("end") or 0.0) - float(block.get("start") or 0.0))


def _block_visual_duration(block: Dict) -> float:
    return max(0.0, _block_source_duration(block) / _safe_video_speed(block.get("video_speed")))


def _estimated_voiceover_seconds_for_chars(narration_chars: int, language: str) -> float:
    chars = max(0, int(narration_chars or 0))
    if chars <= 0:
        return 0.0
    if (language or "").lower().startswith("zh"):
        return chars / 4.2
    return chars / 12.0


def _minimum_voiceover_seconds_for_visual_duration(block_duration: float) -> float:
    visual_seconds = max(0.0, float(block_duration or 0.0))
    if visual_seconds <= 0:
        return 0.0
    min_ratio_seconds = visual_seconds * max(0.0, min(FULL_MODE_MIN_NARRATED_BLOCK_VOICEOVER_RATIO, 1.0))
    max_tail_seconds = max(0.0, FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS)
    min_tail_seconds = max(0.0, visual_seconds - max_tail_seconds)
    return max(min_ratio_seconds, min_tail_seconds)


def _max_visual_seconds_for_actual_voiceover(voice_seconds: float) -> float:
    voice_seconds = max(0.0, float(voice_seconds or 0.0))
    if voice_seconds <= 0:
        return 0.0
    min_ratio = max(0.01, min(FULL_MODE_MIN_NARRATED_BLOCK_VOICEOVER_RATIO, 1.0))
    max_by_voice_ratio = voice_seconds / min_ratio
    max_by_tail = voice_seconds + max(0.0, FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS)
    return max(0.0, min(max_by_voice_ratio, max_by_tail))


def _max_narrated_visual_seconds_for_chars(narration_chars: int, language: str) -> float:
    if narration_chars <= 0:
        return 0.0
    min_scene_chars = max(1, _minimum_scene_matched_narration_chars(language))
    density_seconds = (float(narration_chars) / min_scene_chars) * 24.0
    estimated_voice_seconds = _estimated_voiceover_seconds_for_chars(narration_chars, language)
    min_ratio = max(0.01, min(FULL_MODE_MIN_NARRATED_BLOCK_VOICEOVER_RATIO, 1.0))
    max_by_voice_ratio = estimated_voice_seconds / min_ratio
    max_by_tail = estimated_voice_seconds + max(0.0, FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS)
    # Very short lines still need a sync cap; otherwise a token phrase can hold
    # a long visual range. Grounded scene-length narration is handled by the
    # density floor here and by render-time TTS trimming/splitting later.
    return max(0.0, min(density_seconds, max_by_voice_ratio, max_by_tail))


def _expected_narration_chars_for_visual_duration(block_duration: float, language: str) -> int:
    visual_seconds = float(block_duration or 0.0)
    density_chars = max(
        _minimum_scene_matched_narration_chars(language),
        int(math.ceil((visual_seconds / 24.0) * _minimum_scene_matched_narration_chars(language))),
    )
    if visual_seconds > 30.0:
        return density_chars
    required_voice_seconds = _minimum_voiceover_seconds_for_visual_duration(visual_seconds)
    if (language or "").lower().startswith("zh"):
        sync_chars = int(math.ceil(required_voice_seconds * 4.2))
    else:
        sync_chars = int(math.ceil(required_voice_seconds * 12.0))
    return max(density_chars, sync_chars)


def _minimum_sync_narration_chars_for_visual_duration(block_duration: float, language: str) -> int:
    required_voice_seconds = _minimum_voiceover_seconds_for_visual_duration(block_duration)
    if (language or "").lower().startswith("zh"):
        return max(1, int(math.ceil(required_voice_seconds * 4.2)))
    return max(1, int(math.ceil(required_voice_seconds * 12.0)))


def _density_floor_chars_for_visual_duration(block_duration: float, language: str) -> int:
    visual_seconds = float(block_duration or 0.0)
    return max(
        _minimum_scene_matched_narration_chars(language),
        int(math.ceil((visual_seconds / 24.0) * _minimum_scene_matched_narration_chars(language))),
    )


def _protect_full_mode_visual_budget_after_speed(blocks: List[Dict], duration: float, target_duration: str) -> List[Dict]:
    protected = [dict(block) for block in blocks or []]
    if target_duration != "full" or not protected:
        return protected
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
    min_visual_seconds = _full_mode_min_playable_visual_seconds(duration, target_seconds)
    if sum(_block_visual_duration(block) for block in protected) >= min_visual_seconds - FULL_MODE_VALIDATION_EPSILON_SECONDS:
        return protected
    for block in sorted(protected, key=lambda item: _safe_video_speed(item.get("video_speed")), reverse=True):
        if sum(_block_visual_duration(item) for item in protected) >= min_visual_seconds - FULL_MODE_VALIDATION_EPSILON_SECONDS:
            break
        if _safe_video_speed(block.get("video_speed")) > 1.0:
            block["video_speed"] = 1.0
            block["speed_reason"] = ""
    return protected


def _normalize_narration_blocks(raw_blocks: List[Dict], duration: float) -> List[Dict]:
    normalized = []
    for item in raw_blocks or []:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        is_pause = _coerce_bool(item.get("pause"))
        narration = str(item.get("narration") or item.get("text") or "").strip()
        if is_pause:
            narration = ""
        is_locked_edit_plan = bool(item.get("_locked_edit_plan"))
        if end - start < 1.0 or (not narration and not is_pause and not is_locked_edit_plan):
            continue
        visual = str(item.get("visual") or item.get("reason") or item.get("title") or "").strip()
        normalized_block = {
            "start": round(start, 3),
            "end": round(end, 3),
            "visual": visual,
            "narration": narration,
            "pause": is_pause,
            "rate": _safe_edge_rate(item.get("rate") or "+0%"),
            "pitch": _safe_edge_pitch(item.get("pitch") or "+0Hz"),
            "video_speed": _safe_video_speed(item.get("video_speed", item.get("playback_speed", item.get("speed", item.get("speed_up"))))),
        }
        speed_reason = str(item.get("speed_reason") or item.get("speed_rationale") or "").strip()
        if speed_reason:
            normalized_block["speed_reason"] = speed_reason
        visual_facts = item.get("visual_facts")
        if isinstance(visual_facts, list):
            normalized_block["visual_facts"] = [str(fact).strip() for fact in visual_facts if str(fact).strip()]
        evidence_timestamps = item.get("evidence_timestamps")
        if isinstance(evidence_timestamps, list):
            normalized_block["evidence_timestamps"] = [
                timestamp
                for timestamp in evidence_timestamps
                if isinstance(timestamp, (int, float)) and start <= float(timestamp) <= end
            ]
        for key in (
            "_locked_edit_plan",
            "_min_narration_chars",
            "auto_filled_visual_budget",
            "rendered_duration",
        ):
            if key in item:
                normalized_block[key] = item[key]
        normalized.append(normalized_block)
    normalized.sort(key=lambda block: block["start"])
    merged = []
    for block in normalized:
        if (
            merged
            and bool(merged[-1].get("pause"))
            and bool(block.get("pause"))
            and float(block.get("start") or 0.0) <= float(merged[-1].get("end") or 0.0) + 0.3
        ):
            previous = merged[-1]
            previous["end"] = round(max(float(previous["end"]), float(block["end"])), 3)
            visual_parts = [
                str(previous.get("visual") or "").strip(),
                str(block.get("visual") or "").strip(),
            ]
            previous["visual"] = "; ".join(part for part in visual_parts if part)
            previous["video_speed"] = min(
                _safe_video_speed(previous.get("video_speed")),
                _safe_video_speed(block.get("video_speed")),
            )
            previous["rate"] = "+0%"
            previous["pitch"] = "+0Hz"
            continue
        merged.append(block)
    return merged


def _strip_internal_narration_block_fields(data: Dict) -> None:
    for block in data.get("narration_blocks") or []:
        if not isinstance(block, dict):
            continue
        for key in (
            "_locked_edit_plan",
            "_min_narration_chars",
            "auto_filled_visual_budget",
        ):
            block.pop(key, None)


def _strip_auto_filled_user_visible_fields(blocks: List[Dict]) -> List[Dict]:
    cleaned = []
    for block in blocks or []:
        item = dict(block)
        if _coerce_bool(item.get("auto_filled_visual_budget")):
            item["visual"] = COMMENTARY_AUTO_FILLED_BRIDGE_VISUAL
            item["visual_facts"] = []
            item["narration"] = ""
            item["pause"] = True
            item["rate"] = "+0%"
            item["pitch"] = "+0Hz"
            item.pop("auto_filled_visual_budget", None)
        cleaned.append(item)
    return cleaned


def _apply_auto_video_speed_to_blocks(
    blocks: List[Dict],
    enabled: bool,
    visual_analysis: Optional[Dict] = None,
) -> List[Dict]:
    adjusted = [dict(block) for block in blocks or []]
    if not adjusted:
        return adjusted
    if not enabled:
        for block in adjusted:
            block["video_speed"] = 1.0
        return adjusted
    for block in adjusted:
        block["video_speed"] = _safe_video_speed(block.get("video_speed"))
    return adjusted


def _summarize_auto_video_speed(blocks: List[Dict], enabled: bool) -> Dict:
    accelerated = []
    saved_seconds = 0.0
    for index, block in enumerate(blocks or [], start=1):
        speed = _safe_video_speed(block.get("video_speed"))
        if speed <= 1.0001:
            continue
        source_duration = max(0.0, float(block.get("end") or 0.0) - float(block.get("start") or 0.0))
        saved = max(0.0, source_duration - (source_duration / speed))
        saved_seconds += saved
        accelerated.append({
            "index": index,
            "start": block.get("start"),
            "end": block.get("end"),
            "video_speed": speed,
            "saved_seconds": round(saved, 1),
            "visual": ""
            if _coerce_bool(block.get("auto_filled_visual_budget")) or str(block.get("visual") or "") == COMMENTARY_AUTO_FILLED_BRIDGE_VISUAL
            else str(block.get("visual") or "")[:120],
            "speed_reason": str(block.get("speed_reason") or "")[:160],
        })
    return {
        "enabled": bool(enabled),
        "total_blocks": len(blocks or []),
        "accelerated_count": len(accelerated),
        "saved_seconds": round(saved_seconds, 1),
        "accelerated_blocks": accelerated,
    }


def _narration_blocks_to_edit_segments(blocks: List[Dict]) -> List[Dict]:
    return [
        {
            "start": block["start"],
            "end": block["end"],
            "reason": block.get("visual") or "scene-matched narration block",
            "video_speed": _safe_video_speed(block.get("video_speed")),
            "speed_reason": str(block.get("speed_reason") or "").strip(),
        }
        for block in blocks or []
    ]


def _commit_narration_blocks_to_script(data: Dict, blocks: List[Dict]) -> None:
    data["narration_blocks"] = blocks
    data["edit_segments"] = _narration_blocks_to_edit_segments(blocks)
    data["narration"] = _narration_from_blocks({"narration_blocks": blocks}) or str(data.get("narration") or "")


def _block_evidence_timestamps(block: Dict) -> List[float]:
    timestamps = []
    for value in block.get("evidence_timestamps") or []:
        try:
            timestamps.append(round(float(value), 3))
        except (TypeError, ValueError):
            continue
    return timestamps


def _block_keeps_evidence_timestamps(block: Dict) -> bool:
    evidence = _block_evidence_timestamps(block)
    if not evidence:
        return True
    try:
        start = float(block.get("start"))
        end = float(block.get("end"))
    except (TypeError, ValueError):
        return False
    return all(start - 0.35 <= timestamp <= end + 0.35 for timestamp in evidence)


def _filter_block_evidence_timestamps_for_range(block: Dict, start: float, end: float) -> List[float]:
    low = float(start)
    high = float(end)
    return [
        timestamp
        for timestamp in _block_evidence_timestamps(block)
        if low <= timestamp <= high
    ]


def _visual_analysis_evidence_timestamps_for_range(
    visual_analysis: Optional[Dict],
    start: float,
    end: float,
    limit: int = 4,
) -> List[float]:
    if not visual_analysis or visual_analysis.get("provider") != "openai_compatible":
        return []
    low = float(start)
    high = float(end)
    frame_timestamps = [
        timestamp
        for timestamp in _visual_analysis_frame_timestamps(visual_analysis)
        if low - 0.35 <= timestamp <= high + 0.35
    ]
    if not frame_timestamps:
        return []
    if len(frame_timestamps) <= limit:
        return frame_timestamps
    selected = [frame_timestamps[0], frame_timestamps[-1]]
    if limit > 2:
        middle = frame_timestamps[1:-1]
        if middle:
            step = len(middle) / max(1, limit - 2)
            for offset in range(limit - 2):
                selected.append(middle[min(len(middle) - 1, int(offset * step))])
    return sorted(set(round(float(timestamp), 3) for timestamp in selected))


def _fill_missing_openai_evidence_timestamps(data: Dict, visual_analysis: Optional[Dict]) -> None:
    if not isinstance(data, dict) or not visual_analysis or visual_analysis.get("provider") != "openai_compatible":
        return
    blocks = data.get("narration_blocks") if isinstance(data.get("narration_blocks"), list) else []
    changed = False
    for block in blocks:
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        if _block_evidence_timestamps(block):
            continue
        try:
            start = float(block.get("start"))
            end = float(block.get("end"))
        except (TypeError, ValueError):
            continue
        evidence = _visual_analysis_evidence_timestamps_for_range(visual_analysis, start, end)
        if evidence:
            block["evidence_timestamps"] = evidence
            changed = True
    if changed:
        _commit_narration_blocks_to_script(data, blocks)


def _assert_rendered_blocks_keep_evidence_timestamps(blocks: List[Dict]) -> None:
    for index, block in enumerate(blocks or [], start=1):
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        evidence = _block_evidence_timestamps(block)
        if not evidence:
            continue
        if _block_keeps_evidence_timestamps(block):
            continue
        raise Exception(
            "Rendered commentary block lost its timestamp evidence after TTS sync. "
            f"Block {index} now plays {float(block.get('start') or 0.0):.3f}-{float(block.get('end') or 0.0):.3f}s, "
            f"but its evidence_timestamps are {evidence}. "
            "OpenShorts stopped instead of producing narration that no longer matches the visible frames; regenerate with a shorter source range, fuller narration, or explicit pause=true tail."
        )


def _commentary_transcript_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, "commentary_transcript.json")


def _load_cached_commentary_transcript(output_dir: str) -> Optional[Dict]:
    cache_path = _commentary_transcript_cache_path(output_dir)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    transcript = payload.get("transcript") if isinstance(payload, dict) else None
    if not isinstance(transcript, dict):
        return None
    if not isinstance(transcript.get("segments"), list):
        return None
    return transcript


def _save_commentary_transcript_cache(output_dir: str, transcript: Dict) -> str:
    cache_path = _commentary_transcript_cache_path(output_dir)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"transcript": transcript}, f, ensure_ascii=False, indent=2)
    return cache_path


def _load_or_transcribe_commentary_transcript(
    output_dir: str,
    video_path: str,
    source_language: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = None,
    checkpoint: Optional[Callable[[Dict], None]] = None,
    cached_message: str = "Reusing cached Faster-Whisper transcript from saved task checkpoint...",
    transcribe_message: str = "Transcribing full video with Faster-Whisper...",
) -> Dict:
    transcript = _load_cached_commentary_transcript(output_dir)
    if transcript:
        if progress:
            progress(cached_message)
        return transcript
    if progress:
        progress(transcribe_message)
    transcript = transcribe_video(video_path, language=source_language)
    transcript_path = _save_commentary_transcript_cache(output_dir, transcript)
    if checkpoint:
        checkpoint({"transcript_path": transcript_path})
    return transcript


def _auto_filled_pause_block(start: float, end: float) -> Dict:
    midpoint = round((float(start) + float(end)) / 2.0, 3)
    return {
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "visual": COMMENTARY_AUTO_FILLED_BRIDGE_VISUAL,
        "visual_facts": [],
        "evidence_timestamps": [midpoint],
        "narration": "",
        "pause": True,
        "rate": "+0%",
        "pitch": "+0Hz",
        "video_speed": 1.0,
        "auto_filled_visual_budget": True,
    }


def _insert_auto_pause_block(
    blocks: List[Dict],
    insert_index: int,
    start: float,
    end: float,
) -> None:
    if end - start < 1.0:
        return
    blocks.insert(insert_index, _auto_filled_pause_block(start, end))


def _repair_full_mode_underselected_visual_budget_with_pause_blocks(
    blocks: List[Dict],
    duration: float,
    target_seconds: float,
    language: str = "",
) -> List[Dict]:
    if not blocks or target_seconds <= 0:
        return blocks
    min_visual_seconds = (
        target_seconds
        if _full_mode_preserves_source_process(duration, target_seconds)
        else _full_mode_min_playable_visual_seconds(duration, target_seconds)
    )
    visual_seconds = sum(_block_visual_duration(block) for block in blocks)
    deficit = min_visual_seconds - visual_seconds
    if deficit <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
        return blocks

    repaired = [dict(block) for block in blocks]
    for index in range(len(repaired) - 1, -1, -1):
        block = repaired[index]
        if bool(block.get("pause")):
            continue
        speed = _safe_video_speed(block.get("video_speed"))
        start = float(block.get("start") or 0.0)
        end = float(block.get("end") or start)
        if end <= start:
            continue
        next_start = duration
        for later in repaired[index + 1:]:
            later_start = float(later.get("start") or 0.0)
            if later_start > end + 0.3:
                next_start = min(next_start, later_start)
                break
        max_by_gap = max(0.0, next_start - end) / speed
        if max_by_gap <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
            continue
        narration_chars = len(re.sub(r"\s+", "", str(block.get("narration") or block.get("text") or "")))
        max_playable = _max_narrated_visual_seconds_for_chars(narration_chars, language)
        current_playable = _block_visual_duration(block)
        max_by_density = max(0.0, max_playable - current_playable)
        add_seconds = min(deficit, max_by_gap, max_by_density)
        if add_seconds <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
            continue
        block["end"] = round(end + (add_seconds * speed), 3)
        visual_seconds += add_seconds
        deficit -= add_seconds
        if deficit <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
            return repaired

    max_pause_seconds = (
        max(1.0, target_seconds)
        if _full_mode_preserves_source_process(duration, target_seconds)
        else max(1.0, FULL_MODE_MAX_PAUSE_SECONDS)
    )
    candidate_gaps = []
    for index in range(len(repaired) - 1):
        current = repaired[index]
        following = repaired[index + 1]
        if bool(current.get("pause")):
            continue
        gap_start = float(current.get("end") or 0.0)
        gap_end = float(following.get("start") or 0.0)
        gap_seconds = gap_end - gap_start
        if gap_seconds >= 1.0:
            candidate_gaps.append({
                "index": index + 1,
                "start": gap_start,
                "end": gap_end,
                "seconds": gap_seconds,
            })
    if repaired:
        last = repaired[-1]
        if not bool(last.get("pause")):
            gap_start = float(last.get("end") or 0.0)
            gap_end = duration
            gap_seconds = gap_end - gap_start
            if gap_seconds >= 1.0:
                candidate_gaps.append({
                    "index": len(repaired),
                    "start": gap_start,
                    "end": gap_end,
                    "seconds": gap_seconds,
                })

    for gap in candidate_gaps:
        if deficit <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
            break
        add_seconds = min(max_pause_seconds, float(gap["seconds"]), deficit)
        if add_seconds < 1.0:
            continue
        _insert_auto_pause_block(
            repaired,
            int(gap["index"]),
            float(gap["start"]),
            float(gap["start"]) + add_seconds,
        )
        for later_gap in candidate_gaps:
            if int(later_gap["index"]) > int(gap["index"]):
                later_gap["index"] = int(later_gap["index"]) + 1
        deficit -= add_seconds

    return repaired


def _repair_full_mode_pause_blocks(blocks: List[Dict], min_visual_seconds: float = 0.0, max_pause_seconds: Optional[float] = None) -> List[Dict]:
    if not blocks:
        return blocks
    max_pause_seconds = max(1.0, float(max_pause_seconds if max_pause_seconds is not None else FULL_MODE_MAX_PAUSE_SECONDS))
    repaired = [dict(block) for block in blocks]
    for block in repaired:
        if not bool(block.get("pause")):
            continue
        playable_seconds = _block_visual_duration(block)
        if playable_seconds <= max_pause_seconds + FULL_MODE_VALIDATION_EPSILON_SECONDS:
            continue
        speed = _safe_video_speed(block.get("video_speed"))
        start = float(block.get("start") or 0.0)
        block["end"] = round(start + (max_pause_seconds * speed), 3)

    visual_seconds = sum(_block_visual_duration(block) for block in repaired)
    if min_visual_seconds <= 0 or visual_seconds >= min_visual_seconds - FULL_MODE_VALIDATION_EPSILON_SECONDS:
        return repaired

    for index, block in enumerate(repaired):
        if visual_seconds >= min_visual_seconds - FULL_MODE_VALIDATION_EPSILON_SECONDS:
            break
        if not bool(block.get("pause")):
            continue
        speed = _safe_video_speed(block.get("video_speed"))
        start = float(block.get("start") or 0.0)
        original = blocks[index] if index < len(blocks) else {}
        original_end = float(original.get("end") or block.get("end") or start)
        current_playable = _block_visual_duration(block)
        room = min(
            max_pause_seconds - current_playable,
            max(0.0, (original_end - start) / speed - current_playable),
            max(0.0, min_visual_seconds - visual_seconds),
        )
        if room <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
            continue
        block["end"] = round(float(block.get("end") or start) + (room * speed), 3)
        visual_seconds += room
    return repaired


def _repair_full_mode_overselected_visual_budget(
    blocks: List[Dict],
    duration: float,
    target_seconds: float,
) -> List[Dict]:
    if not blocks or target_seconds <= 0:
        return blocks
    max_visual_seconds = _full_mode_max_playable_visual_seconds(duration, target_seconds)
    visual_seconds = sum(_block_visual_duration(block) for block in blocks)
    excess = visual_seconds - max_visual_seconds
    if excess <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
        return blocks
    if excess > max(30.0, max_visual_seconds * 0.08):
        return blocks

    repaired = [dict(block) for block in blocks]
    required_latest_end = (
        duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION
        if not _full_mode_preserves_source_process(duration, target_seconds)
        else 0.0
    )
    for index in range(len(repaired) - 1, -1, -1):
        if excess <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
            break
        block = repaired[index]
        speed = _safe_video_speed(block.get("video_speed"))
        start = float(block.get("start") or 0.0)
        end = float(block.get("end") or start)
        source_seconds = max(0.0, end - start)
        if source_seconds <= 1.0:
            continue
        max_trim_playable = min(excess, source_seconds / speed - 1.0)
        if max_trim_playable <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
            continue
        if end >= required_latest_end:
            max_trim_playable = min(max_trim_playable, max(0.0, (end - required_latest_end) / speed))
        if max_trim_playable <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
            continue
        new_end = end - (max_trim_playable * speed)
        if new_end <= start:
            continue
        block["end"] = round(new_end, 3)
        excess -= max_trim_playable

    if excess > FULL_MODE_VALIDATION_EPSILON_SECONDS:
        return blocks
    return repaired


def _finalize_full_mode_narration_blocks_for_render(data: Dict, duration: float, target_duration: str, language: str) -> None:
    if target_duration != "full" or not data.get("narration_blocks"):
        return
    blocks = _normalize_narration_blocks(data.get("narration_blocks") or [], duration)
    blocks = _ensure_complete_commentary_ending_blocks(blocks, language)
    blocks = _normalize_full_mode_render_narration_blocks(blocks, language)
    _commit_narration_blocks_to_script(data, blocks)


def _sync_openai_locked_script_for_validation(
    data: Dict,
    duration: float,
    target_duration: str,
    language: str,
    visual_analysis: Optional[Dict],
) -> None:
    if target_duration != "full" or not isinstance(data, dict):
        return
    if not visual_analysis or visual_analysis.get("provider") != "openai_compatible":
        return
    blocks = data.get("narration_blocks") if isinstance(data.get("narration_blocks"), list) else []
    if not any(isinstance(block, dict) and bool(block.get("_locked_edit_plan")) for block in blocks):
        return
    _normalize_script_timeline(data, duration, target_duration, language)
    target_seconds = _target_visual_duration_seconds_for_analysis(duration, target_duration, visual_analysis)
    _fit_locked_plan_narration_to_budget(
        data,
        _maximum_narration_chars_for_target_seconds(target_seconds, target_duration, language),
        language,
    )
    data["narration"] = _narration_from_blocks(data) or str(data.get("narration") or "")


def _resolve_edit_segments_for_target(raw_segments: List[Dict], duration: float, target_duration: str) -> List[Dict]:
    return _normalize_edit_segments(raw_segments or [], duration)


def _require_ai_selected_edit_segments(script: Dict, duration: float, target_duration: str) -> List[Dict]:
    segments = _resolve_edit_segments_for_target(script.get("edit_segments", []), duration, target_duration)
    if segments:
        return segments
    raise Exception(
        "AI did not return any valid kept visual ranges for the commentary edit. "
        "The model must decide which source ranges to keep, cut, and splice from the visible content; "
        "OpenShorts will not invent fallback edit_segments, evenly sample the timeline, or render the whole source by default."
    )


def _edit_segments_playable_timeline(edit_segments: List[Dict]) -> Tuple[List[Dict], float]:
    timeline = []
    cursor = 0.0
    for index, segment in enumerate(edit_segments or [], start=1):
        try:
            source_start = float(segment.get("start"))
            source_end = float(segment.get("end"))
        except (TypeError, ValueError):
            continue
        source_seconds = max(0.0, source_end - source_start)
        if source_seconds <= 0:
            continue
        speed = _safe_video_speed(segment.get("video_speed") or segment.get("speed"))
        output_seconds = source_seconds / speed
        output_start = cursor
        output_end = output_start + output_seconds
        timeline.append({
            "index": index,
            "output_start": round(output_start, 3),
            "output_end": round(output_end, 3),
            "source_start": round(source_start, 3),
            "source_end": round(source_end, 3),
            "source_seconds": round(source_seconds, 3),
            "video_speed": speed,
            "reason": str(segment.get("reason") or "").strip(),
            "speed_reason": str(segment.get("speed_reason") or "").strip(),
        })
        cursor = output_end
    return timeline, cursor


def _remap_transcript_to_edited_timeline(transcript: Optional[Dict], edit_timeline: List[Dict]) -> Dict:
    if not isinstance(transcript, dict):
        return {"text": "", "segments": [], "language": "unknown"}
    remapped_segments = []
    for source_segment in _transcript_spoken_segments(transcript):
        source_start = float(source_segment["start"])
        source_end = float(source_segment["end"])
        for mapping in edit_timeline or []:
            clip_start = float(mapping.get("source_start") or 0.0)
            clip_end = float(mapping.get("source_end") or clip_start)
            overlap_start = max(source_start, clip_start)
            overlap_end = min(source_end, clip_end)
            if overlap_end <= overlap_start:
                continue
            speed = _safe_video_speed(mapping.get("video_speed"))
            output_start = float(mapping.get("output_start") or 0.0) + ((overlap_start - clip_start) / speed)
            output_end = float(mapping.get("output_start") or 0.0) + ((overlap_end - clip_start) / speed)
            if output_end <= output_start:
                continue
            remapped_segments.append({
                "start": round(output_start, 3),
                "end": round(output_end, 3),
                "text": source_segment["text"],
                "source_start": round(overlap_start, 3),
                "source_end": round(overlap_end, 3),
            })
    remapped_segments.sort(key=lambda item: (item["start"], item["end"]))
    return {
        "text": " ".join(segment["text"] for segment in remapped_segments),
        "segments": remapped_segments,
        "language": transcript.get("language", "unknown"),
        "source_timeline": "remapped_from_original_edit",
    }


def _script_from_openai_candidate_edit_plan(
    edit_plan: Optional[Dict],
    video_title: str,
    duration: float,
    target_duration: str,
) -> Dict:
    if not edit_plan or not edit_plan.get("blocks"):
        raise Exception(
            "OpenAI-compatible edit-first flow could not build a locked edit plan from full-video analysis. "
            "The model must provide enough candidate_segments with edit value scores before the final commentary pass."
        )
    blocks = []
    for index, plan_block in enumerate(edit_plan.get("blocks") or [], start=1):
        if not isinstance(plan_block, dict):
            continue
        start = float(plan_block.get("start") or 0.0)
        end = float(plan_block.get("end") or 0.0)
        if end <= start:
            continue
        blocks.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "visual": str(plan_block.get("visual") or "AI-selected useful visual range"),
            "visual_facts": plan_block.get("visual_facts") if isinstance(plan_block.get("visual_facts"), list) else [],
            "evidence_timestamps": plan_block.get("evidence_timestamps") if isinstance(plan_block.get("evidence_timestamps"), list) else [],
            "narration": "",
            "pause": False,
            "rate": "+0%",
            "pitch": "+0Hz",
            "video_speed": _safe_video_speed(plan_block.get("video_speed")),
            "speed_reason": str(plan_block.get("speed_reason") or "").strip(),
            "_locked_edit_plan": True,
            "_source_plan_index": index,
        })
    if not blocks:
        raise Exception("OpenAI-compatible edit-first flow produced no valid visual blocks.")
    script = {
        "title": video_title or "Commentary Remix",
        "summary": "",
        "hook": "",
        "narration": "",
        "narration_blocks": blocks,
        "episode_plan": {"should_split": False, "reason": "not needed"},
        "episodes": [],
        "edit_segments": _narration_blocks_to_edit_segments(blocks),
        "cut_strategy": [],
        "chapters": [],
        "hashtags": [],
    }
    _normalize_script_timeline(script, duration, target_duration, "")
    return script


def _script_from_full_source_edit_plan(
    video_title: str,
    duration: float,
    target_duration: str,
    reason: str = "source duration already fits target; preserve the full edited visual timeline",
) -> Dict:
    safe_duration = max(0.1, float(duration or 0.0))
    block = {
        "start": 0.0,
        "end": round(safe_duration, 3),
        "visual": reason,
        "visual_facts": [reason],
        "evidence_timestamps": [round(safe_duration / 2.0, 3)],
        "narration": "",
        "pause": False,
        "rate": "+0%",
        "pitch": "+0Hz",
        "video_speed": 1.0,
        "speed_reason": "",
        "_locked_edit_plan": True,
        "_source_plan_index": 1,
    }
    script = {
        "title": video_title or "Commentary Remix",
        "summary": "",
        "hook": "",
        "narration": "",
        "narration_blocks": [block],
        "episode_plan": {"should_split": False, "reason": "not needed"},
        "episodes": [],
        "edit_segments": _narration_blocks_to_edit_segments([block]),
        "cut_strategy": [],
        "chapters": [],
        "hashtags": [],
    }
    _normalize_script_timeline(script, safe_duration, target_duration, "")
    return script


def _blocks_from_edited_timeline_script(
    edited_script: Dict,
    edit_timeline: List[Dict],
    language: str,
) -> List[Dict]:
    source_blocks = _script_narration_blocks(edited_script)
    timeline = list(edit_timeline or [])
    result = []
    for index, block in enumerate(source_blocks, start=1):
        try:
            start = float(block.get("start"))
            end = float(block.get("end"))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        overlapping = [
            mapping for mapping in timeline
            if float(mapping.get("output_end") or 0.0) > start and float(mapping.get("output_start") or 0.0) < end
        ]
        source_ranges = [
            {
                "source_start": mapping.get("source_start"),
                "source_end": mapping.get("source_end"),
                "output_start": mapping.get("output_start"),
                "output_end": mapping.get("output_end"),
                "video_speed": mapping.get("video_speed"),
            }
            for mapping in overlapping
        ]
        visual_facts = block.get("visual_facts") if isinstance(block.get("visual_facts"), list) else []
        evidence_timestamps = block.get("evidence_timestamps") if isinstance(block.get("evidence_timestamps"), list) else []
        narration = str(block.get("narration") or "").strip()
        if _contains_visual_analysis_label_artifact(narration, language):
            narration = _strip_visual_analysis_label_artifact(narration, language)
        result.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "visual": str(block.get("visual") or "").strip() or "edited-video visual range",
            "visual_facts": visual_facts,
            "evidence_timestamps": evidence_timestamps,
            "narration": narration,
            "pause": bool(block.get("pause")),
            "rate": block.get("rate") or "+0%",
            "pitch": block.get("pitch") or "+0Hz",
            "video_speed": 1.0,
            "speed_reason": "",
            "source_ranges": source_ranges,
        })
    return _normalize_narration_blocks(result, max((float(item.get("output_end") or 0.0) for item in timeline), default=0.0))


def _openai_commentary_script_response_format(strict: bool = True) -> Dict:
    if not strict or not OPENAI_STRICT_SCRIPT_SCHEMA:
        return {"type": "json_object"}
    block_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "start": {"type": "number"},
            "end": {"type": "number"},
            "visual": {"type": "string"},
            "visual_facts": {"type": "array", "items": {"type": "string"}},
            "evidence_timestamps": {"type": "array", "items": {"type": "number"}},
            "narration": {"type": "string"},
            "pause": {"type": "boolean"},
            "rate": {"type": "string"},
            "pitch": {"type": "string"},
            "video_speed": {"type": "number"},
            "speed_reason": {"type": "string"},
        },
        "required": [
            "start",
            "end",
            "visual",
            "visual_facts",
            "evidence_timestamps",
            "narration",
            "pause",
            "rate",
            "pitch",
            "video_speed",
            "speed_reason",
        ],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "openshorts_commentary_script",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "hook": {"type": "string"},
                    "narration": {"type": "string"},
                    "narration_blocks": {
                        "type": "array",
                        "items": block_schema,
                    },
                    "episode_plan": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "should_split": {"type": "boolean"},
                            "reason": {"type": "string"},
                        },
                        "required": ["should_split", "reason"],
                    },
                    "episodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "episode_number": {"type": "integer"},
                                "title": {"type": "string"},
                                "summary": {"type": "string"},
                                "start_block": {"type": "integer"},
                                "end_block": {"type": "integer"},
                            },
                            "required": ["episode_number", "title", "summary", "start_block", "end_block"],
                        },
                    },
                    "edit_segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": ["start", "end", "reason"],
                        },
                    },
                    "cut_strategy": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "removed_range": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["removed_range", "reason"],
                        },
                    },
                    "chapters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "title": {"type": "string"},
                                "narration": {"type": "string"},
                            },
                            "required": ["start", "end", "title", "narration"],
                        },
                    },
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "title",
                    "summary",
                    "hook",
                    "narration",
                    "narration_blocks",
                    "episode_plan",
                    "episodes",
                    "edit_segments",
                    "cut_strategy",
                    "chapters",
                    "hashtags",
                ],
            },
        },
    }


def _commentary_json_output_contract(block_count: int, target_duration: str) -> str:
    block_rule = (
        f"- For TARGET DURATION full, narration_blocks must contain exactly {block_count} items."
        if target_duration == "full"
        else "- For compact non-full targets, narration_blocks is still preferred when visual timing is available; otherwise edit_segments must be valid."
    )
    return f"""
JSON OUTPUT CONTRACT:
- Return exactly one raw JSON object and nothing else.
- The first non-whitespace character must be {{ and the last non-whitespace character must be }}.
- Do not output markdown fences, XML tags, <think> blocks, reasoning text, comments, apologies, or explanations.
- Use double quotes for every key and string. Never use single quotes.
- Escape literal newlines inside strings as \\n, or replace them with spaces. Do not put unescaped line breaks inside a JSON string.
- Do not use trailing commas, comments, NaN, Infinity, undefined, or Python-style booleans.
- Required top-level keys: title, summary, hook, narration, narration_blocks, episode_plan, episodes, edit_segments, cut_strategy, chapters, hashtags.
- episode_plan must be an object with should_split and reason. If there are no episodes, use {{"should_split": false, "reason": "not needed"}} and episodes=[].
- narration_blocks, episodes, edit_segments, cut_strategy, chapters, and hashtags must always be arrays. Use [] if empty.
- Every narration_blocks item must include start, end, visual, visual_facts, evidence_timestamps, narration, pause, rate, pitch, video_speed, and speed_reason.
- Every edit_segments item must include start, end, and reason.
{block_rule}
- If you are uncertain about an optional value, use an empty string, false, 1.0, or [] as appropriate, but keep the JSON valid.
""".strip()


def _commentary_json_schema_summary() -> str:
    return """
Required JSON shape:
{
  "title": "string",
  "summary": "string",
  "hook": "string",
  "narration": "string",
  "narration_blocks": [
    {
      "start": 0,
      "end": 10,
      "visual": "string",
      "visual_facts": ["string"],
      "evidence_timestamps": [0],
      "narration": "string",
      "pause": false,
      "rate": "+0%",
      "pitch": "+0Hz",
      "video_speed": 1.0,
      "speed_reason": ""
    }
  ],
  "episode_plan": {"should_split": false, "reason": "string"},
  "episodes": [],
  "edit_segments": [{"start": 0, "end": 10, "reason": "string"}],
  "cut_strategy": [],
  "chapters": [{"start": 0, "end": 10, "title": "string", "narration": "string"}],
  "hashtags": ["#tag"]
}
""".strip()


def _block_narration_sync_instruction(language: str) -> str:
    min_scene_chars = _minimum_scene_matched_narration_chars(language)
    return (
        "There is no filler word-count target, and you must not pad narration with meaningless words. Backend rendering preserves each selected source range and the requested video_speed; it will not rescue a sparse narration block by cutting or speeding the visuals after the spoken line. "
        f"Your job is to choose useful timestamp ranges and write concrete, scene-matched commentary that clearly explains the visible action. For ordinary narrated process/action blocks, write enough to make the on-screen content understandable, usually at least {min_scene_chars} non-whitespace characters, but never add filler or repeat obvious words. If a range truly has little to say, shorten the range, split it, or mark a brief pause instead of leaving a long under-explained narrated block."
    )


def _maximum_narration_chars(duration: float, target_duration: str, language: str) -> int:
    if target_duration != "full":
        return 0
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
    return _maximum_narration_chars_for_target_seconds(target_seconds, target_duration, language)


def _maximum_narration_chars_for_target_seconds(target_seconds: float, target_duration: str, language: str) -> int:
    if target_duration != "full":
        return 0
    if (language or "").lower().startswith("zh"):
        readable_limit = int(max(1200, target_seconds * 4.2))
        return int(min(FULL_MODE_MAX_NARRATION_CHARS_ZH, readable_limit))
    readable_limit = int(max(900, target_seconds * 2.8))
    return int(min(FULL_MODE_MAX_NARRATION_CHARS_OTHER, readable_limit))


def _minimum_scene_matched_narration_chars(language: str) -> int:
    if (language or "").lower().startswith("zh"):
        return FULL_MODE_MIN_SCENE_MATCHED_NARRATION_CHARS_ZH
    return FULL_MODE_MIN_SCENE_MATCHED_NARRATION_CHARS_OTHER


def _target_narration_block_count(duration: float, target_duration: str) -> int:
    if target_duration != "full":
        return 0
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
    return _target_narration_block_count_for_target_seconds(target_seconds)


def _target_narration_block_count_for_target_seconds(target_seconds: float) -> int:
    if target_seconds <= 0:
        return 16
    return int(min(48, max(16, math.ceil(target_seconds / 42.0))))


def _narration_from_blocks(data: Dict) -> str:
    parts = []
    for block in data.get("narration_blocks") or []:
        if isinstance(block, dict):
            text = str(block.get("narration") or block.get("text") or "").strip()
        else:
            text = str(block or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _custom_style_requires_operation_logic(custom_style_prompt: Optional[str], language: str) -> bool:
    prompt = re.sub(r"\s+", " ", str(custom_style_prompt or "")).strip()
    if not prompt:
        return False
    lower_prompt = prompt.lower()
    if (language or "").lower().startswith("zh"):
        strong_markers = (
            "解释操作逻辑",
            "操作逻辑",
            "流程逻辑",
            "工序逻辑",
            "这一步目的",
            "目的层",
            "逻辑层",
            "动作层",
            "结果层",
        )
        if any(marker in prompt for marker in strong_markers):
            return True
        process_markers = ("工业", "机械", "设备", "工序", "流程", "操作", "加工", "回收", "生产")
        explanation_markers = ("逻辑", "目的", "解释", "关键", "原因", "原理", "作用")
        return any(marker in prompt for marker in process_markers) and any(
            marker in prompt for marker in explanation_markers
        )
    return bool(re.search(
        r"operation logic|process logic|why|purpose|explain(?:s|ing)? the process|mechanical explanation|industrial explanation",
        lower_prompt,
    ))


def _narration_has_operation_logic(text: str, language: str) -> bool:
    narration = re.sub(r"\s+", "", str(text or ""))
    if not narration:
        return False
    if (language or "").lower().startswith("zh"):
        return bool(re.search(
            r"为了|用来|便于|方便|防止|避免|这样(?:能|可以|才|就)|这么做|"
            r"这一步(?:是|要|先|主要|的目的)|主要是|目的|关键是|好让|"
            r"让.{0,16}(?:能|可以)|就能|才能|保证|保持|为后面|给后面|后续|下一步|"
            r"原因|原理|作用|因为|所以|使(?!用).{0,12}(?:更|能|可以)",
            narration,
        ))
    return bool(re.search(
        r"\b(why|purpose|so that|in order to|this step|the key is|because|therefore|"
        r"prevents?|avoids?|allows?|helps?|ensures?|keeps?|makes sure|so it can)\b",
        str(text or ""),
        flags=re.IGNORECASE,
    ))


def _custom_style_process_block_like(block: Dict, language: str) -> bool:
    narration = str(block.get("narration") or block.get("text") or "")
    visual = str(block.get("visual") or "")
    facts = " ".join(str(fact) for fact in block.get("visual_facts") or [] if str(fact).strip())
    combined = f"{visual} {facts} {narration}"
    if _block_visual_duration(block) >= 4.0 and len(re.sub(r"\s+", "", narration)) >= 6:
        return True
    if (language or "").lower().startswith("zh"):
        return bool(re.search(
            r"机器|机械|设备|工人|材料|工具|液压|刀|辊|刷|机械爪|夹|切|压|拉|送|筛|"
            r"分离|破碎|粉碎|剥|拆|装|焊|磨|打|转|输送|控制|批次|轮胎|橡胶|纤维|钢丝|"
            r"生产|回收|处理|工序|步骤|流程|操作|加工|清洗|烘干|筛选|热压|成型|夹具|料斗|电机",
            combined,
        ))
    return bool(re.search(
        r"\b(machine|mechanical|worker|material|tool|equipment|process|operation|step|cut|press|"
        r"separate|sort|feed|conveyor|hydraulic|roller|blade|recycle|production|batch)\b",
        combined,
        flags=re.IGNORECASE,
    ))


def _custom_style_operation_logic_failure_details(
    data: Dict,
    language: str,
    custom_style_prompt: Optional[str],
    duration: Optional[float] = None,
) -> Optional[Dict]:
    if not _custom_style_requires_operation_logic(custom_style_prompt, language):
        return None
    blocks = data.get("narration_blocks") or []
    if duration is not None:
        try:
            blocks = _normalize_narration_blocks(blocks, float(duration or 0.0))
        except Exception:
            blocks = [block for block in blocks if isinstance(block, dict)]
    eligible = []
    logic_indexes = []
    action_only = []
    for index, block in enumerate(blocks or [], start=1):
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        narration = str(block.get("narration") or block.get("text") or "").strip()
        if not narration or not _custom_style_process_block_like(block, language):
            continue
        eligible.append(index)
        if _narration_has_operation_logic(narration, language):
            logic_indexes.append(index)
        else:
            action_only.append({
                "block_index": index,
                "narration": _limit_text_chars(narration, 80),
                "visual": _limit_text_chars(str(block.get("visual") or ""), 80),
            })
    if not eligible:
        narration = str(data.get("narration") or "").strip()
        if not narration:
            return None
        if _narration_has_operation_logic(narration, language):
            return None
        return {
            "eligible_count": 1,
            "logic_count": 0,
            "required_count": 1,
            "action_only_indexes": [1],
            "action_only_examples": [{
                "block_index": 1,
                "narration": _limit_text_chars(narration, 120),
                "visual": "",
            }],
        }
    required = min(len(eligible), max(1 if len(eligible) <= 2 else 2, int(math.ceil(len(eligible) * 0.45))))
    if len(logic_indexes) >= required:
        return None
    return {
        "eligible_count": len(eligible),
        "logic_count": len(logic_indexes),
        "required_count": required,
        "action_only_indexes": [item["block_index"] for item in action_only],
        "action_only_examples": action_only[:8],
    }


def _validate_custom_style_operation_logic(
    data: Dict,
    language: str,
    custom_style_prompt: Optional[str],
    duration: Optional[float] = None,
) -> None:
    details = _custom_style_operation_logic_failure_details(data, language, custom_style_prompt, duration)
    if not details:
        return
    action_only_indexes = details.get("action_only_indexes") or []
    examples = details.get("action_only_examples") or []
    raise Exception(
        "Custom style operation logic validation failed. "
        "The custom_style_prompt asks for process/operation logic, but only "
        f"{details.get('logic_count', 0)}/{details.get('eligible_count', 0)} eligible narrated process blocks explain purpose, why, operation logic, or same-range result; "
        f"required at least {details.get('required_count', 0)}. "
        f"Action-only block indexes: {action_only_indexes[:16]}. "
        f"Examples: {json.dumps(examples, ensure_ascii=False)}. "
        "Rewrite those blocks so each starts from visible action/tool/material and adds a concise same-range purpose, operation-logic, or result sentence. Do not invent unsupported claims."
    )


def _ensure_complete_commentary_ending_blocks(blocks: List[Dict], language: str) -> List[Dict]:
    return list(blocks or [])


def _episode_block_range_from_source(blocks: List[Dict], start: float, end: float) -> Tuple[int, int]:
    matched = []
    for index, block in enumerate(blocks, start=1):
        block_start = float(block.get("start") or 0.0)
        block_end = float(block.get("end") or block_start)
        if block_end > start and block_start < end:
            matched.append(index)
    if not matched:
        nearest = min(
            range(1, len(blocks) + 1),
            key=lambda idx: abs(float(blocks[idx - 1].get("start") or 0.0) - start),
        )
        return nearest, nearest
    return matched[0], matched[-1]


def _normalize_commentary_episodes(data: Dict, duration: float, target_duration: str) -> None:
    raw_plan = data.get("episode_plan") if isinstance(data.get("episode_plan"), dict) else {}
    raw_episodes = data.get("episodes") if isinstance(data.get("episodes"), list) else []
    reason = str(raw_plan.get("reason") or "").strip()
    blocks = data.get("narration_blocks") or []
    if target_duration != "full" or not blocks:
        data["episode_plan"] = {"should_split": False, "reason": reason}
        data["episodes"] = []
        return

    episodes = []
    previous_end_block = 0
    for raw in raw_episodes:
        if not isinstance(raw, dict):
            continue
        source_start = raw.get("start")
        source_end = raw.get("end")
        if raw.get("start_block") is None and raw.get("end_block") is None and source_start is not None and source_end is not None:
            start_block, end_block = _episode_block_range_from_source(
                blocks,
                _clamp_float(source_start, 0.0, 0.0, duration),
                _clamp_float(source_end, duration, 0.0, duration),
            )
        else:
            start_block = _clamp_int(raw.get("start_block"), previous_end_block + 1, 1, len(blocks))
            end_block = _clamp_int(raw.get("end_block"), start_block, start_block, len(blocks))
        if start_block <= previous_end_block:
            start_block = previous_end_block + 1
        if start_block > len(blocks) or end_block < start_block:
            continue
        end_block = min(end_block, len(blocks))
        first_block = blocks[start_block - 1]
        last_block = blocks[end_block - 1]
        episode_number = _clamp_int(raw.get("episode_number"), len(episodes) + 1, 1, 99)
        episodes.append({
            "episode_number": episode_number,
            "title": str(raw.get("title") or f"第{len(episodes) + 1}集").strip(),
            "summary": str(raw.get("summary") or "").strip(),
            "start_block": start_block,
            "end_block": end_block,
            "start": float(first_block.get("start") or 0.0),
            "end": float(last_block.get("end") or first_block.get("start") or 0.0),
            "block_count": end_block - start_block + 1,
        })
        previous_end_block = end_block

    for index, episode in enumerate(episodes, start=1):
        episode["episode_number"] = index
        if not episode.get("title"):
            episode["title"] = f"第{index}集"

    data["episode_plan"] = {"should_split": bool(episodes), "reason": reason}
    data["episodes"] = episodes


def _normalize_script_timeline(data: Dict, duration: float, target_duration: str, language: str = "") -> None:
    blocks = _normalize_narration_blocks(data.get("narration_blocks") or [], duration)
    if blocks:
        edit_segments = _resolve_edit_segments_for_target(data.get("edit_segments", []), duration, target_duration)
        for index, block in enumerate(blocks):
            if str(block.get("visual") or "").strip():
                continue
            if index < len(edit_segments):
                block["visual"] = str(edit_segments[index].get("reason") or "").strip()
        data["narration_blocks"] = blocks
        data["edit_segments"] = _narration_blocks_to_edit_segments(data.get("narration_blocks") or blocks)
        if target_duration == "full":
            data.setdefault("cut_strategy", [])
    else:
        data["edit_segments"] = _resolve_edit_segments_for_target(data.get("edit_segments", []), duration, target_duration)
    _normalize_commentary_episodes(data, duration, target_duration)


def _normalize_script_narration(data: Dict) -> str:
    narration = str(data.get("narration") or "").strip()
    block_narration = _narration_from_blocks(data)
    if block_narration:
        narration = block_narration
    return narration


def _narration_texts_for_repeat_validation(data: Dict) -> List[str]:
    block_texts = [
        str(block.get("narration") or "").strip()
        for block in data.get("narration_blocks") or []
        if isinstance(block, dict) and str(block.get("narration") or "").strip()
    ]
    if block_texts:
        return block_texts
    narration = str(data.get("narration") or "").strip()
    return [narration] if narration else []


def _first_visual_analysis_label_artifact_index(text: str, language: str = "") -> int:
    value = str(text or "")
    if not value:
        return -1
    if not (language or "").lower().startswith("zh"):
        return -1

    def has_cjk_before(index: int) -> bool:
        return bool(re.search(r"[\u3400-\u9fff]", value[:max(0, index)]))

    candidates = []
    for match in re.finditer(r"(?<![0-9A-Za-z])\d{1,5}(?:\.\d{1,3})?\s*s\s*:\s*[A-Za-z]", value):
        candidates.append(match.start())
    for match in re.finditer(r"[A-Za-z][A-Za-z0-9 _-]*\s*/\s*[A-Za-z]", value):
        if has_cjk_before(match.start()):
            candidates.append(match.start())

    latin_separator = r"[\s,.;:!?/\-\u3001\u3002\uff0c\uff01\uff1f\uff1b\uff1a]+"
    latin_phrase = rf"[A-Za-z]{{2,}}(?:{latin_separator}[A-Za-z]{{2,}})+"
    for match in re.finditer(latin_phrase, value):
        if has_cjk_before(match.start()):
            candidates.append(match.start())
    for match in re.finditer(r"\b[a-z]{10,}\b", value, flags=re.IGNORECASE):
        if has_cjk_before(match.start()):
            candidates.append(match.start())
    return min(candidates) if candidates else -1


def _contains_visual_analysis_label_artifact(text: str, language: str = "") -> bool:
    return _first_visual_analysis_label_artifact_index(text, language) >= 0


def _strip_visual_analysis_label_artifact(text: str, language: str = "") -> str:
    clean = str(text or "").strip()
    if not clean or not _contains_visual_analysis_label_artifact(clean, language):
        return clean
    if not (language or "").lower().startswith("zh"):
        return clean
    timestamp_clean = re.sub(
        r"(?<![0-9A-Za-z])\d{1,5}(?:\.\d{1,3})?\s*s\s*:\s*[A-Za-z][A-Za-z0-9 _-]*",
        "",
        clean,
    ).strip()
    if timestamp_clean and timestamp_clean != clean and not _contains_visual_analysis_label_artifact(timestamp_clean, language):
        return timestamp_clean.rstrip("锛?銆傦紒锛?? ")
    artifact_index = _first_visual_analysis_label_artifact_index(clean, language)
    if artifact_index >= 0:
        prefix = clean[:artifact_index]
    else:
        prefix = re.split(
            r"(?<![0-9A-Za-z])\d{1,5}(?:\.\d{1,3})?\s*s\s*:\s*[A-Za-z][A-Za-z0-9 _-]*|"
            r"[A-Za-z][A-Za-z0-9 _-]*(?:\s*/\s*[A-Za-z][A-Za-z0-9 _-]*)*|"
            r"\b[a-z]{10,}\b",
            clean,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
    return prefix.rstrip(" \t\r\n,.;:!?\u3001\u3002\uff0c\uff01\uff1f\uff1b\uff1a")


def _strip_visual_analysis_label_artifacts_from_script(data: Dict, language: str = "") -> None:
    if not isinstance(data, dict) or not (language or "").lower().startswith("zh"):
        return
    changed = False
    for block in data.get("narration_blocks") or []:
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        narration = str(block.get("narration") or "")
        if not _contains_visual_analysis_label_artifact(narration, language):
            continue
        cleaned = _strip_visual_analysis_label_artifact(narration, language)
        if (
            cleaned
            and re.search(r"[\u3400-\u9fff]", cleaned)
            and not _contains_visual_analysis_label_artifact(cleaned, language)
        ):
            block["narration"] = cleaned
            changed = True
    narration = str(data.get("narration") or "")
    if _contains_visual_analysis_label_artifact(narration, language):
        cleaned = _strip_visual_analysis_label_artifact(narration, language)
        block_narration = _narration_from_blocks(data)
        if block_narration and not _contains_visual_analysis_label_artifact(block_narration, language):
            data["narration"] = block_narration
            changed = False
        elif (
            cleaned
            and re.search(r"[\u3400-\u9fff]", cleaned)
            and not _contains_visual_analysis_label_artifact(cleaned, language)
        ):
            data["narration"] = cleaned
    if changed and data.get("narration_blocks"):
        data["narration"] = _narration_from_blocks(data) or str(data.get("narration") or "")


def _validate_no_visual_analysis_label_artifacts(data: Dict, language: str) -> None:
    if not (language or "").lower().startswith("zh"):
        return
    for index, block in enumerate(data.get("narration_blocks") or [], start=1):
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        narration = str(block.get("narration") or "")
        if _contains_visual_analysis_label_artifact(narration, language):
            raise Exception(
                "AI narration contains raw visual-analysis labels instead of natural commentary. "
                f"Block {index} copied label-like text into narration: {narration[:120]!r}. "
                "Rewrite the block in fluent Chinese using the selected commentary style and the timestamped visual evidence; "
                "do not copy timestamped frame labels, visual facts, slash-separated stage labels, or edit_value/pace labels into the final narration."
            )
    narration = str(data.get("narration") or "")
    if _contains_visual_analysis_label_artifact(narration, language):
        block_narration = _narration_from_blocks(data)
        if block_narration and not _contains_visual_analysis_label_artifact(block_narration, language):
            data["narration"] = block_narration
            return
        raise Exception(
            "AI narration contains raw visual-analysis labels instead of natural commentary. "
            "The top-level narration copied label-like text. "
            "Rewrite the narration in fluent Chinese using the selected commentary style and the timestamped visual evidence; "
            "do not copy timestamped frame labels, visual facts, slash-separated stage labels, or edit_value/pace labels into the final narration."
        )


def _sanitize_and_validate_no_visual_analysis_label_artifacts(data: Dict, language: str) -> None:
    _sanitize_generated_commentary_script(data, language)
    if isinstance(data, dict):
        data["narration"] = _normalize_script_narration(data)
    _validate_no_visual_analysis_label_artifacts(data, language)


def _sentence_repeat_key(sentence: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "", str(sentence or "")).lower()


def _scene_fact_sentence(block: Dict, language: str) -> str:
    visual = str(block.get("visual") or "").strip()
    visual_facts = block.get("visual_facts") if isinstance(block.get("visual_facts"), list) else []
    fact_parts = [str(fact).strip() for fact in visual_facts if str(fact).strip()]
    if (language or "").lower().startswith("zh"):
        zh_fact_parts = [
            part for part in fact_parts
            if re.search(r"[\u3400-\u9fff]", part) and not _contains_visual_analysis_label_artifact(part, language)
        ]
        if zh_fact_parts:
            parts = zh_fact_parts
        elif (
            visual
            and re.search(r"[\u3400-\u9fff]", visual)
            and not _contains_visual_analysis_label_artifact(visual, language)
        ):
            parts = [visual]
        else:
            parts = []
    else:
        parts = fact_parts or ([visual] if visual else [])
    cleaned = []
    for part in parts:
        text = _strip_camera_meta_phrasing_text(part)
        text = re.sub(r"^(?:当前)?(?:画面|视频)(?:显示|展示|里|中)?", "", text).strip(" ，,。")
        text = text.replace("展示最终结果", "呈现最终结果")
        if text:
            cleaned.append(text)
    if not cleaned:
        return ""
    sentence = "，".join(dict.fromkeys(cleaned))
    if (language or "").lower().startswith("zh"):
        return sentence.rstrip("。！？!?") + "。"
    return sentence.rstrip(".!?") + "."


def _normalize_zh_render_narration_text(block: Dict) -> str:
    narration = _strip_camera_meta_phrasing_text(str(block.get("narration") or ""))
    if _contains_visual_analysis_label_artifact(narration, "zh"):
        cleaned = _strip_visual_analysis_label_artifact(narration, "zh")
        if cleaned and re.search(r"[\u3400-\u9fff]", cleaned) and not _contains_visual_analysis_label_artifact(cleaned, "zh"):
            narration = cleaned
        else:
            return narration.rstrip("。")
    visual_sentence = _scene_fact_sentence(block, "zh").rstrip("。")
    original_narration = str(block.get("narration") or "")
    if visual_sentence and (re.search(r"[A-Za-z]{3,}", narration) or "画面里" in original_narration):
        if "画面里" in original_narration:
            prefix = original_narration.split("画面里", 1)[0].strip("，,。 ")
        else:
            prefix = re.split(r"[A-Za-z]{3,}", narration, maxsplit=1)[0].rstrip("，,。 ")
        narration = (prefix + "。" if prefix else "") + visual_sentence
    narration = narration.replace("展示最终结果", "呈现最终结果").replace("展示结果", "呈现结果")
    return narration.rstrip("。")


def _normalize_full_mode_render_narration_text(block: Dict, language: str) -> str:
    if (language or "").lower().startswith("zh"):
        return _normalize_zh_render_narration_text(block)
    return _strip_camera_meta_phrasing_text(str(block.get("narration") or ""))


def _normalize_full_mode_render_narration_blocks(blocks: List[Dict], language: str) -> List[Dict]:
    normalized = []
    for block in blocks or []:
        item = dict(block)
        if not bool(item.get("pause")):
            item["narration"] = _normalize_full_mode_render_narration_text(item, language)
        normalized.append(item)
    return normalized


def _validate_no_repeated_commentary_text(data: Dict) -> None:
    if data.get("_skip_repeat_validation"):
        return
    texts = _narration_texts_for_repeat_validation(data)
    if not texts:
        return
    joined = "\n".join(texts)
    for phrase in COMMENTARY_REPEAT_LIMIT_PHRASES:
        count = len(re.findall(re.escape(phrase), joined))
        if count > 2:
            raise Exception(
                "AI commentary narration repeats a catchphrase too many times. "
                f"Phrase {phrase!r} appears {count} times; rewrite repeated lines instead of relying on backend text cleanup."
            )


def _banned_phrase_instruction() -> str:
    phrases = "、".join(f"“{phrase}”" for phrase in COMMENTARY_BANNED_PHRASES)
    return (
        f"- Never use these banned phrases anywhere in the returned JSON: {phrases}. Describe the concrete scene directly instead of using meta-summary wording.\n"
        "- In narration and narration_blocks.narration, do not use the word '镜头' or camera/meta phrasing like '镜头切到', '镜头拉近', '镜头里', '镜头展示', or '镜头带我们'. Do not use detached phrases like '画面里', '画面中', '视频里', '视频中', '当前画面', '画面显示', '画面展示', '可以看到', or '能看到'. Do not mention the writing itself with phrases like '这段解说', '解说词', '旁白稿', '脚本说明', '交代完整', or '自然收住'. Also avoid any equivalent editorial wrap-up wording such as '到此告一段落', '该交代的都交代完', '作为收尾', '不再展开', 'this narration wraps up', or 'the script is complete'. Describe the subject and action directly instead."
    )


def _validate_no_banned_commentary_phrases(data: Dict) -> None:
    serialized = json.dumps(data, ensure_ascii=False)
    for phrase in COMMENTARY_BANNED_PHRASES:
        if phrase in serialized:
            raise Exception(f"AI commentary output contains banned phrase: {phrase}")
    for phrase in COMMENTARY_AUTO_FILLED_PLACEHOLDER_PHRASES:
        if phrase in serialized:
            raise Exception(
                "AI commentary output contains an auto-filled placeholder phrase. "
                "OpenShorts will not accept backend filler or bridge text; select real scene-matched source ranges from the visual evidence."
            )
    for block in data.get("narration_blocks") or []:
        internal_safe_auto_fill = (
            isinstance(block, dict)
            and _coerce_bool(block.get("auto_filled_visual_budget"))
            and _coerce_bool(block.get("pause"))
            and not str(block.get("narration") or "").strip()
            and str(block.get("visual") or "").strip() == COMMENTARY_AUTO_FILLED_BRIDGE_VISUAL
        )
        if isinstance(block, dict) and _coerce_bool(block.get("auto_filled_visual_budget")) and not internal_safe_auto_fill:
            raise Exception(
                "AI commentary output contains auto_filled_visual_budget. "
                "OpenShorts will not accept backend-filled visual budget blocks; AI must choose real scene-matched source ranges."
            )


def _validate_no_banned_narration_patterns(data: Dict) -> None:
    narration_parts = [str(data.get("narration") or "")]
    for block in data.get("narration_blocks") or []:
        if isinstance(block, dict):
            narration_parts.append(str(block.get("narration") or ""))
    narration = "\n".join(narration_parts)
    for pattern in COMMENTARY_NARRATION_BANNED_PATTERNS:
        match = pattern.search(narration)
        if match:
            raise Exception(f"AI commentary narration contains camera/meta phrasing or editorial phrasing: {match.group(0)}")


def _validate_no_editorial_meta_narration_patterns(data: Dict) -> None:
    narration_parts = [str(data.get("narration") or "")]
    for block in data.get("narration_blocks") or []:
        if isinstance(block, dict):
            narration_parts.append(str(block.get("narration") or ""))
    narration = "\n".join(narration_parts)
    editorial_patterns = COMMENTARY_NARRATION_BANNED_PATTERNS[3:]
    for pattern in editorial_patterns:
        match = pattern.search(narration)
        if match:
            raise Exception(f"AI commentary narration contains editorial phrasing: {match.group(0)}")


def _strip_camera_meta_phrasing_text(text: str) -> str:
    cleaned = str(text or "")
    for phrase in (
        "镜头切到",
        "镜头转到",
        "镜头来到",
        "镜头拉近",
        "镜头推进",
        "镜头对准",
        "镜头展示",
        "镜头给到",
        "镜头带到",
        "镜头里",
        "画面里",
        "画面中",
        "视频里",
        "视频中",
        "当前画面",
        "当前可见",
        "画面显示",
        "画面展示",
        "视频显示",
        "视频展示",
        "可以看到",
        "能看到",
    ):
        cleaned = cleaned.replace(phrase, "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^[，,。！？!?；;：:\s]+", "", cleaned)
    cleaned = re.sub(r"\s+([，,。！？!?；;：:])", r"\1", cleaned)
    return cleaned


def _strip_camera_meta_phrasing(data: Dict) -> None:
    if not isinstance(data, dict):
        return
    if data.get("narration") is not None:
        data["narration"] = _strip_camera_meta_phrasing_text(str(data.get("narration") or ""))
    for block in data.get("narration_blocks") or []:
        if not isinstance(block, dict) or block.get("narration") is None:
            continue
        block["narration"] = _strip_camera_meta_phrasing_text(str(block.get("narration") or ""))
    if data.get("narration_blocks"):
        data["narration"] = _narration_from_blocks(data) or str(data.get("narration") or "")


def _sanitize_generated_commentary_script(data: Dict, language: str = "") -> None:
    _strip_camera_meta_phrasing(data)
    _strip_visual_analysis_label_artifacts_from_script(data, language)
    if isinstance(data, dict):
        data["_generated_sanitized"] = True


def _openai_visual_analysis_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, OPENAI_VISUAL_ANALYSIS_CACHE)


def _load_cached_openai_visual_analysis(output_dir: Optional[str], source_video_path: Optional[str] = None) -> Optional[Dict]:
    if not output_dir:
        return None
    cache_path = _openai_visual_analysis_cache_path(output_dir)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if source_video_path and data.get("source_video_path") != os.path.abspath(source_video_path):
        return None
    if not _openai_visual_analysis_has_edit_value_scores(data):
        return None
    return data


def _openai_visual_analysis_has_edit_value_scores(data: Optional[Dict]) -> bool:
    if not isinstance(data, dict):
        return False
    for item in (data.get("candidate_segments") or []) + (data.get("observations") or []):
        if not isinstance(item, dict):
            continue
        if any(
            key in item
            for key in (
                "importance",
                "importance_score",
                "visual_importance",
                "interest_score",
                "viewer_interest",
                "watch_value",
                "edit_value",
            )
        ):
            return True
    return False


def _text_matches_any_keyword(text: str, keywords: Tuple[str, ...]) -> bool:
    haystack = (text or "").lower()
    if not haystack:
        return False
    for keyword in keywords:
        clean = str(keyword or "").strip().lower()
        if clean and clean in haystack:
            return True
    return False


def _block_visual_grounding_text(block: Dict) -> str:
    parts = [str(block.get("visual") or "")]
    visual_facts = block.get("visual_facts")
    if isinstance(visual_facts, list):
        parts.extend(str(fact or "") for fact in visual_facts)
    return " ".join(part for part in parts if part)


def _visual_analysis_observation_items_for_block(block: Dict, visual_analysis: Optional[Dict]) -> List[Dict]:
    if not visual_analysis:
        return []
    try:
        start = float(block.get("start"))
        end = float(block.get("end"))
    except (TypeError, ValueError):
        return []
    if end <= start:
        return []
    observations = []
    for item in visual_analysis.get("observations") or []:
        if not isinstance(item, dict):
            continue
        timestamp = item.get("timestamp")
        if isinstance(timestamp, (int, float)) and start <= float(timestamp) <= end:
            observations.append(item)
    return observations


def _visual_analysis_items_for_block(block: Dict, visual_analysis: Optional[Dict]) -> List[Dict]:
    if not visual_analysis:
        return []
    try:
        start = float(block.get("start"))
        end = float(block.get("end"))
    except (TypeError, ValueError):
        return []
    if end <= start:
        return []
    observations = _visual_analysis_observation_items_for_block(block, visual_analysis)
    if observations:
        return observations

    items = []
    for item in visual_analysis.get("candidate_segments") or []:
        if not isinstance(item, dict):
            continue
        try:
            item_start = float(item.get("start"))
            item_end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if item_end > start and item_start < end:
            items.append(item)
    return items


def _visual_analysis_text_for_block(block: Dict, visual_analysis: Optional[Dict]) -> str:
    parts = []
    for item in _visual_analysis_items_for_block(block, visual_analysis):
        for key in ("process_stage", "visual", "reason"):
            value = str(item.get(key) or "").strip()
            if value:
                parts.append(value)
    return " ".join(parts)


def _visual_analysis_observation_text_for_block(block: Dict, visual_analysis: Optional[Dict]) -> str:
    parts = []
    for item in _visual_analysis_observation_items_for_block(block, visual_analysis):
        for key in ("process_stage", "visual", "reason"):
            value = str(item.get(key) or "").strip()
            if value:
                parts.append(value)
    return " ".join(parts)


def _visual_analysis_frame_timestamps(visual_analysis: Optional[Dict]) -> List[float]:
    if not visual_analysis:
        return []
    timestamps = []
    for frame in visual_analysis.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        try:
            timestamps.append(round(float(frame.get("timestamp")), 3))
        except (TypeError, ValueError):
            continue
    if timestamps:
        return sorted(set(timestamps))
    for item in visual_analysis.get("observations") or []:
        if not isinstance(item, dict):
            continue
        try:
            timestamps.append(round(float(item.get("timestamp")), 3))
        except (TypeError, ValueError):
            continue
    return sorted(set(timestamps))


def _validate_block_evidence_timestamps(block: Dict, visual_analysis: Optional[Dict], index: int) -> None:
    if not visual_analysis or visual_analysis.get("provider") != "openai_compatible":
        return
    frame_timestamps = _visual_analysis_frame_timestamps(visual_analysis)
    if not frame_timestamps:
        return
    evidence = block.get("evidence_timestamps")
    if not isinstance(evidence, list) or not evidence:
        raise Exception(
            "OpenAI commentary block is missing timestamp evidence. "
            f"Block {index} must include evidence_timestamps copied from the extracted source-video frame timestamps."
        )
    try:
        start = float(block.get("start"))
        end = float(block.get("end"))
    except (TypeError, ValueError):
        return
    valid = False
    frame_set = {round(timestamp, 3) for timestamp in frame_timestamps}
    for value in evidence:
        try:
            timestamp = round(float(value), 3)
        except (TypeError, ValueError):
            continue
        if start - 0.35 <= timestamp <= end + 0.35 and timestamp in frame_set:
            valid = True
            continue
        raise Exception(
            "OpenAI commentary block timestamp evidence does not match its visual range. "
            f"Block {index} uses evidence timestamp {timestamp:.3f}s, but it must be one of the extracted frame timestamps inside {start:.3f}-{end:.3f}s."
        )
    if not valid:
        raise Exception(
            "OpenAI commentary block has no usable timestamp evidence. "
            f"Block {index} must cite at least one extracted frame timestamp inside its start/end range."
        )


def _repair_short_narration_visual_ranges(blocks: List[Dict], language: str) -> List[Dict]:
    repaired = []
    for block in blocks or []:
        item = dict(block)
        if bool(item.get("pause")):
            repaired.append(item)
            continue
        shortfall = _narration_density_shortfall_details(item, len(repaired) + 1, language)
        narration = str(item.get("narration") or item.get("text") or "").strip()
        current_chars = len(re.sub(r"\s+", "", narration))
        expected_chars = int(shortfall.get("expected_chars") or 0) if shortfall else _expected_narration_chars_for_visual_duration(_block_visual_duration(item), language)
        visual_text = str(item.get("visual") or "").strip()
        visual_facts = item.get("visual_facts") if isinstance(item.get("visual_facts"), list) else []
        fact_text = "，".join(str(fact).strip() for fact in visual_facts if str(fact).strip())
        if (language or "").lower().startswith("zh"):
            safe_source_text = _scene_fact_sentence(item, language).strip()
            source_text = safe_source_text.rstrip("。！？!?") if safe_source_text else ""
        else:
            source_text = fact_text or visual_text
        if current_chars < expected_chars and source_text:
            additions = []
            while len(re.sub(r"\s+", "", narration + "".join(additions))) < expected_chars:
                additions.append(f"{source_text}。")
            item["narration"] = narration + "".join(additions)
            current_chars = len(re.sub(r"\s+", "", item["narration"]))
        shortfall = _narration_density_shortfall_details(item, len(repaired) + 1, language)
        if shortfall:
            speed = _safe_video_speed(item.get("video_speed"))
            start = float(item.get("start") or 0.0)
            max_playable = _max_narrated_visual_seconds_for_chars(current_chars, language)
            if max_playable > 0:
                original_end = float(item.get("end") or start)
                split_end = round(start + max_playable * speed, 3)
                item["end"] = split_end
                if original_end - split_end > FULL_MODE_VALIDATION_EPSILON_SECONDS:
                    tail = dict(item)
                    tail["start"] = split_end
                    tail["end"] = round(
                        min(
                            original_end,
                            split_end + FULL_MODE_MAX_NARRATION_SILENCE_TAIL_SECONDS * speed,
                        ),
                        3,
                    )
                    tail["narration"] = ""
                    tail["pause"] = True
                    tail["rate"] = "+0%"
                    tail["pitch"] = "+0Hz"
                    item_evidence = _filter_block_evidence_timestamps_for_range(
                        item,
                        item["start"],
                        item["end"],
                    )
                    if _block_evidence_timestamps(block) and not item_evidence:
                        repaired.append(item)
                        continue
                    item["evidence_timestamps"] = item_evidence
                    tail["evidence_timestamps"] = _filter_block_evidence_timestamps_for_range(
                        block,
                        tail["start"],
                        tail["end"],
                    )
                    repaired.append(item)
                    repaired.append(tail)
                    continue
        repaired.append(item)
    return repaired


def _repair_full_mode_small_density_shortfalls(
    blocks: List[Dict],
    min_visual_seconds: float,
    language: str,
) -> List[Dict]:
    if not blocks:
        return blocks
    repaired = [dict(block) for block in blocks]
    visual_seconds = sum(_block_visual_duration(block) for block in repaired)
    changed = False
    for index, block in enumerate(repaired, start=1):
        shortfall = _narration_density_shortfall_details(block, index, language)
        current_playable = _block_visual_duration(block)
        narration_text = str(block.get("narration") or block.get("text") or "").strip()
        current_chars = len(re.sub(r"\s+", "", narration_text))
        expected_chars = _expected_narration_chars_for_visual_duration(current_playable, language)
        if not shortfall and current_chars >= expected_chars:
            continue
        missing_chars = max(0, expected_chars - current_chars)
        if missing_chars <= 0:
            continue
        # Only trim tiny sync misses. Large sparse ranges need model-level rewrite
        # because cutting them would destroy the selected visual story.
        if missing_chars > max(20, int(math.ceil(expected_chars * 0.30))):
            continue
        max_playable = _max_narrated_visual_seconds_for_chars(current_chars, language)
        trim_playable = current_playable - max_playable
        if trim_playable <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
            continue
        if trim_playable > max(8.0, current_playable * 0.30):
            continue
        if (
            min_visual_seconds > 0
            and visual_seconds - trim_playable < min_visual_seconds - FULL_MODE_VALIDATION_EPSILON_SECONDS
        ):
            continue
        speed = _safe_video_speed(block.get("video_speed"))
        start = float(block.get("start") or 0.0)
        new_end = start + (max_playable * speed)
        if new_end - start < 1.0:
            continue
        block["end"] = round(new_end, 3)
        visual_seconds -= trim_playable
        changed = True
    return repaired if changed else blocks


def _narration_density_shortfall_details(block: Dict, index: int, language: str) -> Optional[Dict]:
    if bool(block.get("pause")):
        return None
    narration_text = str(block.get("narration") or block.get("text") or "").strip()
    narration_chars = len(re.sub(r"\s+", "", narration_text))
    block_duration = _block_visual_duration(block)
    locked_min_chars = 0
    try:
        locked_min_chars = int(math.ceil(float(block.get("_min_narration_chars") or 0)))
    except (TypeError, ValueError):
        locked_min_chars = 0
    if block_duration < 12.0 and not bool(block.get("_locked_edit_plan")):
        return None
    max_duration = _max_narrated_visual_seconds_for_chars(narration_chars, language)
    estimated_voice_seconds = _estimated_voiceover_seconds_for_chars(narration_chars, language)
    minimum_voice_seconds = _minimum_voiceover_seconds_for_visual_duration(block_duration)
    if bool(block.get("_locked_edit_plan")) and locked_min_chars > 0 and narration_chars < locked_min_chars:
        if (
            estimated_voice_seconds + FULL_MODE_VALIDATION_EPSILON_SECONDS >= minimum_voice_seconds
            and block_duration - estimated_voice_seconds <= FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS + FULL_MODE_VALIDATION_EPSILON_SECONDS
        ):
            return None
        return {
            "block_index": index,
            "array_index": index - 1,
            "current_chars": narration_chars,
            "playable_seconds": block_duration,
            "expected_chars": locked_min_chars,
            "estimated_voice_seconds": estimated_voice_seconds,
            "minimum_voice_seconds": minimum_voice_seconds,
            "trailing_silence_seconds": max(0.0, block_duration - estimated_voice_seconds),
            "block": block,
        }
    if (
        block_duration <= max_duration + FULL_MODE_VALIDATION_EPSILON_SECONDS
        and (locked_min_chars <= 0 or narration_chars >= locked_min_chars)
    ):
        return None
    expected_chars = max(
        _expected_narration_chars_for_visual_duration(block_duration, language),
        locked_min_chars,
    )
    if bool(block.get("_locked_edit_plan")) and locked_min_chars > 0:
        density_floor = locked_min_chars
    else:
        density_floor = _density_floor_chars_for_visual_duration(block_duration, language)
    if narration_chars >= int(math.ceil(density_floor * FULL_MODE_NARRATION_DENSITY_MIN_RATIO)):
        return None
    return {
        "block_index": index,
        "array_index": index - 1,
        "current_chars": narration_chars,
        "playable_seconds": block_duration,
        "expected_chars": expected_chars,
        "estimated_voice_seconds": estimated_voice_seconds,
        "minimum_voice_seconds": minimum_voice_seconds,
        "trailing_silence_seconds": max(0.0, block_duration - estimated_voice_seconds),
        "block": block,
    }


def _validate_narration_density_matches_visual_duration(block: Dict, index: int, language: str) -> None:
    shortfall = _narration_density_shortfall_details(block, index, language)
    if not shortfall:
        return
    raise Exception(
        "AI narration block is too short for its selected visual range. "
        f"Block {index} has {shortfall['current_chars']} chars for {shortfall['playable_seconds']:.1f}s of playable visuals; expected at least {shortfall['expected_chars']}. "
        f"Estimated TTS is only {shortfall['estimated_voice_seconds']:.1f}s, leaving about {shortfall['trailing_silence_seconds']:.1f}s without matching narration. "
        "Shorten this block's source range, split it, add concrete scene-matched narration, or mark a brief pause=true moment. "
        "Do not rely on render-time speedup or trimming because that makes the commentary and visuals drift out of sync."
    )


def _estimated_voiceover_seconds_for_text(text: str, language: str) -> float:
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact:
        return 0.0
    if (language or "").lower().startswith("zh"):
        return len(compact) / 4.2
    words = re.findall(r"\S+", str(text or ""))
    return (len(words) / 2.6) if words else (len(compact) / 12.0)


def _shorten_narration_to_fit_visual(text: str, max_voice_seconds: float, language: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    if (language or "").lower().startswith("zh"):
        max_chars = max(1, int(max_voice_seconds * 4.2))
        return _trim_narration_to_compact_chars(clean, max_chars, language)
    words = re.findall(r"\S+", clean)
    max_words = max(1, int(max_voice_seconds * 2.6))
    if len(words) <= max_words:
        return clean
    shortened = " ".join(words[:max_words])
    trimmed = shortened.rstrip(",.!?")
    return (trimmed or shortened) + "."


def _trim_chinese_narration_to_chars(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", "", str(text or "")).strip()
    max_chars = int(max(0, max_chars))
    if max_chars <= 0 or not compact:
        return ""
    if len(compact) <= max_chars:
        return compact
    if max_chars == 1:
        trimmed = compact[:1].rstrip("，,。！？!?")
        return trimmed or compact[:1]
    candidate = compact[:max_chars]
    if candidate[-1:] in "。！？!?":
        return candidate
    if candidate[-1:] in "，,":
        trimmed = candidate.rstrip("，,。！？!?")
        return (trimmed + "。") if trimmed and len(trimmed) < max_chars else (trimmed or candidate)
    body = compact[: max_chars - 1].rstrip("，,。！？!?")
    return (body + "。") if body else candidate


def _trim_narration_to_compact_chars(text: str, max_chars: int, language: str) -> str:
    clean = str(text or "").strip()
    max_chars = int(max(0, max_chars))
    if max_chars <= 0 or not clean:
        return ""
    compact = re.sub(r"\s+", "", clean)
    if len(compact) <= max_chars:
        return clean
    if (language or "").lower().startswith("zh"):
        return _trim_chinese_narration_to_chars(clean, max_chars)
    words = re.findall(r"\S+", clean)
    kept = []
    used = 0
    for word in words:
        compact_word = re.sub(r"\s+", "", word)
        if not compact_word:
            continue
        if used + len(compact_word) > max_chars:
            break
        kept.append(word)
        used += len(compact_word)
    if kept:
        result = " ".join(kept).rstrip(",.!?")
        if result and len(re.sub(r"\s+", "", result)) < max_chars:
            result += "."
        return result or "".join(words)[:max_chars]
    trimmed = compact[:max_chars].rstrip(",.!?")
    return trimmed or compact[:max_chars]


def _visual_analysis_speed_candidate_segments(visual_analysis: Optional[Dict], duration: float = 0.0) -> List[Dict]:
    if not visual_analysis:
        return []
    source_duration = max(0.0, float(duration or 0.0))
    candidates = []
    for item in visual_analysis.get("candidate_segments") or []:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if source_duration > 0:
            start = max(0.0, min(source_duration, start))
            end = max(0.0, min(source_duration, end))
        if end - start < 1.0:
            continue
        speed = _safe_video_speed(
            item.get("suggested_speed", item.get("video_speed", item.get("speed", item.get("recommended_speed"))))
        )
        if speed <= 1.0001:
            continue
        reason = str(item.get("speed_reason") or item.get("reason") or "").strip()
        candidates.append({
            "start": start,
            "end": end,
            "speed": speed,
            "reason": reason,
        })
    return candidates


def _validate_ai_video_speed_decisions(
    blocks: List[Dict],
    language: str,
    visual_analysis: Optional[Dict] = None,
    duration: float = 0.0,
) -> None:
    speed_candidates = (
        []
        if isinstance(visual_analysis, dict) and visual_analysis.get("analysis_stage") == "edited_video_commentary"
        else _visual_analysis_speed_candidate_segments(visual_analysis, duration)
    )
    for index, block in enumerate(blocks or [], start=1):
        speed = _safe_video_speed(block.get("video_speed"))
        speed_reason = str(block.get("speed_reason") or "").strip()
        if speed > 1.0001:
            if not speed_reason:
                raise Exception(
                    "AI video_speed decision is missing a visual reason. "
                    f"Block {index} uses video_speed {speed:g} but has no speed_reason. "
                    "AI must decide acceleration from the visible action and explain why this exact range remains understandable when sped up."
                )
            visual_duration = _block_visual_duration(block)
            estimated_voice_seconds = _estimated_voiceover_seconds_for_text(
                str(block.get("narration") or block.get("text") or ""),
                language,
            )
            if estimated_voice_seconds > visual_duration * FULL_MODE_RENDER_SYNC_MAX_AUDIO_SPEED + 0.75:
                raise Exception(
                    "AI video_speed makes narration too tight for its selected visual range. "
                    f"Block {index} uses video_speed {speed:g}, leaving {visual_duration:.1f}s of visuals, "
                    f"but the narration is estimated around {estimated_voice_seconds:.1f}s before TTS sync. "
                    "Lower video_speed, shorten narration, or expand this block's source range so voiceover, subtitles, and visuals stay aligned."
                )
            continue
        if bool(block.get("pause")) or not speed_candidates or speed_reason:
            continue
        block_start = float(block.get("start") or 0.0)
        block_end = float(block.get("end") or 0.0)
        block_source_seconds = max(0.0, block_end - block_start)
        if block_source_seconds < 12.0:
            continue
        for candidate in speed_candidates:
            overlap = max(0.0, min(block_end, candidate["end"]) - max(block_start, candidate["start"]))
            if overlap < min(block_source_seconds * 0.45, 10.0):
                continue
            raise Exception(
                "AI video_speed decision ignores visual speed evidence. "
                f"Block {index} keeps a source range at 1.0x even though the visual analysis suggested about {candidate['speed']:g}x "
                f"for overlapping slow/repetitive footage from {candidate['start']:.1f}s to {candidate['end']:.1f}s. "
                "Either set video_speed above 1.0 with a concrete speed_reason, shorten/cut that slow range, or explain in speed_reason why this exact kept range must play at normal speed."
                )


def _fit_locked_plan_blocks_to_render_sync(blocks: List[Dict], language: str) -> bool:
    changed = False
    for block in blocks or []:
        if not isinstance(block, dict) or not bool(block.get("_locked_edit_plan")) or bool(block.get("pause")):
            continue
        text = str(block.get("narration") or block.get("text") or "").strip()
        if not text:
            continue
        visual_duration = _block_visual_duration(block)
        if visual_duration <= 0:
            continue
        max_voice_seconds = max(
            0.1,
            visual_duration * max(0.01, FULL_MODE_RENDER_SYNC_MAX_AUDIO_SPEED),
        )
        if _estimated_voiceover_seconds_for_text(text, language) <= max_voice_seconds + 0.75:
            continue
        shortened = _shorten_narration_to_fit_visual(text, max_voice_seconds, language)
        if shortened and shortened != text:
            block["narration"] = shortened
            changed = True
    return changed


def _validate_scene_matched_narration_blocks(
    data: Dict,
    visual_analysis: Optional[Dict] = None,
    strict_scene_actions: bool = False,
) -> None:
    for index, block in enumerate(data.get("narration_blocks") or [], start=1):
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        _validate_block_evidence_timestamps(block, visual_analysis, index)


def _block_best_visual_importance(block: Dict, visual_analysis: Optional[Dict]) -> float:
    if not visual_analysis:
        return 0.0
    values = []
    try:
        start = float(block.get("start"))
        end = float(block.get("end"))
    except (TypeError, ValueError):
        return 0.0
    for item in visual_analysis.get("observations") or []:
        if not isinstance(item, dict):
            continue
        try:
            timestamp = float(item.get("timestamp"))
        except (TypeError, ValueError):
            continue
        if start <= timestamp <= end:
            values.append(_coerce_visual_score(item.get("importance")))
            values.append(_coerce_visual_score(item.get("interest_score")))
            values.append(_coerce_visual_score(item.get("viewer_interest")))
            values.append(_coerce_visual_score(item.get("watch_value")))
            if str(item.get("edit_value") or "").lower() == "must_keep":
                values.append(5.0)
            elif str(item.get("edit_value") or "").lower() == "useful":
                values.append(4.0)
    for item in visual_analysis.get("candidate_segments") or []:
        if not isinstance(item, dict):
            continue
        try:
            item_start = float(item.get("start"))
            item_end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        overlap = _segment_overlap_seconds(start, end, item_start, item_end)
        if overlap < min(max(0.75, (end - start) * 0.25), 6.0):
            continue
        values.append(_candidate_segment_importance(item, visual_analysis))
    return max(values, default=0.0)


def _validate_openai_selected_ranges_are_important(
    blocks: List[Dict],
    visual_analysis: Optional[Dict],
    duration: float,
    target_seconds: float,
) -> None:
    if not visual_analysis or visual_analysis.get("provider") != "openai_compatible" or not blocks:
        return
    if visual_analysis.get("analysis_stage") == "edited_video_commentary":
        return
    if _full_mode_preserves_source_process(duration, target_seconds):
        return
    scored_blocks = []
    for index, block in enumerate(blocks, start=1):
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        block_duration = _block_visual_duration(block)
        if block_duration <= 0:
            continue
        score = _block_best_visual_importance(block, visual_analysis)
        scored_blocks.append((index, score, block_duration))
    if not scored_blocks:
        return
    selected_seconds = sum(item[2] for item in scored_blocks)
    strong_seconds = sum(duration for _index, score, duration in scored_blocks if score >= 3.5)
    strong_ratio = strong_seconds / selected_seconds if selected_seconds > 0 else 0.0
    weak_blocks = [index for index, score, duration in scored_blocks if duration >= 8.0 and score < 2.5]
    if strong_ratio < 0.55:
        raise Exception(
            "OpenAI selected visual ranges are not grounded in important or watchable source content. "
            f"Only {strong_seconds:.1f}s of {selected_seconds:.1f}s selected narrated visuals overlap high-value timestamped frame evidence. "
            "Re-select the final edit from candidate_segments/frames with clear importance, interest_score, edit_value, and evidence_timestamps instead of padding with weak or random-looking ranges."
        )
    if len(weak_blocks) >= max(2, math.ceil(len(scored_blocks) * 0.35)):
        raise Exception(
            "OpenAI selected too many weak visual ranges for the final commentary edit. "
            f"Weak block indexes: {weak_blocks[:8]}. "
            "Choose the video's important, good-looking, story-progressing moments from timestamped visual evidence; do not fill the edit with arbitrary low-value footage."
        )


def _has_visual_plan(data: Dict) -> bool:
    return bool(data.get("narration_blocks") or data.get("chapters"))


def _validate_commentary_script_for_target(
    data: Dict,
    duration: float,
    target_duration: str,
    language: str,
    visual_analysis: Optional[Dict] = None,
    custom_style_prompt: Optional[str] = None,
) -> None:
    generated_sanitized = bool(data.get("_generated_sanitized"))
    if target_duration != "full":
        _validate_no_banned_narration_patterns(data)
    elif not generated_sanitized:
        _validate_no_editorial_meta_narration_patterns(data)
    _strip_camera_meta_phrasing(data)
    _validate_no_banned_commentary_phrases(data)
    _fill_missing_openai_evidence_timestamps(data, visual_analysis)
    _validate_scene_matched_narration_blocks(
        data,
        visual_analysis=visual_analysis,
        strict_scene_actions=bool(visual_analysis),
    )
    if target_duration != "full":
        blocks = _normalize_narration_blocks(data.get("narration_blocks") or [], duration)
        duration_error = _non_full_target_duration_validation_error(blocks, duration, target_duration)
        if duration_error:
            raise duration_error
        _sanitize_and_validate_no_visual_analysis_label_artifacts(data, language)
        _validate_custom_style_operation_logic(data, language, custom_style_prompt, duration)
        _validate_no_repeated_commentary_text(data)
        return
    target_seconds = _target_visual_duration_seconds_for_analysis(duration, target_duration, visual_analysis)
    blocks = _normalize_narration_blocks(data.get("narration_blocks") or [], duration)
    if not blocks:
        raise Exception(
            "AI narration_blocks are required for comprehensive full-mode commentary. "
            "OpenShorts needs timestamped narration blocks so each voiceover section can stay synced with the matching selected visual range."
        )

    def commit_blocks(updated_blocks: List[Dict]) -> None:
        _commit_narration_blocks_to_script(data, updated_blocks)

    blocks = _ensure_complete_commentary_ending_blocks(blocks, language)
    blocks = _normalize_full_mode_render_narration_blocks(blocks, language)
    data["narration_blocks"] = blocks
    data["edit_segments"] = _narration_blocks_to_edit_segments(blocks)
    _commit_narration_blocks_to_script(data, blocks)
    _sanitize_and_validate_no_visual_analysis_label_artifacts(data, language)
    blocks = data.get("narration_blocks") or blocks
    if any(isinstance(block, dict) and bool(block.get("_locked_edit_plan")) for block in blocks):
        _fit_locked_plan_blocks_to_render_sync(blocks, language)
        _commit_narration_blocks_to_script(data, blocks)
        _fit_locked_plan_narration_to_budget(
            data,
            _maximum_narration_chars_for_target_seconds(target_seconds, target_duration, language),
            language,
        )
        blocks = data.get("narration_blocks") or blocks
        data["edit_segments"] = _narration_blocks_to_edit_segments(blocks)
        _commit_narration_blocks_to_script(data, blocks)
        _sanitize_and_validate_no_visual_analysis_label_artifacts(data, language)
        blocks = data.get("narration_blocks") or blocks
    _validate_ai_video_speed_decisions(blocks, language, visual_analysis=visual_analysis, duration=duration)
    _validate_openai_selected_ranges_are_important(blocks, visual_analysis, duration, target_seconds)
    edit_segments = _narration_blocks_to_edit_segments(blocks)
    visual_seconds = sum(_block_visual_duration(block) for block in blocks)
    initial_selected_source_seconds = _segments_total_duration(edit_segments)
    if (
        visual_analysis is None
        and not _full_mode_preserves_source_process(duration, target_seconds)
        and duration > target_seconds * 1.6
        and visual_seconds >= _full_mode_min_playable_visual_seconds(duration, target_seconds) - _full_mode_visual_budget_tolerance_seconds(target_seconds)
    ):
        latest_end = max(float(block.get("end") or 0.0) for block in blocks)
        required_latest_end = duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION
        if latest_end < required_latest_end:
            raise Exception(
                "AI narration_blocks stopped before the end of the full source timeline. "
                f"Latest selected source timestamp is {latest_end:.1f}s, but this {duration:.1f}s source requires at least one selected block after {required_latest_end:.1f}s. "
                "The generated edit would ignore the later part of the video."
            )
    max_visual_seconds = _full_mode_max_playable_visual_seconds(duration, target_seconds)
    if (
        _full_mode_preserves_source_process(duration, target_seconds)
        and initial_selected_source_seconds < duration * 0.82
    ):
        raise Exception(
            "AI selected too little source footage for a complete-process full-mode edit. "
            f"Got {initial_selected_source_seconds:.1f}s selected from a {duration:.1f}s source; expected at least {duration * 0.82:.1f}s before optional video_speed. "
            "Preserve the complete visible workflow and use video_speed for slow or repetitive ranges instead of cutting meaningful process footage."
        )
    if (
        target_seconds > 0
        and not _full_mode_preserves_source_process(duration, target_seconds)
        and initial_selected_source_seconds > max(duration * 0.88, target_seconds * 1.45)
        and visual_seconds > max_visual_seconds
    ):
        raise Exception(
            "AI selected too much source footage for the full-mode edit target. "
            f"Got {initial_selected_source_seconds:.1f}s selected from a {duration:.1f}s source for a {target_seconds:.1f}s target. "
            "Cut repeated, waiting, setup, walking, camera drift, and redundant close-up ranges, and use video_speed for slow-but-useful process footage."
        )
    min_visual_seconds = _full_mode_min_playable_visual_seconds(duration, target_seconds)
    max_visual_seconds = _full_mode_max_playable_visual_seconds(duration, target_seconds)
    visual_budget_tolerance = _full_mode_visual_budget_tolerance_seconds(target_seconds)
    required_visual_seconds = min_visual_seconds
    allow_backend_visual_budget_repair = visual_analysis is None
    if (
        allow_backend_visual_budget_repair
        and target_seconds > 0
        and len(blocks) > 1
        and visual_seconds < required_visual_seconds - visual_budget_tolerance
    ):
        blocks = _repair_full_mode_underselected_visual_budget_with_pause_blocks(
            blocks,
            duration,
            target_seconds,
            language,
        )
        _commit_narration_blocks_to_script(data, blocks)
        edit_segments = _narration_blocks_to_edit_segments(blocks)
        visual_seconds = sum(_block_visual_duration(block) for block in blocks)
        initial_selected_source_seconds = _segments_total_duration(edit_segments)
        if visual_seconds < required_visual_seconds - visual_budget_tolerance:
            raise Exception(
                "AI narration_blocks do not match the selected full-mode edit target. "
                f"Got {visual_seconds:.1f}s of block-matched visuals for a {target_seconds:.1f}s target; expected between {min_visual_seconds:.1f}s and {max_visual_seconds:.1f}s. "
                "AI must select enough useful scene-matched source ranges from the visual evidence; OpenShorts could not safely add enough short original-audio bridge ranges."
            )
    max_visual_seconds = _full_mode_max_playable_visual_seconds(duration, target_seconds)
    if (
        target_seconds > 0
        and (
            len(blocks) <= 1
            or visual_seconds < required_visual_seconds - visual_budget_tolerance
            or visual_seconds > max_visual_seconds + visual_budget_tolerance
        )
    ):
        if len(blocks) > 1 and visual_seconds > max_visual_seconds + visual_budget_tolerance:
            blocks = _repair_full_mode_overselected_visual_budget(blocks, duration, target_seconds)
            _commit_narration_blocks_to_script(data, blocks)
            edit_segments = _narration_blocks_to_edit_segments(blocks)
            visual_seconds = sum(_block_visual_duration(block) for block in blocks)
            max_visual_seconds = _full_mode_max_playable_visual_seconds(duration, target_seconds)
        if (
            len(blocks) > 1
            and visual_seconds >= required_visual_seconds - visual_budget_tolerance
            and visual_seconds <= max_visual_seconds + visual_budget_tolerance
        ):
            pass
        else:
            raise Exception(
                "AI narration_blocks do not match the selected full-mode edit target. "
                f"Got {visual_seconds:.1f}s of block-matched visuals for a {target_seconds:.1f}s target; expected between {min_visual_seconds:.1f}s and {max_visual_seconds:.1f}s."
            )
    if (
        not _full_mode_preserves_source_process(duration, target_seconds)
        and visual_seconds > max_visual_seconds
        and not _segments_have_real_cuts(edit_segments, duration, target_seconds)
    ):
        selected_source_seconds = _segments_total_duration(edit_segments)
        raise Exception(
            "AI returned a near-full-source timeline instead of an edited full-mode cut strategy. "
            f"Got {selected_source_seconds:.1f}s selected from a {duration:.1f}s source for a {target_seconds:.1f}s target."
        )
    pause_repair_limit = target_seconds if _full_mode_preserves_source_process(duration, target_seconds) else FULL_MODE_MAX_PAUSE_SECONDS
    blocks = _repair_full_mode_pause_blocks(blocks, min_visual_seconds, pause_repair_limit)
    _commit_narration_blocks_to_script(data, blocks)
    edit_segments = _narration_blocks_to_edit_segments(blocks)
    visual_seconds = sum(_block_visual_duration(block) for block in blocks)
    if (
        allow_backend_visual_budget_repair
        and target_seconds > 0
        and len(blocks) > 1
        and visual_seconds < required_visual_seconds - visual_budget_tolerance
    ):
        blocks = _repair_full_mode_underselected_visual_budget_with_pause_blocks(
            blocks,
            duration,
            target_seconds,
            language,
        )
        _commit_narration_blocks_to_script(data, blocks)
        edit_segments = _narration_blocks_to_edit_segments(blocks)
        visual_seconds = sum(_block_visual_duration(block) for block in blocks)
    pause_seconds = 0.0
    longest_pause = 0.0
    consecutive_pauses = 0
    max_consecutive_pauses = 0
    spoken_blocks = 0
    allowed_pause_seconds = pause_repair_limit
    for index, block in enumerate(blocks, start=1):
        block_duration = _block_visual_duration(block)
        if bool(block.get("pause")):
            pause_seconds += block_duration
            longest_pause = max(longest_pause, block_duration)
            consecutive_pauses += 1
            max_consecutive_pauses = max(max_consecutive_pauses, consecutive_pauses)
        else:
            spoken_blocks += 1
            consecutive_pauses = 0
    pause_ratio = pause_seconds / visual_seconds if visual_seconds > 0 else 0.0
    if spoken_blocks <= 0:
        raise Exception("AI returned only no-commentary footage. Full-mode commentary needs narrated blocks between any pause blocks.")
    if pause_ratio > FULL_MODE_MAX_PAUSE_RATIO + FULL_MODE_VALIDATION_EPSILON_RATIO:
        raise Exception(
            "AI returned too much no-commentary footage. "
            f"Pause blocks cover {pause_seconds:.1f}s of {visual_seconds:.1f}s selected visuals; "
            f"allowed at most {FULL_MODE_MAX_PAUSE_RATIO * 100:.0f}%. "
            "Shorten pause=true ranges, convert useful footage into narrated scene-matched blocks, or cut low-value footage."
        )
    if longest_pause > allowed_pause_seconds + FULL_MODE_VALIDATION_EPSILON_SECONDS:
        raise Exception(
            "AI returned an overlong no-commentary pause block. "
            f"Longest pause is {longest_pause:.1f}s and allowed at most {allowed_pause_seconds:.1f}s. "
            "Split it, narrate the visible action, or cut the low-value portion."
        )
    if max_consecutive_pauses > FULL_MODE_MAX_CONSECUTIVE_PAUSE_BLOCKS:
        raise Exception(
            "AI returned too many consecutive no-commentary pause blocks; "
            f"allowed at most {FULL_MODE_MAX_CONSECUTIVE_PAUSE_BLOCKS} in a row."
        )
    edit_segments = _narration_blocks_to_edit_segments(blocks)
    visual_seconds = sum(_block_visual_duration(block) for block in blocks)
    min_visual_seconds = _full_mode_min_playable_visual_seconds(duration, target_seconds)
    max_visual_seconds = _full_mode_max_playable_visual_seconds(duration, target_seconds)
    visual_budget_tolerance = _full_mode_visual_budget_tolerance_seconds(target_seconds)
    if target_seconds > 0 and len(blocks) > 1:
        repaired_short_blocks = []
        for index, block in enumerate(blocks, start=1):
            shortfall = _narration_density_shortfall_details(block, index, language)
            if shortfall and _block_visual_duration(block) <= 24.0 + FULL_MODE_VALIDATION_EPSILON_SECONDS:
                repaired_short_blocks.extend(_repair_short_narration_visual_ranges([block], language))
            else:
                repaired_short_blocks.append(block)
        if len(repaired_short_blocks) != len(blocks) or any(
            abs(float((repaired_short_blocks[i] if i < len(repaired_short_blocks) else {}).get("end") or 0.0) - float((blocks[i] if i < len(blocks) else {}).get("end") or 0.0)) > 0.001
            or str((repaired_short_blocks[i] if i < len(repaired_short_blocks) else {}).get("narration") or "") != str((blocks[i] if i < len(blocks) else {}).get("narration") or "")
            for i in range(min(len(repaired_short_blocks), len(blocks)))
        ):
            blocks = repaired_short_blocks
            _commit_narration_blocks_to_script(data, blocks)
            edit_segments = _narration_blocks_to_edit_segments(blocks)
            visual_seconds = sum(_block_visual_duration(block) for block in blocks)
            if (
                allow_backend_visual_budget_repair
                and target_seconds > 0
                and len(blocks) > 1
                and visual_seconds < required_visual_seconds - visual_budget_tolerance
            ):
                blocks = _repair_full_mode_underselected_visual_budget_with_pause_blocks(
                    blocks,
                    duration,
                    target_seconds,
                    language,
                )
                _commit_narration_blocks_to_script(data, blocks)
                edit_segments = _narration_blocks_to_edit_segments(blocks)
                visual_seconds = sum(_block_visual_duration(block) for block in blocks)
    if target_seconds > 0 and len(blocks) > 1 and visual_seconds < required_visual_seconds - visual_budget_tolerance:
        raise Exception(
            "AI narration_blocks do not match the selected full-mode edit target. "
            f"Got {visual_seconds:.1f}s of block-matched visuals for a {target_seconds:.1f}s target; expected between {min_visual_seconds:.1f}s and {max_visual_seconds:.1f}s. "
            "AI must select enough useful scene-matched source ranges from the visual evidence; OpenShorts will not invent filler ranges or evenly sample the timeline."
        )
    blocks = _repair_full_mode_small_density_shortfalls(blocks, min_visual_seconds, language)
    if any(isinstance(block, dict) and bool(block.get("_locked_edit_plan")) for block in blocks):
        _fit_locked_plan_blocks_to_render_sync(blocks, language)
    if any(isinstance(block, dict) and bool(block.get("_locked_edit_plan")) for block in blocks):
        data["narration_blocks"] = blocks
        _fit_locked_plan_narration_to_budget(
            data,
            _maximum_narration_chars_for_target_seconds(target_seconds, target_duration, language),
            language,
        )
        blocks = data.get("narration_blocks") or blocks
    _commit_narration_blocks_to_script(data, blocks)
    edit_segments = _narration_blocks_to_edit_segments(blocks)
    visual_seconds = sum(_block_visual_duration(block) for block in blocks)
    _sanitize_and_validate_no_visual_analysis_label_artifacts(data, language)
    blocks = data.get("narration_blocks") or blocks
    for index, block in enumerate(blocks, start=1):
        _validate_narration_density_matches_visual_duration(block, index, language)

    _validate_ai_video_speed_decisions(blocks, language, visual_analysis=visual_analysis, duration=duration)
    for index, block in enumerate(blocks, start=1):
        if bool(block.get("pause")):
            continue
        block_duration = _block_visual_duration(block)
        visual_text = str(block.get("visual") or "").strip()
        visual_facts = block.get("visual_facts") if isinstance(block.get("visual_facts"), list) else []
        concrete_fact_count = len([fact for fact in visual_facts if len(str(fact).strip()) >= 4])
        if concrete_fact_count < 1 and len(visual_text) < 4:
            raise Exception(
                "AI narration block is missing source visual evidence. "
                f"Block {index} has no concrete visual or visual_facts. "
                "Describe the visible action, tools, materials, and result for this exact timestamp range; OpenShorts will not fill a placeholder visual description."
            )
        if block_duration >= 20.0 and concrete_fact_count < 1 and len(visual_text) < 4:
            raise Exception(
                "AI narration block is not grounded enough in the source visuals. "
                f"Block {index} covers {block_duration:.1f}s but lacks specific visual_facts or a concrete visual description. "
                "Add scene-matched visible facts from this exact timestamp range so the commentary explains what the viewer is seeing."
            )
    if visual_analysis is None and not _full_mode_preserves_source_process(duration, target_seconds) and duration > target_seconds * 1.6:
        latest_end = max(float(block.get("end") or 0.0) for block in blocks)
        required_latest_end = duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION
        if latest_end < required_latest_end:
            raise Exception(
                "AI narration_blocks stopped before the end of the full source timeline. "
                f"Latest selected source timestamp is {latest_end:.1f}s, but this {duration:.1f}s source requires at least one selected block after {required_latest_end:.1f}s. "
                "The generated edit would ignore the later part of the video."
            )
    commit_blocks(blocks)
    narration_source = _narration_from_blocks({"narration_blocks": blocks}) or str(data.get("narration") or "")
    narration = re.sub(r"\s+", "", narration_source)
    max_chars = _maximum_narration_chars_for_target_seconds(target_seconds, target_duration, language)
    if not narration:
        raise Exception(
            "AI returned no spoken narration for full-mode commentary. "
            "Use pause=true only when the timestamped visual or original sound genuinely explains itself better without speech, and keep ordinary process footage in narrated blocks."
        )
    if max_chars and len(narration) > max_chars:
        if any(isinstance(block, dict) and bool(block.get("_locked_edit_plan")) for block in blocks):
            _fit_locked_plan_narration_to_budget(data, max_chars, language)
            blocks = data.get("narration_blocks") or blocks
            _sanitize_and_validate_no_visual_analysis_label_artifacts(data, language)
            blocks = data.get("narration_blocks") or blocks
            narration = re.sub(r"\s+", "", _narration_from_blocks({"narration_blocks": blocks}))
            if len(narration) <= max_chars:
                for index, block in enumerate(blocks, start=1):
                    _validate_narration_density_matches_visual_duration(block, index, language)
                _validate_ai_video_speed_decisions(blocks, language, visual_analysis=visual_analysis, duration=duration)
                _validate_custom_style_operation_logic(data, language, custom_style_prompt, duration)
                _validate_no_repeated_commentary_text(data)
                return
        raise Exception(
            "AI narration is too long for comprehensive full-mode commentary. "
            f"Got {len(narration)} chars; expected at most {max_chars}. "
            "The generated voiceover would run much longer than the selected visuals and can overload local rendering, so OpenShorts rejected it."
        )
    _validate_custom_style_operation_logic(data, language, custom_style_prompt, duration)
    _validate_no_repeated_commentary_text(data)


def _is_rendered_cached_full_mode_script(data: Dict) -> bool:
    blocks = data.get("narration_blocks") if isinstance(data, dict) else None
    if not isinstance(blocks, list) or not blocks:
        return False
    rendered_count = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        try:
            rendered_duration = float(block.get("rendered_duration") or 0.0)
        except (TypeError, ValueError):
            rendered_duration = 0.0
        if rendered_duration > 0:
            rendered_count += 1
    return rendered_count == len(blocks)


def _validate_rendered_cached_full_mode_script(
    data: Dict,
    duration: float,
    target_duration: str,
    language: str,
    custom_style_prompt: Optional[str] = None,
) -> None:
    if target_duration != "full" or not _is_rendered_cached_full_mode_script(data):
        _validate_commentary_script_for_target(
            data,
            duration,
            target_duration,
            language,
            custom_style_prompt=custom_style_prompt,
        )
        return
    _strip_camera_meta_phrasing(data)
    _validate_no_banned_commentary_phrases(data)
    blocks = _normalize_narration_blocks(data.get("narration_blocks") or [], duration)
    if not blocks:
        raise Exception("Cached rendered commentary script has no narration_blocks.")
    spoken_blocks = [block for block in blocks if not bool(block.get("pause")) and str(block.get("narration") or "").strip()]
    if not spoken_blocks:
        raise Exception("Cached rendered commentary script has no spoken narration blocks.")
    for index, block in enumerate(blocks, start=1):
        block_duration = _block_visual_duration(block)
        if block_duration <= 0:
            raise Exception(f"Cached rendered commentary block {index} has invalid duration.")
        if bool(block.get("pause")):
            continue
        visual_text = str(block.get("visual") or "").strip()
        if len(visual_text) < 4:
            raise Exception(f"Cached rendered commentary block {index} is missing visual context.")
    data["narration_blocks"] = blocks
    data["edit_segments"] = _narration_blocks_to_edit_segments(blocks)
    _commit_narration_blocks_to_script(data, blocks)
    _validate_no_banned_narration_patterns(data)
    _validate_custom_style_operation_logic(data, language, custom_style_prompt, duration)
    _validate_no_repeated_commentary_text(data)


def _validate_voiceover_duration_for_target(
    voiceover_path: str,
    edit_segments: List[Dict],
    duration: float,
    target_duration: str,
) -> None:
    if target_duration != "full":
        return
    visual_seconds = _segments_total_duration(edit_segments)
    if visual_seconds <= 0:
        raise Exception(
            "Generated voiceover cannot be validated without AI-selected edit_segments. "
            "The model must choose kept visual ranges before full-mode rendering."
        )
    audio_seconds = _get_audio_duration(voiceover_path)
    max_seconds = max(visual_seconds + 60.0, visual_seconds * FULL_MODE_MAX_VOICEOVER_DURATION_RATIO)
    if audio_seconds > max_seconds:
        raise Exception(
            "Generated voiceover is too long for comprehensive full-mode commentary. "
            f"Got {audio_seconds:.1f}s audio for {visual_seconds:.1f}s selected visuals; allowed at most {max_seconds:.1f}s. "
            "OpenShorts stopped before visual rendering to avoid an oversized local FFmpeg job."
        )


def _visual_budget_error_match(error_text: str) -> Optional[re.Match]:
    return re.search(
        r"Got\s+([0-9.]+)s\s+of block-matched visuals for a\s+([0-9.]+)s\s+target;\s+expected between\s+([0-9.]+)s\s+and\s+([0-9.]+)s",
        str(error_text or ""),
    )


def _target_seconds_from_validation_error(validation_error: Optional[Exception]) -> Optional[float]:
    if not validation_error:
        return None
    budget_match = _visual_budget_error_match(str(validation_error))
    if not budget_match:
        return None
    try:
        return float(budget_match.group(2))
    except (TypeError, ValueError):
        return None


def _validation_error_is_visual_budget(validation_error: Optional[Exception]) -> bool:
    return bool(_visual_budget_error_match(str(validation_error or "")))


def _non_full_target_duration_validation_error(
    blocks: List[Dict],
    duration: float,
    target_duration: str,
) -> Optional[Exception]:
    if not _is_non_full_target_duration(target_duration) or not blocks:
        return None
    min_seconds, max_seconds = _target_duration_window_seconds(duration, target_duration)
    if max_seconds <= 0:
        return None
    playable_seconds = sum(_block_visual_duration(block) for block in blocks)
    tolerance = max(0.0, NON_FULL_TARGET_DURATION_TOLERANCE_SECONDS)
    if playable_seconds > max_seconds + tolerance:
        label = _non_full_target_duration_label(target_duration)
        return Exception(
            f"AI narration_blocks do not match the requested {label} target duration. "
            f"Got {playable_seconds:.1f}s playable visuals; expected no more than {max_seconds:.1f}s. "
            "Select fewer stronger source ranges, apply justified video_speed where useful, and keep narration scene-matched to the selected ranges."
        )
    return None


def _repair_scope_instruction(validation_error: Optional[Exception], attempt_label: str) -> str:
    error_text = str(validation_error or "")
    if "Custom style operation logic validation failed" in error_text:
        return (
            f"This is {attempt_label}. The validation error is a custom-style narration-depth failure, "
            "so keep the selected timeline and visual evidence unless sync is invalid. "
            "Rewrite action-only narration into scene-matched commentary that explains visible action plus same-range purpose, operation logic, or result."
        )
    if re.search(r"requested (?:60-90 second|2-4 minute|3-5 minute) target duration", error_text):
        return (
            f"This is {attempt_label}. The validation error is a global target-duration failure, "
            "so repartition the complete narration_blocks list instead of patching one block. "
            "Keep only the strongest useful source ranges, preserve the source order, and keep the final playable total under the requested duration cap."
        )
    if _validation_error_is_visual_budget(validation_error):
        return (
            f"This is {attempt_label}. The validation error is a global visual-budget/timeline failure, "
            "so do not merely patch one block and do not preserve a block just because it was locally valid. "
            "Repartition the complete narration_blocks list as needed so the final playable total lands inside the target window and every kept range remains scene-matched and TTS-synced."
        )
    if re.search(r"missing narration|raw visual-analysis labels|label-like text", error_text, flags=re.IGNORECASE):
        return (
            f"This is {attempt_label}. The validation error is a narration-writing failure, "
            "so keep the locked timeline, evidence timestamps, visual facts, and video_speed unchanged. "
            "Rewrite the missing or polluted narration_blocks.narration fields as natural commentary in the selected style."
        )
    if re.search(
        r"selected too much source footage|selected too little source footage|near-full-source timeline",
        error_text,
        flags=re.IGNORECASE,
    ):
        return (
            f"This is {attempt_label}. The validation error is a global edit-decision failure, "
            "so re-read the visual evidence and repartition the complete narration_blocks list as needed. "
            "Keep only useful source ranges, preserve the process arc, and keep every narration block matched to its exact visual range."
        )
    return f"This is {attempt_label}. Use the focused repair instructions above when present; otherwise write a fresh full commentary script."


def _script_narration_blocks(script: Dict) -> List[Dict]:
    blocks = script.get("narration_blocks") if isinstance(script, dict) else None
    if not blocks and isinstance(script, dict) and isinstance(script.get("script"), dict):
        blocks = script["script"].get("narration_blocks")
    return [block for block in (blocks or []) if isinstance(block, dict)]


def _visual_budget_validation_failure_details(validation_error: Optional[Exception], invalid_script: Dict) -> Optional[Dict]:
    if not validation_error:
        return None
    budget_match = _visual_budget_error_match(str(validation_error))
    if not budget_match:
        return None
    try:
        actual_seconds = float(budget_match.group(1))
        target_seconds = float(budget_match.group(2))
        min_seconds = float(budget_match.group(3))
        max_seconds = float(budget_match.group(4))
    except (TypeError, ValueError):
        return None
    source_seconds = 0.0
    playable_seconds = 0.0
    speed_saved_seconds = 0.0
    accelerated_count = 0
    latest_end = 0.0
    for block in _script_narration_blocks(invalid_script):
        source_duration = _block_source_duration(block)
        speed = _safe_video_speed(block.get("video_speed"))
        playable_duration = source_duration / speed if speed > 0 else source_duration
        source_seconds += source_duration
        playable_seconds += playable_duration
        latest_end = max(latest_end, float(block.get("end") or 0.0))
        if speed > 1.0001:
            accelerated_count += 1
            speed_saved_seconds += max(0.0, source_duration - playable_duration)
    return {
        "actual_seconds": actual_seconds,
        "target_seconds": target_seconds,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "source_seconds": source_seconds,
        "playable_seconds": playable_seconds,
        "speed_saved_seconds": speed_saved_seconds,
        "accelerated_count": accelerated_count,
        "latest_end": latest_end,
    }


def _density_validation_failure_details(validation_error: Optional[Exception], invalid_script: Dict) -> Optional[Dict]:
    if not validation_error:
        return None
    density_match = re.search(
        r"Block\s+(\d+)\s+has\s+(\d+)\s+chars\s+for\s+([0-9.]+)s\s+of playable visuals;\s+expected at least\s+(\d+)",
        str(validation_error),
    )
    if not density_match:
        return None
    block_index = int(density_match.group(1))
    blocks = _script_narration_blocks(invalid_script)
    if not (1 <= block_index <= len(blocks)) or not isinstance(blocks[block_index - 1], dict):
        return None
    return {
        "block_index": block_index,
        "array_index": block_index - 1,
        "current_chars": int(density_match.group(2)),
        "playable_seconds": float(density_match.group(3)),
        "expected_chars": int(density_match.group(4)),
        "block": blocks[block_index - 1],
    }


def _all_density_shortfall_details(invalid_script: Dict, language: str) -> List[Dict]:
    details = []
    for index, block in enumerate(_script_narration_blocks(invalid_script), start=1):
        if not isinstance(block, dict):
            continue
        shortfall = _narration_density_shortfall_details(block, index, language)
        if shortfall:
            details.append(shortfall)
    return details


def _format_density_repair_targets(details: List[Dict], primary_block_index: int, language: str) -> str:
    if not details:
        return ""
    ordered = sorted(
        details,
        key=lambda item: (
            0 if int(item.get("block_index") or 0) == primary_block_index else 1,
            int(item.get("block_index") or 0),
        ),
    )
    lines = []
    for item in ordered[:8]:
        block = item.get("block") if isinstance(item.get("block"), dict) else {}
        visual = _limit_text_chars(re.sub(r"\s+", " ", str(block.get("visual") or "")).strip(), 180)
        narration = _limit_text_chars(re.sub(r"\s+", " ", str(block.get("narration") or "")).strip(), 220)
        current_chars = int(item.get("current_chars") or 0)
        expected_chars = int(item.get("expected_chars") or 0)
        playable_seconds = float(item.get("playable_seconds") or 0.0)
        estimated_voice_seconds = float(
            item.get("estimated_voice_seconds")
            if item.get("estimated_voice_seconds") is not None
            else _estimated_voiceover_seconds_for_chars(current_chars, language)
        )
        trailing_silence_seconds = max(0.0, playable_seconds - estimated_voice_seconds)
        speed = _safe_video_speed(block.get("video_speed"))
        max_playable_for_current_text = _max_narrated_visual_seconds_for_chars(current_chars, language)
        max_source_for_current_text = max_playable_for_current_text * speed
        missing_chars = max(0, expected_chars - current_chars)
        lines.append(
            "- Block {block_index} / narration_blocks[{array_index}]: {current_chars} chars for "
            "{playable_seconds:.1f}s playable visuals, expected at least {expected_chars}. "
            "Estimated TTS covers about {estimated_voice:.1f}s, leaving about {tail:.1f}s without matched narration. "
            "Required action: add at least {missing_chars} concrete scene-matched chars "
            "(to >= {expected_chars} total), or reduce/split playable visuals to <= {max_playable:.1f}s "
            "for the current narration (about <= {max_source:.1f}s source at {speed:g}x), or use only a brief visually justified pause=true piece. "
            "Do not return this block with the same short narration over the same long range. "
            "Range {start:.1f}-{end:.1f}s, video_speed {speed:g}. visual={visual!r}; narration={narration!r}".format(
                block_index=int(item.get("block_index") or 0),
                array_index=int(item.get("array_index") or 0),
                current_chars=current_chars,
                playable_seconds=playable_seconds,
                expected_chars=expected_chars,
                estimated_voice=estimated_voice_seconds,
                tail=trailing_silence_seconds,
                missing_chars=missing_chars,
                max_playable=max_playable_for_current_text,
                max_source=max_source_for_current_text,
                start=float(block.get("start") or 0.0),
                end=float(block.get("end") or 0.0),
                speed=speed,
                visual=visual,
                narration=narration,
            )
        )
    remaining = len(ordered) - len(lines)
    if remaining > 0:
        lines.append(f"- Plus {remaining} more density-risk block(s); audit every non-pause block before returning JSON.")
    return "\n".join(lines)


def _format_budget_repair_block_timeline(invalid_script: Dict, language: str, limit: int = 24) -> str:
    blocks = _script_narration_blocks(invalid_script)
    if not blocks:
        return "- No valid narration_blocks were present in the previous JSON."
    lines = []
    for index, block in enumerate(blocks[:limit], start=1):
        source_seconds = _block_source_duration(block)
        speed = _safe_video_speed(block.get("video_speed"))
        playable_seconds = _block_visual_duration(block)
        is_pause = bool(block.get("pause"))
        narration_text = str(block.get("narration") or block.get("text") or "").strip()
        narration_chars = len(re.sub(r"\s+", "", narration_text))
        visual = _limit_text_chars(re.sub(r"\s+", " ", str(block.get("visual") or "")).strip(), 120)
        narration = _limit_text_chars(re.sub(r"\s+", " ", narration_text).strip(), 120)
        if is_pause:
            if playable_seconds > FULL_MODE_MAX_PAUSE_SECONDS + FULL_MODE_VALIDATION_EPSILON_SECONDS:
                sync_state = f"pause too long; split/narrate/cut to <= {FULL_MODE_MAX_PAUSE_SECONDS:.1f}s"
            else:
                sync_state = "pause ok if visually justified"
        else:
            estimated_voice_seconds = _estimated_voiceover_seconds_for_chars(narration_chars, language)
            minimum_voice_seconds = _minimum_voiceover_seconds_for_visual_duration(playable_seconds)
            max_playable = _max_narrated_visual_seconds_for_chars(narration_chars, language)
            if playable_seconds >= 12.0 and (
                playable_seconds > max_playable + FULL_MODE_VALIDATION_EPSILON_SECONDS
                or estimated_voice_seconds + FULL_MODE_VALIDATION_EPSILON_SECONDS < minimum_voice_seconds
            ):
                expected_chars = _expected_narration_chars_for_visual_duration(playable_seconds, language)
                sync_state = (
                    f"sync risk; needs >= {expected_chars} chars or <= {max_playable:.1f}s playable "
                    f"for current narration"
                )
            else:
                sync_state = (
                    f"sync ok; est voice {estimated_voice_seconds:.1f}s, "
                    f"tail {max(0.0, playable_seconds - estimated_voice_seconds):.1f}s"
                )
        lines.append(
            "- Block {index}: {start:.1f}-{end:.1f}s source, {source:.1f}s source / {playable:.1f}s playable "
            "at {speed:g}x, pause={pause}, chars={chars}, {sync_state}. visual={visual!r}; narration={narration!r}".format(
                index=index,
                start=float(block.get("start") or 0.0),
                end=float(block.get("end") or 0.0),
                source=source_seconds,
                playable=playable_seconds,
                speed=speed,
                pause=str(is_pause).lower(),
                chars=narration_chars,
                sync_state=sync_state,
                visual=visual,
                narration=narration,
            )
        )
    remaining = len(blocks) - len(lines)
    if remaining > 0:
        lines.append(f"- Plus {remaining} more block(s); audit every block before returning JSON.")
    return "\n".join(lines)


def _full_mode_budget_summary_for_script(
    invalid_script: Dict,
    duration: Optional[float],
    target_duration: str,
    validation_error: Optional[Exception] = None,
    target_seconds_override: Optional[float] = None,
) -> Optional[Dict]:
    if target_duration != "full":
        return None
    blocks = _script_narration_blocks(invalid_script)
    if not blocks:
        return None
    try:
        source_duration = float(duration or 0.0)
    except (TypeError, ValueError):
        source_duration = 0.0
    target_seconds = float(
        target_seconds_override
        or _target_seconds_from_validation_error(validation_error)
        or _target_visual_duration_seconds(source_duration, target_duration)
    )
    if target_seconds <= 0:
        return None
    playable_seconds = sum(_block_visual_duration(block) for block in blocks)
    selected_source_seconds = sum(_block_source_duration(block) for block in blocks)
    min_seconds = _full_mode_min_playable_visual_seconds(source_duration, target_seconds)
    max_seconds = _full_mode_max_playable_visual_seconds(source_duration, target_seconds)
    latest_end = max((float(block.get("end") or 0.0) for block in blocks), default=0.0)
    return {
        "target_seconds": target_seconds,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
        "playable_seconds": playable_seconds,
        "selected_source_seconds": selected_source_seconds,
        "latest_end": latest_end,
    }


def _focused_validation_repair_instruction(
    validation_error: Optional[Exception],
    invalid_script: Dict,
    _language: str,
    block_count: int,
    duration: Optional[float] = None,
    target_duration: str = "full",
    target_seconds: Optional[float] = None,
) -> str:
    if not validation_error:
        return ""
    error_text = str(validation_error)
    if "Custom style operation logic validation failed" in error_text:
        details = _custom_style_operation_logic_failure_details(
            invalid_script,
            _language,
            "解释操作逻辑 为什么 目的 工业机械",
            duration,
        )
        action_only_blocks = []
        action_only_indexes = []
        if details:
            action_only_indexes = details.get("action_only_indexes") or []
            action_only_set = set(action_only_indexes)
        else:
            action_only_set = set()
        for index, block in enumerate(_script_narration_blocks(invalid_script), start=1):
            if action_only_set and index not in action_only_set:
                continue
            if not action_only_set and (bool(block.get("pause")) or _narration_has_operation_logic(str(block.get("narration") or ""), _language)):
                continue
            action_only_blocks.append({
                "block_index": index,
                "block": block,
            })
            if len(action_only_blocks) >= 12:
                break
        return f"""
FOCUSED REPAIR REQUIRED:
- The previous JSON did not actually follow the custom commentary style's process/operation-logic requirement. It mostly described visible actions, but too few narrated process blocks explained why the operation is done, what purpose it serves, or what same-range result it creates.
- Keep exactly {block_count} narration_blocks. Preserve start, end, video_speed, speed_reason, visual_facts, and evidence_timestamps unless a local boundary is already invalid for sync.
- Rewrite action-only narration fields so each ordinary process block has short, natural commentary with this structure: visible action/tool/material first, then one concise purpose, operation-logic, or result sentence for the same timestamp range.
- Stay grounded in that block's visual_facts, evidence_timestamps, visual description, and transcript evidence. Do not invent unseen machine internals, quantities, hazards, worker intent, or later results.
- For Chinese industrial/mechanical custom styles, use compact process language such as "这一步是为了...", "这样能...", "主要是把...", "方便后面...", "防止...", or an equivalent natural phrase where supported by the visuals.
- Do not add generic filler, slogans, or raw visual-analysis labels. The narration should still be speakable inside each block's visual duration.
ACTION-ONLY BLOCK EVIDENCE:
{json.dumps(action_only_blocks, ensure_ascii=False)}
""".strip()
    if "missing narration" in error_text:
        missing_indexes = []
        match = re.search(r"Missing block indexes:\s*([0-9,\s]+)", error_text)
        if match:
            missing_indexes = [int(value) for value in re.findall(r"\d+", match.group(1))]
        repair_targets = []
        blocks = _script_narration_blocks(invalid_script)
        for block_index in missing_indexes[:12]:
            if 1 <= block_index <= len(blocks):
                repair_targets.append({
                    "block_index": block_index,
                    "block": blocks[block_index - 1],
                })
        return f"""
FOCUSED REPAIR REQUIRED:
- The previous JSON left selected locked plan blocks without spoken narration. Missing block indexes: {missing_indexes or "see validation error"}.
- Keep exactly {block_count} narration_blocks and keep every locked block's start, end, video_speed, speed_reason, visual_facts, and evidence_timestamps unchanged.
- Rewrite only the empty narration fields, rate/pitch if useful, and top-level narration/title/summary metadata. Do not add, remove, merge, split, retime, or invent visual ranges.
- For each missing block, write fluent natural commentary in the selected style, grounded only in that block's visual_facts, evidence_timestamps, and transcript evidence. Do not use backend placeholder text.
- Do not copy slash-separated visual analysis labels into narration; convert evidence into normal spoken Chinese.
MISSING BLOCK EVIDENCE:
{json.dumps(repair_targets, ensure_ascii=False)}
""".strip()
    if "raw visual-analysis labels" in error_text or "label-like text" in error_text:
        polluted_blocks = []
        for index, block in enumerate(_script_narration_blocks(invalid_script), start=1):
            if isinstance(block, dict) and _contains_visual_analysis_label_artifact(str(block.get("narration") or ""), _language):
                polluted_blocks.append({
                    "block_index": index,
                    "block": block,
                })
        return f"""
FOCUSED REPAIR REQUIRED:
- The previous JSON copied raw visual-analysis labels into spoken narration. That text is not final commentary.
- Keep exactly {block_count} narration_blocks and keep every locked block's start, end, video_speed, speed_reason, visual_facts, and evidence_timestamps unchanged.
- Rewrite the polluted narration fields as fluent natural commentary in the selected style, grounded only in that block's timestamped visual evidence and transcript evidence.
- Do not include source timestamps, frame labels like "112.324s: Men cutting...", compact labels like "324s:Mencuttingandha", English category labels, slash-separated strings, edit_value labels, pace labels, or raw frame-analysis phrases in narration.
POLLUTED BLOCK EVIDENCE:
{json.dumps(polluted_blocks[:12], ensure_ascii=False)}
""".strip()
    density_details = _density_validation_failure_details(validation_error, invalid_script)
    if density_details:
        block_index = density_details["block_index"]
        current_chars = density_details["current_chars"]
        playable_seconds = density_details["playable_seconds"]
        expected_chars = density_details["expected_chars"]
        block = density_details["block"]
        all_density_shortfalls = _all_density_shortfall_details(invalid_script, _language)
        if not any(item.get("block_index") == block_index for item in all_density_shortfalls):
            all_density_shortfalls.insert(0, density_details)
        density_targets = _format_density_repair_targets(all_density_shortfalls, block_index, _language)
        density_min_chars = _minimum_scene_matched_narration_chars(_language)
        budget_summary = _full_mode_budget_summary_for_script(
            invalid_script,
            duration,
            target_duration,
            validation_error=validation_error,
            target_seconds_override=target_seconds,
        )
        budget_instruction = ""
        if budget_summary:
            playable_total = budget_summary["playable_seconds"]
            min_seconds = budget_summary["min_seconds"]
            max_seconds = budget_summary["max_seconds"]
            target_seconds = budget_summary["target_seconds"]
            if playable_total < min_seconds:
                extra_needed = max(0.0, min_seconds - playable_total)
                budget_instruction = (
                    f"\n- Full-edit timing is also under budget: current total playable visuals are {playable_total:.1f}s "
                    f"for a {target_seconds:.1f}s target; allowed window is {min_seconds:.1f}-{max_seconds:.1f}s. "
                    f"Recover at least {extra_needed:.1f}s of useful playable visual time while fixing Block {block_index}."
                )
            else:
                budget_instruction = (
                    f"\n- Full-edit timing must remain inside the target window while fixing this block: current total playable visuals are {playable_total:.1f}s "
                    f"for a {target_seconds:.1f}s target; allowed window is {min_seconds:.1f}-{max_seconds:.1f}s. "
                    "Do not shorten this block or adjacent ranges in a way that drops the total below the allowed minimum."
                )
        return f"""
FOCUSED REPAIR REQUIRED:
- The previous JSON failed specifically at narration_blocks[{block_index - 1}] / Block {block_index}.
- Historical validator note: that block had {current_chars} non-whitespace characters for {playable_seconds:.1f}s of playable visuals, previously expecting at least {expected_chars}.
- Failing block from previous JSON: {json.dumps(block, ensure_ascii=False)}
- Current density-risk blocks in this JSON, including the primary failed block:
{density_targets}
- Sync math before returning JSON: playable_seconds = (end - start) / video_speed; ordinary narrated blocks need concrete narration that both describes the exact visual range and has enough estimated TTS duration to cover it. For this language, start from expected_chars >= ceil(playable_seconds / 24 * {density_min_chars}), then also make sure estimated voiceover covers at least {FULL_MODE_MIN_NARRATED_BLOCK_VOICEOVER_RATIO:.0%} of playable_seconds and leaves no more than {FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS:.0f}s of trailing visual silence. If you change video_speed, recompute playable_seconds and rewrite/split narration for the accelerated visual duration.
- Fix all listed density-risk blocks in one pass, not only Block {block_index}. Every listed block must pass the required action above before you return JSON. Before returning JSON, audit every non-pause block so another block does not fail the same too-short-for-range validation on the next attempt.
- Do not pad any block with repeated generic filler just to add characters. Fix timing and scene match: shorten the source range, split/repartition adjacent blocks, add concrete narration that names visible actions/tools/materials/results in that exact range, or use a brief pause=true block where the original visual should breathe.
- If you shorten any listed block to fix narration density, recover the removed playable time by extending or selecting other useful, scene-matched source ranges from the visual evidence, or by repartitioning adjacent blocks. Do not let a local density fix break the full-edit visual target.
- Keep unrelated valid narration_blocks unchanged unless changing only the listed blocks would push the full-edit playable duration outside its target window. In that case, adjust nearby boundaries or replace weak ranges with stronger useful ranges from the visual evidence.
- Preserve exactly {block_count} total narration_blocks.
{budget_instruction}
""".strip()
    if "not supported by its selected visual range" in error_text or "concrete claim" in error_text:
        return f"""
FOCUSED REPAIR REQUIRED:
- The previous JSON put a concrete narration claim on a block whose selected visual range did not show that claim.
- Move that claim to a timestamp where the same action/state/result is actually visible, or rewrite the failed block to describe only the visible action in that exact range.
- Do not leave unrelated aftermath or earlier/later footage after a claimed result unless that footage has its own accurately matched narration or is removed from the selected timeline.
- Preserve exactly {block_count} total narration_blocks by adjusting only the failed block or its adjacent boundaries when possible.
""".strip()
    visual_budget_details = _visual_budget_validation_failure_details(validation_error, invalid_script)
    if visual_budget_details:
        actual_seconds = visual_budget_details["actual_seconds"]
        target_seconds = visual_budget_details["target_seconds"]
        min_seconds = visual_budget_details["min_seconds"]
        max_seconds = visual_budget_details["max_seconds"]
        preferred_min = max(min_seconds, target_seconds * 0.96)
        preferred_max = min(max_seconds, target_seconds * 1.04)
        if preferred_min > preferred_max:
            preferred_min = min_seconds
            preferred_max = max_seconds
        block_timeline = _format_budget_repair_block_timeline(invalid_script, _language)
        if actual_seconds < min_seconds:
            deficit_to_min = max(0.0, min_seconds - actual_seconds)
            deficit_to_target = max(0.0, target_seconds - actual_seconds)
            preferred_add_min = max(0.0, preferred_min - actual_seconds)
            preferred_add_max = max(preferred_add_min, preferred_max - actual_seconds)
            recommended_added = max(
                deficit_to_min + 8.0,
                min(deficit_to_target, max(preferred_add_min, deficit_to_min + 24.0)),
            )
            return f"""
FOCUSED REPAIR REQUIRED:
- The previous JSON under-selected the full-mode visual timeline: {actual_seconds:.1f}s playable visuals for a {target_seconds:.1f}s target, below the allowed minimum {min_seconds:.1f}s.
- The next JSON must calculate to {min_seconds:.1f}-{max_seconds:.1f}s playable visuals before it is returned. Aim for {preferred_min:.1f}-{preferred_max:.1f}s, not barely above the minimum; for this failed JSON that means recovering about {preferred_add_min:.1f}-{preferred_add_max:.1f}s useful playable time. Add at least {deficit_to_min:.1f}s; about {recommended_added:.1f}s is a safer first repair target.
- Treat this as a global edit-decision repair, not a one-block patch. If preserving the same block list prevents the target window, repartition the complete narration_blocks list: merge weak short ranges, split longer useful process ranges, replace redundant/low-value ranges, and select additional useful timestamp ranges from the visual evidence.
- The extra time must come from AI-selected, useful, scene-matched source ranges in the visual evidence. Do not invent filler ranges, evenly sample the timeline, or keep low-value footage just to pass validation.
- Do not create a new long sparse narration block to recover budget. Every added, extended, merged, or slowed non-pause range must still pass sync math: concrete narration must describe that exact range, estimated TTS should cover at least {FULL_MODE_MIN_NARRATED_BLOCK_VOICEOVER_RATIO:.0%} of playable duration, and no narrated block should leave more than {FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS:.0f}s of unmatched trailing visuals. Use brief pause=true blocks only where the picture or original sound genuinely carries the moment.
- AI may fix this by extending existing block boundaries into adjacent useful action, replacing weak cuts with stronger useful ranges, repartitioning blocks, or lowering over-aggressive video_speed only when that exact visible action should play slower. Do not blanket-disable acceleration.
- Preserve exactly {block_count} total narration_blocks by merging/splitting/repartitioning adjacent blocks as needed; keep chronological order and keep beginning, middle, and ending coverage.
- Current timing summary: selected source is about {visual_budget_details["source_seconds"]:.1f}s, playable after video_speed is about {actual_seconds:.1f}s, video_speed saved about {visual_budget_details["speed_saved_seconds"]:.1f}s across {visual_budget_details["accelerated_count"]} accelerated blocks, latest selected source end is {visual_budget_details["latest_end"]:.1f}s.
- Current block timing and sync table:
{block_timeline}
- Every changed or added range must have concrete visual, visual_facts, evidence_timestamps when available, matching narration, and speed_reason when video_speed > 1.0.
""".strip()
        if actual_seconds > max_seconds:
            excess = max(0.0, actual_seconds - max_seconds)
            preferred_reduce_min = max(0.0, actual_seconds - preferred_max)
            preferred_reduce_max = max(preferred_reduce_min, actual_seconds - preferred_min)
            return f"""
FOCUSED REPAIR REQUIRED:
- The previous JSON over-selected the full-mode visual timeline: {actual_seconds:.1f}s playable visuals for a {target_seconds:.1f}s target, above the allowed maximum {max_seconds:.1f}s.
- The next JSON must calculate to {min_seconds:.1f}-{max_seconds:.1f}s playable visuals before it is returned. Aim for {preferred_min:.1f}-{preferred_max:.1f}s, so remove or compress about {preferred_reduce_min:.1f}-{preferred_reduce_max:.1f}s useful-playable time; at minimum remove/compress {excess:.1f}s.
- Treat this as a global edit-decision repair, not a one-block patch. Repartition the complete narration_blocks list if needed: cut low-value ranges, merge tiny adjacent ranges, split kept process ranges only where narration remains scene-matched, and keep the beginning, middle, and final payoff.
- Remove or compress playable visual time by cutting low-value ranges or increasing video_speed only where the visible action is genuinely slow, repeated, waiting, setup, walking, transport, camera drift, or redundant.
- Do not make a hardcoded trim. AI must choose the cuts and acceleration from the visual evidence while preserving the process arc and the final payoff.
- After cutting/compressing, every remaining non-pause range must still pass sync math: concrete narration must describe that exact range, estimated TTS should cover at least {FULL_MODE_MIN_NARRATED_BLOCK_VOICEOVER_RATIO:.0%} of playable duration, and no narrated block should leave more than {FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS:.0f}s of unmatched trailing visuals.
- Preserve exactly {block_count} total narration_blocks by merging/splitting/repartitioning adjacent blocks as needed; keep chronological order and keep narration scene-matched to the kept ranges.
- Current block timing and sync table:
{block_timeline}
""".strip()
    near_full_match = re.search(
        r"Got\s+([0-9.]+)s\s+selected from a\s+([0-9.]+)s\s+source for a\s+([0-9.]+)s\s+target",
        error_text,
    )
    if near_full_match:
        selected_seconds = float(near_full_match.group(1))
        source_seconds = float(near_full_match.group(2))
        target_seconds = float(near_full_match.group(3))
        return f"""
FOCUSED REPAIR REQUIRED:
- The previous JSON selected {selected_seconds:.1f}s from a {source_seconds:.1f}s source for a {target_seconds:.1f}s edited target, so it was too close to a continuous source timeline.
- In the next JSON, make real editorial cuts: choose only the strongest process stages, remove waiting/setup/transport/repeated tool operation/camera drift, and keep total playable visual time close to {target_seconds:.1f}s.
- Preserve chronological coverage from beginning, middle, and ending, but do not keep long uncut ranges just to cover time.
""".strip()
    return ""


def _retry_correction_note(previous_error: Optional[str]) -> str:
    if not previous_error:
        return ""
    error_text = str(previous_error)
    visual_budget_match = _visual_budget_error_match(error_text)
    if visual_budget_match:
        try:
            actual_seconds = float(visual_budget_match.group(1))
            target_seconds = float(visual_budget_match.group(2))
            min_seconds = float(visual_budget_match.group(3))
            max_seconds = float(visual_budget_match.group(4))
            deficit = max(0.0, min_seconds - actual_seconds)
            excess = max(0.0, actual_seconds - max_seconds)
        except (TypeError, ValueError):
            actual_seconds = target_seconds = min_seconds = max_seconds = deficit = excess = 0.0
        if deficit > 0:
            repair_action = (
                f"select at least {deficit:.1f}s more useful playable visuals, or lower only those AI-chosen video_speed values "
                "whose exact visible actions should genuinely play slower"
            )
        elif excess > 0:
            repair_action = (
                f"remove or compress at least {excess:.1f}s of low-value playable visuals using AI-chosen cuts and visual reasons"
            )
        else:
            repair_action = "rebalance selected visual ranges so playable timing sits inside the validation window"
        return (
            "\n\nRetry correction note:\n"
            f"The previous run failed full-mode visual budget validation: {actual_seconds:.1f}s playable visuals for a {target_seconds:.1f}s target, allowed {min_seconds:.1f}-{max_seconds:.1f}s. "
            f"In the next run, {repair_action}. "
            "Do not use backend filler, evenly sampled ranges, or fixed timestamp rules. The AI must decide keep/cut/splice/speed from the visible action, and narration must stay scene-matched to those ranges."
        )
    historical_density_error = bool(re.search(
        r"AI narration block is too short for its selected visual range|"
        r"too short for comprehensive full-mode commentary|"
        r"expected at least\s+\d+|"
        r"character-density|"
        r"per-block density",
        error_text,
        flags=re.IGNORECASE,
    ))
    if historical_density_error:
        return (
            "\n\nRetry correction note:\n"
            "The previous run failed because a narration block was too sparse for its selected visual range. "
            "Do not add filler just to satisfy a word-count target, but do fix the timing: shorten the source range, split the block, add concrete scene-matched narration, or use only a brief intentional pause. "
            "Long selected source ranges still need matching spoken detail; do not rely on backend silence or render-time trimming."
        )
    if "Custom style operation logic validation failed" in error_text:
        return (
            "\n\nRetry correction note:\n"
            "The previous run failed because the selected custom commentary style was not actually followed. "
            "The script described actions but did not explain enough same-range operation logic, purpose, or visible results. "
            "In the next script, keep narration grounded in each timestamp range and rewrite ordinary process blocks with visible action first, then a concise why/purpose/result sentence where supported by the evidence."
        )
    unsupported_claim_error = bool(re.search(
        r"concrete action claim|"
        r"concrete claim|"
        r"claim(?:ed|s)? .* not supported|"
        r"not supported by (?:its|the) selected visual range",
        error_text,
        flags=re.IGNORECASE,
    ))
    if unsupported_claim_error:
        return (
            "\n\nRetry correction note:\n"
            "The previous run failed because one narration block claimed an action, state, result, identity, quantity, or risk that the selected visual range did not support. "
            "In the next script, every concrete claim must be grounded in the same block's timestamped keyframes, visual analysis, or transcript evidence. "
            "For any block without that evidence, describe only what is visible in that range and move the unsupported claim to a timestamp where it is actually visible."
        )
    compact_error = re.sub(r"\s+", " ", error_text).strip()
    compact_error = _limit_text_chars(compact_error, 600)
    return (
        "\n\nRetry correction note:\n"
        f"The previous response failed validation with this error: {compact_error}\n"
        "Return narration_blocks that cover about the requested target duration, not the entire raw source timeline. "
        "Keep commentary clear and scene-matched to the timestamped visuals; do not add filler narration, but do not under-explain visible actions just to be short."
    )


def _previous_error_invalidates_cached_script(previous_error: Optional[str]) -> bool:
    if not previous_error:
        return False
    error_text = str(previous_error)
    render_or_system_error = re.search(
        r"No space left on device|"
        r"\bffmpeg\b|"
        r"Error muxing|"
        r"muxer|"
        r"Error writing trailer|"
        r"Error closing file|"
        r"Error submitting a packet|"
        r"Conversion failed",
        error_text,
        flags=re.IGNORECASE,
    )
    if render_or_system_error:
        return False
    script_or_timeline_error = re.search(
        r"AI narration|"
        r"narration_blocks?|"
        r"visual range|"
        r"visual timeline|"
        r"full-mode edit target|"
        r"script validation|"
        r"failed validation|"
        r"invalid .*script|"
        r"JSON|"
        r"OpenAI-compatible model|"
        r"Gemini",
        error_text,
        flags=re.IGNORECASE,
    )
    return bool(script_or_timeline_error)



def _build_regeneration_prompt(
    original_prompt: str,
    short_script: Dict,
    duration: float,
    target_duration: str,
    language: str,
    attempt: int = 1,
    validation_error: Optional[Exception] = None,
) -> str:
    target_seconds = (
        _target_seconds_from_validation_error(validation_error)
        or _target_visual_duration_seconds(duration, target_duration)
    )
    max_chars = _maximum_narration_chars_for_target_seconds(target_seconds, target_duration, language)
    block_count = _target_narration_block_count_for_target_seconds(target_seconds)
    target_block_seconds = target_seconds / max(1, block_count)
    block_sync_instruction = _block_narration_sync_instruction(language)
    focused_repair_instruction = _focused_validation_repair_instruction(
        validation_error,
        short_script,
        language,
        block_count,
        duration,
        target_duration,
        target_seconds=target_seconds,
    )
    timeline_rules = _full_mode_regeneration_timeline_rules(duration, target_seconds)
    repair_scope_instruction = _repair_scope_instruction(validation_error, f"regeneration attempt {attempt}")
    if target_duration != "full":
        min_seconds, max_seconds = _target_duration_window_seconds(duration, target_duration)
        duration_label = _non_full_target_duration_label(target_duration)
        return f"""{original_prompt}

PREVIOUS RESPONSE WAS INVALID:
{json.dumps(short_script, ensure_ascii=False)}

VALIDATION ERROR:
{validation_error or "The previous script failed target-duration validation."}

REGENERATE FROM THE ATTACHED VIDEO:
- {repair_scope_instruction}
- Repartition the complete edit for the requested {duration_label} commentary. The final narration_blocks/edit_segments playable time after video_speed must not exceed {max_seconds:.0f} seconds; aim for at least {min_seconds:.0f} seconds only when the source has enough useful non-repetitive material.
- Do not output the whole source timeline or a near-full-source edit. Select only the strongest useful source ranges, keep chronological order, remove repetitive/waiting/setup/camera-drift/low-value footage, and use justified video_speed for slow-but-useful ranges.
- Each narration_blocks item must match its exact visible range. If a block would be too long for its narration, shorten the source range, split the block, or cut the weak tail.
- Return narration_blocks with start, end, visual, narration, pause, rate, pitch, video_speed, and speed_reason. Keep pause=true blocks brief and only when the original picture or sound should carry the moment.
- Keep title, summary, hook, hashtags, cut_strategy, and chapters consistent with the shorter selected edit.
- Return valid JSON only, using the same JSON FORMAT.
"""
    return f"""{original_prompt}

PREVIOUS RESPONSE WAS INVALID:
{json.dumps(short_script, ensure_ascii=False)}

VALIDATION ERROR:
{validation_error or "The previous script failed full-mode commentary validation."}

{focused_repair_instruction}

REGENERATE FROM THE ATTACHED VIDEO:
- {repair_scope_instruction}
- Use the attached video visual evidence again for any repaired or regenerated ranges.
{timeline_rules}
- Write clear scene-matched narration for the edited visuals. Explain what is happening on screen in each timestamp range; do not add filler, but do not intentionally shorten the explanation.
{_banned_phrase_instruction()}
- Narration must be at most {max_chars} non-whitespace characters so it remains speakable with the selected visuals; shorter commentary is valid only when it still explains the visible content clearly.
- Return exactly {block_count} narration_blocks with start, end, visual, narration, pause, rate, pitch, video_speed, and speed_reason.
- If episode_plan.should_split=true, keep episodes aligned to the repaired 1-based narration_blocks indexes using start_block and end_block.
- Aim for about {target_block_seconds:.0f}s playable visuals per block across the whole edit, but keep ordinary narrated blocks usually 8-16s after video_speed so the actual TTS can cover the selected visuals. Split longer useful ranges into multiple narrated blocks or explicit brief pause=true blocks; do not leave a 40-60s narrated block with a short paragraph.
- Use concrete, clear narration for normal narrated blocks. pause=true blocks must leave narration empty.
- Any concrete action, state change, result, object identity, quantity, danger, or completion claim must be supported by that same block's timestamped visual evidence or transcript evidence.
- {block_sync_instruction}
- Each non-pause narration block must be speakable inside that block's visual duration; do not cram long narration into a short range.
- Use pause=true blocks for key reveals, process sounds, skilled visual moments, transitions, or scenes where the picture genuinely needs to play without commentary; keep pause blocks usually 2-12 seconds, under about 25% of selected visual time, and avoid more than two pause blocks back-to-back.
- Decide video_speed from the actual visible action, not from a fixed rule. Use 1.0 for key reveals, removal moments, packaging/closure, readable text, final results, completed states, tests, installations, and payoff shots. Use moderate speeds such as 1.15-1.5 for slow but still useful process footage; use 1.75-2.5 only when the range is clearly repetitive, waiting, walking, setup, repeated tool operation, transport, or transition footage and remains understandable after acceleration. If the visual evidence marks a kept range as slow/repetitive/waiting/transition or includes suggested_speed > 1.0, either set video_speed above 1.0 with a concrete speed_reason or shorten/cut that range. Every block with video_speed > 1.0 must include a concrete speed_reason tied to that exact visual range; blocks at 1.0 can use speed_reason "".
- Vary rate and pitch across non-pause blocks so the voice has cadence; do not return every block as +0% and +0Hz.
- Do not summarize the whole video in one short paragraph.
- Explain what is happening on screen in each selected visual section and transition naturally between stages.
- Every major narration paragraph must correspond to a visible source-video range in edit_segments or chapters.
- Return valid JSON only, using the same JSON FORMAT.
"""


def _build_visual_plan_finalization_prompt(
    visual_plan: Dict,
    duration: float,
    target_duration: str,
    language: str,
    attempt: int = 1,
    validation_error: Optional[Exception] = None,
    custom_style_prompt: Optional[str] = None,
) -> str:
    target_seconds = (
        _target_seconds_from_validation_error(validation_error)
        or _target_visual_duration_seconds(duration, target_duration)
    )
    max_chars = _maximum_narration_chars_for_target_seconds(target_seconds, target_duration, language)
    block_count = _target_narration_block_count_for_target_seconds(target_seconds)
    target_block_seconds = target_seconds / max(1, block_count)
    block_sync_instruction = _block_narration_sync_instruction(language)
    timeline_rules = _full_mode_timeline_rules(duration, target_seconds)
    focused_repair_instruction = _focused_validation_repair_instruction(
        validation_error,
        visual_plan,
        language,
        block_count,
        duration,
        target_duration,
        target_seconds=target_seconds,
    )
    repair_scope_instruction = _repair_scope_instruction(validation_error, f"finalization attempt {attempt}")
    custom_style_instruction = _custom_style_instruction(custom_style_prompt)
    return f"""You are writing the final voiceover for a commentary remix.

VIDEO-DERIVED VISUAL PLAN:
{json.dumps(visual_plan, ensure_ascii=False)}

CUSTOM STYLE PROMPT:
{custom_style_prompt or ""}

VALIDATION ERROR:
{validation_error or "The previous script needs full-mode validation before rendering."}

{focused_repair_instruction}

FINALIZE COMPLETE COMMENTARY:
- {repair_scope_instruction}
- Use the video-derived visual plan above as the source of visual truth.
- Do not invent unrelated scenes. Every paragraph must follow the timestamps, visual descriptions, chapters, or edit_segments in the visual plan.
- Apply the selected/custom commentary style to the final narration, not only the title or summary.{custom_style_instruction}
{timeline_rules}
- Write a complete Simplified Chinese voiceover for the edited visuals. Explain the timestamped visual content clearly; use pauses only where the picture or original sound genuinely carries the meaning.
{_banned_phrase_instruction()}
- The top-level title must clearly say what the video is doing: name the concrete subject, process/action, and result or purpose. Use titles like "废旧电机拆解回收铜线全过程" instead of vague hype titles like "震撼工厂全过程" or "不可思议的改造".
- The final narration must be at most {max_chars} non-whitespace characters so it remains speakable with the selected visuals; shorter commentary is valid only when it still explains the selected visuals clearly.
- Return exactly {block_count} narration_blocks with start, end, visual, narration, pause, rate, pitch, video_speed, and speed_reason.
- If episode_plan.should_split=true, keep episodes aligned to the repaired 1-based narration_blocks indexes using start_block and end_block.
- Aim for about {target_block_seconds:.0f}s playable visuals per block across the whole edit, but keep ordinary narrated blocks usually 8-16s after video_speed so the actual TTS can cover the selected visuals. Split longer useful ranges into multiple narrated blocks or explicit brief pause=true blocks; do not leave a 40-60s narrated block with a short paragraph.
- Use concrete, clear narration for normal narrated blocks. pause=true blocks must leave narration empty.
- Any concrete action, state change, result, object identity, quantity, danger, or completion claim must be supported by that same block's timestamped visual evidence or transcript evidence.
- {block_sync_instruction}
- Each non-pause narration block must be speakable inside that block's visual duration.
- Use pause=true blocks for key reveals, process sounds, skilled visual moments, transitions, or scenes where the picture genuinely needs to play without commentary; keep pause blocks usually 2-12 seconds, under about 25% of selected visual time, and avoid more than two pause blocks back-to-back.
- Decide video_speed from the actual visible action, not from a fixed rule. Use 1.0 for key reveals, removal moments, packaging/closure, readable text, final results, completed states, tests, installations, and payoff shots. Use moderate speeds such as 1.15-1.5 for slow but still useful process footage; use 1.75-2.5 only when the range is clearly repetitive, waiting, walking, setup, repeated tool operation, transport, or transition footage and remains understandable after acceleration. If the visual evidence marks a kept range as slow/repetitive/waiting/transition or includes suggested_speed > 1.0, either set video_speed above 1.0 with a concrete speed_reason or shorten/cut that range. Every block with video_speed > 1.0 must include a concrete speed_reason tied to that exact visual range; blocks at 1.0 can use speed_reason "".
- Vary rate and pitch across non-pause blocks so the voice has cadence; do not return every block as +0% and +0Hz.
- Preserve chronological order and keep the commentary matched to the visible source process.
- Return valid JSON only.

JSON FORMAT:
{{
  "title": "specific title that says what the video does",
  "summary": "brief summary",
  "hook": "opening hook",
  "narration": "complete final voiceover text",
  "narration_blocks": [
    {{"start": 0, "end": 30, "visual": "visual plan item", "visual_facts": ["concrete visible fact from this range"], "evidence_timestamps": [3.0, 12.0], "narration": "final voiceover for this range", "pause": false, "rate": "+0%", "pitch": "+0Hz", "video_speed": 1.0, "speed_reason": ""}},
    {{"start": 30, "end": 40, "visual": "slow repeated setup or movement", "visual_facts": ["why it can remain understandable when accelerated"], "evidence_timestamps": [35.0], "narration": "short commentary for this sped-up range", "pause": false, "rate": "+6%", "pitch": "+1Hz", "video_speed": 1.5, "speed_reason": "visible setup/repeated motion is slow, so 1.5x keeps the process without dragging"}},
    {{"start": 40, "end": 48, "visual": "original footage moment that should breathe", "visual_facts": ["why the moment should play with ambient sound"], "evidence_timestamps": [44.0], "narration": "", "pause": true, "rate": "+0%", "pitch": "+0Hz", "video_speed": 1.0, "speed_reason": ""}}
  ],
  "edit_segments": [
    {{"start": 0, "end": 30, "reason": "why this range is kept"}}
  ],
  "chapters": [
    {{"start": 0, "end": 30, "title": "chapter title", "narration": "chapter summary"}}
  ],
  "hashtags": ["#tag"]
}}
"""


def _replace_prompt_in_contents(contents, prompt: str):
    if not contents:
        return [prompt]
    if len(contents) == 1 and isinstance(contents[0], str):
        return [prompt]

    first_content = contents[0]
    parts = []
    for part in getattr(first_content, "parts", []) or []:
        if getattr(part, "text", None):
            continue
        parts.append(part)
    parts.append(types.Part.from_text(text=prompt))
    return [
        types.Content(
            role=getattr(first_content, "role", None) or "user",
            parts=parts,
        )
    ]


def _timeline_item_position(item: Dict, fallback_index: int) -> float:
    for key in ("timestamp", "start"):
        try:
            return float(item[key])
        except (KeyError, TypeError, ValueError):
            pass
    try:
        return (float(item["start"]) + float(item["end"])) / 2.0
    except (KeyError, TypeError, ValueError):
        return float(fallback_index)


def _sample_timeline_items(items: List[Dict], max_items: int) -> List[Dict]:
    if max_items <= 0:
        return []

    ordered = sorted(
        enumerate(items),
        key=lambda pair: (_timeline_item_position(pair[1], pair[0]), pair[0]),
    )
    if len(ordered) <= max_items:
        return [item for _index, item in ordered]

    if max_items == 1:
        return [ordered[0][1]]

    source_count = len(ordered)
    selected_positions = []
    seen = set()
    for index in range(max_items):
        position = round(index * (source_count - 1) / (max_items - 1))
        if position not in seen:
            selected_positions.append(position)
            seen.add(position)

    if len(selected_positions) < max_items:
        for position in range(source_count):
            if position not in seen:
                selected_positions.append(position)
                seen.add(position)
                if len(selected_positions) >= max_items:
                    break

    return [ordered[position][1] for position in sorted(selected_positions[:max_items])]


def _compact_openai_visual_analysis(
    visual_analysis: Dict,
    max_observations: int = 220,
    max_candidate_segments: int = 160,
) -> Dict:
    frames = visual_analysis.get("frames") or []
    observations = visual_analysis.get("observations") or []
    candidate_segments = visual_analysis.get("candidate_segments") or []
    compact = {
        "provider": visual_analysis.get("provider"),
        "model": visual_analysis.get("model"),
        "frame_count": visual_analysis.get("frame_count"),
        "batch_count": visual_analysis.get("batch_count"),
        "sampling": visual_analysis.get("sampling"),
        "scene_count": visual_analysis.get("scene_count"),
        "sampling_options": visual_analysis.get("sampling_options"),
        "timeline_coverage": {
            "frames": len(frames),
            "observations": len(observations),
            "candidate_segments": len(candidate_segments),
        },
        "frames": _sample_timeline_items(frames, max_observations),
        "observations": _sample_timeline_items(observations, max_observations),
        "candidate_segments": _sample_timeline_items(candidate_segments, max_candidate_segments),
    }
    return {key: value for key, value in compact.items() if value not in (None, [], {})}


def _openai_timestamped_frame_table(visual_analysis: Optional[Dict], limit: int = 260) -> str:
    if not visual_analysis:
        return ""
    frames = visual_analysis.get("frames") or []
    observations = visual_analysis.get("observations") or []
    by_timestamp = {}
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        try:
            timestamp = round(float(observation.get("timestamp")), 3)
        except (TypeError, ValueError):
            continue
        parts = []
        for key in ("visual", "reason"):
            value = re.sub(r"\s+", " ", str(observation.get(key) or "").strip())
            if value:
                parts.append(value)
        if parts:
            by_timestamp.setdefault(timestamp, []).append("; ".join(parts))

    rows = []
    for frame in sorted((item for item in frames if isinstance(item, dict)), key=lambda item: float(item.get("timestamp") or 0.0)):
        try:
            timestamp = round(float(frame.get("timestamp")), 3)
        except (TypeError, ValueError):
            continue
        row = {
            "timestamp": timestamp,
            "scene_start": frame.get("scene_start"),
            "scene_end": frame.get("scene_end"),
            "sample_role": frame.get("sample_role"),
        }
        evidence = by_timestamp.get(timestamp) or []
        if evidence:
            row["visual_evidence"] = _limit_text_chars(" / ".join(evidence[:2]), 260)
        rows.append({key: value for key, value in row.items() if value not in (None, "", [])})
        if len(rows) >= limit:
            break
    return json.dumps(rows, ensure_ascii=False)


def _openai_frame_infos_for_prompt(frame_infos: List[Dict]) -> List[Dict]:
    frames = []
    for frame in frame_infos or []:
        if not isinstance(frame, dict):
            continue
        try:
            timestamp = round(float(frame.get("timestamp")), 3)
        except (TypeError, ValueError):
            continue
        frames.append({
            key: value
            for key, value in {
                "timestamp": timestamp,
                "scene_index": frame.get("scene_index"),
                "scene_start": frame.get("scene_start"),
                "scene_end": frame.get("scene_end"),
                "scene_duration": frame.get("scene_duration"),
                "sample_role": frame.get("sample_role"),
                "motion_score": frame.get("motion_score"),
            }.items()
            if value is not None
        })
    return frames


def _openai_visual_analysis_prompt_text(visual_analysis: Dict) -> str:
    observation_limit = 220
    candidate_limit = 160
    compact = _compact_openai_visual_analysis(visual_analysis, observation_limit, candidate_limit)
    text = json.dumps(compact, ensure_ascii=False)
    if len(text) <= OPENAI_VISUAL_PROMPT_MAX_CHARS:
        return text

    while len(text) > OPENAI_VISUAL_PROMPT_MAX_CHARS and (observation_limit > 1 or candidate_limit > 1):
        if observation_limit >= candidate_limit and observation_limit > 1:
            observation_limit = max(1, int(observation_limit * 0.75))
        elif candidate_limit > 1:
            candidate_limit = max(1, int(candidate_limit * 0.75))
        compact = _compact_openai_visual_analysis(visual_analysis, observation_limit, candidate_limit)
        text = json.dumps(compact, ensure_ascii=False)
    return text


def _openai_candidate_edit_plan_prompt_text(edit_plan: Optional[Dict]) -> str:
    if not edit_plan:
        return ""
    plan = {
        "target_seconds": round(float(edit_plan.get("target_seconds") or 0.0), 1),
        "allowed_playable_seconds": [
            round(float(edit_plan.get("min_seconds") or 0.0), 1),
            round(float(edit_plan.get("max_seconds") or 0.0), 1),
        ],
        "planned_playable_seconds": round(float(edit_plan.get("playable_seconds") or 0.0), 1),
        "planned_source_seconds": round(float(edit_plan.get("source_seconds") or 0.0), 1),
        "blocks": edit_plan.get("blocks") or [],
    }
    return json.dumps(plan, ensure_ascii=False)


def _locked_plan_block_guidance_text() -> str:
    return (
        "For locked OpenAI-compatible plan blocks, use each block's min_narration_chars as the sync target for non-pause narration; "
        "write concrete scene-matched detail from that exact visual range rather than filler. "
        "Do not add, remove, retime, or reinterpret the locked visual range; any concrete claim must be grounded in that block's visual_facts and evidence_timestamps."
    )


def _apply_openai_candidate_edit_plan(data: Dict, edit_plan: Optional[Dict], language: str = "") -> None:
    if not edit_plan:
        return
    plan_blocks = [block for block in (edit_plan.get("blocks") or []) if isinstance(block, dict)]
    if not plan_blocks:
        return
    model_blocks = _script_narration_blocks(data)
    top_level_narration = str(data.get("narration") or "").strip()
    fallback_chunks: List[str] = []
    if top_level_narration:
        if (language or "").lower().startswith("zh"):
            fallback_chunks = [
                chunk.strip()
                for chunk in re.split(r"(?<=[。！？!?])", re.sub(r"\s+", "", top_level_narration))
                if chunk.strip() and not _contains_visual_analysis_label_artifact(chunk, language)
            ]
        else:
            fallback_chunks = [
                chunk.strip()
                for chunk in re.split(r"(?<=[.!?])\s+", top_level_narration)
                if chunk.strip()
            ]
    fallback_index = 0
    rewritten = []
    used_model_indexes = set()
    for index, plan_block in enumerate(plan_blocks):
        model_block = {}
        plan_start = float(plan_block.get("start") or 0.0)
        plan_end = float(plan_block.get("end") or 0.0)
        best_model_index = None
        best_overlap = 0.0
        best_score = -1.0
        for model_index, candidate in enumerate(model_blocks):
            if model_index in used_model_indexes:
                continue
            try:
                candidate_start = float(candidate.get("start"))
                candidate_end = float(candidate.get("end"))
            except (TypeError, ValueError):
                continue
            if candidate_end <= candidate_start:
                continue
            overlap = _segment_overlap_seconds(plan_start, plan_end, candidate_start, candidate_end)
            if overlap <= 0:
                continue
            union = max(plan_end, candidate_end) - min(plan_start, candidate_start)
            score = overlap / union if union > 0 else 0.0
            if score > best_score:
                best_score = score
                best_overlap = overlap
                best_model_index = model_index
        if (
            best_model_index is not None
            and best_overlap >= min(max(0.5, (plan_end - plan_start) * 0.35), 3.0)
        ):
            model_block = model_blocks[best_model_index]
            used_model_indexes.add(best_model_index)
        elif index < len(model_blocks) and index not in used_model_indexes:
            positional_block = model_blocks[index]
            try:
                float(positional_block.get("start"))
                float(positional_block.get("end"))
                has_positional_timestamps = True
            except (TypeError, ValueError):
                has_positional_timestamps = False
            if not has_positional_timestamps:
                model_block = positional_block
                used_model_indexes.add(index)
        narration = str(model_block.get("narration") or model_block.get("text") or "").strip()
        if _contains_visual_analysis_label_artifact(narration, language):
            narration = _strip_visual_analysis_label_artifact(narration, language)
        if not narration and fallback_chunks:
            remaining_blocks = max(1, len(plan_blocks) - index)
            remaining_chunks = max(0, len(fallback_chunks) - fallback_index)
            take = max(1, int(math.ceil(remaining_chunks / remaining_blocks))) if remaining_chunks else 0
            if take > 0:
                narration = "".join(fallback_chunks[fallback_index:fallback_index + take]).strip()
                fallback_index += take
        pause = bool(plan_block.get("pause")) and not narration
        block_duration = max(
            0.0,
            (float(plan_block.get("end") or 0.0) - float(plan_block.get("start") or 0.0))
            / _safe_video_speed(plan_block.get("video_speed")),
        )
        if narration and block_duration > 0:
            max_voice_seconds = max(
                0.1,
                min(
                    block_duration * max(0.01, FULL_MODE_MAX_VOICEOVER_DURATION_RATIO),
                    block_duration + max(0.0, FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS),
                ),
            )
            narration = _shorten_narration_to_fit_visual(narration, max_voice_seconds, language)
        visual = str(plan_block.get("visual") or "").strip() or str(model_block.get("visual") or "").strip()
        visual_facts = plan_block.get("visual_facts") if isinstance(plan_block.get("visual_facts"), list) else []
        evidence_timestamps = plan_block.get("evidence_timestamps") if isinstance(plan_block.get("evidence_timestamps"), list) else []
        rewritten.append({
            "start": float(plan_block.get("start") or 0.0),
            "end": float(plan_block.get("end") or 0.0),
            "visual": visual or str(plan_block.get("visual") or "AI-selected useful visual range"),
            "visual_facts": [str(fact).strip() for fact in (visual_facts or []) if str(fact).strip()],
            "evidence_timestamps": [
                float(ts)
                for ts in (evidence_timestamps or [])
                if isinstance(ts, (int, float))
            ],
            "narration": "" if pause else narration,
            "pause": pause,
            "rate": _safe_edge_rate(model_block.get("rate") or "+0%"),
            "pitch": _safe_edge_pitch(model_block.get("pitch") or "+0Hz"),
            "video_speed": _safe_video_speed(plan_block.get("video_speed")),
            "speed_reason": str(plan_block.get("speed_reason") or "").strip(),
            "_locked_edit_plan": True,
            "_min_narration_chars": int(plan_block.get("min_narration_chars") or 0),
        })
    data["narration_blocks"] = rewritten
    data["edit_segments"] = _narration_blocks_to_edit_segments(rewritten)
    data["narration"] = _narration_from_blocks({"narration_blocks": rewritten}) or str(data.get("narration") or "")


def _validate_locked_plan_has_required_narration(data: Dict) -> None:
    blocks = data.get("narration_blocks") if isinstance(data, dict) else None
    if not isinstance(blocks, list) or not blocks:
        return
    missing = []
    for index, block in enumerate(blocks, start=1):
        if not isinstance(block, dict) or not bool(block.get("_locked_edit_plan")) or bool(block.get("pause")):
            continue
        if not str(block.get("narration") or "").strip():
            missing.append(index)
    if missing:
        shown = ", ".join(str(index) for index in missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise Exception(
            "OpenAI locked edit plan is missing narration for selected visual blocks. "
            f"Missing block indexes: {shown}{suffix}. "
            "Write concrete scene-matched narration for every non-pause locked block using its visual_facts and evidence_timestamps; do not leave selected important footage empty."
        )


def _fallback_locked_plan_narration(block: Dict, language: str) -> str:
    if not isinstance(block, dict) or bool(block.get("pause")):
        return ""
    fact_sentence = _scene_fact_sentence(block, language).strip()
    visual = str(block.get("visual") or "").strip()
    if (language or "").lower().startswith("zh"):
        if fact_sentence:
            return fact_sentence
        if (
            visual
            and re.search(r"[\u3400-\u9fff]", visual)
            and not _contains_visual_analysis_label_artifact(visual, language)
        ):
            return visual.rstrip("，,。！？!?") + "。"
        return ""
    if fact_sentence:
        return fact_sentence
    if visual:
        return visual.rstrip(".!?") + "."
    return "This kept segment shows the visible process continuing."


def _fill_missing_locked_plan_narration(data: Dict, language: str) -> None:
    blocks = data.get("narration_blocks") if isinstance(data, dict) else None
    if not isinstance(blocks, list) or not blocks:
        return
    changed = False
    for block in blocks:
        if not isinstance(block, dict) or not bool(block.get("_locked_edit_plan")) or bool(block.get("pause")):
            continue
        if str(block.get("narration") or "").strip():
            continue
        fallback = _fallback_locked_plan_narration(block, language)
        if fallback:
            block["narration"] = fallback
            changed = True
    if changed:
        _commit_narration_blocks_to_script(data, blocks)


def _fit_locked_plan_narration_to_budget(data: Dict, max_chars: int, language: str) -> None:
    if max_chars <= 0:
        return
    blocks = data.get("narration_blocks") if isinstance(data, dict) else None
    if not isinstance(blocks, list) or not blocks:
        return
    if not any(isinstance(block, dict) and bool(block.get("_locked_edit_plan")) for block in blocks):
        return
    for block in blocks:
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        text = str(block.get("narration") or "").strip()
        if not text:
            continue
        block_duration = _block_visual_duration(block)
        if block_duration <= 0:
            continue
        max_voice_seconds = max(
            0.1,
            min(
                block_duration * max(0.01, FULL_MODE_MAX_VOICEOVER_DURATION_RATIO),
                block_duration + max(0.0, FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS),
            ),
        )
        shortened = _shorten_narration_to_fit_visual(text, max_voice_seconds, language)
        if shortened:
            block["narration"] = shortened
    narration = re.sub(r"\s+", "", _narration_from_blocks({"narration_blocks": blocks}))
    if len(narration) <= max_chars:
        data["narration_blocks"] = blocks
        data["narration"] = _narration_from_blocks({"narration_blocks": blocks})
        return
    entries = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        text = str(block.get("narration") or "").strip()
        if not text:
            continue
        compact_len = len(re.sub(r"\s+", "", text))
        if compact_len <= 0:
            continue
        raw_min_chars = int(block.get("_min_narration_chars") or _minimum_sync_narration_chars_for_visual_duration(_block_visual_duration(block), language))
        min_chars = min(compact_len, max(1, raw_min_chars))
        entries.append({
            "index": index,
            "block": block,
            "text": text,
            "current": compact_len,
            "min": min_chars,
            "weight": max(0.1, _block_visual_duration(block)),
            "target": min_chars,
        })
    if entries:
        min_total = sum(entry["min"] for entry in entries)
        if min_total > max_chars:
            remaining = max_chars
            total_weight = sum(entry["weight"] for entry in entries) or float(len(entries))
            for entry in entries:
                if remaining <= 0:
                    entry["target"] = 0
                    continue
                share = int(math.floor(max_chars * (entry["weight"] / total_weight)))
                entry["target"] = min(entry["current"], max(1, share))
                remaining -= entry["target"]
            while remaining < 0:
                changed = False
                for entry in sorted(entries, key=lambda item: (item["target"], item["weight"]), reverse=True):
                    if remaining >= 0:
                        break
                    if entry["target"] <= 1:
                        continue
                    entry["target"] -= 1
                    remaining += 1
                    changed = True
                if not changed:
                    break
            while remaining > 0:
                changed = False
                for entry in sorted(entries, key=lambda item: item["weight"], reverse=True):
                    if remaining <= 0:
                        break
                    if entry["target"] >= entry["current"]:
                        continue
                    entry["target"] += 1
                    remaining -= 1
                    changed = True
                if not changed:
                    break
        else:
            remaining = max_chars - min_total
            while remaining > 0:
                expandable = [entry for entry in entries if entry["target"] < entry["current"]]
                if not expandable:
                    break
                total_weight = sum(entry["weight"] for entry in expandable) or float(len(expandable))
                changed = False
                for entry in sorted(expandable, key=lambda item: item["weight"], reverse=True):
                    if remaining <= 0:
                        break
                    capacity = entry["current"] - entry["target"]
                    if capacity <= 0:
                        continue
                    share = max(1, int(math.floor(remaining * (entry["weight"] / total_weight))))
                    add = min(capacity, share, remaining)
                    entry["target"] += add
                    remaining -= add
                    changed = True
                if not changed:
                    break
        for entry in entries:
            if entry["target"] <= 0:
                entry["block"]["narration"] = ""
            elif entry["target"] < entry["current"]:
                entry["block"]["narration"] = _trim_narration_to_compact_chars(
                    entry["text"],
                    entry["target"],
                    language,
                )
    narration = re.sub(r"\s+", "", _narration_from_blocks({"narration_blocks": blocks}))
    if len(narration) <= max_chars:
        data["narration_blocks"] = blocks
        data["narration"] = _narration_from_blocks({"narration_blocks": blocks})
        return
    overage = len(narration) - max_chars
    for block in reversed(blocks):
        if overage <= 0:
            break
        text = str(block.get("narration") or "").strip()
        compact = re.sub(r"\s+", "", text)
        min_chars = int(block.get("_min_narration_chars") or _minimum_sync_narration_chars_for_visual_duration(_block_visual_duration(block), language))
        removable = max(0, len(compact) - min_chars)
        if removable <= 0:
            continue
        remove = min(removable, overage)
        keep = max(min_chars, len(compact) - remove)
        block["narration"] = _trim_narration_to_compact_chars(text, keep, language)
        overage = len(re.sub(r"\s+", "", _narration_from_blocks({"narration_blocks": blocks}))) - max_chars
    data["narration_blocks"] = blocks
    data["narration"] = _narration_from_blocks({"narration_blocks": blocks})


def _build_commentary_prompt(
    transcript: Dict,
    video_title: str,
    duration: float,
    language: str,
    style: str,
    target_duration: str,
    analysis_mode: str,
    visual_analysis: Optional[Dict] = None,
    custom_style_prompt: Optional[str] = None,
    openai_candidate_edit_plan: Optional[Dict] = None,
    source_audio_analysis: Optional[Dict] = None,
) -> str:
    mode = _normalize_analysis_mode(analysis_mode)
    sampled_segments = _sample_transcript_segments(transcript)
    source_commentary_timeline = _format_source_commentary_timeline(transcript)
    source_commentary_available = bool(source_commentary_timeline)
    source_audio_analysis_text = _openai_source_audio_analysis_prompt_text(source_audio_analysis)
    transcript_text = transcript.get("text", "")
    if len(transcript_text) > 60000:
        transcript_text = transcript_text[:60000]
    is_edited_video_commentary = (
        mode == "openai"
        and isinstance(visual_analysis, dict)
        and visual_analysis.get("analysis_stage") == "edited_video_commentary"
    )
    target_seconds = _target_visual_duration_seconds_for_analysis(duration, target_duration, visual_analysis)
    max_chars = _maximum_narration_chars_for_target_seconds(target_seconds, target_duration, language)
    block_count = _target_narration_block_count_for_target_seconds(target_seconds)
    target_block_seconds = target_seconds / max(1, block_count)
    block_sync_instruction = _block_narration_sync_instruction(language)
    timeline_rules = _full_mode_timeline_rules(duration, target_seconds)
    preserves_full_process = _full_mode_preserves_source_process(duration, target_seconds)
    cut_selection_instruction = (
        "- The visual content has already been edited into the final cut before this narration pass. Do not choose new source ranges, do not remove scenes, and do not reorder anything; write narration for the edited video timeline from 0.0 seconds to the edited duration."
        if is_edited_video_commentary
        else (
            "- Preserve the source workflow in chronological order for this full-process edit. Remove only clearly useless dead time, duplicated waiting, setup, walking, camera drift, intro/outro, irrelevant, or failed footage; prefer video_speed for slow/repetitive but meaningful process ranges."
            if target_duration == "full" and preserves_full_process
            else "- Select which original video ranges should be kept for the final edit and which ranges should be removed. Remove repetitive, slow, duplicated, waiting, setup, walking, camera drift, intro/outro, irrelevant, or low-value filler parts; use AI-chosen video_speed for slow-but-useful ranges that should remain understandable instead of being deleted."
        )
    )
    chronological_instruction = (
        f"- Use the edited-video timeline, not the original source timeline. The narration_blocks should cover the already-edited video from 0.0s through about {duration:.1f}s in chronological order, with video_speed=1.0 because speed changes were already baked into the intermediate edit."
        if is_edited_video_commentary
        else (
            "- The kept visual ranges must stay in the same chronological order as the source video and may preserve the full useful workflow when the source itself is shorter than the target."
            if target_duration == "full" and preserves_full_process
            else f"- The kept visual ranges must stay in the same chronological order as the source video, cover the complete process arc, and should total about {int(target_seconds)} playable seconds after video_speed rather than one continuous full-source range."
        )
    )
    openai_one_shot_sync_instruction = ""
    if mode == "openai" and target_duration == "full":
        openai_one_shot_sync_instruction = (
            "- For OpenAI-compatible mode, avoid spending output tokens on audit tables; return the production script and timeline only. "
            "The backend will handle block-level render sync deterministically."
        )
    pause_audio_instruction = (
        "- For TARGET DURATION full, because this source contains original spoken commentary, do not use pause=true blocks to rely on old source narration. Pause=true may only be used for visual breathing room, silence, background music, or non-speech process sound; any needed original explanation must be rewritten into new narration."
        if source_commentary_available
        else "- For TARGET DURATION full, pause blocks should use the original source audio as the main sound, but total pause time must stay under about 25% of selected visual time. Avoid more than two pause blocks back-to-back."
    )
    style_grounding = _style_grounding_instruction(style, language)
    custom_style_instruction = _custom_style_instruction(custom_style_prompt)
    json_output_contract = _commentary_json_output_contract(block_count, target_duration)

    visual_analysis_text = ""
    timestamped_frame_table = ""
    if mode == "video":
        visual_instruction = (
            "- A low-resolution video copy of the complete source is attached for Gemini visual analysis only; final editing, audio bed, "
            "and muxing use the separate high-quality source video.\n"
            "- You must inspect the entire attached video timeline from 0.0 seconds to the source duration before writing the script; do not analyze only the opening, a few highlights, or isolated sampled moments.\n"
            "- All edit_segments and narration_blocks must use timestamps from the original full source video timeline and must be selected from across the complete beginning, middle, and ending timeline."
        )
    elif mode == "openai":
        visual_instruction = (
            "- A timestamped multimodal frame analysis of the complete source timeline is provided below. "
            "Use these source-video timestamps as the primary evidence for edit_segments and narration_blocks.\n"
            "- The frame table lists the extracted source-video timestamps. Each narration_blocks item must use start/end ranges that contain one or more of those frame timestamps, and evidence_timestamps must list the exact frame timestamps used for that block.\n"
            "- You must carefully review every keyframe and timestamp, combine the keyframe content with its exact timestamp, and also consider the overall keyframe timeline analysis before writing narration.\n"
            "- Write the commentary from the timestamped visual evidence first, then use the transcript only as supporting context. Do not let transcript text override what the frame at that timestamp shows.\n"
            "- For every non-pause block, describe the visible action, objects, tools, people, movement, state change, and result shown at that block's evidence_timestamps. The chosen style changes tone, not the timestamp match.\n"
            "- Do not fabricate, do not lie, and do not claim anything that is not supported by the keyframes or transcript. 不允许造假，不允许说谎；如果关键帧看不清，就只写可确认的画面内容或明确不确定。\n"
            "- Do not intentionally shorten the explanation. Use as many clear, natural sentences as needed for the selected visual range, while keeping the narration speakable inside that block's duration.\n"
            "- The final edit must keep the video's important and good-looking content: choose ranges with clear importance, interest_score, edit_value, keep_candidate, payoff, process progress, visual skill/risk, result reveal, or story value. Do not fill duration with arbitrary, random-looking, weak, or low-value timestamp ranges.\n"
            "- When candidate_segments include suggested_speed or speed_reason, use them as visual evidence for narration_blocks.video_speed, but still make the final speed decision from the exact selected timestamp range.\n"
            "- All edit_segments and narration_blocks must use timestamps from the original full source video timeline and must be selected from across the complete beginning, middle, and ending timeline."
        )
        if is_edited_video_commentary:
            visual_instruction = (
                "- A timestamped multimodal frame analysis of the already-edited intermediate video is provided below. "
                "Use these edited-video timestamps as the production timeline for narration_blocks.\n"
                "- The full-source analysis summary is included only as background context to understand the story/process. Do not use original source timestamps as narration_blocks start/end in this final pass.\n"
                "- The frame table lists extracted edited-video timestamps. Each narration_blocks item must use start/end ranges from the edited video that contain one or more of those frame timestamps, and evidence_timestamps must list exact edited-video frame timestamps.\n"
                "- Do not select a new edit. Do not cut, skip, reorder, or retime the already-edited visual content; write commentary for the edited beginning, middle, and ending as it appears.\n"
                "- Set video_speed to 1.0 and speed_reason to an empty string for final commentary blocks because the intermediate edit already baked in source speed decisions.\n"
                "- You must carefully review every keyframe and timestamp, combine the keyframe content with its exact timestamp, and also consider the full-source background summary before writing narration.\n"
                "- Write the commentary from the edited-video visual evidence first, then use the remapped transcript only as supporting context. Do not let transcript text override what the edited frame at that timestamp shows.\n"
                "- For every non-pause block, describe the visible action, objects, tools, people, movement, state change, and result shown at that block's evidence_timestamps. The chosen style changes tone, not the timestamp match.\n"
                "- Do not fabricate, do not lie, and do not claim anything that is not supported by the edited keyframes, edited-video visual analysis, remapped transcript, or full-source background summary.\n"
                "- Do not intentionally shorten the explanation. Use as many clear, natural sentences as needed for the selected visual range, while keeping the narration speakable inside that block's duration.\n"
                "- All edit_segments and narration_blocks must use timestamps from the edited intermediate video timeline."
            )
        if visual_analysis:
            visual_analysis_text = _openai_visual_analysis_prompt_text(visual_analysis)
            timestamped_frame_table = _openai_timestamped_frame_table(visual_analysis)
    else:
        visual_instruction = (
            "- Attached images, if present, are sampled keyframes. Treat them as lightweight visual context, "
            "not as the full source video."
        )
    if openai_candidate_edit_plan:
        plan_blocks = openai_candidate_edit_plan.get("blocks") or []
        if plan_blocks:
            block_count = len(plan_blocks)
            target_block_seconds = (
                float(openai_candidate_edit_plan.get("playable_seconds") or target_seconds) / max(1, block_count)
            )

    openai_visual_section = f"""
OPENAI-COMPATIBLE MULTIMODAL VISUAL TIMELINE:
{visual_analysis_text}
""" if visual_analysis_text else ""
    timestamped_frame_section = f"""
TIMESTAMPED VISUAL FRAME TABLE:
{timestamped_frame_table}
""" if timestamped_frame_table else ""
    locked_edit_plan_text = _openai_candidate_edit_plan_prompt_text(openai_candidate_edit_plan)
    locked_edit_plan_section = f"""
BACKEND-CALCULATED EDIT PLAN FROM AI VISUAL CANDIDATES:
{locked_edit_plan_text}
""" if locked_edit_plan_text else ""
    locked_edit_plan_rule = (
        f"- In OpenAI-compatible mode, when BACKEND-CALCULATED EDIT PLAN is provided, use exactly those plan blocks and exactly their start, end, video_speed, speed_reason, visual_facts, and evidence_timestamps. Do not add, remove, merge, split, or retime plan blocks. Your job is to write narration that matches each locked visual range. {_locked_plan_block_guidance_text()}"
        if locked_edit_plan_text
        else ""
    )
    production_timeline_rule = (
        "- Treat narration_blocks as the production timeline: each block's start/end is the edited-video range that will play while that exact block's narration is spoken."
        if is_edited_video_commentary
        else "- Treat narration_blocks as the production timeline: each block's start/end is the source-video range that will play while that exact block's narration is spoken."
    )
    openai_priority_rule = (
        "- In OpenAI-compatible mode, final narration must follow the already-edited video timeline. Use the full-source analysis only as background context; do not select additional source content."
        if is_edited_video_commentary
        else "- In OpenAI-compatible mode, final edit selection must prioritize important, watchable source content. Use candidate_segments/observations with importance, interest_score, edit_value, keep_candidate, payoff, process progress, or result evidence; do not select random-looking filler just to satisfy duration."
    )
    full_coverage_rule = (
        f"- For TARGET DURATION full, narration_blocks must cover the already-edited video from 0.0s through about {duration:.1f}s and must cover the same ranges as edit_segments."
        if is_edited_video_commentary
        else f"- For TARGET DURATION full, narration_blocks must cover about {int(target_seconds)} seconds of selected visuals across the complete source timeline and must cover the same ranges as edit_segments; do not create narration for ranges that are not kept."
    )
    speed_decision_rule = (
        '- For TARGET DURATION full, set video_speed to 1.0 and speed_reason to "" for final commentary blocks because source speed changes were already baked into the edited intermediate video.'
        if is_edited_video_commentary
        else '- For TARGET DURATION full, decide video_speed from the actual visible action, not from a fixed rule. Use 1.0 for key reveals, removal moments, packaging/closure, readable text, final results, completed states, tests, installations, and payoff shots. Use moderate speeds such as 1.15-1.5 for slow but still useful process footage; use 1.75-2.5 only when the range is clearly repetitive, waiting, walking, setup, repeated tool operation, transport, or transition footage and remains understandable after acceleration. If the provided visual candidate evidence marks a kept range as slow/repetitive/waiting/transition or includes suggested_speed > 1.0, either set video_speed above 1.0 with a concrete speed_reason or shorten/cut that range; do not leave long slow filler at 1.0 without a visual reason. Every block with video_speed > 1.0 must include a concrete speed_reason tied to that exact visual range; blocks at 1.0 can use speed_reason "".'
    )

    return f"""You are an expert video essay writer and short-form commentary producer.

TASK:
Transform the source YouTube video into an original commentary/remix narration script.

SOURCE VIDEO:
Title: {video_title}
Duration seconds: {duration:.1f}
Detected transcript language: {transcript.get('language', 'unknown')}

OUTPUT LANGUAGE: {language}
COMMENTARY STYLE: {style}
CUSTOM STYLE PROMPT: {custom_style_prompt or ""}
TARGET DURATION: {_target_duration_hint(target_duration, duration, target_seconds=target_seconds)}

SOURCE TRANSCRIPT:
{transcript_text}

TIMESTAMPED SAMPLE SEGMENTS:
{json.dumps(sampled_segments, ensure_ascii=False)}

SOURCE AUDIO COMMENTARY TIMELINE:
{source_commentary_timeline or "No usable source commentary transcript was available."}

OPENAI-COMPATIBLE SOURCE AUDIO ANALYSIS:
{source_audio_analysis_text or "No direct OpenAI-compatible source audio analysis was available; use transcript fallback when present."}

VISUAL CONTEXT:
{visual_instruction}
{timestamped_frame_section}
{openai_visual_section}
{locked_edit_plan_section}
RULES:
{_banned_phrase_instruction()}
- Do not merely translate the transcript.
- Rewrite it as an original, natural commentary narration.
- Preserve the important facts, sequence, and context from the source.
- The source transcript/timeline and optional source audio analysis describe the original video's spoken commentary. Use them as timestamped context together with the visuals, especially when the original video already explains what is happening. Do not copy it verbatim; write a new commentary matched to the selected visuals.
- Original spoken commentary must be replaced in the final video. Do not rely on old source narration remaining audible; any needed explanation from source audio must be rewritten into the new narration.
{"- This source appears to contain substantial original spoken commentary; use SOURCE AUDIO COMMENTARY TIMELINE to understand the corresponding visual moments before writing the new narration." if source_commentary_available else ""}
{cut_selection_instruction}
{chronological_instruction}
- The narration must match the selected visual ranges, not the removed parts.
{production_timeline_rule}
- In OpenAI-compatible mode, evidence_timestamps are mandatory for non-pause blocks and must be copied from TIMESTAMPED VISUAL FRAME TABLE. Each evidence timestamp must fall within that block's start/end range.
- In OpenAI-compatible mode, every non-pause block must be justified by its keyframe timestamps plus the overall keyframe analysis. Never write a claim just because it sounds plausible; only write what the keyframes, timestamped visual analysis, or transcript support.
{openai_priority_rule}
- Never put source timestamps, frame labels, evidence labels, visual_facts strings, or compact English visual-analysis text into narration or subtitles. Bad examples: "324s:Mencuttingandha", "112.324s: Men cutting...", "keep_candidate", "edit_value", "AI-selected subrange". Use those fields only as hidden evidence and rewrite the spoken narration as clean natural {language} sentences.
{locked_edit_plan_rule}
- Do not describe a visual before it appears or after it has already passed; if a sentence mentions a machine action, material state, worker movement, comparison, or joke, it must belong to that same block's visible time range.
- Do not fabricate, do not lie, and do not claim unseen actions, causes, outcomes, quantities, identities, danger, or completion states. 不允许造假，不允许说谎；看不清就写不确定或只描述可见外观。
- Any concrete action, state change, result, object identity, quantity, danger, or completion claim must be supported by that same block's timestamped visual evidence or transcript evidence. Do not move a claim onto a later loose close-up, aftermath shot, earlier setup, or unrelated range that does not show the described content.
- Keep each block self-contained: first ground the viewer in the concrete visible action, then add interpretation or commentary for that exact action.
- {style_grounding}{custom_style_instruction}
{timeline_rules}
- For TARGET DURATION full, if the source has a final payoff, result reveal, before/after comparison, effect showcase, completed product, or conclusion, include the visual range where that result actually appears and let it play through.
- For TARGET DURATION full, the selected blocks must not stop in the first half of a long source; at least one narration_blocks item must end after {int(duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION)} seconds.
- For TARGET DURATION full, narration_blocks is required: output exactly {block_count} chronological blocks. Each block must have start, end, visual, narration, pause, rate, pitch, video_speed, and speed_reason.
- For TARGET DURATION full, every non-pause block should also include visual_facts and evidence_timestamps when the model can infer them from the visual timeline; use these fields to prove the narration is grounded in that exact source range.
- For TARGET DURATION full, aim for about {target_block_seconds:.0f}s playable visuals per block across the whole edit, but keep ordinary narrated blocks usually 8-20s after video_speed so the commentary can clearly explain the selected visuals. Split longer useful ranges into multiple narrated blocks; do not leave a 40-60s narrated block with a short paragraph.
{full_coverage_rule}
- For TARGET DURATION full, most selected visual blocks should contain narration that explains the on-screen content clearly. Use pause=true only when the footage, process sound, reveal, or visual proof is better understood without speech.
- For TARGET DURATION full, use pause=true blocks when the original footage genuinely needs to be heard without commentary: key reveals, machine/process sounds, skilled hand work, visual proof, emotional beats, transitions, or moments where the picture explains itself. Pause blocks must leave narration empty and should usually last 2-12 seconds.
{pause_audio_instruction}
- For TARGET DURATION full, each non-pause block's narration must be speakable inside that block's visual duration; do not put 2 minutes of words into a 20-second visual range.
- For TARGET DURATION full, keep normal narrated blocks concrete and clear for their visible action. Each ordinary narrated process/action block should state what is visible, what changes, and why that moment matters.
- For TARGET DURATION full, do not make narration sparse by writing one vague sentence over a long visual block. If a non-pause block plays 12+ seconds, write enough scene-matched commentary to make the visible action clear; if there is not enough to say, shorten that timestamp range or use a brief pause=true block.
- For TARGET DURATION full, each non-pause block's narration must match only that block's visible range. Use enough natural sentences to name concrete objects, actions, state changes, comparisons, results, or risks visible in that range.
- For TARGET DURATION full, {block_sync_instruction}
{openai_one_shot_sync_instruction}
- For TARGET DURATION full, if a selected visual range is too long or too visually sparse for a high-quality natural commentary paragraph, redesign the block: shorten the range, split it, or use a brief pause=true moment where the timestamped visual or original audio carries meaning. The renderer preserves selected source ranges and will not tighten trailing footage after short narration. Do not add meaningless word padding, and do not pad it with meaningless words.
- For TARGET DURATION full, use rate to create cadence: slower values like "-10%" for important reveals or emotional emphasis, faster values like "+12%" for energetic process sections. Valid range: "-30%" to "+30%".
- For TARGET DURATION full, use pitch lightly for tone: lower values like "-3Hz" for weight, higher values like "+3Hz" for excitement. Valid range: "-15Hz" to "+15Hz".
{speed_decision_rule}
- For TARGET DURATION full, vary rate and pitch across blocks; do not leave every non-pause block at "+0%" and "+0Hz".
- For TARGET DURATION full, total narration must be at most {max_chars} non-whitespace characters so the voiceover does not exceed the selected visuals. There is no total minimum word count, but do not intentionally under-explain visible action just to be short.
- For TARGET DURATION full, decide whether the completed commentary should also be released as continuous episodes. Set episode_plan.should_split=true only when the source has natural chapters, process stages, tutorial steps, story progression, interview sections, or a long timeline that benefits from serial viewing.
- For TARGET DURATION full, if should_split=true, return episodes that cover consecutive narration_blocks in order. Each episode must use 1-based start_block and end_block indexes from narration_blocks; do not split inside a block, overlap episodes, or skip blocks between episodes unless that block is an intentional bridge better left only in the full video.
- For TARGET DURATION full, if the source is short or does not have clear episode boundaries, set episode_plan.should_split=false and episodes=[]; the complete commentary video will still be generated.
- The top-level narration must be the complete voiceover text, preferably the non-pause narration_blocks joined in order.
- The opening must follow the 3-second retention rule: the first sentence and the first narration block must immediately create curiosity, contrast, stakes, surprise, or payoff expectation while still matching the first visible action.
- Add a strong hook in the first sentence; do not start with generic setup, greetings, or slow background explanation.
- The top-level title must clearly say what the video is doing: name the concrete subject, process/action, and result or purpose. Use titles like "废旧电机拆解回收铜线全过程" instead of vague hype titles like "震撼工厂全过程" or "不可思议的改造".
- Make it suitable for a video with the original footage under the narration.
- Avoid claiming things that are not supported by the transcript.
- If the original video contains a speaker narration, assume it will be replaced by this new voiceover; keep only useful visual context and ambient sound.
- If output language is zh, write fluent Simplified Chinese.
- Return valid JSON only.

{json_output_contract}

JSON FORMAT:
{{
  "title": "specific title that says what the video does",
  "summary": "brief summary of the source video",
  "hook": "opening hook",
  "narration": "full voiceover narration text",
  "narration_blocks": [
    {{"start": 0, "end": 30, "visual": "what is visible in this range", "visual_facts": ["concrete visible fact from this range"], "evidence_timestamps": [3.0, 12.0], "narration": "voiceover for this visual range", "pause": false, "rate": "+0%", "pitch": "+0Hz", "video_speed": 1.0, "speed_reason": ""}},
    {{"start": 30, "end": 45, "visual": "slow repeated setup or movement", "visual_facts": ["why it can remain understandable when accelerated"], "evidence_timestamps": [35.0, 42.0], "narration": "short commentary for this sped-up range", "pause": false, "rate": "+6%", "pitch": "+1Hz", "video_speed": 1.5, "speed_reason": "visible setup/repeated motion is slow, so 1.5x keeps the process without dragging"}},
    {{"start": 45, "end": 55, "visual": "original footage moment that should breathe", "visual_facts": ["why the moment should play with ambient sound"], "evidence_timestamps": [50.0], "narration": "", "pause": true, "rate": "+0%", "pitch": "+0Hz", "video_speed": 1.0, "speed_reason": ""}}
  ],
  "episode_plan": {{"should_split": true, "reason": "why this full commentary benefits from serialized episodes, or why not"}},
  "episodes": [
    {{"episode_number": 1, "title": "第1集：specific episode title", "summary": "what this episode covers", "start_block": 1, "end_block": 4}}
  ],
  "edit_segments": [
    {{"start": 0, "end": 30, "reason": "why this visual part should be kept and what redundant source time is skipped around it"}}
  ],
  "cut_strategy": [
    {{"removed_range": "30-90", "reason": "repetitive or low-value material removed from the final edit"}}
  ],
  "chapters": [
    {{"start": 0, "end": 30, "title": "chapter title", "narration": "chapter narration summary"}}
  ],
  "hashtags": ["#tag"]
}}
"""


def _build_frame_analysis_contents(prompt: str, frame_paths: Optional[List[str]]):
    inline_parts = []
    for frame_path in (frame_paths or [])[:6]:
        try:
            with open(frame_path, "rb") as f:
                inline_parts.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))
        except Exception:
            pass
    if inline_parts:
        return [types.Content(role="user", parts=inline_parts + [types.Part.from_text(text=prompt)])]
    return [prompt]


def _openai_chat_completions_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        raise Exception("OpenAI-compatible Base URL is required for OpenAI commentary analysis mode")
    if re.search(r"/chat/completions/?$", normalized):
        return normalized
    if re.search(r"/(v\d+|compatible-mode/v\d+|api/v\d+)$", normalized):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _openai_image_content_part(image_path: str) -> Dict:
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encoded}",
            "detail": OPENAI_IMAGE_DETAIL,
        },
    }


def _extract_openai_response_text(data: Dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise Exception("OpenAI-compatible API returned no choices")
    first = choices[0] or {}
    message = first.get("message") or {}
    content = message.get("content", first.get("text", ""))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "output_text"} and item.get("text"):
                    parts.append(str(item.get("text")))
                elif item.get("content"):
                    parts.append(str(item.get("content")))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _call_openai_compatible_chat(
    api_key: str,
    base_url: str,
    model: str,
    messages: List[Dict],
    max_tokens: int,
    timeout_seconds: Optional[int] = None,
    response_format: Optional[Dict] = None,
) -> str:
    if not api_key:
        raise Exception("OpenAI-compatible API Key is required for OpenAI commentary analysis mode")
    if not model:
        raise Exception("OpenAI-compatible model is required for OpenAI commentary analysis mode")
    url = _openai_chat_completions_url(base_url)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": OPENAI_TEMPERATURE,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_timeout = timeout_seconds or OPENAI_REQUEST_TIMEOUT_SECONDS
    response = None
    last_error = None
    response_format_downgraded = False
    response_format_removed = False
    for attempt in range(1, OPENAI_REQUEST_RETRIES + 1):
        try:
            with httpx.Client(timeout=request_timeout, follow_redirects=True) as client:
                response = client.post(url, headers=headers, json=payload)
        except Exception as exc:
            last_error = exc
            if attempt >= OPENAI_REQUEST_RETRIES:
                raise Exception(f"OpenAI-compatible API request failed after {attempt} attempts: {exc}") from exc
            time.sleep(min(2 ** attempt, 8))
            continue
        if response.status_code < 400:
            break
        if (
            response_format
            and not response_format_downgraded
            and response.status_code in {400, 422}
            and "response_format" in response.text
        ):
            if response_format.get("type") == "json_schema":
                payload["response_format"] = {"type": "json_object"}
            else:
                payload.pop("response_format", None)
            response_format_downgraded = True
            continue
        if (
            response_format_downgraded
            and not response_format_removed
            and response.status_code in {400, 422}
            and "response_format" in response.text
        ):
            payload.pop("response_format", None)
            response_format_removed = True
            continue
        if response.status_code < 500 and response.status_code != 429:
            raise Exception(
                "OpenAI-compatible API returned an error: "
                f"HTTP {response.status_code} {response.text[:1200]}"
            )
        if attempt >= OPENAI_REQUEST_RETRIES:
            raise Exception(
                "OpenAI-compatible API returned an error after "
                f"{attempt} attempts: HTTP {response.status_code} {response.text[:1200]}"
            )
        time.sleep(min(2 ** attempt, 8))
    if response is None:
        raise Exception(f"OpenAI-compatible API request failed: {last_error}")
    try:
        data = response.json()
    except Exception as exc:
        raise Exception(f"OpenAI-compatible API returned non-JSON response: {response.text[:1200]}") from exc
    text = _extract_openai_response_text(data)
    if not text:
        raise Exception("OpenAI-compatible API returned empty response text")
    return text


def _select_openai_uniform_frame_samples(duration: float, sampling_options: Optional[Dict] = None) -> List[Dict]:
    if duration <= 0:
        return []

    options = resolve_openai_sampling_options(**(sampling_options or {}))
    frame_count = min(options["max_frames"], max(1, math.ceil(duration / options["frame_interval_seconds"])))
    samples = []
    for index in range(frame_count):
        timestamp = min(max(0.0, duration * (index + 0.5) / frame_count), max(0.0, duration - 0.05))
        samples.append({"timestamp": round(timestamp, 3), "sample_role": "uniform"})
    return samples


def _scene_timecode_seconds(value) -> float:
    if hasattr(value, "get_seconds"):
        return float(value.get_seconds())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _detect_openai_sampling_scenes(
    video_path: str,
    duration: float,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    if duration <= 0:
        return []
    try:
        scene_list, _fps = detect_scenes(video_path)
    except Exception as exc:
        if progress:
            progress(f"Scene detection failed for OpenAI-compatible analysis; falling back to uniform sampling: {exc}")
        return []

    scenes = []
    for index, scene in enumerate(scene_list or [], start=1):
        start_tc, end_tc = scene
        start = max(0.0, min(duration, _scene_timecode_seconds(start_tc)))
        end = max(0.0, min(duration, _scene_timecode_seconds(end_tc)))
        if end - start < OPENAI_SCENE_MIN_SECONDS:
            continue
        scenes.append({
            "scene_index": index,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        })

    return scenes


def _estimate_scene_motion_score(video_path: str, start: float, end: float) -> float:
    if end <= start:
        return 0.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    try:
        duration = end - start
        sample_count = 3 if duration >= 3.0 else 2
        timestamps = [start + duration * (index + 1) / (sample_count + 1) for index in range(sample_count)]
        previous = None
        diffs = []
        for timestamp in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (96, 54))
            if previous is not None:
                diffs.append(float(cv2.absdiff(previous, gray).mean()) / 255.0)
            previous = gray
        return round(sum(diffs) / len(diffs), 4) if diffs else 0.0
    finally:
        cap.release()


def _openai_scene_keyframe_count(scene_duration: float, motion_score: float, sampling_options: Optional[Dict] = None) -> int:
    options = resolve_openai_sampling_options(**(sampling_options or {}))
    if options["scene_max_keyframes"] <= 1 or scene_duration < 4.0:
        return 1
    base_count = max(1, math.ceil(scene_duration / options["frame_interval_seconds"]))
    if motion_score >= OPENAI_SCENE_DYNAMIC_MOTION_THRESHOLD:
        base_count = math.ceil(base_count * 1.5)
    elif motion_score < OPENAI_SCENE_STATIC_MOTION_THRESHOLD:
        base_count = max(1, math.ceil(base_count * 0.5))
    return min(options["scene_max_keyframes"], base_count)


def _openai_scene_sample_timestamps(scene: Dict, keyframe_count: int) -> List[Dict]:
    start = float(scene["start"])
    end = float(scene["end"])
    scene_duration = max(0.0, end - start)
    if scene_duration <= 0:
        return []
    if keyframe_count <= 1:
        return [{"timestamp": start + scene_duration * 0.5, "sample_role": "middle"}]

    samples = []
    for index in range(keyframe_count):
        position = (index + 0.5) / keyframe_count
        if position < 0.25:
            role = "early"
        elif position > 0.75:
            role = "late"
        else:
            role = "middle"
        samples.append({"timestamp": start + scene_duration * position, "sample_role": role})
    return samples


def _cap_openai_scene_samples(samples: List[Dict], max_frames: int) -> List[Dict]:
    if len(samples) <= max_frames:
        return samples

    representatives = {}
    extras = []
    for sample in samples:
        scene_index = sample.get("scene_index")
        current = representatives.get(scene_index)
        if current is None:
            representatives[scene_index] = sample
        elif sample.get("sample_role") == "middle" and current.get("sample_role") != "middle":
            extras.append(current)
            representatives[scene_index] = sample
        else:
            extras.append(sample)

    kept = sorted(representatives.values(), key=lambda item: item["timestamp"])
    if len(kept) >= max_frames:
        step = len(kept) / max_frames
        return [kept[min(len(kept) - 1, int(index * step))] for index in range(max_frames)]

    extras = sorted(extras, key=lambda item: (-float(item.get("motion_score") or 0.0), item["timestamp"]))
    kept.extend(extras[:max_frames - len(kept)])
    return sorted(kept, key=lambda item: item["timestamp"])


def _select_openai_scene_aware_frame_samples(
    video_path: str,
    duration: float,
    progress: Optional[Callable[[str], None]] = None,
    sampling_options: Optional[Dict] = None,
) -> List[Dict]:
    scenes = _detect_openai_sampling_scenes(video_path, duration, progress=progress)
    if not scenes:
        return []

    samples = []
    for scene in scenes:
        motion_score = _estimate_scene_motion_score(video_path, scene["start"], scene["end"])
        keyframe_count = _openai_scene_keyframe_count(scene["duration"], motion_score, sampling_options=sampling_options)
        for sample in _openai_scene_sample_timestamps(scene, keyframe_count):
            timestamp = min(max(0.0, sample["timestamp"]), max(0.0, duration - 0.05))
            samples.append({
                "timestamp": round(timestamp, 3),
                "scene_index": scene["scene_index"],
                "scene_start": scene["start"],
                "scene_end": scene["end"],
                "scene_duration": scene["duration"],
                "sample_role": sample["sample_role"],
                "motion_score": motion_score,
            })

    options = resolve_openai_sampling_options(**(sampling_options or {}))
    return _cap_openai_scene_samples(sorted(samples, key=lambda item: item["timestamp"]), options["max_frames"])


def _openai_frames_manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, OPENAI_ANALYSIS_FRAMES_MANIFEST)


def _load_openai_analysis_frames(
    output_dir: str,
    sampling_options: Optional[Dict] = None,
    source_video_path: Optional[str] = None,
    require_uniform_coverage: bool = False,
    duration: Optional[float] = None,
) -> List[Dict]:
    expected_options = resolve_openai_sampling_options(**(sampling_options or {}))
    manifest_path = _openai_frames_manifest_path(output_dir)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        expected_source = os.path.abspath(source_video_path) if source_video_path else None
        manifest_source = data.get("source_video_path")
        source_matches = not expected_source or manifest_source in {None, expected_source}
        coverage_matches = True
        if require_uniform_coverage:
            coverage_matches = data.get("sampling_mode") == "uniform"
        if data.get("sampling_options") == expected_options and source_matches and coverage_matches:
            frames = data.get("frames") or []
            if isinstance(frames, list) and frames and all(
                isinstance(frame, dict) and frame.get("path") and os.path.exists(frame["path"])
                for frame in frames
            ):
                if require_uniform_coverage and not _openai_frames_have_uniform_coverage(
                    frames,
                    float(duration or 0.0),
                    expected_options["frame_interval_seconds"],
                ):
                    return []
                return frames

    frames_dir = os.path.join(output_dir, "openai_analysis_frames")
    if not os.path.isdir(frames_dir):
        return []
    frames = []
    for filename in sorted(os.listdir(frames_dir)):
        match = re.match(r"^frame_\d+_(\d+)\.jpg$", filename)
        if not match:
            continue
        frame_path = os.path.join(frames_dir, filename)
        if not os.path.exists(frame_path):
            continue
        frames.append({"path": frame_path, "timestamp": round(int(match.group(1)) / 1000.0, 3)})
    if frames and not require_uniform_coverage:
        _save_openai_analysis_frames(
            output_dir,
            frames,
            sampling_options=sampling_options,
            source_video_path=source_video_path,
            sampling_mode="legacy",
        )
    if frames and require_uniform_coverage and not _openai_frames_have_uniform_coverage(
        frames,
        float(duration or 0.0),
        expected_options["frame_interval_seconds"],
    ):
        return []
    return frames


def _save_openai_analysis_frames(
    output_dir: str,
    frames: List[Dict],
    sampling_options: Optional[Dict] = None,
    source_video_path: Optional[str] = None,
    sampling_mode: str = "unknown",
) -> None:
    manifest_path = _openai_frames_manifest_path(output_dir)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({
            "sampling_options": resolve_openai_sampling_options(**(sampling_options or {})),
            "source_video_path": os.path.abspath(source_video_path) if source_video_path else None,
            "sampling_mode": sampling_mode,
            "frames": frames,
        }, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, manifest_path)


def _openai_frames_have_uniform_coverage(
    frames: List[Dict],
    duration: float,
    frame_interval_seconds: float,
) -> bool:
    timestamps = sorted(
        round(float(frame.get("timestamp") or 0.0), 3)
        for frame in frames or []
        if isinstance(frame, dict)
    )
    if not timestamps:
        return False
    if duration <= 0:
        return True
    max_gap = max(OPENAI_MAX_FRAME_GAP_SECONDS, float(frame_interval_seconds or OPENAI_FRAME_INTERVAL_SECONDS) * 2.5)
    previous = 0.0
    for timestamp in timestamps:
        if timestamp - previous > max_gap:
            return False
        previous = timestamp
    return max(0.0, float(duration) - timestamps[-1]) <= max_gap


def _load_openai_visual_analysis(
    output_dir: str,
    model: str,
    frame_infos: List[Dict],
    sampling_options: Optional[Dict] = None,
    transcript: Optional[Dict] = None,
    source_audio_analysis: Optional[Dict] = None,
) -> Optional[Dict]:
    cache_path = _openai_visual_analysis_cache_path(output_dir)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if data.get("model") != model:
        return None
    if data.get("frame_count") != len(frame_infos):
        return None
    if data.get("sampling_options") != resolve_openai_sampling_options(**(sampling_options or {})):
        return None
    source_paths = {
        os.path.abspath(frame.get("source_video_path"))
        for frame in frame_infos
        if isinstance(frame, dict) and frame.get("source_video_path")
    }
    cached_source = data.get("source_video_path")
    if source_paths and cached_source != next(iter(source_paths)):
        return None
    expected_commentary_available = bool(_format_source_commentary_timeline(transcript))
    if expected_commentary_available and not data.get("source_commentary_available"):
        return None
    expected_audio_analysis_available = bool(_format_source_audio_analysis_timeline(source_audio_analysis))
    if expected_audio_analysis_available and not data.get("source_audio_analysis_available"):
        return None
    if not _openai_visual_analysis_has_edit_value_scores(data):
        return None
    return data


def _save_openai_visual_analysis(output_dir: str, visual_analysis: Dict) -> str:
    cache_path = _openai_visual_analysis_cache_path(output_dir)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(visual_analysis, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, cache_path)
    return cache_path


def _extract_openai_analysis_frames(
    video_path: str,
    output_dir: str,
    duration: float,
    progress: Optional[Callable[[str], None]] = None,
    sampling_options: Optional[Dict] = None,
    extraction_video_path: Optional[str] = None,
    force_uniform: bool = False,
) -> List[Dict]:
    frames_dir = os.path.join(output_dir, "openai_analysis_frames")
    os.makedirs(frames_dir, exist_ok=True)
    analysis_video_path = extraction_video_path or video_path
    cached_frames = _load_openai_analysis_frames(
        output_dir,
        sampling_options=sampling_options,
        source_video_path=analysis_video_path,
        require_uniform_coverage=force_uniform,
        duration=duration,
    )
    if cached_frames:
        if progress:
            progress(f"Reusing OpenAI-compatible analysis frames: {len(cached_frames)}")
        return cached_frames
    if duration <= 0:
        return []

    samples = []
    use_scene_aware = not force_uniform and OPENAI_SCENE_AWARE_SAMPLING and (
        OPENAI_SCENE_AWARE_MAX_DURATION_SECONDS <= 0
        or duration <= OPENAI_SCENE_AWARE_MAX_DURATION_SECONDS
    )
    if use_scene_aware:
        if progress:
            progress("Detecting scenes for OpenAI-compatible scene-aware frame sampling on 360p analysis video...")
        samples = _select_openai_scene_aware_frame_samples(
            analysis_video_path,
            duration,
            progress=progress,
            sampling_options=sampling_options,
        )
    elif not force_uniform and OPENAI_SCENE_AWARE_SAMPLING and progress:
        progress(
            "Skipping OpenAI-compatible scene-aware sampling for long source; "
            f"duration {duration:.1f}s exceeds {OPENAI_SCENE_AWARE_MAX_DURATION_SECONDS:.1f}s."
        )
    if not samples:
        if progress and (force_uniform or OPENAI_SCENE_AWARE_SAMPLING):
            progress("Using uniform OpenAI-compatible frame sampling fallback...")
        samples = _select_openai_uniform_frame_samples(duration, sampling_options=sampling_options)
    sampling_mode = "scene_aware" if use_scene_aware and samples else "uniform"

    frames = []
    failures = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        timestamp = float(sample["timestamp"])
        frame_path = os.path.join(frames_dir, f"frame_{index:04d}_{int(timestamp * 1000):09d}.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{timestamp:.3f}",
            "-i", analysis_video_path,
            "-frames:v", "1",
            "-q:v", "4",
            frame_path,
        ]
        try:
            _run_command(cmd)
            if os.path.exists(frame_path):
                frame_info = {
                    "path": frame_path,
                    "timestamp": round(timestamp, 3),
                    "source_video_path": os.path.abspath(analysis_video_path),
                }
                for key in (
                    "scene_index",
                    "scene_start",
                    "scene_end",
                    "scene_duration",
                    "sample_role",
                    "motion_score",
                ):
                    if key in sample:
                        frame_info[key] = sample[key]
                frames.append(frame_info)
        except Exception as exc:
            failures.append(str(exc).strip())
            if progress and len(failures) == 1:
                progress(f"OpenAI-compatible frame extraction failed at {timestamp:.3f}s: {str(exc).strip()[:800]}")
            continue
        if progress and (index == total or index % 10 == 0):
            progress(f"Extracted OpenAI-compatible analysis frames: {index}/{total}")
    if not frames and failures:
        raise Exception(f"OpenAI-compatible analysis mode could not extract video frames: {failures[0]}")
    if frames:
        _save_openai_analysis_frames(
            output_dir,
            frames,
            sampling_options=sampling_options,
            source_video_path=analysis_video_path,
            sampling_mode=sampling_mode,
        )
    return frames


def _openai_visual_batch_prompt(
    video_title: str,
    duration: float,
    frames: List[Dict],
    batch_index: int,
    total_batches: int,
    source_commentary_timeline: str = "",
    source_audio_analysis_timeline: str = "",
) -> str:
    labels = [
        {
            "image": idx + 1,
            "timestamp": frame["timestamp"],
            "scene_index": frame.get("scene_index"),
            "scene_start": frame.get("scene_start"),
            "scene_end": frame.get("scene_end"),
            "sample_role": frame.get("sample_role"),
            "motion_score": frame.get("motion_score"),
        }
        for idx, frame in enumerate(frames)
    ]
    return f"""You are analyzing timestamped frames from a source video for a commentary remix.

SOURCE VIDEO:
Title: {video_title}
Duration seconds: {duration:.1f}
Batch: {batch_index}/{total_batches}
Frame timestamp and scene labels: {json.dumps(labels, ensure_ascii=False)}

SOURCE AUDIO COMMENTARY TIMELINE:
{source_commentary_timeline or "No usable source commentary transcript was available for this batch."}

OPENAI-COMPATIBLE SOURCE AUDIO ANALYSIS TIMELINE:
{source_audio_analysis_timeline or "No direct OpenAI-compatible source audio analysis was available for this batch."}

TASK:
Analyze the visual action in these frames. Return valid JSON only.

RULES:
- Use the timestamp labels as source-video timestamps.
- For every image, return one observation with that image number and the exact timestamp from the labels. Do not omit a frame just because it looks ordinary; ordinary frames still describe the timeline.
- Carefully inspect every keyframe and its timestamp. You must combine the visual content of each keyframe, that keyframe's exact timestamp, and the overall keyframe timeline analysis.
- Use scene_start/scene_end metadata as scene boundaries when present.
- Treat early/middle/late sample roles as positions inside one detected scene.
- Analyze the visuals in detail: describe concrete visible actions, tools, materials, people, hand/foot movement, object state changes, scene changes, reveals, and process stages.
- Separate evidence from uncertainty. If a material or object is ambiguous, describe its appearance instead of forcing a specific label.
- Do not invent facts that are not visible. 不允许造假，不允许说谎；看不清就写不确定或只描述可见外观。
- Mark frames/ranges that look valuable for a commentary edit, and distinguish must-keep payoff/action from slow-but-useful or low-value footage.
- For every observation and candidate segment, provide editing value scores: importance 1-5 for story/process necessity and interest_score 1-5 for visual watchability/appeal. These scores are required editing evidence, not decoration.
- keep_candidate=true only when the frame or range contains important, good-looking, story-progressing, surprising, skilled, risky, result-revealing, or visually clear content. Do not mark random filler, waiting, camera drift, repeated setup, or unclear footage as useful.
- Candidate segments must include evidence_timestamps copied from the frame labels inside that segment, plus a specific reason explaining why this range is important or watchable enough for the final cut.
- For each useful candidate range, judge from the visible motion whether it should play at normal speed or can be accelerated. This is only visual evidence for later AI editing; do not use a fixed duration rule.
- Keep observations specific enough to ground narration later; this detailed visual analysis is evidence for editing and for matching commentary to exact timestamps.
- If SOURCE AUDIO COMMENTARY TIMELINE or OPENAI-COMPATIBLE SOURCE AUDIO ANALYSIS TIMELINE is available, use it only as timestamped context for what the original narrator says near these frames. Combine it with the visual evidence to understand the scene, but do not copy the original narration into the new commentary.

JSON FORMAT:
{{
  "batch_index": {batch_index},
  "observations": [
    {{"image": 1, "timestamp": 12.3, "visual": "what is visible", "process_stage": "stage name", "importance": 1, "interest_score": 1, "keep_candidate": true, "pace": "normal|slow|repetitive|waiting|transition", "edit_value": "must_keep|useful|skippable"}}
  ],
  "candidate_segments": [
    {{"start": 10.0, "end": 25.0, "reason": "why this important/watchable visual range should be kept", "importance": 4, "interest_score": 4, "edit_value": "must_keep|useful|skippable", "evidence_timestamps": [12.3], "suggested_speed": 1.0, "speed_reason": "visible action needs normal speed or can remain understandable accelerated"}}
  ]
}}
"""


def _build_openai_visual_batch_messages(
    prompt: str,
    frames: List[Dict],
) -> List[Dict]:
    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "text", "text": f"Source timestamp: {frame['timestamp']:.3f} seconds"})
        content.append(_openai_image_content_part(frame["path"]))
    return [{"role": "user", "content": content}]


def _parse_openai_json(text: str) -> Dict:
    parsed = json.loads(_clean_json_text(text))
    if isinstance(parsed, dict):
        return parsed
    return {"items": parsed}


def _nearest_openai_frame(frame_infos: List[Dict], timestamp: float) -> Optional[Dict]:
    if not frame_infos:
        return None
    try:
        target = float(timestamp)
    except (TypeError, ValueError):
        return frame_infos[0]
    return min(frame_infos, key=lambda frame: abs(float(frame.get("timestamp") or 0.0) - target))


def _normalize_openai_visual_batch_result(parsed: Dict, batch: List[Dict]) -> Dict:
    normalized = dict(parsed or {})
    labels = [
        {
            "image": index + 1,
            "timestamp": round(float(frame.get("timestamp") or 0.0), 3),
            "scene_index": frame.get("scene_index"),
            "scene_start": frame.get("scene_start"),
            "scene_end": frame.get("scene_end"),
            "sample_role": frame.get("sample_role"),
        }
        for index, frame in enumerate(batch)
    ]
    normalized["frame_labels"] = labels
    observations = []
    used_frame_indexes = set()
    for index, observation in enumerate(normalized.get("observations") or []):
        if not isinstance(observation, dict):
            continue
        item = dict(observation)
        frame = None
        image_index = item.get("image")
        try:
            image_index = int(image_index)
        except (TypeError, ValueError):
            image_index = None
        if image_index and 1 <= image_index <= len(batch):
            frame = batch[image_index - 1]
        elif item.get("timestamp") is not None:
            frame = _nearest_openai_frame(batch, item.get("timestamp"))
        elif index < len(batch):
            frame = next(
                (candidate for candidate_index, candidate in enumerate(batch) if candidate_index not in used_frame_indexes),
                batch[index],
            )
        if frame:
            used_frame_indexes.add(batch.index(frame))
            item["timestamp"] = round(float(frame.get("timestamp") or 0.0), 3)
            item["frame_timestamp"] = item["timestamp"]
            item["image"] = batch.index(frame) + 1
            for key in ("scene_index", "scene_start", "scene_end", "sample_role", "motion_score"):
                if frame.get(key) is not None and item.get(key) is None:
                    item[key] = frame.get(key)
        observations.append(item)
    normalized["observations"] = observations

    candidate_segments = []
    for segment in normalized.get("candidate_segments") or []:
        if not isinstance(segment, dict):
            continue
        item = dict(segment)
        anchor = None
        for key in ("timestamp", "frame_timestamp", "start"):
            if item.get(key) is not None:
                anchor = item.get(key)
                break
        frame = _nearest_openai_frame(batch, anchor) if anchor is not None else (batch[0] if batch else None)
        if frame:
            frame_ts = float(frame.get("timestamp") or 0.0)
            if item.get("start") is None:
                item["start"] = frame.get("scene_start") if frame.get("scene_start") is not None else max(0.0, frame_ts - 1.5)
            if item.get("end") is None:
                item["end"] = frame.get("scene_end") if frame.get("scene_end") is not None else frame_ts + 1.5
            try:
                start = float(item.get("start"))
                end = float(item.get("end"))
            except (TypeError, ValueError):
                start = max(0.0, frame_ts - 1.5)
                end = frame_ts + 1.5
            evidence = []
            for value in item.get("evidence_timestamps") or []:
                try:
                    candidate_ts = float(value)
                except (TypeError, ValueError):
                    continue
                evidence_frame = _nearest_openai_frame(batch, candidate_ts)
                if not evidence_frame:
                    continue
                evidence_ts = round(float(evidence_frame.get("timestamp") or 0.0), 3)
                if start - 0.35 <= evidence_ts <= end + 0.35 and evidence_ts not in evidence:
                    evidence.append(evidence_ts)
            for candidate_frame in batch:
                candidate_ts = round(float(candidate_frame.get("timestamp") or 0.0), 3)
                if start - 0.35 <= candidate_ts <= end + 0.35 and candidate_ts not in evidence:
                    evidence.append(candidate_ts)
            if not evidence:
                evidence = [round(frame_ts, 3)]
            item["evidence_timestamps"] = evidence[:4]
        candidate_segments.append(item)
    normalized["candidate_segments"] = candidate_segments
    return normalized


def _analyze_openai_visual_timeline(
    frame_infos: List[Dict],
    video_title: str,
    duration: float,
    api_key: str,
    base_url: str,
    model: str,
    progress: Optional[Callable[[str], None]] = None,
    sampling_options: Optional[Dict] = None,
    transcript: Optional[Dict] = None,
    source_audio_analysis: Optional[Dict] = None,
) -> Dict:
    if not frame_infos:
        raise Exception("OpenAI-compatible analysis mode could not extract any video frames")
    options = resolve_openai_sampling_options(**(sampling_options or {}))
    batch_size = options["batch_size"]
    visual_concurrency = min(options["visual_concurrency"], max(1, math.ceil(len(frame_infos) / batch_size)))
    batches = [frame_infos[i:i + batch_size] for i in range(0, len(frame_infos), batch_size)]
    parsed_batches = []
    observations = []
    candidate_segments = []
    source_commentary_timeline = _format_source_commentary_timeline(transcript)
    source_audio_analysis_timeline = _format_source_audio_analysis_timeline(source_audio_analysis)

    def analyze_batch(index: int, batch: List[Dict]) -> Dict:
        _raise_if_commentary_cancelled()
        batch_timestamps = []
        for frame in batch:
            try:
                batch_timestamps.append(float(frame.get("timestamp") or 0.0))
            except (TypeError, ValueError):
                continue
        if batch_timestamps:
            local_source_commentary_timeline = _format_source_commentary_timeline(
                transcript,
                max_segments=60,
                max_chars=7000,
                start=min(batch_timestamps),
                end=max(batch_timestamps),
                margin=20.0,
            )
            local_source_audio_analysis_timeline = _format_source_audio_analysis_timeline(
                source_audio_analysis,
                max_items=40,
                max_chars=5000,
                start=min(batch_timestamps),
                end=max(batch_timestamps),
                margin=20.0,
            )
        else:
            local_source_commentary_timeline = source_commentary_timeline
            local_source_audio_analysis_timeline = source_audio_analysis_timeline
        prompt = _openai_visual_batch_prompt(
            video_title,
            duration,
            batch,
            index,
            len(batches),
            source_commentary_timeline=local_source_commentary_timeline,
            source_audio_analysis_timeline=local_source_audio_analysis_timeline,
        )
        text = _call_openai_compatible_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=_build_openai_visual_batch_messages(prompt, batch),
            max_tokens=OPENAI_VISUAL_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        try:
            parsed = _parse_openai_json(text)
        except Exception:
            parsed = {
                "batch_index": index,
                "raw_analysis": text[:4000],
                "frame_timestamps": [frame["timestamp"] for frame in batch],
            }
        return _normalize_openai_visual_batch_result(parsed, batch)

    if progress:
        progress(
            "OpenAI-compatible multimodal visual analysis "
            f"{len(batches)} batches with concurrency {visual_concurrency}..."
        )
    batch_results = [None] * len(batches)
    context_job_id = _current_commentary_job_id()
    context_cancel_event = _current_commentary_cancel_event()

    def analyze_batch_with_context(index: int, batch: List[Dict]) -> Dict:
        if context_job_id:
            with commentary_job_context(context_job_id, context_cancel_event):
                return analyze_batch(index, batch)
        return analyze_batch(index, batch)

    if visual_concurrency <= 1:
        for index, batch in enumerate(batches, start=1):
            _raise_if_commentary_cancelled()
            if progress:
                progress(f"OpenAI-compatible multimodal visual analysis batch {index}/{len(batches)}...")
            batch_results[index - 1] = analyze_batch_with_context(index, batch)
    else:
        _raise_if_commentary_cancelled()
        with ThreadPoolExecutor(max_workers=visual_concurrency) as executor:
            futures = {
                executor.submit(analyze_batch_with_context, index, batch): index
                for index, batch in enumerate(batches, start=1)
            }
            completed = 0
            for future in as_completed(futures):
                _raise_if_commentary_cancelled()
                index = futures[future]
                batch_results[index - 1] = future.result()
                completed += 1
                if progress:
                    progress(f"OpenAI-compatible multimodal visual analysis batch {index}/{len(batches)} done ({completed}/{len(batches)})")

    for parsed in batch_results:
        if not parsed:
            continue
        parsed_batches.append(parsed)
        observations.extend(parsed.get("observations") or [])
        candidate_segments.extend(parsed.get("candidate_segments") or [])
    scene_indexes = {
        frame.get("scene_index")
        for frame in frame_infos
        if frame.get("scene_index") is not None
    }
    source_video_paths = [
        os.path.abspath(frame.get("source_video_path"))
        for frame in frame_infos
        if frame.get("source_video_path")
    ]
    return {
        "provider": "openai_compatible",
        "model": model,
        "frame_count": len(frame_infos),
        "batch_count": len(batches),
        "source_video_path": source_video_paths[0] if source_video_paths else None,
        "sampling": "scene_aware" if scene_indexes else "uniform",
        "scene_count": len(scene_indexes),
        "sampling_options": options,
        "source_commentary_available": bool(source_commentary_timeline),
        "source_commentary_timeline": source_commentary_timeline,
        "source_audio_analysis_available": bool(source_audio_analysis_timeline),
        "source_audio_analysis_timeline": source_audio_analysis_timeline,
        "frames": _openai_frame_infos_for_prompt(frame_infos),
        "observations": observations,
        "candidate_segments": candidate_segments,
        "batches": parsed_batches,
    }


def _build_openai_regeneration_prompt(
    original_prompt: str,
    invalid_script: Dict,
    validation_error: Exception,
    duration: float,
    target_duration: str,
    language: str,
    attempt: int,
    visual_analysis: Optional[Dict] = None,
    openai_candidate_edit_plan: Optional[Dict] = None,
) -> str:
    target_seconds = (
        _target_seconds_from_validation_error(validation_error)
        or _target_visual_duration_seconds_for_analysis(duration, target_duration, visual_analysis)
    )
    max_chars = _maximum_narration_chars_for_target_seconds(target_seconds, target_duration, language)
    plan_blocks = (openai_candidate_edit_plan or {}).get("blocks") or []
    block_count = len(plan_blocks) if plan_blocks else _target_narration_block_count_for_target_seconds(target_seconds)
    target_block_seconds = target_seconds / max(1, block_count)
    block_sync_instruction = _block_narration_sync_instruction(language)
    timeline_rules = _full_mode_regeneration_timeline_rules(duration, target_seconds)
    focused_repair_instruction = _focused_validation_repair_instruction(
        validation_error,
        invalid_script,
        language,
        block_count,
        duration,
        target_duration,
        target_seconds=target_seconds,
    )
    repair_scope_instruction = _repair_scope_instruction(validation_error, f"repair attempt {attempt}")
    if target_duration != "full":
        min_seconds, max_seconds = _target_duration_window_seconds(duration, target_duration)
        duration_label = _non_full_target_duration_label(target_duration)
        return f"""{original_prompt}

PREVIOUS RESPONSE WAS INVALID:
{json.dumps(invalid_script, ensure_ascii=False)}

VALIDATION ERROR:
{validation_error}

REPAIR FROM THE TRANSCRIPT AND MULTIMODAL VISUAL TIMELINE:
- {repair_scope_instruction}
- Repartition the complete edit for the requested {duration_label} commentary. The final narration_blocks/edit_segments playable time after video_speed must not exceed {max_seconds:.0f} seconds; aim for at least {min_seconds:.0f} seconds only when the source has enough useful non-repetitive material.
- Do not output the whole source timeline or a near-full-source edit. Select only the strongest useful source ranges from the available visual evidence, keep chronological order, remove repetitive/waiting/setup/camera-drift/low-value footage, and use justified video_speed for slow-but-useful ranges.
- Each narration_blocks item must match its exact visible range. If a block would be too long for its narration, shorten the source range, split the block, or cut the weak tail.
- Return narration_blocks with start, end, visual, narration, pause, rate, pitch, video_speed, and speed_reason. Keep pause=true blocks brief and only when the original picture or sound should carry the moment.
- Keep title, summary, hook, hashtags, cut_strategy, and chapters consistent with the shorter selected edit.
- Return valid JSON only, using the same JSON FORMAT.
"""
    evidence_scope_instruction = (
        "- Re-use the timestamped visual timeline and source transcript across the whole source when repartitioning the global edit; do not limit changes to the previously failed block."
        if _validation_error_is_visual_budget(validation_error)
        else "- Use the timestamped visual timeline and source transcript again only for failed blocks or nearby blocks that need local boundary changes."
    )
    locked_plan_instruction = (
        f"- BACKEND-CALCULATED EDIT PLAN is locked for this OpenAI-compatible repair. Keep exactly those plan blocks and exactly their start, end, video_speed, speed_reason, visual_facts, and evidence_timestamps; rewrite only narration, rate, pitch, title, summary, hook, episodes, and cut_strategy. {_locked_plan_block_guidance_text()}"
        if plan_blocks
        else ""
    )
    return f"""{original_prompt}

PREVIOUS RESPONSE WAS INVALID:
{json.dumps(invalid_script, ensure_ascii=False)}

VALIDATION ERROR:
{validation_error}

{focused_repair_instruction}

REPAIR FROM THE TRANSCRIPT AND MULTIMODAL VISUAL TIMELINE:
- {repair_scope_instruction}
- Fix the validation failure without creating a new sync failure. Do not pad block narration only to satisfy a character-density target; backend rendering preserves the selected visual ranges and will not hide a too-short narration block.
{evidence_scope_instruction}
{locked_plan_instruction}
{timeline_rules}
{_banned_phrase_instruction()}
- Narration must be at most {max_chars} non-whitespace characters so it remains speakable with the selected visuals; shorter commentary is valid only when it still explains the visible content clearly.
- Return exactly {block_count} narration_blocks with start, end, visual, narration, pause, rate, pitch, video_speed, and speed_reason.
- For each non-pause block, set visual to concrete on-screen evidence from that exact timestamp range. In OpenAI-compatible mode, include evidence_timestamps copied from the extracted frame timestamps; these fields are used to keep narration grounded in the selected visuals.
- If episode_plan.should_split=true, keep episodes aligned to the repaired 1-based narration_blocks indexes using start_block and end_block.
- Aim for about {target_block_seconds:.0f}s playable visuals per block across the whole edit, but keep ordinary narrated blocks usually 8-20s after video_speed so the commentary can clearly explain the selected visuals. Split longer useful ranges into multiple narrated blocks; do not leave a 40-60s narrated block with a short paragraph.
- Use concrete, clear narration for normal narrated blocks. pause=true blocks must leave narration empty.
- Any concrete action, state change, result, object identity, quantity, danger, or completion claim must be supported by that same block's timestamped visual evidence or transcript evidence.
- {block_sync_instruction}
- Each non-pause narration block must be speakable inside that block's visual duration.
- Use pause=true blocks for key reveals, process sounds, skilled visual moments, transitions, or scenes where the picture genuinely needs to play without commentary; keep pause blocks usually 2-12 seconds, under about 25% of selected visual time, and avoid more than two pause blocks back-to-back.
- Decide video_speed from the actual visible action, not from a fixed rule. Use 1.0 for key reveals, removal moments, packaging/closure, readable text, final results, completed states, tests, installations, and payoff shots. Use moderate speeds such as 1.15-1.5 for slow but still useful process footage; use 1.75-2.5 only when the range is clearly repetitive, waiting, walking, setup, repeated tool operation, transport, or transition footage and remains understandable after acceleration. If the visual evidence marks a kept range as slow/repetitive/waiting/transition or includes suggested_speed > 1.0, either set video_speed above 1.0 with a concrete speed_reason or shorten/cut that range. Every block with video_speed > 1.0 must include a concrete speed_reason tied to that exact visual range; blocks at 1.0 can use speed_reason "".
- Vary rate and pitch across non-pause blocks so the voice has cadence.
- Return valid JSON only, using the same JSON FORMAT.
"""


def _build_openai_json_syntax_repair_prompt(
    original_prompt: str,
    invalid_text: str,
    parse_error: Exception,
    attempt: int,
) -> str:
    trimmed_invalid = _limit_text_chars(str(invalid_text or ""), 20000)
    return f"""You are a strict JSON repair engine for an OpenShorts commentary script.

PREVIOUS RESPONSE WAS NOT VALID JSON:
{trimmed_invalid}

JSON PARSE ERROR:
{parse_error}

{_commentary_json_schema_summary()}

REPAIR TASK:
- This is JSON syntax repair attempt {attempt}.
- Return exactly one valid raw JSON object and nothing else.
- The first character must be {{ and the last character must be }}.
- Do not include markdown fences, explanations, reasoning, <think> blocks, comments, or surrounding prose.
- Fix broken commas, quotes, escaping, brackets, and trailing prose.
- Preserve the intended commentary content, timeline ranges, narration_blocks, edit_segments, episode_plan, episodes, cut_strategy, chapters, and hashtags wherever possible.
- If a required field is missing, synthesize the smallest schema-valid value from the available response instead of leaving the field absent.
- Ensure episode_plan is present. Use {{"should_split": false, "reason": "not needed"}} when no split plan is clear.
- Ensure array fields are arrays: narration_blocks, episodes, edit_segments, cut_strategy, chapters, hashtags.
- Do not re-analyze the video and do not add new scenes. Repair only the JSON syntax and required shape.
"""


def generate_openai_commentary_script(
    transcript: Dict,
    video_title: str,
    duration: float,
    openai_key: str,
    openai_base_url: str,
    openai_model: str,
    frame_infos: List[Dict],
    language: str = "zh",
    style: str = "hustle",
    custom_style_prompt: Optional[str] = None,
    target_duration: str = "two_to_four",
    progress: Optional[Callable[[str], None]] = None,
    openai_sampling_options: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    checkpoint: Optional[Callable[[Dict], None]] = None,
    preserve_internal_sync_fields: bool = False,
    source_audio_analysis: Optional[Dict] = None,
    precomputed_visual_analysis: Optional[Dict] = None,
    lock_candidate_edit_plan: Optional[bool] = None,
) -> Dict:
    visual_analysis = precomputed_visual_analysis
    if not visual_analysis:
        visual_analysis = _load_openai_visual_analysis(
            output_dir,
            openai_model,
            frame_infos,
            sampling_options=openai_sampling_options,
            transcript=transcript,
            source_audio_analysis=source_audio_analysis,
        ) if output_dir else None
    if visual_analysis:
        if frame_infos and not visual_analysis.get("frames"):
            visual_analysis["frames"] = _openai_frame_infos_for_prompt(frame_infos)
        if progress:
            progress(
                "Reusing cached OpenAI-compatible multimodal visual analysis "
                f"{visual_analysis.get('batch_count', 0)} batches."
            )
    elif not precomputed_visual_analysis:
        visual_analysis = _analyze_openai_visual_timeline(
            frame_infos=frame_infos,
            video_title=video_title,
            duration=duration,
            api_key=openai_key,
            base_url=openai_base_url,
            model=openai_model,
            progress=progress,
            sampling_options=openai_sampling_options,
            transcript=transcript,
            source_audio_analysis=source_audio_analysis,
        )
        if output_dir:
            cache_path = _save_openai_visual_analysis(output_dir, visual_analysis)
            if checkpoint:
                checkpoint({"openai_visual_analysis_path": cache_path})
    if not visual_analysis:
        raise Exception("OpenAI-compatible analysis mode did not produce visual analysis.")
    use_locked_candidate_plan = OPENAI_LOCK_CANDIDATE_EDIT_PLAN if lock_candidate_edit_plan is None else bool(lock_candidate_edit_plan)
    candidate_edit_plan = (
        _build_openai_candidate_edit_plan(
            visual_analysis,
            duration,
            target_duration,
            language,
        )
        if use_locked_candidate_plan
        else None
    )
    if candidate_edit_plan and progress:
        progress(
            "OpenAI-compatible candidate edit plan locked: "
            f"{len(candidate_edit_plan.get('blocks') or [])} blocks, "
            f"{float(candidate_edit_plan.get('playable_seconds') or 0.0):.1f}s playable visuals "
            f"for target window {float(candidate_edit_plan.get('min_seconds') or 0.0):.1f}-"
            f"{float(candidate_edit_plan.get('max_seconds') or 0.0):.1f}s."
        )
    prompt = _build_commentary_prompt(
        transcript=transcript,
        video_title=video_title,
        duration=duration,
        language=language,
        style=style,
        target_duration=target_duration,
        analysis_mode="openai",
        visual_analysis=visual_analysis,
        custom_style_prompt=custom_style_prompt,
        openai_candidate_edit_plan=candidate_edit_plan,
        source_audio_analysis=source_audio_analysis,
    )
    if progress:
        progress("OpenAI-compatible model is writing commentary script from transcript and visual timeline...")
    script_response_format = _openai_commentary_script_response_format()
    response_text = _call_openai_compatible_chat(
        api_key=openai_key,
        base_url=openai_base_url,
        model=openai_model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        max_tokens=OPENAI_SCRIPT_MAX_TOKENS,
        timeout_seconds=OPENAI_SCRIPT_REQUEST_TIMEOUT_SECONDS,
        response_format=script_response_format,
    )
    validation_error = None
    for script_attempt in range(1, GEMINI_SCRIPT_VALIDATION_ATTEMPTS + 1):
        try:
            data = _parse_openai_json(response_text)
        except Exception as exc:
            validation_error = exc
            if script_attempt >= GEMINI_SCRIPT_VALIDATION_ATTEMPTS:
                raise
            if progress:
                progress(
                    f"OpenAI-compatible model returned invalid JSON on correction attempt "
                    f"{script_attempt}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS}: {exc} "
                    "Asking model to repair the JSON syntax without changing the script intent..."
                )
            response_text = _call_openai_compatible_chat(
                api_key=openai_key,
                base_url=openai_base_url,
                model=openai_model,
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": _build_openai_json_syntax_repair_prompt(
                            prompt,
                            response_text,
                            exc,
                            script_attempt,
                        ),
                    }],
                }],
                max_tokens=OPENAI_SCRIPT_MAX_TOKENS,
                timeout_seconds=OPENAI_SCRIPT_REQUEST_TIMEOUT_SECONDS,
                response_format=script_response_format,
            )
            if progress:
                progress("OpenAI-compatible model returned JSON syntax repair; validating timeline sync...")
            continue
        narration = _normalize_script_narration(data)
        if not narration:
            raise Exception("OpenAI-compatible model did not return narration text")
        data["narration"] = narration
        data.setdefault("title", video_title or "Commentary Remix")
        data.setdefault("summary", "")
        if candidate_edit_plan:
            _apply_openai_candidate_edit_plan(data, candidate_edit_plan, language)
            _fit_locked_plan_narration_to_budget(
                data,
                _maximum_narration_chars_for_target_seconds(
                    float(candidate_edit_plan.get("target_seconds") or 0.0),
                    target_duration,
                    language,
                ),
                language,
            )
        _normalize_script_timeline(data, duration, target_duration, language)
        if candidate_edit_plan:
            _fit_locked_plan_narration_to_budget(
                data,
                _maximum_narration_chars_for_target_seconds(
                    float(candidate_edit_plan.get("target_seconds") or 0.0),
                    target_duration,
                    language,
                ),
                language,
            )
        _sanitize_generated_commentary_script(data, language)
        data["narration"] = _normalize_script_narration(data)
        try:
            if candidate_edit_plan:
                _validate_locked_plan_has_required_narration(data)
            _validate_commentary_script_for_target(
                data,
                duration,
                target_duration,
                language,
                visual_analysis=visual_analysis,
                custom_style_prompt=custom_style_prompt,
            )
            data.setdefault("chapters", [])
            data.setdefault("hashtags", [])
            data["_openai_analysis"] = {
                "model": openai_model,
                "frame_count": visual_analysis.get("frame_count", 0),
                "batch_count": visual_analysis.get("batch_count", 0),
                "sampling": visual_analysis.get("sampling", "unknown"),
                "scene_count": visual_analysis.get("scene_count", 0),
                "sampling_options": visual_analysis.get("sampling_options"),
                "source_commentary_available": visual_analysis.get("source_commentary_available"),
                "source_audio_analysis_available": bool(source_audio_analysis),
            }
            if data.get("narration_blocks"):
                data["narration_blocks"] = _strip_auto_filled_user_visible_fields(data.get("narration_blocks") or [])
                data["edit_segments"] = _narration_blocks_to_edit_segments(data["narration_blocks"])
            if not preserve_internal_sync_fields:
                _strip_internal_narration_block_fields(data)
            return data
        except Exception as exc:
            validation_error = exc
            if script_attempt >= GEMINI_SCRIPT_VALIDATION_ATTEMPTS:
                raise
            if progress:
                scope_label = "full-mode" if target_duration == "full" else f"{target_duration} target-duration"
                repair_scope = "global timeline" if _validation_error_is_visual_budget(exc) else "focused block"
                progress(
                    f"OpenAI-compatible script validation failed on correction attempt {script_attempt}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS}: "
                    f"{exc} Asking model to repair the invalid {scope_label} script with {repair_scope} instructions..."
                )
            response_text = _call_openai_compatible_chat(
                api_key=openai_key,
                base_url=openai_base_url,
                model=openai_model,
                messages=[{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": _build_openai_regeneration_prompt(
                            prompt,
                            data,
                            exc,
                            duration,
                            target_duration,
                            language,
                            script_attempt,
                            visual_analysis=visual_analysis,
                            openai_candidate_edit_plan=candidate_edit_plan,
                        ),
                    }],
                }],
                max_tokens=OPENAI_SCRIPT_MAX_TOKENS,
                timeout_seconds=OPENAI_SCRIPT_REQUEST_TIMEOUT_SECONDS,
                response_format=script_response_format,
            )
            if progress:
                progress("OpenAI-compatible model returned a repaired commentary script; validating timeline sync...")
    raise validation_error or Exception("OpenAI-compatible model returned invalid commentary script")


def _openai_generate_edit_first_commentary_script(
    transcript: Dict,
    video_title: str,
    duration: float,
    source_video_path: str,
    output_dir: str,
    openai_key: str,
    openai_base_url: str,
    openai_model: str,
    frame_infos: List[Dict],
    language: str,
    style: str,
    custom_style_prompt: Optional[str],
    target_duration: str,
    aspect_mode: str,
    openai_sampling_options: Optional[Dict],
    source_audio_analysis: Optional[Dict],
    auto_video_speed: bool,
    progress: Optional[Callable[[str], None]] = None,
    checkpoint: Optional[Callable[[Dict], None]] = None,
) -> Tuple[Dict, Optional[Dict], Optional[Dict], Optional[str], List[Dict], float]:
    if not OPENAI_TWO_STAGE_EDIT_THEN_COMMENTARY:
        script = generate_openai_commentary_script(
            transcript=transcript,
            video_title=video_title,
            duration=duration,
            openai_key=openai_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
            frame_infos=frame_infos,
            language=language,
            style=style,
            custom_style_prompt=custom_style_prompt,
            target_duration=target_duration,
            progress=progress,
            openai_sampling_options=openai_sampling_options,
            output_dir=output_dir,
            checkpoint=checkpoint,
            preserve_internal_sync_fields=True,
            source_audio_analysis=source_audio_analysis,
        )
        visual_analysis = _load_openai_visual_analysis(
            output_dir,
            openai_model,
            frame_infos,
            sampling_options=openai_sampling_options,
            transcript=transcript,
            source_audio_analysis=source_audio_analysis,
        )
        return script, visual_analysis, None, None, [], duration

    source_visual_analysis = _load_openai_visual_analysis(
        output_dir,
        openai_model,
        frame_infos,
        sampling_options=openai_sampling_options,
        transcript=transcript,
        source_audio_analysis=source_audio_analysis,
    )
    if source_visual_analysis:
        if frame_infos and not source_visual_analysis.get("frames"):
            source_visual_analysis["frames"] = _openai_frame_infos_for_prompt(frame_infos)
        if progress:
            progress(
                "Reusing cached full-source OpenAI-compatible multimodal visual analysis "
                f"{source_visual_analysis.get('batch_count', 0)} batches."
            )
    else:
        source_visual_analysis = _analyze_openai_visual_timeline(
            frame_infos=frame_infos,
            video_title=video_title,
            duration=duration,
            api_key=openai_key,
            base_url=openai_base_url,
            model=openai_model,
            progress=progress,
            sampling_options=openai_sampling_options,
            transcript=transcript,
            source_audio_analysis=source_audio_analysis,
        )
        cache_path = _save_openai_visual_analysis(output_dir, source_visual_analysis)
        if checkpoint:
            checkpoint({"openai_visual_analysis_path": cache_path})

    candidate_edit_plan = _build_openai_candidate_edit_plan(
        source_visual_analysis,
        duration,
        target_duration,
        language,
    )
    if candidate_edit_plan and candidate_edit_plan.get("blocks"):
        edit_plan_script = _script_from_openai_candidate_edit_plan(
            candidate_edit_plan,
            video_title,
            duration,
            target_duration,
        )
    elif duration <= _target_visual_duration_seconds(duration, target_duration) + FULL_MODE_VALIDATION_EPSILON_SECONDS:
        edit_plan_script = _script_from_full_source_edit_plan(
            video_title,
            duration,
            target_duration,
        )
    else:
        raise Exception(
            "OpenAI-compatible edit-first flow could not build a locked edit plan from full-video analysis. "
            "The full-video multimodal analysis must provide enough candidate_segments with edit value scores before the final commentary pass."
        )
    if auto_video_speed:
        edit_plan_script["narration_blocks"] = _apply_auto_video_speed_to_blocks(
            edit_plan_script.get("narration_blocks") or [],
            auto_video_speed,
            visual_analysis=source_visual_analysis,
        )
    else:
        for block in edit_plan_script.get("narration_blocks") or []:
            if isinstance(block, dict):
                block["video_speed"] = 1.0
                block["speed_reason"] = ""
    edit_plan_script["edit_segments"] = _narration_blocks_to_edit_segments(edit_plan_script.get("narration_blocks") or [])
    edit_segments = _require_ai_selected_edit_segments(edit_plan_script, duration, target_duration)
    edit_timeline, edited_duration = _edit_segments_playable_timeline(edit_segments)
    if not edit_timeline or edited_duration <= 0:
        raise Exception("OpenAI-compatible edit-first flow could not build a playable edited-video timeline.")

    if progress:
        progress(
            "OpenAI-compatible edit-first flow locked visual cut: "
            f"{len(edit_segments)} source ranges, {edited_duration:.1f}s edited video. "
            "Rendering intermediate visual edit before final commentary..."
        )
    intermediate_dir = os.path.join(output_dir, "openai_edit_first")
    os.makedirs(intermediate_dir, exist_ok=True)
    edited_video_path = os.path.join(output_dir, "openai_selected_visual_edit.mp4")
    _create_visual_edit(
        source_video_path,
        edit_segments,
        edited_video_path,
        aspect_mode,
        intermediate_dir,
        preserve_source_resolution=False,
    )
    actual_edited_duration = _get_video_duration(edited_video_path)
    if actual_edited_duration > 0:
        edited_duration = actual_edited_duration
    if checkpoint:
        checkpoint({
            "openai_intermediate_edit_path": edited_video_path,
            "openai_intermediate_edit_filename": os.path.basename(edited_video_path),
        })

    remapped_transcript = _remap_transcript_to_edited_timeline(transcript, edit_timeline)
    edited_frame_options = dict(openai_sampling_options or {})
    if edited_duration > 0 and _is_non_full_target_duration(target_duration):
        edited_frame_options["max_frames"] = min(
            resolve_openai_sampling_options(**(openai_sampling_options or {}))["max_frames"],
            max(30, int(math.ceil(edited_duration / max(1.0, OPENAI_FRAME_INTERVAL_SECONDS))) + 8),
        )
    if progress:
        progress("Extracting frames from the intermediate edited video for final commentary alignment...")
    edited_frame_infos = _extract_openai_analysis_frames(
        edited_video_path,
        intermediate_dir,
        edited_duration,
        progress=progress,
        sampling_options=edited_frame_options,
        extraction_video_path=edited_video_path,
        force_uniform=True,
    )
    if progress:
        progress(
            "Analyzing the intermediate edited video with OpenAI-compatible multimodal model; "
            "final narration will use edited-video timestamps."
        )
    edited_visual_analysis = _analyze_openai_visual_timeline(
        frame_infos=edited_frame_infos,
        video_title=f"{video_title} - edited cut",
        duration=edited_duration,
        api_key=openai_key,
        base_url=openai_base_url,
        model=openai_model,
        progress=progress,
        sampling_options=edited_frame_options,
        transcript=remapped_transcript,
        source_audio_analysis=None,
    )
    edited_visual_analysis["source_video_path"] = os.path.abspath(edited_video_path)
    edited_visual_analysis["analysis_stage"] = "edited_video_commentary"
    edited_visual_analysis["full_source_visual_analysis_summary"] = _compact_openai_visual_analysis(
        source_visual_analysis,
        max_observations=80,
        max_candidate_segments=80,
    )
    edited_visual_analysis["source_edit_timeline"] = edit_timeline
    edited_visual_analysis_path = os.path.join(output_dir, "openai_edited_visual_analysis.json")
    with open(edited_visual_analysis_path, "w", encoding="utf-8") as f:
        json.dump(edited_visual_analysis, f, ensure_ascii=False, indent=2)
    if checkpoint:
        checkpoint({"openai_edited_visual_analysis_path": edited_visual_analysis_path})

    if progress:
        progress("Writing final commentary from the edited-video analysis and full-source context...")
    edited_script = generate_openai_commentary_script(
        transcript=remapped_transcript,
        video_title=video_title,
        duration=edited_duration,
        openai_key=openai_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        frame_infos=edited_frame_infos,
        language=language,
        style=style,
        custom_style_prompt=custom_style_prompt,
        target_duration="full",
        progress=progress,
        openai_sampling_options=edited_frame_options,
        output_dir=None,
        checkpoint=None,
        preserve_internal_sync_fields=True,
        source_audio_analysis=None,
        precomputed_visual_analysis=edited_visual_analysis,
        lock_candidate_edit_plan=False,
    )
    final_blocks = _blocks_from_edited_timeline_script(edited_script, edit_timeline, language)
    if not final_blocks:
        raise Exception("OpenAI-compatible edit-first flow returned no final narration blocks for the edited video.")
    final_script = dict(edited_script)
    final_script["narration_blocks"] = final_blocks
    final_script["edit_segments"] = _narration_blocks_to_edit_segments(final_blocks)
    final_script["narration"] = _narration_from_blocks({"narration_blocks": final_blocks}) or str(final_script.get("narration") or "")
    final_script["_openai_analysis"] = {
        "model": openai_model,
        "mode": "edit_first_then_commentary",
        "source_frame_count": source_visual_analysis.get("frame_count", 0),
        "source_batch_count": source_visual_analysis.get("batch_count", 0),
        "edited_frame_count": edited_visual_analysis.get("frame_count", 0),
        "edited_batch_count": edited_visual_analysis.get("batch_count", 0),
        "sampling": edited_visual_analysis.get("sampling", "unknown"),
        "scene_count": edited_visual_analysis.get("scene_count", 0),
        "sampling_options": edited_visual_analysis.get("sampling_options"),
        "source_commentary_available": source_visual_analysis.get("source_commentary_available"),
        "source_audio_analysis_available": bool(source_audio_analysis),
        "intermediate_edit": os.path.basename(edited_video_path),
        "intermediate_duration": round(edited_duration, 3),
    }
    final_script["_openai_edit_first"] = {
        "enabled": True,
        "intermediate_edit_path": edited_video_path,
        "source_edit_segments": edit_segments,
        "source_edit_timeline": edit_timeline,
        "source_visual_analysis_path": _openai_visual_analysis_cache_path(output_dir),
        "edited_visual_analysis_path": edited_visual_analysis_path,
        "source_target_duration": target_duration,
        "render_target_duration": "full",
    }
    _normalize_script_timeline(final_script, edited_duration, "full", language)
    _sanitize_generated_commentary_script(final_script, language)
    _validate_commentary_script_for_target(
        final_script,
        edited_duration,
        "full",
        language,
        visual_analysis=edited_visual_analysis,
        custom_style_prompt=custom_style_prompt,
    )
    return final_script, source_visual_analysis, edited_visual_analysis, edited_video_path, edit_timeline, edited_duration


def _gemini_file_processing_timeout(duration: float, file_size: int) -> int:
    duration = max(0.0, float(duration or 0.0))
    size_mb = max(0.0, float(file_size or 0) / 1024 / 1024)
    adaptive_seconds = max(
        GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS,
        int(duration * 0.8),
        int(size_mb * 12),
    )
    return int(min(GEMINI_FILE_PROCESSING_MAX_TIMEOUT_SECONDS, adaptive_seconds))


def _analysis_video_fps_for_quota(duration: float) -> Optional[float]:
    duration = max(0.0, float(duration or 0.0))
    if duration <= 0:
        return None
    fps = GEMINI_SAFE_INPUT_TOKEN_BUDGET / (duration * GEMINI_LOW_RES_TOKENS_PER_SECOND)
    if fps >= 1.0:
        return None
    return round(max(0.1, fps), 3)


def _gemini_video_part_from_uri(file_uri: str, mime_type: str, duration: float = 0.0):
    part = types.Part.from_uri(file_uri=file_uri, mime_type=mime_type or "video/mp4")
    analysis_fps = _analysis_video_fps_for_quota(duration)
    if analysis_fps is not None:
        part.video_metadata = types.VideoMetadata(fps=analysis_fps)
    return part


def _is_ascii_path(path: str) -> bool:
    try:
        os.fspath(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _copy_to_ascii_safe_upload_path(path: str) -> Optional[str]:
    if _is_ascii_path(path):
        return None
    directory = os.path.dirname(os.path.abspath(path))
    stem, extension = os.path.splitext(os.path.basename(path))
    safe_extension = extension if _is_ascii_path(extension) else ".mp4"
    safe_path = os.path.join(directory, f"gemini_upload_{_safe_slug(stem)}{safe_extension}")
    if os.path.abspath(safe_path) == os.path.abspath(path):
        return None
    shutil.copy2(path, safe_path)
    return safe_path


def _upload_gemini_video_part(
    client,
    analysis_video_path: str,
    duration: float = 0.0,
    progress: Optional[Callable[[str], None]] = None,
    checkpoint: Optional[Callable[[Dict], None]] = None,
    gemini_file: Optional[Dict] = None,
    pool_session=None,
):
    reusable_uri = (gemini_file or {}).get("uri") or (gemini_file or {}).get("file_uri")
    if reusable_uri:
        mime_type = (gemini_file or {}).get("mime_type") or "video/mp4"
        if progress:
            progress("Reusing processed Gemini analysis video from previous task...")
        return _gemini_video_part_from_uri(reusable_uri, mime_type, duration)

    if not getattr(client, "files", None) or not getattr(client.files, "upload", None):
        raise Exception(
            "Gemini 视频输入模式需要使用官方 Files API 上传完整分析视频，"
            "但当前 Gemini Base URL/SDK 客户端不支持 files.upload。"
            "请使用官方 Gemini API，或切换回当前模式。"
        )

    uploaded = None
    last_error = None
    file_size = os.path.getsize(analysis_video_path)
    processing_timeout = _gemini_file_processing_timeout(duration, file_size)
    if progress:
        size_mb = file_size / 1024 / 1024
        progress(f"Uploading 360p Gemini analysis video ({size_mb:.1f} MB)...")
    upload_path = _copy_to_ascii_safe_upload_path(analysis_video_path) or analysis_video_path
    try:
        for attempt in range(1, GEMINI_FILE_UPLOAD_RETRIES + 1):
            try:
                if progress and attempt > 1:
                    progress(f"Retrying Gemini analysis video upload ({attempt}/{GEMINI_FILE_UPLOAD_RETRIES})...")
                uploaded = client.files.upload(file=upload_path)
                break
            except Exception as exc:
                last_error = exc
                classification = pool_session.record_error(exc, "files.upload") if pool_session else None
                error_text = str(exc).lower()
                transient = any(marker in error_text for marker in [
                    "server disconnected",
                    "remote protocol",
                    "timeout",
                    "temporarily unavailable",
                    "connection reset",
                    "connection aborted",
                ])
                if classification and classification.state in {"disabled", "exhausted"}:
                    raise
                if not transient or attempt == GEMINI_FILE_UPLOAD_RETRIES:
                    raise
                time.sleep(min(10, attempt * 2))
    finally:
        if upload_path != analysis_video_path and os.path.exists(upload_path):
            os.remove(upload_path)
    if uploaded is None:
        raise last_error or Exception("Gemini Files API upload failed")
    if progress:
        progress(f"Gemini analysis video uploaded; waiting for Files API processing for up to {processing_timeout}s...")
    file_name = getattr(uploaded, "name", None)
    processing_started_at = time.monotonic()
    last_processing_log = -1
    while True:
        state = str(getattr(uploaded, "state", "") or "").upper()
        if "ACTIVE" in state or not state:
            break
        if "FAILED" in state:
            raise Exception("Gemini Files API failed to process the analysis video.")
        if not file_name or not getattr(client.files, "get", None):
            break
        elapsed = time.monotonic() - processing_started_at
        if progress and int(elapsed) >= last_processing_log + 15:
            last_processing_log = int(elapsed)
            progress(f"Gemini Files API processing analysis video... {int(elapsed)}s")
        if elapsed >= processing_timeout:
            break
        time.sleep(min(GEMINI_FILE_PROCESSING_POLL_SECONDS, max(1, processing_timeout - int(elapsed))))
        uploaded = client.files.get(name=file_name)
    final_state = str(getattr(uploaded, "state", "") or "").upper()
    if final_state and "ACTIVE" not in final_state:
        raise Exception(
            "Gemini Files API did not finish processing the analysis video in time. "
            f"Waited {processing_timeout}s; "
            "for large videos, retry after a few minutes or increase "
            "OPENSHORTS_GEMINI_FILE_PROCESSING_TIMEOUT_SECONDS or "
            "OPENSHORTS_GEMINI_FILE_PROCESSING_MAX_TIMEOUT_SECONDS."
        )

    file_uri = getattr(uploaded, "uri", None)
    if not file_uri:
        raise Exception("Gemini Files API upload did not return a usable file URI.")
    mime_type = getattr(uploaded, "mime_type", None) or "video/mp4"
    if checkpoint:
        checkpoint({
            "gemini_file_uri": file_uri,
            "gemini_file_name": file_name,
            "gemini_file_mime_type": mime_type,
        })
    if progress:
        progress("Gemini analysis video is ready for model analysis.")
    return _gemini_video_part_from_uri(file_uri, mime_type, duration)


def _build_video_analysis_contents(
    prompt: str,
    analysis_video_path: str,
    client=None,
    duration: float = 0.0,
    progress: Optional[Callable[[str], None]] = None,
    checkpoint: Optional[Callable[[Dict], None]] = None,
    gemini_file: Optional[Dict] = None,
    pool_session=None,
):
    reusable_uri = (gemini_file or {}).get("uri") or (gemini_file or {}).get("file_uri")
    if reusable_uri:
        video_part = _upload_gemini_video_part(
            client,
            analysis_video_path or "",
            duration=duration,
            progress=progress,
            checkpoint=checkpoint,
            gemini_file=gemini_file,
            pool_session=pool_session,
        )
        return [types.Content(role="user", parts=[video_part, types.Part.from_text(text=prompt)])]
    if not analysis_video_path:
        raise Exception("Missing analysis video path for Gemini video input mode")
    if not os.path.exists(analysis_video_path):
        raise Exception(f"Missing analysis video file: {analysis_video_path}")
    file_size = os.path.getsize(analysis_video_path)
    if file_size > GEMINI_FILES_API_HARD_MAX_BYTES:
        raise Exception(
            "分析视频超过 Gemini Files API 单文件上限 2GB。"
            "请降低分析视频质量、缩短视频，或切换回当前模式。"
        )
    video_part = _upload_gemini_video_part(
        client,
        analysis_video_path,
        duration=duration,
        progress=progress,
        checkpoint=checkpoint,
        gemini_file=gemini_file,
        pool_session=pool_session,
    )
    return [types.Content(role="user", parts=[video_part, types.Part.from_text(text=prompt)])]


def _prepare_analysis_video_for_gemini(
    analysis_video_path: str,
    output_dir: str,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    if not analysis_video_path or not os.path.exists(analysis_video_path):
        raise Exception("Missing analysis video path for Gemini video input mode")

    stem, _ = os.path.splitext(os.path.basename(analysis_video_path))
    label = f"{GEMINI_ANALYSIS_HEIGHT}p"
    prepared_path = os.path.join(output_dir, f"{stem}_gemini_{label}.mp4")
    if os.path.abspath(prepared_path) == os.path.abspath(analysis_video_path):
        prepared_path = os.path.join(output_dir, f"{stem}_gemini_analysis_{label}.mp4")

    source_duration = _get_video_duration(analysis_video_path)
    if progress:
        progress(f"Compressing Gemini analysis video to {GEMINI_ANALYSIS_HEIGHT}p...")
    cmd = [
        "ffmpeg", "-y",
        "-i", analysis_video_path,
        "-vf", f"scale=-2:{GEMINI_ANALYSIS_HEIGHT}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(GEMINI_ANALYSIS_CRF),
        "-c:a", "aac",
        "-b:a", GEMINI_ANALYSIS_AUDIO_BITRATE,
        "-ac", "1",
        "-movflags", "+faststart",
        prepared_path,
    ]
    _run_ffmpeg_with_progress(cmd, duration=source_duration, progress=progress, label=f"Compressing Gemini analysis video to {GEMINI_ANALYSIS_HEIGHT}p")
    if not os.path.exists(prepared_path):
        raise Exception("Failed to create compressed Gemini analysis video.")
    prepared_size = os.path.getsize(prepared_path)
    if prepared_size <= GEMINI_ANALYSIS_TARGET_MAX_BYTES:
        if progress:
            progress(f"Gemini analysis video ready: {prepared_size / 1024 / 1024:.1f} MB")
        return prepared_path

    if progress:
        progress("360p analysis video is still large; creating no-audio fallback...")
    no_audio_path = os.path.join(output_dir, f"{stem}_gemini_{label}_noaudio.mp4")
    no_audio_cmd = [
        "ffmpeg", "-y",
        "-i", analysis_video_path,
        "-vf", f"scale=-2:{GEMINI_ANALYSIS_HEIGHT}",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(max(GEMINI_ANALYSIS_CRF, 34)),
        "-movflags", "+faststart",
        no_audio_path,
    ]
    _run_ffmpeg_with_progress(no_audio_cmd, duration=source_duration, progress=progress, label=f"Compressing no-audio Gemini analysis video to {GEMINI_ANALYSIS_HEIGHT}p")
    if os.path.exists(no_audio_path) and os.path.getsize(no_audio_path) <= GEMINI_ANALYSIS_TARGET_MAX_BYTES:
        if progress:
            progress(f"No-audio Gemini analysis video ready: {os.path.getsize(no_audio_path) / 1024 / 1024:.1f} MB")
        return no_audio_path

    target_mb = GEMINI_ANALYSIS_TARGET_MAX_BYTES // (1024 * 1024)
    raise Exception(
        f"分析视频压缩后仍超过 OpenShorts 当前配置的 {target_mb}MB 上传目标。"
        "请降低视频时长/质量，或切换回当前模式。"
        f" Last analysis file: {no_audio_path if os.path.exists(no_audio_path) else prepared_path}"
    )


def _prepare_analysis_video_for_openai_frames(
    source_video_path: str,
    output_dir: str,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    if not source_video_path or not os.path.exists(source_video_path):
        raise Exception("Missing source video path for OpenAI-compatible frame analysis")

    stem, _ = os.path.splitext(os.path.basename(source_video_path))
    label = f"{OPENAI_FRAME_HEIGHT}p"
    prepared_path = os.path.join(output_dir, f"{stem}_openai_frames_{label}.mp4")
    if os.path.abspath(prepared_path) == os.path.abspath(source_video_path):
        prepared_path = os.path.join(output_dir, f"{stem}_openai_analysis_{label}.mp4")
    if os.path.exists(prepared_path) and os.path.getsize(prepared_path) > 0:
        if progress:
            progress(f"Reusing OpenAI-compatible {label} frame-analysis video.")
        return prepared_path

    source_duration = _get_video_duration(source_video_path)
    if progress:
        progress(f"Preparing OpenAI-compatible {label} frame-analysis video...")
    cmd = [
        "ffmpeg", "-y",
        "-i", source_video_path,
        "-vf", f"scale=-2:{OPENAI_FRAME_HEIGHT}",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(max(GEMINI_ANALYSIS_CRF, 32)),
        "-movflags", "+faststart",
        prepared_path,
    ]
    _run_ffmpeg_with_progress(
        cmd,
        duration=source_duration,
        progress=progress,
        label=f"Preparing OpenAI-compatible {label} frame-analysis video",
    )
    if not os.path.exists(prepared_path) or os.path.getsize(prepared_path) <= 0:
        raise Exception("Failed to create OpenAI-compatible frame-analysis video.")
    if progress:
        progress(f"OpenAI-compatible frame-analysis video ready: {os.path.getsize(prepared_path) / 1024 / 1024:.1f} MB")
    return prepared_path


def _extract_zero_region_quota_location(error_text: str) -> Optional[str]:
    if "ApiRequestsPerMinutePerProjectPerRegion" not in error_text:
        return None
    if "quota_limit_value" not in error_text or not re.search(r"quota_limit_value['\"]?\s*:\s*['\"]0['\"]", error_text):
        return None
    match = re.search(r"quota_location['\"]?\s*:\s*['\"]([^'\"]+)['\"]", error_text)
    return match.group(1) if match else "当前区域"


def _format_zero_region_quota_error(error_text: str) -> Optional[str]:
    location = _extract_zero_region_quota_location(error_text)
    if not location:
        return None
    return (
        f"Gemini 当前区域 {location} 的每分钟请求配额为 0，后端已被 Google 拒绝请求。"
        "这不是视频下载、转码或 Whisper 转录错误；继续重试同一个 Key 通常不会成功。"
        "请更换有 Gemini 配额的 API Key/Google Cloud 项目，或在设置里配置可用的 Gemini Base URL/代理，"
        "也可以在 Google Cloud 为该项目申请提高 Gemini API regional RPM quota。"
        f" Technical details: {error_text}"
    )


def _gemini_video_proxy_timeout_fallback_reason(error_text: Optional[str]) -> Optional[str]:
    lowered = str(error_text or "").lower()
    if not lowered:
        return None
    if "ali-oss" not in lowered and "responsetimeouterror" not in lowered:
        return None
    if "response timeout" not in lowered and "timeout for 60000ms" not in lowered:
        return None
    return (
        "Gemini video Files analysis through the configured proxy timed out while the proxy read the uploaded "
        "video from ali-oss (60s response timeout). Falling back to local Faster-Whisper transcript plus "
        "keyframe visual context instead of resending the large video file URI."
    )


def _is_retryable_gemini_error(error_text: str) -> bool:
    return any(marker in error_text for marker in [
        "429",
        "RESOURCE_EXHAUSTED",
        "502",
        "Bad Gateway",
        "bad gateway",
        "503",
        "UNAVAILABLE",
        "500",
        "INTERNAL",
        "temporarily unavailable",
        "high demand",
        "empty response text",
    ])


def _should_failover_gemini_pool(gemini_pool: Optional[GeminiKeyPool], classification) -> bool:
    return bool(
        gemini_pool
        and gemini_pool.mode == "official_pool"
        and classification
        and classification.state in {"disabled", "exhausted", "cooldown"}
    )


def _is_gemini_file_permission_error(classification) -> bool:
    return bool(classification and classification.state == "file_permission")


def _text_only_gemini_config_kwargs(config_kwargs: Dict) -> Dict:
    text_config = dict(config_kwargs or {})
    text_config.pop("media_resolution", None)
    return text_config


def _raise_pool_exhausted(pool_error: Exception, last_error: Exception) -> None:
    raise Exception(
        "所有配置的 Gemini Key 都不可用或正在冷却，已尝试切换到下一个 Key 但没有可用 Key。"
        f"最后一次 Gemini 错误：{last_error}"
    ) from pool_error


def _generate_content_with_retry(
    client,
    model: str,
    contents,
    config_kwargs: Dict,
    pool_session=None,
    gemini_pool: Optional[GeminiKeyPool] = None,
    max_attempts: int = 5,
):
    last_error = None
    attempts = max(max_attempts, len(gemini_pool.keys) if gemini_pool and gemini_pool.mode == "official_pool" else max_attempts)
    for attempt in range(1, attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            if pool_session:
                pool_session.record_success("models.generate_content", response)
            _ensure_gemini_response_has_text(response, "Gemini follow-up request")
            return response
        except Exception as exc:
            last_error = exc
            failed_fingerprint = pool_session.fingerprint if pool_session else "current key"
            classification = pool_session.record_error(exc, "models.generate_content") if pool_session else None
            error_text = str(exc)
            if _should_failover_gemini_pool(gemini_pool, classification):
                try:
                    pool_session = gemini_pool.checkout("generate")
                    client = pool_session.client
                    print(f"[Commentary] Gemini follow-up key {failed_fingerprint} failed ({classification.state}); trying next configured key...")
                    continue
                except Exception as pool_error:
                    _raise_pool_exhausted(pool_error, exc)
            if not _is_retryable_gemini_error(error_text) or attempt >= attempts:
                raise
            wait_seconds = min(60, 5 * attempt)
            print(f"[Commentary] Gemini follow-up request failed on attempt {attempt}/{attempts}, retrying in {wait_seconds}s: {error_text}")
            time.sleep(wait_seconds)
    raise last_error or Exception("Gemini request failed")


def generate_commentary_script(
    transcript: Dict,
    video_title: str,
    duration: float,
    gemini_key: str,
    language: str = "zh",
    style: str = "hustle",
    custom_style_prompt: Optional[str] = None,
    target_duration: str = "two_to_four",
    base_url: Optional[str] = None,
    frame_paths: Optional[List[str]] = None,
    analysis_video_path: Optional[str] = None,
    analysis_mode: str = DEFAULT_ANALYSIS_MODE,
    gemini_model: Optional[str] = None,
    gemini_pool: Optional[GeminiKeyPool] = None,
    progress: Optional[Callable[[str], None]] = None,
    checkpoint: Optional[Callable[[Dict], None]] = None,
    gemini_file: Optional[Dict] = None,
    previous_error: Optional[str] = None,
) -> Dict:
    analysis_mode = _normalize_analysis_mode(analysis_mode)
    resolved_model = gemini_model or DEFAULT_GEMINI_MODEL
    pool_session = gemini_pool.checkout("files+generate") if gemini_pool else None
    client = pool_session.client if pool_session else create_gemini_client(gemini_key, base_url)
    prompt = _build_commentary_prompt(
        transcript=transcript,
        video_title=video_title,
        duration=duration,
        language=language,
        style=style,
        target_duration=target_duration,
        analysis_mode=analysis_mode,
        custom_style_prompt=custom_style_prompt,
    )
    if previous_error:
        prompt += _retry_correction_note(previous_error)
    reusable_gemini_file = None if gemini_pool and gemini_pool.mode == "official_pool" else gemini_file
    if analysis_mode == "video":
        while True:
            try:
                contents = _build_video_analysis_contents(
                    prompt,
                    analysis_video_path or "",
                    client=client,
                    duration=duration,
                    progress=progress,
                    checkpoint=checkpoint,
                    gemini_file=reusable_gemini_file,
                    pool_session=pool_session,
                )
                break
            except Exception as exc:
                classification = None
                if pool_session:
                    classification = pool_session.record_error(exc, "files.upload")
                if not _should_failover_gemini_pool(gemini_pool, classification):
                    raise
                failed_fingerprint = pool_session.fingerprint if pool_session else "current key"
                try:
                    pool_session = gemini_pool.checkout("files+generate")
                    client = pool_session.client
                    reusable_gemini_file = None
                    if progress:
                        progress(f"Gemini file upload key {failed_fingerprint} failed ({classification.state}); switching to next configured key...")
                except Exception as pool_error:
                    _raise_pool_exhausted(pool_error, exc)
    else:
        contents = _build_frame_analysis_contents(prompt, frame_paths)
    config_kwargs = {
        "response_mime_type": "application/json",
    }
    if analysis_mode == "video":
        config_kwargs["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_LOW

    response = None
    last_error = None
    using_proxy = bool(normalize_gemini_base_url(base_url))
    for attempt in range(1, 6):
        try:
            if progress:
                progress(f"Gemini is analyzing the video and writing commentary script (attempt {attempt}/5)...")
            response = client.models.generate_content(
                model=resolved_model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            if pool_session:
                pool_session.record_success("models.generate_content", response)
            _ensure_gemini_response_has_text(response, "Gemini script generation")
            if progress:
                progress("Gemini returned a commentary script; validating timeline sync...")
            break
        except Exception as exc:
            last_error = exc
            error_text = str(exc)
            classification = pool_session.record_error(exc, "models.generate_content") if pool_session else None
            zero_region_quota_error = _format_zero_region_quota_error(error_text)
            if zero_region_quota_error:
                raise Exception(zero_region_quota_error) from exc
            if _is_gemini_file_permission_error(classification) and analysis_mode == "video":
                if progress:
                    progress("Reusable Gemini file is not accessible with the current key; re-uploading analysis video...")
                reusable_gemini_file = None
                contents = _build_video_analysis_contents(
                    prompt,
                    analysis_video_path or "",
                    client=client,
                    duration=duration,
                    progress=progress,
                    checkpoint=checkpoint,
                    gemini_file=None,
                    pool_session=pool_session,
                )
                continue
            if _should_failover_gemini_pool(gemini_pool, classification):
                failed_fingerprint = pool_session.fingerprint if pool_session else "current key"
                try:
                    pool_session = gemini_pool.checkout("files+generate")
                    client = pool_session.client
                    if analysis_mode == "video":
                        reusable_gemini_file = None
                        contents = _build_video_analysis_contents(
                            prompt,
                            analysis_video_path or "",
                            client=client,
                            duration=duration,
                            progress=progress,
                            checkpoint=checkpoint,
                            gemini_file=None,
                            pool_session=pool_session,
                        )
                    if progress:
                        progress(f"Gemini key {failed_fingerprint} failed ({classification.state}); switching to next configured key...")
                    continue
                except Exception as pool_error:
                    _raise_pool_exhausted(pool_error, exc)
            retryable_error = any(marker in error_text for marker in [
                "PERMISSION_DENIED",
                "403",
                "denied access",
                "429",
                "RESOURCE_EXHAUSTED",
                "502",
                "Bad Gateway",
                "bad gateway",
                "503",
                "UNAVAILABLE",
                "500",
                "INTERNAL",
                "empty response text",
            ])
            if not retryable_error or attempt == 5:
                if "PERMISSION_DENIED" in error_text or "403" in error_text or "denied access" in error_text:
                    if using_proxy:
                        raise Exception(
                            "Gemini 代理/Key 连续重试后仍返回 403 PERMISSION_DENIED：当前配置的 Gemini Base URL 或 API Key 被上游拒绝访问。"
                            "这不是 YouTube cookies 问题，视频下载和转录已经完成。请稍后重试，或在设置页更换可用的 Gemini Key。"
                        ) from exc
                    raise Exception(
                        "Gemini 官方接口连续重试后仍返回 403 PERMISSION_DENIED：当前 API Key/项目没有权限访问 Gemini 模型，"
                        "请更换可用 Key 或确认该 Google Cloud/AI Studio 项目已启用 Gemini API。"
                    ) from exc
                unsupported_video = analysis_mode == "video" and any(marker in error_text.lower() for marker in [
                    "video",
                    "mime",
                    "unsupported",
                    "invalid_argument",
                    "400",
                ])
                if unsupported_video:
                    raise Exception(
                        "当前 Gemini Base URL 或模型可能不支持视频输入模式。"
                        "请使用支持视频输入的 Gemini API/模型，或切换回当前模式。"
                        f" Technical details: {error_text}"
                    ) from exc
                default_model_unavailable = resolved_model == DEFAULT_GEMINI_MODEL and any(marker in error_text.lower() for marker in [
                    "model",
                    "not found",
                    "404",
                    "unsupported",
                    "permission_denied",
                ])
                if default_model_unavailable:
                    raise Exception(
                        f"默认 Gemini 模型 {DEFAULT_GEMINI_MODEL} 可能不被当前 Key/Base URL 支持。"
                        "请在配置的提供方启用该模型，或指定可用的 Gemini 模型。"
                        f" Technical details: {error_text}"
                    ) from exc
                raise
            wait_seconds = min(60, 5 * attempt)
            print(f"[Commentary] Gemini request failed on attempt {attempt}/5, retrying in {wait_seconds}s: {error_text}")
            time.sleep(wait_seconds)
    if response is None:
        raise last_error or Exception("Gemini request failed")

    validation_error = None
    for script_attempt in range(1, GEMINI_SCRIPT_VALIDATION_ATTEMPTS + 1):
        try:
            data = json.loads(_clean_json_text(_gemini_response_text(response)))
        except Exception as exc:
            validation_error = exc
            if script_attempt >= GEMINI_SCRIPT_VALIDATION_ATTEMPTS:
                raise Exception(f"Gemini returned invalid JSON for the commentary script: {exc}") from exc
            if progress:
                progress(
                    f"Gemini returned invalid JSON on correction attempt "
                    f"{script_attempt}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS}: {exc} "
                    "Asking Gemini to repair the JSON response..."
                )
            repair_prompt = _build_openai_json_syntax_repair_prompt(
                prompt,
                _gemini_response_text(response),
                exc,
                script_attempt,
            )
            response = _generate_content_with_retry(
                client,
                resolved_model,
                [repair_prompt],
                _text_only_gemini_config_kwargs(config_kwargs),
                pool_session=pool_session,
                gemini_pool=gemini_pool,
            )
            if progress:
                progress("Gemini returned JSON repair; validating timeline sync...")
            continue
        narration = _normalize_script_narration(data)
        if not narration:
            validation_error = Exception("Gemini did not return narration text")
            if script_attempt >= GEMINI_SCRIPT_VALIDATION_ATTEMPTS:
                raise validation_error
            if progress:
                scope_label = "full-mode" if target_duration == "full" else f"{target_duration} target-duration"
                progress(
                    f"Gemini script validation failed on correction attempt {script_attempt}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS}: "
                    f"{validation_error} Asking Gemini to rewrite the {scope_label} script..."
                )
            response = _generate_content_with_retry(
                client,
                resolved_model,
                _replace_prompt_in_contents(contents, _build_regeneration_prompt(
                    prompt,
                    data if isinstance(data, dict) else {},
                    duration,
                    target_duration,
                    language,
                    attempt=script_attempt,
                    validation_error=validation_error,
                )),
                config_kwargs,
                pool_session=pool_session,
                gemini_pool=gemini_pool,
            )
            if progress:
                progress("Gemini returned a corrected commentary script; validating timeline sync...")
            continue
        data["narration"] = narration
        data.setdefault("title", video_title or "Commentary Remix")
        data.setdefault("summary", "")
        _normalize_script_timeline(data, duration, target_duration, language)
        _sanitize_generated_commentary_script(data, language)
        data["narration"] = _normalize_script_narration(data)
        try:
            _validate_commentary_script_for_target(
                data,
                duration,
                target_duration,
                language,
                custom_style_prompt=custom_style_prompt,
            )
            data.setdefault("chapters", [])
            data.setdefault("hashtags", [])
            if data.get("narration_blocks"):
                data["narration_blocks"] = _strip_auto_filled_user_visible_fields(data.get("narration_blocks") or [])
                data["edit_segments"] = _narration_blocks_to_edit_segments(data["narration_blocks"])
            _strip_internal_narration_block_fields(data)
            return data
        except Exception as exc:
            validation_error = exc
            if script_attempt >= GEMINI_SCRIPT_VALIDATION_ATTEMPTS:
                raise
            if progress:
                scope_label = "full-mode" if target_duration == "full" else f"{target_duration} target-duration"
                progress(
                    f"Gemini script validation failed on correction attempt {script_attempt}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS}: "
                    f"{exc} Asking Gemini to rewrite the {scope_label} script..."
                )
            if target_duration == "full" and _has_visual_plan(data):
                next_contents = [
                    _build_visual_plan_finalization_prompt(
                        data,
                        duration,
                        target_duration,
                        language,
                        attempt=script_attempt,
                        validation_error=exc,
                        custom_style_prompt=custom_style_prompt,
                    )
                ]
            else:
                regeneration_prompt = _build_regeneration_prompt(
                    prompt,
                    data,
                    duration,
                    target_duration,
                    language,
                    attempt=script_attempt,
                    validation_error=exc,
                )
                next_contents = _replace_prompt_in_contents(contents, regeneration_prompt)
            if progress:
                scope_label = "full-mode" if target_duration == "full" else f"{target_duration} target-duration"
                progress(f"Gemini is rewriting the {scope_label} commentary script (correction {script_attempt + 1}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS})...")
            response = _generate_content_with_retry(
                client,
                resolved_model,
                next_contents,
                config_kwargs,
                pool_session=pool_session,
                gemini_pool=gemini_pool,
            )
            if progress:
                progress("Gemini returned a corrected commentary script; validating timeline sync...")

    raise validation_error or Exception("Gemini returned invalid commentary script")


def _resolve_aspect_mode(video_info: Dict, aspect_mode: str = "auto") -> str:
    mode = (aspect_mode or "auto").lower()
    if mode in {"9:16", "vertical"}:
        return "9:16"
    if mode in {"16:9", "horizontal"}:
        return "16:9"
    width = float(video_info.get("width") or 0)
    height = float(video_info.get("height") or 0)
    if width > 0 and height > 0 and height > width:
        return "9:16"
    return "16:9"


def _video_filter_for_aspect(aspect_mode: str) -> str:
    if aspect_mode == "9:16":
        return "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1"
    if aspect_mode == "16:9":
        return "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1"
    return "setsar=1"


def _create_visual_edit(
    video_path: str,
    edit_segments: List[Dict],
    output_path: str,
    aspect_mode: str,
    work_dir: str,
    preserve_source_resolution: bool = False,
) -> None:
    segments = list(edit_segments or [])
    if not segments:
        raise Exception(
            "Cannot create commentary visual edit without AI-selected edit_segments. "
            "The model must choose kept source ranges before rendering."
        )
    part_paths = []
    vf = None if preserve_source_resolution else _video_filter_for_aspect(aspect_mode)
    for index, segment in enumerate(segments, start=1):
        part_path = os.path.join(work_dir, f"edit_part_{index:03d}.mp4")
        duration = max(0.1, float(segment["end"]) - float(segment["start"]))
        speed = _safe_render_video_speed(segment.get("video_speed") or segment.get("speed"))
        filters = []
        if speed > 1.0001:
            filters.append(f"setpts=PTS*{(1.0 / speed):.6f}")
        if vf:
            filters.append(vf)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{float(segment['start']):.3f}",
            "-t", f"{duration:.3f}",
            "-i", video_path,
            "-an",
        ]
        if filters:
            cmd.extend([
                "-vf", ",".join(filters),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
            ])
        else:
            cmd.extend(["-c:v", "copy"])
        cmd.append(part_path)
        _run_command(cmd)
        part_paths.append(part_path)

    list_path = os.path.join(work_dir, "edit_parts.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for part_path in part_paths:
            safe_path = part_path.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ]
    _run_command(cmd)


def _ambient_audio_filter(volume: float) -> str:
    volume = max(0.0, min(volume, 1.0))
    return f"volume={volume}"


def get_background_music_tracks() -> List[Dict]:
    tracks = []
    for track in BACKGROUND_MUSIC_TRACKS.values():
        track_path = os.path.join(COMMENTARY_MUSIC_DIR, track["filename"])
        tracks.append({
            **track,
            "available": os.path.exists(track_path),
        })
    return tracks


def _resolve_background_music_track(track_id: Optional[str]) -> Dict:
    resolved_id = (track_id or "aodebiao_caravan").strip()
    track = BACKGROUND_MUSIC_TRACKS.get(resolved_id)
    if not track:
        raise ValueError(f"Unsupported background music track: {track_id}")
    track_path = os.path.join(COMMENTARY_MUSIC_DIR, track["filename"])
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Background music file not found: {track_path}")
    return {**track, "path": track_path}


def _create_background_music_bed(
    source_audio_path: str,
    output_path: str,
    duration: float,
    volume: float = DEFAULT_BACKGROUND_MUSIC_VOLUME,
) -> str:
    target_duration = max(0.1, float(duration or 0.0))
    safe_volume = max(0.0, min(float(volume or 0.0), 1.0))
    if safe_volume <= 0:
        raise ValueError("Background music volume must be greater than 0")
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", source_audio_path,
        "-t", f"{target_duration:.3f}",
        "-af", f"volume={safe_volume},afade=t=out:st={max(0.0, target_duration - 1.5):.3f}:d={min(1.5, target_duration):.3f}",
        "-c:a", "aac",
        "-b:a", "128k",
        *_ffmpeg_output_format_args(output_path),
        output_path,
    ]
    _run_command(cmd)
    return output_path


def _create_ambient_audio_bed(
    video_path: str,
    edit_segments: List[Dict],
    output_path: str,
    volume: float,
    work_dir: str,
) -> Optional[str]:
    volume = max(0.0, min(volume, 1.0))
    if volume <= 0:
        return None

    segments = list(edit_segments or [])
    if not segments:
        return None
    part_paths = []
    for index, segment in enumerate(segments, start=1):
        part_path = os.path.join(work_dir, f"ambient_part_{index:03d}.m4a")
        duration = max(0.1, float(segment["end"]) - float(segment["start"]))
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{float(segment['start']):.3f}",
            "-t", f"{duration:.3f}",
            "-i", video_path,
            "-vn",
            "-af", _ambient_audio_filter(volume),
            "-c:a", "aac",
            "-b:a", "128k",
            *_ffmpeg_output_format_args(part_path),
            part_path,
        ]
        try:
            _run_command(cmd)
            part_paths.append(part_path)
        except Exception:
            continue

    if not part_paths:
        return None

    list_path = os.path.join(work_dir, "ambient_parts.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for part_path in part_paths:
            safe_path = part_path.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        *_ffmpeg_output_format_args(output_path),
        output_path,
    ]
    _run_command(cmd)
    return output_path


def _fit_video_to_voiceover(video_path: str, voiceover_path: str, output_path: str) -> None:
    video_duration = _get_video_duration(video_path)
    audio_duration = _get_audio_duration(voiceover_path)
    if video_duration <= 0 or audio_duration <= 0:
        shutil.copyfile(video_path, output_path)
        return
    ratio = video_duration / audio_duration
    if 0.92 <= ratio <= 1.08:
        shutil.copyfile(video_path, output_path)
        return
    ratio = max(0.5, min(ratio, 2.0))
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-filter:v", f"setpts=PTS*{(1.0 / ratio):.6f}",
        "-an",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        output_path,
    ]
    _run_command(cmd)


def _write_concat_list(paths: List[str], list_path: str) -> None:
    use_windows_paths = os.path.basename(str(FFMPEG_BINARY)).lower() == "ffmpeg.exe"
    with open(list_path, "w", encoding="utf-8") as f:
        for path in paths:
            concat_path = os.path.abspath(path)
            concat_path = _windows_path_from_wsl(concat_path) if use_windows_paths else concat_path
            safe_path = concat_path.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")


def _concat_media_parts(paths: List[str], output_path: str, work_dir: str, codec: str = "copy", media_type: str = "video") -> None:
    if not paths:
        raise Exception("No media parts to concatenate")
    if media_type == "audio":
        _concat_audio_parts_precise(paths, output_path, codec=codec if codec != "copy" else "aac")
        return
    list_path = os.path.join(work_dir, f"concat_{_safe_slug(os.path.basename(output_path))}.txt")
    _write_concat_list(paths, list_path)
    codec_args = ["-c", codec]
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        *codec_args,
        output_path,
    ]
    _run_command(cmd)


def _concat_audio_parts_precise(paths: List[str], output_path: str, codec: str = "aac", bitrate: str = "192k") -> None:
    if not paths:
        raise Exception("No audio parts to concatenate")
    audio_codec = codec if codec and codec != "copy" else "aac"
    cmd = ["ffmpeg", "-y"]
    for path in paths:
        cmd.extend(["-i", path])

    filters = []
    labels = []
    for index in range(len(paths)):
        label = f"a{index}"
        filters.append(
            f"[{index}:a]aresample=48000,"
            "aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"asetpts=N/SR/TB[{label}]"
        )
        labels.append(f"[{label}]")
    filter_complex = ";".join(filters)
    filter_complex = f"{filter_complex};{''.join(labels)}concat=n={len(paths)}:v=0:a=1[aout]"

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-vn",
        "-c:a", audio_codec,
    ])
    if audio_codec == "aac" and bitrate:
        cmd.extend(["-b:a", bitrate])
    cmd.extend(_ffmpeg_output_format_args(output_path))
    cmd.append(output_path)
    _run_command(cmd)


def _atempo_filter(speed: float) -> str:
    speed = max(0.25, min(float(speed or 1.0), 4.0))
    filters = []
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    filters.append(f"atempo={speed:.6f}")
    return ",".join(filters)


def _fit_audio_part_to_duration(input_audio_path: str, output_audio_path: str, target_duration: float, max_speedup: Optional[float] = None) -> None:
    target_duration = max(0.1, float(target_duration or 0.0))
    max_speedup = max(1.0, float(max_speedup or FULL_MODE_RENDER_SYNC_MAX_AUDIO_SPEED))
    audio_duration = _get_audio_duration(input_audio_path)
    filters = []
    if audio_duration > target_duration:
        speedup = audio_duration / target_duration
        if speedup > max_speedup:
            raise Exception(
                "A timestamped commentary block is too long for its visual range. "
                f"Got {audio_duration:.1f}s audio for {target_duration:.1f}s visuals; shorten that block's narration."
            )
        filters.append(_atempo_filter(speedup))
    filters.extend(["apad", f"atrim=0:{target_duration:.3f}", "asetpts=N/SR/TB"])
    cmd = [
        "ffmpeg", "-y",
        "-i", input_audio_path,
        "-af", ",".join(filters),
        "-t", f"{target_duration:.3f}",
        "-c:a", "aac",
        "-b:a", "192k",
        *_ffmpeg_output_format_args(output_audio_path),
        output_audio_path,
    ]
    _run_command(cmd)


def _trim_audio_part_to_duration(input_audio_path: str, output_audio_path: str, target_duration: float) -> None:
    target_duration = max(0.1, float(target_duration or 0.0))
    cmd = [
        "ffmpeg", "-y",
        "-i", input_audio_path,
        "-af", f"atrim=0:{target_duration:.3f},asetpts=N/SR/TB",
        "-t", f"{target_duration:.3f}",
        "-c:a", "aac",
        "-b:a", "192k",
        *_ffmpeg_output_format_args(output_audio_path),
        output_audio_path,
    ]
    _run_command(cmd)


def _create_silent_audio_clip(output_path: str, duration: float) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-t", f"{max(0.1, float(duration or 0.0)):.3f}",
        "-c:a", "aac",
        "-b:a", "48k",
        *_ffmpeg_output_format_args(output_path),
        output_path,
    ]
    _run_command(cmd)


def _force_audio_clip_duration(audio_path: str, duration: float, work_dir: str, bitrate: str = "192k") -> None:
    target_duration = max(0.1, float(duration or 0.0))
    tmp_path = os.path.join(work_dir, f"duration_fixed_{_safe_slug(os.path.basename(audio_path))}.m4a")
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-af", f"apad,atrim=0:{target_duration:.3f},asetpts=N/SR/TB",
        "-t", f"{target_duration:.3f}",
        "-c:a", "aac",
        "-b:a", bitrate,
        *_ffmpeg_output_format_args(tmp_path),
        tmp_path,
    ]
    _run_command(cmd)
    os.replace(tmp_path, audio_path)


def _probe_media_format_duration(path: str) -> Optional[float]:
    if not path or not os.path.exists(path):
        return None
    ffprobe_name = os.path.basename(str(FFPROBE_BINARY)).lower()
    probe_path = _windows_path_from_wsl(path) if ffprobe_name == "ffprobe.exe" else path
    returncode, stdout, _stderr = _run_capture_command([
        FFPROBE_BINARY, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        probe_path,
    ])
    if returncode != 0:
        return None
    try:
        return float((stdout or "").strip() or 0.0)
    except ValueError:
        return None


def _assert_media_duration_close(path: str, expected_duration: float, label: str, tolerance: float = 0.75) -> None:
    if not path or not os.path.exists(path):
        return
    expected = max(0.1, float(expected_duration or 0.0))
    actual = _probe_media_format_duration(path)
    if actual is None:
        return
    if abs(actual - expected) > tolerance:
        raise Exception(
            f"{label} duration mismatch after render sync. "
            f"Got {actual:.2f}s, expected {expected:.2f}s. "
            "OpenShorts stopped to avoid an abrupt or incomplete commentary ending."
        )


def _extract_original_audio_clip(
    video_path: str,
    start: float,
    duration: float,
    output_path: str,
    volume: float = 1.0,
    speed: float = 1.0,
    output_duration: Optional[float] = None,
) -> None:
    volume = max(0.0, min(volume, 1.0))
    speed = _safe_render_video_speed(speed)
    filters = [f"volume={volume}"]
    if speed > 1.0001:
        filters.append(_atempo_filter(speed))
    if output_duration is not None:
        filters.extend(["apad", f"atrim=0:{max(0.1, float(output_duration or 0.0)):.3f}", "asetpts=N/SR/TB"])
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{float(start):.3f}",
        "-t", f"{max(0.1, float(duration or 0.0)):.3f}",
        "-i", video_path,
        "-vn",
        "-af", ",".join(filters),
    ]
    if output_duration is not None:
        cmd.extend(["-t", f"{max(0.1, float(output_duration or 0.0)):.3f}"])
    cmd.extend([
        "-c:a", "aac",
        "-b:a", "128k",
        *_ffmpeg_output_format_args(output_path),
        output_path,
    ])
    _run_command(cmd)


def _block_video_filter(aspect_filter: str, speed: float) -> str:
    filters = []
    speed = _safe_render_video_speed(speed)
    if speed > 1.0001:
        filters.append(f"setpts=PTS*{(1.0 / speed):.6f}")
    if aspect_filter:
        filters.append(aspect_filter)
    return ",".join(filters) or "setsar=1"


def _create_block_synced_visuals_and_audio(
    video_path: str,
    narration_blocks: List[Dict],
    timed_video_path: str,
    voiceover_path: str,
    ambient_audio_path: str,
    aspect_mode: str,
    work_dir: str,
    tts_provider: str,
    language: str,
    elevenlabs_key: Optional[str],
    voice_id: str,
    edge_voice: Optional[str],
    original_audio_volume: float,
    pause_original_audio_volume: float = 0.6,
    preserve_source_resolution: bool = True,
    block_concurrency: Optional[int] = None,
    trim_short_tts_tails: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[str], List[float]]:
    blocks = _normalize_narration_blocks(narration_blocks, _get_video_duration(video_path))
    if not blocks:
        raise Exception("Cannot build synced commentary without narration_blocks")

    part_dir = os.path.join(work_dir, "synced_blocks")
    os.makedirs(part_dir, exist_ok=True)
    vf_aspect = "setsar=1" if preserve_source_resolution else _video_filter_for_aspect(aspect_mode)
    spoken_volume = max(0.0, min(original_audio_volume, 1.0))
    pause_volume = max(0.0, min(pause_original_audio_volume, 1.0))
    needs_ambient_track = spoken_volume > 0 or pause_volume > 0
    total_blocks = len(blocks)
    resolved_concurrency = min(resolve_commentary_block_concurrency(block_concurrency), total_blocks)
    progress_lock = threading.Lock()

    def report(message: str) -> None:
        if not progress:
            return
        with progress_lock:
            progress(message)

    def render_segment(
        index: int,
        sub_index: int,
        block: Dict,
        source_start: float,
        render_source_duration: float,
        video_speed: float,
        visual_duration: float,
        is_pause: bool,
        voice_path: Optional[str],
        narration_text: str,
    ) -> Dict:
        rendered_block = dict(block)
        rendered_block["start"] = round(float(source_start), 3)
        rendered_block["end"] = round(float(source_start) + float(render_source_duration), 3)
        rendered_block["narration"] = "" if is_pause else narration_text
        rendered_block["pause"] = is_pause
        rendered_block["video_speed"] = _safe_video_speed(video_speed)
        rendered_block["rendered_duration"] = round(visual_duration, 3)
        rendered_block["evidence_timestamps"] = _filter_block_evidence_timestamps_for_range(
            block,
            rendered_block["start"],
            rendered_block["end"],
        )
        speed_token = int(round(video_speed * 1000))
        source_token = int(round(render_source_duration * 1000))
        duration_token = int(round(visual_duration * 1000))
        segment_token = f"{index:03d}" if sub_index == 0 else f"{index:03d}_{sub_index:02d}"
        cache_token = f"s{speed_token}_src{source_token}_dur{duration_token}"
        fitted_voice_path = os.path.join(part_dir, f"block_voice_fit_{segment_token}_{cache_token}.m4a")
        block_video_path = os.path.join(part_dir, f"block_video_{segment_token}_{cache_token}.mp4")
        block_ambient_path = os.path.join(part_dir, f"block_ambient_{segment_token}_{cache_token}.m4a") if needs_ambient_track else None
        if is_pause:
            _create_silent_audio_clip(fitted_voice_path, visual_duration)
        else:
            if not voice_path:
                raise Exception("Cannot render narrated commentary block without generated voice audio")
            _fit_audio_part_to_duration(voice_path, fitted_voice_path, visual_duration)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{source_start:.3f}",
            "-t", f"{render_source_duration:.3f}",
            "-i", video_path,
            "-an",
            "-vf", _block_video_filter(vf_aspect, video_speed),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            block_video_path,
        ]
        _run_command(cmd)

        if block_ambient_path:
            try:
                if is_pause and pause_volume > 0:
                    _extract_original_audio_clip(
                        video_path,
                        source_start,
                        render_source_duration,
                        block_ambient_path,
                        volume=pause_volume,
                        speed=video_speed,
                        output_duration=visual_duration,
                    )
                elif not is_pause and spoken_volume > 0:
                    _extract_original_audio_clip(
                        video_path,
                        source_start,
                        render_source_duration,
                        block_ambient_path,
                        volume=spoken_volume,
                        speed=video_speed,
                        output_duration=visual_duration,
                    )
                else:
                    _create_silent_audio_clip(block_ambient_path, visual_duration)
            except Exception:
                _create_silent_audio_clip(block_ambient_path, visual_duration)
        return {
            "index": index,
            "sub_index": sub_index,
            "video_path": block_video_path,
            "voice_path": fitted_voice_path,
            "ambient_path": block_ambient_path,
            "duration": visual_duration,
            "block": rendered_block,
        }

    def process_block(index: int, block: Dict) -> List[Dict]:
        _raise_if_commentary_cancelled()
        is_pause = bool(block.get("pause"))
        source_start = float(block["start"])
        source_duration = max(0.1, float(block["end"]) - source_start)
        requested_video_speed = _safe_render_video_speed(block.get("video_speed"))
        video_speed = requested_video_speed
        visual_duration = max(0.1, source_duration / video_speed)
        speed_label = f" at {video_speed:g}x" if video_speed > 1.0001 else ""
        if is_pause:
            report(f"Adding original-audio pause block {index}/{total_blocks}{speed_label}...")
            return [
                render_segment(
                    index=index,
                    sub_index=0,
                    block=block,
                    source_start=source_start,
                    render_source_duration=source_duration,
                    video_speed=video_speed,
                    visual_duration=visual_duration,
                    is_pause=True,
                    voice_path=None,
                    narration_text="",
                )
            ]

        report(f"Generating synced commentary block {index}/{total_blocks}{speed_label}...")
        block_voice_path = os.path.join(part_dir, f"block_voice_{index:03d}.mp3")
        narration_text = str(block.get("narration") or "").strip()
        max_audio_speedup = max(1.0, FULL_MODE_RENDER_SYNC_MAX_AUDIO_SPEED)
        generate_commentary_voiceover(
            text=narration_text,
            output_path=block_voice_path,
            tts_provider=tts_provider,
            language=language,
            elevenlabs_key=elevenlabs_key,
            voice_id=voice_id,
            edge_voice=edge_voice,
            rate=block.get("rate") or "+0%",
            pitch=block.get("pitch") or "+0Hz",
        )
        last_voice_duration = max(0.1, _get_audio_duration(block_voice_path))

        if last_voice_duration / max(0.1, visual_duration) > max_audio_speedup + 0.01:
            required_visual_duration = last_voice_duration / max_audio_speedup
            slower_speed = min(video_speed, source_duration / max(0.1, required_visual_duration))
            if slower_speed >= 1.0 and slower_speed < video_speed - 0.001:
                video_speed = round(slower_speed, 3)
                visual_duration = max(0.1, source_duration / video_speed)
                report(
                    f"Slowing commentary block {index}/{total_blocks}: "
                    f"TTS needs {last_voice_duration:.1f}s, so video_speed changes from {requested_video_speed:g}x to {video_speed:g}x."
                )
            elif FULL_MODE_RENDER_SYNC_TTS_REWRITE_ATTEMPTS > 1 and narration_text:
                shortened_text = _shorten_narration_to_fit_visual(
                    narration_text,
                    visual_duration * max_audio_speedup,
                    language,
                )
                if not shortened_text or len(shortened_text) >= len(narration_text):
                    shortened_text = narration_text[:max(1, int(len(narration_text) * 0.7))].rstrip(" ,，.。!?！？") + ("。" if (language or "").lower().startswith("zh") else ".")
                narration_text = shortened_text
                generate_commentary_voiceover(
                    text=narration_text,
                    output_path=block_voice_path,
                    tts_provider=tts_provider,
                    language=language,
                    elevenlabs_key=elevenlabs_key,
                    voice_id=voice_id,
                    edge_voice=edge_voice,
                    rate=block.get("rate") or "+0%",
                    pitch=block.get("pitch") or "+0Hz",
                )
                last_voice_duration = max(0.1, _get_audio_duration(block_voice_path))
                if last_voice_duration / max(0.1, visual_duration) > max_audio_speedup + 0.01:
                    raise Exception(
                        "A timestamped commentary block is too long for its visual range. "
                        f"Block {index} has {last_voice_duration:.1f}s audio for {visual_duration:.1f}s visuals after "
                        "AI-selected video_speed and source range are preserved; shorten that block's narration, lower video_speed, or expand its source range before rendering."
                    )
            else:
                raise Exception(
                    "A timestamped commentary block is too long for its visual range. "
                    f"Block {index} has {last_voice_duration:.1f}s audio for {visual_duration:.1f}s visuals after "
                    "AI-selected video_speed and source range are preserved; shorten that block's narration, lower video_speed, or expand its source range before rendering."
                )
        min_voice_duration = _minimum_voiceover_seconds_for_visual_duration(visual_duration)
        if (
            visual_duration >= 12.0
            and last_voice_duration + FULL_MODE_VALIDATION_EPSILON_SECONDS < min_voice_duration
        ):
            narrated_visual_duration = min(
                visual_duration,
                _max_visual_seconds_for_actual_voiceover(last_voice_duration),
            )
            tail_visual_duration = visual_duration - narrated_visual_duration
            if narrated_visual_duration <= 0.1 or tail_visual_duration <= FULL_MODE_VALIDATION_EPSILON_SECONDS:
                trailing_silence = max(0.0, visual_duration - last_voice_duration)
                raise Exception(
                    "A timestamped commentary block is too short for its visual range after TTS. "
                    f"Block {index} has {last_voice_duration:.1f}s audio for {visual_duration:.1f}s visuals, "
                    f"leaving about {trailing_silence:.1f}s without matching narration. "
                    "Shorten or split this block's source range, add concrete scene-matched narration, or move the silent tail into a brief pause=true block before rendering."
                )
            if trim_short_tts_tails:
                narrated_source_duration = max(0.1, narrated_visual_duration * video_speed)
                trimmed_block_end = source_start + narrated_source_duration
                evidence = _block_evidence_timestamps(block)
                if evidence and not all(source_start - 0.35 <= timestamp <= trimmed_block_end + 0.35 for timestamp in evidence):
                    raise Exception(
                        "A timestamped commentary block cannot be trimmed without losing its visual evidence. "
                        f"Block {index} would keep {source_start:.3f}-{trimmed_block_end:.3f}s, "
                        f"but its evidence_timestamps are {evidence}. "
                        "Shorten the selected source range around the evidence timestamps, add fuller scene-matched narration, or render the leftover visual as pause=true."
                    )
                report(
                    f"Trimming short-TTS commentary block {index}/{total_blocks}: "
                    f"{last_voice_duration:.1f}s audio supports {narrated_visual_duration:.1f}s of "
                    f"{visual_duration:.1f}s visuals; dropping the unmatched tail for compact sync."
                )
                return [
                    render_segment(
                        index=index,
                        sub_index=0,
                        block=block,
                        source_start=source_start,
                        render_source_duration=narrated_source_duration,
                        video_speed=video_speed,
                        visual_duration=narrated_visual_duration,
                        is_pause=False,
                        voice_path=block_voice_path,
                        narration_text=narration_text,
                    )
                ]
            narrated_source_duration = max(0.1, narrated_visual_duration * video_speed)
            tail_source_duration = max(0.1, source_duration - narrated_source_duration)
            tail_visual_duration = tail_source_duration / video_speed
            report(
                f"Splitting short-TTS commentary block {index}/{total_blocks}: "
                f"{last_voice_duration:.1f}s audio supports {narrated_visual_duration:.1f}s of "
                f"{visual_duration:.1f}s visuals; keeping {tail_visual_duration:.1f}s as original-audio pause."
            )
            return [
                render_segment(
                    index=index,
                    sub_index=0,
                    block=block,
                    source_start=source_start,
                    render_source_duration=narrated_source_duration,
                    video_speed=video_speed,
                    visual_duration=narrated_visual_duration,
                    is_pause=False,
                    voice_path=block_voice_path,
                    narration_text=narration_text,
                ),
                render_segment(
                    index=index,
                    sub_index=1,
                    block=block,
                    source_start=source_start + narrated_source_duration,
                    render_source_duration=tail_source_duration,
                    video_speed=video_speed,
                    visual_duration=tail_visual_duration,
                    is_pause=True,
                    voice_path=None,
                    narration_text="",
                ),
            ]

        return [
            render_segment(
                index=index,
                sub_index=0,
                block=block,
                source_start=source_start,
                render_source_duration=source_duration,
                video_speed=video_speed,
                visual_duration=visual_duration,
                is_pause=False,
                voice_path=block_voice_path,
                narration_text=narration_text,
            )
        ]

    context_job_id = _current_commentary_job_id()
    context_cancel_event = _current_commentary_cancel_event()

    def process_block_with_context(index: int, block: Dict) -> List[Dict]:
        if context_job_id:
            with commentary_job_context(context_job_id, context_cancel_event):
                return process_block(index, block)
        return process_block(index, block)

    if resolved_concurrency > 1:
        _raise_if_commentary_cancelled()
        with ThreadPoolExecutor(max_workers=resolved_concurrency) as executor:
            futures = [executor.submit(process_block_with_context, index, block) for index, block in enumerate(blocks, start=1)]
            block_results = [item for future in as_completed(futures) for item in future.result()]
    else:
        block_results = [item for index, block in enumerate(blocks, start=1) for item in process_block_with_context(index, block)]

    block_results.sort(key=lambda item: (item["index"], item.get("sub_index", 0)))
    video_parts = [item["video_path"] for item in block_results]
    voice_parts = [item["voice_path"] for item in block_results]
    part_durations = [item["duration"] for item in block_results]
    ambient_parts = [item["ambient_path"] for item in block_results if item.get("ambient_path")]
    rendered_blocks = [item["block"] for item in block_results]
    _assert_rendered_blocks_keep_evidence_timestamps(rendered_blocks)
    narration_blocks[:] = rendered_blocks

    visual_duration_total = sum(part_durations)
    _concat_media_parts(video_parts, timed_video_path, part_dir)
    _concat_media_parts(voice_parts, voiceover_path, part_dir, codec="aac", media_type="audio")
    _force_audio_clip_duration(voiceover_path, visual_duration_total, part_dir, bitrate="192k")
    _assert_media_duration_close(voiceover_path, visual_duration_total, "Synced commentary voiceover")
    if ambient_parts:
        _concat_media_parts(ambient_parts, ambient_audio_path, part_dir, codec="aac", media_type="audio")
        _force_audio_clip_duration(ambient_audio_path, visual_duration_total, part_dir, bitrate="128k")
        _assert_media_duration_close(ambient_audio_path, visual_duration_total, "Synced ambient audio")
        return ambient_audio_path, part_durations
    return None, part_durations


def _mix_voiceover_with_video(
    video_path: str,
    voiceover_path: str,
    output_path: str,
    original_audio_volume: float = 0.3,
    ambient_audio_path: Optional[str] = None,
    background_music_path: Optional[str] = None,
    trim_to_voiceover: bool = True,
) -> None:
    audio_inputs = [voiceover_path]
    if ambient_audio_path and os.path.exists(ambient_audio_path):
        audio_inputs.append(ambient_audio_path)
    if background_music_path and os.path.exists(background_music_path):
        audio_inputs.append(background_music_path)

    if len(audio_inputs) > 1:
        audio_duration_mode = "first" if trim_to_voiceover else "longest"
        filter_parts = []
        mix_labels = []
        for input_index in range(1, len(audio_inputs) + 1):
            label = f"a{input_index}"
            filter_parts.append(f"[{input_index}:a]volume=1.0[{label}]")
            mix_labels.append(f"[{label}]")
        filter_parts.append(
            f"{''.join(mix_labels)}amix=inputs={len(audio_inputs)}:duration={audio_duration_mode}:dropout_transition=0[a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            *[arg for audio_path in audio_inputs for arg in ("-i", audio_path)],
            "-filter_complex", ";".join(filter_parts),
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path,
        ]
        if trim_to_voiceover:
            cmd.insert(-1, "-shortest")
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", voiceover_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path,
        ]
        if trim_to_voiceover:
            cmd.insert(-1, "-shortest")
    _run_command(cmd)


def _get_audio_duration(audio_path: str) -> float:
    returncode, stdout, stderr = _run_capture_command([
        FFPROBE_BINARY, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        _windows_path_from_wsl(audio_path) if os.path.basename(str(FFPROBE_BINARY)).lower() == "ffprobe.exe" else audio_path,
    ])
    if returncode != 0:
        raise Exception(stderr or stdout or f"Failed to probe audio: {audio_path}")
    return float(stdout.strip() or 0)


def _format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    if centiseconds >= 100:
        whole_seconds += 1
        centiseconds = 0
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _split_narration_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])", text or "") if s.strip()]
    if sentences:
        return sentences
    return [text.strip()] if text and text.strip() else []


def _normalize_ass_dimensions(width: Optional[int] = None, height: Optional[int] = None) -> Tuple[int, int]:
    safe_width = int(width or ASS_SUBTITLE_DEFAULT_WIDTH)
    safe_height = int(height or ASS_SUBTITLE_DEFAULT_HEIGHT)
    return max(1, safe_width), max(1, safe_height)


def _probe_video_dimensions(video_path: str) -> Tuple[int, int]:
    ffprobe_name = os.path.basename(str(FFPROBE_BINARY)).lower()
    probe_path = _windows_path_from_wsl(video_path) if ffprobe_name == "ffprobe.exe" else video_path
    returncode, stdout, _stderr = _run_capture_command([
        FFPROBE_BINARY, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        probe_path,
    ])
    if returncode != 0:
        return _normalize_ass_dimensions()
    first_line = (stdout or "").strip().splitlines()[0:1]
    if not first_line or "x" not in first_line[0]:
        return _normalize_ass_dimensions()
    width, height = first_line[0].split("x", 1)
    try:
        return _normalize_ass_dimensions(int(width), int(height))
    except ValueError:
        return _normalize_ass_dimensions()


def _subtitle_style_values(width: Optional[int] = None, height: Optional[int] = None) -> Tuple[int, int, int, int, int]:
    safe_width, safe_height = _normalize_ass_dimensions(width, height)
    font_size = int(round(safe_height * ASS_SUBTITLE_FONT_HEIGHT_RATIO))
    font_size = max(ASS_SUBTITLE_MIN_FONT_SIZE, min(ASS_SUBTITLE_MAX_FONT_SIZE, font_size))
    margin_x = max(32, int(round(safe_width * ASS_SUBTITLE_MARGIN_X_RATIO)))
    margin_v = max(48, int(round(safe_height * ASS_SUBTITLE_MARGIN_Y_RATIO)))
    outline = max(1, int(round(font_size * 0.04)))
    shadow = 0
    return font_size, margin_x, margin_v, outline, shadow


def _subtitle_max_line_units(width: Optional[int] = None, height: Optional[int] = None) -> int:
    safe_width, safe_height = _normalize_ass_dimensions(width, height)
    font_size, margin_x, _, _, _ = _subtitle_style_values(safe_width, safe_height)
    available_width = max(font_size * 4, safe_width - (2 * margin_x))
    calculated_units = int(available_width / max(1.0, font_size / 2))
    default_width, default_height = _normalize_ass_dimensions(ASS_SUBTITLE_DEFAULT_WIDTH, ASS_SUBTITLE_DEFAULT_HEIGHT)
    if safe_width == default_width and safe_height == default_height:
        return max(12, min(ASS_SUBTITLE_MAX_LINE_UNITS, calculated_units))
    return max(12, calculated_units)


def _ass_header_lines(width: Optional[int] = None, height: Optional[int] = None) -> List[str]:
    safe_width, safe_height = _normalize_ass_dimensions(width, height)
    font_size, margin_x, margin_v, outline, shadow = _subtitle_style_values(safe_width, safe_height)
    return [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {safe_width}",
        f"PlayResY: {safe_height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Microsoft YaHei,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_x},{margin_x},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]


def _subtitle_char_units(char: str) -> int:
    return 1 if char.isascii() else 2


def _wrap_ass_subtitle_lines(text: str, max_units: int = ASS_SUBTITLE_MAX_LINE_UNITS) -> List[str]:
    clean = (text or "").replace("\n", " ").replace("{", "").replace("}", "")
    words = re.split(r"(\s+)", clean)
    lines = []
    current = ""
    current_units = 0

    def push_current() -> None:
        nonlocal current, current_units
        if current.strip():
            lines.append(current.strip())
        current = ""
        current_units = 0

    for token in words:
        if not token:
            continue
        token_units = sum(_subtitle_char_units(char) for char in token)
        if token.isspace():
            if current and current_units + 1 <= max_units:
                current += " "
                current_units += 1
            continue
        if token_units > max_units:
            for char in token:
                char_units = _subtitle_char_units(char)
                if current and current_units + char_units > max_units:
                    push_current()
                current += char
                current_units += char_units
            continue
        if current and current_units + token_units > max_units:
            push_current()
        current += token
        current_units += token_units
    push_current()
    return lines if lines else ([clean.strip()] if clean.strip() else [])


def _wrap_ass_subtitle_text(text: str, max_units: int = ASS_SUBTITLE_MAX_LINE_UNITS) -> str:
    return r"\N".join(_wrap_ass_subtitle_lines(text, max_units=max_units))


def _chunk_ass_subtitle_text(text: str, max_units: int = ASS_SUBTITLE_MAX_LINE_UNITS, max_lines: int = ASS_SUBTITLE_MAX_VISIBLE_LINES) -> List[str]:
    wrapped_lines = _wrap_ass_subtitle_lines(text, max_units=max_units)
    if not wrapped_lines:
        return []
    safe_max_lines = max(1, max_lines)
    chunks = []
    for index in range(0, len(wrapped_lines), safe_max_lines):
        chunks.append(r"\N".join(wrapped_lines[index:index + safe_max_lines]))
    return chunks


def _append_weighted_subtitle_lines(lines: List[str], sentences: List[str], start_time: float, duration: float, max_units: int = ASS_SUBTITLE_MAX_LINE_UNITS) -> None:
    if not sentences or duration <= 0:
        return
    if len(sentences) == 1:
        chunk = _wrap_ass_subtitle_text(sentences[0], max_units=max_units)
        if chunk:
            lines.append(f"Dialogue: 0,{_format_ass_time(start_time)},{_format_ass_time(start_time + duration)},Default,,0,0,0,,{chunk}")
        return
    sentence_chunks = []
    for sentence in sentences:
        chunks = _chunk_ass_subtitle_text(sentence, max_units=max_units)
        if chunks:
            sentence_chunks.extend(chunks)
    if not sentence_chunks:
        return
    weights = [max(len(chunk.replace(r"\N", "")), 1) for chunk in sentence_chunks]
    total_weight = sum(weights)
    cursor = start_time
    block_end = start_time + duration
    for index, chunk in enumerate(sentence_chunks):
        segment_duration = duration * weights[index] / total_weight
        start = cursor
        end = block_end if index == len(sentence_chunks) - 1 else min(block_end, cursor + segment_duration)
        cursor = end
        lines.append(f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},Default,,0,0,0,,{chunk}")


def _write_text_timed_ass(narration: str, audio_path: str, output_path: str, video_dimensions: Optional[Tuple[int, int]] = None) -> None:
    duration = _get_audio_duration(audio_path)
    sentences = _split_narration_sentences(narration)
    if not sentences:
        raise Exception("Cannot generate subtitles from empty narration")

    width, height = _normalize_ass_dimensions(*(video_dimensions or (ASS_SUBTITLE_DEFAULT_WIDTH, ASS_SUBTITLE_DEFAULT_HEIGHT)))
    lines = _ass_header_lines(width, height)
    _append_weighted_subtitle_lines(lines, sentences, 0.0, duration, max_units=_subtitle_max_line_units(width, height))
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))


def _write_block_timed_ass(narration_blocks: List[Dict], output_path: str, block_durations: Optional[List[float]] = None, video_dimensions: Optional[Tuple[int, int]] = None) -> None:
    if block_durations:
        # Render-time blocks may include sub-second pause tails split from a
        # narrated block. The generic script normalizer drops ranges below 1s,
        # which would shift block_durations onto the wrong subtitle text.
        blocks = [dict(block) for block in narration_blocks or [] if isinstance(block, dict)]
    else:
        blocks = _normalize_narration_blocks(
            narration_blocks,
            max((float(block.get("end") or 0) for block in narration_blocks or [] if isinstance(block, dict)), default=0.0),
        )
    if not blocks:
        raise Exception("Cannot generate subtitles from empty narration blocks")
    width, height = _normalize_ass_dimensions(*(video_dimensions or (ASS_SUBTITLE_DEFAULT_WIDTH, ASS_SUBTITLE_DEFAULT_HEIGHT)))
    lines = _ass_header_lines(width, height)
    max_units = _subtitle_max_line_units(width, height)
    cursor = 0.0
    for index, block in enumerate(blocks):
        if block_durations and index < len(block_durations):
            block_duration = max(0.1, float(block_durations[index] or 0.0))
        else:
            block_duration = max(0.1, float(block["end"]) - float(block["start"]))
        _append_weighted_subtitle_lines(lines, _split_narration_sentences(str(block.get("narration") or "")), cursor, block_duration, max_units=max_units)
        cursor += block_duration
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))


def _ass_last_dialogue_end_seconds(subtitle_path: str) -> float:
    try:
        with open(subtitle_path, "r", encoding="utf-8-sig") as f:
            text = f.read()
    except Exception:
        return 0.0
    pattern = re.compile(r"^Dialogue:\s*\d+,\d+:\d\d:\d\d\.\d\d,(\d+):(\d\d):(\d\d)\.(\d\d),", re.M)
    ends = []
    for match in pattern.finditer(text):
        hours, minutes, seconds, centiseconds = (int(value) for value in match.groups())
        ends.append(hours * 3600 + minutes * 60 + seconds + centiseconds / 100.0)
    return max(ends, default=0.0)


def _assert_full_mode_output_alignment(
    final_path: str,
    voiceover_path: str,
    subtitle_path: Optional[str],
    block_durations: List[float],
    source_duration: Optional[float] = None,
    narration_blocks: Optional[List[Dict]] = None,
) -> None:
    if not block_durations:
        return
    expected_duration = max(0.1, sum(float(duration or 0.0) for duration in block_durations))
    final_duration = _probe_media_format_duration(final_path)
    voiceover_duration = _probe_media_format_duration(voiceover_path)
    if final_duration is None or voiceover_duration is None:
        return
    if abs(final_duration - expected_duration) > 1.0:
        raise Exception(
            "Final commentary video duration does not match the synced narration blocks. "
            f"Got {final_duration:.2f}s video for {expected_duration:.2f}s block timeline."
        )
    if source_duration and final_duration > float(source_duration) + 1.0:
        raise Exception(
            "Final commentary video is longer than the source video. "
            f"Got {final_duration:.2f}s final video for a {float(source_duration):.2f}s source. "
            "OpenShorts stopped because full-process commentary may preserve or modestly accelerate the source, but it must not extend beyond the original timeline."
        )
    if abs(voiceover_duration - expected_duration) > 1.0:
        raise Exception(
            "Final commentary voiceover duration does not match the synced narration blocks. "
            f"Got {voiceover_duration:.2f}s audio for {expected_duration:.2f}s block timeline."
        )
    if subtitle_path and os.path.exists(subtitle_path):
        subtitle_end = _ass_last_dialogue_end_seconds(subtitle_path)
        expected_subtitle_end = expected_duration
        if narration_blocks:
            cursor = 0.0
            last_spoken_end = 0.0
            for index, block in enumerate(narration_blocks):
                if index < len(block_durations):
                    block_duration = max(0.1, float(block_durations[index] or 0.0))
                else:
                    block_duration = _block_visual_duration(block) if isinstance(block, dict) else 0.0
                if (
                    isinstance(block, dict)
                    and not bool(block.get("pause"))
                    and str(block.get("narration") or "").strip()
                ):
                    last_spoken_end = cursor + block_duration
                cursor += block_duration
            if last_spoken_end > 0:
                expected_subtitle_end = last_spoken_end
        if subtitle_end > 0 and abs(subtitle_end - expected_subtitle_end) > 1.25:
            raise Exception(
                "Final commentary subtitles do not match the synced narration blocks. "
                f"Last subtitle ends at {subtitle_end:.2f}s for {expected_subtitle_end:.2f}s spoken block timeline."
            )


def _assert_non_full_output_duration_target(
    block_durations: List[float],
    source_duration: float,
    target_duration: str,
) -> None:
    if not _is_non_full_target_duration(target_duration) or not block_durations:
        return
    min_seconds, max_seconds = _target_duration_window_seconds(source_duration, target_duration)
    if max_seconds <= 0:
        return
    actual_seconds = sum(max(0.0, float(duration or 0.0)) for duration in block_durations)
    tolerance = max(0.0, NON_FULL_TARGET_DURATION_TOLERANCE_SECONDS)
    label = _non_full_target_duration_label(target_duration)
    if actual_seconds > max_seconds + tolerance:
        raise Exception(
            f"Rendered commentary video does not match the requested {label} target duration. "
            f"Got {actual_seconds:.1f}s after TTS/block sync; expected no more than {max_seconds:.1f}s. "
            "Regenerate with fewer selected visual ranges or stronger justified video_speed."
        )
    if actual_seconds + tolerance < min_seconds:
        raise Exception(
            f"Rendered commentary video does not match the requested {label} target duration. "
            f"Got {actual_seconds:.1f}s after TTS/block sync; expected at least {min_seconds:.1f}s. "
            "OpenShorts stopped because render-time trimming or sparse narration made the actual output much shorter than requested; regenerate with enough important timestamped ranges and scene-matched narration."
        )


def _burn_subtitles(video_path: str, subtitle_path: str, output_path: str) -> None:
    work_dir = os.path.dirname(os.path.abspath(video_path)) or None
    video_name = os.path.basename(video_path)
    subtitle_name = os.path.basename(subtitle_path)
    output_name = os.path.basename(output_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_name,
        "-vf", f"subtitles={subtitle_name}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "copy",
        output_name,
    ]
    _run_command(cmd, cwd=work_dir)


def _commentary_block_output_ranges(narration_blocks: List[Dict], block_durations: Optional[List[float]]) -> List[Dict]:
    ranges = []
    cursor = 0.0
    for index, block in enumerate(narration_blocks):
        if block_durations and index < len(block_durations):
            duration = max(0.1, float(block_durations[index] or 0.0))
        else:
            duration = max(0.1, _block_visual_duration(block))
        ranges.append({"start": cursor, "end": cursor + duration, "duration": duration})
        cursor += duration
    return ranges


def _render_commentary_episode_videos(
    final_path: str,
    episodes: List[Dict],
    narration_blocks: List[Dict],
    block_durations: Optional[List[float]],
    output_dir: str,
    slug: str,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Dict]:
    if not episodes or not narration_blocks:
        return []
    block_ranges = _commentary_block_output_ranges(narration_blocks, block_durations)
    rendered = []
    job_id = os.path.basename(output_dir)
    for episode in episodes:
        start_block = _clamp_int(episode.get("start_block"), 1, 1, len(block_ranges))
        end_block = _clamp_int(episode.get("end_block"), start_block, start_block, len(block_ranges))
        output_start = block_ranges[start_block - 1]["start"]
        output_end = block_ranges[end_block - 1]["end"]
        output_duration = max(0.1, output_end - output_start)
        episode_number = _clamp_int(episode.get("episode_number"), len(rendered) + 1, 1, 99)
        episode_filename = f"{slug}_episode_{episode_number}.mp4"
        episode_path = os.path.join(output_dir, episode_filename)
        if progress:
            progress(f"Rendering commentary episode {episode_number}: blocks {start_block}-{end_block}...")
        _run_command([
            "ffmpeg", "-y",
            "-ss", f"{output_start:.3f}",
            "-i", final_path,
            "-t", f"{output_duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            episode_path,
        ])
        rendered_episode = dict(episode)
        rendered_episode.update({
            "episode_number": len(rendered) + 1,
            "video_path": episode_path,
            "video_filename": episode_filename,
            "video_url": f"/videos/{job_id}/{episode_filename}",
            "output_start": round(output_start, 3),
            "output_end": round(output_end, 3),
            "duration": round(output_duration, 3),
        })
        rendered.append(rendered_episode)
    return rendered


def generate_commentary_video(
    source: str,
    output_dir: str,
    gemini_key: str,
    elevenlabs_key: Optional[str] = None,
    source_type: str = "url",
    voice_id: str = DEFAULT_VOICE_ID,
    edge_voice: Optional[str] = None,
    tts_provider: str = "edge",
    language: str = "zh",
    style: str = "hustle",
    custom_style_prompt: Optional[str] = None,
    target_duration: str = "two_to_four",
    original_audio_volume: float = 0.3,
    pause_original_audio_volume: float = 0.6,
    subtitles: bool = True,
    vertical: bool = False,
    aspect_mode: str = "auto",
    source_language: Optional[str] = None,
    gemini_base_url: Optional[str] = None,
    analysis_mode: str = DEFAULT_ANALYSIS_MODE,
    gemini_model: Optional[str] = None,
    openai_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_frame_interval_seconds: Optional[float] = None,
    openai_max_frames: Optional[int] = None,
    openai_scene_max_keyframes: Optional[int] = None,
    openai_batch_size: Optional[int] = None,
    openai_visual_concurrency: Optional[int] = None,
    commentary_block_concurrency: Optional[int] = None,
    auto_video_speed: bool = True,
    background_music_enabled: bool = False,
    background_music_track: str = "aodebiao_caravan",
    background_music_volume: float = DEFAULT_BACKGROUND_MUSIC_VOLUME,
    gemini_pool: Optional[GeminiKeyPool] = None,
    progress: Optional[Callable[[str], None]] = None,
    checkpoint: Optional[Callable[[Dict], None]] = None,
    prepared_analysis_video_path: Optional[str] = None,
    gemini_file: Optional[Dict] = None,
    previous_error: Optional[str] = None,
) -> Dict:
    analysis_mode = _normalize_analysis_mode(analysis_mode)
    requested_analysis_mode = analysis_mode
    resolved_gemini_model = gemini_model or DEFAULT_GEMINI_MODEL
    analysis_fallback_reason = (
        _gemini_video_proxy_timeout_fallback_reason(previous_error)
        if analysis_mode == "video"
        else None
    )
    if analysis_fallback_reason:
        analysis_mode = "current"
    openai_sampling_options = resolve_openai_sampling_options(
        frame_interval_seconds=openai_frame_interval_seconds,
        max_frames=openai_max_frames,
        scene_max_keyframes=openai_scene_max_keyframes,
        batch_size=openai_batch_size,
        visual_concurrency=openai_visual_concurrency,
    ) if analysis_mode == "openai" else None
    resolved_block_concurrency = resolve_commentary_block_concurrency(commentary_block_concurrency)
    resolved_background_music = _resolve_background_music_track(background_music_track) if background_music_enabled else None
    resolved_background_music_volume = max(0.0, min(float(background_music_volume or 0.0), 1.0))

    def log(message: str) -> None:
        _raise_if_commentary_cancelled()
        print(f"[Commentary] {message}")
        if progress:
            progress(message)

    os.makedirs(output_dir, exist_ok=True)
    _raise_if_commentary_cancelled()
    if openai_sampling_options:
        log(
            "OpenAI-compatible sampling settings: "
            f"interval={openai_sampling_options['frame_interval_seconds']}s, "
            f"max_frames={openai_sampling_options['max_frames']}, "
            f"scene_max_keyframes={openai_sampling_options['scene_max_keyframes']}, "
            f"batch_size={openai_sampling_options['batch_size']}, "
            f"visual_concurrency={openai_sampling_options['visual_concurrency']}"
        )
    log(f"Commentary block generation concurrency: {resolved_block_concurrency}")
    if resolved_background_music:
        log(
            f"Background music enabled: {resolved_background_music['label']} "
            f"({resolved_background_music['title']} - {resolved_background_music['artist']}), "
            f"volume {resolved_background_music_volume:.2f}."
        )
    if analysis_fallback_reason:
        log(analysis_fallback_reason)
    log("Preparing source video...")
    _raise_if_commentary_cancelled()

    video_path = None
    analysis_video_path = None
    has_reusable_gemini_file = bool((gemini_file or {}).get("uri") or (gemini_file or {}).get("file_uri"))

    if source_type == "url":
        video_path, video_title = download_youtube_video(source, output_dir, quality="high")
        if checkpoint:
            checkpoint({"source_path": video_path, "source_filename": os.path.basename(video_path)})
        if analysis_mode == "video" and not has_reusable_gemini_file:
            log("Downloading low-resolution analysis video for Gemini video input...")
            try:
                analysis_video_path, _ = download_youtube_video(
                    source,
                    output_dir,
                    quality="low",
                    filename_suffix="_analysis_low",
                )
            except Exception as exc:
                raise Exception(
                    "Gemini 视频输入模式需要下载低清分析视频，但低清视频下载失败。"
                    "请检查 YouTube cookies，或改用当前模式。"
                ) from exc
    else:
        if not os.path.exists(source):
            raise Exception(f"Input video not found: {source}")
        video_title = os.path.splitext(os.path.basename(source))[0]
        video_path = os.path.join(output_dir, os.path.basename(source))
        if os.path.abspath(source) != os.path.abspath(video_path):
            shutil.copyfile(source, video_path)
        if checkpoint:
            checkpoint({"source_path": video_path, "source_filename": os.path.basename(video_path)})
        if analysis_mode == "video":
            analysis_video_path = video_path

    if analysis_mode == "video":
        _raise_if_commentary_cancelled()
        reusable_analysis_path = prepared_analysis_video_path if prepared_analysis_video_path and os.path.exists(prepared_analysis_video_path) else None
        if reusable_analysis_path:
            analysis_video_path = reusable_analysis_path
            log("Reusing compressed Gemini analysis video from previous task...")
        elif has_reusable_gemini_file:
            log("Reusing processed Gemini analysis video from previous task...")
        else:
            if not analysis_video_path:
                raise Exception("Missing analysis video path for Gemini video input mode")
            log("Preparing complete Gemini analysis video for Files API upload...")
            analysis_video_path = _prepare_analysis_video_for_gemini(analysis_video_path, output_dir, progress=log)
        if checkpoint and analysis_video_path:
            checkpoint({
                "analysis_video_path": analysis_video_path,
                "analysis_video_filename": os.path.basename(analysis_video_path),
            })

    video_info = _get_video_info(video_path)
    _raise_if_commentary_cancelled()
    duration = float(video_info.get("duration") or 0)
    resolved_aspect = "9:16" if vertical else _resolve_aspect_mode(video_info, aspect_mode)
    log(f"Resolved output aspect ratio: {resolved_aspect}")

    cached_script_path = None
    if checkpoint:
        task_script_path = os.path.join(output_dir, "commentary_task.json")
        try:
            with open(task_script_path, "r", encoding="utf-8") as f:
                cached_script_path = (json.load(f) or {}).get("script_path")
        except Exception:
            cached_script_path = None
    if _previous_error_invalidates_cached_script(previous_error):
        cached_script_path = None

    frame_paths = []
    openai_frame_infos = []
    openai_analysis_video_path = None
    openai_visual_analysis = None
    openai_edited_visual_analysis = None
    openai_intermediate_edit_path = None
    openai_source_edit_timeline = []
    render_video_path = None
    render_duration = None
    openai_source_audio_analysis = None
    openai_audio_probe = None
    using_rendered_cached_script = False
    source_has_spoken_commentary = False
    effective_original_audio_volume = max(0.0, min(float(original_audio_volume or 0.0), 1.0))
    effective_pause_original_audio_volume = max(0.0, min(float(pause_original_audio_volume or 0.0), 1.0))
    if cached_script_path and os.path.exists(cached_script_path):
        log("Reusing cached commentary script from saved task checkpoint...")
        with open(cached_script_path, "r", encoding="utf-8") as f:
            cached_payload = json.load(f)
        script = cached_payload.get("script") or cached_payload
        transcript = cached_payload.get("transcript") or {
            "text": "",
            "segments": [],
            "language": source_language or "unknown",
        }
        source_has_spoken_commentary = _transcript_has_source_commentary(transcript)
        if (
            not source_has_spoken_commentary
            and _should_check_source_commentary_for_audio_muting(
                effective_original_audio_volume,
                effective_pause_original_audio_volume,
                target_duration,
            )
            and _has_audio_stream(video_path)
        ):
            try:
                transcript = _load_or_transcribe_commentary_transcript(
                    output_dir,
                    video_path,
                    source_language=source_language,
                    progress=log,
                    checkpoint=checkpoint,
                    cached_message="Reusing cached Faster-Whisper transcript for source-commentary audio check...",
                    transcribe_message="Transcribing source audio with Faster-Whisper for original-commentary muting check...",
                )
                source_has_spoken_commentary = _transcript_has_source_commentary(transcript)
            except Exception as exc:
                log(f"Could not transcribe source audio for original-commentary muting check: {str(exc)[:300]}")
        if source_has_spoken_commentary:
            cached_metadata_path = os.path.splitext(cached_script_path)[0].replace("_script", "_metadata") + ".json"
            cached_metadata = {}
            if os.path.exists(cached_metadata_path):
                try:
                    with open(cached_metadata_path, "r", encoding="utf-8") as f:
                        cached_metadata = json.load(f) or {}
                except Exception:
                    cached_metadata = {}
            if cached_metadata.get("source_has_spoken_commentary") is not True:
                log("Cached commentary script was generated before source-commentary audio muting; regenerating script.")
                cached_script_path = None
                using_rendered_cached_script = False
                script = None
                transcript = {
                    "text": "",
                    "segments": [],
                    "language": source_language or "unknown",
                }
                source_has_spoken_commentary = False
        try:
            if cached_script_path:
                _normalize_script_timeline(script, duration, target_duration, language)
                using_rendered_cached_script = target_duration == "full" and _is_rendered_cached_full_mode_script(script)
                _validate_rendered_cached_full_mode_script(
                    script,
                    duration,
                    target_duration,
                    language,
                    custom_style_prompt=custom_style_prompt,
                )
        except Exception as exc:
            log(f"Cached commentary script failed current sync validation; regenerating script: {exc}")
            cached_script_path = None
            using_rendered_cached_script = False
    else:
        cached_script_path = None

    if not cached_script_path:
        _raise_if_commentary_cancelled()
        if analysis_mode in {"current", "openai"}:
            transcript = _load_or_transcribe_commentary_transcript(
                output_dir,
                video_path,
                source_language=source_language,
                progress=log,
                checkpoint=checkpoint,
            )
            source_has_spoken_commentary = _transcript_has_source_commentary(transcript)
            if analysis_mode == "current":
                log("Extracting keyframes for visual context...")
                frame_paths = _extract_keyframes(video_path, output_dir, duration)
            else:
                if _has_audio_stream(video_path):
                    log("Testing whether the configured OpenAI-compatible model can inspect source audio...")
                    openai_audio_probe = _probe_openai_audio_analysis_support(
                        api_key=openai_key or "",
                        base_url=openai_base_url or "",
                        model=openai_model or "",
                        video_path=video_path,
                        output_dir=output_dir,
                        duration=duration,
                        transcript=transcript,
                        progress=log,
                    )
                    if openai_audio_probe.get("supported"):
                        log("OpenAI-compatible source audio analysis is supported; analyzing original narration audio...")
                        openai_source_audio_analysis = _analyze_openai_source_audio(
                            api_key=openai_key or "",
                            base_url=openai_base_url or "",
                            model=openai_model or "",
                            video_path=video_path,
                            output_dir=output_dir,
                            duration=duration,
                            audio_mode=openai_audio_probe.get("mode") or "input_audio",
                            transcript=transcript,
                            progress=log,
                        )
                        source_has_spoken_commentary = (
                            source_has_spoken_commentary
                            or _source_audio_analysis_has_spoken_commentary(openai_source_audio_analysis)
                        )
                    else:
                        log(
                            "Configured OpenAI-compatible model did not confirm source audio support; "
                            "using Faster-Whisper transcript as the audio-understanding fallback."
                        )
                openai_analysis_video_path = _prepare_analysis_video_for_openai_frames(
                    video_path,
                    output_dir,
                    progress=log,
                )
                analysis_video_path = openai_analysis_video_path
                if checkpoint:
                    checkpoint({
                        "analysis_video_path": openai_analysis_video_path,
                        "analysis_video_filename": os.path.basename(openai_analysis_video_path),
                    })
                log("Extracting dense timestamped frames for OpenAI-compatible multimodal analysis...")
                openai_frame_infos = _extract_openai_analysis_frames(
                    video_path,
                    output_dir,
                    duration,
                    progress=log,
                    sampling_options=openai_sampling_options,
                    extraction_video_path=openai_analysis_video_path,
                )
        else:
            transcript = {
                "text": "",
                "segments": [],
                "language": source_language or "unknown",
            }
            source_has_spoken_commentary = False
            if (
                _should_check_source_commentary_for_audio_muting(
                    effective_original_audio_volume,
                    effective_pause_original_audio_volume,
                    target_duration,
                )
                and _has_audio_stream(video_path)
            ):
                try:
                    transcript = _load_or_transcribe_commentary_transcript(
                        output_dir,
                        video_path,
                        source_language=source_language,
                        progress=log,
                        checkpoint=checkpoint,
                        cached_message="Reusing cached Faster-Whisper transcript for source-commentary audio check...",
                        transcribe_message="Transcribing source audio with Faster-Whisper to detect original commentary before render...",
                    )
                    source_has_spoken_commentary = _transcript_has_source_commentary(transcript)
                except Exception as exc:
                    log(f"Could not transcribe source audio for original-commentary muting check: {str(exc)[:300]}")
                    transcript = {
                        "text": "",
                        "segments": [],
                        "language": source_language or "unknown",
                    }
            else:
                log("Skipping Faster-Whisper transcription; Gemini will analyze the attached video directly...")

        if analysis_mode == "openai":
            _raise_if_commentary_cancelled()
            log(
                "Generating OpenAI-compatible edit-first commentary: "
                "full-video analysis, intermediate visual cut, then final narration on the edited video..."
            )
            (
                script,
                openai_visual_analysis,
                openai_edited_visual_analysis,
                openai_intermediate_edit_path,
                openai_source_edit_timeline,
                render_duration,
            ) = _openai_generate_edit_first_commentary_script(
                transcript=transcript,
                video_title=video_title,
                duration=duration,
                source_video_path=video_path,
                output_dir=output_dir,
                openai_key=openai_key or "",
                openai_base_url=openai_base_url or "",
                openai_model=openai_model or "",
                frame_infos=openai_frame_infos,
                language=language,
                style=style,
                custom_style_prompt=custom_style_prompt,
                target_duration=target_duration,
                aspect_mode=resolved_aspect,
                progress=log,
                openai_sampling_options=openai_sampling_options,
                source_audio_analysis=openai_source_audio_analysis,
                auto_video_speed=auto_video_speed,
                checkpoint=checkpoint,
            )
            if openai_intermediate_edit_path:
                render_video_path = openai_intermediate_edit_path
        else:
            _raise_if_commentary_cancelled()
            log("Generating original commentary script with Gemini...")
            try:
                script = generate_commentary_script(
                    transcript=transcript,
                    video_title=video_title,
                    duration=duration,
                    gemini_key=gemini_key,
                    language=language,
                    style=style,
                    custom_style_prompt=custom_style_prompt,
                    target_duration=target_duration,
                    base_url=gemini_base_url,
                    frame_paths=frame_paths,
                    analysis_video_path=analysis_video_path if analysis_mode == "video" else None,
                    analysis_mode=analysis_mode,
                    gemini_model=gemini_model,
                    gemini_pool=gemini_pool,
                    progress=log,
                    checkpoint=checkpoint,
                    gemini_file=gemini_file if analysis_mode == "video" else None,
                    previous_error=previous_error,
                )
            except Exception as exc:
                fallback_reason = (
                    _gemini_video_proxy_timeout_fallback_reason(str(exc))
                    if analysis_mode == "video"
                    else None
                )
                if not fallback_reason:
                    raise
                analysis_fallback_reason = fallback_reason
                analysis_mode = "current"
                log(fallback_reason)
                transcript = _load_or_transcribe_commentary_transcript(
                    output_dir,
                    video_path,
                    source_language=source_language,
                    progress=log,
                    checkpoint=checkpoint,
                    cached_message="Reusing cached Faster-Whisper transcript for Gemini fallback...",
                    transcribe_message="Transcribing full video with Faster-Whisper for Gemini fallback...",
                )
                source_has_spoken_commentary = _transcript_has_source_commentary(transcript)
                log("Extracting keyframes for Gemini fallback visual context...")
                frame_paths = _extract_keyframes(video_path, output_dir, duration)
                log("Generating original commentary script with Gemini fallback inputs...")
                script = generate_commentary_script(
                    transcript=transcript,
                    video_title=video_title,
                    duration=duration,
                    gemini_key=gemini_key,
                    language=language,
                    style=style,
                    custom_style_prompt=custom_style_prompt,
                    target_duration=target_duration,
                    base_url=gemini_base_url,
                    frame_paths=frame_paths,
                    analysis_video_path=None,
                    analysis_mode="current",
                    gemini_model=gemini_model,
                    gemini_pool=gemini_pool,
                    progress=log,
                    checkpoint=checkpoint,
                    gemini_file=None,
                    previous_error=previous_error,
                )

    if source_has_spoken_commentary:
        if effective_original_audio_volume > 0 or effective_pause_original_audio_volume > 0:
            log(
                "Source video contains spoken commentary; muting original source audio in the final remix "
                "so the old narration does not overlap the new voiceover."
            )
        effective_original_audio_volume = 0.0
        effective_pause_original_audio_volume = 0.0
        for block in (script.get("narration_blocks") or []):
            if isinstance(block, dict) and bool(block.get("pause")):
                block["source_audio_muted"] = True

    _raise_if_commentary_cancelled()
    active_video_path = render_video_path or video_path
    active_duration = float(render_duration or _get_video_duration(active_video_path) or duration)
    active_visual_analysis = openai_edited_visual_analysis if openai_edited_visual_analysis else openai_visual_analysis
    active_target_duration = "full" if openai_intermediate_edit_path else target_duration
    active_auto_video_speed = False if openai_intermediate_edit_path else auto_video_speed
    _normalize_script_timeline(script, active_duration, active_target_duration, language)
    if not using_rendered_cached_script:
        _sanitize_generated_commentary_script(script, language)
        _sync_openai_locked_script_for_validation(
            script,
            active_duration,
            active_target_duration,
            language,
            active_visual_analysis,
        )
    if using_rendered_cached_script:
        _validate_rendered_cached_full_mode_script(
            script,
            active_duration,
            active_target_duration,
            language,
            custom_style_prompt=custom_style_prompt,
        )
    else:
        _validate_commentary_script_for_target(
            script,
            active_duration,
            active_target_duration,
            language,
            visual_analysis=active_visual_analysis,
            custom_style_prompt=custom_style_prompt,
        )
    if script.get("narration_blocks") and not using_rendered_cached_script:
        script["narration_blocks"] = _apply_auto_video_speed_to_blocks(
            script.get("narration_blocks") or [],
            active_auto_video_speed,
            visual_analysis=active_visual_analysis,
        )
        script["narration_blocks"] = _protect_full_mode_visual_budget_after_speed(
            script.get("narration_blocks") or [],
            active_duration,
            active_target_duration,
        )
        script["edit_segments"] = _narration_blocks_to_edit_segments(script["narration_blocks"])
        _sync_openai_locked_script_for_validation(
            script,
            active_duration,
            active_target_duration,
            language,
            active_visual_analysis,
        )
        _validate_commentary_script_for_target(
            script,
            active_duration,
            active_target_duration,
            language,
            visual_analysis=active_visual_analysis,
            custom_style_prompt=custom_style_prompt,
        )
    if not using_rendered_cached_script:
        _finalize_full_mode_narration_blocks_for_render(script, active_duration, active_target_duration, language)
        _sync_openai_locked_script_for_validation(
            script,
            active_duration,
            active_target_duration,
            language,
            active_visual_analysis,
        )
    if script.get("narration_blocks"):
        if using_rendered_cached_script:
            _validate_rendered_cached_full_mode_script(
                script,
                active_duration,
                active_target_duration,
                language,
                custom_style_prompt=custom_style_prompt,
            )
        else:
            _validate_commentary_script_for_target(
                script,
                active_duration,
                active_target_duration,
                language,
                visual_analysis=active_visual_analysis,
                custom_style_prompt=custom_style_prompt,
            )
        script["narration_blocks"] = _strip_auto_filled_user_visible_fields(script.get("narration_blocks") or [])
        script["edit_segments"] = _narration_blocks_to_edit_segments(script["narration_blocks"])
        script["narration"] = _narration_from_blocks({"narration_blocks": script["narration_blocks"]}) or str(script.get("narration") or "")
        _strip_internal_narration_block_fields(script)

    slug = _safe_slug(script.get("title") or video_title)
    script_path = os.path.join(output_dir, f"{slug}_commentary_script.json")
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump({"script": script, "transcript": transcript}, f, ensure_ascii=False, indent=2)
    if checkpoint:
        checkpoint({"script_path": script_path})

    voiceover_path = os.path.join(output_dir, f"{slug}_voiceover.mp3")
    narration_blocks = _normalize_narration_blocks(script.get("narration_blocks") or [], active_duration)
    if narration_blocks:
        narration_blocks = _normalize_narration_blocks(script.get("narration_blocks") or [], active_duration)
        script["narration_blocks"] = narration_blocks
        script["edit_segments"] = _narration_blocks_to_edit_segments(narration_blocks)
        script["narration"] = _narration_from_blocks({"narration_blocks": narration_blocks}) or str(script.get("narration") or "")
        if using_rendered_cached_script:
            _validate_rendered_cached_full_mode_script(
                script,
                active_duration,
                active_target_duration,
                language,
                custom_style_prompt=custom_style_prompt,
            )
    auto_video_speed_summary = _summarize_auto_video_speed(narration_blocks, active_auto_video_speed)
    use_block_synced_render = bool(narration_blocks)
    if use_block_synced_render:
        _raise_if_commentary_cancelled()
        if not active_auto_video_speed:
            log("AI auto video speed disabled; rendering all commentary blocks at 1.0x.")
        elif auto_video_speed_summary["accelerated_count"] > 0:
            log(
                "AI auto video speed: "
                f"{auto_video_speed_summary['accelerated_count']}/{auto_video_speed_summary['total_blocks']} blocks accelerated, "
                f"saved about {auto_video_speed_summary['saved_seconds']:.1f}s."
            )
        else:
            log("AI auto video speed: AI returned no accelerated blocks; rendering all blocks at 1.0x.")
        voiceover_path = os.path.join(output_dir, f"{slug}_voiceover.m4a")
        edit_segments = _narration_blocks_to_edit_segments(narration_blocks)
    else:
        edit_segments = _require_ai_selected_edit_segments(script, active_duration, active_target_duration)

    work_dir = os.path.join(output_dir, f"{slug}_work")
    os.makedirs(work_dir, exist_ok=True)
    edited_video_path = os.path.join(output_dir, f"{slug}_edited_visual.mp4")
    timed_video_path = os.path.join(output_dir, f"{slug}_timed_visual.mp4")
    ambient_audio_path = os.path.join(output_dir, f"{slug}_ambient.m4a")
    background_music_bed_path = os.path.join(output_dir, f"{slug}_background_music.m4a") if resolved_background_music else None
    trim_to_voiceover = True
    preserve_source_resolution = use_block_synced_render and active_target_duration == "full"
    synced_block_durations = []

    if use_block_synced_render:
        _raise_if_commentary_cancelled()
        log(f"Generating {len(narration_blocks)} timestamp-synced commentary blocks with {tts_provider} TTS...")
        ambient_audio, synced_block_durations = _create_block_synced_visuals_and_audio(
            video_path=active_video_path,
            narration_blocks=narration_blocks,
            timed_video_path=timed_video_path,
            voiceover_path=voiceover_path,
            ambient_audio_path=ambient_audio_path,
            aspect_mode=resolved_aspect,
            work_dir=work_dir,
            tts_provider=tts_provider,
            language=language,
            elevenlabs_key=elevenlabs_key,
            voice_id=voice_id,
            edge_voice=edge_voice,
            original_audio_volume=effective_original_audio_volume,
            pause_original_audio_volume=effective_pause_original_audio_volume,
            preserve_source_resolution=preserve_source_resolution,
            block_concurrency=resolved_block_concurrency,
            trim_short_tts_tails=active_target_duration != "full",
            progress=log,
        )
        script["narration_blocks"] = narration_blocks
        script["edit_segments"] = _narration_blocks_to_edit_segments(narration_blocks)
        script["narration"] = _narration_from_blocks({"narration_blocks": narration_blocks}) or str(script.get("narration") or "")
        edit_segments = script["edit_segments"]
        auto_video_speed_summary = _summarize_auto_video_speed(narration_blocks, active_auto_video_speed)
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump({"script": script, "transcript": transcript}, f, ensure_ascii=False, indent=2)
        edited_video_path = timed_video_path
        if not synced_block_durations:
            _validate_voiceover_duration_for_target(voiceover_path, edit_segments, active_duration, active_target_duration)
    else:
        _raise_if_commentary_cancelled()
        log(f"Generating commentary voiceover with {tts_provider} TTS...")
        generate_commentary_voiceover(
            text=script["narration"],
            output_path=voiceover_path,
            tts_provider=tts_provider,
            language=language,
            elevenlabs_key=elevenlabs_key,
            voice_id=voice_id,
            edge_voice=edge_voice,
        )
        _validate_voiceover_duration_for_target(voiceover_path, edit_segments, active_duration, active_target_duration)
        log(f"Creating AI-selected visual edit with {len(edit_segments)} kept segments...")
        _create_visual_edit(
            active_video_path,
            edit_segments,
            edited_video_path,
            resolved_aspect,
            work_dir,
            preserve_source_resolution=preserve_source_resolution,
        )
        if active_target_duration == "full":
            log("Skipping full-length visual retiming because no timestamped narration blocks were returned...")
            timed_video_path = edited_video_path
        else:
            log("Aligning edited visuals to the voiceover duration...")
            _fit_video_to_voiceover(edited_video_path, voiceover_path, timed_video_path)
        if effective_original_audio_volume > 0:
            log("Preparing low-volume original audio bed as ambient sound...")
        else:
            log("Skipping original source audio bed for this commentary remix.")
        ambient_audio = _create_ambient_audio_bed(
            active_video_path,
            edit_segments,
            ambient_audio_path,
            effective_original_audio_volume,
            work_dir,
        )
    background_music_bed = None
    _raise_if_commentary_cancelled()
    if resolved_background_music and background_music_bed_path and resolved_background_music_volume > 0:
        background_duration = _get_audio_duration(voiceover_path) if trim_to_voiceover else _get_video_duration(timed_video_path)
        log(f"Preparing background music bed: {resolved_background_music['label']}...")
        background_music_bed = _create_background_music_bed(
            resolved_background_music["path"],
            background_music_bed_path,
            background_duration,
            resolved_background_music_volume,
        )
    mixed_path = os.path.join(output_dir, f"{slug}_mixed.mp4")
    mix_has_source_audio = bool(ambient_audio and os.path.exists(ambient_audio))
    if background_music_bed and mix_has_source_audio:
        log("Mixing new voiceover with ambient source audio and background music...")
    elif background_music_bed:
        log("Mixing new voiceover with background music...")
    elif mix_has_source_audio:
        log("Mixing new voiceover with ambient source audio...")
    else:
        log("Mixing new voiceover only...")
    _mix_voiceover_with_video(
        video_path=timed_video_path,
        voiceover_path=voiceover_path,
        output_path=mixed_path,
        original_audio_volume=effective_original_audio_volume,
        ambient_audio_path=ambient_audio,
        background_music_path=background_music_bed,
        trim_to_voiceover=trim_to_voiceover,
    )

    subtitle_path = None
    final_path = mixed_path
    if subtitles:
        _raise_if_commentary_cancelled()
        subtitle_path = os.path.join(output_dir, f"{slug}_commentary.ass")
        subtitled_path = os.path.join(output_dir, f"{slug}_final.mp4")
        log("Generating text-timed subtitles from the commentary narration...")
        subtitle_dimensions = _probe_video_dimensions(mixed_path)
        log(f"Subtitle canvas size: {subtitle_dimensions[0]}x{subtitle_dimensions[1]}")
        if use_block_synced_render:
            _write_block_timed_ass(narration_blocks, subtitle_path, synced_block_durations, video_dimensions=subtitle_dimensions)
        else:
            _write_text_timed_ass(script["narration"], voiceover_path, subtitle_path, video_dimensions=subtitle_dimensions)
        log("Burning subtitles into final video...")
        _burn_subtitles(mixed_path, subtitle_path, subtitled_path)
        final_path = subtitled_path

    if use_block_synced_render and synced_block_durations:
        _assert_full_mode_output_alignment(
            final_path,
            voiceover_path,
            subtitle_path,
            synced_block_durations,
            active_duration,
            narration_blocks=narration_blocks,
        )
        _assert_non_full_output_duration_target(synced_block_durations, active_duration, active_target_duration)

    episode_plan = script.get("episode_plan") if isinstance(script.get("episode_plan"), dict) else {"should_split": False, "reason": ""}
    commentary_episodes = script.get("episodes") if isinstance(script.get("episodes"), list) else []
    rendered_episodes = []
    if active_target_duration == "full" and narration_blocks and commentary_episodes and episode_plan.get("should_split"):
        _raise_if_commentary_cancelled()
        log(f"Rendering {len(commentary_episodes)} AI-planned commentary episodes while keeping the complete video...")
        rendered_episodes = _render_commentary_episode_videos(
            final_path=final_path,
            episodes=commentary_episodes,
            narration_blocks=narration_blocks,
            block_durations=synced_block_durations,
            output_dir=output_dir,
            slug=slug,
            progress=log,
        )

    publish_fields = _build_douyin_publish_fields(script)
    _raise_if_commentary_cancelled()
    log("Generating local commentary cover images from the current task video frame/source thumbnail...")
    cover_source_path = next(
        (
            path for path in (mixed_path, timed_video_path, edited_video_path, video_path, final_path)
            if path and os.path.exists(path)
        ),
        final_path,
    )
    cover_fields = _generate_commentary_covers(
        cover_source_path,
        output_dir,
        slug,
        _get_video_duration(cover_source_path) or duration,
        cover_title=publish_fields.get("publish_title") or script.get("title") or slug,
        source_type=source_type,
        source_url=source if source_type == "url" else None,
    )

    final_duration = _probe_media_format_duration(final_path) or _get_video_duration(final_path) or sum(synced_block_durations or []) or duration
    openai_edit_first_metadata = script.get("_openai_edit_first") if analysis_mode == "openai" else None
    if openai_intermediate_edit_path:
        openai_edit_first_metadata = {
            **(openai_edit_first_metadata if isinstance(openai_edit_first_metadata, dict) else {}),
            "enabled": True,
            "intermediate_edit_path": openai_intermediate_edit_path,
            "source_edit_timeline": openai_source_edit_timeline,
            "source_target_duration": target_duration,
            "render_target_duration": active_target_duration,
        }

    metadata = {
        "video_path": final_path,
        "video_filename": os.path.basename(final_path),
        "video_url": f"/videos/{os.path.basename(output_dir)}/{os.path.basename(final_path)}",
        "source_video": os.path.basename(video_path),
        "analysis_mode": analysis_mode,
        "requested_analysis_mode": requested_analysis_mode,
        "analysis_fallback_reason": analysis_fallback_reason,
        "analysis_video": os.path.basename(analysis_video_path) if analysis_video_path else None,
        "gemini_model": resolved_gemini_model if analysis_mode != "openai" else None,
        "gemini_events": gemini_pool.event_dicts() if gemini_pool else [],
        "openai_model": openai_model if analysis_mode == "openai" else None,
        "openai_analysis": script.get("_openai_analysis") if analysis_mode == "openai" else None,
        "openai_sampling_options": openai_sampling_options if analysis_mode == "openai" else None,
        "openai_edit_first": openai_edit_first_metadata,
        "openai_intermediate_edit": os.path.basename(openai_intermediate_edit_path) if openai_intermediate_edit_path else None,
        "openai_edited_visual_analysis": bool(openai_edited_visual_analysis) if analysis_mode == "openai" else None,
        "openai_source_edit_timeline": openai_source_edit_timeline if analysis_mode == "openai" else [],
        "openai_audio_probe": openai_audio_probe if analysis_mode == "openai" else None,
        "openai_source_audio_analysis": openai_source_audio_analysis if analysis_mode == "openai" else None,
        "commentary_block_concurrency": resolved_block_concurrency,
        "auto_video_speed": bool(auto_video_speed),
        "auto_video_speed_summary": auto_video_speed_summary,
        "edited_visual": os.path.basename(edited_video_path),
        "timed_visual": os.path.basename(timed_video_path),
        "ambient_audio": os.path.basename(ambient_audio) if ambient_audio else None,
        "background_music_enabled": bool(resolved_background_music),
        "background_music_track": resolved_background_music["id"] if resolved_background_music else None,
        "background_music_label": resolved_background_music["label"] if resolved_background_music else None,
        "background_music_audio": os.path.basename(background_music_bed) if background_music_bed else None,
        "background_music_volume": resolved_background_music_volume if resolved_background_music else 0,
        "voiceover": os.path.basename(voiceover_path),
        "tts_provider": tts_provider,
        "voice": edge_voice or voice_id,
        "subtitle": os.path.basename(subtitle_path) if subtitle_path else None,
        "subtitle_block_durations": synced_block_durations,
        "script_path": os.path.basename(script_path),
        "duration": final_duration,
        "source_duration": duration,
        "render_source_video": os.path.basename(active_video_path) if active_video_path else None,
        "render_source_duration": active_duration,
        "output_aspect": resolved_aspect,
        "source_has_spoken_commentary": source_has_spoken_commentary,
        "original_audio_volume": effective_original_audio_volume,
        "pause_original_audio_volume": effective_pause_original_audio_volume,
        "requested_original_audio_volume": original_audio_volume,
        "requested_pause_original_audio_volume": pause_original_audio_volume,
        "edit_segments": edit_segments,
        "title": script.get("title"),
        "summary": script.get("summary"),
        "hook": script.get("hook"),
        "narration": script.get("narration"),
        "narration_blocks": narration_blocks,
        "episode_plan": episode_plan,
        "episodes": rendered_episodes,
        "chapters": script.get("chapters", []),
        "cut_strategy": script.get("cut_strategy", []),
        "hashtags": script.get("hashtags", []),
        **publish_fields,
        **cover_fields,
    }
    metadata_path = os.path.join(output_dir, f"{slug}_commentary_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    if checkpoint:
        checkpoint({"metadata_path": metadata_path})

    log("Commentary remix video completed.")
    return metadata
