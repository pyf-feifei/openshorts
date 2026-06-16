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
OPENAI_ANALYSIS_FRAMES_MANIFEST = "openai_analysis_frames_manifest.json"
OPENAI_VISUAL_ANALYSIS_CACHE = "openai_visual_analysis.json"
OPENAI_VISUAL_PROMPT_MAX_CHARS = max(10000, int(os.environ.get("OPENSHORTS_OPENAI_VISUAL_PROMPT_MAX_CHARS", "45000")))
OPENAI_TEMPERATURE = float(os.environ.get("OPENSHORTS_OPENAI_TEMPERATURE", "0.7"))
OPENAI_STRICT_SCRIPT_SCHEMA = os.environ.get(
    "OPENSHORTS_OPENAI_STRICT_SCRIPT_SCHEMA",
    "true",
).strip().lower() not in {"0", "false", "no", "off"}
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
COMMENTARY_FINAL_COMPLETION_NARRATION_PATTERNS = (
    re.compile(r"(?:装好|做好|做完|完工|带走|收工|搞定|结束|完成|收尾)"),
    re.compile(r"\b(?:finished|done|complete|wrapped up|job is done)\b", re.I),
)
COMMENTARY_PACKING_NARRATION_PATTERNS = (
    re.compile(r"(?:装满|装袋|打包|封箱|装箱|袋口[^。！？!?]{0,8}(?:拧紧|扎紧|收紧|绑紧|封住|封好)|(?:装进|装入|放进|放入|塞进)[^。！？!?]{0,8}(?:袋|箱|盒|桶|容器))"),
    re.compile(r"\b(?:packed|packing|bagged|bagging|boxed|boxing|wrapped|wrapping|put into (?:a )?(?:bag|box|container)|placed into (?:a )?(?:bag|box|container))\b", re.I),
)
COMMENTARY_COMPLETION_NARRATION_PATTERNS = (
    *COMMENTARY_FINAL_COMPLETION_NARRATION_PATTERNS,
    *COMMENTARY_PACKING_NARRATION_PATTERNS,
)
COMMENTARY_COMPLETION_VISUAL_KEYWORDS = (
    "finished", "finish", "final", "complete", "completed", "completion", "done", "result",
    "output", "payoff", "ending", "conclusion", "before and after", "after result",
    "final product", "completed product", "ready", "working", "tested", "assembled",
    "installed", "secured", "sealed", "closed", "packed", "packing",
    "收工", "装好", "做好", "做完", "完工", "完成", "收尾", "结束", "结果", "最终",
    "成品", "成果", "效果", "安装好", "组装好", "固定好", "封好", "测试完成",
)
COMMENTARY_FINAL_COMPLETION_VISUAL_KEYWORDS = (
    "finished", "finish", "final", "complete", "completed", "completion", "done",
    "job is done", "result", "output", "payoff", "ending", "conclusion",
    "before and after", "after result", "final product", "completed product",
    "ready", "working", "tested", "assembled", "installed", "secured", "securing",
    "sealed", "closed", "tightened", "locked", "fixed in place", "cleaned up",
    "收工", "装好", "做好", "做完", "完工", "完成", "收尾", "结束", "最终", "结果",
    "成品", "成果", "效果", "安装好", "组装好", "固定好", "封好", "测试完成",
)
COMMENTARY_PACKING_VISUAL_KEYWORDS = (
    "package", "packaging", "bagging", "packing", "packed", "boxed", "wrapped",
    "placed into bag", "placed into a bag", "put into bag", "put into a bag",
    "placed into box", "placed into a box", "put into box", "put into a box",
    "placed into container", "sealed package", "loaded into bag", "loaded into box",
    "bag is held close", "bag is tied", "bag is sealed", "bag mouth", "袋口",
    "装袋", "装箱", "装盒", "装进袋", "装进箱", "装进盒",
    "放进袋", "放进箱", "放进盒", "塞进袋", "塞进箱", "打包", "包装", "封箱",
    "放入袋", "放入箱", "放入盒", "装入袋", "装入箱", "装入盒", "绑紧袋", "扎紧袋",
)
COMMENTARY_PACKING_CONTAINER_KEYWORDS = (
    "bag", "bags", "plastic bag", "sack", "box", "boxes", "carton", "container",
    "袋子", "塑料袋", "箱子", "纸箱", "盒子", "容器",
)
COMMENTARY_PACKING_ACTION_KEYWORDS = (
    "placed", "put into", "loaded", "sealed", "tied", "closed", "secured",
    "装", "放进", "放入", "塞进", "打包", "包装", "封", "绑紧", "扎紧", "收紧", "固定",
)
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


def _target_duration_hint(mode: str, source_duration: float, target_seconds: Optional[float] = None) -> str:
    if mode == "short":
        return "AI must select only the most important visual moments and create a tight 60-90 second commentary edit; backend will not invent fallback kept ranges."
    if mode == "medium":
        return "AI must select enough important visual moments for a 3-5 minute commentary edit, but remove repetitive or low-value parts; backend will not invent fallback kept ranges."
    full_target = float(target_seconds) if target_seconds and target_seconds > 0 else _target_visual_duration_seconds(source_duration, "full")
    if _full_mode_preserves_source_process(source_duration, full_target):
        return (
            "Create a comprehensive full-process commentary edit. For this source length, preserve the complete visible workflow in chronological order, "
            "remove only clearly useless dead time, duplicate waiting, setup, walking, camera drift, or failed/irrelevant footage, and use video_speed for visibly slow or repetitive ranges instead of cutting away important process steps. "
            "AI must decide the kept source ranges and splice order from the visual evidence; backend will validate and render those ranges, not choose replacements. "
            "The visual analysis should be detailed; the narration should be concise, scene-matched, and allowed to leave breathing room."
        )
    return (
        "Create a comprehensive long-form commentary edit with an explicit editing strategy, not a raw full-length copy of the source. "
        f"For this source, select about {int(full_target)} seconds of useful original footage across the whole timeline, preserving the complete process arc while removing repetitive, slow, duplicated, waiting, setup, walking, camera drift, and low-value filler time. "
        "Do not preserve the entire source unless the source itself is already shorter than the target, and do not stretch a concise process into a long edit just to fill time. AI must decide the kept source ranges, skipped ranges, splice order, and video_speed from the visual evidence; backend will validate and render those ranges, not choose replacements. Use video_speed for slow-but-useful ranges; the narration must be selective, scene-matched, and leave visual breathing room instead of talking over every second."
    )


def _style_grounding_instruction(style: str, language: str) -> str:
    normalized = (style or "").strip().lower()
    if normalized in {"first_person_hustle", "first-person-hustle", "整活第一视角", "第一视角整活"}:
        if (language or "").lower().startswith("zh"):
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
        if (language or "").lower().startswith("zh"):
            return (
                "整活解说风格要求：保持第三人称或旁观者口播，不要装成正在参与动作的人。"
                "先说清楚当前画面正在发生什么，再用短促、有梗、带反差的方式吐槽或强化看点。"
                "梗必须来自当前可见的动作、表情、风险、工具、材料、环境或结果；口头禅全片最多点两次，禁止同一句反复刷屏。"
            )
        return (
            "Hustle commentary style: use an energetic observer voice, not first person. First describe the visible action, then add a short joke or punchy reaction grounded in the same timestamp range. Avoid repeating the same catchphrase."
        )
    if normalized in {"funny", "roast", "吐槽", "轻松吐槽"}:
        if (language or "").lower().startswith("zh"):
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
        "\n- Custom user style instruction: follow this additional style direction for wording, tone, pacing, and point of view, "
        "as long as it does not conflict with visual grounding, timeline sync, factuality, safety, or JSON schema rules: "
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
        normalized.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "reason": str(item.get("reason") or item.get("title") or "selected visual segment"),
        })
    normalized.sort(key=lambda segment: segment["start"])
    merged = []
    for segment in normalized:
        if merged and segment["start"] <= merged[-1]["end"] + 0.3:
            merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
            if segment.get("reason") and segment["reason"] not in merged[-1].get("reason", ""):
                merged[-1]["reason"] = f"{merged[-1]['reason']}; {segment['reason']}"
        else:
            merged.append(segment)
    return merged


def _target_visual_duration_seconds(source_duration: float, target_duration: str) -> float:
    duration = max(0.0, float(source_duration or 0.0))
    if target_duration == "short":
        return min(duration, 90.0)
    if target_duration == "medium":
        return min(duration, 300.0)
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
        parts = []
        for key in ("process_stage", "visual", "edit_value", "pace"):
            value = str(item.get(key) or "").strip()
            if value:
                parts.append(value)
        text = re.sub(r"\s+", " ", " / ".join(parts)).strip()
        if text:
            observations.append(text)
        if len(observations) >= limit:
            break
    return observations


def _candidate_segment_importance(segment: Dict, visual_analysis: Optional[Dict]) -> float:
    try:
        start = float(segment.get("start"))
        end = float(segment.get("end"))
    except (TypeError, ValueError):
        return 0.0
    score = 0.0
    reason = str(segment.get("reason") or "").lower()
    if re.search(
        r"must|core|payoff|final|result|reveal|finish|complete|output|effect|test|install|assembl|pack|secur|"
        r"完成|收工|结果|成品|成果|效果|测试|安装|组装|固定|包装|打包",
        reason,
    ):
        score += 1.0
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
    if observations:
        importances = []
        for item in observations:
            try:
                importances.append(float(item.get("importance") or 0.0))
            except (TypeError, ValueError):
                pass
            if bool(item.get("keep_candidate")):
                score += 0.35
            edit_value = str(item.get("edit_value") or "").lower()
            if edit_value == "must_keep":
                score += 1.0
            elif edit_value == "useful":
                score += 0.45
        if importances:
            score += max(importances) / 5.0
    return score


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
    text = f"{segment.get('reason') or ''} {segment.get('speed_reason') or ''}".lower()
    payoff_score = 1.0 if re.search(
        r"final|ending|result|payoff|complete|finish|secured?|pack|output|effect|test|install|assembl|reveal|"
        r"完成|收工|结果|成品|成果|效果|测试|安装|组装|固定|打包|包装",
        text,
    ) else 0.0
    duration = max(1.0, float(source_duration or 0.0))
    return (
        payoff_score,
        float(segment.get("importance") or 0.0),
        float(segment.get("end") or 0.0) / duration,
        -float(segment.get("playable_seconds") or 0.0),
    )


def _openai_plan_completion_guidance_text(visual_text: str) -> Tuple[bool, str]:
    text = str(visual_text or "")
    if _visual_supports_final_completion(text):
        return (
            True,
            "Completion wording is allowed only if it describes the visible final result, completed state, closure, test, installation, secured state, or clear ending action in this exact range.",
        )
    if _visual_supports_packing(text):
        return (
            False,
            "Packaging/container-loading action may be described, but do not say finished/done/收工/装好 unless a final result, completed state, closure, test, installation, secured state, or clear ending is visible.",
        )
    return (
        False,
        "No final completion evidence in this range; do not use finished/done/收工/装好/完成 wording here.",
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
    if target_duration != "full" or not visual_analysis:
        return None
    source_duration = max(0.0, float(duration or 0.0))
    target_seconds = _target_visual_duration_seconds_for_analysis(source_duration, target_duration, visual_analysis)
    if target_seconds <= 0 or _full_mode_preserves_source_process(source_duration, target_seconds):
        return None
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
        })
    if len(raw_segments) < 4:
        return None
    raw_segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    preferred_seconds = min(max_seconds - 4.0, max(min_seconds + 24.0, target_seconds))
    if preferred_seconds < min_seconds:
        preferred_seconds = min_seconds
    selected = []
    total = 0.0
    for segment in raw_segments:
        selected.append(segment)
        total += segment["playable_seconds"]
        if total >= preferred_seconds:
            break
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
        completion_allowed, completion_note = _openai_plan_completion_guidance_text(visual)
        speed_reason = segment["speed_reason"]
        if segment["video_speed"] > 1.0001 and not speed_reason:
            speed_reason = "AI visual analysis marked this exact range as slow or repetitive enough for acceleration"
        blocks.append({
            "index": index,
            "start": round(segment["start"], 3),
            "end": round(segment["end"], 3),
            "visual": visual or "AI-selected useful visual range",
            "visual_facts": observations[:3] or [segment["reason"] or "AI-selected useful visual range"],
            "evidence_timestamps": [
                round((segment["start"] + segment["end"]) / 2.0, 3),
            ],
            "pause": False,
            "video_speed": segment["video_speed"],
            "speed_reason": speed_reason,
            "playable_seconds": round(segment["playable_seconds"], 3),
            "min_narration_chars": _expected_narration_chars_for_visual_duration(segment["playable_seconds"], language),
            "completion_allowed": completion_allowed,
            "completion_note": completion_note,
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
- For TARGET DURATION full, write concise scene-matched commentary over the preserved workflow. The visual analysis should be detailed, but the spoken narration should be selective and natural, with short pause=true breathing-room blocks where the original picture or sound should carry the moment.
""".strip()
    return f"""
- For TARGET DURATION full, first analyze the complete source visual timeline from 0.0 seconds through {duration:.1f} seconds; do not stop after a short highlight scan, and do not summarize only the first few minutes.
- For TARGET DURATION full, produce a real edit decision list: select about {int(target_seconds)} seconds of useful visual ranges from the complete source timeline, and intentionally skip redundant or low-value ranges.
- For TARGET DURATION full, do not output one continuous 0-to-{duration:.1f} timeline. For this source, the selected playable visual duration should be near {int(target_seconds)} seconds, not {int(duration)} seconds.
- For TARGET DURATION full, preserve the complete process arc by keeping the beginning, middle, and ending payoff, but compress repeated manual actions, repeated tool operations, waiting, setup, walking, camera drift, and redundant close-ups with cuts and video_speed.
- For TARGET DURATION full, for example, compress repeated hammering, repeated climbing/setup motions, and other slow-but-useful process ranges with cuts or video_speed when the visual evidence supports it.
- For TARGET DURATION full, if the visual evidence shows the useful process is naturally tighter than the raw source, keep the edit tight instead of padding with weak ranges; the detailed analysis can be much longer than the spoken/kept edit.
- For TARGET DURATION full, write selective scene-matched commentary that covers the chosen visual ranges from start to finish across the full source timeline. Do not return a 60-second summary over a long source, but also do not talk over every second.
""".strip()


def _full_mode_regeneration_timeline_rules(duration: float, target_seconds: float) -> str:
    if _full_mode_preserves_source_process(duration, target_seconds):
        return (
            "- Keep chronological edit_segments that preserve the useful source workflow; remove only clearly useless dead time, duplicate waiting, setup, walking, camera drift, failed footage, or irrelevant ranges.\n"
            "- The final narration_blocks must include the beginning, middle, and ending portions of the source; for this source length, do not collapse the video into a much shorter highlights reel.\n"
            "- Let the AI decide video_speed from the visible action. Use video_speed above 1.0 for visibly slow or repetitive ranges instead of deleting meaningful process footage, and explain that decision in speed_reason.\n"
            "- Keep ordinary narrated blocks short enough for spoken commentary to cover them, usually 8-16s after video_speed. Split longer useful process ranges into multiple narrated blocks or explicit brief pause=true blocks."
        )
    return (
        f"- Keep chronological edit_segments that cover every major visible process stage across the source timeline while cutting repetitive, slow, duplicated, waiting, setup, walking, camera drift, and low-value filler ranges; selected playable visual time should be near {int(target_seconds)} seconds.\n"
        f"- The final narration_blocks must include selected source ranges from the beginning, middle, and later ending portion of the source; at least one block must end after {int(duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION)} seconds.\n"
        f"- Do not return a continuous near-full-source timeline; select about {int(target_seconds)} seconds of useful visuals, not {int(duration)} seconds. If useful visual evidence is naturally concise, do not pad the edit with weak ranges.\n"
        "- Keep ordinary narrated blocks short enough for spoken commentary to cover them, usually 8-16s after video_speed. Split longer useful process ranges into multiple narrated blocks or explicit brief pause=true blocks."
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
        if end - start < 1.0 or (not narration and not is_pause):
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
            "_completion_allowed",
            "_completion_note",
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
            "_completion_allowed",
            "_completion_note",
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
        }
        for block in blocks or []
    ]


def _commit_narration_blocks_to_script(data: Dict, blocks: List[Dict]) -> None:
    data["narration_blocks"] = blocks
    data["edit_segments"] = _narration_blocks_to_edit_segments(blocks)
    data["narration"] = _narration_from_blocks({"narration_blocks": blocks}) or str(data.get("narration") or "")


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
    while len(blocks) > 1 and bool(blocks[-1].get("pause")):
        previous_text = str(blocks[-2].get("narration") or "")
        previous_visual = " ".join([
            str(blocks[-2].get("visual") or ""),
            " ".join(str(fact) for fact in (blocks[-2].get("visual_facts") or []) if isinstance(blocks[-2].get("visual_facts"), list)),
        ])
        if _narration_claims_final_completion(previous_text) or _visual_supports_final_completion(previous_visual):
            blocks.pop()
        else:
            break
    _commit_narration_blocks_to_script(data, blocks)


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


def _block_narration_sync_instruction(language: str) -> str:
    min_scene_chars = _minimum_scene_matched_narration_chars(language)
    return (
        "There is no filler word-count target, and you must not pad narration with meaningless words. Backend rendering preserves each selected source range and the requested video_speed; it will not rescue a sparse narration block by cutting or speeding the visuals after the spoken line. "
        f"Your job is to choose useful timestamp ranges and write concise, concrete, scene-matched commentary. For ordinary narrated process/action blocks, write enough to clearly explain the visible action, usually at least {min_scene_chars} non-whitespace characters, but never add filler or repeat obvious words. If a range truly has little to say, shorten the range, split it, or mark a brief pause instead of leaving a long under-explained narrated block."
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
    if len(re.sub(r"\s+", "", block_narration)) > len(re.sub(r"\s+", "", narration)):
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


def _sentence_repeat_key(sentence: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "", str(sentence or "")).lower()


def _scene_fact_sentence(block: Dict, language: str) -> str:
    visual = str(block.get("visual") or "").strip()
    visual_facts = block.get("visual_facts") if isinstance(block.get("visual_facts"), list) else []
    fact_parts = [str(fact).strip() for fact in visual_facts if str(fact).strip()]
    if (language or "").lower().startswith("zh"):
        zh_fact_parts = [part for part in fact_parts if not re.search(r"[A-Za-z]{3,}", part)]
        if zh_fact_parts:
            parts = zh_fact_parts
        elif visual and not re.search(r"[A-Za-z]{3,}", visual):
            parts = [visual]
        else:
            parts = fact_parts or ([visual] if visual else [])
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
    replacements = (
        (r"镜头(?:切到|转到|来到|拉近|推进|对准|展示|给到|带到)?", ""),
        (r"(?:画面里|画面中|视频里|视频中|当前画面|当前可见|画面显示|画面展示|视频显示|视频展示)", ""),
        (r"(?:可以看到|能看到)", ""),
    )
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)
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


def _sanitize_generated_commentary_script(data: Dict) -> None:
    _strip_camera_meta_phrasing(data)
    if isinstance(data, dict):
        data["_generated_sanitized"] = True


def _openai_visual_analysis_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, OPENAI_VISUAL_ANALYSIS_CACHE)


def _load_cached_openai_visual_analysis(output_dir: Optional[str]) -> Optional[Dict]:
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
    return data if isinstance(data, dict) else None


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
    observations = []
    for item in visual_analysis.get("observations") or []:
        if not isinstance(item, dict):
            continue
        timestamp = item.get("timestamp")
        if isinstance(timestamp, (int, float)) and start <= float(timestamp) <= end:
            observations.append(item)
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


def _narration_claims_completion(text: str) -> bool:
    return _narration_claims_final_completion(text) or _narration_claims_packing(text)


def _narration_claims_final_completion(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in COMMENTARY_FINAL_COMPLETION_NARRATION_PATTERNS)


def _narration_claims_packing(text: str) -> bool:
    return any(pattern.search(text or "") for pattern in COMMENTARY_PACKING_NARRATION_PATTERNS)


def _rewrite_unsupported_completion_claim_text(text: str) -> str:
    rewritten = str(text or "")
    replacements = (
        (r"装好收工", "继续处理"),
        (r"做好收工", "继续处理"),
        (r"完工收尾", "继续处理"),
        (r"完工", "继续推进"),
        (r"收工", "继续处理"),
        (r"装好", "继续整理"),
        (r"做好", "继续处理"),
        (r"做完", "继续处理"),
        (r"搞定", "继续处理"),
        (r"完成", "继续推进"),
        (r"结束", "继续推进"),
        (r"打包", "整理"),
        (r"包装", "整理"),
        (r"封箱", "整理"),
        (r"装袋", "整理进袋子旁"),
        (r"装箱", "整理到箱子旁"),
        (r"(?:装进|装入|放进|放入|塞进)[^。！？!?]{0,8}(?:袋子|袋|箱子|箱|盒子|盒|桶|容器)(?:里|中|内)?", "继续处理"),
        (r"袋口[^。！？!?]{0,8}(?:拧紧|扎紧|收紧|绑紧|封住|封好)", "袋子还在旁边调整"),
        (r"(?:被|已)?(?:装进|装入|放进|放入|塞进)", "继续处理"),
        (r"\b(?:finished|done|completed|complete|wrapped up)\b", "keeps going"),
        (r"\b(?:put|placed|loaded)\s+into\s+(?:a\s+)?(?:bag|box|container)\b", "keeps moving near the container"),
        (r"\b(?:packed|boxed|wrapped|packing|boxing|wrapping|bagged|bagging)\b", "organized"),
    )
    for pattern, replacement in replacements:
        rewritten = re.sub(pattern, replacement, rewritten, flags=re.I)
    rewritten = re.sub(r"\s+", " ", rewritten).strip() if re.search(r"[A-Za-z]", rewritten) else rewritten.strip()
    return rewritten


def _rewrite_unsupported_completion_claims(data: Dict, visual_analysis: Optional[Dict] = None) -> None:
    changed = False
    for block in data.get("narration_blocks") or []:
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        model_visual_text = _block_visual_grounding_text(block)
        actual_visual_text = _visual_analysis_text_for_block(block, visual_analysis)
        evidence_text = actual_visual_text or model_visual_text
        narration_text = str(block.get("narration") or block.get("text") or "")
        if _narration_claims_final_completion(narration_text) and not _visual_supports_final_completion(evidence_text):
            block["narration"] = _rewrite_unsupported_completion_claim_text(narration_text)
            changed = True
            narration_text = str(block.get("narration") or "")
        if _narration_claims_packing(narration_text) and not _visual_supports_packing(evidence_text):
            block["narration"] = _rewrite_unsupported_completion_claim_text(narration_text)
            changed = True
    if changed:
        data["narration"] = _narration_from_blocks(data) or str(data.get("narration") or "")


def _visual_supports_completion(text: str) -> bool:
    return _visual_supports_final_completion(text) or _visual_supports_packing(text)


def _visual_supports_final_completion(text: str) -> bool:
    return _text_matches_any_keyword(text, COMMENTARY_FINAL_COMPLETION_VISUAL_KEYWORDS)


def _visual_supports_packing(text: str) -> bool:
    visual_text = str(text or "")
    if _text_matches_any_keyword(visual_text, COMMENTARY_PACKING_VISUAL_KEYWORDS):
        return True
    has_container = _text_matches_any_keyword(visual_text, COMMENTARY_PACKING_CONTAINER_KEYWORDS)
    has_action = _text_matches_any_keyword(visual_text, COMMENTARY_PACKING_ACTION_KEYWORDS)
    return has_container and has_action


def _validate_no_post_completion_visual_regression(blocks: List[Dict], visual_analysis: Optional[Dict]) -> None:
    if not visual_analysis:
        return
    saw_secured_or_final = False
    for index, block in enumerate(blocks or [], start=1):
        evidence = " ".join([
            str(block.get("visual") or ""),
            " ".join(str(fact) for fact in (block.get("visual_facts") or []) if isinstance(block.get("visual_facts"), list)),
            _visual_analysis_text_for_block(block, visual_analysis),
        ]).lower()
        if any(token in evidence for token in ("secured", "final", "packed", "completion", "绑好", "固定住", "收尾")):
            saw_secured_or_final = True
            continue
        if saw_secured_or_final and any(token in evidence for token in ("climb", "climber", "ascent", "gripping trunk", "boots", "爬")):
            raise Exception(
                "AI narration_blocks regress to an earlier source action after the harvest/secured result. "
                f"Block {index} returns to climbing/ascent footage after a secured or final-result moment; keep chronology aligned with the actual visual timeline."
            )


def _validate_completion_claim_matches_visual(
    block: Dict,
    visual_text: str,
    actual_visual_text: str,
    model_visual_text: str,
    narration_text: str,
    index: int,
) -> None:
    evidence_text = actual_visual_text or model_visual_text or visual_text
    if _narration_claims_final_completion(narration_text):
        if _visual_supports_final_completion(evidence_text) or _visual_supports_packing(evidence_text):
            return
        evidence_source = "actual OpenAI visual timeline" if actual_visual_text else "block visual description"
        raise Exception(
            "AI narration block claims a completed packing/ending action that is not supported by its selected visual range. "
            f"Block {index} says the work is finished, completed, or 收工, but the {evidence_source} does not show a final result, completed state, closure, test, installation, secured state, or clear ending moment. "
            "Move the ending line to a timestamp where that completion is visible, or rewrite this block to describe only the visible action in its range."
        )
    if not _narration_claims_packing(narration_text):
        return
    if _visual_supports_packing(evidence_text):
        return
    evidence_source = "actual OpenAI visual timeline" if actual_visual_text else "block visual description"
    raise Exception(
        "AI narration block claims a completed packing/ending action that is not supported by its selected visual range. "
        f"Block {index} says the work is packed, boxed, wrapped, or put into a container, but the {evidence_source} does not show visible packaging or container-loading action. "
        "Move that line to a timestamp where the packaging is visible, or rewrite this block to describe only the visible action in its range."
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
        if (language or "").lower().startswith("zh") and visual_text and not re.search(r"[A-Za-z]{3,}", visual_text):
            source_text = visual_text
        else:
            source_text = fact_text or visual_text
        if current_chars < expected_chars and source_text:
            additions = []
            while len(re.sub(r"\s+", "", narration + "".join(additions))) < expected_chars:
                additions.append(f"{source_text}。")
            item["narration"] = narration + "".join(additions)
        elif shortfall:
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
    grounding_text = _block_visual_grounding_text(block)
    if (
        block_duration >= 60.0
        and narration_chars >= _minimum_scene_matched_narration_chars(language) * FULL_MODE_NARRATION_DENSITY_MIN_RATIO
        and _visual_supports_final_completion(grounding_text)
    ):
        return None
    if bool(block.get("_locked_edit_plan")) and locked_min_chars > 0 and narration_chars < locked_min_chars:
        locked_density_floor = int(math.ceil(locked_min_chars * FULL_MODE_NARRATION_DENSITY_MIN_RATIO))
        short_locked_block_voice_covers_range = (
            block_duration < 12.0
            and estimated_voice_seconds + FULL_MODE_VALIDATION_EPSILON_SECONDS >= minimum_voice_seconds
            and block_duration - estimated_voice_seconds <= FULL_MODE_MAX_NARRATED_BLOCK_SILENCE_SECONDS + FULL_MODE_VALIDATION_EPSILON_SECONDS
        )
        if (
            (
                narration_chars >= locked_density_floor
                or short_locked_block_voice_covers_range
            )
            and estimated_voice_seconds + FULL_MODE_VALIDATION_EPSILON_SECONDS >= minimum_voice_seconds
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
    density_floor = max(
        _density_floor_chars_for_visual_duration(block_duration, language),
        locked_min_chars if locked_min_chars > 0 else 0,
    )
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
        compact = re.sub(r"\s+", "", clean)
        if len(compact) <= max_chars:
            return clean
        return compact[:max_chars].rstrip("，,。！？!?") + "。"
    words = re.findall(r"\S+", clean)
    max_words = max(1, int(max_voice_seconds * 2.6))
    if len(words) <= max_words:
        return clean
    return " ".join(words[:max_words]).rstrip(",.!?") + "."


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
    speed_candidates = _visual_analysis_speed_candidate_segments(visual_analysis, duration)
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


def _validate_scene_matched_narration_blocks(data: Dict, visual_analysis: Optional[Dict] = None) -> None:
    for index, block in enumerate(data.get("narration_blocks") or [], start=1):
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        model_visual_text = _block_visual_grounding_text(block)
        actual_visual_text = _visual_analysis_text_for_block(block, visual_analysis)
        visual_text = actual_visual_text or model_visual_text
        narration_text = str(block.get("narration") or block.get("text") or "")
        _validate_completion_claim_matches_visual(
            block,
            visual_text,
            actual_visual_text,
            model_visual_text,
            narration_text,
            index,
        )


def _has_visual_plan(data: Dict) -> bool:
    return bool(data.get("narration_blocks") or data.get("chapters"))


def _validate_commentary_script_for_target(
    data: Dict,
    duration: float,
    target_duration: str,
    language: str,
    visual_analysis: Optional[Dict] = None,
) -> None:
    generated_sanitized = bool(data.get("_generated_sanitized"))
    if target_duration != "full":
        _validate_no_banned_narration_patterns(data)
    elif not generated_sanitized:
        _validate_no_editorial_meta_narration_patterns(data)
    _strip_camera_meta_phrasing(data)
    _validate_no_banned_commentary_phrases(data)
    _rewrite_unsupported_completion_claims(data, visual_analysis=visual_analysis)
    _validate_scene_matched_narration_blocks(data, visual_analysis=visual_analysis)
    _validate_no_post_completion_visual_regression(data.get("narration_blocks") or [], visual_analysis)
    if target_duration != "full":
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
    _validate_ai_video_speed_decisions(blocks, language, visual_analysis=visual_analysis, duration=duration)
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
    required_visual_seconds = target_seconds if _full_mode_preserves_source_process(duration, target_seconds) else min_visual_seconds
    if target_seconds > 0 and len(blocks) > 1 and visual_seconds < required_visual_seconds - visual_budget_tolerance:
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
    if target_seconds > 0 and len(blocks) > 1 and visual_seconds < required_visual_seconds - visual_budget_tolerance:
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
            if target_seconds > 0 and len(blocks) > 1 and visual_seconds < required_visual_seconds - visual_budget_tolerance:
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
    _commit_narration_blocks_to_script(data, blocks)
    edit_segments = _narration_blocks_to_edit_segments(blocks)
    visual_seconds = sum(_block_visual_duration(block) for block in blocks)
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
            "Use pause=true only for intentional visual breathing room, and keep ordinary process footage in narrated blocks."
        )
    if max_chars and len(narration) > max_chars:
        raise Exception(
            "AI narration is too long for comprehensive full-mode commentary. "
            f"Got {len(narration)} chars; expected at most {max_chars}. "
            "The generated voiceover would run much longer than the selected visuals and can overload local rendering, so OpenShorts rejected it."
        )
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
) -> None:
    if target_duration != "full" or not _is_rendered_cached_full_mode_script(data):
        _validate_commentary_script_for_target(data, duration, target_duration, language)
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


def _repair_scope_instruction(validation_error: Optional[Exception], attempt_label: str) -> str:
    error_text = str(validation_error or "")
    if _validation_error_is_visual_budget(validation_error):
        return (
            f"This is {attempt_label}. The validation error is a global visual-budget/timeline failure, "
            "so do not merely patch one block and do not preserve a block just because it was locally valid. "
            "Repartition the complete narration_blocks list as needed so the final playable total lands inside the target window and every kept range remains scene-matched and TTS-synced."
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
    if "claims a completed packing/ending action" in error_text:
        return f"""
FOCUSED REPAIR REQUIRED:
- The previous JSON put a completion/收工/packed/finished line on a block whose selected visual range does not show the claimed completion or packaging action.
- Move that ending line to a timestamp where the final result, completed state, closure, test, installation, secured state, clear ending, packaging, or container-loading action is actually visible; otherwise rewrite the failed block to describe only the visible action in that range.
- Do not leave unrelated aftermath footage after a spoken "收工" line unless that later footage has its own accurately matched narration or is removed from the selected timeline.
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
    unsupported_completion_or_packing_error = bool(re.search(
        r"completed packing/ending action|"
        r"finished, completed, or\s*收工|"
        r"packed, boxed, wrapped|"
        r"does not show a final result|"
        r"does not show visible packaging|"
        r"completion_allowed is false",
        error_text,
        flags=re.IGNORECASE,
    ))
    if unsupported_completion_or_packing_error:
        return (
            "\n\nRetry correction note:\n"
            "The previous run failed because one narration block claimed a completed packing/ending action that the selected visual range did not support. "
            "In the next script, use completion words such as finished, done, completed, 收工, 装好, 打包, or 完成 only when that exact final result, completed state, closure, test, installation, secured state, or packaging action is visible inside the same block. "
            "For any block without that evidence, describe only what is visible in that range and move any ending line to a timestamp where the ending is actually visible."
        )
    compact_error = re.sub(r"\s+", " ", error_text).strip()
    compact_error = _limit_text_chars(compact_error, 600)
    return (
        "\n\nRetry correction note:\n"
        f"The previous response failed validation with this error: {compact_error}\n"
        "Return narration_blocks that cover about the requested target duration, not the entire raw source timeline. "
        "Keep commentary concise and scene-matched; do not add filler narration just to make the script longer."
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
- Write selective scene-matched narration for the edited visuals, leaving moments for the picture and original sound to breathe. Do not add words just to make the script longer.
{_banned_phrase_instruction()}
- Narration must be at most {max_chars} non-whitespace characters; shorter concise commentary is valid when it matches the visuals.
- Return exactly {block_count} narration_blocks with start, end, visual, narration, pause, rate, pitch, video_speed, and speed_reason.
- If episode_plan.should_split=true, keep episodes aligned to the repaired 1-based narration_blocks indexes using start_block and end_block.
- Aim for about {target_block_seconds:.0f}s playable visuals per block across the whole edit, but keep ordinary narrated blocks usually 8-16s after video_speed so the actual TTS can cover the selected visuals. Split longer useful ranges into multiple narrated blocks or explicit brief pause=true blocks; do not leave a 40-60s narrated block with a short paragraph.
- Use concise, concrete narration for normal narrated blocks. pause=true blocks must leave narration empty.
- Completion words such as "finished", "packed", "done", "收工", "装好", "打包", or "完成" must only appear in a block where that exact completion, packaging, closure, final result, completed state, test, installation, secured state, or clear ending is visible.
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
    return f"""You are writing the final voiceover for a commentary remix.

VIDEO-DERIVED VISUAL PLAN:
{json.dumps(visual_plan, ensure_ascii=False)}

VALIDATION ERROR:
{validation_error or "The previous script needs full-mode validation before rendering."}

{focused_repair_instruction}

FINALIZE COMPLETE COMMENTARY:
- {repair_scope_instruction}
- Use the video-derived visual plan above as the source of visual truth.
- Do not invent unrelated scenes. Every paragraph must follow the timestamps, visual descriptions, chapters, or edit_segments in the visual plan.
{timeline_rules}
- Write a complete but breathable Simplified Chinese voiceover for the edited visuals. Keep commentary selective and do not talk over every second.
{_banned_phrase_instruction()}
- The top-level title must clearly say what the video is doing: name the concrete subject, process/action, and result or purpose. Use titles like "废旧电机拆解回收铜线全过程" instead of vague hype titles like "震撼工厂全过程" or "不可思议的改造".
- The final narration must be at most {max_chars} non-whitespace characters; shorter concise commentary is valid when it matches the selected visuals.
- Return exactly {block_count} narration_blocks with start, end, visual, narration, pause, rate, pitch, video_speed, and speed_reason.
- If episode_plan.should_split=true, keep episodes aligned to the repaired 1-based narration_blocks indexes using start_block and end_block.
- Aim for about {target_block_seconds:.0f}s playable visuals per block across the whole edit, but keep ordinary narrated blocks usually 8-16s after video_speed so the actual TTS can cover the selected visuals. Split longer useful ranges into multiple narrated blocks or explicit brief pause=true blocks; do not leave a 40-60s narrated block with a short paragraph.
- Use concise, concrete narration for normal narrated blocks. pause=true blocks must leave narration empty.
- Completion words such as "finished", "packed", "done", "收工", "装好", "打包", or "完成" must only appear in a block where that exact completion, packaging, closure, final result, completed state, test, installation, secured state, or clear ending is visible.
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
            "observations": len(observations),
            "candidate_segments": len(candidate_segments),
        },
        "observations": _sample_timeline_items(observations, max_observations),
        "candidate_segments": _sample_timeline_items(candidate_segments, max_candidate_segments),
    }
    return {key: value for key, value in compact.items() if value not in (None, [], {})}


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
        "If completion_allowed is false, do not use final completion words such as finished/done/收工/装好/完成 in that block; follow completion_note. "
        "Packaging or container-loading may be described only when the block's visual facts show packaging, a container, or loading action."
    )


def _apply_openai_candidate_edit_plan(data: Dict, edit_plan: Optional[Dict]) -> None:
    if not edit_plan:
        return
    plan_blocks = [block for block in (edit_plan.get("blocks") or []) if isinstance(block, dict)]
    if not plan_blocks:
        return
    model_blocks = _script_narration_blocks(data)
    rewritten = []
    for index, plan_block in enumerate(plan_blocks):
        model_block = model_blocks[index] if index < len(model_blocks) else {}
        narration = str(model_block.get("narration") or model_block.get("text") or "").strip()
        pause = bool(model_block.get("pause")) and not narration
        visual = str(model_block.get("visual") or "").strip() or str(plan_block.get("visual") or "").strip()
        visual_facts = model_block.get("visual_facts") if isinstance(model_block.get("visual_facts"), list) else None
        if not visual_facts:
            visual_facts = plan_block.get("visual_facts") if isinstance(plan_block.get("visual_facts"), list) else []
        evidence_timestamps = (
            model_block.get("evidence_timestamps")
            if isinstance(model_block.get("evidence_timestamps"), list)
            else plan_block.get("evidence_timestamps")
        )
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
            "_completion_allowed": bool(plan_block.get("completion_allowed")),
            "_completion_note": str(plan_block.get("completion_note") or "").strip(),
        })
    data["narration_blocks"] = rewritten
    data["edit_segments"] = _narration_blocks_to_edit_segments(rewritten)
    data["narration"] = _narration_from_blocks({"narration_blocks": rewritten}) or str(data.get("narration") or "")


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
) -> str:
    mode = _normalize_analysis_mode(analysis_mode)
    sampled_segments = _sample_transcript_segments(transcript)
    transcript_text = transcript.get("text", "")
    if len(transcript_text) > 60000:
        transcript_text = transcript_text[:60000]
    target_seconds = _target_visual_duration_seconds_for_analysis(duration, target_duration, visual_analysis)
    max_chars = _maximum_narration_chars_for_target_seconds(target_seconds, target_duration, language)
    block_count = _target_narration_block_count_for_target_seconds(target_seconds)
    target_block_seconds = target_seconds / max(1, block_count)
    block_sync_instruction = _block_narration_sync_instruction(language)
    timeline_rules = _full_mode_timeline_rules(duration, target_seconds)
    preserves_full_process = _full_mode_preserves_source_process(duration, target_seconds)
    cut_selection_instruction = (
        "- Preserve the source workflow in chronological order for this full-process edit. Remove only clearly useless dead time, duplicated waiting, setup, walking, camera drift, intro/outro, irrelevant, or failed footage; prefer video_speed for slow/repetitive but meaningful process ranges."
        if target_duration == "full" and preserves_full_process
        else "- Select which original video ranges should be kept for the final edit and which ranges should be removed. Remove repetitive, slow, duplicated, waiting, setup, walking, camera drift, intro/outro, irrelevant, or low-value filler parts; use AI-chosen video_speed for slow-but-useful ranges that should remain understandable instead of being deleted."
    )
    chronological_instruction = (
        "- The kept visual ranges must stay in the same chronological order as the source video and may preserve the full useful workflow when the source itself is shorter than the target."
        if target_duration == "full" and preserves_full_process
        else f"- The kept visual ranges must stay in the same chronological order as the source video, cover the complete process arc, and should total about {int(target_seconds)} playable seconds after video_speed rather than one continuous full-source range."
    )
    openai_one_shot_sync_instruction = ""
    if mode == "openai" and target_duration == "full":
        openai_one_shot_sync_instruction = (
            "- For OpenAI-compatible mode, avoid spending output tokens on audit tables; return the production script and timeline only. "
            "The backend will handle block-level render sync deterministically."
        )
    style_grounding = _style_grounding_instruction(style, language)
    custom_style_instruction = _custom_style_instruction(custom_style_prompt)

    visual_analysis_text = ""
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
            "Use it as the visual evidence for edit_segments and narration_blocks.\n"
            "- The analysis was generated from scene-aware keyframes across the complete source video, "
            "with 1-3 frames per detected scene depending on visual change. Do not treat it as only a few isolated highlights.\n"
            "- Align narration and edit_segments with both the Faster-Whisper transcript timestamps and the scene-aware visual timeline.\n"
            "- When candidate_segments include suggested_speed or speed_reason, use them as visual evidence for narration_blocks.video_speed, but still make the final speed decision from the exact selected range.\n"
            "- All edit_segments and narration_blocks must use timestamps from the original full source video timeline and must be selected from across the complete beginning, middle, and ending timeline."
        )
        if visual_analysis:
            visual_analysis_text = _openai_visual_analysis_prompt_text(visual_analysis)
    else:
        visual_instruction = (
            "- Attached images, if present, are sampled keyframes. Treat them as lightweight visual context, "
            "not as the full source video."
        )
    if openai_candidate_edit_plan and target_duration == "full":
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
    locked_edit_plan_text = _openai_candidate_edit_plan_prompt_text(openai_candidate_edit_plan)
    locked_edit_plan_section = f"""
BACKEND-CALCULATED EDIT PLAN FROM AI VISUAL CANDIDATES:
{locked_edit_plan_text}
""" if locked_edit_plan_text else ""

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

VISUAL CONTEXT:
{visual_instruction}
{openai_visual_section}
{locked_edit_plan_section}
RULES:
{_banned_phrase_instruction()}
- Do not merely translate the transcript.
- Rewrite it as an original, natural commentary narration.
- Preserve the important facts, sequence, and context from the source.
{cut_selection_instruction}
{chronological_instruction}
- The narration must match the selected visual ranges, not the removed parts.
- Treat narration_blocks as the production timeline: each block's start/end is the source-video range that will play while that exact block's narration is spoken.
- Do not describe a visual before it appears or after it has already passed; if a sentence mentions a machine action, material state, worker movement, comparison, or joke, it must belong to that same block's visible time range.
- Completion words such as "finished", "packed", "done", "收工", "装好", "打包", or "完成" must only appear in a block where that exact completion, packaging, closure, final result, completed state, test, installation, secured state, or clear ending is visible. Do not say "收工" on a later loose close-up or aftermath shot that does not show the described completion.
- Keep each block self-contained: first ground the viewer in the concrete visible action, then add interpretation or commentary for that exact action.
- {style_grounding}{custom_style_instruction}
{timeline_rules}
- For TARGET DURATION full, if the source has a final payoff, result reveal, before/after comparison, effect showcase, completed product, or conclusion, include the visual range where that result actually appears and let it play through.
- For TARGET DURATION full, the selected blocks must not stop in the first half of a long source; at least one narration_blocks item must end after {int(duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION)} seconds.
- For TARGET DURATION full, narration_blocks is required: output exactly {block_count} chronological blocks. Each block must have start, end, visual, narration, pause, rate, pitch, video_speed, and speed_reason.
- For TARGET DURATION full in OpenAI-compatible mode, when BACKEND-CALCULATED EDIT PLAN is provided, use exactly those plan blocks and exactly their start, end, video_speed, speed_reason, visual_facts, and evidence_timestamps. Do not add, remove, merge, split, or retime plan blocks. Your job is to write narration that matches each locked visual range. {_locked_plan_block_guidance_text()}
- For TARGET DURATION full, every non-pause block should also include visual_facts and evidence_timestamps when the model can infer them from the visual timeline; use these fields to prove the narration is grounded in that exact source range.
- For TARGET DURATION full, aim for about {target_block_seconds:.0f}s playable visuals per block across the whole edit, but keep ordinary narrated blocks usually 8-16s after video_speed so the actual TTS can cover the selected visuals. Split longer useful ranges into multiple narrated blocks or explicit brief pause=true blocks; do not leave a 40-60s narrated block with a short paragraph.
- For TARGET DURATION full, narration_blocks must cover about {int(target_seconds)} seconds of selected visuals across the complete source timeline and must cover the same ranges as edit_segments; do not create narration for ranges that are not kept.
- For TARGET DURATION full, most selected visual blocks should contain narration, but do not narrate every second like a robot; intentionally leave short breathing room when the footage, process sound, reveal, or visual proof benefits from it.
- For TARGET DURATION full, use pause=true blocks when the original footage genuinely needs to be heard without commentary: key reveals, machine/process sounds, skilled hand work, visual proof, emotional beats, transitions, or moments where the picture explains itself. Pause blocks must leave narration empty and should usually last 2-12 seconds.
- For TARGET DURATION full, pause blocks should use the original source audio as the main sound, but total pause time must stay under about 25% of selected visual time. Avoid more than two pause blocks back-to-back.
- For TARGET DURATION full, each non-pause block's narration must be speakable inside that block's visual duration; do not put 2 minutes of words into a 20-second visual range.
- For TARGET DURATION full, keep normal narrated blocks concise and concrete for their visible action. This means not too dense, but also not empty: each ordinary narrated process/action block should clearly state what is visible, what changes, and why that moment matters.
- For TARGET DURATION full, do not make narration sparse by writing one vague sentence over a long visual block. If a non-pause block plays 12+ seconds, write enough scene-matched commentary to make the visible action clear; if there is not enough to say, shorten that timestamp range or use a brief pause=true block.
- For TARGET DURATION full, each non-pause block's narration must match only that block's visible range. Prefer 1-3 compact sentences that name concrete objects, actions, state changes, comparisons, results, or risks visible in that range.
- For TARGET DURATION full, {block_sync_instruction}
{openai_one_shot_sync_instruction}
- For TARGET DURATION full, if a selected visual range is too long or too visually sparse for a high-quality natural commentary paragraph, redesign the block: shorten the range, split it, or use a brief pause=true moment for original audio/visual breathing room. The renderer preserves selected source ranges and will not tighten trailing footage after short narration. Do not add meaningless word padding, and do not pad it with meaningless words.
- For TARGET DURATION full, use rate to create cadence: slower values like "-10%" for important reveals or emotional emphasis, faster values like "+12%" for energetic process sections. Valid range: "-30%" to "+30%".
- For TARGET DURATION full, use pitch lightly for tone: lower values like "-3Hz" for weight, higher values like "+3Hz" for excitement. Valid range: "-15Hz" to "+15Hz".
- For TARGET DURATION full, decide video_speed from the actual visible action, not from a fixed rule. Use 1.0 for key reveals, removal moments, packaging/closure, readable text, final results, completed states, tests, installations, and payoff shots. Use moderate speeds such as 1.15-1.5 for slow but still useful process footage; use 1.75-2.5 only when the range is clearly repetitive, waiting, walking, setup, repeated tool operation, transport, or transition footage and remains understandable after acceleration. If the provided visual candidate evidence marks a kept range as slow/repetitive/waiting/transition or includes suggested_speed > 1.0, either set video_speed above 1.0 with a concrete speed_reason or shorten/cut that range; do not leave long slow filler at 1.0 without a visual reason. Every block with video_speed > 1.0 must include a concrete speed_reason tied to that exact visual range; blocks at 1.0 can use speed_reason "".
- For TARGET DURATION full, vary rate and pitch across blocks; do not leave every non-pause block at "+0%" and "+0Hz".
- For TARGET DURATION full, total narration must be at most {max_chars} non-whitespace characters so the voiceover does not exceed the selected visuals. There is no total minimum word count; do not add filler to make the script longer.
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


def _load_openai_analysis_frames(output_dir: str, sampling_options: Optional[Dict] = None) -> List[Dict]:
    manifest_path = _openai_frames_manifest_path(output_dir)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        expected_options = resolve_openai_sampling_options(**(sampling_options or {}))
        if data.get("sampling_options") == expected_options:
            frames = data.get("frames") or []
            if isinstance(frames, list) and frames and all(
                isinstance(frame, dict) and frame.get("path") and os.path.exists(frame["path"])
                for frame in frames
            ):
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
    if frames:
        _save_openai_analysis_frames(output_dir, frames, sampling_options=sampling_options)
    return frames


def _save_openai_analysis_frames(output_dir: str, frames: List[Dict], sampling_options: Optional[Dict] = None) -> None:
    manifest_path = _openai_frames_manifest_path(output_dir)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({
            "sampling_options": resolve_openai_sampling_options(**(sampling_options or {})),
            "frames": frames,
        }, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, manifest_path)


def _load_openai_visual_analysis(output_dir: str, model: str, frame_infos: List[Dict], sampling_options: Optional[Dict] = None) -> Optional[Dict]:
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
) -> List[Dict]:
    frames_dir = os.path.join(output_dir, "openai_analysis_frames")
    os.makedirs(frames_dir, exist_ok=True)
    cached_frames = _load_openai_analysis_frames(output_dir, sampling_options=sampling_options)
    if cached_frames:
        if progress:
            progress(f"Reusing OpenAI-compatible analysis frames: {len(cached_frames)}")
        return cached_frames
    if duration <= 0:
        return []

    samples = []
    if OPENAI_SCENE_AWARE_SAMPLING:
        if progress:
            progress("Detecting scenes for OpenAI-compatible scene-aware frame sampling...")
        samples = _select_openai_scene_aware_frame_samples(video_path, duration, progress=progress, sampling_options=sampling_options)
    if not samples:
        if progress and OPENAI_SCENE_AWARE_SAMPLING:
            progress("Using uniform OpenAI-compatible frame sampling fallback...")
        samples = _select_openai_uniform_frame_samples(duration, sampling_options=sampling_options)

    frames = []
    failures = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        timestamp = float(sample["timestamp"])
        frame_path = os.path.join(frames_dir, f"frame_{index:04d}_{int(timestamp * 1000):09d}.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{timestamp:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale=-2:{OPENAI_FRAME_HEIGHT}",
            "-q:v", "4",
            frame_path,
        ]
        try:
            _run_command(cmd)
            if os.path.exists(frame_path):
                frame_info = {"path": frame_path, "timestamp": round(timestamp, 3)}
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
        _save_openai_analysis_frames(output_dir, frames, sampling_options=sampling_options)
    return frames


def _openai_visual_batch_prompt(
    video_title: str,
    duration: float,
    frames: List[Dict],
    batch_index: int,
    total_batches: int,
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

TASK:
Analyze the visual action in these frames. Return valid JSON only.

RULES:
- Use the timestamp labels as source-video timestamps.
- Use scene_start/scene_end metadata as scene boundaries when present.
- Treat early/middle/late sample roles as positions inside one detected scene.
- Analyze the visuals in detail: describe concrete visible actions, tools, materials, people, hand/foot movement, object state changes, scene changes, reveals, and process stages.
- Separate evidence from uncertainty. If a material or object is ambiguous, describe its appearance instead of forcing a specific label.
- Do not invent facts that are not visible.
- Mark frames/ranges that look valuable for a commentary edit, and distinguish must-keep payoff/action from slow-but-useful or low-value footage.
- For each useful candidate range, judge from the visible motion whether it should play at normal speed or can be accelerated. This is only visual evidence for later AI editing; do not use a fixed duration rule.
- Keep observations concise but specific enough to ground narration later; this detailed visual analysis is evidence for editing, not a requirement to make the spoken narration long.

JSON FORMAT:
{{
  "batch_index": {batch_index},
  "observations": [
    {{"timestamp": 12.3, "visual": "what is visible", "process_stage": "stage name", "importance": 1, "keep_candidate": true, "pace": "normal|slow|repetitive|waiting|transition", "edit_value": "must_keep|useful|skippable"}}
  ],
  "candidate_segments": [
    {{"start": 10.0, "end": 25.0, "reason": "why this visual range should be kept", "suggested_speed": 1.0, "speed_reason": "visible action needs normal speed or can remain understandable accelerated"}}
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


def _analyze_openai_visual_timeline(
    frame_infos: List[Dict],
    video_title: str,
    duration: float,
    api_key: str,
    base_url: str,
    model: str,
    progress: Optional[Callable[[str], None]] = None,
    sampling_options: Optional[Dict] = None,
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

    def analyze_batch(index: int, batch: List[Dict]) -> Dict:
        _raise_if_commentary_cancelled()
        prompt = _openai_visual_batch_prompt(video_title, duration, batch, index, len(batches))
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
        return parsed

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
    return {
        "provider": "openai_compatible",
        "model": model,
        "frame_count": len(frame_infos),
        "batch_count": len(batches),
        "sampling": "scene_aware" if scene_indexes else "uniform",
        "scene_count": len(scene_indexes),
        "sampling_options": options,
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
- Narration must be at most {max_chars} non-whitespace characters; shorter concise commentary is valid when it matches the selected visuals.
- Return exactly {block_count} narration_blocks with start, end, visual, narration, pause, rate, pitch, video_speed, and speed_reason.
- For each non-pause block, set visual to concrete on-screen evidence from that exact timestamp range. When possible also include visual_facts and evidence_timestamps; these fields are used to keep narration grounded in the selected visuals.
- If episode_plan.should_split=true, keep episodes aligned to the repaired 1-based narration_blocks indexes using start_block and end_block.
- Aim for about {target_block_seconds:.0f}s playable visuals per block across the whole edit, but keep ordinary narrated blocks usually 8-16s after video_speed so the actual TTS can cover the selected visuals. Split longer useful ranges into multiple narrated blocks or explicit brief pause=true blocks; do not leave a 40-60s narrated block with a short paragraph.
- Use concise, concrete narration for normal narrated blocks. pause=true blocks must leave narration empty.
- Completion words such as "finished", "packed", "done", "收工", "装好", "打包", or "完成" must only appear in a block where that exact completion, packaging, closure, final result, completed state, test, installation, secured state, or clear ending is visible.
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
    return f"""{original_prompt}

PREVIOUS RESPONSE WAS NOT VALID JSON:
{invalid_text[:12000]}

JSON PARSE ERROR:
{parse_error}

REPAIR TASK:
- This is JSON syntax repair attempt {attempt}.
- Return valid JSON only, using exactly the same JSON FORMAT requested above.
- Fix broken commas, quotes, escaping, brackets, and trailing prose.
- Preserve the intended commentary content, timeline ranges, narration_blocks, edit_segments, episode_plan, chapters, and hashtags wherever possible.
- Do not wrap the JSON in markdown fences.
- Do not include explanations before or after the JSON.
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
    target_duration: str = "medium",
    progress: Optional[Callable[[str], None]] = None,
    openai_sampling_options: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    checkpoint: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    visual_analysis = _load_openai_visual_analysis(
        output_dir,
        openai_model,
        frame_infos,
        sampling_options=openai_sampling_options,
    ) if output_dir else None
    if visual_analysis:
        if progress:
            progress(
                "Reusing cached OpenAI-compatible multimodal visual analysis "
                f"{visual_analysis.get('batch_count', 0)} batches."
            )
    else:
        visual_analysis = _analyze_openai_visual_timeline(
            frame_infos=frame_infos,
            video_title=video_title,
            duration=duration,
            api_key=openai_key,
            base_url=openai_base_url,
            model=openai_model,
            progress=progress,
            sampling_options=openai_sampling_options,
        )
        if output_dir:
            cache_path = _save_openai_visual_analysis(output_dir, visual_analysis)
            if checkpoint:
                checkpoint({"openai_visual_analysis_path": cache_path})
    candidate_edit_plan = _build_openai_candidate_edit_plan(
        visual_analysis,
        duration,
        target_duration,
        language,
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
            if target_duration != "full" or script_attempt >= GEMINI_SCRIPT_VALIDATION_ATTEMPTS:
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
        _apply_openai_candidate_edit_plan(data, candidate_edit_plan)
        _normalize_script_timeline(data, duration, target_duration, language)
        data["narration"] = _normalize_script_narration(data)
        try:
            _validate_commentary_script_for_target(
                data,
                duration,
                target_duration,
                language,
                visual_analysis=visual_analysis,
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
            }
            if data.get("narration_blocks"):
                data["narration_blocks"] = _strip_auto_filled_user_visible_fields(data.get("narration_blocks") or [])
                data["edit_segments"] = _narration_blocks_to_edit_segments(data["narration_blocks"])
            _strip_internal_narration_block_fields(data)
            return data
        except Exception as exc:
            validation_error = exc
            if target_duration != "full" or script_attempt >= GEMINI_SCRIPT_VALIDATION_ATTEMPTS:
                raise
            if progress:
                repair_scope = "global timeline" if _validation_error_is_visual_budget(exc) else "focused block"
                progress(
                    f"OpenAI-compatible script validation failed on correction attempt {script_attempt}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS}: "
                    f"{exc} Asking model to repair the invalid full-mode script with {repair_scope} instructions..."
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
        "503",
        "UNAVAILABLE",
        "500",
        "INTERNAL",
        "temporarily unavailable",
        "high demand",
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
    target_duration: str = "medium",
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
                "503",
                "UNAVAILABLE",
                "500",
                "INTERNAL",
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
            data = json.loads(_clean_json_text(response.text))
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
                str(getattr(response, "text", "") or ""),
                exc,
                script_attempt,
            )
            response = _generate_content_with_retry(
                client,
                resolved_model,
                _replace_prompt_in_contents(contents, repair_prompt),
                config_kwargs,
                pool_session=pool_session,
                gemini_pool=gemini_pool,
            )
            if progress:
                progress("Gemini returned JSON repair; validating timeline sync...")
            continue
        narration = _normalize_script_narration(data)
        if not narration:
            validation_error = Exception("Gemini did not return narration text")
            if target_duration != "full" or script_attempt >= GEMINI_SCRIPT_VALIDATION_ATTEMPTS:
                raise validation_error
            if progress:
                progress(
                    f"Gemini script validation failed on correction attempt {script_attempt}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS}: "
                    f"{validation_error} Asking Gemini to rewrite the full-mode script..."
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
        _sanitize_generated_commentary_script(data)
        data["narration"] = _normalize_script_narration(data)
        try:
            _validate_commentary_script_for_target(data, duration, target_duration, language)
            data.setdefault("chapters", [])
            data.setdefault("hashtags", [])
            if data.get("narration_blocks"):
                data["narration_blocks"] = _strip_auto_filled_user_visible_fields(data.get("narration_blocks") or [])
                data["edit_segments"] = _narration_blocks_to_edit_segments(data["narration_blocks"])
            _strip_internal_narration_block_fields(data)
            return data
        except Exception as exc:
            validation_error = exc
            if target_duration != "full" or script_attempt >= GEMINI_SCRIPT_VALIDATION_ATTEMPTS:
                raise
            if progress:
                progress(
                    f"Gemini script validation failed on correction attempt {script_attempt}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS}: "
                    f"{exc} Asking Gemini to rewrite the full-mode script..."
                )
            if _has_visual_plan(data):
                next_contents = [
                    _build_visual_plan_finalization_prompt(
                        data,
                        duration,
                        target_duration,
                        language,
                        attempt=script_attempt,
                        validation_error=exc,
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
                progress(f"Gemini is rewriting the full-mode commentary script (correction {script_attempt + 1}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS})...")
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
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{float(segment['start']):.3f}",
            "-t", f"{duration:.3f}",
            "-i", video_path,
            "-an",
        ]
        if vf:
            cmd.extend([
                "-vf", vf,
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
    blocks = _normalize_narration_blocks(narration_blocks, max((float(block.get("end") or 0) for block in narration_blocks or [] if isinstance(block, dict)), default=0.0))
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
        _append_weighted_subtitle_lines(lines, _split_narration_sentences(block["narration"]), cursor, block_duration, max_units=max_units)
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
    target_duration: str = "medium",
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
    openai_visual_analysis = _load_cached_openai_visual_analysis(output_dir) if analysis_mode == "openai" else None
    using_rendered_cached_script = False
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
        try:
            _normalize_script_timeline(script, duration, target_duration, language)
            using_rendered_cached_script = target_duration == "full" and _is_rendered_cached_full_mode_script(script)
            _validate_rendered_cached_full_mode_script(
                script,
                duration,
                target_duration,
                language,
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
            transcript = _load_cached_commentary_transcript(output_dir)
            if transcript:
                log("Reusing cached Faster-Whisper transcript from saved task checkpoint...")
            else:
                log("Transcribing full video with Faster-Whisper...")
                transcript = transcribe_video(video_path, language=source_language)
                transcript_path = _save_commentary_transcript_cache(output_dir, transcript)
                if checkpoint:
                    checkpoint({"transcript_path": transcript_path})
            if analysis_mode == "current":
                log("Extracting keyframes for visual context...")
                frame_paths = _extract_keyframes(video_path, output_dir, duration)
            else:
                log("Extracting dense timestamped frames for OpenAI-compatible multimodal analysis...")
                openai_frame_infos = _extract_openai_analysis_frames(
                    video_path,
                    output_dir,
                    duration,
                    progress=log,
                    sampling_options=openai_sampling_options,
                )
        else:
            log("Skipping Faster-Whisper transcription; Gemini will analyze the attached video directly...")
            transcript = {
                "text": "",
                "segments": [],
                "language": source_language or "unknown",
            }

        if analysis_mode == "openai":
            _raise_if_commentary_cancelled()
            log("Generating original commentary script with OpenAI-compatible multimodal model...")
            script = generate_openai_commentary_script(
                transcript=transcript,
                video_title=video_title,
                duration=duration,
                openai_key=openai_key or "",
                openai_base_url=openai_base_url or "",
                openai_model=openai_model or "",
                frame_infos=openai_frame_infos,
                language=language,
                style=style,
                custom_style_prompt=custom_style_prompt,
                target_duration=target_duration,
                progress=log,
                openai_sampling_options=openai_sampling_options,
                output_dir=output_dir,
                checkpoint=checkpoint,
            )
            openai_visual_analysis = _load_cached_openai_visual_analysis(output_dir)
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
                transcript = _load_cached_commentary_transcript(output_dir)
                if transcript:
                    log("Reusing cached Faster-Whisper transcript for Gemini fallback...")
                else:
                    log("Transcribing full video with Faster-Whisper for Gemini fallback...")
                    transcript = transcribe_video(video_path, language=source_language)
                    transcript_path = _save_commentary_transcript_cache(output_dir, transcript)
                    if checkpoint:
                        checkpoint({"transcript_path": transcript_path})
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

    _raise_if_commentary_cancelled()
    _normalize_script_timeline(script, duration, target_duration, language)
    if not using_rendered_cached_script:
        _sanitize_generated_commentary_script(script)
    if using_rendered_cached_script:
        _validate_rendered_cached_full_mode_script(script, duration, target_duration, language)
    else:
        _validate_commentary_script_for_target(
            script,
            duration,
            target_duration,
            language,
            visual_analysis=openai_visual_analysis,
        )
    if script.get("narration_blocks") and not using_rendered_cached_script:
        script["narration_blocks"] = _apply_auto_video_speed_to_blocks(
            script.get("narration_blocks") or [],
            auto_video_speed,
            visual_analysis=openai_visual_analysis,
        )
        script["narration_blocks"] = _protect_full_mode_visual_budget_after_speed(
            script.get("narration_blocks") or [],
            duration,
            target_duration,
        )
        script["edit_segments"] = _narration_blocks_to_edit_segments(script["narration_blocks"])
        _validate_commentary_script_for_target(
            script,
            duration,
            target_duration,
            language,
            visual_analysis=openai_visual_analysis,
        )
    if not using_rendered_cached_script:
        _finalize_full_mode_narration_blocks_for_render(script, duration, target_duration, language)
    if script.get("narration_blocks"):
        if using_rendered_cached_script:
            _validate_rendered_cached_full_mode_script(script, duration, target_duration, language)
        else:
            _validate_commentary_script_for_target(
                script,
                duration,
                target_duration,
                language,
                visual_analysis=openai_visual_analysis,
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
    narration_blocks = _normalize_narration_blocks(script.get("narration_blocks") or [], duration)
    if narration_blocks:
        if not using_rendered_cached_script:
            _finalize_full_mode_narration_blocks_for_render(script, duration, target_duration, language)
        narration_blocks = _normalize_narration_blocks(script.get("narration_blocks") or [], duration)
        script["narration_blocks"] = narration_blocks
        script["edit_segments"] = _narration_blocks_to_edit_segments(narration_blocks)
        script["narration"] = _narration_from_blocks({"narration_blocks": narration_blocks}) or str(script.get("narration") or "")
        if using_rendered_cached_script:
            _validate_rendered_cached_full_mode_script(script, duration, target_duration, language)
        else:
            _validate_commentary_script_for_target(
                script,
                duration,
                target_duration,
                language,
                visual_analysis=openai_visual_analysis,
            )
    auto_video_speed_summary = _summarize_auto_video_speed(narration_blocks, auto_video_speed)
    use_block_synced_render = bool(narration_blocks)
    if use_block_synced_render:
        _raise_if_commentary_cancelled()
        if not auto_video_speed:
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
        edit_segments = _require_ai_selected_edit_segments(script, duration, target_duration)

    work_dir = os.path.join(output_dir, f"{slug}_work")
    os.makedirs(work_dir, exist_ok=True)
    edited_video_path = os.path.join(output_dir, f"{slug}_edited_visual.mp4")
    timed_video_path = os.path.join(output_dir, f"{slug}_timed_visual.mp4")
    ambient_audio_path = os.path.join(output_dir, f"{slug}_ambient.m4a")
    background_music_bed_path = os.path.join(output_dir, f"{slug}_background_music.m4a") if resolved_background_music else None
    trim_to_voiceover = True
    preserve_source_resolution = use_block_synced_render and target_duration == "full"
    synced_block_durations = []

    if use_block_synced_render:
        _raise_if_commentary_cancelled()
        log(f"Generating {len(narration_blocks)} timestamp-synced commentary blocks with {tts_provider} TTS...")
        ambient_audio, synced_block_durations = _create_block_synced_visuals_and_audio(
            video_path=video_path,
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
            original_audio_volume=original_audio_volume,
            pause_original_audio_volume=pause_original_audio_volume,
            preserve_source_resolution=preserve_source_resolution,
            block_concurrency=resolved_block_concurrency,
            trim_short_tts_tails=target_duration != "full",
            progress=log,
        )
        script["narration_blocks"] = narration_blocks
        script["edit_segments"] = _narration_blocks_to_edit_segments(narration_blocks)
        script["narration"] = _narration_from_blocks({"narration_blocks": narration_blocks}) or str(script.get("narration") or "")
        edit_segments = script["edit_segments"]
        auto_video_speed_summary = _summarize_auto_video_speed(narration_blocks, auto_video_speed)
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump({"script": script, "transcript": transcript}, f, ensure_ascii=False, indent=2)
        edited_video_path = timed_video_path
        if not synced_block_durations:
            _validate_voiceover_duration_for_target(voiceover_path, edit_segments, duration, target_duration)
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
        _validate_voiceover_duration_for_target(voiceover_path, edit_segments, duration, target_duration)
        log(f"Creating AI-selected visual edit with {len(edit_segments)} kept segments...")
        _create_visual_edit(
            video_path,
            edit_segments,
            edited_video_path,
            resolved_aspect,
            work_dir,
            preserve_source_resolution=preserve_source_resolution,
        )
        if target_duration == "full":
            log("Skipping full-length visual retiming because no timestamped narration blocks were returned...")
            timed_video_path = edited_video_path
        else:
            log("Aligning edited visuals to the voiceover duration...")
            _fit_video_to_voiceover(edited_video_path, voiceover_path, timed_video_path)
        log("Preparing low-volume original audio bed as ambient sound...")
        ambient_audio = _create_ambient_audio_bed(video_path, edit_segments, ambient_audio_path, original_audio_volume, work_dir)
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
    log("Mixing new voiceover with ambient source audio and background music..." if background_music_bed else "Mixing new voiceover with ambient source audio...")
    _mix_voiceover_with_video(
        video_path=timed_video_path,
        voiceover_path=voiceover_path,
        output_path=mixed_path,
        original_audio_volume=original_audio_volume,
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
            duration,
            narration_blocks=narration_blocks,
        )

    episode_plan = script.get("episode_plan") if isinstance(script.get("episode_plan"), dict) else {"should_split": False, "reason": ""}
    commentary_episodes = script.get("episodes") if isinstance(script.get("episodes"), list) else []
    rendered_episodes = []
    if target_duration == "full" and narration_blocks and commentary_episodes and episode_plan.get("should_split"):
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
        "duration": duration,
        "output_aspect": resolved_aspect,
        "original_audio_volume": original_audio_volume,
        "pause_original_audio_volume": pause_original_audio_volume,
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
