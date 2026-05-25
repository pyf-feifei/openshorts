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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import httpx
from google.genai import types

from gemini_client import create_gemini_client, normalize_gemini_base_url
from gemini_pool import GeminiKeyPool
from main import detect_scenes, download_youtube_video, transcribe_video
from resource_limits import resolve_thread_count
from saasshorts import generate_voiceover


DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
SUPPORTED_ANALYSIS_MODES = {"current", "video", "openai"}
DEFAULT_ANALYSIS_MODE = "video"
OPENAI_FRAME_INTERVAL_SECONDS = max(1.0, float(os.environ.get("OPENSHORTS_OPENAI_FRAME_INTERVAL_SECONDS", "3")))
OPENAI_MAX_FRAMES = max(1, int(os.environ.get("OPENSHORTS_OPENAI_MAX_FRAMES", "1800")))
OPENAI_BATCH_SIZE_LIMIT = 128
OPENAI_VISUAL_CONCURRENCY_LIMIT = 8
COMMENTARY_BLOCK_CONCURRENCY_LIMIT = 8
OPENAI_BATCH_SIZE = max(1, min(OPENAI_BATCH_SIZE_LIMIT, int(os.environ.get("OPENSHORTS_OPENAI_BATCH_SIZE", "46"))))
OPENAI_VISUAL_CONCURRENCY = max(1, min(OPENAI_VISUAL_CONCURRENCY_LIMIT, int(os.environ.get("OPENSHORTS_OPENAI_VISUAL_CONCURRENCY", "5"))))
COMMENTARY_BLOCK_CONCURRENCY = max(1, min(COMMENTARY_BLOCK_CONCURRENCY_LIMIT, int(os.environ.get("OPENSHORTS_COMMENTARY_BLOCK_CONCURRENCY", "5"))))
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
FULL_MODE_MIN_VISUAL_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_MIN_VISUAL_SECONDS", "600"))
FULL_MODE_MAX_VISUAL_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_VISUAL_SECONDS", "1800"))
FULL_MODE_SOURCE_FRACTION = float(os.environ.get("OPENSHORTS_FULL_MODE_SOURCE_FRACTION", "0.30"))
FULL_MODE_MAX_SOURCE_RETENTION_FRACTION = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_SOURCE_RETENTION_FRACTION", "0.45"))
FULL_MODE_MAX_NARRATION_CHARS_ZH = int(os.environ.get("OPENSHORTS_FULL_MODE_MAX_NARRATION_CHARS_ZH", "22000"))
FULL_MODE_MAX_NARRATION_CHARS_OTHER = int(os.environ.get("OPENSHORTS_FULL_MODE_MAX_NARRATION_CHARS_OTHER", "32000"))
FULL_MODE_MIN_NARRATION_ACCEPTANCE_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_MIN_NARRATION_ACCEPTANCE_RATIO", "0.88"))
FULL_MODE_MAX_VOICEOVER_DURATION_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_VOICEOVER_DURATION_RATIO", "1.15"))
FULL_MODE_MAX_PAUSE_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_PAUSE_RATIO", "0.18"))
FULL_MODE_MAX_PAUSE_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_PAUSE_SECONDS", "12"))
FULL_MODE_MAX_CONSECUTIVE_PAUSE_BLOCKS = int(os.environ.get("OPENSHORTS_FULL_MODE_MAX_CONSECUTIVE_PAUSE_BLOCKS", "1"))
FULL_MODE_MAX_NARRATION_SILENCE_TAIL_SECONDS = float(os.environ.get("OPENSHORTS_FULL_MODE_MAX_NARRATION_SILENCE_TAIL_SECONDS", "1.5"))
FULL_MODE_RENDER_SYNC_MAX_VIDEO_SPEED = float(os.environ.get("OPENSHORTS_FULL_MODE_RENDER_SYNC_MAX_VIDEO_SPEED", "4.0"))
FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION = float(os.environ.get("OPENSHORTS_FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION", "0.85"))
FULL_MODE_NEAR_MISS_REPAIR_MAX_SPEEDUP = float(os.environ.get("OPENSHORTS_FULL_MODE_NEAR_MISS_REPAIR_MAX_SPEEDUP", "1.75"))
FULL_MODE_NEAR_MISS_REPAIR_MIN_RATIO = float(os.environ.get("OPENSHORTS_FULL_MODE_NEAR_MISS_REPAIR_MIN_RATIO", "0.55"))
GEMINI_SAFE_INPUT_TOKEN_BUDGET = int(os.environ.get("OPENSHORTS_GEMINI_SAFE_INPUT_TOKEN_BUDGET", "180000"))
GEMINI_LOW_RES_TOKENS_PER_SECOND = 100.0
GEMINI_SCRIPT_VALIDATION_ATTEMPTS = max(2, int(os.environ.get("OPENSHORTS_GEMINI_SCRIPT_VALIDATION_ATTEMPTS", "6")))
COMMENTARY_BANNED_PHRASES = ("画面汇总",)
COMMENTARY_NARRATION_BANNED_PATTERNS = (
    re.compile(r"镜头", re.I),
)
ASS_SUBTITLE_MAX_LINE_UNITS = 34
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
    for candidate in (
        "/mnt/c/Apps/ffmpeg-2025-04-14-git-3b2a9410ef-full_build/ffmpeg-2025-04-14-git-3b2a9410ef-full_build/bin/ffmpeg.exe",
        "/mnt/c/Apps/OrderEXE/ffmpeg.exe",
        "/mnt/c/Apps/mediago/resources/bin/ffmpeg.exe",
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


def _run_command(cmd: List[str], cwd: Optional[str] = None) -> None:
    try:
        result = subprocess.run(_limit_ffmpeg_threads(cmd), cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise Exception(
            f"FFmpeg executable not found: {FFMPEG_BINARY}. Install ffmpeg or set OPENSHORTS_FFMPEG_BINARY to a valid binary."
        ) from exc
    if result.returncode != 0:
        raise Exception(result.stderr or result.stdout or f"Command failed: {' '.join(cmd)}")


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
    process = subprocess.Popen(
        run_cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    last_percent = -1
    if process.stdout:
        for raw_line in process.stdout:
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
    return_code = process.wait()
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


def _target_duration_hint(mode: str, source_duration: float) -> str:
    if mode == "short":
        return "Select only the most important visual moments and create a tight 60-90 second commentary edit."
    if mode == "medium":
        return "Select enough important visual moments for a 3-5 minute commentary edit, but remove repetitive or low-value parts."
    full_target = _target_visual_duration_seconds(source_duration, "full")
    return (
        "Create a comprehensive long-form commentary edit with an explicit editing strategy, not a raw full-length copy of the source. "
        f"For this source, select about {int(full_target)} seconds of useful original footage across the whole timeline, keeping the strongest process stages and removing repetitive, slow, duplicated, waiting, setup, walking, camera drift, and low-value filler time. "
        "Do not preserve the entire source unless the source itself is already shorter than the target. The narration must be detailed, scene-by-scene, and matched only to the selected edit_segments."
    )


def _style_grounding_instruction(style: str, language: str) -> str:
    normalized = (style or "").strip().lower()
    if normalized in {"funny", "roast", "吐槽", "轻松吐槽"}:
        if (language or "").lower().startswith("zh"):
            return (
                "轻松吐槽风格要求：每个 narration_blocks 段落先描述这个时间段正在发生的具体画面，再基于同一画面做轻松吐槽。"
                "可以做国际工厂/海外回收流程与中国工厂/中国回收效率的对比，但对比必须围绕当前画面里的材料、设备、人工动作、工序节奏或安全细节；"
                "不要写脱离画面的国际形势、宏大政治、地域刻板印象或没有画面证据的段子。"
            )
        return (
            "Funny/roast style: each narration block must first describe the visible action in that exact range, then make a light joke grounded in the same visible material, equipment, worker action, process rhythm, or safety detail. Avoid unrelated world affairs, politics, stereotypes, or claims not supported by the frame."
        )
    return "Keep the selected style grounded in the visible action of each timestamped range."


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


def _fallback_edit_segments(duration: float, target_duration: str) -> List[Dict]:
    if duration <= 0:
        return []
    wanted = _target_visual_duration_seconds(duration, target_duration)
    if wanted >= duration * 0.9:
        return [{"start": 0.0, "end": round(duration, 3), "reason": "full source retained"}]
    if target_duration == "full":
        parts = 16
    elif target_duration == "medium":
        parts = 6
    else:
        parts = 4
    segment_length = max(8.0, wanted / parts)
    segments = []
    for index in range(parts):
        center = duration * (index + 1) / (parts + 1)
        start = max(0.0, center - segment_length / 2)
        end = min(duration, start + segment_length)
        segments.append({"start": round(start, 3), "end": round(end, 3), "reason": "fallback evenly sampled segment"})
    return _normalize_edit_segments(segments, duration)


def _target_visual_duration_seconds(source_duration: float, target_duration: str) -> float:
    duration = max(0.0, float(source_duration or 0.0))
    if target_duration == "short":
        return min(duration, 90.0)
    if target_duration == "medium":
        return min(duration, 300.0)
    if duration <= 0:
        return 0.0
    if duration <= FULL_MODE_MIN_VISUAL_SECONDS:
        return duration
    return min(
        duration,
        max(
            FULL_MODE_MIN_VISUAL_SECONDS,
            min(duration * FULL_MODE_SOURCE_FRACTION, FULL_MODE_MAX_VISUAL_SECONDS),
        ),
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
    target_sized_limit = target_seconds * 1.10 if target_seconds > 0 else 0.0
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
    if isinstance(value, str) and value.strip().lower() in {"true", "yes", "y"}:
        return 1.5
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
        normalized.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "visual": visual or "visible scene in this source range",
            "narration": narration,
            "pause": is_pause,
            "rate": _safe_edge_rate(item.get("rate") or "+0%"),
            "pitch": _safe_edge_pitch(item.get("pitch") or "+0Hz"),
            "video_speed": _safe_video_speed(
                item.get("video_speed", item.get("playback_speed", item.get("speed", item.get("speed_up"))))
            ),
        })
    normalized.sort(key=lambda block: block["start"])
    return normalized


_AUTO_SPEED_CANDIDATE_KEYWORDS = {
    "slow", "repetitive", "waiting", "setup", "walking", "transport", "transition", "moving", "loading", "unloading",
    "conveyor", "sorting", "cleanup", "preparing", "carry", "push", "pull", "move", "repeat",
    "缓慢", "慢慢", "重复", "等待", "准备", "运输", "搬运", "转运", "移动", "推进", "拖动", "传送",
    "上料", "下料", "分拣", "清理", "过渡", "转场", "路上", "来回", "反复", "铺垫",
}
_AUTO_SPEED_PROTECTED_KEYWORDS = {
    "reveal", "final", "result", "ending", "showcase", "close-up", "text", "label", "title", "readable",
    "关键", "揭晓", "最终", "结果", "成品", "亮相", "展示", "特写", "文字", "标签", "标题", "结尾", "完成", "成果",
}


def _auto_speed_text(block: Dict) -> str:
    return f"{block.get('visual') or ''} {block.get('narration') or ''}".lower()


def _auto_speed_candidate_score(block: Dict) -> float:
    if bool(block.get("pause")):
        return 0.0
    duration = max(0.0, float(block.get("end") or 0.0) - float(block.get("start") or 0.0))
    if duration < 8.0:
        return 0.0
    text = _auto_speed_text(block)
    if any(keyword in text for keyword in _AUTO_SPEED_PROTECTED_KEYWORDS):
        return 0.0
    keyword_hits = sum(1 for keyword in _AUTO_SPEED_CANDIDATE_KEYWORDS if keyword in text)
    if keyword_hits <= 0 and duration < 18.0:
        return 0.0
    return keyword_hits * 10.0 + min(duration, 45.0)


def _apply_auto_video_speed_to_blocks(blocks: List[Dict], enabled: bool) -> List[Dict]:
    adjusted = [dict(block) for block in blocks or []]
    if not adjusted:
        return adjusted
    if not enabled:
        for block in adjusted:
            block["video_speed"] = 1.0
        return adjusted
    if any(_safe_video_speed(block.get("video_speed")) > 1.0001 for block in adjusted):
        for block in adjusted:
            block["video_speed"] = _safe_video_speed(block.get("video_speed"))
        return adjusted

    scored = [
        (score, index, block)
        for index, block in enumerate(adjusted)
        for score in [_auto_speed_candidate_score(block)]
        if score > 0
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    max_count = min(3, max(1, len([block for block in adjusted if not bool(block.get("pause"))]) // 5))
    for _, _, block in scored[:max_count]:
        duration = max(0.0, float(block.get("end") or 0.0) - float(block.get("start") or 0.0))
        block["video_speed"] = 1.5 if duration >= 20.0 else 1.25
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
            "visual": str(block.get("visual") or "")[:120],
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


def _resolve_edit_segments_for_target(raw_segments: List[Dict], duration: float, target_duration: str) -> List[Dict]:
    segments = _normalize_edit_segments(raw_segments or [], duration)
    if not segments:
        segments = _fallback_edit_segments(duration, target_duration)

    if target_duration == "full" and duration > 0:
        wanted = _target_visual_duration_seconds(duration, target_duration)
        total = _segments_total_duration(segments)
        if len(segments) <= 1 or total < wanted * 0.65 or total > wanted * 1.6:
            return _fallback_edit_segments(duration, target_duration)

    return segments


def _minimum_narration_chars_for_seconds(target_seconds: float, language: str) -> int:
    if (language or "").lower().startswith("zh"):
        return int(min(16000, max(1200, target_seconds * 4.2)))
    return int(min(12000, max(900, target_seconds * 2.6)))


def _spoken_block_chars_per_second(language: str) -> float:
    if (language or "").lower().startswith("zh"):
        return 4.2
    return 2.6


def _prompt_spoken_block_chars_per_second(language: str) -> float:
    if (language or "").lower().startswith("zh"):
        return 5.4
    return 3.4


def _minimum_spoken_block_chars(target_seconds: float, language: str) -> int:
    if (language or "").lower().startswith("zh"):
        return int(max(36, target_seconds * _spoken_block_chars_per_second(language)))
    return int(max(24, target_seconds * _spoken_block_chars_per_second(language)))


def _minimum_narration_chars(duration: float, target_duration: str, language: str) -> int:
    if target_duration != "full":
        return 1
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
    return _minimum_narration_chars_for_seconds(target_seconds, language)


def _minimum_narration_chars_for_blocks(blocks: List[Dict], duration: float, target_duration: str, language: str) -> int:
    base_min = _minimum_narration_chars(duration, target_duration, language)
    if target_duration != "full":
        return base_min
    spoken_seconds = sum(
        _block_visual_duration(block)
        for block in blocks
        if not bool(block.get("pause"))
    )
    if spoken_seconds <= 0:
        return base_min
    return min(base_min, _minimum_narration_chars_for_seconds(spoken_seconds, language))


def _accepted_minimum_narration_chars(min_chars: int) -> int:
    ratio = max(0.5, min(1.0, FULL_MODE_MIN_NARRATION_ACCEPTANCE_RATIO))
    return max(1, int(math.floor(min_chars * ratio)))


def _block_narration_density_instruction(language: str) -> str:
    validation_chars_per_second = _spoken_block_chars_per_second(language)
    prompt_chars_per_second = _prompt_spoken_block_chars_per_second(language)
    return (
        "Hard requirement before style, jokes, title, or pacing: every non-pause narration_blocks item must either pass the character-density check or be redesigned. "
        f"For every non-pause item, calculate playable_visual_seconds = (end - start) / video_speed and required_min_chars = ceil(playable_visual_seconds * {prompt_chars_per_second:.1f}) non-whitespace characters; this safely clears the backend validator at {validation_chars_per_second:.1f} chars/second. "
        "Before returning JSON, audit every block one by one and write those numbers into density_audit. "
        "If a sparse visual moment does not justify that much natural commentary, do not pad it with meaningless words: shorten the source range, split the block, mark only a brief pause=true section with empty narration, or use video_speed only when the visible action is genuinely slow or repetitive. "
        "The correct fix for a low-information scene is better block design, not filler narration."
    )


def _maximum_narration_chars(duration: float, target_duration: str, language: str) -> int:
    if target_duration != "full":
        return 0
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
    min_chars = _minimum_narration_chars(duration, target_duration, language)
    if (language or "").lower().startswith("zh"):
        readable_limit = int(max(min_chars + 1800, target_seconds * 6.2))
        return int(min(FULL_MODE_MAX_NARRATION_CHARS_ZH, readable_limit))
    readable_limit = int(max(min_chars + 2200, target_seconds * 4.2))
    return int(min(FULL_MODE_MAX_NARRATION_CHARS_OTHER, readable_limit))


def _target_narration_block_count(duration: float, target_duration: str) -> int:
    if target_duration != "full":
        return 0
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
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


def _repair_near_miss_narration_density_blocks(data: Dict, language: str) -> None:
    blocks = data.get("narration_blocks") or []
    changed = False
    for block in blocks:
        if not isinstance(block, dict) or bool(block.get("pause")):
            continue
        block_duration = _block_visual_duration(block)
        if block_duration <= 0:
            continue
        block_chars = len(re.sub(r"\s+", "", str(block.get("narration") or "")))
        min_block_chars = _accepted_minimum_narration_chars(_minimum_spoken_block_chars(block_duration, language))
        if block_chars <= 0 or block_chars >= min_block_chars:
            continue
        if block_chars / max(1, min_block_chars) < FULL_MODE_NEAR_MISS_REPAIR_MIN_RATIO:
            continue
        source_duration = _block_source_duration(block)
        if source_duration <= 0:
            continue
        accepted_chars_per_second = _spoken_block_chars_per_second(language) * max(0.5, min(1.0, FULL_MODE_MIN_NARRATION_ACCEPTANCE_RATIO))
        target_visual_duration = max(0.1, block_chars / accepted_chars_per_second)
        required_speed = source_duration / target_visual_duration
        current_speed = _safe_video_speed(block.get("video_speed"))
        max_allowed_speed = min(FULL_MODE_RENDER_SYNC_MAX_VIDEO_SPEED, current_speed * FULL_MODE_NEAR_MISS_REPAIR_MAX_SPEEDUP)
        if required_speed <= current_speed or required_speed > max_allowed_speed:
            continue
        block["video_speed"] = round(required_speed, 3)
        changed = True
    if changed:
        data["narration_blocks"] = blocks
        data["edit_segments"] = _narration_blocks_to_edit_segments(blocks)


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


def _derive_commentary_episodes_from_blocks(blocks: List[Dict], title: str) -> List[Dict]:
    if len(blocks) < 4:
        return []
    episode_count = min(6, max(2, math.ceil(len(blocks) / 4)))
    chunk_size = math.ceil(len(blocks) / episode_count)
    episodes = []
    for index, start_block in enumerate(range(1, len(blocks) + 1, chunk_size), start=1):
        end_block = min(len(blocks), start_block + chunk_size - 1)
        episodes.append({
            "episode_number": index,
            "title": f"第{index}集：{title or '二创解说'}",
            "summary": "按完整解说时间线拆分的连续观看分集。",
            "start_block": start_block,
            "end_block": end_block,
        })
    return episodes


def _normalize_commentary_episodes(data: Dict, duration: float, target_duration: str) -> None:
    raw_plan = data.get("episode_plan") if isinstance(data.get("episode_plan"), dict) else {}
    raw_episodes = data.get("episodes") if isinstance(data.get("episodes"), list) else []
    should_split = bool(raw_plan.get("should_split")) or bool(raw_episodes)
    reason = str(raw_plan.get("reason") or "").strip()
    blocks = data.get("narration_blocks") or []
    if target_duration != "full" or not blocks:
        data["episode_plan"] = {"should_split": False, "reason": reason}
        data["episodes"] = []
        return

    episodes = []
    previous_end_block = 0
    for default_number, raw in enumerate(raw_episodes, start=1):
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

    if should_split and not episodes:
        episodes = _derive_commentary_episodes_from_blocks(blocks, str(data.get("title") or ""))
        reason = reason or "AI 判断适合分集，已按完整解说时间线对齐拆分。"

    for index, episode in enumerate(episodes, start=1):
        episode["episode_number"] = index
        if not episode.get("title"):
            episode["title"] = f"第{index}集"

    data["episode_plan"] = {"should_split": bool(episodes), "reason": reason}
    data["episodes"] = episodes


def _normalize_script_timeline(data: Dict, duration: float, target_duration: str, language: str = "") -> None:
    blocks = _normalize_narration_blocks(data.get("narration_blocks") or [], duration)
    if target_duration == "full" and blocks:
        data["narration_blocks"] = blocks
        _repair_near_miss_narration_density_blocks(data, language)
        data["edit_segments"] = _narration_blocks_to_edit_segments(data.get("narration_blocks") or blocks)
        data.setdefault("cut_strategy", [])
    else:
        data["edit_segments"] = _resolve_edit_segments_for_target(data.get("edit_segments", []), duration, target_duration)
        if blocks:
            data["narration_blocks"] = blocks
    _normalize_commentary_episodes(data, duration, target_duration)


def _normalize_script_narration(data: Dict) -> str:
    narration = str(data.get("narration") or "").strip()
    block_narration = _narration_from_blocks(data)
    if len(re.sub(r"\s+", "", block_narration)) > len(re.sub(r"\s+", "", narration)):
        narration = block_narration
    return narration


def _banned_phrase_instruction() -> str:
    phrases = "、".join(f"“{phrase}”" for phrase in COMMENTARY_BANNED_PHRASES)
    return (
        f"- Never use these banned phrases anywhere in the returned JSON: {phrases}. Describe the concrete scene directly instead of using meta-summary wording.\n"
        "- In narration and narration_blocks.narration, do not use the word '镜头' or camera/meta phrasing like '镜头切到', '镜头拉近', '镜头里', '镜头展示', or '镜头带我们'. Describe the subject and action directly instead."
    )


def _validate_no_banned_commentary_phrases(data: Dict) -> None:
    serialized = json.dumps(data, ensure_ascii=False)
    for phrase in COMMENTARY_BANNED_PHRASES:
        if phrase in serialized:
            raise Exception(f"AI commentary output contains banned phrase: {phrase}")


def _validate_no_banned_narration_patterns(data: Dict) -> None:
    narration_parts = [str(data.get("narration") or "")]
    for block in data.get("narration_blocks") or []:
        if isinstance(block, dict):
            narration_parts.append(str(block.get("narration") or ""))
    narration = "\n".join(narration_parts)
    for pattern in COMMENTARY_NARRATION_BANNED_PATTERNS:
        match = pattern.search(narration)
        if match:
            raise Exception(f"AI commentary narration contains camera/meta phrasing: {match.group(0)}")


def _has_visual_plan(data: Dict) -> bool:
    return bool(data.get("narration_blocks") or data.get("chapters"))


def _validate_commentary_script_for_target(data: Dict, duration: float, target_duration: str, language: str) -> None:
    _validate_no_banned_commentary_phrases(data)
    _validate_no_banned_narration_patterns(data)
    if target_duration != "full":
        return
    blocks = _normalize_narration_blocks(data.get("narration_blocks") or [], duration)
    if not blocks:
        raise Exception(
            "AI narration_blocks are required for comprehensive full-mode commentary. "
            "OpenShorts needs timestamped narration blocks so each voiceover section can stay synced with the matching selected visual range."
        )
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
    edit_segments = _narration_blocks_to_edit_segments(blocks)
    visual_seconds = sum(_block_visual_duration(block) for block in blocks)
    if target_seconds > 0 and (len(blocks) <= 1 or visual_seconds < target_seconds * 0.65 or visual_seconds > target_seconds * 1.6):
        raise Exception(
            "AI narration_blocks do not match the selected full-mode edit target. "
            f"Got {visual_seconds:.1f}s of block-matched visuals for a {target_seconds:.1f}s target."
        )
    if not _segments_have_real_cuts(edit_segments, duration, target_seconds):
        selected_source_seconds = _segments_total_duration(edit_segments)
        raise Exception(
            "AI returned a near-full-source timeline instead of an edited full-mode cut strategy. "
            f"Got {selected_source_seconds:.1f}s selected from a {duration:.1f}s source for a {target_seconds:.1f}s target."
        )
    pause_seconds = 0.0
    longest_pause = 0.0
    consecutive_pauses = 0
    max_consecutive_pauses = 0
    spoken_blocks = 0
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
            block_chars = len(re.sub(r"\s+", "", str(block.get("narration") or "")))
            min_block_chars = _accepted_minimum_narration_chars(_minimum_spoken_block_chars(block_duration, language))
            if block_chars < min_block_chars:
                raise Exception(
                    "AI narration block is too short for its selected visual range. "
                    f"Block {index} has {block_chars} chars for {block_duration:.1f}s of playable visuals; "
                    f"expected at least {min_block_chars}. Rewrite this block from the source visuals: add richer scene-matched commentary, shorten the range, split it, or mark only a brief intentional pause. "
                    "Do not leave visible footage with no commentary."
                )
    pause_ratio = pause_seconds / visual_seconds if visual_seconds > 0 else 0.0
    if spoken_blocks <= 0:
        raise Exception("AI returned only no-commentary footage. Full-mode commentary needs narrated blocks between any pause blocks.")
    if pause_ratio > FULL_MODE_MAX_PAUSE_RATIO:
        raise Exception(
            "AI returned too much no-commentary footage. "
            f"Pause blocks cover {pause_seconds:.1f}s of {visual_seconds:.1f}s selected visuals; "
            f"allowed at most {FULL_MODE_MAX_PAUSE_RATIO * 100:.0f}%."
        )
    if longest_pause > FULL_MODE_MAX_PAUSE_SECONDS:
        raise Exception(
            "AI returned an overlong no-commentary pause block. "
            f"Got {longest_pause:.1f}s; allowed at most {FULL_MODE_MAX_PAUSE_SECONDS:.1f}s."
        )
    if max_consecutive_pauses > FULL_MODE_MAX_CONSECUTIVE_PAUSE_BLOCKS:
        raise Exception("AI returned consecutive no-commentary pause blocks; pauses must be separated by narrated blocks.")
    if duration > target_seconds * 1.6:
        latest_end = max(float(block.get("end") or 0.0) for block in blocks)
        required_latest_end = duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION
        if latest_end < required_latest_end:
            raise Exception(
                "AI narration_blocks stopped before the end of the full source timeline. "
                f"Latest selected source timestamp is {latest_end:.1f}s, but this {duration:.1f}s source requires at least one selected block after {required_latest_end:.1f}s. "
                "The generated edit would ignore the later part of the video."
            )
    narration = re.sub(r"\s+", "", str(data.get("narration") or ""))
    min_chars = _minimum_narration_chars_for_blocks(blocks, duration, target_duration, language)
    accepted_min_chars = _accepted_minimum_narration_chars(min_chars)
    max_chars = _maximum_narration_chars(duration, target_duration, language)
    if len(narration) < accepted_min_chars:
        raise Exception(
            "AI narration is too short for comprehensive full-mode commentary. "
            f"Got {len(narration)} chars; expected at least {accepted_min_chars}. "
            "The generated video would not match the selected visuals, so OpenShorts rejected it."
        )
    if max_chars and len(narration) > max_chars:
        raise Exception(
            "AI narration is too long for comprehensive full-mode commentary. "
            f"Got {len(narration)} chars; expected at most {max_chars}. "
            "The generated voiceover would run much longer than the selected visuals and can overload local rendering, so OpenShorts rejected it."
        )


def _validate_voiceover_duration_for_target(
    voiceover_path: str,
    edit_segments: List[Dict],
    duration: float,
    target_duration: str,
) -> None:
    if target_duration != "full":
        return
    visual_seconds = _segments_total_duration(edit_segments) or _target_visual_duration_seconds(duration, target_duration)
    if visual_seconds <= 0:
        return
    audio_seconds = _get_audio_duration(voiceover_path)
    max_seconds = max(visual_seconds + 60.0, visual_seconds * FULL_MODE_MAX_VOICEOVER_DURATION_RATIO)
    if audio_seconds > max_seconds:
        raise Exception(
            "Generated voiceover is too long for comprehensive full-mode commentary. "
            f"Got {audio_seconds:.1f}s audio for {visual_seconds:.1f}s selected visuals; allowed at most {max_seconds:.1f}s. "
            "OpenShorts stopped before visual rendering to avoid an oversized local FFmpeg job."
        )


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
    blocks = invalid_script.get("narration_blocks") or []
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


def _focused_validation_repair_instruction(
    validation_error: Optional[Exception],
    invalid_script: Dict,
    language: str,
    block_count: int,
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
        validation_chars_per_second = _spoken_block_chars_per_second(language)
        prompt_chars_per_second = _prompt_spoken_block_chars_per_second(language)
        repair_chars = max(expected_chars + 24, int(math.ceil(playable_seconds * prompt_chars_per_second)))
        max_seconds_for_current_text = max(1.0, current_chars / validation_chars_per_second)
        return f"""
FOCUSED REPAIR REQUIRED:
- The previous JSON failed specifically at narration_blocks[{block_index - 1}] / Block {block_index}.
- That block has {current_chars} non-whitespace characters for {playable_seconds:.1f}s of playable visuals, but it needs at least {expected_chars} characters to pass validation.
- Failing block from previous JSON: {json.dumps(block, ensure_ascii=False)}
- First priority: make this exact block pass the character-density validator; do not spend tokens rewriting already-valid blocks.
- In the next JSON, keep all unrelated narration_blocks unchanged. Fix only this block, or this block plus its immediate neighbors if a timestamp boundary must move.
- Preferred fix: expand this block's narration to at least {repair_chars} scene-matched characters while staying speakable. Alternative fixes: shorten its source range so playable visuals are about {max_seconds_for_current_text:.1f}s or less, increase video_speed only if the footage is visibly slow/repetitive, or split this visual idea across adjacent chronological blocks while preserving exactly {block_count} total narration_blocks.
- Re-check every non-pause block before returning JSON: len(non_whitespace(narration)) must be >= ceil(((end - start) / video_speed) * {prompt_chars_per_second:.1f}) so it clears the hard validator at {validation_chars_per_second:.1f} chars/second.
- Do not convert this block to pause=true unless it is a short intentional visual-only reveal; do not leave ordinary process footage without commentary.
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
- In the next JSON, make real editorial cuts: choose only the strongest process stages, remove waiting/setup/transport/repeated hammering/camera drift, and keep total playable visual time close to {target_seconds:.1f}s.
- Preserve chronological coverage from beginning, middle, and ending, but do not keep long uncut ranges just to cover time.
""".strip()
    return ""



def _build_regeneration_prompt(
    original_prompt: str,
    short_script: Dict,
    duration: float,
    target_duration: str,
    language: str,
    attempt: int = 1,
    validation_error: Optional[Exception] = None,
) -> str:
    min_chars = _minimum_narration_chars(duration, target_duration, language)
    max_chars = _maximum_narration_chars(duration, target_duration, language)
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
    block_count = _target_narration_block_count(duration, target_duration)
    min_chars_per_block = max(120, math.ceil((min_chars / max(1, block_count)) * 1.25))
    target_block_seconds = target_seconds / max(1, block_count)
    block_density_instruction = _block_narration_density_instruction(language)
    focused_repair_instruction = _focused_validation_repair_instruction(validation_error, short_script, language, block_count)
    return f"""{original_prompt}

PREVIOUS RESPONSE WAS INVALID:
{json.dumps(short_script, ensure_ascii=False)}

VALIDATION ERROR:
{validation_error or "The previous script failed full-mode commentary validation."}

{focused_repair_instruction}

REGENERATE FROM THE ATTACHED VIDEO:
- This is regeneration attempt {attempt}. Discard the previous narration; do not expand it.
- Use the attached video visual evidence again and write a fresh full commentary script.
- Select chronological edit_segments that cover every major visible process stage across the source timeline while cutting repetitive, slow, duplicated, waiting, setup, walking, camera drift, and low-value filler ranges.
- The final narration_blocks must include selected source ranges from the beginning, middle, and later ending portion of the source; at least one block must end after {int(duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION)} seconds.
- Do not return a continuous near-full-source timeline; select about {int(target_seconds)} seconds of useful visuals, not {int(duration)} seconds.
- Write detailed scene-by-scene narration for about {int(target_seconds)} seconds of edited visuals.
{_banned_phrase_instruction()}
- Narration must be at least {min_chars} non-whitespace characters.
- Narration must be at most {max_chars} non-whitespace characters; do not create a voiceover longer than the selected visuals.
- Return exactly {block_count} narration_blocks with start, end, visual, narration, pause, rate, pitch, and video_speed.
- If episode_plan.should_split=true, keep episodes aligned to the repaired 1-based narration_blocks indexes using start_block and end_block.
- Aim for about {target_block_seconds:.0f}s playable visuals per block; most non-pause blocks should be 25-55s after video_speed, not 70-100s long ranges.
- Non-pause narration_blocks must each contain at least {min_chars_per_block} non-whitespace characters; this is a blocking requirement, not a style suggestion. pause=true blocks must leave narration empty.
- {block_density_instruction}
- Each non-pause narration block must be speakable inside that block's visual duration; do not cram long narration into a short range.
- Use pause=true blocks sparingly for key reveals, process sounds, skilled visual moments, transitions, or scenes where the picture genuinely needs to play without commentary; keep pause blocks short, usually 2-8 seconds, under about 15% of selected visual time, and never back-to-back.
- Use video_speed above 1.0 only for visibly slow, repetitive, waiting, setup, walking, transport, or process-transition ranges; valid range is 1.0 to 2.5. Keep key reveals, endings, readable text, final results, and effect showcases at 1.0 unless the footage is clearly slow and still understandable.
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
    min_chars = _minimum_narration_chars(duration, target_duration, language)
    max_chars = _maximum_narration_chars(duration, target_duration, language)
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
    block_count = _target_narration_block_count(duration, target_duration)
    min_chars_per_block = max(120, math.ceil((min_chars / max(1, block_count)) * 1.25))
    target_block_seconds = target_seconds / max(1, block_count)
    block_density_instruction = _block_narration_density_instruction(language)
    focused_repair_instruction = _focused_validation_repair_instruction(validation_error, visual_plan, language, block_count)
    return f"""You are writing the final voiceover for a commentary remix.

VIDEO-DERIVED VISUAL PLAN:
{json.dumps(visual_plan, ensure_ascii=False)}

VALIDATION ERROR:
{validation_error or "The previous script needs full-mode validation before rendering."}

{focused_repair_instruction}

FINALIZE COMPLETE COMMENTARY:
- This is finalization attempt {attempt}; use the video-derived visual plan above as the source of visual truth.
- Do not invent unrelated scenes. Every paragraph must follow the timestamps, visual descriptions, chapters, or edit_segments in the visual plan.
- If the visual plan preserves a near-full-source continuous timeline, replace it with a real edit decision list that selects about {int(target_seconds)} seconds and removes repetitive, slow, duplicated, waiting, setup, walking, camera drift, and low-value filler ranges.
- Write a complete Simplified Chinese voiceover for about {int(target_seconds)} seconds of edited visuals.
{_banned_phrase_instruction()}
- The top-level title must clearly say what the video is doing: name the concrete subject, process/action, and result or purpose. Use titles like "废旧电机拆解回收铜线全过程" instead of vague hype titles like "震撼工厂全过程" or "不可思议的改造".
- The final narration must be at least {min_chars} non-whitespace characters.
- The final narration must be at most {max_chars} non-whitespace characters; keep it paced for the selected visuals instead of stretching the edit.
- Return exactly {block_count} narration_blocks with start, end, visual, narration, pause, rate, pitch, and video_speed.
- If episode_plan.should_split=true, keep episodes aligned to the repaired 1-based narration_blocks indexes using start_block and end_block.
- Aim for about {target_block_seconds:.0f}s playable visuals per block; most non-pause blocks should be 25-55s after video_speed, not 70-100s long ranges.
- Non-pause narration_blocks must each contain at least {min_chars_per_block} non-whitespace characters; this is a blocking requirement, not a style suggestion. pause=true blocks must leave narration empty.
- {block_density_instruction}
- Each non-pause narration block must be speakable inside that block's visual duration.
- Use pause=true blocks sparingly for key reveals, process sounds, skilled visual moments, transitions, or scenes where the picture genuinely needs to play without commentary; keep pause blocks short, usually 2-8 seconds, under about 15% of selected visual time, and never back-to-back.
- Use video_speed above 1.0 only for visibly slow, repetitive, waiting, setup, walking, transport, or process-transition ranges; valid range is 1.0 to 2.5. Keep key reveals, endings, readable text, final results, and effect showcases at 1.0 unless the footage is clearly slow and still understandable.
- Vary rate and pitch across non-pause blocks so the voice has cadence; do not return every block as +0% and +0Hz.
- Preserve chronological order and keep the commentary matched to the visible factory process.
- Return valid JSON only.

JSON FORMAT:
{{
  "title": "specific title that says what the video does",
  "summary": "brief summary",
  "hook": "opening hook",
  "narration": "complete final voiceover text",
  "narration_blocks": [
    {{"start": 0, "end": 30, "visual": "visual plan item", "narration": "final voiceover for this range", "pause": false, "rate": "+0%", "pitch": "+0Hz", "video_speed": 1.0}},
    {{"start": 30, "end": 40, "visual": "original footage moment that should breathe", "narration": "", "pause": true, "rate": "+0%", "pitch": "+0Hz", "video_speed": 1.0}}
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


def _build_commentary_prompt(
    transcript: Dict,
    video_title: str,
    duration: float,
    language: str,
    style: str,
    target_duration: str,
    analysis_mode: str,
    visual_analysis: Optional[Dict] = None,
) -> str:
    mode = _normalize_analysis_mode(analysis_mode)
    sampled_segments = _sample_transcript_segments(transcript)
    transcript_text = transcript.get("text", "")
    if len(transcript_text) > 60000:
        transcript_text = transcript_text[:60000]
    min_chars = _minimum_narration_chars(duration, target_duration, language)
    max_chars = _maximum_narration_chars(duration, target_duration, language)
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
    block_count = _target_narration_block_count(duration, target_duration)
    min_chars_per_block = max(120, math.ceil((min_chars / max(1, block_count)) * 1.25))
    target_block_seconds = target_seconds / max(1, block_count)
    block_density_instruction = _block_narration_density_instruction(language)
    openai_one_shot_density_instruction = ""
    if mode == "openai" and target_duration == "full":
        openai_one_shot_density_instruction = (
            "- For OpenAI-compatible mode, the first complete JSON response must pass every per-block density check; "
            "the backend will not send follow-up repair requests just to fix under-length narration blocks."
        )
    style_grounding = _style_grounding_instruction(style, language)

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
            "- All edit_segments and narration_blocks must use timestamps from the original full source video timeline and must be selected from across the complete beginning, middle, and ending timeline."
        )
        if visual_analysis:
            visual_analysis_text = _openai_visual_analysis_prompt_text(visual_analysis)
    else:
        visual_instruction = (
            "- Attached images, if present, are sampled keyframes. Treat them as lightweight visual context, "
            "not as the full source video."
        )

    openai_visual_section = f"""
OPENAI-COMPATIBLE MULTIMODAL VISUAL TIMELINE:
{visual_analysis_text}
""" if visual_analysis_text else ""

    return f"""You are an expert video essay writer and short-form commentary producer.

TASK:
Transform the source YouTube video into an original commentary/remix narration script.

SOURCE VIDEO:
Title: {video_title}
Duration seconds: {duration:.1f}
Detected transcript language: {transcript.get('language', 'unknown')}

OUTPUT LANGUAGE: {language}
COMMENTARY STYLE: {style}
TARGET DURATION: {_target_duration_hint(target_duration, duration)}

SOURCE TRANSCRIPT:
{transcript_text}

TIMESTAMPED SAMPLE SEGMENTS:
{json.dumps(sampled_segments, ensure_ascii=False)}

VISUAL CONTEXT:
{visual_instruction}
{openai_visual_section}
RULES:
{_banned_phrase_instruction()}
- Do not merely translate the transcript.
- Rewrite it as an original, natural commentary narration.
- Preserve the important facts, sequence, and context from the source.
- Select which original video ranges should be kept for the final edit and which ranges should be removed. Remove repetitive, slow, duplicated, waiting, setup, walking, camera drift, intro/outro, irrelevant, or low-value filler parts.
- The kept visual ranges must stay in the same chronological order as the source video, but they should not form one continuous full-source range when the source is longer than the target.
- The narration must match the selected visual ranges, not the removed parts.
- Treat narration_blocks as the production timeline: each block's start/end is the source-video range that will play while that exact block's narration is spoken.
- Do not describe a visual before it appears or after it has already passed; if a sentence mentions a machine action, material state, worker movement, comparison, or joke, it must belong to that same block's visible time range.
- Keep each block self-contained: first ground the viewer in the concrete visible action, then add interpretation or commentary for that exact action.
- {style_grounding}
- For TARGET DURATION full, first analyze the complete source visual timeline from 0.0 seconds through {duration:.1f} seconds; do not stop after a short highlight scan, and do not summarize only the first few minutes.
- For TARGET DURATION full, produce a real edit decision list: select only about {int(target_seconds)} seconds of the best visual ranges from the complete source timeline, and intentionally skip redundant or low-value ranges.
- For TARGET DURATION full, do not output one continuous 0-to-{duration:.1f} timeline unless the source duration is already close to {int(target_seconds)} seconds. For this source, the selected visual duration should be near {int(target_seconds)} seconds, not {int(duration)} seconds.
- For TARGET DURATION full, write a detailed scene-by-scene commentary that covers the chosen visual ranges from start to finish across the full source timeline, including beginning, middle, and ending portions. Do not return a 60-second summary over a long source.
- For TARGET DURATION full, if the source has a final payoff, result reveal, before/after comparison, effect showcase, completed product, or conclusion, include the visual range where that result actually appears and let it play through.
- For TARGET DURATION full, the selected blocks must not stop in the first half of a long source; at least one narration_blocks item must end after {int(duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION)} seconds.
- For TARGET DURATION full, narration_blocks is required: output exactly {block_count} chronological blocks. Each block must have start, end, visual, narration, pause, rate, pitch, and video_speed.
- For TARGET DURATION full, aim for about {target_block_seconds:.0f}s playable visuals per block; most non-pause blocks should be 25-55s after video_speed. Do not create 70-100s blocks unless the narration is dense enough for the entire range.
- For TARGET DURATION full, narration_blocks must cover about {int(target_seconds)} seconds of selected visuals across the complete source timeline and must cover the same ranges as edit_segments; do not create narration for ranges that are not kept.
- For TARGET DURATION full, most selected visual blocks should contain narration, but do not narrate every second like a robot; use short breathing room only when the footage benefits from it, and avoid long stretches where footage plays without commentary unless the source audio itself is essential.
- For TARGET DURATION full, use pause=true blocks sparingly when the original footage genuinely needs to be heard without commentary: key reveals, machine/process sounds, skilled hand work, visual proof, emotional beats, transitions, or moments where the picture explains itself. Pause blocks must leave narration empty and should usually last 2-8 seconds.
- For TARGET DURATION full, pause blocks should use the original source audio as the main sound, but total pause time must stay under about 15% of selected visual time. Do not place pause blocks back-to-back.
- For TARGET DURATION full, each non-pause block's narration must be speakable inside that block's visual duration; do not put 2 minutes of words into a 20-second visual range.
- For TARGET DURATION full, use about {min_chars_per_block} non-whitespace characters as a planning average for a normal {target_block_seconds:.0f}s narrated block, not as a reason to overfill short or sparse visual moments.
- For TARGET DURATION full, {block_density_instruction}
{openai_one_shot_density_instruction}
- For TARGET DURATION full, if a selected visual range is too long or too visually sparse for a high-quality natural commentary paragraph, redesign the block: shorten the range, split it, or use a brief pause=true moment for original audio/visual breathing room. Do not rely on silent padding, meaningless word padding, extreme TTS pacing, or later render-time fixes.
- For TARGET DURATION full, use rate to create cadence: slower values like "-10%" for important reveals or emotional emphasis, faster values like "+12%" for energetic process sections. Valid range: "-30%" to "+30%".
- For TARGET DURATION full, use pitch lightly for tone: lower values like "-3Hz" for weight, higher values like "+3Hz" for excitement. Valid range: "-15Hz" to "+15Hz".
- For TARGET DURATION full, use video_speed above 1.0 only for visibly slow, repetitive, waiting, setup, walking, transport, or process-transition ranges; valid range is 1.0 to 2.5. Keep key reveals, endings, readable text, final results, and effect showcases at 1.0 unless the footage is clearly slow and still understandable.
- For TARGET DURATION full, vary rate and pitch across blocks; do not leave every non-pause block at "+0%" and "+0Hz".
- For TARGET DURATION full, total narration must be at least {min_chars} non-whitespace characters and should cover about {int(target_seconds)} seconds of edited visuals.
- For TARGET DURATION full, total narration must be at most {max_chars} non-whitespace characters so the voiceover does not exceed the selected visuals.
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
    {{"start": 0, "end": 30, "visual": "what is visible in this range", "narration": "voiceover for this visual range", "pause": false, "rate": "+0%", "pitch": "+0Hz", "video_speed": 1.0}},
    {{"start": 30, "end": 40, "visual": "original footage moment that should breathe", "narration": "", "pause": true, "rate": "+0%", "pitch": "+0Hz", "video_speed": 1.0}}
  ],
  "density_audit": [
    {{"block": 1, "playable_visual_seconds": 30, "required_min_chars": 162, "actual_non_whitespace_chars": 190, "passes": true}}
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
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_timeout = timeout_seconds or OPENAI_REQUEST_TIMEOUT_SECONDS
    response = None
    last_error = None
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


def _openai_visual_analysis_cache_path(output_dir: str) -> str:
    return os.path.join(output_dir, OPENAI_VISUAL_ANALYSIS_CACHE)


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
- Describe concrete visible actions, materials, people, machines, scene changes, reveals, and process stages.
- Do not invent facts that are not visible.
- Mark frames/ranges that look valuable for a commentary edit.
- Keep observations concise but specific enough to ground narration later.

JSON FORMAT:
{{
  "batch_index": {batch_index},
  "observations": [
    {{"timestamp": 12.3, "visual": "what is visible", "process_stage": "stage name", "importance": 1, "keep_candidate": true}}
  ],
  "candidate_segments": [
    {{"start": 10.0, "end": 25.0, "reason": "why this visual range should be kept"}}
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
        prompt = _openai_visual_batch_prompt(video_title, duration, batch, index, len(batches))
        text = _call_openai_compatible_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=_build_openai_visual_batch_messages(prompt, batch),
            max_tokens=OPENAI_VISUAL_MAX_TOKENS,
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
    if visual_concurrency <= 1:
        for index, batch in enumerate(batches, start=1):
            if progress:
                progress(f"OpenAI-compatible multimodal visual analysis batch {index}/{len(batches)}...")
            batch_results[index - 1] = analyze_batch(index, batch)
    else:
        with ThreadPoolExecutor(max_workers=visual_concurrency) as executor:
            futures = {
                executor.submit(analyze_batch, index, batch): index
                for index, batch in enumerate(batches, start=1)
            }
            completed = 0
            for future in as_completed(futures):
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
) -> str:
    min_chars = _minimum_narration_chars(duration, target_duration, language)
    max_chars = _maximum_narration_chars(duration, target_duration, language)
    target_seconds = _target_visual_duration_seconds(duration, target_duration)
    block_count = _target_narration_block_count(duration, target_duration)
    min_chars_per_block = max(120, math.ceil((min_chars / max(1, block_count)) * 1.25))
    target_block_seconds = target_seconds / max(1, block_count)
    block_density_instruction = _block_narration_density_instruction(language)
    focused_repair_instruction = _focused_validation_repair_instruction(validation_error, invalid_script, language, block_count)
    return f"""{original_prompt}

PREVIOUS RESPONSE WAS INVALID:
{json.dumps(invalid_script, ensure_ascii=False)}

VALIDATION ERROR:
{validation_error}

{focused_repair_instruction}

REPAIR FROM THE TRANSCRIPT AND MULTIMODAL VISUAL TIMELINE:
- This is repair attempt {attempt}. Preserve valid blocks from the previous JSON; do not rewrite the whole script unless the validation error makes the timeline globally impossible.
- Fix the specific validation failure first, then re-check every non-pause block's character density before returning JSON.
- Use the timestamped visual timeline and source transcript again only for the failed block or nearby blocks that need local boundary changes.
- Keep chronological edit_segments that cover every major visible process stage across the source timeline while cutting repetitive, slow, duplicated, waiting, setup, walking, camera drift, and low-value filler ranges.
- The final narration_blocks must include selected source ranges from the beginning, middle, and later ending portion of the source; at least one block must end after {int(duration * FULL_MODE_MIN_TIMELINE_COVERAGE_FRACTION)} seconds.
- Do not return a continuous near-full-source timeline; select about {int(target_seconds)} seconds of useful visuals, not {int(duration)} seconds.
{_banned_phrase_instruction()}
- Narration must be at least {min_chars} non-whitespace characters.
- Narration must be at most {max_chars} non-whitespace characters; do not create a voiceover longer than the selected visuals.
- Return exactly {block_count} narration_blocks with start, end, visual, narration, pause, rate, pitch, and video_speed.
- If episode_plan.should_split=true, keep episodes aligned to the repaired 1-based narration_blocks indexes using start_block and end_block.
- Aim for about {target_block_seconds:.0f}s playable visuals per block; most non-pause blocks should be 25-55s after video_speed, not 70-100s long ranges.
- Non-pause narration_blocks must each contain at least {min_chars_per_block} non-whitespace characters; this is a blocking requirement, not a style suggestion. pause=true blocks must leave narration empty.
- {block_density_instruction}
- Each non-pause narration block must be speakable inside that block's visual duration.
- Use pause=true blocks sparingly for key reveals, process sounds, skilled visual moments, transitions, or scenes where the picture genuinely needs to play without commentary; keep pause blocks short, usually 2-8 seconds, under about 15% of selected visual time, and never back-to-back.
- Use video_speed above 1.0 only for visibly slow, repetitive, waiting, setup, walking, transport, or process-transition ranges; valid range is 1.0 to 2.5. Keep key reveals, endings, readable text, final results, and effect showcases at 1.0 unless the footage is clearly slow and still understandable.
- Vary rate and pitch across non-pause blocks so the voice has cadence.
- Return valid JSON only, using the same JSON FORMAT.
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
    style: str = "documentary",
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
    prompt = _build_commentary_prompt(
        transcript=transcript,
        video_title=video_title,
        duration=duration,
        language=language,
        style=style,
        target_duration=target_duration,
        analysis_mode="openai",
        visual_analysis=visual_analysis,
    )
    if progress:
        progress("OpenAI-compatible model is writing commentary script from transcript and visual timeline...")
    response_text = _call_openai_compatible_chat(
        api_key=openai_key,
        base_url=openai_base_url,
        model=openai_model,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        max_tokens=OPENAI_SCRIPT_MAX_TOKENS,
        timeout_seconds=OPENAI_SCRIPT_REQUEST_TIMEOUT_SECONDS,
    )
    validation_error = None
    for script_attempt in range(1, GEMINI_SCRIPT_VALIDATION_ATTEMPTS + 1):
        data = _parse_openai_json(response_text)
        narration = _normalize_script_narration(data)
        if not narration:
            raise Exception("OpenAI-compatible model did not return narration text")
        data["narration"] = narration
        data.setdefault("title", video_title or "Commentary Remix")
        data.setdefault("summary", "")
        _normalize_script_timeline(data, duration, target_duration, language)
        data["narration"] = _normalize_script_narration(data)
        try:
            _validate_commentary_script_for_target(data, duration, target_duration, language)
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
            return data
        except Exception as exc:
            validation_error = exc
            if target_duration != "full" or script_attempt >= GEMINI_SCRIPT_VALIDATION_ATTEMPTS:
                raise
            density_details = _density_validation_failure_details(exc, data)
            if density_details:
                if progress:
                    progress(
                        f"OpenAI-compatible script validation failed: {exc} "
                        "The first full-script response did not satisfy the required per-block narration density; "
                        "not sending extra OpenAI repair requests for under-length blocks."
                    )
                raise
            if progress:
                progress(
                    f"OpenAI-compatible script validation failed on correction attempt {script_attempt}/{GEMINI_SCRIPT_VALIDATION_ATTEMPTS}: "
                    f"{exc} Asking model to repair the invalid full-mode script without rewriting valid blocks..."
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
                        ),
                    }],
                }],
                max_tokens=OPENAI_SCRIPT_MAX_TOKENS,
                timeout_seconds=OPENAI_SCRIPT_REQUEST_TIMEOUT_SECONDS,
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
    for attempt in range(1, GEMINI_FILE_UPLOAD_RETRIES + 1):
        try:
            if progress and attempt > 1:
                progress(f"Retrying Gemini analysis video upload ({attempt}/{GEMINI_FILE_UPLOAD_RETRIES})...")
            uploaded = client.files.upload(file=analysis_video_path)
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
    style: str = "documentary",
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
    )
    if previous_error and target_duration == "full":
        prompt += (
            "\n\nRetry correction note:\n"
            f"The previous response failed validation with this error: {previous_error}\n"
            "Return narration_blocks that cover about the requested full-mode target duration, not the entire raw source timeline. "
            "Do not return a short highlight summary; every selected visual range in the comprehensive edit must have matching narration."
        )
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
        data = json.loads(_clean_json_text(response.text))
        narration = _normalize_script_narration(data)
        if not narration:
            raise Exception("Gemini did not return narration text")
        data["narration"] = narration
        data.setdefault("title", video_title or "Commentary Remix")
        data.setdefault("summary", "")
        _normalize_script_timeline(data, duration, target_duration, language)
        data["narration"] = _normalize_script_narration(data)
        try:
            _validate_commentary_script_for_target(data, duration, target_duration, language)
            data.setdefault("chapters", [])
            data.setdefault("hashtags", [])
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
    segments = edit_segments or [{"start": 0.0, "end": _get_video_duration(video_path)}]
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

    segments = edit_segments or [{"start": 0.0, "end": _get_video_duration(video_path)}]
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
        "-filter:v", f"setpts=PTS/{ratio:.6f}",
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
            concat_path = _windows_path_from_wsl(path) if use_windows_paths else path
            safe_path = concat_path.replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")


def _concat_media_parts(paths: List[str], output_path: str, work_dir: str, codec: str = "copy") -> None:
    if not paths:
        raise Exception("No media parts to concatenate")
    list_path = os.path.join(work_dir, f"concat_{_safe_slug(os.path.basename(output_path))}.txt")
    _write_concat_list(paths, list_path)
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", codec,
        output_path,
    ]
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


def _fit_audio_part_to_duration(input_audio_path: str, output_audio_path: str, target_duration: float, max_speedup: float = 1.35) -> None:
    target_duration = max(0.1, float(target_duration or 0.0))
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
        tmp_path,
    ]
    _run_command(cmd)
    os.replace(tmp_path, audio_path)


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
        output_path,
    ])
    _run_command(cmd)


def _block_video_filter(aspect_filter: str, speed: float) -> str:
    filters = []
    speed = _safe_render_video_speed(speed)
    if speed > 1.0001:
        filters.append(f"setpts=PTS/{speed:.6f}")
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
    needs_ambient_track = spoken_volume > 0 or any(bool(block.get("pause")) and pause_volume > 0 for block in blocks)
    total_blocks = len(blocks)
    resolved_concurrency = min(resolve_commentary_block_concurrency(block_concurrency), total_blocks)
    progress_lock = threading.Lock()

    def report(message: str) -> None:
        if not progress:
            return
        with progress_lock:
            progress(message)

    def process_block(index: int, block: Dict) -> Dict:
        is_pause = bool(block.get("pause"))
        source_start = float(block["start"])
        source_duration = max(0.1, float(block["end"]) - source_start)
        requested_video_speed = _safe_render_video_speed(block.get("video_speed"))
        video_speed = requested_video_speed
        visual_duration = max(0.1, source_duration / video_speed)
        render_source_duration = source_duration
        speed_label = f" at {video_speed:g}x" if video_speed > 1.0001 else ""
        if is_pause:
            report(f"Adding original-audio pause block {index}/{total_blocks}{speed_label}...")
        else:
            report(f"Generating synced commentary block {index}/{total_blocks}{speed_label}...")

        block_voice_path = os.path.join(part_dir, f"block_voice_{index:03d}.mp3")
        if not is_pause:
            generate_commentary_voiceover(
                text=block["narration"],
                output_path=block_voice_path,
                tts_provider=tts_provider,
                language=language,
                elevenlabs_key=elevenlabs_key,
                voice_id=voice_id,
                edge_voice=edge_voice,
                rate=block.get("rate") or "+0%",
                pitch=block.get("pitch") or "+0Hz",
            )

        if not is_pause:
            voice_duration = max(0.1, _get_audio_duration(block_voice_path))
            max_synced_visual_duration = max(0.1, voice_duration + FULL_MODE_MAX_NARRATION_SILENCE_TAIL_SECONDS)
            if visual_duration > max_synced_visual_duration:
                desired_speed = source_duration / max_synced_visual_duration
                video_speed = _safe_render_video_speed(max(requested_video_speed, desired_speed))
                visual_duration = min(visual_duration, max_synced_visual_duration)
                render_source_duration = min(source_duration, max(0.1, visual_duration * video_speed))
                visual_duration = max(0.1, render_source_duration / video_speed)
                if render_source_duration < source_duration * 0.75:
                    raise Exception(
                        "Generated TTS is much shorter than its selected visual range. "
                        f"Block {index} has {voice_duration:.1f}s of narration for {source_duration:.1f}s of source visuals. "
                        "Regenerate the commentary with denser scene-matched narration or shorter visual ranges; refusing to over-trim footage because that would hurt commentary quality."
                    )
                report(
                    f"Tightening commentary block {index}/{total_blocks} to {visual_duration:.1f}s "
                    "so narration stays synced with the visuals..."
                )

        speed_token = int(round(video_speed * 1000))
        source_token = int(round(render_source_duration * 1000))
        duration_token = int(round(visual_duration * 1000))
        cache_token = f"s{speed_token}_src{source_token}_dur{duration_token}"
        fitted_voice_path = os.path.join(part_dir, f"block_voice_fit_{index:03d}_{cache_token}.m4a")
        block_video_path = os.path.join(part_dir, f"block_video_{index:03d}_{cache_token}.mp4")
        block_ambient_path = os.path.join(part_dir, f"block_ambient_{index:03d}_{cache_token}.m4a") if needs_ambient_track else None
        if is_pause:
            _create_silent_audio_clip(fitted_voice_path, visual_duration)
        else:
            _fit_audio_part_to_duration(block_voice_path, fitted_voice_path, visual_duration)
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
            "video_path": block_video_path,
            "voice_path": fitted_voice_path,
            "ambient_path": block_ambient_path,
            "duration": visual_duration,
        }

    if resolved_concurrency > 1:
        with ThreadPoolExecutor(max_workers=resolved_concurrency) as executor:
            futures = [executor.submit(process_block, index, block) for index, block in enumerate(blocks, start=1)]
            block_results = [future.result() for future in as_completed(futures)]
    else:
        block_results = [process_block(index, block) for index, block in enumerate(blocks, start=1)]

    block_results.sort(key=lambda item: item["index"])
    video_parts = [item["video_path"] for item in block_results]
    voice_parts = [item["voice_path"] for item in block_results]
    part_durations = [item["duration"] for item in block_results]
    ambient_parts = [item["ambient_path"] for item in block_results if item.get("ambient_path")]

    visual_duration_total = sum(part_durations)
    _concat_media_parts(video_parts, timed_video_path, part_dir)
    _concat_media_parts(voice_parts, voiceover_path, part_dir, codec="aac")
    _force_audio_clip_duration(voiceover_path, visual_duration_total, part_dir, bitrate="192k")
    if ambient_parts:
        _concat_media_parts(ambient_parts, ambient_audio_path, part_dir, codec="aac")
        _force_audio_clip_duration(ambient_audio_path, visual_duration_total, part_dir, bitrate="128k")
        return ambient_audio_path, part_durations
    return None, part_durations


def _mix_voiceover_with_video(
    video_path: str,
    voiceover_path: str,
    output_path: str,
    original_audio_volume: float = 0.3,
    ambient_audio_path: Optional[str] = None,
    trim_to_voiceover: bool = True,
) -> None:
    if ambient_audio_path and os.path.exists(ambient_audio_path):
        audio_duration_mode = "first" if trim_to_voiceover else "longest"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", voiceover_path,
            "-i", ambient_audio_path,
            "-filter_complex", f"[1:a]volume=1.0[a1];[2:a]volume=1.0[a2];[a1][a2]amix=inputs=2:duration={audio_duration_mode}:dropout_transition=0[a]",
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
    result = subprocess.run(
        [
            FFPROBE_BINARY, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            _windows_path_from_wsl(audio_path) if os.path.basename(str(FFPROBE_BINARY)).lower() == "ffprobe.exe" else audio_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise Exception(result.stderr or result.stdout or f"Failed to probe audio: {audio_path}")
    return float(result.stdout.strip() or 0)


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
    result = subprocess.run(
        [
            FFPROBE_BINARY, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            probe_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return _normalize_ass_dimensions()
    first_line = (result.stdout or "").strip().splitlines()[0:1]
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
    return max(12, min(90, int(available_width / max(1.0, font_size / 2))))


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


def _wrap_ass_subtitle_text(text: str, max_units: int = ASS_SUBTITLE_MAX_LINE_UNITS) -> str:
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
    return r"\N".join(lines) if lines else clean.strip()


def _append_weighted_subtitle_lines(lines: List[str], sentences: List[str], start_time: float, duration: float, max_units: int = ASS_SUBTITLE_MAX_LINE_UNITS) -> None:
    if not sentences or duration <= 0:
        return
    weights = [max(len(sentence), 1) for sentence in sentences]
    total_weight = sum(weights)
    cursor = start_time
    block_end = start_time + duration
    for index, sentence in enumerate(sentences):
        segment_duration = duration * weights[index] / total_weight
        start = cursor
        end = block_end if index == len(sentences) - 1 else min(block_end, cursor + segment_duration)
        cursor = end
        clean = _wrap_ass_subtitle_text(sentence, max_units=max_units)
        lines.append(f"Dialogue: 0,{_format_ass_time(start)},{_format_ass_time(end)},Default,,0,0,0,,{clean}")


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
    style: str = "documentary",
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
    gemini_pool: Optional[GeminiKeyPool] = None,
    progress: Optional[Callable[[str], None]] = None,
    checkpoint: Optional[Callable[[Dict], None]] = None,
    prepared_analysis_video_path: Optional[str] = None,
    gemini_file: Optional[Dict] = None,
    previous_error: Optional[str] = None,
) -> Dict:
    analysis_mode = _normalize_analysis_mode(analysis_mode)
    resolved_gemini_model = gemini_model or DEFAULT_GEMINI_MODEL
    openai_sampling_options = resolve_openai_sampling_options(
        frame_interval_seconds=openai_frame_interval_seconds,
        max_frames=openai_max_frames,
        scene_max_keyframes=openai_scene_max_keyframes,
        batch_size=openai_batch_size,
        visual_concurrency=openai_visual_concurrency,
    ) if analysis_mode == "openai" else None
    resolved_block_concurrency = resolve_commentary_block_concurrency(commentary_block_concurrency)

    def log(message: str) -> None:
        print(f"[Commentary] {message}")
        if progress:
            progress(message)

    os.makedirs(output_dir, exist_ok=True)
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
    log("Preparing source video...")

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
    duration = float(video_info.get("duration") or 0)
    resolved_aspect = "9:16" if vertical else _resolve_aspect_mode(video_info, aspect_mode)
    log(f"Resolved output aspect ratio: {resolved_aspect}")
    frame_paths = []
    openai_frame_infos = []
    if analysis_mode in {"current", "openai"}:
        log("Transcribing full video with Faster-Whisper...")
        transcript = transcribe_video(video_path, language=source_language)
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

    cached_script_path = None
    if checkpoint and previous_error:
        task_script_path = os.path.join(output_dir, "commentary_task.json")
        try:
            with open(task_script_path, "r", encoding="utf-8") as f:
                cached_script_path = (json.load(f) or {}).get("script_path")
        except Exception:
            cached_script_path = None
    if cached_script_path and os.path.exists(cached_script_path):
        log("Reusing cached commentary script from previous attempt...")
        with open(cached_script_path, "r", encoding="utf-8") as f:
            cached_payload = json.load(f)
        script = cached_payload.get("script") or cached_payload
        transcript = cached_payload.get("transcript") or transcript
        _normalize_script_timeline(script, duration, target_duration)
        _validate_commentary_script_for_target(script, duration, target_duration, language)
    elif analysis_mode == "openai":
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
            target_duration=target_duration,
            progress=log,
            openai_sampling_options=openai_sampling_options,
            output_dir=output_dir,
            checkpoint=checkpoint,
        )
    else:
        log("Generating original commentary script with Gemini...")
        script = generate_commentary_script(
            transcript=transcript,
            video_title=video_title,
            duration=duration,
            gemini_key=gemini_key,
            language=language,
            style=style,
            target_duration=target_duration,
            base_url=gemini_base_url,
            frame_paths=frame_paths,
            analysis_video_path=analysis_video_path,
            analysis_mode=analysis_mode,
            gemini_model=gemini_model,
            gemini_pool=gemini_pool,
            progress=log,
            checkpoint=checkpoint,
            gemini_file=gemini_file,
            previous_error=previous_error,
        )

    _normalize_script_timeline(script, duration, target_duration)
    if target_duration != "full" or script.get("narration_blocks"):
        _validate_commentary_script_for_target(script, duration, target_duration, language)
    if target_duration == "full" and script.get("narration_blocks"):
        script["narration_blocks"] = _apply_auto_video_speed_to_blocks(script.get("narration_blocks") or [], auto_video_speed)
        script["edit_segments"] = _narration_blocks_to_edit_segments(script["narration_blocks"])

    slug = _safe_slug(script.get("title") or video_title)
    script_path = os.path.join(output_dir, f"{slug}_commentary_script.json")
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump({"script": script, "transcript": transcript}, f, ensure_ascii=False, indent=2)
    if checkpoint:
        checkpoint({"script_path": script_path})

    voiceover_path = os.path.join(output_dir, f"{slug}_voiceover.mp3")
    narration_blocks = _normalize_narration_blocks(script.get("narration_blocks") or [], duration)
    auto_video_speed_summary = _summarize_auto_video_speed(narration_blocks, auto_video_speed)
    if target_duration == "full" and narration_blocks:
        if not auto_video_speed:
            log("AI auto video speed disabled; rendering all commentary blocks at 1.0x.")
        elif auto_video_speed_summary["accelerated_count"] > 0:
            log(
                "AI auto video speed: "
                f"{auto_video_speed_summary['accelerated_count']}/{auto_video_speed_summary['total_blocks']} blocks accelerated, "
                f"saved about {auto_video_speed_summary['saved_seconds']:.1f}s."
            )
        else:
            log("AI auto video speed: no eligible slow or repetitive blocks selected; rendering all blocks at 1.0x.")
        voiceover_path = os.path.join(output_dir, f"{slug}_voiceover.m4a")
        edit_segments = _narration_blocks_to_edit_segments(narration_blocks)
    else:
        edit_segments = _resolve_edit_segments_for_target(script.get("edit_segments", []), duration, target_duration)

    work_dir = os.path.join(output_dir, f"{slug}_work")
    os.makedirs(work_dir, exist_ok=True)
    edited_video_path = os.path.join(output_dir, f"{slug}_edited_visual.mp4")
    timed_video_path = os.path.join(output_dir, f"{slug}_timed_visual.mp4")
    ambient_audio_path = os.path.join(output_dir, f"{slug}_ambient.m4a")
    trim_to_voiceover = True
    preserve_source_resolution = target_duration == "full"
    synced_block_durations = []

    if target_duration == "full" and narration_blocks:
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
            progress=log,
        )
        edited_video_path = timed_video_path
        _validate_voiceover_duration_for_target(voiceover_path, edit_segments, duration, target_duration)
    else:
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
    mixed_path = os.path.join(output_dir, f"{slug}_mixed.mp4")
    log("Mixing new voiceover with ambient source audio...")
    _mix_voiceover_with_video(
        video_path=timed_video_path,
        voiceover_path=voiceover_path,
        output_path=mixed_path,
        original_audio_volume=original_audio_volume,
        ambient_audio_path=ambient_audio,
        trim_to_voiceover=trim_to_voiceover,
    )

    subtitle_path = None
    final_path = mixed_path
    if subtitles:
        subtitle_path = os.path.join(output_dir, f"{slug}_commentary.ass")
        subtitled_path = os.path.join(output_dir, f"{slug}_final.mp4")
        log("Generating text-timed subtitles from the commentary narration...")
        subtitle_dimensions = _probe_video_dimensions(mixed_path)
        log(f"Subtitle canvas size: {subtitle_dimensions[0]}x{subtitle_dimensions[1]}")
        if target_duration == "full" and narration_blocks:
            _write_block_timed_ass(narration_blocks, subtitle_path, synced_block_durations, video_dimensions=subtitle_dimensions)
        else:
            _write_text_timed_ass(script["narration"], voiceover_path, subtitle_path, video_dimensions=subtitle_dimensions)
        log("Burning subtitles into final video...")
        _burn_subtitles(mixed_path, subtitle_path, subtitled_path)
        final_path = subtitled_path

    episode_plan = script.get("episode_plan") if isinstance(script.get("episode_plan"), dict) else {"should_split": False, "reason": ""}
    commentary_episodes = script.get("episodes") if isinstance(script.get("episodes"), list) else []
    rendered_episodes = []
    if target_duration == "full" and narration_blocks and commentary_episodes and episode_plan.get("should_split"):
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

    metadata = {
        "video_path": final_path,
        "video_filename": os.path.basename(final_path),
        "video_url": f"/videos/{os.path.basename(output_dir)}/{os.path.basename(final_path)}",
        "source_video": os.path.basename(video_path),
        "analysis_mode": analysis_mode,
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
    }
    metadata_path = os.path.join(output_dir, f"{slug}_commentary_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    if checkpoint:
        checkpoint({"metadata_path": metadata_path})

    log("Commentary remix video completed.")
    return metadata
