import os
import re
import uuid
import subprocess
import threading
import json
import shutil
import glob
import time
import asyncio
import yt_dlp
import tempfile
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Dict, Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from s3_uploader import upload_job_artifacts, list_all_clips, upload_actor_to_s3, list_actor_gallery, upload_video_to_gallery, list_video_gallery
from gemini_pool import parse_gemini_pool_config

load_dotenv()

GEMINI_BASE_URL_HEADER = "X-Gemini-Base-URL"
OPENAI_COMPAT_KEY_HEADER = "X-OpenAI-Compatible-Key"
OPENAI_COMPAT_BASE_URL_HEADER = "X-OpenAI-Compatible-Base-URL"
OPENAI_COMPAT_MODEL_HEADER = "X-OpenAI-Compatible-Model"
PROJECT_COOKIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local", "youtube_cookies.txt")
PROJECT_COOKIES_BACKUP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local", "youtube_cookies.last_good.txt")


def normalize_gemini_base_url(base_url: Optional[str]) -> str:
    if not base_url:
        return ""
    return base_url.strip().rstrip("/")


def get_gemini_base_url(request: Request = None, header_value: Optional[str] = None) -> str:
    value = header_value or ""
    if not value and request is not None:
        value = request.headers.get(GEMINI_BASE_URL_HEADER, "")
    return normalize_gemini_base_url(value or os.environ.get("GEMINI_BASE_URL", ""))


def add_gemini_base_url_to_env(env: dict, base_url: str) -> None:
    normalized = normalize_gemini_base_url(base_url)
    if normalized:
        env["GEMINI_BASE_URL"] = normalized
    else:
        env.pop("GEMINI_BASE_URL", None)


def normalize_openai_compat_base_url(base_url: Optional[str]) -> str:
    return (base_url or "").strip().rstrip("/")


def resolve_openai_compat_config(
    header_key: Optional[str] = None,
    header_base_url: Optional[str] = None,
    header_model: Optional[str] = None,
    request_model: Optional[str] = None,
) -> Dict[str, str]:
    return {
        "api_key": (header_key or os.environ.get("OPENAI_COMPAT_API_KEY") or "").strip(),
        "base_url": normalize_openai_compat_base_url(header_base_url or os.environ.get("OPENAI_COMPAT_BASE_URL") or ""),
        "model": (header_model or request_model or os.environ.get("OPENAI_COMPAT_MODEL") or "").strip(),
    }


def resolve_gemini_access(
    request: Request,
    header_key: Optional[str] = None,
    header_base_url: Optional[str] = None,
    body: Optional[Dict] = None,
    form: Optional[Dict] = None,
):
    pool = parse_gemini_pool_config(headers=dict(request.headers), body=body, form=form)
    if pool.mode == "official_pool":
        session = pool.checkout()
        return session.api_key, "", pool
    api_key = header_key or (pool.keys[0] if pool.keys else os.environ.get("GEMINI_API_KEY"))
    base_url = get_gemini_base_url(request, header_base_url)
    return api_key, base_url, pool

# Constants
UPLOAD_DIR = os.environ.get("OPENSHORTS_UPLOAD_DIR", "uploads")
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
# Default to 1 if not set, but user can set higher for powerful servers
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
MAX_FILE_SIZE_MB = 2048  # 2GB limit
COMMENTARY_MAX_UPLOAD_MB = int(os.environ.get("COMMENTARY_MAX_UPLOAD_MB", "8192"))
JOB_RETENTION_SECONDS = 3600  # 1 hour retention

# Application State
job_queue = asyncio.Queue()
jobs: Dict[str, Dict] = {}
thumbnail_sessions: Dict[str, Dict] = {}
publish_jobs: Dict[str, Dict] = {}  # {publish_id: {status, result, error}}
# Semester to limit concurrency to MAX_CONCURRENT_JOBS
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

def _relocate_root_job_artifacts(job_id: str, job_output_dir: str) -> bool:
    """
    Backward-compat rescue:
    If main.py accidentally wrote metadata/clips into OUTPUT_DIR root (e.g. output/<jobid>_...),
    move them into output/<job_id>/ so the API can find and serve them.
    """
    try:
        os.makedirs(job_output_dir, exist_ok=True)
        root = OUTPUT_DIR
        pattern = os.path.join(root, f"{job_id}_*_metadata.json")
        meta_candidates = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
        if not meta_candidates:
            return False

        # Move the newest metadata and its associated clips.
        metadata_path = meta_candidates[0]
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")

        # Move metadata
        dest_metadata = os.path.join(job_output_dir, os.path.basename(metadata_path))
        if os.path.abspath(metadata_path) != os.path.abspath(dest_metadata):
            shutil.move(metadata_path, dest_metadata)

        # Move any clips that match the same base_name into the job folder
        clip_pattern = os.path.join(root, f"{base_name}_clip_*.mp4")
        for clip_path in glob.glob(clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        # Also move any temp_ clips that might remain
        temp_clip_pattern = os.path.join(root, f"temp_{base_name}_clip_*.mp4")
        for clip_path in glob.glob(temp_clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        return True
    except Exception:
        return False

async def cleanup_jobs():
    """Background task to remove old jobs and files."""
    import time
    print("🧹 Cleanup task started.")
    while True:
        try:
            await asyncio.sleep(300) # Check every 5 minutes
            now = time.time()
            
            # Simple directory cleanup based on modification time
            # Check OUTPUT_DIR
            for job_id in os.listdir(OUTPUT_DIR):
                job_path = os.path.join(OUTPUT_DIR, job_id)
                if os.path.isdir(job_path):
                    if now - os.path.getmtime(job_path) > JOB_RETENTION_SECONDS:
                        print(f"🧹 Purging old job: {job_id}")
                        shutil.rmtree(job_path, ignore_errors=True)
                        if job_id in jobs:
                            del jobs[job_id]

            # Cleanup SaaSShorts jobs from memory
            try:
                saas_expired = [
                    jid for jid, jdata in list(saas_jobs.items())
                    if jdata.get("status") in ("completed", "failed")
                    and jdata.get("output_dir")
                    and os.path.isdir(jdata["output_dir"])
                    and now - os.path.getmtime(jdata["output_dir"]) > JOB_RETENTION_SECONDS
                ]
                for jid in saas_expired:
                    del saas_jobs[jid]
            except NameError:
                pass

            # Cleanup Uploads
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if now - os.path.getmtime(file_path) > JOB_RETENTION_SECONDS:
                         os.remove(file_path)
                except Exception: pass

        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

async def process_queue():
    """Background worker to process jobs from the queue with concurrency limit."""
    print(f"🚀 Job Queue Worker started with {MAX_CONCURRENT_JOBS} concurrent slots.")
    while True:
        try:
            # Wait for a job
            job_id = await job_queue.get()
            
            # Acquire semaphore slot (waits if max jobs are running)
            await concurrency_semaphore.acquire()
            print(f"🔄 Acquired slot for job: {job_id}")

            # Process in background task to not block the loop (allowing other slots to fill)
            asyncio.create_task(run_job_wrapper(job_id))
            
        except Exception as e:
            print(f"❌ Queue dispatch error: {e}")
            await asyncio.sleep(1)

async def run_job_wrapper(job_id):
    """Wrapper to run job and release semaphore"""
    try:
        job = jobs.get(job_id)
        if job:
            await run_job(job_id, job)
    except Exception as e:
         print(f"❌ Job wrapper error {job_id}: {e}")
    finally:
        # Always release semaphore and mark queue task done
        concurrency_semaphore.release()
        job_queue.task_done()
        print(f"✅ Released slot for job: {job_id}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start worker and cleanup
    worker_task = asyncio.create_task(process_queue())
    cleanup_task = asyncio.create_task(cleanup_jobs())
    yield
    # Cleanup (optional: cancel worker)

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving videos
app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")

# Mount static files for serving thumbnails
THUMBNAILS_DIR = os.path.join(OUTPUT_DIR, "thumbnails")
os.makedirs(THUMBNAILS_DIR, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")

class ProcessRequest(BaseModel):
    url: str

def enqueue_output(out, job_id):
    """Reads output from a subprocess and appends it to jobs logs."""
    try:
        for line in iter(out.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                print(f"📝 [Job Output] {decoded_line}")
                if job_id in jobs:
                    jobs[job_id]['logs'].append(decoded_line)
    except Exception as e:
        print(f"Error reading output for job {job_id}: {e}")
    finally:
        out.close()

async def run_job(job_id, job_data):
    """Executes the subprocess for a specific job."""
    
    cmd = job_data['cmd']
    env = job_data['env']
    output_dir = job_data['output_dir']
    
    jobs[job_id]['status'] = 'processing'
    jobs[job_id]['logs'].append("Job started by worker.")
    print(f"🎬 [run_job] Executing command for {job_id}: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr to stdout
            env=env,
            cwd=os.getcwd()
        )
        
        # We need to capture logs in a thread because Popen isn't async
        t_log = threading.Thread(target=enqueue_output, args=(process.stdout, job_id))
        t_log.daemon = True
        t_log.start()
        
        # Async wait for process with incremental updates
        start_wait = time.time()
        while process.poll() is None:
            await asyncio.sleep(2)
            
            # Check for partial results every 2 seconds
            # Look for metadata file
            try:
                json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
                if json_files:
                    target_json = json_files[0]
                    # Read metadata (it might be being written to, so simple try/except or just read)
                    # Use a lock or just robust read? json.load might fail if file is partial.
                    # Usually main.py writes it once at start (based on my review).
                    if os.path.getsize(target_json) > 0:
                        with open(target_json, 'r') as f:
                            data = json.load(f)
                            
                        base_name = os.path.basename(target_json).replace('_metadata.json', '')
                        clips = data.get('shorts', [])
                        cost_analysis = data.get('cost_analysis')
                        
                        # Check which clips actually exist on disk
                        ready_clips = []
                        for i, clip in enumerate(clips):
                             clip_filename = f"{base_name}_clip_{i+1}.mp4"
                             clip_path = os.path.join(output_dir, clip_filename)
                             if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                                 # Checking if file is growing? For now assume if it exists and main.py moves it there, it's done.
                                 # main.py writes to temp_... then moves to final name. So presence means ready!
                                 clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                                 ready_clips.append(clip)
                        
                        if ready_clips:
                             jobs[job_id]['result'] = {'clips': ready_clips, 'cost_analysis': cost_analysis}
            except Exception as e:
                # Ignore read errors during processing
                pass

        returncode = process.returncode
        
        if returncode == 0:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['logs'].append("Process finished successfully.")
            
            # Start S3 upload in background (silent, non-blocking)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, upload_job_artifacts, output_dir, job_id)
            
            # Find result JSON
            json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if not json_files:
                # Backward-compat rescue if outputs were written to OUTPUT_DIR root
                if _relocate_root_job_artifacts(job_id, output_dir):
                    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if json_files:
                target_json = json_files[0] 
                with open(target_json, 'r') as f:
                    data = json.load(f)
                
                # Enhance result with video URLs
                base_name = os.path.basename(target_json).replace('_metadata.json', '')
                clips = data.get('shorts', [])
                cost_analysis = data.get('cost_analysis')

                for i, clip in enumerate(clips):
                     clip_filename = f"{base_name}_clip_{i+1}.mp4"
                     clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                
                jobs[job_id]['result'] = {'clips': clips, 'cost_analysis': cost_analysis}
            else:
                 jobs[job_id]['status'] = 'failed'
                 jobs[job_id]['logs'].append("No metadata file generated.")
        else:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['logs'].append(f"Process failed with exit code {returncode}")
            
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['logs'].append(f"Execution error: {str(e)}")

@app.post("/api/process")
async def process_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    # Handle JSON body manually for URL payload
    content_type = request.headers.get("content-type", "")
    body = None
    if "application/json" in content_type:
        body = await request.json()
        url = body.get("url")
    api_key, gemini_base_url, gemini_pool = resolve_gemini_access(request, body=body)
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")
    
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    
    # Prepare Command
    cmd = ["python", "-u", "main.py"] # -u for unbuffered
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key
    add_gemini_base_url_to_env(env, gemini_base_url)
    
    if url:
        cmd.extend(["-u", url])
    else:
        # Save uploaded file with size limit check
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        
        # Read file in chunks to check size
        size = 0
        limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        
        with open(input_path, "wb") as buffer:
            while content := await file.read(1024 * 1024): # Read 1MB chunks
                size += len(content)
                if size > limit_bytes:
                    os.remove(input_path)
                    shutil.rmtree(job_output_dir)
                    raise HTTPException(status_code=413, detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB")
                buffer.write(content)
                
        cmd.extend(["-i", input_path])

    cmd.extend(["-o", job_output_dir])

    # Enqueue Job
    jobs[job_id] = {
        'status': 'queued',
        'logs': [f"Job {job_id} queued."],
        'cmd': cmd,
        'env': env,
        'output_dir': job_output_dir,
        'gemini_events': gemini_pool.event_dicts() if gemini_pool else [],
    }
    
    await job_queue.put(job_id)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "status": job['status'],
        "logs": job['logs'],
        "result": job.get('result'),
        "gemini_events": job.get("gemini_events", []),
    }

from editor import VideoEditor
from subtitles import generate_srt, burn_subtitles, generate_srt_from_video
from hooks import add_hook_to_video
from translate import translate_video, get_supported_languages
from thumbnail import analyze_video_for_titles, refine_titles, generate_thumbnail, generate_youtube_description
from commentary import DEFAULT_ANALYSIS_MODE, generate_commentary_video, generate_edge_voiceover, resolve_commentary_block_concurrency, resolve_openai_sampling_options

commentary_jobs: Dict[str, Dict] = {}
COMMENTARY_TASK_FILE = "commentary_task.json"
COMMENTARY_PERSISTED_FIELDS = {
    "job_id",
    "status",
    "stage",
    "stage_label",
    "stage_progress",
    "logs",
    "created_at",
    "updated_at",
    "request",
    "source_type",
    "source_path",
    "source_filename",
    "source_value",
    "analysis_video_path",
    "analysis_video_filename",
    "gemini_file_uri",
    "gemini_file_name",
    "gemini_file_mime_type",
    "script_path",
    "metadata_path",
    "result",
    "error",
    "gemini_events",
}

COMMENTARY_STAGE_PATTERNS = [
    ("source", "准备源视频", "Preparing source video"),
    ("analysis_compress", "压缩 Gemini 分析视频", "Compressing Gemini analysis video"),
    ("analysis_compress", "复用 Gemini 分析视频", "Reusing compressed Gemini analysis video"),
    ("analysis_upload", "上传 Gemini 分析副本", "Uploading 360p Gemini analysis video"),
    ("analysis_upload", "复用 Gemini 分析副本", "Reusing processed Gemini analysis video"),
    ("analysis_upload", "等待 Gemini 文件处理", "Gemini Files API processing"),
    ("openai", "OpenAI 兼容多模态分析", "OpenAI-compatible multimodal visual analysis"),
    ("openai", "OpenAI 兼容模型写解说", "OpenAI-compatible model is writing"),
    ("openai", "OpenAI 兼容模型写解说", "OpenAI-compatible model returned"),
    ("gemini", "Gemini 分析并写解说", "Gemini is analyzing"),
    ("gemini", "校验解说时间线", "validating timeline sync"),
    ("voice", "生成语音并同步画面", "Generating synced commentary block"),
    ("voice", "生成解说语音", "Generating commentary voiceover"),
    ("render", "合成最终视频", "Mixing new voiceover"),
    ("render", "生成字幕", "Generating text-timed subtitles"),
    ("done", "生成完成", "Commentary remix video completed"),
]


def commentary_job_dir(job_id: str) -> str:
    return os.path.abspath(os.path.join(OUTPUT_DIR, job_id))


def commentary_task_path(job_id: str) -> str:
    return os.path.join(commentary_job_dir(job_id), COMMENTARY_TASK_FILE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def commentary_request_to_dict(req) -> Dict:
    data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    data.pop("gemini_pool", None)
    return {key: value for key, value in data.items() if value is not None}


def persistable_commentary_task(job: Dict) -> Dict:
    return {key: job.get(key) for key in COMMENTARY_PERSISTED_FIELDS if key in job}


def save_commentary_task(job_id: str) -> None:
    job = commentary_jobs.get(job_id)
    if not job:
        return
    job["updated_at"] = now_iso()
    os.makedirs(commentary_job_dir(job_id), exist_ok=True)
    tmp_path = f"{commentary_task_path(job_id)}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(persistable_commentary_task(job), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, commentary_task_path(job_id))


def load_commentary_task(job_id: str) -> Optional[Dict]:
    task_path = commentary_task_path(job_id)
    if not os.path.exists(task_path):
        return None
    try:
        with open(task_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("job_id", job_id)
    data.setdefault("logs", [])
    commentary_jobs[job_id] = data
    return data


def list_commentary_tasks(limit: int = 30) -> List[Dict]:
    tasks = []
    for task_path in glob.glob(os.path.join(os.path.abspath(OUTPUT_DIR), "*", COMMENTARY_TASK_FILE)):
        try:
            with open(task_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if isinstance(data, dict):
            tasks.append(data)
    tasks.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or "", reverse=True)
    return tasks[:limit]


def update_commentary_stage(job_id: str, message: str) -> None:
    job = commentary_jobs.get(job_id)
    if not job:
        return
    for stage, label, marker in COMMENTARY_STAGE_PATTERNS:
        if marker in message:
            job["stage"] = stage
            job["stage_label"] = label
            break
    match = re.search(r"(\d+)%", message)
    if match:
        job["stage_progress"] = int(match.group(1))
    elif job.get("stage") not in {"analysis_compress", "upload"}:
        job["stage_progress"] = None

class CommentaryVoicePreviewRequest(BaseModel):
    language: str = "zh"
    edge_voice: str
    text: Optional[str] = None

COMMENTARY_VOICE_PREVIEW_TEXT = {
    "zh": "你好，这是当前选择的中文解说声音试听。",
    "en": "Hello, this is a preview of the selected English narration voice.",
    "es": "Hola, esta es una prueba de la voz de narración seleccionada.",
    "ja": "こんにちは。これは選択した日本語ナレーション音声のプレビューです。",
}

class CommentaryRequest(BaseModel):
    url: str
    language: str = "zh"
    style: str = "documentary"
    target_duration: str = "medium"
    analysis_mode: str = DEFAULT_ANALYSIS_MODE
    gemini_model: Optional[str] = None
    openai_model: Optional[str] = None
    openai_frame_interval_seconds: Optional[float] = None
    openai_max_frames: Optional[int] = None
    openai_scene_max_keyframes: Optional[int] = None
    openai_batch_size: Optional[int] = None
    openai_visual_concurrency: Optional[int] = None
    commentary_block_concurrency: Optional[int] = None
    auto_video_speed: bool = True
    tts_provider: str = "edge"
    voice_id: Optional[str] = None
    edge_voice: Optional[str] = None
    original_audio_volume: float = 0.3
    pause_original_audio_volume: float = 0.6
    subtitles: bool = True
    vertical: bool = False
    aspect_mode: str = "auto"
    source_language: Optional[str] = None
    gemini_pool: Optional[str] = None

def parse_form_bool(value, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def parse_form_float(value, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_form_optional_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_form_optional_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def apply_openai_sampling_options_to_request(req: CommentaryRequest) -> Dict:
    options = resolve_openai_sampling_options(
        frame_interval_seconds=req.openai_frame_interval_seconds,
        max_frames=req.openai_max_frames,
        scene_max_keyframes=req.openai_scene_max_keyframes,
        batch_size=req.openai_batch_size,
        visual_concurrency=req.openai_visual_concurrency,
    )
    req.openai_frame_interval_seconds = options["frame_interval_seconds"]
    req.openai_max_frames = options["max_frames"]
    req.openai_scene_max_keyframes = options["scene_max_keyframes"]
    req.openai_batch_size = options["batch_size"]
    req.openai_visual_concurrency = options["visual_concurrency"]
    return options

def commentary_request_from_form(form) -> CommentaryRequest:
    return CommentaryRequest(
        url=str(form.get("url") or ""),
        language=str(form.get("language") or "zh"),
        style=str(form.get("style") or "documentary"),
        target_duration=str(form.get("target_duration") or "medium"),
        analysis_mode=str(form.get("analysis_mode") or DEFAULT_ANALYSIS_MODE),
        gemini_model=str(form.get("gemini_model") or "") or None,
        openai_model=str(form.get("openai_model") or "") or None,
        openai_frame_interval_seconds=parse_form_optional_float(form.get("openai_frame_interval_seconds")),
        openai_max_frames=parse_form_optional_int(form.get("openai_max_frames")),
        openai_scene_max_keyframes=parse_form_optional_int(form.get("openai_scene_max_keyframes")),
        openai_batch_size=parse_form_optional_int(form.get("openai_batch_size")),
        openai_visual_concurrency=parse_form_optional_int(form.get("openai_visual_concurrency")),
        commentary_block_concurrency=parse_form_optional_int(form.get("commentary_block_concurrency")),
        auto_video_speed=parse_form_bool(form.get("auto_video_speed"), True),
        tts_provider=str(form.get("tts_provider") or "edge"),
        voice_id=str(form.get("voice_id") or "") or None,
        edge_voice=str(form.get("edge_voice") or "") or None,
        original_audio_volume=parse_form_float(form.get("original_audio_volume"), 0.3),
        pause_original_audio_volume=parse_form_float(form.get("pause_original_audio_volume"), 0.6),
        subtitles=parse_form_bool(form.get("subtitles"), True),
        vertical=parse_form_bool(form.get("vertical"), False),
        aspect_mode=str(form.get("aspect_mode") or "auto"),
        source_language=str(form.get("source_language") or "") or None,
        gemini_pool=str(form.get("gemini_pool") or "") or None,
    )

async def save_commentary_upload(file: UploadFile, job_id: str, progress=None) -> str:
    safe_filename = os.path.basename(file.filename or "uploaded_video.mp4") or "uploaded_video.mp4"
    upload_dir = commentary_job_dir(job_id)
    os.makedirs(upload_dir, exist_ok=True)
    input_path = os.path.abspath(os.path.join(upload_dir, f"{job_id}_source_{safe_filename}"))
    size = 0
    limit_bytes = COMMENTARY_MAX_UPLOAD_MB * 1024 * 1024
    total = int(file.headers.get("content-length") or 0) if getattr(file, "headers", None) else 0
    last_logged_percent = -1
    if progress:
        progress("upload", "Saving uploaded video on server...", 0)
    with open(input_path, "wb") as buffer:
        while content := await file.read(1024 * 1024):
            size += len(content)
            if progress and total > 0:
                percent = int(min(99, (size / total) * 100))
                if percent >= last_logged_percent + 10:
                    last_logged_percent = percent
                    progress("upload", f"Saving uploaded video on server: {percent}%", percent)
            if size > limit_bytes:
                buffer.close()
                if os.path.exists(input_path):
                    os.remove(input_path)
                raise HTTPException(status_code=413, detail=f"File too large. Max size {COMMENTARY_MAX_UPLOAD_MB}MB")
            buffer.write(content)
    if progress:
        progress("upload", "Uploaded video saved on server: 100%", 100)
    return input_path

class YouTubeCookiesRequest(BaseModel):
    cookies: str

class YouTubeCookiesVerifyRequest(BaseModel):
    url: Optional[str] = None

def normalize_youtube_cookies(cookies: str) -> str:
    split_field_rows = normalize_split_youtube_cookie_fields(cookies)
    if split_field_rows:
        cookies = split_field_rows

    normalized_lines = []
    allowed_domains = ("youtube.com", ".youtube.com", "google.com", ".google.com")
    for raw_line in cookies.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if not normalized_lines:
                normalized_lines.append(line)
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            parts = line.split(None, 6)
        if len(parts) >= 7:
            domain = parts[0].lower()
            if any(domain == allowed or domain.endswith(allowed) for allowed in allowed_domains):
                normalized_lines.append("\t".join(parts[:7]))
        elif "youtube" in line.lower() or "google" in line.lower():
            normalized_lines.append(line)
    return "\n".join(normalized_lines).strip()


def normalize_split_youtube_cookie_fields(cookies: str) -> str:
    fields = [
        line.strip()
        for line in cookies.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(fields) < 7 or len(fields) % 7 != 0:
        return ""
    rows = []
    for index in range(0, len(fields), 7):
        chunk = fields[index:index + 7]
        domain, include_subdomains, path, secure, expiry, name, value = chunk
        if not (domain.startswith(".") or "." in domain):
            return ""
        if include_subdomains.upper() not in {"TRUE", "FALSE"}:
            return ""
        if secure.upper() not in {"TRUE", "FALSE"}:
            return ""
        if not expiry.isdigit():
            return ""
        if not ("youtube" in domain.lower() or "google" in domain.lower()):
            return ""
        rows.append("\t".join([domain, include_subdomains.upper(), path, secure.upper(), expiry, name, value]))
    return "\n".join(rows)

def inspect_youtube_cookies(cookies: str) -> Dict:
    names = set()
    rows = 0
    domains = set()
    for line in normalize_youtube_cookies(cookies).splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            rows += 1
            domains.add(parts[0])
            names.add(parts[5])
    login_names = {"SID", "HSID", "SSID", "APISID", "SAPISID", "__Secure-1PSID", "__Secure-3PSID", "LOGIN_INFO"}
    has_login_cookies = bool(names & login_names)
    strong_login_names = {"SID", "SAPISID", "LOGIN_INFO"}
    found_strong_login_cookies = sorted(names & strong_login_names)
    missing_strong_login_cookies = sorted(strong_login_names - names)
    has_strong_login_cookies = strong_login_names.issubset(names)
    youtube_domains = [domain for domain in domains if "youtube" in domain or "google" in domain]
    return {
        "rows": rows,
        "domains": sorted(youtube_domains),
        "has_login_cookies": has_login_cookies,
        "has_strong_login_cookies": has_strong_login_cookies,
        "found_strong_login_cookies": found_strong_login_cookies,
        "missing_strong_login_cookies": missing_strong_login_cookies,
        "missing_warning": not has_strong_login_cookies,
    }


def read_cookie_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def write_cookie_file(path: str, cookies: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(cookies.rstrip() + "\n")


def restore_complete_cookies_from_backup() -> bool:
    if not os.path.exists(PROJECT_COOKIES_BACKUP_PATH):
        return False
    backup_cookies = normalize_youtube_cookies(read_cookie_file(PROJECT_COOKIES_BACKUP_PATH))
    backup_info = inspect_youtube_cookies(backup_cookies)
    if backup_info.get("missing_warning"):
        return False
    current_info = {"missing_warning": True}
    if os.path.exists(PROJECT_COOKIES_PATH):
        current_info = inspect_youtube_cookies(read_cookie_file(PROJECT_COOKIES_PATH))
    if current_info.get("missing_warning"):
        write_cookie_file(PROJECT_COOKIES_PATH, backup_cookies)
        os.environ["YOUTUBE_COOKIES_PATH"] = PROJECT_COOKIES_PATH
        return True
    return False


def verify_youtube_cookies_for_url(url: str, cookies_path: str) -> Dict:
    if not cookies_path or not os.path.exists(cookies_path):
        return {"ok": False, "message": "YouTube cookies file is not configured."}

    verify_url = (url or "https://www.youtube.com/watch?v=dQw4w9WgXcQ").strip()
    runtime_cookies_path = cookies_path
    runtime_dir = None
    try:
        runtime_dir = tempfile.mkdtemp(prefix="openshorts_ytdlp_verify_")
        runtime_cookies_path = os.path.join(runtime_dir, "youtube_cookies.runtime.txt")
        shutil.copyfile(cookies_path, runtime_cookies_path)
    except Exception:
        runtime_cookies_path = cookies_path

    js_runtimes = {}
    for runtime_name in ("node", "bun", "deno"):
        if shutil.which(runtime_name):
            js_runtimes[runtime_name] = {}
            break
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "cookiefile": runtime_cookies_path,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "nocheckcertificate": True,
        "cachedir": False,
        "js_runtimes": js_runtimes,
        "remote_components": ["ejs:github"],
        "extractor_args": {"youtube": {"player_client": ["web"]}},
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(verify_url, download=False)
        formats = info.get("formats") or []
        video_formats = [fmt for fmt in formats if fmt.get("vcodec") and fmt.get("vcodec") != "none"]
        if not video_formats:
            return {
                "ok": False,
                "message": "Cookies reached YouTube, but no downloadable video formats were returned.",
                "title": info.get("title"),
                "duration": info.get("duration"),
            }
        return {
            "ok": True,
            "message": "YouTube cookies are valid for yt-dlp video extraction.",
            "title": info.get("title"),
            "duration": info.get("duration"),
            "format_count": len(video_formats),
        }
    except Exception as exc:
        error_text = str(exc)
        if "Sign in to confirm" in error_text or "not a bot" in error_text or "cookies" in error_text.lower():
            message = "YouTube cookies appear invalid or expired; YouTube still asks for login/bot verification."
        elif "Requested format is not available" in error_text or "no downloadable video formats" in error_text:
            message = "YouTube responded, but yt-dlp could not find a downloadable video format."
        else:
            message = "YouTube cookies verification failed."
        return {"ok": False, "message": message, "error": error_text[:800]}


@app.get("/api/settings/youtube-cookies")
async def get_youtube_cookies_status():
    exists = os.path.exists(PROJECT_COOKIES_PATH)
    if not exists:
        restored = restore_complete_cookies_from_backup()
        if not restored:
            return {"configured": False, "size": 0}
    else:
        restore_complete_cookies_from_backup()
    with open(PROJECT_COOKIES_PATH, "r", encoding="utf-8", errors="ignore") as f:
        info = inspect_youtube_cookies(f.read())
    return {
        "configured": True,
        "size": os.path.getsize(PROJECT_COOKIES_PATH),
        **info,
    }

@app.post("/api/settings/youtube-cookies")
async def save_youtube_cookies(req: YouTubeCookiesRequest):
    cookies = normalize_youtube_cookies(req.cookies or "")
    if not cookies:
        raise HTTPException(status_code=400, detail="Missing YouTube cookies")
    if "youtube.com" not in cookies and "youtube" not in cookies.lower():
        raise HTTPException(status_code=400, detail="Invalid YouTube cookies format")
    info = inspect_youtube_cookies(cookies)
    if info["rows"] == 0:
        raise HTTPException(status_code=400, detail="Invalid Netscape cookies format")
    existing_info = None
    if os.path.exists(PROJECT_COOKIES_PATH):
        with open(PROJECT_COOKIES_PATH, "r", encoding="utf-8", errors="ignore") as f:
            existing_info = inspect_youtube_cookies(f.read())
    if info.get("missing_warning"):
        if existing_info and not existing_info.get("missing_warning"):
            raise HTTPException(status_code=400, detail="拒绝覆盖当前完整 cookies：你这次上传/粘贴的 cookies 缺少 SID / SAPISID / LOGIN_INFO 等关键字段，仍保留原来的完整 cookies。")
        raise HTTPException(status_code=400, detail="当前 cookies 不是完整登录 cookies：缺少 SID / SAPISID / LOGIN_INFO 等关键字段。请用隐身窗口登录 YouTube 后重新导出 cookies，或粘贴包含这些字段的完整 cookies。")
    write_cookie_file(PROJECT_COOKIES_PATH, cookies)
    write_cookie_file(PROJECT_COOKIES_BACKUP_PATH, cookies)
    os.environ["YOUTUBE_COOKIES_PATH"] = PROJECT_COOKIES_PATH
    return {"configured": True, "size": os.path.getsize(PROJECT_COOKIES_PATH), **info}

@app.post("/api/settings/youtube-cookies/verify")
async def verify_youtube_cookies(req: YouTubeCookiesVerifyRequest):
    restore_complete_cookies_from_backup()
    if not os.path.exists(PROJECT_COOKIES_PATH):
        raise HTTPException(status_code=400, detail="YouTube cookies are not configured")
    result = verify_youtube_cookies_for_url(req.url or "https://www.youtube.com/watch?v=dQw4w9WgXcQ", PROJECT_COOKIES_PATH)
    status_code = 200 if result.get("ok") else 400
    if not result.get("ok"):
        raise HTTPException(status_code=status_code, detail=result)
    return result

@app.post("/api/commentary/voice-preview")
async def commentary_voice_preview(req: CommentaryVoicePreviewRequest):
    voice = (req.edge_voice or "").strip()
    if not voice:
        raise HTTPException(status_code=400, detail="Missing Edge voice")
    text = (req.text or COMMENTARY_VOICE_PREVIEW_TEXT.get(req.language, COMMENTARY_VOICE_PREVIEW_TEXT["zh"])).strip()
    if not text:
        raise HTTPException(status_code=400, detail="Missing preview text")

    preview_id = str(uuid.uuid4())
    preview_dir = os.path.abspath(os.path.join(OUTPUT_DIR, "voice_previews"))
    os.makedirs(preview_dir, exist_ok=True)
    preview_path = os.path.join(preview_dir, f"{preview_id}.mp3")

    try:
        await asyncio.to_thread(generate_edge_voiceover, text[:180], preview_path, voice)
        return FileResponse(preview_path, media_type="audio/mpeg", filename="voice-preview.mp3")
    except Exception as e:
        if os.path.exists(preview_path):
            os.remove(preview_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/commentary/generate")
async def commentary_generate(
    request: Request,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key"),
    x_gemini_base_url: Optional[str] = Header(None, alias=GEMINI_BASE_URL_HEADER),
    x_openai_compatible_key: Optional[str] = Header(None, alias=OPENAI_COMPAT_KEY_HEADER),
    x_openai_compatible_base_url: Optional[str] = Header(None, alias=OPENAI_COMPAT_BASE_URL_HEADER),
    x_openai_compatible_model: Optional[str] = Header(None, alias=OPENAI_COMPAT_MODEL_HEADER),
):
    gemini_key = x_gemini_key or os.environ.get("GEMINI_API_KEY")
    elevenlabs_key = x_elevenlabs_key or os.environ.get("ELEVENLABS_API_KEY")
    gemini_base_url = get_gemini_base_url(header_value=x_gemini_base_url)
    upload_file = None

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload_file = form.get("file")
        req = commentary_request_from_form(form)
        gemini_pool = parse_gemini_pool_config(headers=dict(request.headers), form=form)
    else:
        body = await request.json()
        req = CommentaryRequest(**body)
        gemini_pool = parse_gemini_pool_config(headers=dict(request.headers), body=body)

    has_upload = bool(upload_file and getattr(upload_file, "filename", None))
    if not req.url.strip() and not has_upload:
        raise HTTPException(status_code=400, detail="Missing YouTube URL or uploaded video")
    effective_gemini_pool = gemini_pool if gemini_pool.keys else None
    analysis_mode = (req.analysis_mode or DEFAULT_ANALYSIS_MODE).strip().lower()
    if analysis_mode not in {"current", "video", "openai"}:
        raise HTTPException(status_code=400, detail="Unsupported commentary analysis mode")
    openai_config = resolve_openai_compat_config(
        header_key=x_openai_compatible_key,
        header_base_url=x_openai_compatible_base_url,
        header_model=x_openai_compatible_model,
        request_model=req.openai_model,
    )
    if analysis_mode == "openai":
        if not openai_config["api_key"]:
            raise HTTPException(status_code=400, detail="Missing OpenAI-compatible API Key")
        if not openai_config["base_url"]:
            raise HTTPException(status_code=400, detail="Missing OpenAI-compatible Base URL")
        if not openai_config["model"]:
            raise HTTPException(status_code=400, detail="Missing OpenAI-compatible model")
        req.openai_model = openai_config["model"]
        apply_openai_sampling_options_to_request(req)
    elif not gemini_key and not gemini_pool.keys:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key")
    tts_provider = (req.tts_provider or "edge").lower()
    if tts_provider not in {"edge", "elevenlabs"}:
        raise HTTPException(status_code=400, detail="Unsupported TTS provider")
    if tts_provider == "elevenlabs" and not elevenlabs_key:
        raise HTTPException(status_code=400, detail="Missing ElevenLabs API Key")
    req.commentary_block_concurrency = resolve_commentary_block_concurrency(req.commentary_block_concurrency)

    job_id = str(uuid.uuid4())
    job_output_dir = commentary_job_dir(job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    created_at = now_iso()
    commentary_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "stage": "upload" if has_upload else "queued",
        "stage_label": "上传原视频" if has_upload else "等待开始",
        "stage_progress": 0 if has_upload else None,
        "logs": ["Queued commentary remix job..."],
        "created_at": created_at,
        "updated_at": created_at,
        "request": commentary_request_to_dict(req),
        "source_type": "file" if has_upload else "url",
        "source_value": req.url.strip(),
        "source_path": None,
        "source_filename": getattr(upload_file, "filename", None) if has_upload else None,
        "analysis_video_path": None,
        "analysis_video_filename": None,
        "result": None,
        "error": None,
        "gemini_events": [],
    }
    save_commentary_task(job_id)

    def log(message: str):
        commentary_jobs[job_id]["logs"].append(message)
        update_commentary_stage(job_id, message)
        save_commentary_task(job_id)

    def upload_log(stage: str, message: str, percent=None):
        commentary_jobs[job_id]["stage"] = stage
        commentary_jobs[job_id]["stage_label"] = "上传原视频"
        commentary_jobs[job_id]["stage_progress"] = percent
        commentary_jobs[job_id]["logs"].append(message)
        save_commentary_task(job_id)

    def checkpoint(fields: Dict):
        commentary_jobs[job_id].update(fields)
        save_commentary_task(job_id)

    input_path = None
    source_type = "url"
    source_value = req.url.strip()
    try:
        if has_upload:
            input_path = await save_commentary_upload(upload_file, job_id, progress=upload_log)
            source_type = "file"
            source_value = input_path
            checkpoint({
                "source_value": source_value,
                "source_path": source_value,
                "source_filename": os.path.basename(source_value),
            })
    except HTTPException as e:
        commentary_jobs[job_id]["logs"].append(f"Error: {e.detail}")
        commentary_jobs[job_id]["stage"] = "failed"
        commentary_jobs[job_id]["stage_label"] = "生成失败"
        commentary_jobs[job_id]["stage_progress"] = None
        commentary_jobs[job_id]["status"] = "failed"
        commentary_jobs[job_id]["error"] = str(e.detail)
        save_commentary_task(job_id)
        raise
    except Exception as e:
        error_text = f"Failed to save uploaded video: {e}"
        commentary_jobs[job_id]["logs"].append(f"Error: {error_text}")
        commentary_jobs[job_id]["stage"] = "failed"
        commentary_jobs[job_id]["stage_label"] = "生成失败"
        commentary_jobs[job_id]["stage_progress"] = None
        commentary_jobs[job_id]["status"] = "failed"
        commentary_jobs[job_id]["error"] = error_text
        save_commentary_task(job_id)
        raise HTTPException(status_code=500, detail=error_text)
    finally:
        if has_upload:
            await upload_file.close()

    def run_job():
        try:
            result = generate_commentary_video(
                source=source_value,
                source_type=source_type,
                output_dir=job_output_dir,
                gemini_key=gemini_key or "",
                gemini_pool=effective_gemini_pool,
                elevenlabs_key=elevenlabs_key,
                tts_provider=tts_provider,
                voice_id=req.voice_id or "21m00Tcm4TlvDq8ikWAM",
                edge_voice=req.edge_voice,
                language=req.language,
                style=req.style,
                target_duration=req.target_duration,
                original_audio_volume=req.original_audio_volume,
                pause_original_audio_volume=req.pause_original_audio_volume,
                subtitles=req.subtitles,
                vertical=req.vertical,
                aspect_mode=req.aspect_mode,
                source_language=req.source_language,
                gemini_base_url=gemini_base_url,
                analysis_mode=analysis_mode,
                gemini_model=req.gemini_model,
                openai_key=openai_config["api_key"],
                openai_base_url=openai_config["base_url"],
                openai_model=openai_config["model"],
                openai_frame_interval_seconds=req.openai_frame_interval_seconds,
                openai_max_frames=req.openai_max_frames,
                openai_scene_max_keyframes=req.openai_scene_max_keyframes,
                openai_batch_size=req.openai_batch_size,
                openai_visual_concurrency=req.openai_visual_concurrency,
                commentary_block_concurrency=req.commentary_block_concurrency,
                auto_video_speed=req.auto_video_speed,
                progress=log,
                checkpoint=checkpoint,
            )
            commentary_jobs[job_id]["result"] = result
            commentary_jobs[job_id]["gemini_events"] = result.get("gemini_events", [])
            commentary_jobs[job_id]["stage"] = "done"
            commentary_jobs[job_id]["stage_label"] = "生成完成"
            commentary_jobs[job_id]["stage_progress"] = 100
            commentary_jobs[job_id]["status"] = "completed"
            commentary_jobs[job_id]["error"] = None
            save_commentary_task(job_id)
        except Exception as e:
            error_text = str(e)
            commentary_jobs[job_id]["logs"].append(f"Error: {error_text}")
            commentary_jobs[job_id]["gemini_events"] = effective_gemini_pool.event_dicts() if effective_gemini_pool else []
            commentary_jobs[job_id]["stage"] = "failed"
            commentary_jobs[job_id]["stage_label"] = "生成失败"
            commentary_jobs[job_id]["stage_progress"] = None
            commentary_jobs[job_id]["status"] = "failed"
            commentary_jobs[job_id]["error"] = error_text
            save_commentary_task(job_id)

    threading.Thread(target=run_job, daemon=True).start()
    return {"job_id": job_id, "status": "processing"}

@app.get("/api/commentary/jobs")
async def commentary_job_list():
    return {"jobs": list_commentary_tasks()}


@app.post("/api/commentary/jobs/{job_id}/retry")
async def commentary_retry(
    job_id: str,
    request: Request,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key"),
    x_gemini_base_url: Optional[str] = Header(None, alias=GEMINI_BASE_URL_HEADER),
    x_openai_compatible_key: Optional[str] = Header(None, alias=OPENAI_COMPAT_KEY_HEADER),
    x_openai_compatible_base_url: Optional[str] = Header(None, alias=OPENAI_COMPAT_BASE_URL_HEADER),
    x_openai_compatible_model: Optional[str] = Header(None, alias=OPENAI_COMPAT_MODEL_HEADER),
):
    job = commentary_jobs.get(job_id) or load_commentary_task(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Commentary job not found")
    if job.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Commentary job is already processing")

    body = {}
    if "application/json" in request.headers.get("content-type", ""):
        try:
            parsed_body = await request.json()
            body = parsed_body if isinstance(parsed_body, dict) else {}
        except Exception:
            body = {}
    gemini_key = x_gemini_key or os.environ.get("GEMINI_API_KEY")
    elevenlabs_key = x_elevenlabs_key or os.environ.get("ELEVENLABS_API_KEY")
    gemini_base_url = get_gemini_base_url(header_value=x_gemini_base_url)
    gemini_pool = parse_gemini_pool_config(headers=dict(request.headers), body=body)

    request_data = job.get("request") or {}
    if not request_data:
        request_data = {"url": job.get("source_value") or ""}
    req = CommentaryRequest(**request_data)
    effective_gemini_pool = gemini_pool if gemini_pool.keys else None
    analysis_mode = (req.analysis_mode or DEFAULT_ANALYSIS_MODE).strip().lower()
    if analysis_mode not in {"current", "video", "openai"}:
        raise HTTPException(status_code=400, detail="Unsupported commentary analysis mode")
    openai_config = resolve_openai_compat_config(
        header_key=x_openai_compatible_key,
        header_base_url=x_openai_compatible_base_url,
        header_model=x_openai_compatible_model,
        request_model=req.openai_model,
    )
    has_cached_script = bool(job.get("script_path") and os.path.exists(job.get("script_path")))
    if analysis_mode == "openai":
        if not has_cached_script:
            if not openai_config["api_key"]:
                raise HTTPException(status_code=400, detail="Missing OpenAI-compatible API Key")
            if not openai_config["base_url"]:
                raise HTTPException(status_code=400, detail="Missing OpenAI-compatible Base URL")
            if not openai_config["model"]:
                raise HTTPException(status_code=400, detail="Missing OpenAI-compatible model")
        if openai_config["model"]:
            req.openai_model = openai_config["model"]
        apply_openai_sampling_options_to_request(req)
        job["request"] = commentary_request_to_dict(req)
    elif not gemini_key and not gemini_pool.keys:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key")
    tts_provider = (req.tts_provider or "edge").lower()
    if tts_provider == "elevenlabs" and not elevenlabs_key:
        raise HTTPException(status_code=400, detail="Missing ElevenLabs API Key")
    req.commentary_block_concurrency = resolve_commentary_block_concurrency(req.commentary_block_concurrency)
    job["request"] = commentary_request_to_dict(req)

    source_path = job.get("source_path")
    if source_path and not os.path.exists(source_path):
        source_path = None
    if source_path:
        source_type = "file"
        source_value = source_path
    elif job.get("source_type") == "url" and (job.get("source_value") or req.url):
        source_type = "url"
        source_value = job.get("source_value") or req.url
    else:
        raise HTTPException(status_code=400, detail="Missing reusable source video for retry")

    job_output_dir = commentary_job_dir(job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    previous_error = job.get("error") or ""
    job.update({
        "status": "processing",
        "stage": "queued",
        "stage_label": "准备重试",
        "stage_progress": None,
        "result": None,
        "error": None,
    })
    job.setdefault("logs", []).append("Retrying commentary remix from saved task checkpoints...")
    save_commentary_task(job_id)

    def log(message: str):
        commentary_jobs[job_id]["logs"].append(message)
        update_commentary_stage(job_id, message)
        save_commentary_task(job_id)

    def checkpoint(fields: Dict):
        commentary_jobs[job_id].update(fields)
        save_commentary_task(job_id)

    gemini_file = None
    if job.get("gemini_file_uri"):
        gemini_file = {
            "uri": job.get("gemini_file_uri"),
            "name": job.get("gemini_file_name"),
            "mime_type": job.get("gemini_file_mime_type") or "video/mp4",
        }

    prepared_analysis_video_path = job.get("analysis_video_path")
    if prepared_analysis_video_path and not os.path.exists(prepared_analysis_video_path):
        prepared_analysis_video_path = None

    def run_retry_job():
        try:
            result = generate_commentary_video(
                source=source_value,
                source_type=source_type,
                output_dir=job_output_dir,
                gemini_key=gemini_key or "",
                gemini_pool=effective_gemini_pool,
                elevenlabs_key=elevenlabs_key,
                tts_provider=tts_provider,
                voice_id=req.voice_id or "21m00Tcm4TlvDq8ikWAM",
                edge_voice=req.edge_voice,
                language=req.language,
                style=req.style,
                target_duration=req.target_duration,
                original_audio_volume=req.original_audio_volume,
                pause_original_audio_volume=req.pause_original_audio_volume,
                subtitles=req.subtitles,
                vertical=req.vertical,
                aspect_mode=req.aspect_mode,
                source_language=req.source_language,
                gemini_base_url=gemini_base_url,
                analysis_mode=analysis_mode,
                gemini_model=req.gemini_model,
                openai_key=openai_config["api_key"],
                openai_base_url=openai_config["base_url"],
                openai_model=openai_config["model"],
                openai_frame_interval_seconds=req.openai_frame_interval_seconds,
                openai_max_frames=req.openai_max_frames,
                openai_scene_max_keyframes=req.openai_scene_max_keyframes,
                openai_batch_size=req.openai_batch_size,
                openai_visual_concurrency=req.openai_visual_concurrency,
                commentary_block_concurrency=req.commentary_block_concurrency,
                auto_video_speed=req.auto_video_speed,
                progress=log,
                checkpoint=checkpoint,
                prepared_analysis_video_path=prepared_analysis_video_path,
                gemini_file=gemini_file,
                previous_error=previous_error,
            )
            commentary_jobs[job_id]["result"] = result
            commentary_jobs[job_id]["gemini_events"] = result.get("gemini_events", [])
            commentary_jobs[job_id]["stage"] = "done"
            commentary_jobs[job_id]["stage_label"] = "生成完成"
            commentary_jobs[job_id]["stage_progress"] = 100
            commentary_jobs[job_id]["status"] = "completed"
            commentary_jobs[job_id]["error"] = None
            save_commentary_task(job_id)
        except Exception as e:
            error_text = str(e)
            commentary_jobs[job_id]["logs"].append(f"Error: {error_text}")
            commentary_jobs[job_id]["gemini_events"] = effective_gemini_pool.event_dicts() if effective_gemini_pool else []
            commentary_jobs[job_id]["stage"] = "failed"
            commentary_jobs[job_id]["stage_label"] = "生成失败"
            commentary_jobs[job_id]["stage_progress"] = None
            commentary_jobs[job_id]["status"] = "failed"
            commentary_jobs[job_id]["error"] = error_text
            save_commentary_task(job_id)

    threading.Thread(target=run_retry_job, daemon=True).start()
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/commentary/status/{job_id}")
async def commentary_status(job_id: str):
    job = commentary_jobs.get(job_id) or load_commentary_task(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Commentary job not found")
    return job

class EditRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: Optional[str] = None
    input_filename: Optional[str] = None

@app.post("/api/edit")
async def edit_clip(
    request: Request,
    req: EditRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    x_gemini_base_url: Optional[str] = Header(None, alias=GEMINI_BASE_URL_HEADER)
):
    # Determine API Key
    final_api_key, gemini_base_url, _gemini_pool = resolve_gemini_access(
        request,
        header_key=req.api_key or x_gemini_key,
        header_base_url=x_gemini_base_url,
        body=req.dict(),
    )
    
    if not final_api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key (Header or Body)")

    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")
        
    try:
        # Resolve Input Path: Prefer explict input_filename from frontend (chaining edits)
        if req.input_filename:
            # Security: Ensure just a filename, no paths
            safe_name = os.path.basename(req.input_filename)
            input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_name)
            filename = safe_name
        else:
            # Fallback to original clip
            clip = job['result']['clips'][req.clip_index]
            filename = clip['video_url'].split('/')[-1]
            input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
        
        if not os.path.exists(input_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

        # Define output path for edited video
        edited_filename = f"edited_{filename}"
        output_path = os.path.join(OUTPUT_DIR, req.job_id, edited_filename)
        
        # Run editing in a thread to avoid blocking main loop
        # Since VideoEditor uses blocking calls (subprocess, API wait)
        def run_edit():
            editor = VideoEditor(api_key=final_api_key, base_url=gemini_base_url)
            
            # SAFE FILE RENAMING STRATEGY (Avoid UnicodeEncodeError in Docker)
            # Create a safe ASCII filename in the same directory
            safe_filename = f"temp_input_{req.job_id}.mp4"
            safe_input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_filename)
            
            # Copy original file to safe path
            # (Copy is safer than rename if something crashes, we keep original)
            shutil.copy(input_path, safe_input_path)
            
            try:
                # 1. Upload (using safe path)
                vid_file = editor.upload_video(safe_input_path)
                
                # 2. Get duration
                import cv2
                cap = cv2.VideoCapture(safe_input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps else 0
                cap.release()
                
                # Load transcript from metadata
                transcript = None
                try:
                    meta_files = glob.glob(os.path.join(OUTPUT_DIR, req.job_id, "*_metadata.json"))
                    if meta_files:
                        with open(meta_files[0], 'r') as f:
                            data = json.load(f)
                            transcript = data.get('transcript')
                except Exception as e:
                    print(f"⚠️ Could not load transcript for editing context: {e}")

                # 3. Get Plan (Filter String)
                filter_data = editor.get_ffmpeg_filter(vid_file, duration, fps=fps, width=width, height=height, transcript=transcript)
                
                # 4. Apply
                # Use safe output name first
                safe_output_path = os.path.join(OUTPUT_DIR, req.job_id, f"temp_output_{req.job_id}.mp4")
                editor.apply_edits(safe_input_path, safe_output_path, filter_data)
                
                # Move result to final destination (rename works even if dest name has unicode if filesystem supports it, 
                # but python might still struggle if locale is broken? No, os.rename usually handles it better than subprocess args)
                # Actually, output_path is defined above: f"edited_{filename}"
                # If filename has unicode, output_path has unicode.
                # Let's hope shutil.move / os.rename works.
                if os.path.exists(safe_output_path):
                    shutil.move(safe_output_path, output_path)
                
                return filter_data
            finally:
                # Cleanup temp safe input
                if os.path.exists(safe_input_path):
                    os.remove(safe_input_path)

        # Run in thread pool
        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(None, run_edit)
        
        # Update clip URL in the job result? 
        # Or return new URL and let frontend handle it?
        # Updating job result allows persistence if page refreshes.
        
        new_video_url = f"/videos/{req.job_id}/{edited_filename}"
        
        # Start a new "edited" clip entry or just update the current one?
        # Let's update the current one's video_url but keep backup?
        # Or return the new URL to the frontend to display.
        
        return {
            "success": True, 
            "new_video_url": new_video_url,
            "edit_plan": plan
        }

    except Exception as e:
        print(f"❌ Edit Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SubtitleRequest(BaseModel):
    job_id: str
    clip_index: int
    position: str = "bottom" # top, middle, bottom
    font_size: int = 16
    font_name: str = "Verdana"
    font_color: str = "#FFFFFF"
    border_color: str = "#000000"
    border_width: int = 2
    bg_color: str = "#000000"
    bg_opacity: float = 0.0
    input_filename: Optional[str] = None


@app.get("/api/clip/{job_id}/{clip_index}/transcript")
async def get_clip_transcript(job_id: str, clip_index: int):
    """Return word-level captions for a specific clip, formatted for Remotion."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(OUTPUT_DIR, job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata")

    clips = data.get('shorts', [])
    if clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[clip_index]
    clip_start = clip_data.get('start', 0)
    clip_end = clip_data.get('end', 0)

    # Extract words within clip range and convert to CaptionWord format
    captions = []
    for segment in transcript.get('segments', []):
        for word_info in segment.get('words', []):
            if word_info['end'] > clip_start and word_info['start'] < clip_end:
                captions.append({
                    "text": word_info.get('word', '').strip(),
                    "startMs": int((max(0, word_info['start'] - clip_start)) * 1000),
                    "endMs": int((max(0, word_info['end'] - clip_start)) * 1000),
                })

    duration_sec = clip_end - clip_start

    return {
        "captions": captions,
        "durationSec": duration_sec,
        "language": transcript.get('language', 'en'),
    }


# --- Remotion Render Proxy ---
RENDER_SERVICE_URL = os.getenv("RENDER_SERVICE_URL", "http://renderer:3100")

@app.post("/api/render")
async def proxy_render(request: Request):
    """Proxy render requests to the Node.js Remotion render service."""
    import httpx
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{RENDER_SERVICE_URL}/render", json=body)
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Render service unavailable: {e}")

@app.get("/api/render/{render_id}")
async def proxy_render_status(render_id: str):
    """Proxy render status polling to the Node.js Remotion render service."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{RENDER_SERVICE_URL}/render/{render_id}")
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Render service unavailable: {e}")


class EffectsGenerateRequest(BaseModel):
    job_id: str
    clip_index: int
    input_filename: Optional[str] = None

@app.post("/api/effects/generate")
async def generate_effects_config(
    request: Request,
    req: EffectsGenerateRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    x_gemini_base_url: Optional[str] = Header(None, alias=GEMINI_BASE_URL_HEADER)
):
    """Generate structured EffectsConfig JSON for Remotion rendering via Gemini AI."""
    final_api_key, gemini_base_url, _gemini_pool = resolve_gemini_access(
        request,
        header_key=x_gemini_key,
        header_base_url=x_gemini_base_url,
        body=req.dict(),
    )

    if not final_api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key (Header)")

    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")

    try:
        # Resolve input path
        if req.input_filename:
            safe_name = os.path.basename(req.input_filename)
            input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_name)
        else:
            clip = job['result']['clips'][req.clip_index]
            filename = clip['video_url'].split('/')[-1]
            input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)

        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

        def run_effects_generation():
            editor = VideoEditor(api_key=final_api_key, base_url=gemini_base_url)

            # Create safe ASCII filename to avoid encoding issues
            safe_filename = f"temp_effects_{req.job_id}.mp4"
            safe_input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_filename)
            shutil.copy(input_path, safe_input_path)

            try:
                # Upload video to Gemini
                vid_file = editor.upload_video(safe_input_path)

                # Get video metadata via ffprobe
                probe_cmd = [
                    'ffprobe', '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=width,height,r_frame_rate,duration',
                    '-show_entries', 'format=duration',
                    '-of', 'json',
                    safe_input_path
                ]
                probe_result = subprocess.check_output(probe_cmd).decode().strip()
                probe_data = json.loads(probe_result)

                stream = probe_data.get('streams', [{}])[0]
                width = int(stream.get('width', 1080))
                height = int(stream.get('height', 1920))

                # Parse fps from r_frame_rate (e.g. "30/1")
                r_frame_rate = stream.get('r_frame_rate', '30/1')
                num, den = r_frame_rate.split('/')
                fps = round(int(num) / int(den), 2)

                # Get duration from stream or format
                duration = float(stream.get('duration', 0))
                if duration == 0:
                    duration = float(probe_data.get('format', {}).get('duration', 0))

                # Load transcript from metadata
                transcript = None
                try:
                    meta_files = glob.glob(os.path.join(OUTPUT_DIR, req.job_id, "*_metadata.json"))
                    if meta_files:
                        with open(meta_files[0], 'r') as f:
                            data = json.load(f)
                            transcript = data.get('transcript')
                except Exception as e:
                    print(f"⚠️ Could not load transcript for effects config: {e}")

                # Generate effects config
                effects_config = editor.get_effects_config(
                    vid_file, duration, fps=fps, width=width, height=height, transcript=transcript
                )

                return effects_config
            finally:
                if os.path.exists(safe_input_path):
                    os.remove(safe_input_path)

        loop = asyncio.get_event_loop()
        effects_config = await loop.run_in_executor(None, run_effects_generation)

        if effects_config is None:
            raise HTTPException(status_code=500, detail="Failed to generate effects config from Gemini")

        return {"effects": effects_config}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Effects Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/subtitle")
async def add_subtitles(req: SubtitleRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Reload job data from disk just in case metadata was updated
    job = jobs[req.job_id]
    
    # We need to access metadata.json to get the transcript
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
        
    with open(json_files[0], 'r') as f:
        data = json.load(f)
        
    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata. Please process a new video.")
        
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
        
    clip_data = clips[req.clip_index]
    
    # Video Path
    if req.input_filename:
        # Use chained file
        filename = os.path.basename(req.input_filename)
    else:
        # Fallback to standard naming
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"
         
    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        # Try looking for edited version if url implied it?
        # Just fail if not found.
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")
        
    # Define outputs
    srt_filename = f"subs_{req.clip_index}_{int(time.time())}.srt"
    srt_path = os.path.join(output_dir, srt_filename)
    
    # Output video
    # We create a new file "subtitled_..."
    output_filename = f"subtitled_{filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # 1. Generate SRT
        # Check if this is a dubbed video - if so, transcribe it fresh
        is_dubbed = filename.startswith("translated_")

        if is_dubbed:
            print(f"🎙️ Dubbed video detected, transcribing audio for subtitles...")
            def run_transcribe_srt():
                return generate_srt_from_video(input_path, srt_path)

            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, run_transcribe_srt)
        else:
            success = generate_srt(transcript, clip_data['start'], clip_data['end'], srt_path)

        if not success:
             raise HTTPException(status_code=400, detail="No words found for this clip range.")

        # 2. Burn Subtitles
        # Run in thread pool
        def run_burn():
             burn_subtitles(input_path, srt_path, output_path,
                           alignment=req.position, fontsize=req.font_size,
                           font_name=req.font_name, font_color=req.font_color,
                           border_color=req.border_color, border_width=req.border_width,
                           bg_color=req.bg_color, bg_opacity=req.bg_opacity)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_burn)
        
    except Exception as e:
        print(f"❌ Subtitle Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    # 3. Update Result and Metadata
    # Update InMemory Jobs
    if req.clip_index < len(job['result']['clips']):
         job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
    
    # Update Metadata on Disk (Persistence)
    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            # Update the main data structure
            data['shorts'] = clips
            
            # Write back
            with open(json_files[0], 'w') as f:
                json.dump(data, f, indent=4)
                print(f"✅ Metadata updated with subtitled video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")
        # Non-critical, but good for persistence

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }

class HookRequest(BaseModel):
    job_id: str
    clip_index: int
    text: str
    input_filename: Optional[str] = None
    position: Optional[str] = "top" # top, center, bottom
    size: Optional[str] = "M" # S, M, L

@app.post("/api/hook")
async def add_hook(req: HookRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[req.job_id]
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
        
    with open(json_files[0], 'r') as f:
        data = json.load(f)
        
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
        
    clip_data = clips[req.clip_index]
    
    # Video Path
    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"
         
    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")
        
    # Output video
    output_filename = f"hook_{filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    # Map Size to Scale
    size_map = {"S": 0.8, "M": 1.0, "L": 1.3}
    font_scale = size_map.get(req.size, 1.0)
    
    try:
        # Run in thread pool
        def run_hook():
             add_hook_to_video(input_path, req.text, output_path, position=req.position, font_scale=font_scale)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_hook)
        
    except Exception as e:
        print(f"❌ Hook Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    # Update Persistence (Same logic as subtitles)
    # Update InMemory Jobs
    if req.clip_index < len(job['result']['clips']):
         job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
    
    # Update Metadata on Disk
    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            data['shorts'] = clips
            with open(json_files[0], 'w') as f:
                json.dump(data, f, indent=4)
                print(f"✅ Metadata updated with hook video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }

class TranslateRequest(BaseModel):
    job_id: str
    clip_index: int
    target_language: str
    source_language: Optional[str] = None
    input_filename: Optional[str] = None

@app.get("/api/translate/languages")
async def get_languages():
    """Return supported languages for translation."""
    return {"languages": get_supported_languages()}

@app.post("/api/translate")
async def translate_clip(
    req: TranslateRequest,
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key")
):
    """Translate a video clip to a different language using ElevenLabs dubbing."""
    if not x_elevenlabs_key:
        raise HTTPException(status_code=400, detail="Missing X-ElevenLabs-Key header")

    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))

    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[req.clip_index]

    # Video Path
    if req.input_filename:
        filename = os.path.basename(req.input_filename)
    else:
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"

    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

    # Output video with language suffix
    base, ext = os.path.splitext(filename)
    output_filename = f"translated_{req.target_language}_{base}{ext}"
    output_path = os.path.join(output_dir, output_filename)

    try:
        # Run translation in thread pool (blocking API calls)
        def run_translate():
            return translate_video(
                video_path=input_path,
                output_path=output_path,
                target_language=req.target_language,
                api_key=x_elevenlabs_key,
                source_language=req.source_language,
            )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_translate)

    except Exception as e:
        print(f"❌ Translation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Update InMemory Jobs
    if req.clip_index < len(job['result']['clips']):
         job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"

    # Update Metadata on Disk
    try:
        if req.clip_index < len(clips):
            clips[req.clip_index]['video_url'] = f"/videos/{req.job_id}/{output_filename}"
            data['shorts'] = clips
            with open(json_files[0], 'w') as f:
                json.dump(data, f, indent=4)
                print(f"✅ Metadata updated with translated video for clip {req.clip_index}")
    except Exception as e:
        print(f"⚠️ Failed to update metadata.json: {e}")

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{output_filename}"
    }

class SocialPostRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: str
    user_id: str
    platforms: List[str] # ["tiktok", "instagram", "youtube"]
    # Optional overrides if frontend wants to edit them
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[str] = None # ISO-8601 string
    timezone: Optional[str] = "UTC"

import httpx

@app.post("/api/social/post")
async def post_to_socials(req: SocialPostRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")
        
    try:
        clip = job['result']['clips'][req.clip_index]
        # Video URL is relative /videos/..., we need absolute file path
        # clip['video_url'] is like "/videos/{job_id}/{filename}"
        # We constructed it as: f"/videos/{job_id}/{clip_filename}"
        # And file is at f"{OUTPUT_DIR}/{job_id}/{clip_filename}"
        
        filename = clip['video_url'].split('/')[-1]
        file_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
        
        if not os.path.exists(file_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {file_path}")

        # Construct parameters for Upload-Post API
        # Fallbacks
        final_title = req.title or clip.get('title', 'Viral Short')
        final_description = req.description or clip.get('video_description_for_instagram') or clip.get('video_description_for_tiktok') or "Check this out!"
        
        # Prepare form data
        url = "https://api.upload-post.com/api/upload"
        headers = {
            "Authorization": f"Apikey {req.api_key}"
        }
        
        # Prepare data as dict (httpx handles lists for multiple values)
        data_payload = {
            "user": req.user_id,
            "title": final_title,
            "platform[]": req.platforms, # Pass list directly
            "async_upload": "true"  # Enable async upload
        }

        # Add scheduling if present
        if req.scheduled_date:
            data_payload["scheduled_date"] = req.scheduled_date
            if req.timezone:
                data_payload["timezone"] = req.timezone
        
        # Add Platform specifics
        if "tiktok" in req.platforms:
             data_payload["tiktok_title"] = final_description
             
        if "instagram" in req.platforms:
             data_payload["instagram_title"] = final_description
             data_payload["media_type"] = "REELS"

        if "youtube" in req.platforms:
             yt_title = req.title or clip.get('video_title_for_youtube_short', final_title)
             data_payload["youtube_title"] = yt_title
             data_payload["youtube_description"] = final_description
             data_payload["privacyStatus"] = "public"

        # Send File
        # httpx AsyncClient requires async file reading or bytes. 
        # Since we have MAX_FILE_SIZE_MB, reading into memory is safe-ish.
        with open(file_path, "rb") as f:
            file_content = f.read()
            
        files = {
            "video": (filename, file_content, "video/mp4")
        }

        # Switch to synchronous Client to avoid "sync request with AsyncClient" error with multipart/files
        with httpx.Client(timeout=120.0) as client:
            print(f"📡 Sending to Upload-Post for platforms: {req.platforms}")
            response = client.post(url, headers=headers, data=data_payload, files=files)
            
        if response.status_code not in [200, 201, 202]: # Added 201
             print(f"❌ Upload-Post Error: {response.text}")
             raise HTTPException(status_code=response.status_code, detail=f"Vendor API Error: {response.text}")

        return response.json()

    except Exception as e:
        print(f"❌ Social Post Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/user")
async def get_social_user(api_key: str = Header(..., alias="X-Upload-Post-Key")):
    """Proxy to fetch user ID from Upload-Post"""
    if not api_key:
         raise HTTPException(status_code=400, detail="Missing X-Upload-Post-Key header")
         
    url = "https://api.upload-post.com/api/uploadposts/users"
    print(f"🔍 Fetching User ID from: {url}")
    headers = {"Authorization": f"Apikey {api_key}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"❌ Upload-Post User Fetch Error: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch user: {resp.text}")
            
            data = resp.json()
            print(f"🔍 Upload-Post User Response: {data}")
            
            user_id = None
            # The structure is {'success': True, 'profiles': [{'username': '...'}, ...]}
            profiles_list = []
            if isinstance(data, dict):
                 raw_profiles = data.get('profiles', [])
                 if isinstance(raw_profiles, list):
                     for p in raw_profiles:
                         username = p.get('username')
                         if username:
                             # Determine connected platforms
                             socials = p.get('social_accounts', {})
                             connected = []
                             # Check typical platforms
                             for platform in ['tiktok', 'instagram', 'youtube']:
                                 account_info = socials.get(platform)
                                 # If it's a dict and typically has data, or just not empty string
                                 if isinstance(account_info, dict):
                                     connected.append(platform)
                             
                             profiles_list.append({
                                 "username": username,
                                 "connected": connected
                             })
            
            if not profiles_list:
                # Fallback if no profiles found
                return {"profiles": [], "error": "No profiles found"}
                
            return {"profiles": profiles_list}
            
            
        except Exception as e:
             raise HTTPException(status_code=500, detail=str(e))

# --- Thumbnail Studio Endpoints ---

@app.post("/api/thumbnail/upload")
async def thumbnail_upload(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
):
    """Upload video and start background Whisper transcription immediately."""
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")

    session_id = str(uuid.uuid4())
    transcript_event = asyncio.Event()

    # Save file if uploaded directly
    video_path = None
    if file:
        video_path = os.path.join(UPLOAD_DIR, f"thumb_{session_id}_{file.filename}")
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

    # Initialize session
    thumbnail_sessions[session_id] = {
        "video_path": video_path,
        "transcript_event": transcript_event,
        "transcript_ready": False,
        "transcript": None,
        "transcript_segments": [],
        "video_duration": 0,
        "language": "en",
        "context": "",
        "titles": [],
        "conversation": [],
        "_url": url,  # Store URL for deferred download
    }

    async def run_background_whisper():
        try:
            vpath = video_path
            # Download YouTube video if URL was provided
            if not vpath and url:
                from main import download_youtube_video
                loop = asyncio.get_event_loop()
                vpath, _ = await loop.run_in_executor(None, download_youtube_video, url, UPLOAD_DIR)
                thumbnail_sessions[session_id]["video_path"] = vpath

            from main import transcribe_video
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(None, transcribe_video, vpath)
            segments = transcript.get("segments", [])
            duration = segments[-1]["end"] if segments else 0

            thumbnail_sessions[session_id].update({
                "transcript_ready": True,
                "transcript": transcript,
                "transcript_segments": segments,
                "video_duration": duration,
                "language": transcript.get("language", "en"),
            })
            print(f"✅ [Thumbnail] Background Whisper complete for session {session_id}")
        except Exception as e:
            print(f"❌ [Thumbnail] Background Whisper failed: {e}")
            thumbnail_sessions[session_id]["transcript_error"] = str(e)
        finally:
            transcript_event.set()

    asyncio.create_task(run_background_whisper())

    return {"session_id": session_id}


@app.post("/api/thumbnail/analyze")
async def thumbnail_analyze(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    x_gemini_base_url: Optional[str] = Header(None, alias=GEMINI_BASE_URL_HEADER)
):
    """Analyze a video and suggest viral YouTube titles."""
    api_key, gemini_base_url, _gemini_pool = resolve_gemini_access(
        request,
        header_key=x_gemini_key,
        header_base_url=x_gemini_base_url,
    )
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    pre_transcript = None

    # Check for pre-existing session with background Whisper
    if session_id and session_id in thumbnail_sessions:
        session = thumbnail_sessions[session_id]

        # Wait for background Whisper to complete
        transcript_event = session.get("transcript_event")
        if transcript_event:
            print(f"⏳ [Thumbnail] Waiting for background Whisper to finish...")
            await transcript_event.wait()

        if session.get("transcript_error"):
            raise HTTPException(status_code=500, detail=f"Transcription failed: {session['transcript_error']}")

        video_path = session["video_path"]
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found in session")

        if session.get("transcript_ready"):
            pre_transcript = session["transcript"]
    else:
        # No pre-existing session — need file or URL
        if not url and not file:
            raise HTTPException(status_code=400, detail="Must provide URL, File, or session_id")

        session_id = str(uuid.uuid4())

        if url:
            from main import download_youtube_video
            video_path, _ = download_youtube_video(url, UPLOAD_DIR)
        else:
            video_path = os.path.join(UPLOAD_DIR, f"thumb_{session_id}_{file.filename}")
            with open(video_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

    try:
        # Run analysis in thread pool (skips Whisper if pre_transcript is available)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_video_for_titles, api_key, video_path, pre_transcript, gemini_base_url)

        # Store/update session context
        if session_id not in thumbnail_sessions:
            thumbnail_sessions[session_id] = {}

        thumbnail_sessions[session_id].update({
            "context": result.get("transcript_summary", ""),
            "titles": result.get("titles", []),
            "language": result.get("language", "en"),
            "conversation": thumbnail_sessions[session_id].get("conversation", []),
            "video_path": video_path,
            "transcript_segments": result.get("segments", []),
            "video_duration": result.get("video_duration", 0)
        })

        return {
            "session_id": session_id,
            "titles": result.get("titles", []),
            "context": result.get("transcript_summary", ""),
            "language": result.get("language", "en"),
            "recommended": result.get("recommended", [])
        }

    except Exception as e:
        print(f"❌ Thumbnail Analyze Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ThumbnailTitlesRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    title: Optional[str] = None

@app.post("/api/thumbnail/titles")
async def thumbnail_titles(
    request: Request,
    req: ThumbnailTitlesRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    x_gemini_base_url: Optional[str] = Header(None, alias=GEMINI_BASE_URL_HEADER)
):
    """Refine title suggestions or accept a manual title."""
    api_key, gemini_base_url, _gemini_pool = resolve_gemini_access(
        request,
        header_key=x_gemini_key,
        header_base_url=x_gemini_base_url,
        body=req.dict(),
    )
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    # Manual title mode - just create a session with the user's title
    if req.title:
        session_id = req.session_id or str(uuid.uuid4())
        if session_id not in thumbnail_sessions:
            thumbnail_sessions[session_id] = {
                "context": "",
                "titles": [req.title],
                "language": "en",
                "conversation": []
            }
        return {"session_id": session_id, "titles": [req.title]}

    # Refinement mode
    if not req.session_id or req.session_id not in thumbnail_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if not req.message:
        raise HTTPException(status_code=400, detail="Must provide message or title")

    session = thumbnail_sessions[req.session_id]

    # Add user message to conversation history
    session["conversation"].append({"role": "user", "content": req.message})

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            refine_titles,
            api_key,
            session["context"],
            req.message,
            session["conversation"],
            gemini_base_url
        )

        new_titles = result.get("titles", [])
        session["titles"] = new_titles
        session["conversation"].append({"role": "assistant", "content": json.dumps(new_titles)})

        return {"titles": new_titles}

    except Exception as e:
        print(f"❌ Thumbnail Titles Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/thumbnail/generate")
async def thumbnail_generate(
    request: Request,
    session_id: str = Form(...),
    title: str = Form(...),
    extra_prompt: str = Form(""),
    count: int = Form(3),
    face: Optional[UploadFile] = File(None),
    background: Optional[UploadFile] = File(None),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    x_gemini_base_url: Optional[str] = Header(None, alias=GEMINI_BASE_URL_HEADER)
):
    """Generate YouTube thumbnails with Gemini image generation."""
    api_key, gemini_base_url, _gemini_pool = resolve_gemini_access(
        request,
        header_key=x_gemini_key,
        header_base_url=x_gemini_base_url,
    )
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    # Clamp count
    count = min(max(1, count), 6)

    # Save optional uploaded images
    face_path = None
    bg_path = None
    thumb_upload_dir = os.path.join(UPLOAD_DIR, f"thumb_{session_id}")
    os.makedirs(thumb_upload_dir, exist_ok=True)

    try:
        if face and face.filename:
            face_path = os.path.join(thumb_upload_dir, f"face_{face.filename}")
            with open(face_path, "wb") as f:
                f.write(await face.read())

        if background and background.filename:
            bg_path = os.path.join(thumb_upload_dir, f"bg_{background.filename}")
            with open(bg_path, "wb") as f:
                f.write(await background.read())

        # Get video context from session (transcript summary from analysis step)
        video_context = ""
        if session_id in thumbnail_sessions:
            video_context = thumbnail_sessions[session_id].get("context", "")

        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        thumbnails = await loop.run_in_executor(
            None,
            generate_thumbnail,
            api_key,
            title,
            session_id,
            face_path,
            bg_path,
            extra_prompt,
            count,
            video_context,
            gemini_base_url
        )

        if not thumbnails:
            raise HTTPException(status_code=500, detail="Thumbnail generation failed. Please check your Gemini API key has access to image generation (gemini-3.1-flash-image-preview model).")

        return {"thumbnails": thumbnails}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Thumbnail Generate Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ThumbnailDescribeRequest(BaseModel):
    session_id: str
    title: str

@app.post("/api/thumbnail/describe")
async def thumbnail_describe(
    request: Request,
    req: ThumbnailDescribeRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    x_gemini_base_url: Optional[str] = Header(None, alias=GEMINI_BASE_URL_HEADER)
):
    """Generate a YouTube description with chapters from the transcript."""
    api_key, gemini_base_url, _gemini_pool = resolve_gemini_access(
        request,
        header_key=x_gemini_key,
        header_base_url=x_gemini_base_url,
        body=req.dict(),
    )
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    if req.session_id not in thumbnail_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = thumbnail_sessions[req.session_id]
    segments = session.get("transcript_segments", [])
    if not segments:
        raise HTTPException(status_code=400, detail="No transcript segments available. Please analyze a video first.")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            generate_youtube_description,
            api_key,
            req.title,
            segments,
            session.get("language", "en"),
            session.get("video_duration", 0),
            gemini_base_url
        )
        return {"description": result.get("description", "")}

    except Exception as e:
        print(f"❌ Thumbnail Describe Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/thumbnail/publish")
async def thumbnail_publish(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    thumbnail_url: str = Form(...),
    api_key: str = Form(...),
    user_id: str = Form(...),
):
    """Kick off a background upload to YouTube via Upload-Post and return immediately."""
    if session_id not in thumbnail_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = thumbnail_sessions[session_id]
    video_path = session.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Original video file not found")

    # Resolve thumbnail path from URL
    thumb_relative = thumbnail_url.lstrip("/")
    if thumb_relative.startswith("thumbnails/"):
        thumb_path = os.path.join(OUTPUT_DIR, thumb_relative)
    else:
        thumb_path = os.path.join(THUMBNAILS_DIR, thumb_relative)

    if not os.path.exists(thumb_path):
        raise HTTPException(status_code=404, detail=f"Thumbnail file not found: {thumb_path}")

    # Generate a unique ID for this publish job so the frontend can poll
    publish_id = str(uuid.uuid4())
    publish_jobs[publish_id] = {"status": "uploading", "result": None, "error": None}

    def do_upload():
        """Runs in a thread via BackgroundTasks — does the actual multipart upload."""
        try:
            upload_url = "https://api.upload-post.com/api/upload"
            headers = {"Authorization": f"Apikey {api_key}"}
            data_payload = {
                "user": user_id,
                "platform[]": ["youtube"],
                "title": title,          # required base field (fallback)
                "async_upload": "true",
                "youtube_title": title,
                "youtube_description": description,
                "privacyStatus": "public",
            }
            video_filename = os.path.basename(video_path)
            thumb_filename = os.path.basename(thumb_path)

            print(f"📡 [Thumbnail] Publishing to YouTube via Upload-Post... (publish_id={publish_id})")
            with open(video_path, "rb") as vf, open(thumb_path, "rb") as tf:
                files = {
                    "video": (video_filename, vf.read(), "video/mp4"),
                    "thumbnail": (thumb_filename, tf.read(), "image/jpeg"),
                }

            # Use a long timeout — video uploads can take several minutes
            with httpx.Client(timeout=600.0) as client:
                response = client.post(upload_url, headers=headers, data=data_payload, files=files)

            if response.status_code not in [200, 201, 202]:
                err = f"Upload-Post API Error ({response.status_code}): {response.text}"
                print(f"❌ {err}")
                publish_jobs[publish_id]["status"] = "failed"
                publish_jobs[publish_id]["error"] = err
            else:
                print(f"✅ [Thumbnail] Published successfully (publish_id={publish_id})")
                publish_jobs[publish_id]["status"] = "done"
                publish_jobs[publish_id]["result"] = response.json()

        except Exception as e:
            err = str(e)
            print(f"❌ Thumbnail Publish Background Error: {err}")
            publish_jobs[publish_id]["status"] = "failed"
            publish_jobs[publish_id]["error"] = err

    background_tasks.add_task(do_upload)
    return {"publish_id": publish_id, "status": "uploading"}


@app.get("/api/thumbnail/publish/status/{publish_id}")
async def thumbnail_publish_status(publish_id: str):
    """Poll the status of a background publish job."""
    if publish_id not in publish_jobs:
        raise HTTPException(status_code=404, detail="Publish job not found")
    return publish_jobs[publish_id]


# @app.get("/api/gallery/clips")
# async def get_gallery_clips(limit: int = 20, offset: int = 0, refresh: bool = False):
#     """
#     Fetch clips from S3 for the gallery with pagination.
#
#     Args:
#         limit: Number of clips to return (default 20, max 100)
#         offset: Starting position for pagination
#         refresh: Force refresh cache
#     """
#     try:
#         # Clamp limit to reasonable values
#         limit = min(max(1, limit), 100)
#
#         # Get clips (uses cache internally)
#         all_clips = list_all_clips(limit=limit + offset, force_refresh=refresh)
#
#         # Apply offset for pagination
#         clips = all_clips[offset:offset + limit]
#
#         return {
#             "clips": clips,
#             "total": len(all_clips),
#             "limit": limit,
#             "offset": offset,
#             "has_more": len(all_clips) > offset + limit
#         }
#     except Exception as e:
#         print(f"❌ Gallery Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════
# SaaSShorts: AI UGC Video Generator for SaaS Products
# ═══════════════════════════════════════════════════════════════════════

from saasshorts import (
    scrape_website,
    research_saas_online,
    analyze_saas,
    generate_scripts,
    generate_full_video,
    generate_actor_images,
    get_elevenlabs_voices,
    DEFAULT_VOICES,
)

# State for SaaSShorts jobs (separate from video processing jobs)
saas_jobs: Dict[str, Dict] = {}


class SaaSAnalyzeRequest(BaseModel):
    url: Optional[str] = None
    description: Optional[str] = None  # Manual product/business description
    num_scripts: int = 3
    style: str = "ugc"
    language: str = "en"
    actor_gender: str = "female"


@app.post("/api/saasshorts/analyze")
async def saasshorts_analyze(
    request: Request,
    req: SaaSAnalyzeRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    x_gemini_base_url: Optional[str] = Header(None, alias=GEMINI_BASE_URL_HEADER),
):
    """Analyze a URL or manual description and generate video scripts."""
    gemini_key, gemini_base_url, _gemini_pool = resolve_gemini_access(
        request,
        header_key=x_gemini_key,
        header_base_url=x_gemini_base_url,
        body=req.dict(),
    )
    if not gemini_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key")

    if not req.url and not req.description:
        raise HTTPException(status_code=400, detail="Provide a URL or a product description")

    try:
        loop = asyncio.get_event_loop()

        def run_analysis():
            web_research = None

            if req.url and req.url.strip():
                # URL provided: full scrape + research pipeline
                scraped = scrape_website(req.url)
                web_research = research_saas_online(req.url, gemini_key, gemini_base_url)
                analysis = analyze_saas(scraped, gemini_key, web_research=web_research, base_url=gemini_base_url)
            else:
                # Manual description: build analysis from description
                analysis = {
                    "product_name": req.description.split(",")[0].strip()[:60] if req.description else "Product",
                    "description": req.description,
                    "value_proposition": req.description,
                    "target_audience": "general audience",
                    "key_features": [req.description],
                    "pain_points": [],
                    "tone": "casual and authentic",
                }

            scripts = generate_scripts(analysis, gemini_key, req.num_scripts, req.style, req.language, req.actor_gender, gemini_base_url)
            return {
                "analysis": analysis,
                "scripts": scripts,
                "web_research": web_research,
            }

        result = await loop.run_in_executor(None, run_analysis)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaaSActorRequest(BaseModel):
    actor_description: str
    num_options: int = 3
    product_description: Optional[str] = None


@app.post("/api/saasshorts/actor-upload")
async def saasshorts_actor_upload(file: UploadFile = File(...)):
    """Upload a custom actor image (stored locally only, not S3)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        content = await file.read()

        # Validate minimum size
        if len(content) < 1000:
            raise HTTPException(status_code=400, detail="File too small to be a valid image")

        upload_id = uuid.uuid4().hex[:8]
        upload_dir = os.path.join(OUTPUT_DIR, "actor_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        filename = f"custom_{upload_id}.png"
        file_path = os.path.join(upload_dir, filename)

        with open(file_path, "wb") as f:
            f.write(content)

        return {"url": f"/videos/actor_uploads/{filename}"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/saasshorts/actor-options")
async def saasshorts_actor_options(
    req: SaaSActorRequest,
    x_fal_key: Optional[str] = Header(None, alias="X-Fal-Key"),
):
    """Generate multiple actor image options for the user to choose from."""
    fal_key = x_fal_key
    if not fal_key:
        raise HTTPException(status_code=400, detail="Missing fal.ai API Key")

    try:
        job_id = str(uuid.uuid4())
        out_dir = os.path.join(OUTPUT_DIR, f"saas_actors_{job_id}")
        os.makedirs(out_dir, exist_ok=True)

        loop = asyncio.get_running_loop()
        import functools
        paths = await loop.run_in_executor(
            None,
            functools.partial(
                generate_actor_images,
                req.actor_description, fal_key, out_dir, "actor", req.num_options,
                product_description=req.product_description,
            ),
        )

        # Upload each actor image to public S3 with description
        desc = req.actor_description
        if req.product_description:
            desc += f" (holding {req.product_description})"
        urls = []
        for p in paths:
            s3_url = upload_actor_to_s3(p, description=desc)
            if s3_url:
                urls.append(s3_url)
            else:
                # Fallback to local URL if S3 fails
                urls.append(f"/videos/saas_actors_{job_id}/{os.path.basename(p)}")

        return {"images": urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/saasshorts/gallery")
async def saasshorts_video_gallery(limit: int = 50):
    """List all UGC videos from the public gallery."""
    try:
        loop = asyncio.get_running_loop()
        videos = await loop.run_in_executor(None, list_video_gallery, limit)
        return {"videos": videos, "total": len(videos)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaaSPostRequest(BaseModel):
    job_id: str
    api_key: str
    user_id: str
    platforms: List[str]
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[str] = None
    timezone: Optional[str] = "UTC"


@app.post("/api/saasshorts/post")
async def saasshorts_post_to_socials(req: SaaSPostRequest):
    """Post an AI Shorts video to social media via Upload-Post."""
    if req.job_id not in saas_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = saas_jobs[req.job_id]
    result = job.get("result")
    if not result or not result.get("video_url"):
        raise HTTPException(status_code=400, detail="No video available for this job")

    try:
        # Resolve video file path
        video_url = result["video_url"]  # e.g. /videos/saas_xxx/slug_final.mp4
        rel_path = video_url.replace("/videos/", "")
        file_path = os.path.join(OUTPUT_DIR, rel_path)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Video file not found")

        script = result.get("script", {})
        final_title = req.title or script.get("title", "AI Short")
        final_description = req.description or script.get("caption", "")
        if not final_description:
            final_description = script.get("full_narration", "Check this out!")

        url = "https://api.upload-post.com/api/upload"
        headers = {"Authorization": f"Apikey {req.api_key}"}

        data_payload = {
            "user": req.user_id,
            "title": final_title,
            "platform[]": req.platforms,
            "async_upload": "true",
        }

        if req.scheduled_date:
            data_payload["scheduled_date"] = req.scheduled_date
            if req.timezone:
                data_payload["timezone"] = req.timezone

        if "tiktok" in req.platforms:
            data_payload["tiktok_title"] = final_description
        if "instagram" in req.platforms:
            data_payload["instagram_title"] = final_description
            data_payload["media_type"] = "REELS"
        if "youtube" in req.platforms:
            data_payload["youtube_title"] = final_title
            data_payload["youtube_description"] = final_description
            data_payload["privacyStatus"] = "public"

        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_content = f.read()

        files = {"video": (filename, file_content, "video/mp4")}

        with httpx.Client(timeout=120.0) as client:
            print(f"📡 [AI Shorts] Sending to Upload-Post: {req.platforms}")
            response = client.post(url, headers=headers, data=data_payload, files=files)

        if response.status_code not in [200, 201, 202]:
            raise HTTPException(status_code=response.status_code, detail=f"Upload-Post Error: {response.text}")

        return response.json()

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AI Shorts] Post Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gallery", response_class=HTMLResponse)
async def gallery_html_page():
    """SEO gallery page with all generated UGC videos."""
    import html as html_mod
    loop = asyncio.get_running_loop()
    videos = await loop.run_in_executor(None, list_video_gallery, 100)

    cards_html = ""
    ld_items = []
    for i, v in enumerate(videos):
        title = html_mod.escape(v.get("title", "Untitled"))
        video_url = v.get("video_url", "")
        actor_url = v.get("actor_url", "")
        video_id = v.get("video_id", "")
        duration = v.get("duration", 0)
        mode = v.get("video_mode", "")
        product = html_mod.escape(v.get("product_name", ""))
        caption = html_mod.escape(v.get("caption", "")[:120])

        mode_badge = '<span style="background:#22c55e;color:#000;padding:2px 8px;border-radius:9999px;font-size:10px;font-weight:700">LOW COST</span>' if mode == "lowcost" else '<span style="background:#8b5cf6;color:#fff;padding:2px 8px;border-radius:9999px;font-size:10px;font-weight:700">PREMIUM</span>'

        cards_html += f'''
        <a href="/video/{video_id}" style="text-decoration:none;color:inherit">
          <div style="background:#18181b;border-radius:16px;overflow:hidden;border:1px solid #27272a;transition:transform 0.2s" onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
            <div style="position:relative;aspect-ratio:9/16;background:#000">
              <video src="{video_url}" poster="{actor_url}" muted playsinline preload="metadata"
                     onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0"
                     style="width:100%;height:100%;object-fit:cover"></video>
              <div style="position:absolute;top:8px;right:8px">{mode_badge}</div>
            </div>
            <div style="padding:12px">
              <h2 style="font-size:14px;font-weight:600;margin:0 0 4px 0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{title}</h2>
              <p style="font-size:11px;color:#71717a;margin:0">{duration:.0f}s · {product}</p>
            </div>
          </div>
        </a>'''

        ld_items.append(f'{{"@type":"ListItem","position":{i+1},"url":"https://openshorts.app/video/{video_id}","name":"{title}"}}')

    ld_json = f'{{"@context":"https://schema.org","@type":"CollectionPage","name":"AI UGC Video Gallery","mainEntity":{{"@type":"ItemList","numberOfItems":{len(videos)},"itemListElement":[{",".join(ld_items)}]}}}}'

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI UGC Video Gallery | OpenShorts</title>
<meta name="description" content="Browse {len(videos)} AI-generated UGC marketing videos. Create viral TikTok and Instagram Reels for your SaaS product.">
<meta name="robots" content="index, follow">
<meta property="og:title" content="AI UGC Video Gallery | OpenShorts">
<meta property="og:type" content="website">
<meta property="og:description" content="Browse AI-generated UGC marketing videos for SaaS products.">
<script type="application/ld+json">{ld_json}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0c;color:#e4e4e7;font-family:-apple-system,BlinkMacSystemFont,sans-serif}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:20px;padding:20px;max-width:1400px;margin:0 auto}}
nav{{padding:20px 40px;border-bottom:1px solid #27272a;display:flex;align-items:center;justify-content:space-between}}
h1{{font-size:28px;font-weight:700;padding:40px 20px 0;text-align:center}}
.subtitle{{text-align:center;color:#71717a;font-size:14px;padding:8px 20px 20px}}
.cta{{display:inline-block;background:#8b5cf6;color:#fff;padding:10px 24px;border-radius:12px;text-decoration:none;font-weight:600;font-size:14px}}
</style>
</head>
<body>
<nav><strong style="font-size:18px">OpenShorts</strong><a href="/" class="cta">Create Your Video</a></nav>
<h1>AI-Generated UGC Videos</h1>
<p class="subtitle">{len(videos)} videos generated · Low Cost & Premium modes</p>
<div class="grid">{cards_html}</div>
<div style="text-align:center;padding:40px"><a href="/" class="cta">Create Your Own UGC Video</a></div>
</body></html>'''


@app.get("/video/{video_id}", response_class=HTMLResponse)
async def video_html_page(video_id: str):
    """SEO individual video page with og:video meta tags."""
    import html as html_mod
    loop = asyncio.get_running_loop()
    videos = await loop.run_in_executor(None, list_video_gallery, 200)
    meta = next((v for v in videos if v.get("video_id") == video_id), None)
    if not meta:
        raise HTTPException(status_code=404, detail="Video not found")

    title = html_mod.escape(meta.get("title", "Untitled"))
    caption = html_mod.escape(meta.get("caption", ""))
    narration = html_mod.escape(meta.get("full_narration", ""))
    video_url = meta.get("video_url", "")
    actor_url = meta.get("actor_url", "")
    duration = meta.get("duration", 0)
    mode = meta.get("video_mode", "")
    product = html_mod.escape(meta.get("product_name", ""))
    product_url = html_mod.escape(meta.get("product_url", ""))
    language = meta.get("language", "en")
    hashtags = " ".join(meta.get("hashtags", []))
    cost = meta.get("cost_estimate", {}).get("total", 0)
    created = meta.get("created_at", "")
    actor_desc = html_mod.escape(meta.get("actor_description", ""))

    ld_json = f'{{"@context":"https://schema.org","@type":"VideoObject","name":"{title}","description":"{caption}","thumbnailUrl":"{actor_url}","contentUrl":"{video_url}","uploadDate":"{created}","duration":"PT{int(duration)}S","width":1080,"height":1920,"inLanguage":"{language}"}}'

    mode_label = "Low Cost" if mode == "lowcost" else "Premium"

    return f'''<!DOCTYPE html>
<html lang="{language}">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} - AI UGC Video | OpenShorts</title>
<meta name="description" content="{caption} {hashtags}">
<meta property="og:type" content="video.other">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{caption}">
<meta property="og:video" content="{video_url}">
<meta property="og:video:type" content="video/mp4">
<meta property="og:video:width" content="1080">
<meta property="og:video:height" content="1920">
<meta property="og:image" content="{actor_url}">
<meta name="twitter:card" content="player">
<meta name="twitter:title" content="{title}">
<meta name="twitter:image" content="{actor_url}">
<script type="application/ld+json">{ld_json}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0c;color:#e4e4e7;font-family:-apple-system,BlinkMacSystemFont,sans-serif}}
nav{{padding:20px 40px;border-bottom:1px solid #27272a;display:flex;align-items:center;gap:16px}}
nav a{{color:#a1a1aa;text-decoration:none;font-size:14px}}
.container{{max-width:1000px;margin:0 auto;padding:40px 20px;display:grid;grid-template-columns:1fr 1fr;gap:40px}}
@media(max-width:768px){{.container{{grid-template-columns:1fr}}}}
video{{width:100%;border-radius:16px;background:#000}}
h1{{font-size:22px;font-weight:700;margin-bottom:8px}}
.meta{{color:#71717a;font-size:13px;margin-bottom:20px}}
.section{{margin-bottom:20px}}
.section h2{{font-size:13px;color:#71717a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}}
.section p{{font-size:14px;line-height:1.6}}
.badge{{display:inline-block;padding:3px 10px;border-radius:9999px;font-size:11px;font-weight:700}}
.cta{{display:inline-block;background:#8b5cf6;color:#fff;padding:10px 24px;border-radius:12px;text-decoration:none;font-weight:600;font-size:14px;margin-top:20px}}
</style>
</head>
<body>
<nav><strong>OpenShorts</strong><a href="/gallery">Gallery</a><span style="color:#3f3f46">›</span><span style="color:#e4e4e7;font-size:14px">{title}</span></nav>
<div class="container">
<div><video src="{video_url}" poster="{actor_url}" controls autoplay playsinline style="aspect-ratio:9/16;object-fit:cover"></video></div>
<div>
<h1>{title}</h1>
<p class="meta">{duration:.0f}s · {mode_label} · ${cost:.2f} · {product}</p>
<div class="section"><h2>Caption</h2><p>{caption}</p><p style="color:#8b5cf6;margin-top:4px">{hashtags}</p></div>
<div class="section"><h2>Script</h2><p>{narration}</p></div>
<div class="section"><h2>Actor</h2><p>{actor_desc}</p></div>
{f'<div class="section"><h2>Product</h2><p><a href="{product_url}" style="color:#8b5cf6" target="_blank">{product}</a></p></div>' if product_url else ''}
<a href="/gallery">← Back to Gallery</a>
<br><a href="/" class="cta">Create Your Own</a>
</div>
</div>
</body></html>'''


@app.get("/api/saasshorts/actor-gallery")
async def saasshorts_actor_gallery():
    """List all previously generated actor images from public S3."""
    try:
        loop = asyncio.get_running_loop()
        images = await loop.run_in_executor(None, list_actor_gallery)
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SaaSGenerateRequest(BaseModel):
    script: dict
    voice_id: Optional[str] = None
    actor_description: Optional[str] = None
    selected_actor_url: Optional[str] = None  # Pre-selected actor image URL
    retry_job_id: Optional[str] = None
    video_mode: str = "lowcost"  # "lowcost" or "premium"


@app.post("/api/saasshorts/generate")
async def saasshorts_generate(
    req: SaaSGenerateRequest,
    x_fal_key: Optional[str] = Header(None, alias="X-Fal-Key"),
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key"),
):
    """Generate a SaaS UGC video from a script. Returns a job_id for polling."""
    fal_key = x_fal_key
    elevenlabs_key = x_elevenlabs_key

    if not fal_key:
        raise HTTPException(status_code=400, detail="Missing fal.ai API Key (X-Fal-Key header)")
    if not elevenlabs_key:
        raise HTTPException(status_code=400, detail="Missing ElevenLabs API Key (X-ElevenLabs-Key header)")

    # Support retry: reuse output_dir so cached assets (image, voice, head, broll) are kept
    reused = False
    if req.retry_job_id:
        # Check memory first, then disk
        old_dir = os.path.join(OUTPUT_DIR, f"saas_{req.retry_job_id}")
        if req.retry_job_id in saas_jobs:
            old_dir = saas_jobs[req.retry_job_id]["output_dir"]

        if os.path.isdir(old_dir):
            job_id = req.retry_job_id
            job_output_dir = old_dir
            reused = True
            # Clear the 0-byte final video so pipeline re-generates it
            for f in os.listdir(old_dir):
                fp = os.path.join(old_dir, f)
                if f.endswith("_final.mp4") and os.path.getsize(fp) == 0:
                    os.remove(fp)
            saas_jobs[job_id] = {
                "status": "processing",
                "logs": [f"Retrying job {job_id[:8]}... reusing cached assets from disk."],
                "result": None,
                "output_dir": job_output_dir,
            }

    if not reused:
        job_id = str(uuid.uuid4())
        job_output_dir = os.path.join(OUTPUT_DIR, f"saas_{job_id}")
        os.makedirs(job_output_dir, exist_ok=True)
        saas_jobs[job_id] = {
            "status": "processing",
            "logs": ["SaaSShorts job started."],
            "result": None,
            "output_dir": job_output_dir,
        }

    # If user selected a pre-generated actor, resolve it to a local path
    selected_actor_path = None
    if req.selected_actor_url:
        if req.selected_actor_url.startswith("http"):
            # Download from S3 public URL to job output dir
            import httpx
            try:
                actor_local = os.path.join(job_output_dir, "selected_actor.png")
                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(req.selected_actor_url)
                    if resp.status_code == 200:
                        with open(actor_local, "wb") as f:
                            f.write(resp.content)
                        selected_actor_path = actor_local
            except Exception:
                pass
        else:
            src = os.path.join(OUTPUT_DIR, req.selected_actor_url.replace("/videos/", ""))
            if os.path.exists(src):
                selected_actor_path = src

    config = {
        "fal_key": fal_key,
        "elevenlabs_key": elevenlabs_key,
        "voice_id": req.voice_id or "21m00Tcm4TlvDq8ikWAM",
        "actor_description": req.actor_description,
        "selected_actor_path": selected_actor_path,
        "video_mode": req.video_mode,
    }

    async def run_generation():
        await concurrency_semaphore.acquire()
        try:
            loop = asyncio.get_running_loop()

            def log_msg(msg):
                print(f"[SaaSShorts Job {job_id[:8]}] {msg}")
                if job_id in saas_jobs:
                    saas_jobs[job_id]["logs"].append(msg)

            def run():
                return generate_full_video(req.script, config, job_output_dir, log_msg)

            result = await loop.run_in_executor(None, run)

            if job_id in saas_jobs:
                video_filename = result["video_filename"]
                saas_jobs[job_id]["status"] = "completed"
                saas_jobs[job_id]["result"] = {
                    "video_url": f"/videos/saas_{job_id}/{video_filename}",
                    "video_filename": video_filename,
                    "duration": result.get("duration", 0),
                    "cost_estimate": result.get("cost_estimate", {}),
                    "script": req.script,
                }
                saas_jobs[job_id]["logs"].append("Video generation completed!")

                # Upload to public gallery (non-blocking)
                try:
                    gallery_meta = {
                        "title": req.script.get("title", "Untitled"),
                        "hook_text": req.script.get("hook_text", ""),
                        "caption": req.script.get("caption", ""),
                        "hashtags": req.script.get("hashtags", []),
                        "full_narration": req.script.get("full_narration", ""),
                        "actor_description": req.script.get("actor_description", ""),
                        "style": req.script.get("style", "ugc"),
                        "language": req.script.get("language", "en"),
                        "duration": result.get("duration", 0),
                        "video_mode": req.video_mode,
                        "product_name": req.script.get("_product_name", ""),
                        "product_url": req.script.get("_product_url", ""),
                        "segments": req.script.get("segments", []),
                        "cost_estimate": result.get("cost_estimate", {}),
                    }
                    gallery_result = upload_video_to_gallery(
                        video_path=result["video_path"],
                        actor_image_path=result.get("actor_image", ""),
                        metadata=gallery_meta,
                        video_id=job_id[:8],
                    )
                    if gallery_result:
                        saas_jobs[job_id]["result"]["gallery_video_id"] = gallery_result["video_id"]
                        log_msg("📤 Uploaded to public gallery.")
                except Exception as gallery_err:
                    log_msg(f"⚠️ Gallery upload skipped: {gallery_err}")

        except Exception as e:
            print(f"[SaaSShorts] ❌ Job {job_id} failed: {e}")
            if job_id in saas_jobs:
                saas_jobs[job_id]["status"] = "failed"
                saas_jobs[job_id]["logs"].append(f"Error: {str(e)}")
        finally:
            concurrency_semaphore.release()

    asyncio.create_task(run_generation())

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/saasshorts/status/{job_id}")
async def saasshorts_status(job_id: str):
    """Poll SaaSShorts job status."""
    if job_id not in saas_jobs:
        raise HTTPException(status_code=404, detail="SaaSShorts job not found")

    job = saas_jobs[job_id]
    return {
        "status": job["status"],
        "logs": job["logs"],
        "result": job.get("result"),
    }


@app.get("/api/saasshorts/voices")
async def saasshorts_voices(
    x_elevenlabs_key: Optional[str] = Header(None, alias="X-ElevenLabs-Key"),
):
    """List available ElevenLabs voices."""
    if x_elevenlabs_key:
        try:
            loop = asyncio.get_event_loop()
            voices = await loop.run_in_executor(
                None, get_elevenlabs_voices, x_elevenlabs_key
            )
            if voices:
                return {"voices": voices, "source": "elevenlabs"}
        except Exception:
            pass

    # Fallback to default voices
    return {
        "voices": [
            {"voice_id": vid, "name": name, "category": "default"}
            for name, vid in DEFAULT_VOICES.items()
        ],
        "source": "defaults",
    }
