import asyncio
import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

import httpx
import yt_dlp

from commentary import _call_openai_compatible_chat
from main import transcribe_video


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(PROJECT_ROOT, ".local")
DOUYIN_COOKIES_PATH = os.path.join(LOCAL_DIR, "douyin_cookies.txt")
COMMENTARY_STYLES_PATH = os.path.join(LOCAL_DIR, "commentary_styles.json")
COMMENTARY_STYLE_PREFIX = "custom:"
STYLE_PROMPT_MAX_CHARS = 5000

DOUYIN_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
DOUYIN_WEB_API_HOST = "https://www.douyin.com"
DOUYIN_POST_API_PATH = "/aweme/v1/web/aweme/post/"
DOUYIN_DETAIL_API_PATH = "/aweme/v1/web/aweme/detail/"
DOUYIN_ALLOWED_COOKIE_DOMAINS = (
    "douyin.com",
    "iesdouyin.com",
)

STYLE_STORAGE_LOCK = threading.RLock()


class StyleLearningCancelled(Exception):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _raise_if_cancelled(cancel_event=None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise StyleLearningCancelled("Commentary style learning job was cancelled.")


def parse_douyin_user_url(profile_url: str) -> str:
    value = str(profile_url or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Douyin profile URL must start with http:// or https://")
    host = parsed.netloc.lower()
    if host not in {"douyin.com", "www.douyin.com"}:
        raise ValueError("Only douyin.com user profile URLs are supported")
    match = re.match(r"^/user/([^/?#]+)", parsed.path or "")
    if not match:
        raise ValueError("Douyin profile URL must look like https://www.douyin.com/user/<sec_uid>")
    sec_uid = match.group(1).strip()
    if not sec_uid:
        raise ValueError("Douyin profile URL is missing sec_uid")
    return sec_uid


def _is_allowed_douyin_cookie_domain(domain: str) -> bool:
    value = str(domain or "").strip().lower().lstrip(".")
    return any(value == allowed or value.endswith(f".{allowed}") for allowed in DOUYIN_ALLOWED_COOKIE_DOMAINS)


def normalize_douyin_cookies(cookies: str) -> str:
    normalized_lines = []
    for raw_line in str(cookies or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if not normalized_lines:
                normalized_lines.append("# Netscape HTTP Cookie File")
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            parts = line.split(None, 6)
        if len(parts) < 7:
            continue
        domain = parts[0].lower()
        if not _is_allowed_douyin_cookie_domain(domain):
            continue
        normalized_lines.append("\t".join(parts[:7]))
    return "\n".join(normalized_lines).strip()


def inspect_douyin_cookies(cookies: str) -> Dict:
    names = set()
    domains = set()
    rows = 0
    for line in normalize_douyin_cookies(cookies).splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            rows += 1
            domains.add(parts[0])
            names.add(parts[5])
    login_markers = {"sessionid", "sid_tt", "passport_csrf_token", "s_v_web_id", "ttwid", "msToken"}
    return {
        "rows": rows,
        "domains": sorted(domains),
        "has_login_cookies": bool(names & login_markers),
        "found_login_cookies": sorted(names & login_markers),
        "missing_warning": not bool(names & login_markers),
    }


def write_text_file(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text or "").rstrip() + "\n")


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def cookie_header_from_netscape_file(cookie_path: str) -> str:
    if not cookie_path or not os.path.exists(cookie_path):
        return ""
    pairs = []
    for raw_line in read_text_file(cookie_path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            pairs.append(f"{parts[5]}={parts[6]}")
    return "; ".join(pairs)


def _int_value(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _first_value(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _find_aweme_lists(value) -> List[List[Dict]]:
    lists = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"aweme_list", "aweme_list_v2", "awemeList"} and isinstance(child, list):
                lists.append(child)
            else:
                lists.extend(_find_aweme_lists(child))
    elif isinstance(value, list):
        if value and all(isinstance(item, dict) and (item.get("aweme_id") or item.get("awemeId")) for item in value[:3]):
            lists.append(value)
        else:
            for child in value:
                lists.extend(_find_aweme_lists(child))
    return lists


def _recursive_first(value, keys: set):
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for child in value.values():
            found = _recursive_first(child, keys)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for child in value:
            found = _recursive_first(child, keys)
            if found not in (None, ""):
                return found
    return None


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _find_chrome_executable() -> Optional[str]:
    candidates = [
        os.environ.get("OPENSHORTS_CHROME_PATH"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("msedge"),
        shutil.which("msedge.exe"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _douyin_public_capture_script() -> str:
    return r"""
(() => {
  window.__openshortsDouyinPostResponses = [];
  const save = (url, body, source, status) => {
    try {
      const value = String(url || "");
      if (value.includes("/aweme/v1/web/aweme/post/") || value.includes("/aweme/v1/web/aweme/detail/")) {
        window.__openshortsDouyinPostResponses.push({
          url: value,
          body: String(body || ""),
          source,
          status,
          at: Date.now()
        });
      }
    } catch (error) {}
  };

  const originalFetch = window.fetch;
  if (originalFetch && !window.__openshortsFetchPatched) {
    window.__openshortsFetchPatched = true;
    window.fetch = async function(...args) {
      const response = await originalFetch.apply(this, args);
      try {
        const url = response && response.url || args[0];
        if (String(url || "").includes("/aweme/v1/web/aweme/post/") || String(url || "").includes("/aweme/v1/web/aweme/detail/")) {
          response.clone().text().then((text) => save(url, text, "fetch", response.status)).catch(() => {});
        }
      } catch (error) {}
      return response;
    };
  }

  const proto = window.XMLHttpRequest && window.XMLHttpRequest.prototype;
  if (proto && !proto.__openshortsXhrPatched) {
    proto.__openshortsXhrPatched = true;
    const originalOpen = proto.open;
    const originalSend = proto.send;
    proto.open = function(method, url, ...rest) {
      this.__openshortsUrl = url;
      return originalOpen.call(this, method, url, ...rest);
    };
    proto.send = function(...args) {
      this.addEventListener("load", function() {
        try {
          const url = this.responseURL || this.__openshortsUrl;
          if (String(url || "").includes("/aweme/v1/web/aweme/post/") || String(url || "").includes("/aweme/v1/web/aweme/detail/")) {
            save(url, this.responseText, "xhr", this.status);
          }
        } catch (error) {}
      });
      return originalSend.apply(this, args);
    };
  }
})();
""".strip()


class _ChromeDevToolsClient:
    def __init__(self, websocket):
        self.websocket = websocket
        self._next_id = 1
        self._pending = {}
        self.post_responses: Dict[str, Dict] = {}
        self.finished_post_request_ids = []
        self.media_urls: List[str] = []

    async def send(self, method: str, params: Optional[Dict] = None) -> Dict:
        message_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[message_id] = future
        await self.websocket.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        return await future

    async def recv_loop(self):
        async for raw in self.websocket:
            message = json.loads(raw)
            if "id" in message:
                future = self._pending.pop(message["id"], None)
                if future and not future.done():
                    future.set_result(message)
                continue
            method = message.get("method")
            params = message.get("params") or {}
            if method == "Network.responseReceived":
                response = params.get("response") or {}
                url = response.get("url") or ""
                if DOUYIN_POST_API_PATH in url or DOUYIN_DETAIL_API_PATH in url:
                    request_id = params.get("requestId")
                    self.post_responses[request_id] = {
                        "url": url,
                        "status": response.get("status"),
                        "mimeType": response.get("mimeType"),
                    }
                elif (response.get("mimeType") or "").startswith("video/") and "douyinvod.com" in url:
                    self.media_urls.append(url)
            elif method == "Network.loadingFinished":
                request_id = params.get("requestId")
                if request_id in self.post_responses:
                    self.finished_post_request_ids.append(request_id)


def _extract_cdp_result_value(message: Dict):
    return (((message or {}).get("result") or {}).get("result") or {}).get("value")


async def _read_json_url(url: str, method: str = "GET") -> Dict:
    def request():
        request = urllib.request.Request(url, method=method)
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    return await asyncio.to_thread(request)


async def _wait_for_chrome_debug_port(port: int, timeout_seconds: int = 12) -> Dict:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            return await _read_json_url(f"http://127.0.0.1:{port}/json/version")
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.25)
    raise Exception(f"Chrome remote debugging did not become ready: {last_error}")


async def _collect_douyin_public_post_payloads(
    profile_url: str,
    progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
) -> List[Dict]:
    return await _collect_douyin_public_payloads(
        page_url=profile_url,
        progress=progress,
        cancel_event=cancel_event,
        initial_wait_seconds=int(os.environ.get("OPENSHORTS_DOUYIN_CAPTURE_SECONDS", "12")),
        scroll_rounds=int(os.environ.get("OPENSHORTS_DOUYIN_SCROLL_ROUNDS", "18")),
        open_message="Opening public Douyin profile in Chrome...",
    )


async def _collect_douyin_public_payloads(
    page_url: str,
    progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
    initial_wait_seconds: int = 12,
    scroll_rounds: int = 0,
    open_message: str = "Opening public Douyin page in Chrome...",
) -> List[Dict]:
    try:
        import websockets
    except Exception as exc:
        raise Exception("Python package 'websockets' is required for public Douyin browser capture.") from exc

    chrome_path = _find_chrome_executable()
    if not chrome_path:
        raise Exception("Chrome or Edge was not found. Install Chrome or set OPENSHORTS_CHROME_PATH.")

    port = int(os.environ.get("OPENSHORTS_CHROME_DEBUG_PORT") or _find_available_port())
    user_data_dir = tempfile.mkdtemp(prefix="openshorts_douyin_chrome_")
    page_url = str(page_url or "").strip()
    chrome_args = [
        chrome_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
        "--disable-blink-features=AutomationControlled",
        "--window-position=-32000,-32000",
        "--window-size=1200,900",
        "--no-first-run",
        "--no-default-browser-check",
        "about:blank",
    ]
    if os.environ.get("OPENSHORTS_DOUYIN_HEADLESS", "0").strip().lower() in {"1", "true", "yes"}:
        chrome_args.insert(-1, "--headless=new")
        chrome_args.insert(-1, "--disable-gpu")

    process = None
    try:
        if progress:
            progress(open_message)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            chrome_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        await _wait_for_chrome_debug_port(port)
        target = await _read_json_url(
            f"http://127.0.0.1:{port}/json/new?{urllib.parse.quote(page_url, safe='')}",
            method="PUT",
        )
        websocket_url = target.get("webSocketDebuggerUrl")
        if not websocket_url:
            raise Exception("Chrome did not expose a debuggable page target.")

        payloads = []
        async with websockets.connect(websocket_url, max_size=100_000_000) as websocket:
            client = _ChromeDevToolsClient(websocket)
            recv_task = asyncio.create_task(client.recv_loop())
            try:
                await client.send("Page.enable")
                await client.send("Runtime.enable")
                await client.send(
                    "Network.enable",
                    {"maxTotalBufferSize": 100_000_000, "maxResourceBufferSize": 50_000_000},
                )
                await client.send("Page.addScriptToEvaluateOnNewDocument", {"source": _douyin_public_capture_script()})
                await client.send("Page.navigate", {"url": page_url})

                started = time.time()
                while time.time() - started < initial_wait_seconds:
                    _raise_if_cancelled(cancel_event)
                    await asyncio.sleep(0.5)

                for _ in range(max(0, scroll_rounds)):
                    _raise_if_cancelled(cancel_event)
                    await client.send(
                        "Runtime.evaluate",
                        {
                            "expression": "window.scrollBy(0, Math.max(1000, window.innerHeight)); 0",
                            "returnByValue": True,
                        },
                    )
                    await asyncio.sleep(0.8)
                await asyncio.sleep(2)

                captured_message = await client.send(
                    "Runtime.evaluate",
                    {
                        "expression": "JSON.stringify(window.__openshortsDouyinPostResponses || [])",
                        "returnByValue": True,
                    },
                )
                captured = json.loads(_extract_cdp_result_value(captured_message) or "[]")
                for item in captured:
                    body = item.get("body") or ""
                    if not body:
                        continue
                    try:
                        payloads.append(json.loads(body))
                    except Exception:
                        continue

                for request_id in client.finished_post_request_ids:
                    info = client.post_responses.get(request_id) or {}
                    if info.get("status") != 200:
                        continue
                    try:
                        body_message = await client.send("Network.getResponseBody", {"requestId": request_id})
                        body = (body_message.get("result") or {}).get("body") or ""
                        if body:
                            payloads.append(json.loads(body))
                    except Exception:
                        continue
                for media_url in client.media_urls:
                    payloads.append({"__openshorts_media_url": media_url})
            finally:
                recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await recv_task
        return payloads
    finally:
        if process:
            with contextlib.suppress(Exception):
                process.terminate()
                process.wait(timeout=3)
            with contextlib.suppress(Exception):
                process.kill()
        shutil.rmtree(user_data_dir, ignore_errors=True)


def _public_douyin_payloads_to_videos(payloads: List[Dict]) -> List[Dict]:
    videos_by_id = {}
    for payload in payloads or []:
        aweme_lists = _find_aweme_lists(payload)
        for aweme_list in aweme_lists:
            for raw_item in aweme_list:
                item = normalize_douyin_aweme(raw_item)
                if not item:
                    continue
                aweme_id = item["aweme_id"]
                videos_by_id[aweme_id] = _merge_douyin_video_item(videos_by_id.get(aweme_id), item)
    return list(videos_by_id.values())


def _has_douyin_direct_media(item: Dict) -> bool:
    return bool(str((item or {}).get("direct_audio_url") or "").strip() or str((item or {}).get("direct_video_url") or "").strip())


def _merge_douyin_video_item(existing: Optional[Dict], incoming: Dict) -> Dict:
    if not existing:
        return dict(incoming or {})
    if not incoming:
        return dict(existing)
    merged = dict(existing)
    existing_media = _has_douyin_direct_media(existing)
    incoming_media = _has_douyin_direct_media(incoming)

    for key, value in incoming.items():
        if value not in (None, "", [], {}):
            if key not in merged or merged.get(key) in (None, "", [], {}):
                merged[key] = value

    if incoming_media and not existing_media:
        for key, value in incoming.items():
            if value not in (None, "", [], {}):
                merged[key] = value

    existing_score = _int_value(existing.get("rank_score"), _int_value(existing.get("like_count")) + _int_value(existing.get("save_count")))
    incoming_score = _int_value(incoming.get("rank_score"), _int_value(incoming.get("like_count")) + _int_value(incoming.get("save_count")))
    if incoming_score >= existing_score:
        for key in ("like_count", "save_count", "rank_score", "timestamp", "title", "author_name", "metadata_partial"):
            value = incoming.get(key)
            if value not in (None, "", [], {}):
                merged[key] = value
    merged["rank_score"] = _int_value(merged.get("rank_score"), _int_value(merged.get("like_count")) + _int_value(merged.get("save_count")))
    return merged


def _extract_aweme_detail_items(payload: Dict) -> List[Dict]:
    if not isinstance(payload, dict):
        return []
    candidates = []
    for key in ("aweme_detail", "aweme_detail_info", "aweme", "item"):
        value = payload.get(key)
        if isinstance(value, dict) and (value.get("aweme_id") or value.get("awemeId")):
            candidates.append(value)
    for aweme_list in _find_aweme_lists(payload):
        candidates.extend(aweme_list)
    return candidates


async def _fetch_douyin_public_aweme_detail_payloads(
    aweme_id: str,
    progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
) -> List[Dict]:
    aweme_id = str(aweme_id or "").strip()
    if not aweme_id:
        return []
    return await _collect_douyin_public_payloads(
        page_url=f"https://www.douyin.com/video/{aweme_id}",
        progress=progress,
        cancel_event=cancel_event,
        initial_wait_seconds=int(os.environ.get("OPENSHORTS_DOUYIN_DETAIL_CAPTURE_SECONDS", "14")),
        scroll_rounds=0,
        open_message=f"Opening public Douyin video detail in Chrome: {aweme_id}",
    )


def fetch_douyin_public_aweme_detail(
    video: Dict,
    progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
) -> Optional[Dict]:
    aweme_id = str((video or {}).get("aweme_id") or "").strip()
    if not aweme_id:
        return None
    payloads = asyncio.run(_fetch_douyin_public_aweme_detail_payloads(aweme_id, progress=progress, cancel_event=cancel_event))
    best = None
    for payload in payloads:
        media_url = str((payload or {}).get("__openshorts_media_url") or "").strip()
        if media_url and (aweme_id in media_url or "__vid=" in media_url):
            best = _merge_douyin_video_item(best, {
                "aweme_id": aweme_id,
                "video_url": f"https://www.douyin.com/video/{aweme_id}",
                "direct_video_url": media_url,
            })
        for raw in _extract_aweme_detail_items(payload):
            item = normalize_douyin_aweme(raw)
            if item and item.get("aweme_id") == aweme_id:
                best = _merge_douyin_video_item(best, item)
    return best


def ensure_douyin_direct_media(
    video: Dict,
    progress: Optional[Callable[[str], None]] = None,
    cancel_event=None,
) -> Dict:
    if _has_douyin_direct_media(video):
        return video
    aweme_id = str((video or {}).get("aweme_id") or "").strip()
    if progress:
        progress(f"Fetching public Douyin video detail for media URL: {aweme_id}")
    try:
        detail = fetch_douyin_public_aweme_detail(video, progress=progress, cancel_event=cancel_event)
    except StyleLearningCancelled:
        raise
    except Exception as exc:
        if progress:
            progress(f"Public Douyin detail did not return media for {aweme_id}: {str(exc)[:240]}")
        return video
    if detail and _has_douyin_direct_media(detail):
        return _merge_douyin_video_item(video, detail)
    return video


def _url_list_from_douyin_addr(value) -> List[str]:
    if not isinstance(value, dict):
        return []
    urls = value.get("url_list") or value.get("urlList") or []
    if not isinstance(urls, list):
        return []
    return [str(url or "").strip() for url in urls if str(url or "").strip().startswith("http")]


def _extract_douyin_direct_video_url(raw: Dict) -> str:
    video = raw.get("video") if isinstance(raw.get("video"), dict) else {}
    candidates = []
    bitrates = [
        item for item in (video.get("bit_rate") if isinstance(video.get("bit_rate"), list) else [])
        if isinstance(item, dict)
    ]
    bitrates = sorted(
        bitrates,
        key=lambda item: (
            0 if str(item.get("format") or "").lower() == "mp4" else 1,
            _int_value(item.get("is_h265")) + _int_value(item.get("is_bytevc1")),
            _int_value(item.get("bit_rate"), 10**12),
        ),
    )
    for bitrate in bitrates:
        candidates.extend(_url_list_from_douyin_addr(bitrate.get("play_addr")))
    for key in ("play_addr_h264", "play_addr_265", "play_addr"):
        candidates.extend(_url_list_from_douyin_addr(video.get(key)))
    for url in candidates:
        if "media-video-hvc1" in url or "/dash/" in url:
            continue
        if "douyinvod.com" in url or "mime_type=video_mp4" in url:
            return url
    return candidates[0] if candidates else ""


def _extract_douyin_direct_audio_url(raw: Dict) -> str:
    video = raw.get("video") if isinstance(raw.get("video"), dict) else {}
    entries = video.get("bit_rate_audio") if isinstance(video.get("bit_rate_audio"), list) else []
    candidates = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        audio_meta = entry.get("audio_meta") if isinstance(entry.get("audio_meta"), dict) else {}
        url_data = audio_meta.get("url_list")
        if isinstance(url_data, dict):
            for key in ("main_url", "backup_url", "fallback_url"):
                url = str(url_data.get(key) or "").strip()
                if url.startswith("http"):
                    candidates.append(url)
        elif isinstance(url_data, list):
            candidates.extend(str(url or "").strip() for url in url_data if str(url or "").strip().startswith("http"))
    for url in candidates:
        if "media-audio" in url or "audio" in url:
            return url
    return candidates[0] if candidates else ""


def normalize_douyin_aweme(raw: Dict) -> Optional[Dict]:
    if not isinstance(raw, dict):
        return None
    aweme_id = str(_first_value(raw.get("aweme_id"), raw.get("awemeId"), raw.get("id")) or "").strip()
    if not aweme_id:
        return None
    stats = raw.get("statistics") if isinstance(raw.get("statistics"), dict) else {}
    author = raw.get("author") if isinstance(raw.get("author"), dict) else {}
    title = str(_first_value(raw.get("desc"), raw.get("title"), raw.get("caption")) or "").strip()
    like_count = _int_value(_first_value(stats.get("digg_count"), raw.get("digg_count"), raw.get("like_count")))
    save_count = _int_value(_first_value(stats.get("collect_count"), raw.get("collect_count"), raw.get("save_count")))
    timestamp = _int_value(_first_value(raw.get("create_time"), raw.get("createTime"), raw.get("timestamp")))
    direct_video_url = _extract_douyin_direct_video_url(raw)
    direct_audio_url = _extract_douyin_direct_audio_url(raw)
    return {
        "aweme_id": aweme_id,
        "video_url": f"https://www.douyin.com/video/{aweme_id}",
        "direct_video_url": direct_video_url,
        "direct_audio_url": direct_audio_url,
        "title": title,
        "like_count": like_count,
        "save_count": save_count,
        "rank_score": like_count + save_count,
        "timestamp": timestamp,
        "author_name": str(_first_value(author.get("nickname"), author.get("unique_id")) or "").strip(),
        "metadata_partial": not bool(stats),
    }


def rank_douyin_videos(videos: List[Dict], max_videos: int = 100) -> List[Dict]:
    unique = {}
    for raw in videos or []:
        item = normalize_douyin_aweme(raw) or raw
        aweme_id = str(item.get("aweme_id") or "").strip()
        if not aweme_id:
            continue
        item = {
            **item,
            "like_count": _int_value(item.get("like_count")),
            "save_count": _int_value(item.get("save_count")),
            "timestamp": _int_value(item.get("timestamp")),
        }
        item["rank_score"] = _int_value(item.get("rank_score"), item["like_count"] + item["save_count"])
        existing = unique.get(aweme_id)
        unique[aweme_id] = _merge_douyin_video_item(existing, item)
    ranked = sorted(
        unique.values(),
        key=lambda item: (-_int_value(item.get("rank_score")), -_int_value(item.get("timestamp")), str(item.get("aweme_id") or "")),
    )
    return [
        {
            **item,
            "rank_index": index + 1,
        }
        for index, item in enumerate(ranked[: max(1, int(max_videos or 100))])
    ]


class DouyinProfileProvider:
    def __init__(self, timeout_seconds: int = 30, page_size: int = 18, max_pages: Optional[int] = None):
        self.timeout_seconds = timeout_seconds
        self.page_size = page_size
        self.max_pages = max_pages or int(os.environ.get("OPENSHORTS_DOUYIN_MAX_PAGES", "200"))

    def fetch_videos(self, profile_url: str, cookie_path: str, progress: Optional[Callable[[str], None]] = None, cancel_event=None) -> List[Dict]:
        sec_uid = parse_douyin_user_url(profile_url)
        browser_error = None
        try:
            if progress:
                progress("Fetching public Douyin profile videos with browser capture...")
            payloads = asyncio.run(_collect_douyin_public_post_payloads(profile_url, progress=progress, cancel_event=cancel_event))
            videos = _public_douyin_payloads_to_videos(payloads)
            if videos:
                if progress:
                    progress(f"Fetched {len(videos)} public Douyin videos from the profile page.")
                return videos
        except StyleLearningCancelled:
            raise
        except Exception as exc:
            browser_error = exc
            if progress:
                progress(f"Public Douyin browser capture did not return videos: {str(exc)[:240]}")

        if not cookie_path or not os.path.exists(cookie_path):
            detail = f" Browser capture error: {browser_error}" if browser_error else ""
            raise Exception(
                "No visible Douyin videos were found from the public profile page. "
                "OpenShorts no longer requires Douyin cookies for list fetching, but this run could not read the public page data."
                f"{detail}"
            )
        cookie_header = cookie_header_from_netscape_file(cookie_path)
        if not cookie_header:
            detail = f" Browser capture error: {browser_error}" if browser_error else ""
            raise Exception(
                "No visible Douyin videos were found from the public profile page, and the optional Douyin cookies file is empty or invalid."
                f"{detail}"
            )

        headers = {
            "User-Agent": DOUYIN_USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": profile_url,
            "Cookie": cookie_header,
        }
        videos = []
        seen_ids = set()
        max_cursor = 0
        previous_cursor = None
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True, headers=headers) as client:
            client.get(profile_url)
            for page in range(1, self.max_pages + 1):
                _raise_if_cancelled(cancel_event)
                query = {
                    "device_platform": "webapp",
                    "aid": "6383",
                    "channel": "channel_pc_web",
                    "sec_user_id": sec_uid,
                    "max_cursor": str(max_cursor),
                    "count": str(self.page_size),
                    "publish_video_strategy_type": "2",
                    "pc_client_type": "1",
                    "version_code": "170400",
                    "version_name": "17.4.0",
                    "cookie_enabled": "true",
                    "platform": "PC",
                    "downlink": "10",
                }
                if progress:
                    progress(f"Fetching Douyin video list page {page}...")
                response = client.get(f"{DOUYIN_WEB_API_HOST}/aweme/v1/web/aweme/post/", params=query)
                if response.status_code >= 400:
                    raise Exception(
                        "Douyin video list request failed. Fresh logged-in Douyin cookies may be required. "
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                try:
                    data = response.json()
                except Exception as exc:
                    raise Exception(f"Douyin returned a non-JSON video list response: {response.text[:500]}") from exc

                aweme_lists = _find_aweme_lists(data)
                page_items = []
                for aweme_list in aweme_lists:
                    for raw_item in aweme_list:
                        item = normalize_douyin_aweme(raw_item)
                        if not item or item["aweme_id"] in seen_ids:
                            continue
                        seen_ids.add(item["aweme_id"])
                        page_items.append(item)
                videos.extend(page_items)

                has_more = bool(_recursive_first(data, {"has_more", "hasMore"}))
                next_cursor = _recursive_first(data, {"max_cursor", "maxCursor", "cursor"})
                next_cursor = _int_value(next_cursor, max_cursor)
                if progress:
                    progress(f"Fetched {len(videos)} unique Douyin videos so far.")
                if not has_more or not page_items or next_cursor == previous_cursor:
                    break
                previous_cursor = max_cursor
                max_cursor = next_cursor
        if not videos:
            detail = f" Browser capture error: {browser_error}" if browser_error else ""
            raise Exception(f"No visible Douyin videos were found for this profile.{detail}")
        return videos


def download_douyin_audio(video: Dict, output_dir: str, cookie_path: str, cancel_event=None) -> Dict:
    _raise_if_cancelled(cancel_event)
    os.makedirs(output_dir, exist_ok=True)
    aweme_id = str(video.get("aweme_id") or uuid.uuid4().hex)
    output_template = os.path.join(output_dir, f"{aweme_id}.%(ext)s")
    source_url = str(video.get("direct_audio_url") or "").strip() or str(video.get("direct_video_url") or "").strip() or video["video_url"]
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "cookiefile": cookie_path if cookie_path and os.path.exists(cookie_path) else None,
        "http_headers": {
            "User-Agent": DOUYIN_USER_AGENT,
            "Referer": str(video.get("video_url") or "https://www.douyin.com/"),
        },
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "overwrites": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 30,
        "nocheckcertificate": True,
        "cachedir": False,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(source_url, download=True)
        prepared = ydl.prepare_filename(info)
    candidates = [
        os.path.splitext(prepared)[0] + ".mp3",
        prepared,
    ]
    candidates.extend(
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith(f"{aweme_id}.")
    )
    audio_path = next((path for path in candidates if os.path.exists(path) and os.path.getsize(path) > 0), None)
    if not audio_path:
        raise Exception(f"yt-dlp did not produce an audio file for Douyin video {aweme_id}")
    return {
        "audio_path": audio_path,
        "title": info.get("title") or video.get("title") or aweme_id,
        "duration": info.get("duration"),
        "like_count": _int_value(info.get("like_count"), _int_value(video.get("like_count"))),
        "save_count": _int_value(info.get("save_count"), _int_value(video.get("save_count"))),
    }


def _parse_json_object(text: str) -> Dict:
    raw = str(text or "").strip()
    if not raw:
        raise Exception("AI returned empty response")
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise
        return json.loads(match.group(0))


def _compact_text(text: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def _sanitize_style_prompt(prompt: str, profile_url: str = "", max_chars: int = STYLE_PROMPT_MAX_CHARS) -> str:
    value = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"\b\d{13,}\b", "", value)
    value = re.sub(r"MS4wLjAB[0-9A-Za-z_\-]+", "", value)
    if profile_url:
        value = value.replace(profile_url, "")
    lines = []
    for raw_line in value.split("\n"):
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    value = "\n".join(lines).strip()
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value[: max(1, int(max_chars or STYLE_PROMPT_MAX_CHARS))].rstrip()


def _normalize_style_label(label: str) -> str:
    value = re.sub(r"\s+", " ", str(label or "")).strip()
    return value[:40] or "Douyin learned commentary style"


def _style_language_name(language: str) -> str:
    value = str(language or "").strip().lower()
    return {
        "zh": "中文",
        "zh-cn": "中文",
        "cn": "中文",
        "en": "English",
        "es": "Español",
        "ja": "日本語",
        "jp": "日本語",
    }.get(value, value or "中文")


def _default_style_label(language: str) -> str:
    value = str(language or "").strip().lower()
    if value in {"zh", "zh-cn", "cn"}:
        return "抖音学习解说风格"
    if value == "es":
        return "Estilo aprendido de Douyin"
    if value in {"ja", "jp"}:
        return "Douyin学習ナレーションスタイル"
    return "Douyin learned commentary style"


def normalize_style_record(style: Dict) -> Dict:
    now = utc_now_iso()
    label = _normalize_style_label(style.get("label") or style.get("name"))
    prompt = _sanitize_style_prompt(style.get("prompt") or style.get("custom_style_prompt") or "")
    if not prompt:
        raise ValueError("Commentary style prompt is empty")
    style_id = str(style.get("id") or "").strip()
    if not style_id.startswith(COMMENTARY_STYLE_PREFIX):
        style_id = f"{COMMENTARY_STYLE_PREFIX}learned-{uuid.uuid4().hex[:12]}"
    return {
        "id": style_id,
        "label": label,
        "prompt": prompt,
        "custom": True,
        "source": style.get("source") or "douyin_style_learning",
        "summary": _compact_text(style.get("summary") or "", 1200),
        "style_traits": style.get("style_traits") if isinstance(style.get("style_traits"), list) else [],
        "metadata": style.get("metadata") if isinstance(style.get("metadata"), dict) else {},
        "created_at": style.get("created_at") or now,
        "updated_at": now,
    }


def list_commentary_styles(storage_path: str = COMMENTARY_STYLES_PATH) -> List[Dict]:
    if not os.path.exists(storage_path):
        return []
    with STYLE_STORAGE_LOCK:
        try:
            with open(storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
    raw_styles = data.get("styles") if isinstance(data, dict) else data
    if not isinstance(raw_styles, list):
        return []
    styles = []
    for item in raw_styles:
        try:
            styles.append(normalize_style_record(item))
        except Exception:
            continue
    return styles


def save_commentary_style(style: Dict, storage_path: str = COMMENTARY_STYLES_PATH) -> Dict:
    record = normalize_style_record(style)
    with STYLE_STORAGE_LOCK:
        existing = list_commentary_styles(storage_path)
        label_key = record["label"].lower()
        merged = [item for item in existing if item.get("id") != record["id"] and str(item.get("label") or "").lower() != label_key]
        merged.append(record)
        merged = merged[-200:]
        os.makedirs(os.path.dirname(os.path.abspath(storage_path)), exist_ok=True)
        tmp_path = f"{storage_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"styles": merged}, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, storage_path)
    return record


def delete_commentary_style(style_id: str, storage_path: str = COMMENTARY_STYLES_PATH) -> Dict:
    target = str(style_id or "").strip()
    if not target:
        return {"deleted": False}
    with STYLE_STORAGE_LOCK:
        existing = list_commentary_styles(storage_path)
        remaining = [item for item in existing if item.get("id") != target]
        deleted = len(remaining) != len(existing)
        os.makedirs(os.path.dirname(os.path.abspath(storage_path)), exist_ok=True)
        tmp_path = f"{storage_path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"styles": remaining}, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, storage_path)
    return {"deleted": deleted, "style_id": target}


def _style_analysis_chat(openai_config: Dict) -> Callable:
    def call(messages, max_tokens=2000, response_format=None, timeout_seconds=None):
        return _call_openai_compatible_chat(
            api_key=openai_config["api_key"],
            base_url=openai_config["base_url"],
            model=openai_config["model"],
            messages=messages,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            response_format=response_format,
        )
    return call


def _has_openai_style_config(openai_config: Optional[Dict]) -> bool:
    config = openai_config or {}
    return all(str(config.get(key) or "").strip() for key in ("api_key", "base_url", "model"))


def summarize_single_video_style(video: Dict, transcript: Dict, openai_chat: Callable) -> Dict:
    text = _compact_text(transcript.get("text") or "", 12000)
    prompt = f"""
你是短视频解说风格分析师。分析下面一条视频的完整解说转写，目标是提炼“可迁移到新视频上的说话方式”，不是复述这条视频的题材。

分析边界：
- 只抽取表达机制：叙述人设、开头钩子、句式节奏、画面证据顺序、解释深度、情绪强度、转折方式、常用语气。
- 不要写账号身份、视频链接、视频 ID、具体人物、具体产品、固定地名、固定剧情或可识别原文。
- 可以概括“句式模式”和“语气功能”，但不要复制原作者长句、完整口头禅或连续 8 个字以上的原句。
- 如果转写太短、像歌词/字幕噪声、或没有明显解说风格，把 usable 设为 false 并说明原因。

返回 JSON：
{{
  "usable": true,
  "coverage": "这条转写对风格学习是否充分",
  "summary": "这条转写体现出的风格摘要",
  "voice_persona": "叙述者像什么角色/视角，用什么距离感说话",
  "script_structure": "从开头到结尾的组织方式",
  "opening_hook": "开头通常如何抓人，抽象成模式",
  "visual_grounding": "如何先说画面证据、动作、物体和变化",
  "sentence_rhythm": "句长、停顿、短句/长句比例、推进方式",
  "pacing": "信息密度、语速感、每句话承担的功能",
  "rhetoric": "常见修辞和转折方式，不要照抄原句",
  "word_choice": "词汇偏好、语气词、判断词、动作词类型，必须抽象描述",
  "emotion": "情绪强度和态度",
  "do_rules": ["复用时必须保留的写法规则"],
  "dont_rules": ["复用时必须避免的写法规则"],
  "template": "可迁移到新画面的单段解说公式"
}}

视频标题或描述（仅用于理解，不要写进风格）：{_compact_text(video.get("title") or "", 180)}
完整转写：
{text}
""".strip()
    response = openai_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=2600,
        response_format={"type": "json_object"},
        timeout_seconds=180,
    )
    parsed = _parse_json_object(response)
    parsed["aweme_id"] = video.get("aweme_id")
    parsed["rank_index"] = video.get("rank_index")
    return parsed


def aggregate_style_summaries(
    summaries: List[Dict],
    openai_chat: Callable,
    style_name: str = "",
    profile_url: str = "",
    video_count: int = 0,
    transcript_count: int = 0,
    language: str = "zh",
) -> Dict:
    compact_summaries = [
        {
            "rank_index": item.get("rank_index"),
            "coverage": _compact_text(item.get("coverage") or "", 160),
            "summary": _compact_text(item.get("summary") or "", 650),
            "voice_persona": _compact_text(item.get("voice_persona") or "", 260),
            "script_structure": _compact_text(item.get("script_structure") or "", 360),
            "opening_hook": _compact_text(item.get("opening_hook") or "", 260),
            "visual_grounding": _compact_text(item.get("visual_grounding") or "", 360),
            "sentence_rhythm": _compact_text(item.get("sentence_rhythm") or "", 300),
            "traits": item.get("traits") if isinstance(item.get("traits"), list) else [],
            "pacing": _compact_text(item.get("pacing") or "", 320),
            "rhetoric": _compact_text(item.get("rhetoric") or "", 320),
            "word_choice": _compact_text(item.get("word_choice") or "", 320),
            "emotion": _compact_text(item.get("emotion") or "", 240),
            "do_rules": item.get("do_rules") if isinstance(item.get("do_rules"), list) else [],
            "dont_rules": item.get("dont_rules") if isinstance(item.get("dont_rules"), list) else [],
            "template": _compact_text(item.get("template") or "", 360),
        }
        for item in summaries
        if item.get("usable", True) is not False
    ]
    if not compact_summaries:
        raise Exception("OpenAI did not produce any usable per-video style summaries.")

    chunk_summaries = compact_summaries
    if len(compact_summaries) > 25:
        chunk_summaries = []
        for index in range(0, len(compact_summaries), 20):
            chunk = compact_summaries[index:index + 20]
            response = openai_chat(
                [{
                    "role": "user",
                    "content": (
                        "把这些单条视频解说风格摘要压缩成一个批次风格画像。只保留跨视频反复出现的通用说话方式，"
                        "不要包含视频 ID、链接、账号身份、固定人物或具体题材。返回 JSON: "
                        '{"summary":"...", "voice_persona":"...", "script_structure":"...", "opening_hook":"...", '
                        '"visual_grounding":"...", "sentence_rhythm":"...", "pacing":"...", "rhetoric":"...", '
                        '"word_choice":"...", "emotion":"...", "do_rules":["..."], "dont_rules":["..."], "template":"..."}\n'
                        f"{json.dumps(chunk, ensure_ascii=False)}"
                    ),
                }],
                max_tokens=2400,
                response_format={"type": "json_object"},
                timeout_seconds=180,
            )
            chunk_summaries.append(_parse_json_object(response))

    desired_label = _normalize_style_label(style_name or _default_style_label(language))
    target_language = _style_language_name(language)
    response = openai_chat(
        [{
            "role": "user",
            "content": f"""
你是解说风格提示词设计师。基于多条视频的风格摘要，生成一个可直接用于 OpenShorts 自定义解说风格的通用提示词。
这个提示词要让另一个 AI 在新视频上尽量复刻同一种“解说说话方式”：像同一个类型的解说员在说话，而不是只写一句风格标签。

输出语言要求：
- label、prompt、summary、style_traits 和 warnings 必须使用 {target_language} 输出。
- 不要因为原始转写或摘要里出现其他语言，就切换输出语言。

硬性要求：
- 输出必须是通用风格，不要写账号名、平台链接、视频 ID、具体人物、具体产品、具体场景或固定题材。
- 不要复制原作者长句、完整口头禅或可识别文案；只抽象成节奏、句式、视角、结构、情绪、信息密度。
- prompt 必须完整，可直接保存到 custom_style_prompt 并用于生产新解说。
- prompt 必须写成“给解说生成模型的操作指令”，不要写成分析报告。
- prompt 必须明确要求先描述画面证据，不编造画面外剧情；任何判断、反转、情绪和解释都要来自当前画面或转写证据。
- prompt 建议 1800 到 4200 个字符之间；不要为了短而丢掉可复刻细节。

prompt 至少包含这些小节，使用清晰标题：
1. 核心风格定位：叙述者人设、观看距离、整体气质。
2. 画面优先规则：每段如何从可见动作/物体/变化开始。
3. 结构模板：开头钩子、主体推进、转折、收束分别怎么写。
4. 句式和节奏：短句/长句比例、停顿、信息密度、每句功能。
5. 用词和语气：动作词、判断词、疑问/感叹/反差的使用边界。
6. 复刻公式：给出 3 到 5 条可套用的抽象句式模板，模板只能用占位符，不能复制原句。
7. 禁止事项：禁止账号信息、固定题材、无证据剧情、照抄原文、泛泛鸡汤。

返回 JSON：
{{
  "label": "{desired_label}",
  "prompt": "可直接保存到 custom_style_prompt 的{target_language}提示词",
  "summary": "风格摘要",
  "style_traits": ["通用特征1", "通用特征2"],
  "confidence": "high|medium|low",
  "warnings": []
}}

视频数量：{video_count}
成功转写数量：{transcript_count}
风格摘要：
{json.dumps(chunk_summaries, ensure_ascii=False)}
""".strip(),
        }],
        max_tokens=5000,
        response_format={"type": "json_object"},
        timeout_seconds=240,
    )
    parsed = _parse_json_object(response)
    parsed["label"] = _normalize_style_label(parsed.get("label") or desired_label)
    parsed["prompt"] = _sanitize_style_prompt(parsed.get("prompt") or "", profile_url)
    parsed["source"] = "douyin_style_learning"
    parsed["metadata"] = {
        "source_platform": "douyin",
        "language": language or "zh",
        "video_count": video_count,
        "transcript_count": transcript_count,
        "confidence": parsed.get("confidence") or "medium",
        "warnings": parsed.get("warnings") if isinstance(parsed.get("warnings"), list) else [],
    }
    return parsed


def _dedupe_failed_videos(failures: List[Dict], successful_aweme_ids: Optional[set] = None) -> List[Dict]:
    successful = {str(item or "").strip() for item in (successful_aweme_ids or set()) if str(item or "").strip()}
    deduped = {}
    for item in failures or []:
        if not isinstance(item, dict):
            continue
        aweme_id = str(item.get("aweme_id") or "").strip()
        if aweme_id and aweme_id in successful:
            continue
        stage = str(item.get("stage") or "unknown").strip() or "unknown"
        key = (aweme_id, stage)
        deduped[key] = {
            "aweme_id": aweme_id,
            "stage": stage,
            "error": str(item.get("error") or "")[:800],
        }
    return list(deduped.values())


def _split_commentary_units(text: str) -> List[str]:
    units = []
    for chunk in re.split(r"[。！？!?；;，,\n]+|\s{2,}", str(text or "")):
        value = re.sub(r"\s+", " ", chunk).strip()
        if len(value) >= 2:
            units.append(value)
    return units


def _count_terms(text: str, terms: List[str]) -> List[str]:
    counts = Counter()
    for term in terms:
        count = str(text or "").count(term)
        if count:
            counts[term] = count
    return [term for term, _ in counts.most_common(10)]


def _infer_local_style_position(all_text: str, titles_text: str, language: str) -> str:
    value = str(language or "").strip().lower()
    combined = f"{titles_text}\n{all_text}"
    documentary_hits = sum(combined.count(term) for term in ("纪录片", "生活", "清晨", "山", "一家人", "回家", "孩子", "丈夫", "婆婆", "收获"))
    process_hits = sum(combined.count(term) for term in ("准备", "随后", "接着", "最后", "开始", "继续", "带回", "发现", "找到"))
    if value in {"zh", "zh-cn", "cn"}:
        if documentary_hits >= 8:
            return "生活纪录片式中文短视频解说"
        if process_hits >= 8:
            return "流程推进式中文短视频解说"
        return "画面先行的中文短视频解说"
    if documentary_hits >= 8:
        return "documentary-style short-video narration"
    if process_hits >= 8:
        return "process-driven short-video narration"
    return "visual-first short-video narration"


def _build_local_chinese_style_prompt(style_position: str, recurring_terms: List[str]) -> str:
    term_hint = "、".join(recurring_terms[:6]) if recurring_terms else "人物动作、环境变化、材料处理、家庭处境、收获结果"
    return f"""
# 核心风格定位
你要模仿一种“{style_position}”的解说方式，但不要绑定任何固定账号、人物、地区、作品、产品或题材。叙述者像一个在现场旁观、熟悉生活流程的纪录片解说员，用平实、连续、带一点惊奇感的语气，把观众带进一个正在发生的真实过程里。整体不是喊麦、不是鸡汤、不是营销口播，也不是百科讲解；它更像耐心讲一个见闻：先看见动作，再解释处境，再点出结果、风险、压力或反差。

# 画面优先规则
每一段都必须先落到当前画面里真实可见的动作、人物位置、工具、动物、植物、道路、山坡、天气、容器、搬运过程、材料变化或表情反应。不要一上来空喊情绪，也不要编造画面外剧情。可以补充判断，但判断必须来自画面证据或用户提供的转写证据，例如“看起来收获不少”“这一步并不轻松”“家里还等着结果”。如果画面只展示动作，就写动作和动作造成的变化；如果画面展示人物关系，就写谁在做什么、谁在等待、谁受到影响。

# 结构模板
开头先用一个具体动作或处境抓人，不用泛泛介绍人物身份。常用结构是：可见动作/突发发现 -> 当前处境或目标 -> 连续动作推进 -> 中途转折或压力 -> 结果落点。主体段落要像跟着画面往前走，每一句承担一个功能：交代动作、补足原因、指出难度、承接下一步、给出克制判断。结尾不要强行升华，可以落在“带回了什么”“问题是否解决”“下一步还要面对什么”这类具体结果上。

# 开头写法
开头不要先喊“今天我们来看”或直接给宏大结论，而是把观众拉到一个正在发生的瞬间。优先从“人物已经在做的动作”“刚发现的东西”“眼前环境带来的压力”里选择一个钩子。开头可以带一点悬念，但悬念必须具体，例如某个收获不够、路程还很远、工具不好用、天气或环境增加难度。第一句不要超过一个判断，第二句再交代为什么这个动作值得看。不要用标题党式夸张词，也不要为了抓人而编出画面外冲突。

# 主体推进
主体要像跟拍一样按时间顺序推进。每切换一个画面，先说画面变化，再说动作目的；每出现一个结果，先说结果长什么样，再说它意味着什么。不要把所有信息堆在一句里，而是拆成连续的小步：谁在动、动了什么、为什么这样做、这一步带来什么变化。可以反复使用“先做一件小事，再发现新的麻烦，再继续处理”的链条，让解说有现场感。解释不要离画面太远，宁愿少讲一点，也不要把人物心理、家庭冲突、收入状况写成无证据事实。

# 转折和反差
这类风格的反差不是夸张反转，而是生活流程里的现实落差：辛苦走了很远但收获有限，准备很久但工具不顺手，发现好东西但还要背回去，表面动作简单但过程很耗体力。写转折时先给画面证据，再给克制判断。可以用“没想到、可问题是、但真正麻烦的是、这时候、原来”来转折，但每次转折都要对应一个可见变化。不要把普通动作硬写成惊天反转。

# 信息密度
每 10 秒左右的解说应至少包含一个可见动作或物体变化，不要连续多句空泛抒情。画面信息少时，可以扩写动作的目的、难度、顺序和后果；画面信息多时，优先选择最能推动流程的 1 到 2 个细节，不要流水账列清单。保持“动作密度高、解释适中、情绪低饱和”的比例。可以补充背景常识，但只能作为一句辅助解释，不能盖过当前画面。

# 人物和关系写法
可以写人物之间的配合、等待、催促、分工和压力，但要用中性旁观口吻。不要替人物编内心独白，不要强行拔高苦难，也不要把关系写成狗血冲突。人物称谓优先使用画面或输入资料给出的通用身份，例如“老人、丈夫、孩子、家人、同伴、村民”，除非用户明确要求保留姓名。即便保留姓名，也不要复刻原账号里的固定人物设定。

# 句式和节奏
以短句和中短句为主，节奏连续但不碎。多数句子只表达一个动作或一个判断，少量长句用于串起原因和结果。可以使用“随后、接着、没想到、原来、这时、最后、为了、因为、但”来推动时间和转折，但不要机械堆叠。句子要有纪录片旁白感：不抢画面，不夸张，不用密集网络热词。每 2 到 4 句形成一个小推进：先描述画面，再解释为什么这样做，最后补一个结果或反差。

# 用词和语气
多用具体动作词和过程词，例如寻找、背起、放下、捡起、带回、清理、压榨、翻找、继续、准备、赶回。判断词要克制，可以说“不容易、并不轻松、看起来、显然、没想到、只好、终于”，但不要把情绪写成夸张口号。允许出现生活压力、家庭分工、自然环境、收获多少、路途远近等判断，但必须由画面或转写支撑。可参考这类高频关注点：{term_hint}，但不要直接复制原视频里的姓名、地名、口头禅或连续原句。

# 段落组织
如果要生成 1 到 3 分钟的完整解说，按“开场 10% + 过程 70% + 收束 20%”组织。开场负责给出动作钩子和处境；过程负责持续跟随画面，每段都要有新动作；收束负责落到具体结果、未解决的问题或下一步任务。不要每段都重复“这很不容易”，要换成具体难点：山路、重量、时间、工具、天气、距离、材料状态、家人的等待。

# 复刻公式
1. “[画面里正在发生的具体动作]，说明[人物当前目标或处境]。”
2. “他/她先[动作A]，又[动作B]，因为[从画面或资料能看出的原因]。”
3. “看似只是[表层动作]，其实难点在[工具、环境、体力、时间或家庭压力]。”
4. “没想到[画面里的转折]，这让接下来的[行动/结果]变得更麻烦。”
5. “最后[可见结果]，但[仍然存在的现实问题或下一步悬念]。”
6. “镜头里[细节变化]，也说明[当前处境正在变好/变难]。”
7. “为了[具体目标]，他/她只能[下一步动作]，这一步比看上去更费时间。”
8. “如果只看[表面结果]，可能以为很简单，但[画面证据]才是关键。”

# 可直接套用的分镜旁白骨架
第一段：从最清楚的动作切入，交代人物正在做什么，以及这件事和眼前目标有什么关系。
第二段：顺着画面说下一步动作，补一句为什么必须这样做，不要跳到画面外故事。
第三段：抓一个困难或变化，用克制语气写出反差，让观众知道流程并不轻松。
第四段：回到动作推进，写人物如何继续处理、搬运、寻找、清理、等待或准备。
第五段：落到结果，说明眼前收获、损失、进展或未解决的问题，并自然留下下一步。

# 生成要求
写新解说时，必须优先复刻“说话方式”，不要复刻原题材。输出应像同一类型解说员在讲一个新画面：画面证据先行，信息逐步推进，情绪克制，判断具体。遇到不确定内容时，用“看起来、似乎、可能是因为画面中……”这类保守表达；不要把推测写成事实。不要只给风格标签，要真正按这种节奏写完整旁白。

# 自检标准
生成后逐段检查：第一，每段是否至少有一个当前画面证据；第二，判断是否能从画面或输入资料推出；第三，是否有连续动作链，而不是孤立点评；第四，是否避免复制原视频人物、地名、标题和长句；第五，语气是否像旁观纪录片解说，而不是广告、新闻稿、鸡汤或短剧旁白。达不到这五点，就重写。

# 禁止事项
禁止写账号名、主页链接、视频 ID、固定人物姓名、固定地名、原作品标题或可识别原文。禁止连续照搬原作者句子。禁止编造画面外故事、家庭关系、冲突、收入、结局或动机。禁止用泛泛鸡汤、营销话术、夸张反转标题、密集感叹号、网络段子替代画面描述。禁止只总结“这是纪录片风格”，必须在每段具体写出动作、原因、转折和结果。
""".strip()


def _build_local_english_style_prompt(style_position: str, recurring_terms: List[str]) -> str:
    term_hint = ", ".join(recurring_terms[:6]) if recurring_terms else "visible actions, setting changes, tools, materials, relationships, outcomes"
    return f"""
# Core Style
Write in a {style_position} voice without copying any account identity, fixed people, places, titles, links, or source wording. The narrator should sound like an observant documentary commentator: grounded, sequential, lightly curious, and focused on what the viewer can actually see.

# Visual-First Rule
Start each beat from visible evidence: actions, positions, tools, terrain, weather, containers, materials, facial reactions, or changes on screen. Any judgement must be supported by the current visuals or provided transcript evidence. Do not invent off-screen story.

# Structure
Use this flow: visible action or discovery -> immediate goal or situation -> next actions -> obstacle or turn -> concrete result. Each sentence should do one job: describe, explain, connect, contrast, or conclude.

# Rhythm
Prefer short and medium sentences. Keep the pace continuous but calm. Use simple connectors such as then, after that, but, because, at this point, and finally. Avoid hype, slogans, and generic motivational lines.

# Diction
Use concrete process verbs and cautious judgements. Recurring attention points in the source pattern include: {term_hint}. Treat them as style signals, not content that must be copied.

# Reusable Formulas
1. "[Visible action] shows [current goal or situation]."
2. "They first [action A], then [action B], because [evidence-based reason]."
3. "It looks like [surface action], but the difficult part is [tool, environment, time, strength, or pressure]."
4. "The turn comes when [visible change], which makes [next step] harder."
5. "By the end, [visible result], but [remaining concrete problem]."

# Do Not
Do not include account names, links, video IDs, fixed names, fixed places, original titles, copied phrases, invented backstory, unsupported emotions, or exaggerated clickbait. Recreate the narration mechanics, not the source topics.
""".strip()


def synthesize_style_from_transcripts_locally(
    transcripts: List[Dict],
    style_name: str = "",
    profile_url: str = "",
    video_count: int = 0,
    language: str = "zh",
) -> Dict:
    transcript_items = [item for item in (transcripts or []) if isinstance(item, dict)]
    texts = [str(((item.get("transcript") or {}).get("text") or "")).strip() for item in transcript_items]
    texts = [text for text in texts if text]
    if not texts:
        raise Exception("No transcript text was available for local style synthesis.")

    titles = [
        str(((item.get("video") or {}).get("title") or "")).strip()
        for item in transcript_items
        if str(((item.get("video") or {}).get("title") or "")).strip()
    ]
    all_text = "\n".join(texts)
    titles_text = "\n".join(titles)
    units = _split_commentary_units(all_text)
    avg_unit_chars = round(sum(len(unit) for unit in units) / max(1, len(units)), 1)
    short_unit_ratio = round(sum(1 for unit in units if len(unit) <= 18) / max(1, len(units)), 3)
    recurring_terms = _count_terms(
        f"{titles_text}\n{all_text}",
        [
            "随后", "接着", "这时", "最后", "没想到", "原来", "因为", "为了", "继续", "准备",
            "找到", "发现", "带回", "收获", "回家", "一家人", "孩子", "丈夫", "婆婆", "山坡",
            "山上", "清晨", "野菜", "药材", "工具", "赶紧", "终于",
        ],
    )
    style_position = _infer_local_style_position(all_text, titles_text, language)
    lang = str(language or "").strip().lower()
    if lang in {"zh", "zh-cn", "cn"}:
        prompt = _build_local_chinese_style_prompt(style_position, recurring_terms)
        label = _normalize_style_label(style_name or "真实转写归纳解说风格")
        summary = (
            f"基于 {len(texts)} 条真实转写做本地归纳：画面先行、流程推进、短句为主、判断克制。"
            f"共抽取约 {len(all_text)} 字转写，平均语义单元约 {avg_unit_chars} 字。"
        )
        traits = ["画面先行", "流程推进", "短句旁白", "克制判断", "禁止编造画面外剧情"]
        warnings = [
            "未配置 OpenAI 兼容模型，使用本地规则从真实转写归纳；不是模型聚合结果。",
            "只复刻通用说话方式，不复制原账号身份、人物姓名、地名或原文。",
        ]
    else:
        prompt = _build_local_english_style_prompt(style_position, recurring_terms)
        label = _normalize_style_label(style_name or "Transcript-derived commentary style")
        summary = (
            f"Local synthesis from {len(texts)} real transcripts: visual-first, process-driven, concise, and grounded. "
            f"Approx. {len(all_text)} transcript characters; average unit length {avg_unit_chars}."
        )
        traits = ["visual-first", "process-driven", "concise documentary voice", "grounded judgements"]
        warnings = [
            "No OpenAI-compatible model was configured; this is a local rule-based synthesis from real transcripts.",
            "Use it to reproduce generic narration mechanics, not source identities or exact wording.",
        ]

    return {
        "label": label,
        "prompt": _sanitize_style_prompt(prompt, profile_url),
        "source": "douyin_style_learning_local_transcript_synthesis",
        "summary": summary,
        "style_traits": traits,
        "metadata": {
            "source_platform": "douyin",
            "language": language or "zh",
            "video_count": video_count,
            "transcript_count": len(texts),
            "total_transcript_chars": len(all_text),
            "commentary_unit_count": len(units),
            "avg_commentary_unit_chars": avg_unit_chars,
            "short_commentary_unit_ratio": short_unit_ratio,
            "recurring_terms": recurring_terms,
            "analysis_method": "local_transcript_synthesis",
            "confidence": "medium" if len(texts) >= 6 else "low",
            "warnings": warnings,
        },
    }


def _write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_commentary_style_learning(
    profile_url: str,
    output_dir: str,
    openai_config: Dict,
    cookie_path: str = DOUYIN_COOKIES_PATH,
    style_name: str = "",
    max_videos: int = 100,
    language: str = "zh",
    progress: Optional[Callable[[str], None]] = None,
    checkpoint: Optional[Callable[[Dict], None]] = None,
    cancel_event=None,
    provider: Optional[DouyinProfileProvider] = None,
    download_audio_fn: Optional[Callable] = None,
    transcribe_fn: Optional[Callable] = None,
    openai_chat_fn: Optional[Callable] = None,
    style_storage_path: str = COMMENTARY_STYLES_PATH,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    provider = provider or DouyinProfileProvider()
    download_audio_fn = download_audio_fn or download_douyin_audio
    transcribe_fn = transcribe_fn or transcribe_video
    use_openai_analysis = bool(openai_chat_fn) or _has_openai_style_config(openai_config)
    openai_chat = openai_chat_fn or (_style_analysis_chat(openai_config) if use_openai_analysis else None)
    download_concurrency = max(1, int(os.environ.get("OPENSHORTS_STYLE_DOWNLOAD_CONCURRENCY", "2")))
    analysis_concurrency = max(1, int(os.environ.get("OPENSHORTS_STYLE_ANALYSIS_CONCURRENCY", "2")))
    media_concurrency = max(1, int(os.environ.get("OPENSHORTS_STYLE_MEDIA_CONCURRENCY", "2")))

    def log(message: str) -> None:
        if progress:
            progress(message)

    def cp(fields: Dict) -> None:
        if checkpoint:
            checkpoint(fields)

    _raise_if_cancelled(cancel_event)
    log("Fetching Douyin profile videos...")
    videos = provider.fetch_videos(profile_url, cookie_path, progress=log, cancel_event=cancel_event)
    ranked = rank_douyin_videos(videos, max_videos=max_videos)
    if not ranked:
        raise Exception("No Douyin videos were available after ranking.")
    selected_path = os.path.join(output_dir, "selected_videos.json")
    _write_json(selected_path, ranked)
    cp({
        "total_videos": len(videos),
        "selected_count": len(ranked),
        "selected_videos": ranked,
        "selected_videos_path": selected_path,
    })
    log(f"Selected {len(ranked)} videos from {len(videos)} visible Douyin videos.")

    log(f"Resolving public media URLs for {len(ranked)} selected Douyin videos...")
    enriched_ranked = [None] * len(ranked)
    with ThreadPoolExecutor(max_workers=media_concurrency) as executor:
        future_map = {
            executor.submit(ensure_douyin_direct_media, video, log, cancel_event): index
            for index, video in enumerate(ranked)
        }
        for future in as_completed(future_map):
            index = future_map[future]
            _raise_if_cancelled(cancel_event)
            video = ranked[index]
            try:
                enriched = future.result()
            except StyleLearningCancelled:
                raise
            except Exception as exc:
                log(f"Media URL resolution failed for {video.get('aweme_id')}: {str(exc)[:240]}")
                enriched = video
            enriched_ranked[index] = {
                **enriched,
                "rank_index": index + 1,
            }
            resolved_count = sum(1 for item in enriched_ranked if item is not None)
            cp({"media_resolved_count": resolved_count})
            log(f"Resolved media URL metadata {resolved_count}/{len(ranked)}: {video.get('aweme_id')}")
    ranked = [item or {**ranked[index], "rank_index": index + 1} for index, item in enumerate(enriched_ranked)]
    _write_json(selected_path, ranked)
    cp({
        "selected_videos": ranked,
        "selected_videos_path": selected_path,
        "media_resolved_count": len(ranked),
        "media_direct_count": sum(1 for item in ranked if _has_douyin_direct_media(item)),
    })

    audio_dir = os.path.join(output_dir, "audio")
    downloaded = []
    failures = []
    with ThreadPoolExecutor(max_workers=download_concurrency) as executor:
        future_map = {
            executor.submit(download_audio_fn, video, audio_dir, cookie_path, cancel_event): video
            for video in ranked
        }
        for future in as_completed(future_map):
            video = future_map[future]
            _raise_if_cancelled(cancel_event)
            try:
                info = future.result()
                downloaded.append({"video": video, **(info or {})})
                log(f"Downloaded audio {len(downloaded)}/{len(ranked)}: {video.get('aweme_id')}")
            except Exception as exc:
                failures.append({"aweme_id": video.get("aweme_id"), "stage": "download", "error": str(exc)[:800]})
                log(f"Audio download failed for {video.get('aweme_id')}: {str(exc)[:240]}")
            cp({"downloaded_count": len(downloaded), "failed_videos": _dedupe_failed_videos(failures)[-100:]})

    downloaded.sort(key=lambda item: _int_value((item.get("video") or {}).get("rank_index")))
    transcripts = []
    transcript_dir = os.path.join(output_dir, "transcripts")
    os.makedirs(transcript_dir, exist_ok=True)
    for item in downloaded:
        _raise_if_cancelled(cancel_event)
        video = item["video"]
        audio_path = item.get("audio_path")
        try:
            log(f"Transcribing audio {len(transcripts) + 1}/{len(downloaded)}: {video.get('aweme_id')}")
            transcript = transcribe_fn(audio_path)
            text = str((transcript or {}).get("text") or "").strip()
            if len(text) < 20:
                raise Exception("Transcript is too short to analyze commentary style.")
            transcript_path = os.path.join(transcript_dir, f"{video.get('aweme_id')}.json")
            payload = {"video": video, "audio_path": audio_path, "transcript": transcript}
            _write_json(transcript_path, payload)
            transcripts.append({"video": video, "audio_path": audio_path, "transcript_path": transcript_path, "transcript": transcript})
            cp({"transcript_count": len(transcripts)})
        except Exception as exc:
            failures.append({"aweme_id": video.get("aweme_id"), "stage": "transcribe", "error": str(exc)[:800]})
            log(f"Transcription failed for {video.get('aweme_id')}: {str(exc)[:240]}")
            cp({"failed_videos": _dedupe_failed_videos(failures)[-100:]})
    if not transcripts:
        raise Exception("No Douyin video audio could be transcribed, so no commentary style can be learned.")

    summaries = []
    summaries_path = ""
    if use_openai_analysis:
        log(f"Analyzing commentary style from {len(transcripts)} transcripts with OpenAI-compatible model...")
        with ThreadPoolExecutor(max_workers=analysis_concurrency) as executor:
            future_map = {
                executor.submit(summarize_single_video_style, item["video"], item["transcript"], openai_chat): item
                for item in transcripts
            }
            for future in as_completed(future_map):
                item = future_map[future]
                _raise_if_cancelled(cancel_event)
                video = item["video"]
                try:
                    summary = future.result()
                    summaries.append(summary)
                    log(f"Analyzed style summary {len(summaries)}/{len(transcripts)}: {video.get('aweme_id')}")
                    cp({"style_summary_count": len(summaries)})
                except Exception as exc:
                    failures.append({"aweme_id": video.get("aweme_id"), "stage": "analyze", "error": str(exc)[:800]})
                    log(f"Style analysis failed for {video.get('aweme_id')}: {str(exc)[:240]}")
                    cp({"failed_videos": _dedupe_failed_videos(failures)[-100:]})
        if not summaries:
            raise Exception("OpenAI-compatible model did not return any usable style summaries.")

        summaries.sort(key=lambda item: _int_value(item.get("rank_index")))
        summaries_path = os.path.join(output_dir, "style_summaries.json")
        _write_json(summaries_path, summaries)
        log("Aggregating final reusable commentary style...")
        style = aggregate_style_summaries(
            summaries,
            openai_chat,
            style_name=style_name,
            profile_url=profile_url,
            video_count=len(ranked),
            transcript_count=len(transcripts),
            language=language,
        )
        analysis_method = "openai_compatible"
    else:
        log(f"Synthesizing reusable commentary style locally from {len(transcripts)} real transcripts...")
        style = synthesize_style_from_transcripts_locally(
            transcripts,
            style_name=style_name,
            profile_url=profile_url,
            video_count=len(ranked),
            language=language,
        )
        analysis_method = "local_transcript_synthesis"
        cp({"style_summary_count": 0})

    successful_aweme_ids = {str((item.get("video") or {}).get("aweme_id") or "").strip() for item in transcripts}
    failed_videos = _dedupe_failed_videos(failures, successful_aweme_ids=successful_aweme_ids)
    style["metadata"] = {
        **(style.get("metadata") or {}),
        "profile_url": profile_url,
        "selected_count": len(ranked),
        "downloaded_count": len(downloaded),
        "failed_count": len(failed_videos),
        "analysis_method": analysis_method,
    }
    if summaries_path:
        style["metadata"]["summaries_path"] = summaries_path
    saved_style = save_commentary_style(style, storage_path=style_storage_path)
    result = {
        "style": saved_style,
        "profile_url": profile_url,
        "video_count": len(videos),
        "selected_count": len(ranked),
        "downloaded_count": len(downloaded),
        "transcript_count": len(transcripts),
        "style_summary_count": len(summaries),
        "failed_videos": failed_videos[-100:],
        "selected_videos": ranked,
        "selected_videos_path": selected_path,
        "style_summaries_path": summaries_path,
        "analysis_method": analysis_method,
    }
    result_path = os.path.join(output_dir, "style_learning_result.json")
    _write_json(result_path, result)
    result["result_path"] = result_path
    cp({"result": result, "style": saved_style})
    try:
        if os.environ.get("OPENSHORTS_STYLE_KEEP_AUDIO", "0").strip().lower() not in {"1", "true", "yes"}:
            shutil.rmtree(audio_dir, ignore_errors=True)
    except Exception:
        pass
    return result
