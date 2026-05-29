import os
import uuid
import time
import json
import re
from google.genai import types
from gemini_client import create_gemini_client
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def _load_thumbnail_font(size):
    root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, "fonts", "NotoSansCJK-Bold.ttc"),
        os.path.join(root, "fonts", "NotoSerif-Bold.ttf"),
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\msyhbd.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
    ]
    for path in candidates:
        try:
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _clean_thumbnail_hook(title):
    text = re.sub(r"[#@]\S+", "", str(title or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return "全程高能"
    if re.search(r"[\u4e00-\u9fff]", text):
        return text[:10].strip("，。！？、,.!? ") or "全程高能"
    return " ".join(text.upper().split()[:4]) or "MUST WATCH"


def _wrap_thumbnail_text(text, draw, font, max_width, stroke_width):
    if re.search(r"[\u4e00-\u9fff]", text):
        tokens = list(text)
    else:
        tokens = text.split()
    lines = []
    current = ""
    joiner = "" if re.search(r"[\u4e00-\u9fff]", text) else " "
    for token in tokens:
        candidate = token if not current else f"{current}{joiner}{token}"
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
        if current and bbox[2] - bbox[0] > max_width:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:3]


def _add_art_text_to_thumbnail(image_path, title):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rectangle((0, int(height * 0.50), width, height), fill=(0, 0, 0, 150))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(12, width // 80)))
        overlay.alpha_composite(shadow)

        draw = ImageDraw.Draw(overlay)
        hook = _clean_thumbnail_hook(title)
        font_size = max(58, int(height * 0.19))
        stroke_width = max(5, font_size // 13)
        font = _load_thumbnail_font(font_size)
        max_width = int(width * 0.88)
        lines = _wrap_thumbnail_text(hook, draw, font, max_width, stroke_width)
        while len(lines) > 2 and font_size > 48:
            font_size = int(font_size * 0.9)
            stroke_width = max(5, font_size // 13)
            font = _load_thumbnail_font(font_size)
            lines = _wrap_thumbnail_text(hook, draw, font, max_width, stroke_width)

        boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width) for line in lines]
        line_gap = int(font_size * 0.04)
        total_height = sum(box[3] - box[1] for box in boxes) + line_gap * max(0, len(lines) - 1)
        y = height - total_height - int(height * 0.08)
        colors = [(196, 255, 54, 255), (255, 255, 255, 255)]
        for index, line in enumerate(lines):
            bbox = boxes[index]
            text_w = bbox[2] - bbox[0]
            x = max(int(width * 0.05), (width - text_w) // 2)
            wobble = int(font_size * (0.04 if index % 2 == 0 else -0.025))
            draw.text(
                (x, y + wobble),
                line,
                font=font,
                fill=colors[index % len(colors)],
                stroke_width=stroke_width,
                stroke_fill=(0, 0, 0, 255),
            )
            highlight_y = y + bbox[3] - bbox[1] + wobble - max(3, font_size // 20)
            draw.line(
                (x + int(text_w * 0.05), highlight_y, x + int(text_w * 0.95), highlight_y),
                fill=(255, 225, 30, 210),
                width=max(4, font_size // 18),
            )
            y += bbox[3] - bbox[1] + line_gap

        composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        composed.save(image_path, quality=94)


def analyze_video_for_titles(api_key, video_path, transcript=None, base_url=None):
    """
    Transcribes a video and uses Gemini to suggest viral YouTube titles.
    If transcript is provided, skips Whisper transcription.
    Returns: { "titles": [...], "transcript_summary": "...", "language": "...", "segments": [...], "video_duration": ... }
    """
    if transcript is None:
        from main import transcribe_video
        print("🎬 [Thumbnail] Transcribing video...")
        transcript = transcribe_video(video_path)
    else:
        print("🎬 [Thumbnail] Using pre-computed transcript (Whisper already done)...")

    print("📤 [Thumbnail] Uploading video to Gemini...")
    client = create_gemini_client(api_key, base_url)

    file_upload = client.files.upload(file=video_path)
    while True:
        file_info = client.files.get(name=file_upload.name)
        if file_info.state == "ACTIVE":
            break
        elif file_info.state == "FAILED":
            raise Exception("Video processing failed by Gemini.")
        time.sleep(2)

    prompt = f"""You are a YouTube title expert who creates viral, click-worthy titles.

Analyze this video and its transcript, then suggest 10 YouTube titles that would maximize CTR (click-through rate).

TRANSCRIPT:
{transcript['text']}

RULES:
- Titles must be under 70 characters
- Use power words, curiosity gaps, and emotional triggers
- Mix styles: how-to, listicle, story-driven, controversial, question-based
- Make them specific to the actual content, not generic
- Include numbers where appropriate
- Consider the language of the video (detected: {transcript['language']})
- Titles should be in the SAME LANGUAGE as the video transcript

Also provide a brief summary of the video content (2-3 sentences).

After generating all 10 titles, pick the TOP 2 you most recommend and explain concisely WHY (CTR potential, emotional hook, uniqueness, etc.). Reference them by their 0-based index in the titles array.

OUTPUT JSON:
{{
    "titles": ["title1", "title2", ...],
    "transcript_summary": "Brief summary of the video content...",
    "language": "{transcript['language']}",
    "recommended": [
        {{"index": 0, "reason": "Why this title is best..."}},
        {{"index": 3, "reason": "Why this title is second best..."}}
    ]
}}"""

    print("🤖 [Thumbnail] Asking Gemini for title suggestions...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[file_upload, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    # Extract segments and duration from transcript for later use
    segments = transcript.get("segments", [])
    video_duration = segments[-1]["end"] if segments else 0

    try:
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]

        result = json.loads(text)
        result["transcript_summary"] = result.get("transcript_summary", "")
        result["language"] = result.get("language", transcript["language"])
        result["segments"] = segments
        result["video_duration"] = video_duration
        return result
    except json.JSONDecodeError:
        print(f"❌ [Thumbnail] Failed to parse titles JSON: {response.text}")
        return {
            "titles": ["Could not generate titles - please try again"],
            "transcript_summary": transcript["text"][:500],
            "language": transcript["language"],
            "segments": segments,
            "video_duration": video_duration
        }


def refine_titles(api_key, context, user_message, conversation_history=None, base_url=None):
    """
    Takes video context + user feedback and returns refined title suggestions.
    """
    client = create_gemini_client(api_key, base_url)

    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            role = msg.get("role", "user")
            history_text += f"\n{role.upper()}: {msg['content']}"

    prompt = f"""You are a YouTube title expert. Based on the video context and the user's feedback, suggest 8 new refined YouTube titles.

VIDEO CONTEXT:
{context}

CONVERSATION HISTORY:{history_text}

USER'S NEW REQUEST:
{user_message}

RULES:
- Titles must be under 70 characters
- Incorporate the user's feedback/direction
- Keep titles viral and click-worthy
- If the user asks for a specific style, follow it
- Titles should be in the same language as the original content

OUTPUT JSON:
{{
    "titles": ["title1", "title2", ...]
}}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx + 1]

        return json.loads(text)
    except json.JSONDecodeError:
        print(f"❌ [Thumbnail] Failed to parse refined titles: {response.text}")
        return {"titles": ["Could not refine titles - please try again"]}


def generate_thumbnail(api_key, title, session_id, face_image_path=None, bg_image_path=None, extra_prompt="", count=3, video_context="", base_url=None):
    """
    Generates YouTube thumbnails using Gemini image generation.
    Returns list of saved image paths (relative URLs).
    """
    client = create_gemini_client(api_key, base_url)

    output_dir = os.path.join("output", "thumbnails", session_id)
    os.makedirs(output_dir, exist_ok=True)

    prompt_parts = []

    # Add face image if provided
    if face_image_path and os.path.exists(face_image_path):
        face_img = Image.open(face_image_path)
        prompt_parts.append(face_img)

    # Add background image if provided
    if bg_image_path and os.path.exists(bg_image_path):
        bg_img = Image.open(bg_image_path)
        prompt_parts.append(bg_img)

    # Build video context block
    context_block = ""
    if video_context:
        context_block = f"""
VIDEO CONTEXT (use this to understand the video and design a relevant thumbnail):
{video_context}
"""

    # Build extra instructions block (high priority)
    extra_block = ""
    if extra_prompt:
        extra_block = f"""
⚠️ MANDATORY USER INSTRUCTIONS (MUST follow these exactly — they override any default behavior):
{extra_prompt}
"""

    text_prompt = f"""Generate a professional, eye-catching YouTube thumbnail image.

VIDEO TITLE (for reference — do NOT put the full title on the thumbnail): "{title}"
{context_block}
TEXT ON THE THUMBNAIL:
- Do NOT render any letters, captions, subtitles, words, logos, or UI text inside the generated image
- Leave a clean lower-third or side area with enough contrast for a later text overlay
{extra_block}
DESIGN REQUIREMENTS:
- Use vibrant, eye-catching colors that match the video's mood
- Professional YouTube thumbnail aesthetic
- Clean composition — text and face/subject as clear focal points
- NO clutter, NO small text, NO watermarks"""

    if face_image_path and os.path.exists(face_image_path):
        text_prompt += "\n- Include the provided face/person prominently with an exaggerated expression (surprise, excitement, shock)"

    if bg_image_path and os.path.exists(bg_image_path):
        text_prompt += "\n- Use the provided background image as the base/backdrop"

    prompt_parts.append(text_prompt)

    thumbnails = []
    last_error = None
    for i in range(count):
        print(f"🎨 [Thumbnail] Generating thumbnail {i + 1}/{count}...")
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-image-preview",
                contents=prompt_parts,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio="16:9",
                        image_size="2K"
                    )
                )
            )

            for part in response.parts:
                if part.text is not None:
                    print(f"📝 [Thumbnail] Gemini text: {part.text}")
                elif image := part.as_image():
                    filename = f"thumb_{i + 1}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    image.save(filepath)
                    _add_art_text_to_thumbnail(filepath, title)
                    thumbnails.append(f"/thumbnails/{session_id}/{filename}")
                    print(f"✅ [Thumbnail] Saved: {filepath}")
                    break

        except Exception as e:
            last_error = str(e)
            print(f"❌ [Thumbnail] Generation {i + 1} failed: {e}")

    if not thumbnails and last_error:
        raise RuntimeError(f"All thumbnail generations failed. Last error: {last_error}")

    return thumbnails


def generate_youtube_description(api_key, title, transcript_segments, language, video_duration, base_url=None):
    """
    Uses Gemini to generate a YouTube description with chapter markers from transcript segments.
    Returns: { "description": "full description text with chapters" }
    """
    client = create_gemini_client(api_key, base_url)

    # Format segments for the prompt
    formatted_segments = []
    for seg in transcript_segments:
        start = seg.get("start", 0)
        mins = int(start // 60)
        secs = int(start % 60)
        timestamp = f"{mins}:{secs:02d}"
        formatted_segments.append(f"[{timestamp}] {seg.get('text', '').strip()}")

    segments_text = "\n".join(formatted_segments)

    # Format total duration
    dur_mins = int(video_duration // 60)
    dur_secs = int(video_duration % 60)
    duration_str = f"{dur_mins}:{dur_secs:02d}"

    prompt = f"""You are a YouTube SEO expert. Generate a complete YouTube video description for the following video.

VIDEO TITLE: "{title}"
VIDEO LANGUAGE: {language}
VIDEO DURATION: {duration_str}

TRANSCRIPT WITH TIMESTAMPS:
{segments_text}

REQUIREMENTS:
1. Write the description in the SAME LANGUAGE as the video ({language})
2. Start with a compelling 2-3 sentence summary/hook
3. Add relevant CTAs (subscribe, like, comment)
4. Generate YouTube CHAPTERS based on the transcript timestamps:
   - First chapter MUST start at 0:00
   - Minimum 3 chapters, each at least 10 seconds apart
   - Chapter titles should be concise and descriptive
   - Format: 0:00 Chapter Title
   - Place chapters in their own section with a blank line before and after
5. Add 5-10 relevant hashtags at the end
6. Keep the total description under 5000 characters

OUTPUT: Return ONLY the description text (no JSON wrapper, no markdown code blocks). The description should be ready to paste directly into YouTube."""

    print("🤖 [Thumbnail] Generating YouTube description with chapters...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
    )

    description = response.text.strip()
    # Clean up any accidental markdown wrappers
    if description.startswith("```"):
        lines = description.split("\n")
        description = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return {"description": description}
