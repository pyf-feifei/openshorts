# Commentary Publish Assets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Douyin-ready title/description copy fields and two downloadable commentary cover images.

**Architecture:** Backend derives publish metadata from the existing commentary script and extracts two covers from the final video with ffmpeg. Frontend renders those fields in the completed commentary result card with clipboard buttons and cover preview/download actions.

**Tech Stack:** Python `unittest`/`pytest`, FastAPI result metadata, ffmpeg, React, static source tests.

---

### Task 1: Backend Publish Metadata

**Files:**
- Modify: `commentary.py`
- Test: `tests/test_commentary_analysis_mode.py`

- [ ] Add tests for `_build_douyin_publish_fields` that assert a <=30 character title, <=1000 character description, no banned meta phrases, and appended hashtags.
- [ ] Run `pytest tests/test_commentary_analysis_mode.py -k "publish_fields" -q` and confirm it fails because the helper does not exist.
- [ ] Implement `_build_douyin_publish_fields(script)` and include `publish_title` / `publish_description` in `generate_commentary_video` metadata.
- [ ] Re-run the targeted test and confirm it passes.

### Task 2: Backend Cover Extraction

**Files:**
- Modify: `commentary.py`
- Test: `tests/test_commentary_analysis_mode.py`

- [ ] Add a test that patches `subprocess.run` and calls `_generate_commentary_covers`, asserting both `4:3` and `3:4` ffmpeg crop/scale commands are issued and URLs use `/videos/<job>/<file>`.
- [ ] Run `pytest tests/test_commentary_analysis_mode.py -k "commentary_covers" -q` and confirm it fails because the helper does not exist.
- [ ] Implement `_generate_commentary_covers(final_path, output_dir, slug, duration)` and add `covers`, `cover_landscape_url`, and `cover_portrait_url` to metadata.
- [ ] Re-run the targeted test and confirm it passes.

### Task 3: Frontend Result UI

**Files:**
- Modify: `dashboard/src/components/CommentaryTab.jsx`
- Test: `dashboard/src/components/commentaryDefaults.test.js`

- [ ] Add static assertions for `navigator.clipboard.writeText`, `publish_title`, `publish_description`, `cover_landscape_url`, and `cover_portrait_url`.
- [ ] Run `cd dashboard; npm test -- --runInBand` if available, otherwise run the package's existing test command, and confirm the new assertions fail.
- [ ] Add a `copyText` helper, import `Copy`, render title/description publish blocks with copy buttons and counters, and render two cover previews with download buttons.
- [ ] Re-run frontend tests.

### Task 4: Verification

**Files:**
- Validate changed backend and frontend files.

- [ ] Run `pytest tests/test_commentary_analysis_mode.py -k "publish_fields or commentary_covers" -q`.
- [ ] Run the dashboard test command.
- [ ] Run `git diff --check`.
