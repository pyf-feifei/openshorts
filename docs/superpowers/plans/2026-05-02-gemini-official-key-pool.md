# Gemini Official Key Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an official Gemini multi-key pool mode with health-aware failover, statistics events, and compatibility with all existing Gemini-backed OpenShorts features.

**Architecture:** Introduce a focused backend `gemini_pool.py` module that normalizes single-key/proxy and official key-pool requests into one selection interface. Frontend request helpers build either legacy headers or a pool payload, while existing feature components keep calling the shared helper. Backend endpoints parse the pool config and pass it into Gemini-backed modules through a lightweight context.

**Tech Stack:** FastAPI, google-genai, Python unittest, React, localStorage, Node test runner.

---

### Task 1: Backend Pool Core

**Files:**
- Create: `gemini_pool.py`
- Test: `tests/test_gemini_pool.py`

- [ ] **Step 1: Write failing tests for pool parsing, selection, and error classification.**
- [ ] **Step 2: Run `python -m unittest discover -s tests -p "test_gemini_pool.py"` and verify failures.**
- [ ] **Step 3: Implement `GeminiKeyPool`, `GeminiPoolSession`, `GeminiEvent`, and error classification helpers.**
- [ ] **Step 4: Re-run the pool tests and verify they pass.**

### Task 2: Backend Request Integration

**Files:**
- Modify: `app.py`
- Modify: `commentary.py`
- Modify: `editor.py`
- Modify: `thumbnail.py`
- Modify: `saasshorts.py`
- Test: `tests/test_commentary_upload.py`

- [ ] **Step 1: Write failing endpoint tests that send a Gemini pool payload to commentary generation and verify the pool is passed through.**
- [ ] **Step 2: Run the commentary upload tests and verify the new test fails.**
- [ ] **Step 3: Add backend helpers that parse pool JSON from headers, JSON bodies, and multipart forms.**
- [ ] **Step 4: Update Gemini-backed modules to accept either a raw key/base URL or a pool object.**
- [ ] **Step 5: Re-run commentary and pool tests.**

### Task 3: Frontend Pool Helpers

**Files:**
- Modify: `dashboard/src/lib/geminiHeaders.js`
- Test: `dashboard/src/lib/geminiHeaders.test.js`

- [ ] **Step 1: Write failing tests for custom-proxy headers and official-pool payload building.**
- [ ] **Step 2: Run `npm test -- --test-name-pattern Gemini` from `dashboard` and verify failure.**
- [ ] **Step 3: Implement `fingerprintGeminiKey`, `buildGeminiConfig`, and pool-aware header/form helpers.**
- [ ] **Step 4: Re-run frontend helper tests.**

### Task 4: Settings UI And Call-Site Coverage

**Files:**
- Modify: `dashboard/src/App.jsx`
- Modify: `dashboard/src/components/CommentaryTab.jsx`
- Modify: `dashboard/src/components/SaaShortsTab.jsx`
- Modify: `dashboard/src/components/ThumbnailStudio.jsx`
- Modify: `dashboard/src/components/ResultCard.jsx`

- [ ] **Step 1: Add settings state for Gemini access mode, official key pool, and local stats.**
- [ ] **Step 2: Add UI controls for custom proxy vs official key pool, bulk key entry, key status rows, and clear stats.**
- [ ] **Step 3: Update all Gemini-backed components to receive and send the unified Gemini config.**
- [ ] **Step 4: Run frontend tests and build.**

### Task 5: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Run `python -m unittest discover -s tests -p "test_*.py"`.**
- [ ] **Step 2: Run `npm test` and `npm run build` in `dashboard`.**
- [ ] **Step 3: Restart the backend if code changed.**
- [ ] **Step 4: Run a full commentary generation using `C:\Apps\yt-dlp\Amazing Factory Process of Recycling Giant Copper Motor Scrap Converting Into New Copper Materials [8TVlG65UNoE] 480p.mp4` with official Gemini key-pool mode.**
- [ ] **Step 5: Report generated output path, Gemini key events, and any limits encountered.**
