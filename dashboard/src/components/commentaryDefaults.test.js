import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';

import {
  COMMENTARY_DEFAULTS,
  getDefaultEdgeVoiceForLanguage,
} from './commentaryDefaults.js';

test('commentary tab defaults match preferred Chinese remix setup', () => {
  assert.equal(COMMENTARY_DEFAULTS.style, 'funny');
  assert.equal(COMMENTARY_DEFAULTS.targetDuration, 'full');
  assert.equal(COMMENTARY_DEFAULTS.analysisMode, 'openai');
  assert.equal(COMMENTARY_DEFAULTS.edgeVoice, 'zh-CN-YunjianNeural');
  assert.equal(COMMENTARY_DEFAULTS.originalAudioVolume, 0.08);
  assert.equal(COMMENTARY_DEFAULTS.pauseOriginalAudioVolume, 0.6);
  assert.equal(COMMENTARY_DEFAULTS.openAIFrameIntervalSeconds, 3);
  assert.equal(COMMENTARY_DEFAULTS.openAIMaxFrames, 1800);
  assert.equal(COMMENTARY_DEFAULTS.openAISceneMaxKeyframes, 60);
  assert.equal(COMMENTARY_DEFAULTS.openAIBatchSize, 32);
  assert.equal(COMMENTARY_DEFAULTS.openAIVisualConcurrency, 3);
});

test('Chinese Edge voice default is Yunjian and other languages keep first option fallback', () => {
  assert.equal(getDefaultEdgeVoiceForLanguage('zh'), 'zh-CN-YunjianNeural');
  assert.equal(getDefaultEdgeVoiceForLanguage('en'), 'en-US-JennyNeural');
});

test('commentary status polling tolerates transient status endpoint failures', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /statusPollFailuresRef/);
  assert.match(source, /状态检查失败，正在重试/);
  assert.match(source, /statusPollFailuresRef\.current >= 5/);
});

test('commentary task manager lists jobs and retries saved tasks', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /\/api\/commentary\/jobs/);
  assert.match(source, /\/api\/commentary\/jobs\/\$\{task\.job_id\}\/retry/);
  assert.match(source, /setJobId\(data\.job_id\)/);
  assert.match(source, /历史任务/);
});

test('commentary supports OpenAI-compatible multimodal analysis mode', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /value="openai"/);
  assert.match(source, /buildOpenAICompatibleHeaders/);
  assert.match(source, /hasOpenAICompatibleAccess/);
  assert.match(source, /openai_model/);
  assert.match(source, /openai_visual_concurrency/);
});

test('commentary sends separate pause original audio volume', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /pauseOriginalAudioVolume/);
  assert.match(source, /pause_original_audio_volume/);
  assert.match(source, /无解说片段原视频音量/);
});

test('commentary status steps use the active job analysis mode', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /activeAnalysisMode/);
  assert.match(source, /data\.request\?\.analysis_mode/);
  assert.match(source, /task\.request\?\.analysis_mode/);
  assert.match(source, /displayAnalysisMode === 'openai'/);
});
