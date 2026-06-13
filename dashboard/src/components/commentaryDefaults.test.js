import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';

import {
  COMMENTARY_DEFAULTS,
  getDefaultEdgeVoiceForLanguage,
} from './commentaryDefaults.js';

test('commentary tab defaults match hustle Chinese remix setup', () => {
  assert.equal(COMMENTARY_DEFAULTS.style, 'hustle');
  assert.equal(COMMENTARY_DEFAULTS.customStylePrompt, '');
  assert.equal(COMMENTARY_DEFAULTS.targetDuration, 'full');
  assert.equal(COMMENTARY_DEFAULTS.analysisMode, 'openai');
  assert.equal(COMMENTARY_DEFAULTS.edgeVoice, 'zh-CN-YunyangNeural');
  assert.equal(COMMENTARY_DEFAULTS.originalAudioVolume, 0.3);
  assert.equal(COMMENTARY_DEFAULTS.pauseOriginalAudioVolume, 0.6);
  assert.equal(COMMENTARY_DEFAULTS.openAIFrameIntervalSeconds, 3);
  assert.equal(COMMENTARY_DEFAULTS.openAIMaxFrames, 1800);
  assert.equal(COMMENTARY_DEFAULTS.openAISceneMaxKeyframes, 60);
  assert.equal(COMMENTARY_DEFAULTS.openAIBatchSize, 46);
  assert.equal(COMMENTARY_DEFAULTS.openAIVisualConcurrency, 2);
  assert.equal(COMMENTARY_DEFAULTS.commentaryBlockConcurrency, 2);
  assert.equal(COMMENTARY_DEFAULTS.backgroundMusicEnabled, false);
  assert.equal(COMMENTARY_DEFAULTS.backgroundMusicTrack, 'aodebiao_caravan');
  assert.equal(COMMENTARY_DEFAULTS.backgroundMusicVolume, 0.16);
});

test('Chinese Edge voice default is Yunyang and other languages keep first option fallback', () => {
  assert.equal(getDefaultEdgeVoiceForLanguage('zh'), 'zh-CN-YunyangNeural');
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
  assert.match(source, /commentary_block_concurrency/);
  assert.match(source, /解说分块生成并发数/);
});

test('commentary exposes first-person hustle commentary style', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /first_person_hustle/);
  assert.match(source, /整活第一视角/);
  assert.match(source, /id: 'hustle'/);
  assert.match(source, /整活解说/);
});

test('commentary allows a custom style prompt under commentary style', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /id: 'custom'/);
  assert.match(source, /自定义提示词/);
  assert.match(source, /customStylePrompt/);
  assert.match(source, /custom_style_prompt/);
  assert.match(source, /自定义风格提示词/);
  assert.match(source, /customStyleOptions/);
  assert.match(source, /CUSTOM_STYLE_STORAGE_KEY/);
  assert.match(source, /添加到下拉框/);
  assert.match(source, /更新下拉框选项/);
});

test('commentary sends separate pause original audio volume', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /pauseOriginalAudioVolume/);
  assert.match(source, /pause_original_audio_volume/);
  assert.match(source, /无解说片段原视频音量/);
});

test('commentary lets users configure background music volume', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /backgroundMusicVolume/);
  assert.match(source, /background_music_volume/);
  assert.match(source, /背景音乐音量/);
});

test('commentary allows larger OpenAI-compatible visual batches', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /max="128"/);
  assert.match(source, /默认 46，最大 128/);
});

test('commentary status steps use the active job analysis mode', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /activeAnalysisMode/);
  assert.match(source, /data\.request\?\.analysis_mode/);
  assert.match(source, /task\.request\?\.analysis_mode/);
  assert.match(source, /displayAnalysisMode === 'openai'/);
});

test('commentary result exposes Douyin publish copy and cover downloads', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /navigator\.clipboard\.writeText/);
  assert.match(source, /publish_title/);
  assert.match(source, /publish_description/);
  assert.match(source, /cover_landscape_url/);
  assert.match(source, /cover_portrait_url/);
  assert.match(source, /横封面 4:3/);
  assert.match(source, /竖封面 3:4/);
});
