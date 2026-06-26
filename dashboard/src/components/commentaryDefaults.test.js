import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';

import {
  COMMENTARY_DEFAULTS,
  getDefaultEdgeVoiceForLanguage,
} from './commentaryDefaults.js';
import {
  findCustomStyleOption,
  normalizeCustomStyleOptions,
  resolveCommentaryStyleRequest,
} from './commentaryCustomStyles.js';
import {
  COMMENTARY_LOG_PANEL_BODY_CLASS,
  getCommentaryLogPanelState,
} from './commentaryLogPanel.js';
import {
  mergeStyleIntoLocalStorage,
} from './commentaryStyleSync.js';

test('commentary tab defaults match hustle Chinese remix setup', () => {
  assert.equal(COMMENTARY_DEFAULTS.style, 'hustle');
  assert.equal(COMMENTARY_DEFAULTS.customStylePrompt, '');
  assert.equal(COMMENTARY_DEFAULTS.targetDuration, 'two_to_four');
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

test('commentary target length includes default selected 2-4 minute option', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /id: 'two_to_four'/);
  assert.match(source, /label: '2-4 分钟'/);
  assert.match(source, /useState\(COMMENTARY_DEFAULTS\.targetDuration\)/);
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
  assert.match(source, /\/api\/commentary\/jobs\/\$\{ids\[0\]\}/);
  assert.match(source, /\/api\/commentary\/jobs\/delete/);
  assert.match(source, /job_ids: ids/);
  assert.match(source, /analysis_mode: selectedAnalysisMode/);
  assert.match(source, /gemini_model: selectedAnalysisMode === 'openai' \? undefined : geminiModel\.trim\(\)/);
  assert.match(source, /setJobId\(data\.job_id\)/);
  assert.match(source, /历史任务/);
  assert.match(source, /selectedTaskIds/);
  assert.match(source, /CheckSquare/);
  assert.match(source, /Trash2/);
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

test('commentary lets users fetch or manually enter the Gemini analysis model', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /geminiModel/);
  assert.match(source, /\/api\/settings\/gemini-models/);
  assert.match(source, /gemini_model/);
  assert.match(source, /Qwen3\.7-Plus-thinking/);
  assert.match(source, /Gemini 模型/);
  assert.match(source, /获取模型/);
  assert.match(source, /手动填写/);
  assert.match(source, /request\.gemini_model/);
});

test('commentary allows manually entered Gemini models for video-input analysis', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /const selectedGeminiModel = geminiModel\.trim\(\)/);
  assert.match(source, /analysisMode !== 'openai' && !selectedGeminiModel/);
  assert.match(source, /手动填写当前 Key\/Base URL 支持的模型名/);
  assert.doesNotMatch(source, /analysisMode === 'video' && geminiModels\.length === 0/);
});

test('commentary selects the first fetched Gemini model when still using the built-in default', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /DEFAULT_GEMINI_MODEL/);
  assert.match(source, /geminiModel\s*===\s*DEFAULT_GEMINI_MODEL/);
  assert.match(source, /setGeminiModel\(models\[0\]\.id\)/);
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

  assert.match(source, /setStyle\('custom'\)/);
  assert.match(source, /新建自定义风格/);
  assert.doesNotMatch(source, /label: '自定义提示词'/);
  assert.match(source, /customStylePrompt/);
  assert.match(source, /custom_style_prompt/);
  assert.match(source, /自定义风格提示词/);
  assert.match(source, /customStyleOptions/);
  assert.match(source, /CUSTOM_STYLE_STORAGE_KEY/);
  assert.match(source, /添加到下拉框/);
  assert.match(source, /更新下拉框选项/);
  assert.match(source, /handleCreateCustomStyleOption/);
  assert.match(source, /新建自定义解说风格/);
  assert.match(source, /handleDeleteCustomStyleOption/);
  assert.match(source, /删除这个自定义风格/);
  assert.match(source, /event\.stopPropagation\(\)/);
  assert.match(source, /customStyleOptionsRef/);
});

test('commentary custom styles are unique by style name after restart cleanup', () => {
  const options = normalizeCustomStyleOptions([
    { id: 'custom:first', label: '自定义视频解说风格', prompt: '旧提示词' },
    { id: 'custom:second', label: ' 自定义视频解说风格 ', prompt: '新提示词' },
    { id: 'custom:third', label: '另一种风格', prompt: '提示词' },
  ]);

  assert.deepEqual(options.map((item) => item.label), ['自定义视频解说风格', '另一种风格']);
  assert.equal(options[0].id, 'custom:second');
  assert.equal(options[0].prompt, '新提示词');
});

test('commentary task recovery reuses an existing custom style with the same name', () => {
  const options = normalizeCustomStyleOptions([
    { id: 'custom:existing', label: '自定义视频解说风格', prompt: '本地保存的最新版提示词' },
  ]);

  const recovered = findCustomStyleOption(options, '自定义视频解说风格', '历史任务里保存的旧提示词');

  assert.equal(recovered.id, 'custom:existing');
  assert.equal(recovered.prompt, '本地保存的最新版提示词');
});

test('commentary sends saved custom style prompt when a dropdown custom style is selected', () => {
  const options = normalizeCustomStyleOptions([
    { id: 'custom:existing', label: '自定义视频解说风格', prompt: '用保存的风格写，先说画面再加语气。' },
  ]);

  const request = resolveCommentaryStyleRequest('custom:existing', '', options);

  assert.equal(request.style, '自定义视频解说风格');
  assert.equal(request.customStylePrompt, '用保存的风格写，先说画面再加语气。');
  assert.equal(request.isCustomStyle, true);
});

test('commentary clears stale custom prompt for built-in style requests', () => {
  const request = resolveCommentaryStyleRequest('documentary', '上一轮自定义提示词', []);

  assert.equal(request.style, 'documentary');
  assert.equal(request.customStylePrompt, '');
  assert.equal(request.isCustomStyle, false);
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

test('commentary OpenAI status steps follow the edit-first chain', () => {
  const source = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');

  assert.match(source, /OpenAI 兼容多模态：先剪辑再解说/);
  assert.match(source, /检查原片解说音频/);
  assert.match(source, /全片抽帧/);
  assert.match(source, /全片多模态分析/);
  assert.match(source, /生成中间剪辑/);
  assert.match(source, /剪辑片抽帧/);
  assert.match(source, /剪辑片多模态分析/);
  assert.match(source, /基于剪辑片写解说/);
  assert.match(source, /openai_edited_frames/);
  assert.doesNotMatch(source, /label: 'OpenAI 写解说脚本'/);
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

test('commentary log panel can collapse while expanded logs stay internally scrollable', () => {
  const expandedState = getCommentaryLogPanelState(['Queued commentary remix job...', 'Rendering final video...'], true);
  assert.equal(expandedState.countLabel, '2 条日志');
  assert.equal(expandedState.toggleLabel, '收起运行日志');
  assert.match(COMMENTARY_LOG_PANEL_BODY_CLASS, /max-h-\[320px\]/);
  assert.match(COMMENTARY_LOG_PANEL_BODY_CLASS, /overflow-y-auto/);

  const collapsedState = getCommentaryLogPanelState([], false);
  assert.equal(collapsedState.countLabel, '暂无日志');
  assert.equal(collapsedState.toggleLabel, '展开运行日志');
  assert.equal(collapsedState.emptyText, '等待开始...');
});
test('commentary style learner is wired into the sidebar and style APIs', () => {
  const appSource = readFileSync(resolve(import.meta.dirname, '../App.jsx'), 'utf8');
  const learnerSource = readFileSync(resolve(import.meta.dirname, 'CommentaryStyleLearner.jsx'), 'utf8');
  const commentarySource = readFileSync(resolve(import.meta.dirname, 'CommentaryTab.jsx'), 'utf8');
  const translationsSource = readFileSync(resolve(import.meta.dirname, '../i18n/translations.js'), 'utf8');

  assert.match(appSource, /setActiveTab\('style-learning'\)/);
  assert.match(appSource, /<CommentaryStyleLearner/);
  assert.match(appSource, /openAICompatibleConfig=\{\{/);
  assert.match(appSource, /nav\.commentaryStyleLearner/);
  assert.doesNotMatch(learnerSource, /\/api\/settings\/douyin-cookies/);
  assert.match(learnerSource, /\/api\/commentary\/style-learning\/jobs/);
  assert.match(learnerSource, /language/);
  assert.match(learnerSource, /styleLearning\.language/);
  assert.match(learnerSource, /viewFullPrompt/);
  assert.match(learnerSource, /downloadPrompt/);
  assert.match(learnerSource, /downloadStyleJson/);
  assert.match(learnerSource, /max_videos: 100/);
  assert.match(learnerSource, /rankingCount/);
  assert.match(learnerSource, /totalVideos/);
  assert.match(learnerSource, /mergeStyleIntoLocalStorage/);
  assert.match(learnerSource, /useI18n/);
  assert.match(learnerSource, /styleLearning\.description/);
  assert.match(learnerSource, /formatElapsedDuration/);
  assert.match(learnerSource, /styleLearning\.metrics\.elapsed/);
  assert.match(learnerSource, /styleLearning\.elapsedHistory/);
  assert.match(learnerSource, /viewMode/);
  assert.match(learnerSource, /styleLearning\.newMode/);
  assert.match(learnerSource, /styleLearning\.historyMode/);
  assert.match(learnerSource, /noHistorySelection/);
  assert.match(translationsSource, /likes plus saves/);
  assert.match(translationsSource, /elapsedHistory/);
  assert.match(translationsSource, /查看历史/);
  assert.match(translationsSource, /公开主页抓到 \{total\} 条/);
  assert.match(translationsSource, /获取解说风格/);
  assert.match(commentarySource, /\/api\/commentary\/styles/);
  assert.match(commentarySource, /openshorts:commentary-styles-updated/);
});

test('style learner merges learned backend styles into local custom style storage', () => {
  const events = [];
  global.window = {
    localStorage: {
      value: JSON.stringify([
        { id: 'custom:old', label: '旧风格', prompt: '旧提示词' },
      ]),
      getItem() {
        return this.value;
      },
      setItem(_key, value) {
        this.value = value;
      },
    },
    dispatchEvent(event) {
      events.push(event.type);
    },
  };
  global.CustomEvent = class CustomEvent {
    constructor(type, options = {}) {
      this.type = type;
      this.detail = options.detail;
    }
  };

  const merged = mergeStyleIntoLocalStorage({
    id: 'custom:learned',
    label: '学到的风格',
    prompt: '先说画面动作，再给短促判断。',
    custom: true,
  });
  const stored = JSON.parse(window.localStorage.value);

  assert.equal(merged.id, 'custom:learned');
  assert.equal(stored.length, 2);
  assert.equal(stored[1].label, '学到的风格');
  assert.deepEqual(events, ['openshorts:commentary-styles-updated']);

  delete global.window;
  delete global.CustomEvent;
});
