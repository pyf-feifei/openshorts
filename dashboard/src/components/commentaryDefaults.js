export const COMMENTARY_DEFAULTS = {
  language: 'zh',
  style: 'documentary',
  targetDuration: 'full',
  analysisMode: 'openai',
  edgeVoice: 'zh-CN-YunyangNeural',
  originalAudioVolume: 0.3,
  pauseOriginalAudioVolume: 0.6,
  openAIFrameIntervalSeconds: 3,
  openAIMaxFrames: 1800,
  openAISceneMaxKeyframes: 60,
  openAIBatchSize: 46,
  openAIVisualConcurrency: 5,
  commentaryBlockConcurrency: 5,
  autoVideoSpeed: true,
};

export const DEFAULT_EDGE_VOICES_BY_LANGUAGE = {
  zh: COMMENTARY_DEFAULTS.edgeVoice,
  en: 'en-US-JennyNeural',
  es: 'es-ES-ElviraNeural',
  ja: 'ja-JP-NanamiNeural',
};

export function getDefaultEdgeVoiceForLanguage(language) {
  return DEFAULT_EDGE_VOICES_BY_LANGUAGE[language] || COMMENTARY_DEFAULTS.edgeVoice;
}
