export const COMMENTARY_DEFAULTS = {
  language: 'zh',
  style: 'funny',
  targetDuration: 'full',
  analysisMode: 'openai',
  edgeVoice: 'zh-CN-YunjianNeural',
  originalAudioVolume: 0.3,
  pauseOriginalAudioVolume: 0.6,
  openAIFrameIntervalSeconds: 3,
  openAIMaxFrames: 1800,
  openAISceneMaxKeyframes: 60,
  openAIBatchSize: 32,
  openAIVisualConcurrency: 3,
  commentaryBlockConcurrency: 3,
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
