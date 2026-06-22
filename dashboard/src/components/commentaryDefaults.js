export const COMMENTARY_DEFAULTS = {
  language: 'zh',
  style: 'hustle',
  customStylePrompt: '',
  targetDuration: 'two_to_four',
  analysisMode: 'openai',
  edgeVoice: 'zh-CN-YunyangNeural',
  originalAudioVolume: 0.3,
  pauseOriginalAudioVolume: 0.6,
  openAIFrameIntervalSeconds: 3,
  openAIMaxFrames: 1800,
  openAISceneMaxKeyframes: 60,
  openAIBatchSize: 46,
  openAIVisualConcurrency: 2,
  commentaryBlockConcurrency: 2,
  autoVideoSpeed: true,
  backgroundMusicEnabled: false,
  backgroundMusicTrack: 'aodebiao_caravan',
  backgroundMusicVolume: 0.16,
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
