import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Activity, CheckCircle, CheckSquare, ChevronDown, ChevronUp, Copy, Download, FileText, FileVideo, Film, History, Loader2, Mic2, Music2, Play, RefreshCcw, Square, Trash2, Upload, Volume2, X, Youtube } from 'lucide-react'
import { getApiUrl } from '../config'
import { buildGeminiHeaders, getGeminiAccessMissingMessage, hasGeminiAccess as hasGeminiConfigAccess, mergeGeminiEvents } from '../lib/geminiHeaders'
import { buildOpenAICompatibleHeaders, hasOpenAICompatibleAccess } from '../lib/openaiCompatibleHeaders'
import { COMMENTARY_DEFAULTS, getDefaultEdgeVoiceForLanguage } from './commentaryDefaults'
import {
  createCustomStyleId,
  CUSTOM_STYLE_STORAGE_KEY,
  findCustomStyleOption,
  getCustomStyleLabelKey,
  normalizeCustomStyleOptions,
  normalizeCustomStylePrompt,
  resolveCommentaryStyleRequest,
} from './commentaryCustomStyles'
import { COMMENTARY_LOG_PANEL_BODY_CLASS, getCommentaryLogPanelState } from './commentaryLogPanel'

const STYLE_OPTIONS = [
  { id: 'documentary', label: '纪录片解说' },
  { id: 'first_person_hustle', label: '整活第一视角' },
  { id: 'hustle', label: '整活解说' },
  { id: 'news', label: '新闻解读' },
  { id: 'storytelling', label: '故事化旁白' },
  { id: 'funny', label: '轻松吐槽' },
  { id: 'educational', label: '知识科普' },
  { id: 'custom', label: '自定义提示词' },
]

const loadCustomStyleOptions = () => {
  if (typeof window === 'undefined') return []
  try {
    return normalizeCustomStyleOptions(JSON.parse(window.localStorage.getItem(CUSTOM_STYLE_STORAGE_KEY) || '[]'))
  } catch {
    return []
  }
}

const saveCustomStyleOptions = (items) => {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(CUSTOM_STYLE_STORAGE_KEY, JSON.stringify(normalizeCustomStyleOptions(items)))
}

const DURATION_OPTIONS = [
  { id: 'two_to_four', label: '2-4 分钟' },
  { id: 'medium', label: '3-5 分钟' },
  { id: 'short', label: '60-90 秒' },
  { id: 'full', label: '尽量完整（按时间戳解说）' },
]

const FALLBACK_BACKGROUND_MUSIC_TRACKS = [
  { id: 'aodebiao_caravan', label: '默认 奥德彪专属音乐', available: true },
]

const DEFAULT_GEMINI_MODEL = 'Qwen3.7-Plus-thinking'

const VOICE_PREVIEW_TEXT = {
  zh: '你好，这是当前选择的中文解说声音试听。',
  en: 'Hello, this is a preview of the selected English narration voice.',
  es: 'Hola, esta es una prueba de la voz de narración seleccionada.',
  ja: 'こんにちは。これは選択した日本語ナレーション音声のプレビューです。',
}

const EDGE_VOICE_OPTIONS = {
  zh: [
    { id: 'zh-CN-XiaoxiaoNeural', label: '晓晓 · 普通话女声' },
    { id: 'zh-CN-XiaoyiNeural', label: '晓伊 · 普通话女声' },
    { id: 'zh-CN-YunjianNeural', label: '云健 · 普通话男声' },
    { id: 'zh-CN-YunxiNeural', label: '云希 · 普通话男声' },
    { id: 'zh-CN-YunxiaNeural', label: '云夏 · 普通话男声' },
    { id: 'zh-CN-YunyangNeural', label: '云扬 · 普通话男声' },
    { id: 'zh-CN-liaoning-XiaobeiNeural', label: '晓北 · 东北话女声' },
    { id: 'zh-CN-shaanxi-XiaoniNeural', label: '晓妮 · 陕西话女声' },
    { id: 'zh-HK-HiuGaaiNeural', label: '曉佳 · 粤语女声' },
    { id: 'zh-HK-HiuMaanNeural', label: '曉曼 · 粤语女声' },
    { id: 'zh-HK-WanLungNeural', label: '雲龍 · 粤语男声' },
    { id: 'zh-TW-HsiaoChenNeural', label: '曉臻 · 台湾女声' },
    { id: 'zh-TW-HsiaoYuNeural', label: '曉雨 · 台湾女声' },
    { id: 'zh-TW-YunJheNeural', label: '雲哲 · 台湾男声' },
  ],
  en: [
    { id: 'en-US-JennyNeural', label: 'Jenny · US Female' },
    { id: 'en-US-GuyNeural', label: 'Guy · US Male' },
    { id: 'en-US-AriaNeural', label: 'Aria · US Female' },
    { id: 'en-US-AvaNeural', label: 'Ava · US Female' },
    { id: 'en-US-AndrewNeural', label: 'Andrew · US Male' },
    { id: 'en-US-EmmaNeural', label: 'Emma · US Female' },
    { id: 'en-US-BrianNeural', label: 'Brian · US Male' },
    { id: 'en-US-AnaNeural', label: 'Ana · US Female' },
    { id: 'en-US-ChristopherNeural', label: 'Christopher · US Male' },
    { id: 'en-US-EricNeural', label: 'Eric · US Male' },
    { id: 'en-US-MichelleNeural', label: 'Michelle · US Female' },
    { id: 'en-US-RogerNeural', label: 'Roger · US Male' },
    { id: 'en-US-SteffanNeural', label: 'Steffan · US Male' },
    { id: 'en-US-AvaMultilingualNeural', label: 'Ava Multilingual · US Female' },
    { id: 'en-US-AndrewMultilingualNeural', label: 'Andrew Multilingual · US Male' },
    { id: 'en-US-BrianMultilingualNeural', label: 'Brian Multilingual · US Male' },
    { id: 'en-US-EmmaMultilingualNeural', label: 'Emma Multilingual · US Female' },
    { id: 'en-GB-LibbyNeural', label: 'Libby · UK Female' },
    { id: 'en-GB-MaisieNeural', label: 'Maisie · UK Female' },
    { id: 'en-GB-RyanNeural', label: 'Ryan · UK Male' },
    { id: 'en-GB-SoniaNeural', label: 'Sonia · UK Female' },
    { id: 'en-GB-ThomasNeural', label: 'Thomas · UK Male' },
    { id: 'en-AU-NatashaNeural', label: 'Natasha · AU Female' },
    { id: 'en-AU-WilliamMultilingualNeural', label: 'William Multilingual · AU Male' },
    { id: 'en-CA-ClaraNeural', label: 'Clara · CA Female' },
    { id: 'en-CA-LiamNeural', label: 'Liam · CA Male' },
    { id: 'en-HK-YanNeural', label: 'Yan · HK Female' },
    { id: 'en-HK-SamNeural', label: 'Sam · HK Male' },
    { id: 'en-IN-NeerjaExpressiveNeural', label: 'Neerja Expressive · IN Female' },
    { id: 'en-IN-NeerjaNeural', label: 'Neerja · IN Female' },
    { id: 'en-IN-PrabhatNeural', label: 'Prabhat · IN Male' },
    { id: 'en-IE-ConnorNeural', label: 'Connor · IE Male' },
    { id: 'en-IE-EmilyNeural', label: 'Emily · IE Female' },
    { id: 'en-KE-AsiliaNeural', label: 'Asilia · KE Female' },
    { id: 'en-KE-ChilembaNeural', label: 'Chilemba · KE Male' },
    { id: 'en-NZ-MitchellNeural', label: 'Mitchell · NZ Male' },
    { id: 'en-NZ-MollyNeural', label: 'Molly · NZ Female' },
    { id: 'en-NG-AbeoNeural', label: 'Abeo · NG Male' },
    { id: 'en-NG-EzinneNeural', label: 'Ezinne · NG Female' },
    { id: 'en-PH-JamesNeural', label: 'James · PH Male' },
    { id: 'en-PH-RosaNeural', label: 'Rosa · PH Female' },
    { id: 'en-SG-LunaNeural', label: 'Luna · SG Female' },
    { id: 'en-SG-WayneNeural', label: 'Wayne · SG Male' },
    { id: 'en-ZA-LeahNeural', label: 'Leah · ZA Female' },
    { id: 'en-ZA-LukeNeural', label: 'Luke · ZA Male' },
    { id: 'en-TZ-ElimuNeural', label: 'Elimu · TZ Male' },
    { id: 'en-TZ-ImaniNeural', label: 'Imani · TZ Female' },
  ],
  es: [
    { id: 'es-ES-ElviraNeural', label: 'Elvira · Spain Female' },
    { id: 'es-ES-AlvaroNeural', label: 'Alvaro · Spain Male' },
    { id: 'es-ES-XimenaNeural', label: 'Ximena · Spain Female' },
    { id: 'es-MX-DaliaNeural', label: 'Dalia · Mexico Female' },
    { id: 'es-MX-JorgeNeural', label: 'Jorge · Mexico Male' },
    { id: 'es-US-AlonsoNeural', label: 'Alonso · US Male' },
    { id: 'es-US-PalomaNeural', label: 'Paloma · US Female' },
    { id: 'es-AR-ElenaNeural', label: 'Elena · Argentina Female' },
    { id: 'es-AR-TomasNeural', label: 'Tomas · Argentina Male' },
    { id: 'es-BO-MarceloNeural', label: 'Marcelo · Bolivia Male' },
    { id: 'es-BO-SofiaNeural', label: 'Sofia · Bolivia Female' },
    { id: 'es-CL-CatalinaNeural', label: 'Catalina · Chile Female' },
    { id: 'es-CL-LorenzoNeural', label: 'Lorenzo · Chile Male' },
    { id: 'es-CO-GonzaloNeural', label: 'Gonzalo · Colombia Male' },
    { id: 'es-CO-SalomeNeural', label: 'Salome · Colombia Female' },
    { id: 'es-CR-JuanNeural', label: 'Juan · Costa Rica Male' },
    { id: 'es-CR-MariaNeural', label: 'Maria · Costa Rica Female' },
    { id: 'es-CU-BelkysNeural', label: 'Belkys · Cuba Female' },
    { id: 'es-CU-ManuelNeural', label: 'Manuel · Cuba Male' },
    { id: 'es-DO-EmilioNeural', label: 'Emilio · Dominican Republic Male' },
    { id: 'es-DO-RamonaNeural', label: 'Ramona · Dominican Republic Female' },
    { id: 'es-EC-AndreaNeural', label: 'Andrea · Ecuador Female' },
    { id: 'es-EC-LuisNeural', label: 'Luis · Ecuador Male' },
    { id: 'es-SV-LorenaNeural', label: 'Lorena · El Salvador Female' },
    { id: 'es-SV-RodrigoNeural', label: 'Rodrigo · El Salvador Male' },
    { id: 'es-GQ-JavierNeural', label: 'Javier · Equatorial Guinea Male' },
    { id: 'es-GQ-TeresaNeural', label: 'Teresa · Equatorial Guinea Female' },
    { id: 'es-GT-AndresNeural', label: 'Andres · Guatemala Male' },
    { id: 'es-GT-MartaNeural', label: 'Marta · Guatemala Female' },
    { id: 'es-HN-CarlosNeural', label: 'Carlos · Honduras Male' },
    { id: 'es-HN-KarlaNeural', label: 'Karla · Honduras Female' },
    { id: 'es-NI-FedericoNeural', label: 'Federico · Nicaragua Male' },
    { id: 'es-NI-YolandaNeural', label: 'Yolanda · Nicaragua Female' },
    { id: 'es-PA-MargaritaNeural', label: 'Margarita · Panama Female' },
    { id: 'es-PA-RobertoNeural', label: 'Roberto · Panama Male' },
    { id: 'es-PY-MarioNeural', label: 'Mario · Paraguay Male' },
    { id: 'es-PY-TaniaNeural', label: 'Tania · Paraguay Female' },
    { id: 'es-PE-AlexNeural', label: 'Alex · Peru Male' },
    { id: 'es-PE-CamilaNeural', label: 'Camila · Peru Female' },
    { id: 'es-PR-KarinaNeural', label: 'Karina · Puerto Rico Female' },
    { id: 'es-PR-VictorNeural', label: 'Victor · Puerto Rico Male' },
    { id: 'es-UY-MateoNeural', label: 'Mateo · Uruguay Male' },
    { id: 'es-UY-ValentinaNeural', label: 'Valentina · Uruguay Female' },
    { id: 'es-VE-PaolaNeural', label: 'Paola · Venezuela Female' },
    { id: 'es-VE-SebastianNeural', label: 'Sebastian · Venezuela Male' },
  ],
  ja: [
    { id: 'ja-JP-NanamiNeural', label: 'Nanami · Female' },
    { id: 'ja-JP-KeitaNeural', label: 'Keita · Male' },
  ],
}

async function readErrorMessage(res, fallback) {
  const rawText = await res.text().catch(() => '')
  if (!rawText) return fallback
  try {
    const data = JSON.parse(rawText)
    return data.detail || data.message || rawText
  } catch {
    return rawText
  }
}

export default function CommentaryTab({ geminiApiKey, geminiBaseUrl, geminiConfig, openAICompatibleConfig, elevenLabsKey, setGeminiKeyPoolStats }) {
  const [sourceMode, setSourceMode] = useState('url')
  const [url, setUrl] = useState('')
  const [videoFile, setVideoFile] = useState(null)
  const [language, setLanguage] = useState(COMMENTARY_DEFAULTS.language)
  const [style, setStyle] = useState(COMMENTARY_DEFAULTS.style)
  const [customStylePrompt, setCustomStylePrompt] = useState(COMMENTARY_DEFAULTS.customStylePrompt)
  const [customStyleName, setCustomStyleName] = useState('')
  const [customStyleOptions, setCustomStyleOptions] = useState(loadCustomStyleOptions)
  const [targetDuration, setTargetDuration] = useState(COMMENTARY_DEFAULTS.targetDuration)
  const [analysisMode, setAnalysisMode] = useState(COMMENTARY_DEFAULTS.analysisMode)
  const [activeAnalysisMode, setActiveAnalysisMode] = useState(COMMENTARY_DEFAULTS.analysisMode)
  const [geminiModel, setGeminiModel] = useState(DEFAULT_GEMINI_MODEL)
  const [geminiModels, setGeminiModels] = useState([])
  const [geminiModelsStatus, setGeminiModelsStatus] = useState('idle')
  const [geminiModelsError, setGeminiModelsError] = useState('')
  const [ttsProvider, setTtsProvider] = useState('edge')
  const [voiceId, setVoiceId] = useState('21m00Tcm4TlvDq8ikWAM')
  const [edgeVoice, setEdgeVoice] = useState(COMMENTARY_DEFAULTS.edgeVoice)
  const [originalAudioVolume, setOriginalAudioVolume] = useState(COMMENTARY_DEFAULTS.originalAudioVolume)
  const [pauseOriginalAudioVolume, setPauseOriginalAudioVolume] = useState(COMMENTARY_DEFAULTS.pauseOriginalAudioVolume)
  const [openAIFrameIntervalSeconds, setOpenAIFrameIntervalSeconds] = useState(COMMENTARY_DEFAULTS.openAIFrameIntervalSeconds)
  const [openAIMaxFrames, setOpenAIMaxFrames] = useState(COMMENTARY_DEFAULTS.openAIMaxFrames)
  const [openAISceneMaxKeyframes, setOpenAISceneMaxKeyframes] = useState(COMMENTARY_DEFAULTS.openAISceneMaxKeyframes)
  const [openAIBatchSize, setOpenAIBatchSize] = useState(COMMENTARY_DEFAULTS.openAIBatchSize)
  const [openAIVisualConcurrency, setOpenAIVisualConcurrency] = useState(COMMENTARY_DEFAULTS.openAIVisualConcurrency)
  const [commentaryBlockConcurrency, setCommentaryBlockConcurrency] = useState(COMMENTARY_DEFAULTS.commentaryBlockConcurrency)
  const [autoVideoSpeed, setAutoVideoSpeed] = useState(COMMENTARY_DEFAULTS.autoVideoSpeed)
  const [backgroundMusicEnabled, setBackgroundMusicEnabled] = useState(COMMENTARY_DEFAULTS.backgroundMusicEnabled)
  const [backgroundMusicTrack, setBackgroundMusicTrack] = useState(COMMENTARY_DEFAULTS.backgroundMusicTrack)
  const [backgroundMusicVolume, setBackgroundMusicVolume] = useState(COMMENTARY_DEFAULTS.backgroundMusicVolume)
  const [backgroundMusicTracks, setBackgroundMusicTracks] = useState(FALLBACK_BACKGROUND_MUSIC_TRACKS)
  const [subtitles, setSubtitles] = useState(true)
  const [aspectMode, setAspectMode] = useState('auto')
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState('idle')
  const [logs, setLogs] = useState([])
  const [logsExpanded, setLogsExpanded] = useState(true)
  const [backendStage, setBackendStage] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [uploadProgress, setUploadProgress] = useState(null)
  const [uploadPhase, setUploadPhase] = useState('idle')
  const [commentaryTasks, setCommentaryTasks] = useState([])
  const [taskListStatus, setTaskListStatus] = useState('idle')
  const [retryingJobId, setRetryingJobId] = useState(null)
  const [selectedTaskIds, setSelectedTaskIds] = useState([])
  const [deletingTaskIds, setDeletingTaskIds] = useState([])
  const [voicePreviewStatus, setVoicePreviewStatus] = useState('idle')
  const [attachedTaskRequest, setAttachedTaskRequest] = useState(null)
  const audioRef = useRef(null)
  const audioUrlRef = useRef(null)
  const statusPollFailuresRef = useRef(0)
  const eventsMergedRef = useRef(null)
  const customStyleOptionsRef = useRef(customStyleOptions)
  const hasGeminiAccess = hasGeminiConfigAccess(geminiConfig || geminiApiKey)
  const geminiAccessMissingMessage = getGeminiAccessMissingMessage(geminiConfig || geminiApiKey)
  const hasSelectedAnalysisAccess = (mode = analysisMode) => (
    mode === 'openai' ? hasOpenAICompatibleAccess(openAICompatibleConfig) : hasGeminiAccess
  )
  const buildAnalysisHeaders = (mode = analysisMode, extraHeaders = {}) => (
    mode === 'openai'
      ? buildOpenAICompatibleHeaders(openAICompatibleConfig, extraHeaders)
      : buildGeminiHeaders(geminiConfig || geminiApiKey, geminiBaseUrl, extraHeaders)
  )
  const styleOptions = [...STYLE_OPTIONS, ...customStyleOptions]
  const selectedCustomStyle = customStyleOptions.find((item) => item.id === style)
  const isCustomStyleEditorVisible = style === 'custom' || Boolean(selectedCustomStyle)

  useEffect(() => {
    setEdgeVoice(getDefaultEdgeVoiceForLanguage(language))
  }, [language])

  useEffect(() => {
    customStyleOptionsRef.current = normalizeCustomStyleOptions(customStyleOptions)
    saveCustomStyleOptions(customStyleOptions)
  }, [customStyleOptions])

  useEffect(() => {
    return () => {
      if (audioRef.current) audioRef.current.pause()
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current)
    }
  }, [])

  const refreshCommentaryTasks = useCallback(async () => {
    setTaskListStatus('loading')
    try {
      const res = await fetch(getApiUrl('/api/commentary/jobs'))
      if (!res.ok) throw new Error(`Task list failed (${res.status})`)
      const data = await res.json()
      setCommentaryTasks(data.jobs || [])
      setTaskListStatus('idle')
    } catch {
      setTaskListStatus('failed')
    }
  }, [])

  useEffect(() => {
    refreshCommentaryTasks()
  }, [refreshCommentaryTasks])

  useEffect(() => {
    let cancelled = false
    const loadBackgroundMusicTracks = async () => {
      try {
        const res = await fetch(getApiUrl('/api/commentary/background-music'))
        if (!res.ok) throw new Error('Failed to load background music tracks')
        const data = await res.json()
        if (!cancelled && Array.isArray(data.tracks) && data.tracks.length > 0) {
          setBackgroundMusicTracks(data.tracks)
          if (!data.tracks.some((track) => track.id === backgroundMusicTrack)) {
            setBackgroundMusicTrack(data.tracks[0].id)
          }
        }
      } catch {
        if (!cancelled) setBackgroundMusicTracks(FALLBACK_BACKGROUND_MUSIC_TRACKS)
      }
    }
    loadBackgroundMusicTracks()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!jobId || status !== 'processing') return undefined
    const timer = setInterval(async () => {
      try {
        const res = await fetch(getApiUrl(`/api/commentary/status/${jobId}`))
        if (!res.ok) throw new Error(`Status check failed (${res.status})`)
        const data = await res.json()
        statusPollFailuresRef.current = 0
        setLogs(data.logs || [])
        setBackendStage(data.stage ? { stage: data.stage, label: data.stage_label, progress: data.stage_progress } : null)
        setStatus(data.status)
        if (data.request) applyCommentaryRequestToControls(data.request, data.source_type)
        if (data.request?.analysis_mode) setActiveAnalysisMode(data.request.analysis_mode)
        if (data.result) setResult(data.result)
        if (data.status === 'failed') setError((data.logs || []).slice(-1)[0] || 'Generation failed')
        if ((data.status === 'completed' || data.status === 'failed') && data.gemini_events?.length && eventsMergedRef.current !== jobId) {
          eventsMergedRef.current = jobId
          setGeminiKeyPoolStats?.((prev) => mergeGeminiEvents(prev, data.gemini_events))
        }
        if (data.status === 'completed' || data.status === 'failed') refreshCommentaryTasks()
      } catch (e) {
        statusPollFailuresRef.current += 1
        if (statusPollFailuresRef.current >= 5) {
          setError(e.message)
          setStatus('failed')
          return
        }
        setError(`状态检查失败，正在重试... (${statusPollFailuresRef.current}/5)`)
      }
    }, 2500)
    return () => clearInterval(timer)
  }, [jobId, status, setGeminiKeyPoolStats, refreshCommentaryTasks])

  const handleVoicePreview = async () => {
    if (voicePreviewStatus === 'loading') return

    setError('')
    setVoicePreviewStatus('loading')

    try {
      if (audioRef.current) audioRef.current.pause()
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current)
        audioUrlRef.current = null
      }

      const res = await fetch(getApiUrl('/api/commentary/voice-preview'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          language,
          edge_voice: edgeVoice,
          text: VOICE_PREVIEW_TEXT[language] || VOICE_PREVIEW_TEXT.zh,
        }),
      })

      if (!res.ok) {
        const fallbackMessage = `试听失败（HTTP ${res.status}）`
        let message = fallbackMessage
        const rawText = await res.text()
        if (rawText) {
          try {
            const data = JSON.parse(rawText)
            message = data.detail || data.message || rawText
          } catch {
            message = rawText
          }
        }
        throw new Error(message)
      }

      const blob = await res.blob()
      const audioUrl = URL.createObjectURL(blob)
      const audio = new Audio(audioUrl)
      audioRef.current = audio
      audioUrlRef.current = audioUrl
      audio.onended = () => setVoicePreviewStatus('idle')
      audio.onerror = () => {
        setVoicePreviewStatus('idle')
        setError('试听音频播放失败')
      }
      setVoicePreviewStatus('playing')
      await audio.play()
    } catch (e) {
      setVoicePreviewStatus('idle')
      setError(e.message)
    }
  }

  const copyText = async (text) => {
    const value = String(text || '').trim()
    if (!value) return
    try {
      await navigator.clipboard.writeText(value)
    } catch {
      const textarea = document.createElement('textarea')
      textarea.value = value
      textarea.setAttribute('readonly', '')
      textarea.style.position = 'fixed'
      textarea.style.opacity = '0'
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
    }
  }

  const applyCommentaryRequestToControls = (request = {}, sourceType = null) => {
    setAttachedTaskRequest(request)
    if (sourceType) setSourceMode(sourceType === 'file' ? 'file' : 'url')
    if (request.url !== undefined) setUrl(request.url || '')
    if (request.language) setLanguage(request.language)
    if (request.style) {
      const customPrompt = normalizeCustomStylePrompt(request.custom_style_prompt)
      if (customPrompt && request.style !== 'custom') {
        const savedCustomStyle = findCustomStyleOption(customStyleOptionsRef.current, request.style, customPrompt)
        const recoveredOption = savedCustomStyle || {
          id: createCustomStyleId(),
          label: request.style,
          prompt: customPrompt,
          custom: true,
        }
        if (!savedCustomStyle) {
          setCustomStyleOptions((items) => normalizeCustomStyleOptions([...items, recoveredOption]))
          customStyleOptionsRef.current = normalizeCustomStyleOptions([...customStyleOptionsRef.current, recoveredOption])
        }
        setStyle(recoveredOption.id)
        setCustomStyleName(recoveredOption.label)
      } else if (request.style === 'custom') {
        setStyle('custom')
        setCustomStyleName('')
      } else {
        setStyle(request.style)
      }
    }
    if (request.custom_style_prompt !== undefined) setCustomStylePrompt(request.custom_style_prompt || '')
    if (request.target_duration) setTargetDuration(request.target_duration)
    if (request.analysis_mode) setAnalysisMode(request.analysis_mode)
    if (request.gemini_model) setGeminiModel(request.gemini_model)
    if (request.tts_provider) setTtsProvider(request.tts_provider)
    if (request.voice_id) setVoiceId(request.voice_id)
    if (request.edge_voice) setEdgeVoice(request.edge_voice)
    if (request.original_audio_volume !== undefined) setOriginalAudioVolume(request.original_audio_volume)
    if (request.pause_original_audio_volume !== undefined) setPauseOriginalAudioVolume(request.pause_original_audio_volume)
    if (request.openai_frame_interval_seconds !== undefined) setOpenAIFrameIntervalSeconds(request.openai_frame_interval_seconds)
    if (request.openai_max_frames !== undefined) setOpenAIMaxFrames(request.openai_max_frames)
    if (request.openai_scene_max_keyframes !== undefined) setOpenAISceneMaxKeyframes(request.openai_scene_max_keyframes)
    if (request.openai_batch_size !== undefined) setOpenAIBatchSize(request.openai_batch_size)
    if (request.openai_visual_concurrency !== undefined) setOpenAIVisualConcurrency(request.openai_visual_concurrency)
    if (request.commentary_block_concurrency !== undefined) setCommentaryBlockConcurrency(request.commentary_block_concurrency)
    if (request.auto_video_speed !== undefined) setAutoVideoSpeed(Boolean(request.auto_video_speed))
    if (request.background_music_enabled !== undefined) setBackgroundMusicEnabled(Boolean(request.background_music_enabled))
    if (request.background_music_track) setBackgroundMusicTrack(request.background_music_track)
    if (request.background_music_volume !== undefined) setBackgroundMusicVolume(request.background_music_volume)
    if (request.subtitles !== undefined) setSubtitles(Boolean(request.subtitles))
    if (request.aspect_mode) setAspectMode(request.aspect_mode)
  }

  const handleStyleChange = (nextStyle) => {
    setAttachedTaskRequest(null)
    setStyle(nextStyle)
    const customOption = customStyleOptions.find((item) => item.id === nextStyle)
    if (customOption) {
      setCustomStyleName(customOption.label)
      setCustomStylePrompt(customOption.prompt)
    } else if (nextStyle === 'custom') {
      setCustomStyleName('')
      setCustomStylePrompt('')
    }
  }

  const handleSaveCustomStyleOption = () => {
    const label = customStyleName.trim()
    const prompt = customStylePrompt.trim()
    if (!label || !prompt) {
      setError('请填写自定义风格名称和提示词')
      return
    }
    const nextOption = {
      id: selectedCustomStyle?.id || findCustomStyleOption(customStyleOptions, label, prompt)?.id || createCustomStyleId(),
      label,
      prompt,
      custom: true,
    }
    setCustomStyleOptions((items) => {
      const nextLabelKey = getCustomStyleLabelKey(nextOption.label)
      const withoutCurrent = items.filter((item) => item.id !== nextOption.id && getCustomStyleLabelKey(item.label) !== nextLabelKey)
      return normalizeCustomStyleOptions([...withoutCurrent, nextOption])
    })
    setStyle(nextOption.id)
    setError('')
  }

  const createCommentaryJobWithUploadProgress = (headers, requestBody) => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', getApiUrl('/api/commentary/generate'))
    Object.entries(headers).forEach(([key, value]) => {
      if (value !== undefined && value !== null) xhr.setRequestHeader(key, value)
    })
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) {
        setUploadPhase('uploading')
        return
      }
      setUploadPhase('uploading')
      setUploadProgress(Math.min(99, Math.round((event.loaded / event.total) * 100)))
    }
    xhr.onload = () => {
      const rawText = xhr.responseText || ''
      if (xhr.status < 200 || xhr.status >= 300) {
        try {
          const data = JSON.parse(rawText)
          reject(new Error(data.detail || 'Generation failed'))
        } catch {
          reject(new Error(rawText || 'Generation failed'))
        }
        return
      }
      setUploadProgress(100)
      setUploadPhase('creating')
      try {
        resolve(JSON.parse(rawText))
      } catch {
        reject(new Error('Invalid server response'))
      }
    }
    xhr.onerror = () => reject(new Error('视频上传失败，请检查网络或后端服务'))
    xhr.onabort = () => reject(new Error('视频上传已取消'))
    xhr.send(requestBody)
  })

  const attachCommentaryTask = async (task) => {
    let currentTask = task
    try {
      const res = await fetch(getApiUrl(`/api/commentary/status/${task.job_id}`))
      if (res.ok) currentTask = await res.json()
    } catch {
      currentTask = task
    }

    setError(currentTask.error || '')
    if (currentTask.request) applyCommentaryRequestToControls(currentTask.request, currentTask.source_type)
    setActiveAnalysisMode(currentTask.request?.analysis_mode || analysisMode)
    setJobId(currentTask.job_id)
    setStatus(currentTask.status || 'processing')
    setLogs(currentTask.logs || [])
    setLogsExpanded(true)
    setBackendStage(currentTask.stage ? { stage: currentTask.stage, label: currentTask.stage_label, progress: currentTask.stage_progress } : null)
    setResult(currentTask.result || null)
    setUploadProgress(null)
    setUploadPhase('idle')
    statusPollFailuresRef.current = 0
  }

  const resetAttachedTaskIfDeleted = (deletedIds) => {
    if (!jobId || !deletedIds.includes(jobId)) return
    setJobId(null)
    setStatus('idle')
    setLogs([])
    setLogsExpanded(true)
    setBackendStage(null)
    setResult(null)
    setError('')
    setUploadProgress(null)
    setUploadPhase('idle')
    setAttachedTaskRequest(null)
    statusPollFailuresRef.current = 0
  }

  const toggleCommentaryTaskSelection = (taskId) => {
    setSelectedTaskIds((ids) => (
      ids.includes(taskId) ? ids.filter((id) => id !== taskId) : [...ids, taskId]
    ))
  }

  const toggleAllCommentaryTaskSelection = () => {
    const visibleIds = commentaryTasks.map((task) => task.job_id).filter(Boolean)
    setSelectedTaskIds((ids) => (
      visibleIds.length > 0 && visibleIds.every((id) => ids.includes(id)) ? [] : visibleIds
    ))
  }

  const deleteCommentaryTasks = async (taskIds) => {
    const ids = Array.from(new Set((taskIds || []).filter(Boolean)))
    if (ids.length === 0 || deletingTaskIds.length > 0) return
    const message = ids.length === 1
      ? '确定删除这个历史任务？会同时删除对应输出文件，并尝试停止进行中的处理。'
      : `确定删除选中的 ${ids.length} 个历史任务？会同时删除对应输出文件，并尝试停止进行中的处理。`
    if (!window.confirm(message)) return

    setError('')
    setDeletingTaskIds(ids)
    try {
      let res
      if (ids.length === 1) {
        res = await fetch(getApiUrl(`/api/commentary/jobs/${ids[0]}`), { method: 'DELETE' })
      } else {
        res = await fetch(getApiUrl('/api/commentary/jobs/delete'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ job_ids: ids }),
        })
      }
      if (!res.ok) {
        throw new Error(await readErrorMessage(res, '删除任务失败'))
      }
      const data = await res.json().catch(() => ({}))
      if (Array.isArray(data.errors) && data.errors.length > 0) {
        throw new Error(data.errors.map((item) => `${item.job_id}: ${item.error}`).join('\n'))
      }
      resetAttachedTaskIfDeleted(ids)
      setSelectedTaskIds((selected) => selected.filter((id) => !ids.includes(id)))
      setCommentaryTasks((tasks) => tasks.filter((task) => !ids.includes(task.job_id)))
      refreshCommentaryTasks()
    } catch (e) {
      setError(e.message || '删除任务失败')
    } finally {
      setDeletingTaskIds([])
    }
  }

  const retryCommentaryTask = async (task) => {
    const taskAnalysisMode = task.request?.analysis_mode || analysisMode
    const selectedAnalysisMode = jobId === task.job_id ? analysisMode : taskAnalysisMode
    if (!hasSelectedAnalysisAccess(selectedAnalysisMode)) {
      setError(selectedAnalysisMode === 'openai' ? '请先在 Settings 配置 OpenAI 兼容 API URL、Key 和模型' : geminiAccessMissingMessage)
      return
    }
    setError('')
    setActiveAnalysisMode(selectedAnalysisMode)
    setRetryingJobId(task.job_id)
    try {
      const headers = {
        ...buildAnalysisHeaders(selectedAnalysisMode, { 'Content-Type': 'application/json' }),
        ...(elevenLabsKey ? { 'X-ElevenLabs-Key': elevenLabsKey } : {}),
      }
      const res = await fetch(getApiUrl(`/api/commentary/jobs/${task.job_id}/retry`), {
        method: 'POST',
        headers,
        body: JSON.stringify({
          analysis_mode: selectedAnalysisMode,
          gemini_model: selectedAnalysisMode === 'openai' ? undefined : geminiModel.trim(),
          openai_model: selectedAnalysisMode === 'openai' ? openAICompatibleConfig?.model : undefined,
        }),
      })
      if (!res.ok) {
        throw new Error(await readErrorMessage(res, 'Retry failed'))
      }
      const data = await res.json()
      setAttachedTaskRequest(task.request || null)
      setJobId(data.job_id)
      setStatus('processing')
      setLogs([...(task.logs || []), 'Retrying commentary remix from saved task checkpoints...'])
      setLogsExpanded(true)
      setBackendStage({ stage: 'queued', label: '准备重试', progress: null })
      setResult(null)
      setUploadProgress(null)
      setUploadPhase('idle')
      statusPollFailuresRef.current = 0
      refreshCommentaryTasks()
    } catch (e) {
      setError(e.message)
    } finally {
      setRetryingJobId(null)
    }
  }

  const positiveNumberOrDefault = (value, fallback) => {
    const parsed = Number(value)
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
  }

  const buildOpenAISamplingPayload = () => ({
    openai_frame_interval_seconds: positiveNumberOrDefault(openAIFrameIntervalSeconds, COMMENTARY_DEFAULTS.openAIFrameIntervalSeconds),
    openai_max_frames: Math.round(positiveNumberOrDefault(openAIMaxFrames, COMMENTARY_DEFAULTS.openAIMaxFrames)),
    openai_scene_max_keyframes: Math.round(positiveNumberOrDefault(openAISceneMaxKeyframes, COMMENTARY_DEFAULTS.openAISceneMaxKeyframes)),
    openai_batch_size: Math.round(positiveNumberOrDefault(openAIBatchSize, COMMENTARY_DEFAULTS.openAIBatchSize)),
    openai_visual_concurrency: Math.round(positiveNumberOrDefault(openAIVisualConcurrency, COMMENTARY_DEFAULTS.openAIVisualConcurrency)),
    commentary_block_concurrency: Math.round(positiveNumberOrDefault(commentaryBlockConcurrency, COMMENTARY_DEFAULTS.commentaryBlockConcurrency)),
  })

  const fetchGeminiModels = async () => {
    if (!hasGeminiAccess) {
      setGeminiModelsError(geminiAccessMissingMessage)
      return
    }
    setGeminiModelsStatus('loading')
    setGeminiModelsError('')
    try {
      const res = await fetch(getApiUrl('/api/settings/gemini-models'), {
        headers: buildAnalysisHeaders('video'),
      })
      if (!res.ok) {
        throw new Error(await readErrorMessage(res, '获取 Gemini 模型失败'))
      }
      const data = await res.json()
      const models = Array.isArray(data.models) ? data.models : []
      setGeminiModels(models)
      if ((!geminiModel.trim() || geminiModel === DEFAULT_GEMINI_MODEL) && models[0]?.id) setGeminiModel(models[0].id)
    } catch (e) {
      setGeminiModelsError(e.message || '获取 Gemini 模型失败，可以手动填写模型名')
    } finally {
      setGeminiModelsStatus('idle')
    }
  }

  const handleGenerate = async () => {
    const usingFile = sourceMode === 'file'
    const selectedGeminiModel = geminiModel.trim()
    if (!usingFile && !url.trim()) {
      setError('请输入 YouTube URL')
      return
    }
    if (usingFile && !videoFile) {
      setError('请选择要上传的视频文件')
      return
    }
    if (!hasSelectedAnalysisAccess(analysisMode)) {
      setError(analysisMode === 'openai' ? '请先在 Settings 配置 OpenAI 兼容 API URL、Key 和模型' : geminiAccessMissingMessage)
      return
    }
    if (analysisMode !== 'openai' && !selectedGeminiModel) {
      setError('请先选择 Gemini 模型，或手动填写当前 Key/Base URL 支持的模型名。')
      return
    }
    if (ttsProvider === 'elevenlabs' && !elevenLabsKey) {
      setError('选择 ElevenLabs 时需要先在 Settings 配置 ElevenLabs API Key')
      return
    }

    setError('')
    setAttachedTaskRequest(null)
    setActiveAnalysisMode(analysisMode)
    setStatus('processing')
    setLogs([usingFile ? 'Preparing to upload local video...' : 'Starting commentary remix...'])
    setLogsExpanded(true)
    setBackendStage(null)
    setResult(null)
    setUploadProgress(usingFile ? 0 : null)
    setUploadPhase(usingFile ? 'uploading' : 'idle')
    statusPollFailuresRef.current = 0

    try {
      const styleRequest = resolveCommentaryStyleRequest(style, customStylePrompt, customStyleOptions)
      const headers = {
        ...buildAnalysisHeaders(analysisMode),
        ...(elevenLabsKey ? { 'X-ElevenLabs-Key': elevenLabsKey } : {}),
      }
      let requestBody
      if (usingFile) {
        const formData = new FormData()
        formData.append('file', videoFile)
        formData.append('language', language)
        formData.append('style', styleRequest.style)
        formData.append('custom_style_prompt', styleRequest.customStylePrompt)
        formData.append('target_duration', targetDuration)
        formData.append('analysis_mode', analysisMode)
        if (analysisMode === 'openai') {
          formData.append('openai_model', openAICompatibleConfig?.model || '')
          Object.entries(buildOpenAISamplingPayload()).forEach(([key, value]) => {
            formData.append(key, String(value))
          })
        } else {
          formData.append('gemini_model', selectedGeminiModel)
        }
        if (analysisMode !== 'openai' && geminiConfig?.mode === 'official_pool') formData.append('gemini_pool', JSON.stringify(geminiConfig))
        formData.append('tts_provider', ttsProvider)
        if (voiceId) formData.append('voice_id', voiceId)
        if (edgeVoice) formData.append('edge_voice', edgeVoice)
        formData.append('original_audio_volume', String(Number(originalAudioVolume)))
        formData.append('pause_original_audio_volume', String(Number(pauseOriginalAudioVolume)))
        formData.append('auto_video_speed', String(autoVideoSpeed))
        formData.append('background_music_enabled', String(backgroundMusicEnabled))
        formData.append('background_music_track', backgroundMusicTrack)
        formData.append('background_music_volume', String(Number(backgroundMusicVolume)))
        formData.append('subtitles', String(subtitles))
        formData.append('aspect_mode', aspectMode)
        formData.append('vertical', String(aspectMode === '9:16'))
        requestBody = formData
      } else {
        headers['Content-Type'] = 'application/json'
        requestBody = JSON.stringify({
          url: url.trim(),
          language,
          style: styleRequest.style,
          custom_style_prompt: styleRequest.customStylePrompt,
          target_duration: targetDuration,
          analysis_mode: analysisMode,
          gemini_model: analysisMode === 'openai' ? undefined : selectedGeminiModel,
          openai_model: analysisMode === 'openai' ? openAICompatibleConfig?.model : undefined,
          ...(analysisMode === 'openai' ? buildOpenAISamplingPayload() : {}),
          tts_provider: ttsProvider,
          voice_id: voiceId || undefined,
          edge_voice: edgeVoice || undefined,
          original_audio_volume: Number(originalAudioVolume),
          pause_original_audio_volume: Number(pauseOriginalAudioVolume),
          auto_video_speed: autoVideoSpeed,
          background_music_enabled: backgroundMusicEnabled,
          background_music_track: backgroundMusicTrack,
          background_music_volume: Number(backgroundMusicVolume),
          subtitles,
          aspect_mode: aspectMode,
          vertical: aspectMode === '9:16',
        })
      }

      let data
      if (usingFile) {
        data = await createCommentaryJobWithUploadProgress(headers, requestBody)
      } else {
        const res = await fetch(getApiUrl('/api/commentary/generate'), {
          method: 'POST',
          headers,
          body: requestBody,
        })

        if (!res.ok) {
          throw new Error(await readErrorMessage(res, 'Generation failed'))
        }

        data = await res.json()
      }
      setUploadPhase(usingFile ? 'processing' : 'idle')
      setJobId(data.job_id)
      refreshCommentaryTasks()
    } catch (e) {
      setError(e.message)
      setStatus('failed')
      setUploadPhase('idle')
    }
  }

  const formatTaskTime = (value) => {
    if (!value) return ''
    try {
      return new Date(value).toLocaleString()
    } catch {
      return value
    }
  }

  const taskTitle = (task) => task.result?.title || task.source_filename || task.source_value || task.job_id
  const visibleTaskIds = commentaryTasks.map((task) => task.job_id).filter(Boolean)
  const selectedVisibleTaskIds = selectedTaskIds.filter((id) => visibleTaskIds.includes(id))
  const allVisibleTasksSelected = visibleTaskIds.length > 0 && visibleTaskIds.every((id) => selectedTaskIds.includes(id))
  const isDeletingTasks = deletingTaskIds.length > 0

  const latestLog = logs[logs.length - 1] || ''
  const displayAnalysisMode = status === 'idle' ? analysisMode : activeAnalysisMode
  const hasLogMatching = (pattern) => logs.some((line) => pattern.test(line))
  const backendStepState = (stageName, doneStages = []) => {
    if (status === 'completed' || doneStages.includes(backendStage?.stage)) return 'done'
    if (backendStage?.stage === stageName) return 'active'
    return 'pending'
  }
  const sourceStep = {
    label: sourceMode === 'file' ? '上传原视频' : '准备源视频',
    detail: sourceMode === 'file'
      ? uploadProgress === null ? '等待上传' : uploadProgress >= 100 ? '上传完成' : `正在上传 ${uploadProgress}%`
      : hasLogMatching(/Preparing source video|Downloading/i) ? '已开始' : '等待开始',
    state: sourceMode === 'file'
      ? uploadProgress >= 100 ? 'done' : uploadPhase === 'uploading' ? 'active' : status === 'processing' ? 'done' : 'pending'
      : hasLogMatching(/Preparing source video|Downloading/i) ? 'done' : status === 'processing' ? 'active' : 'pending',
    percent: sourceMode === 'file' ? uploadProgress : null,
  }

  const voiceStep = {
    label: '生成语音并同步画面',
    detail: backendStage?.stage === 'voice' ? (backendStage.label || latestLog) : hasLogMatching(/Mixing new voiceover/i) ? '语音和画面已同步' : /Generating synced commentary block|Adding original-audio pause block|Generating commentary voiceover|Creating AI-selected visual edit|Aligning edited visuals/i.test(latestLog) ? latestLog : '等待中',
    state: backendStepState('voice', ['render', 'done']) === 'pending'
      ? hasLogMatching(/Mixing new voiceover/i) ? 'done' : /Generating synced commentary block|Adding original-audio pause block|Generating commentary voiceover|Creating AI-selected visual edit|Aligning edited visuals/i.test(latestLog) ? 'active' : 'pending'
      : backendStepState('voice', ['render', 'done']),
  }

  const renderStep = {
    label: '合成最终视频',
    detail: status === 'completed' ? '生成完成' : backendStage?.stage === 'render' ? (backendStage.label || latestLog) : /Mixing new voiceover|Generating text-timed subtitles|Burning subtitles/i.test(latestLog) ? latestLog : '等待中',
    state: status === 'completed' ? 'done' : backendStage?.stage === 'render' ? 'active' : /Mixing new voiceover|Generating text-timed subtitles|Burning subtitles/i.test(latestLog) ? 'active' : 'pending',
  }

  const openAIFrameMatch = latestLog.match(/Extracted OpenAI-compatible analysis frames:\s*(\d+)\/(\d+)/i)
  const openAIFramePercent = openAIFrameMatch ? Math.round((Number(openAIFrameMatch[1]) / Number(openAIFrameMatch[2])) * 100) : null
  const voiceStarted = backendStage?.stage === 'voice' || hasLogMatching(/Generating synced commentary block|Adding original-audio pause block|Generating \d+ timestamp-synced commentary blocks|Mixing new voiceover/i)
  const renderStarted = backendStage?.stage === 'render' || status === 'completed' || hasLogMatching(/Mixing new voiceover|Generating text-timed subtitles|Burning subtitles/i)
  const normalizedOpenAIStage = backendStage?.stage === 'openai_frames'
    ? 'openai_source_frames'
    : backendStage?.stage === 'openai'
      ? 'openai_source_analysis'
      : backendStage?.stage
  const openAIStageOrder = [
    'transcribe',
    'openai_audio',
    'openai_source_frames',
    'openai_source_analysis',
    'openai_edit',
    'openai_edited_frames',
    'openai_edited_analysis',
    'openai_final_script',
    'voice',
    'render',
    'done',
  ]
  const openAIStageRank = (stageName) => openAIStageOrder.indexOf(stageName)
  const openAIHasReachedStage = (stageName) => {
    if (status === 'completed') return true
    const currentRank = openAIStageRank(normalizedOpenAIStage)
    const targetRank = openAIStageRank(stageName)
    return currentRank >= targetRank && targetRank >= 0
  }
  const openAIIsAfterStage = (stageName) => {
    if (status === 'completed') return true
    const currentRank = openAIStageRank(normalizedOpenAIStage)
    const targetRank = openAIStageRank(stageName)
    return currentRank > targetRank && targetRank >= 0
  }
  const openAIStepState = (stageName, started, done) => {
    if (done || openAIIsAfterStage(stageName)) return 'done'
    if (normalizedOpenAIStage === stageName || started) return 'active'
    return 'pending'
  }

  const openAITranscriptStarted = backendStage?.stage === 'transcribe' || hasLogMatching(/Transcribing full video|Transcribing video with Faster-Whisper|Reusing cached Faster-Whisper transcript/i)
  const openAIAudioStarted = openAIHasReachedStage('openai_audio') || hasLogMatching(/Testing whether the configured OpenAI-compatible model can inspect source audio|OpenAI-compatible source audio analysis is supported|Configured OpenAI-compatible model did not confirm source audio support|OpenAI-compatible audio analysis failed|OpenAI-compatible audio probe/i)
  const openAISourceFrameStarted = openAIHasReachedStage('openai_source_frames') || hasLogMatching(/Extracting dense timestamped frames|Extracted OpenAI-compatible analysis frames|Detecting scenes for OpenAI-compatible scene-aware/i)
  const openAIEditStarted = openAIHasReachedStage('openai_edit') || hasLogMatching(/OpenAI-compatible edit-first flow locked visual cut/i)
  const openAIEditedFrameStarted = openAIHasReachedStage('openai_edited_frames') || hasLogMatching(/Extracting frames from the intermediate edited video/i)
  const openAIEditedAnalysisStarted = openAIHasReachedStage('openai_edited_analysis') || hasLogMatching(/Analyzing the intermediate edited video with OpenAI-compatible multimodal model/i)
  const openAIFinalScriptStarted = openAIHasReachedStage('openai_final_script') || hasLogMatching(/Writing final commentary from the edited-video analysis|OpenAI-compatible model is writing|script validation failed|returned a corrected commentary script|returned a repaired commentary script|OpenAI-compatible model returned/i)
  const openAISourceAnalysisStarted = openAIHasReachedStage('openai_source_analysis') || (
    !openAIEditedAnalysisStarted && hasLogMatching(/OpenAI-compatible multimodal visual analysis|Reusing cached full-source OpenAI-compatible multimodal visual analysis/i)
  )
  const openAITranscriptDone = openAIAudioStarted || openAISourceFrameStarted || openAISourceAnalysisStarted || openAIEditStarted || voiceStarted || renderStarted
  const openAIAudioDone = openAISourceFrameStarted || openAISourceAnalysisStarted || openAIEditStarted || voiceStarted || renderStarted
  const openAISourceFrameDone = openAISourceAnalysisStarted || openAIEditStarted || voiceStarted || renderStarted
  const openAISourceAnalysisDone = openAIEditStarted || openAIEditedFrameStarted || voiceStarted || renderStarted
  const openAIEditDone = openAIEditedFrameStarted || openAIEditedAnalysisStarted || openAIFinalScriptStarted || voiceStarted || renderStarted
  const openAIEditedFrameDone = openAIEditedAnalysisStarted || openAIFinalScriptStarted || voiceStarted || renderStarted
  const openAIEditedAnalysisDone = openAIFinalScriptStarted || voiceStarted || renderStarted
  const openAIFinalScriptDone = voiceStarted || renderStarted
  const openAIAudioDetail = openAIAudioDone
    ? hasLogMatching(/OpenAI-compatible source audio analysis is supported/i)
      ? 'API 音频分析已完成'
      : hasLogMatching(/Configured OpenAI-compatible model did not confirm source audio support|OpenAI-compatible audio analysis failed/i)
        ? 'API 不支持音频，已用转录兜底'
        : '音频检查完成'
    : openAIAudioStarted
      ? backendStage?.stage === 'openai_audio' ? (backendStage.label || latestLog) : latestLog
      : '等待中'

  const openAIAnalysisSteps = [
    {
      label: '转录完整视频',
      detail: openAITranscriptDone ? '转录完成' : /Transcribing full video|Transcribing video with Faster-Whisper/i.test(latestLog) ? latestLog : '等待中',
      state: openAIStepState('transcribe', openAITranscriptStarted, openAITranscriptDone),
    },
    {
      label: '检查原片解说音频',
      detail: openAIAudioDetail,
      state: openAIStepState('openai_audio', openAIAudioStarted, openAIAudioDone),
    },
    {
      label: '全片抽帧',
      detail: openAISourceFrameDone ? '全片抽帧完成' : openAIFrameMatch ? `正在抽帧 ${openAIFrameMatch[1]}/${openAIFrameMatch[2]}` : /Extracting dense timestamped frames|Detecting scenes for OpenAI-compatible scene-aware/i.test(latestLog) ? latestLog : backendStage?.stage === 'openai_source_frames' ? (backendStage.label || latestLog) : '等待中',
      state: openAIStepState('openai_source_frames', openAISourceFrameStarted, openAISourceFrameDone),
      percent: openAISourceFrameDone ? 100 : openAIFramePercent,
    },
    {
      label: '全片多模态分析',
      detail: openAISourceAnalysisDone ? '全片视觉分析完成' : /OpenAI-compatible multimodal visual analysis|Reusing cached full-source OpenAI-compatible multimodal visual analysis/i.test(latestLog) && !openAIEditedAnalysisStarted ? latestLog : backendStage?.stage === 'openai_source_analysis' ? (backendStage.label || latestLog) : '等待中',
      state: openAIStepState('openai_source_analysis', openAISourceAnalysisStarted, openAISourceAnalysisDone),
    },
    {
      label: '生成中间剪辑',
      detail: openAIEditDone ? '中间剪辑已生成' : /OpenAI-compatible edit-first flow locked visual cut/i.test(latestLog) ? latestLog : backendStage?.stage === 'openai_edit' ? (backendStage.label || latestLog) : '等待中',
      state: openAIStepState('openai_edit', openAIEditStarted, openAIEditDone),
    },
    {
      label: '剪辑片抽帧',
      detail: openAIEditedFrameDone ? '剪辑片抽帧完成' : openAIEditedFrameStarted && openAIFrameMatch ? `正在抽帧 ${openAIFrameMatch[1]}/${openAIFrameMatch[2]}` : /Extracting frames from the intermediate edited video|Detecting scenes for OpenAI-compatible scene-aware/i.test(latestLog) && openAIEditedFrameStarted ? latestLog : backendStage?.stage === 'openai_edited_frames' ? (backendStage.label || latestLog) : '等待中',
      state: openAIStepState('openai_edited_frames', openAIEditedFrameStarted, openAIEditedFrameDone),
      percent: openAIEditedFrameDone ? 100 : openAIEditedFrameStarted ? openAIFramePercent : null,
    },
    {
      label: '剪辑片多模态分析',
      detail: openAIEditedAnalysisDone ? '剪辑片视觉分析完成' : openAIEditedAnalysisStarted && /Analyzing the intermediate edited video|OpenAI-compatible multimodal visual analysis/i.test(latestLog) ? latestLog : backendStage?.stage === 'openai_edited_analysis' ? (backendStage.label || latestLog) : '等待中',
      state: openAIStepState('openai_edited_analysis', openAIEditedAnalysisStarted, openAIEditedAnalysisDone),
    },
    {
      label: '基于剪辑片写解说',
      detail: openAIFinalScriptDone ? '最终解说脚本已通过校验' : openAIFinalScriptStarted ? latestLog : '等待中',
      state: openAIStepState('openai_final_script', openAIFinalScriptStarted, openAIFinalScriptDone),
    },
  ]

  const currentGeminiAnalysisSteps = [
    {
      label: '转录并抽取关键帧',
      detail: hasLogMatching(/Generating original commentary script|Gemini is analyzing/i) ? '视觉上下文已准备' : /Transcribing full video|Extracting keyframes/i.test(latestLog) ? latestLog : '等待中',
      state: hasLogMatching(/Generating original commentary script|Gemini is analyzing/i) ? 'done' : /Transcribing full video|Extracting keyframes/i.test(latestLog) ? 'active' : 'pending',
    },
    {
      label: 'Gemini 分析并写解说',
      detail: backendStage?.stage === 'gemini' ? (backendStage.label || latestLog) : hasLogMatching(/returned a commentary script/i) ? '脚本已返回' : /Gemini is analyzing|Generating original commentary script/i.test(latestLog) ? latestLog : '等待中',
      state: backendStepState('gemini', ['voice', 'render', 'done']) === 'pending'
        ? hasLogMatching(/returned a commentary script/i) ? 'done' : /Gemini is analyzing|Generating original commentary script/i.test(latestLog) ? 'active' : 'pending'
        : backendStepState('gemini', ['voice', 'render', 'done']),
    },
  ]

  const geminiVideoAnalysisSteps = [
    {
      label: '压缩 Gemini 分析视频',
      detail: backendStage?.stage === 'analysis_compress' && typeof backendStage.progress === 'number'
        ? `正在压缩 ${backendStage.progress}%`
        : latestLog.match(/Compressing .*?(\d+)%/)?.[1]
          ? `正在压缩 ${latestLog.match(/Compressing .*?(\d+)%/)?.[1]}%`
          : hasLogMatching(/Gemini analysis video ready|No-audio Gemini analysis video ready/i) ? '压缩完成' : '等待中',
      state: backendStepState('analysis_compress', ['analysis_upload', 'gemini', 'voice', 'render', 'done']) === 'pending'
        ? hasLogMatching(/Gemini analysis video ready|No-audio Gemini analysis video ready/i)
          ? 'done'
          : /Compressing Gemini analysis video|Compressing no-audio Gemini analysis video|creating no-audio fallback/i.test(latestLog)
            ? 'active'
            : 'pending'
        : backendStepState('analysis_compress', ['analysis_upload', 'gemini', 'voice', 'render', 'done']),
      percent: backendStage?.stage === 'analysis_compress' && typeof backendStage.progress === 'number'
        ? backendStage.progress
        : Number(latestLog.match(/Compressing .*?(\d+)%/)?.[1] || '') || null,
    },
    {
      label: '上传 Gemini 分析副本',
      detail: backendStage?.stage === 'analysis_upload' ? (backendStage.label || latestLog) : hasLogMatching(/ready for model analysis/i) ? 'Gemini 文件已就绪' : /Uploading 360p Gemini|Files API processing|waiting for Files API/i.test(latestLog) ? latestLog : '等待中',
      state: backendStepState('analysis_upload', ['gemini', 'voice', 'render', 'done']) === 'pending'
        ? hasLogMatching(/ready for model analysis/i) ? 'done' : /Uploading 360p Gemini|Files API processing|waiting for Files API/i.test(latestLog) ? 'active' : 'pending'
        : backendStepState('analysis_upload', ['gemini', 'voice', 'render', 'done']),
    },
    ...currentGeminiAnalysisSteps.slice(1),
  ]

  const progressSteps = [
    sourceStep,
    ...(displayAnalysisMode === 'openai' ? openAIAnalysisSteps : displayAnalysisMode === 'video' ? geminiVideoAnalysisSteps : currentGeminiAnalysisSteps),
    voiceStep,
    renderStep,
  ]

  const submitLabel = status === 'processing'
    ? sourceMode === 'file' && uploadPhase === 'uploading' && uploadProgress !== null
      ? `正在上传视频 ${uploadProgress}%`
      : sourceMode === 'file' && uploadPhase === 'creating'
        ? '上传完成，正在创建任务...'
        : '生成中...'
    : '生成二创解说视频'

  const displayedOpenAIFrameIntervalSeconds = attachedTaskRequest?.openai_frame_interval_seconds ?? openAIFrameIntervalSeconds
  const displayedGeminiModel = attachedTaskRequest?.gemini_model ?? geminiModel
  const displayedOpenAIMaxFrames = attachedTaskRequest?.openai_max_frames ?? openAIMaxFrames
  const displayedOpenAISceneMaxKeyframes = attachedTaskRequest?.openai_scene_max_keyframes ?? openAISceneMaxKeyframes
  const displayedOpenAIBatchSize = attachedTaskRequest?.openai_batch_size ?? openAIBatchSize
  const displayedOpenAIVisualConcurrency = attachedTaskRequest?.openai_visual_concurrency ?? openAIVisualConcurrency
  const displayedCommentaryBlockConcurrency = attachedTaskRequest?.commentary_block_concurrency ?? commentaryBlockConcurrency
  const displayedAutoVideoSpeed = attachedTaskRequest?.auto_video_speed ?? autoVideoSpeed
  const displayedBackgroundMusicEnabled = attachedTaskRequest?.background_music_enabled ?? backgroundMusicEnabled
  const displayedBackgroundMusicTrack = attachedTaskRequest?.background_music_track ?? backgroundMusicTrack
  const displayedBackgroundMusicVolume = attachedTaskRequest?.background_music_volume ?? backgroundMusicVolume
  const selectedBackgroundMusicTrack = backgroundMusicTracks.find((track) => track.id === displayedBackgroundMusicTrack) || backgroundMusicTracks[0]
  const speedSummary = result?.auto_video_speed_summary
  const commentaryEpisodes = Array.isArray(result?.episodes) ? result.episodes : []
  const episodePlan = result?.episode_plan
  const publishTitle = result?.publish_title || result?.title || ''
  const publishDescription = result?.publish_description || [
    result?.summary,
    Array.isArray(result?.hashtags) ? result.hashtags.join(' ') : '',
  ].filter(Boolean).join('\n\n')
  const logPanelState = getCommentaryLogPanelState(logs, logsExpanded)
  const commentaryCovers = [
    result?.cover_landscape_url ? { label: '横封面 4:3', url: result.cover_landscape_url, className: 'aspect-[4/3]' } : null,
    result?.cover_portrait_url ? { label: '竖封面 3:4', url: result.cover_portrait_url, className: 'aspect-[3/4]' } : null,
  ].filter(Boolean)

  const reset = () => {
    setStatus('idle')
    setJobId(null)
    setAttachedTaskRequest(null)
    setLogs([])
    setLogsExpanded(true)
    setBackendStage(null)
    setResult(null)
    setError('')
    setUploadProgress(null)
    setUploadPhase('idle')
    statusPollFailuresRef.current = 0
  }

  return (
    <div className="h-full overflow-y-auto p-6 md:p-8 animate-[fadeIn_0.3s_ease-out]">
      <div className="max-w-[88rem] mx-auto space-y-8">
        <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 text-xs font-medium mb-4">
              <Mic2 size={14} /> Commentary Remix
            </div>
            <h1 className="text-3xl md:text-4xl font-black bg-gradient-to-b from-white to-white/60 bg-clip-text text-transparent">
              整视频二创解说
            </h1>
            <p className="text-zinc-400 mt-3 max-w-2xl">
              输入 YouTube 链接或上传本地视频，自动转录、生成原创解说稿、合成旁白，并与原视频画面混音输出一个完整解说视频。
            </p>
          </div>
          {status !== 'idle' && (
            <button onClick={reset} className="btn-secondary flex items-center gap-2">
              <RefreshCcw size={16} /> 新任务
            </button>
          )}
        </div>

        <div className="grid xl:grid-cols-[minmax(0,1fr)_minmax(360px,0.55fr)] gap-6">
          <div className="glass-panel p-6 space-y-5">
            <div className="space-y-3">
              <div className="flex gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-1">
                <button
                  type="button"
                  onClick={() => setSourceMode('url')}
                  disabled={status === 'processing'}
                  className={`flex-1 inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all ${sourceMode === 'url' ? 'bg-cyan-500/15 text-cyan-200' : 'text-zinc-400 hover:text-white'}`}
                >
                  <Youtube size={16} /> YouTube URL
                </button>
                <button
                  type="button"
                  onClick={() => setSourceMode('file')}
                  disabled={status === 'processing'}
                  className={`flex-1 inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all ${sourceMode === 'file' ? 'bg-cyan-500/15 text-cyan-200' : 'text-zinc-400 hover:text-white'}`}
                >
                  <Upload size={16} /> 上传视频
                </button>
              </div>

              {sourceMode === 'url' ? (
                <div>
                  <label className="block text-sm text-zinc-300 mb-2">YouTube URL</label>
                  <div className="relative">
                    <Youtube size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
                    <input
                      value={url}
                      onChange={(e) => setUrl(e.target.value)}
                      className="input-field pl-10"
                      placeholder="https://www.youtube.com/watch?v=..."
                      disabled={status === 'processing'}
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <label className="block text-sm text-zinc-300 mb-2">上传本地视频</label>
                  <div className={`rounded-xl border border-dashed p-4 transition-all ${videoFile ? 'border-cyan-500/40 bg-cyan-500/5' : 'border-white/15 bg-white/[0.03]'}`}>
                    {videoFile ? (
                      <div className="flex items-center gap-3">
                        <FileVideo className="shrink-0 text-cyan-300" size={22} />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm font-medium text-zinc-100">{videoFile.name}</div>
                          <div className="text-xs text-zinc-500">{(videoFile.size / 1024 / 1024).toFixed(1)} MB</div>
                        </div>
                        {uploadProgress !== null && (
                          <div className="hidden sm:flex min-w-[120px] flex-col gap-1">
                            <div className="flex items-center justify-between text-[11px] text-cyan-200">
                              <span>{uploadProgress >= 100 ? '上传完成' : '上传中'}</span>
                              <span>{uploadProgress}%</span>
                            </div>
                            <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
                              <div className="h-full rounded-full bg-cyan-400 transition-all" style={{ width: `${uploadProgress}%` }} />
                            </div>
                          </div>
                        )}
                        <button
                          type="button"
                          onClick={() => setVideoFile(null)}
                          disabled={status === 'processing'}
                          className="rounded-lg p-2 text-zinc-400 hover:bg-white/10 hover:text-white disabled:opacity-50"
                          aria-label="移除上传视频"
                        >
                          <X size={16} />
                        </button>
                      </div>
                    ) : (
                      <label className="block cursor-pointer text-center">
                        <input
                          type="file"
                          accept="video/*"
                          className="hidden"
                          disabled={status === 'processing'}
                          onChange={(e) => setVideoFile(e.target.files?.[0] || null)}
                        />
                        <Upload className="mx-auto mb-2 text-zinc-500" size={24} />
                        <div className="text-sm text-zinc-300">点击选择本地视频</div>
                        <div className="mt-1 text-xs text-zinc-500">上传后服务器会生成 360p Gemini 分析副本；最终剪辑仍使用原始上传视频。</div>
                      </label>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-zinc-300 mb-2">输出语言</label>
                <select value={language} onChange={(e) => setLanguage(e.target.value)} className="input-field">
                  <option value="zh">中文</option>
                  <option value="en">English</option>
                  <option value="es">Español</option>
                  <option value="ja">日本語</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-zinc-300 mb-2">解说风格</label>
                <select value={style} onChange={(e) => handleStyleChange(e.target.value)} className="input-field">
                  {styleOptions.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
                </select>
              </div>
              {isCustomStyleEditorVisible && (
                <div className="md:col-span-2">
                  <label className="block text-sm text-zinc-300 mb-2">自定义风格名称</label>
                  <input
                    value={customStyleName}
                    onChange={(e) => {
                      setAttachedTaskRequest(null)
                      setCustomStyleName(e.target.value)
                    }}
                    className="input-field mb-3"
                    maxLength={40}
                    placeholder="例如：第一人称紧张整活"
                  />
                  <label className="block text-sm text-zinc-300 mb-2">自定义风格提示词</label>
                  <textarea
                    value={customStylePrompt}
                    onChange={(e) => {
                      setAttachedTaskRequest(null)
                      setCustomStylePrompt(e.target.value)
                    }}
                    className="input-field min-h-[120px] resize-y"
                    maxLength={2000}
                    placeholder="例如：用第一人称紧张整活口吻，句子短一点，先说画面动作，再加反差吐槽；不要脱离画面编剧情。"
                  />
                  <div className="mt-3 flex flex-col sm:flex-row sm:items-center gap-3">
                    <button type="button" onClick={handleSaveCustomStyleOption} className="btn-secondary text-sm px-4 py-2">
                      {selectedCustomStyle ? '更新下拉框选项' : '添加到下拉框'}
                    </button>
                    <p className="text-xs text-zinc-500">保存后会出现在“解说风格”下拉框里，并自动使用这段提示词。</p>
                  </div>
                </div>
              )}
              <div>
                <label className="block text-sm text-zinc-300 mb-2">目标长度</label>
                <select value={targetDuration} onChange={(e) => setTargetDuration(e.target.value)} className="input-field">
                  {DURATION_OPTIONS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm text-zinc-300 mb-2">AI 分析模式</label>
                <select value={analysisMode} onChange={(e) => setAnalysisMode(e.target.value)} className="input-field">
                  <option value="current">当前模式：转录文本 + 关键帧</option>
                  <option value="video">Gemini 视频输入：完整视频分析</option>
                  <option value="openai">OpenAI 兼容多模态：先剪辑再解说</option>
                </select>
                <p className="text-xs text-zinc-500 mt-1">Gemini 视频模式会上传 360p 分析副本；OpenAI 兼容模式会先转录并抽帧分析全片，生成中间剪辑后再分析剪辑片，最后按剪辑片画面写解说。最终剪辑仍使用高清源视频。</p>
              </div>
              {analysisMode !== 'openai' && (
                <div className="sm:col-span-2 rounded-xl border border-white/10 bg-white/[0.03] p-4 space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium text-zinc-100">Gemini 模型</div>
                      <p className="text-xs text-zinc-500 mt-1">用于二创解说的 Gemini 分析和脚本生成；Gemini 视频输入模式需要先点击获取模型，也可以手动填写代理支持的模型名。</p>
                    </div>
                    <button type="button" onClick={fetchGeminiModels} disabled={geminiModelsStatus === 'loading'} className="btn-secondary inline-flex items-center justify-center gap-2 text-sm px-4 py-2 disabled:opacity-60">
                      {geminiModelsStatus === 'loading' && <Loader2 size={14} className="animate-spin" />}
                      {geminiModelsStatus === 'loading' ? '获取中...' : '获取模型'}
                    </button>
                  </div>
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-zinc-300 mb-2">模型列表</label>
                      <select
                        value={geminiModels.some((model) => model.id === geminiModel) ? geminiModel : ''}
                        onChange={(e) => {
                          setAttachedTaskRequest(null)
                          if (e.target.value) setGeminiModel(e.target.value)
                        }}
                        className="input-field"
                      >
                        <option value="">{geminiModels.length ? '选择已获取模型' : '先点击获取模型'}</option>
                        {geminiModels.map((model) => (
                          <option key={model.id} value={model.id}>{model.display_name || model.id}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-300 mb-2">手动填写</label>
                      <input
                        value={geminiModel}
                        onChange={(e) => {
                          setAttachedTaskRequest(null)
                          setGeminiModel(e.target.value)
                        }}
                        className="input-field font-mono"
                        placeholder="Qwen3.7-Plus-thinking"
                      />
                    </div>
                  </div>
                  {geminiModelsError && <p className="text-xs text-yellow-300">{geminiModelsError}</p>}
                </div>
              )}
              {analysisMode === 'openai' && (
                <div className="sm:col-span-2 rounded-xl border border-cyan-500/20 bg-cyan-500/[0.06] p-4 space-y-4">
                  <div>
                    <div className="text-sm font-medium text-cyan-100">OpenAI 兼容抽帧设置</div>
                    <p className="text-xs text-zinc-400 mt-1">这些参数仅在 OpenAI 兼容多模态模式生效；Gemini 模式会忽略。更密集抽帧能提升长视频画面理解，但会增加处理时间和 API 成本。</p>
                  </div>
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-zinc-300 mb-2">抽帧间隔（秒）</label>
                      <input type="number" min="1" max="60" step="0.5" value={displayedOpenAIFrameIntervalSeconds} onChange={(e) => { setAttachedTaskRequest(null); setOpenAIFrameIntervalSeconds(e.target.value) }} className="input-field" />
                      <p className="text-xs text-zinc-500 mt-1">每隔多少秒采样一帧；数值越小，画面理解越细，但会增加抽帧时间和多模态调用成本。默认 3 秒。</p>
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-300 mb-2">全片最多分析帧数</label>
                      <input type="number" min="1" max="2000" step="1" value={displayedOpenAIMaxFrames} onChange={(e) => { setAttachedTaskRequest(null); setOpenAIMaxFrames(e.target.value) }} className="input-field" />
                      <p className="text-xs text-zinc-500 mt-1">限制整条视频最多发送给 OpenAI 兼容模型的帧数；上限越高，批次数、耗时和费用越高。默认 1800。</p>
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-300 mb-2">单场景最多关键帧</label>
                      <input type="number" min="1" max="600" step="1" value={displayedOpenAISceneMaxKeyframes} onChange={(e) => { setAttachedTaskRequest(null); setOpenAISceneMaxKeyframes(e.target.value) }} className="input-field" />
                      <p className="text-xs text-zinc-500 mt-1">场景感知抽帧时，每个镜头/场景最多保留多少关键帧；动态场景可用更高值覆盖细节。默认 60。</p>
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-300 mb-2">每批图片数</label>
                      <input type="number" min="1" max="128" step="1" value={displayedOpenAIBatchSize} onChange={(e) => { setAttachedTaskRequest(null); setOpenAIBatchSize(e.target.value) }} className="input-field" />
                      <p className="text-xs text-zinc-500 mt-1">每次多模态请求携带的图片数量；如果模型或网关限制较低，可以调小。默认 46，最大 128。</p>
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-300 mb-2">视觉分析并发数</label>
                      <input type="number" min="1" max="8" step="1" value={displayedOpenAIVisualConcurrency} onChange={(e) => { setAttachedTaskRequest(null); setOpenAIVisualConcurrency(e.target.value) }} className="input-field" />
                      <p className="text-xs text-zinc-500 mt-1">同时请求多少个视觉 batch；提高可加速全片分析，但可能触发限流。默认 2。</p>
                    </div>
                    <div>
                      <label className="block text-sm text-zinc-300 mb-2">解说分块生成并发数</label>
                      <input type="number" min="1" max="8" step="1" value={displayedCommentaryBlockConcurrency} onChange={(e) => { setAttachedTaskRequest(null); setCommentaryBlockConcurrency(e.target.value) }} className="input-field" />
                      <p className="text-xs text-zinc-500 mt-1">整视频二创解说时，同时生成多少个配音/画面同步 block；提高可加速语音阶段，但可能触发 TTS 限流或增加本机负载。默认 2。</p>
                    </div>
                  </div>
                </div>
              )}
              <div>
                <label className="block text-sm text-zinc-300 mb-2">TTS 引擎</label>
                <select value={ttsProvider} onChange={(e) => setTtsProvider(e.target.value)} className="input-field">
                  <option value="edge">Edge TTS 免费</option>
                  <option value="elevenlabs">ElevenLabs 高质量</option>
                </select>
              </div>
              {ttsProvider === 'edge' ? (
                <div>
                  <label className="block text-sm text-zinc-300 mb-2">Edge 语音</label>
                  <div className="flex gap-2">
                    <select value={edgeVoice} onChange={(e) => setEdgeVoice(e.target.value)} className="input-field flex-1">
                      {(EDGE_VOICE_OPTIONS[language] || EDGE_VOICE_OPTIONS.zh).map((item) => (
                        <option key={item.id} value={item.id}>{item.label}</option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={handleVoicePreview}
                      disabled={status === 'processing' || voicePreviewStatus === 'loading'}
                      className="shrink-0 inline-flex items-center justify-center gap-2 rounded-xl border border-cyan-500/30 bg-cyan-500/10 px-4 py-3 text-sm font-medium text-cyan-200 hover:bg-cyan-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                      {voicePreviewStatus === 'loading' ? <Loader2 size={16} className="animate-spin" /> : voicePreviewStatus === 'playing' ? <Volume2 size={16} /> : <Play size={16} />}
                      试听
                    </button>
                  </div>
                </div>
              ) : (
                <div>
                  <label className="block text-sm text-zinc-300 mb-2">ElevenLabs Voice ID</label>
                  <input value={voiceId} onChange={(e) => setVoiceId(e.target.value)} className="input-field" placeholder="21m00Tcm4TlvDq8ikWAM" />
                </div>
              )}
            </div>

            <div className="grid sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-zinc-300 mb-2">解说时原视频音量：{Math.round(originalAudioVolume * 100)}%</label>
                <input
                  type="range"
                  min="0"
                  max="0.5"
                  step="0.01"
                  value={originalAudioVolume}
                  onChange={(e) => setOriginalAudioVolume(e.target.value)}
                  className="w-full"
                />
                <p className="text-xs text-zinc-500 mt-1">控制 AI 解说说话时保留多少原片声音，建议 5%-10%。</p>
              </div>
              <div>
                <label className="block text-sm text-zinc-300 mb-2">无解说片段原视频音量：{Math.round(pauseOriginalAudioVolume * 100)}%</label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={pauseOriginalAudioVolume}
                  onChange={(e) => setPauseOriginalAudioVolume(e.target.value)}
                  className="w-full"
                />
                <p className="text-xs text-zinc-500 mt-1">控制 pause 无解说片段的原声大小，想保留现场声可设为 50%-100%。</p>
              </div>
            </div>

            <div>
              <label className="block text-sm text-zinc-300 mb-2">视频比例</label>
              <select value={aspectMode} onChange={(e) => setAspectMode(e.target.value)} className="input-field">
                <option value="auto">自动保持原比例</option>
                <option value="9:16">强制 9:16 竖屏</option>
                <option value="16:9">强制 16:9 横屏</option>
              </select>
              <p className="text-xs text-zinc-500 mt-1">未选择强制比例时，原视频是 9:16 就保持竖屏，是 16:9 就保持横屏。</p>
            </div>

            <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 space-y-3">
              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  className="mt-1"
                  checked={backgroundMusicEnabled}
                  onChange={(e) => {
                    setAttachedTaskRequest(null)
                    setBackgroundMusicEnabled(e.target.checked)
                  }}
                />
                <span>
                  <span className="flex items-center gap-2 text-sm text-zinc-300">
                    <Music2 size={16} className="text-cyan-300" /> 添加背景音乐
                  </span>
                  <span className="block text-xs text-zinc-500 mt-1">默认关闭；开启后会把所选音乐低音量循环铺底。</span>
                </span>
              </label>
              {backgroundMusicEnabled && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm text-zinc-300 mb-2">背景音乐曲目</label>
                    <select
                      value={backgroundMusicTrack}
                      onChange={(e) => {
                        setAttachedTaskRequest(null)
                        setBackgroundMusicTrack(e.target.value)
                      }}
                      className="input-field"
                    >
                      {backgroundMusicTracks.map((track) => (
                        <option key={track.id} value={track.id} disabled={track.available === false}>
                          {track.label || track.title || track.id}{track.available === false ? '（文件缺失）' : ''}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm text-zinc-300 mb-2">背景音乐音量：{Math.round(backgroundMusicVolume * 100)}%</label>
                    <input
                      type="range"
                      min="0"
                      max="0.5"
                      step="0.01"
                      value={backgroundMusicVolume}
                      onChange={(e) => {
                        setAttachedTaskRequest(null)
                        setBackgroundMusicVolume(e.target.value)
                      }}
                      className="w-full"
                    />
                    <p className="text-xs text-zinc-500 mt-1">控制背景音乐铺底音量；默认 16%，想突出解说可调到 8%-12%。</p>
                  </div>
                </div>
              )}
            </div>

            <div className="grid sm:grid-cols-2 gap-3">
              <label className="flex items-start gap-3 p-4 rounded-xl border border-white/10 bg-white/[0.03] cursor-pointer">
                <input type="checkbox" className="mt-1" checked={autoVideoSpeed} onChange={(e) => { setAttachedTaskRequest(null); setAutoVideoSpeed(e.target.checked) }} />
                <span>
                  <span className="block text-sm text-zinc-300">AI 自动变速</span>
                  <span className="block text-xs text-zinc-500 mt-1">慢节奏、重复、搬运或转场片段会自动加速，关键展示保持 1x。</span>
                </span>
              </label>
              <label className="flex items-center gap-3 p-4 rounded-xl border border-white/10 bg-white/[0.03] cursor-pointer">
                <input type="checkbox" checked={subtitles} onChange={(e) => setSubtitles(e.target.checked)} />
                <span className="text-sm text-zinc-300">{targetDuration === 'full' ? '生成外挂字幕' : '生成并烧录字幕'}</span>
              </label>
            </div>

            {error && <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-300 text-sm">{error}</div>}

            <button
              onClick={handleGenerate}
              disabled={status === 'processing' || (sourceMode === 'url' && !url.trim()) || (sourceMode === 'file' && !videoFile)}
              className="btn-primary w-full flex items-center justify-center gap-2 py-3"
            >
              {status === 'processing' ? <Loader2 size={18} className="animate-spin" /> : <Film size={18} />}
              {submitLabel}
            </button>
          </div>

          <div className="space-y-6">
          <div className="glass-panel p-6 flex flex-col min-h-[420px]">
            <h2 className="text-lg font-semibold flex items-center gap-2 mb-4">
              <Activity size={18} className={status === 'processing' ? 'text-cyan-400 animate-pulse' : 'text-zinc-400'} />
              任务状态
            </h2>
            {attachedTaskRequest && (
              <div className="mb-4 rounded-xl border border-cyan-500/20 bg-cyan-500/[0.06] p-4">
                <div className="text-sm font-medium text-cyan-100 mb-3">当前任务参数</div>
                <div className="grid sm:grid-cols-3 gap-3 text-xs">
                  {attachedTaskRequest.analysis_mode !== 'openai' && (
                    <div className="rounded-lg bg-black/20 border border-white/10 p-3 sm:col-span-3">
                      <div className="text-zinc-500 mb-1">Gemini 模型</div>
                      <div className="text-zinc-100 font-semibold font-mono">{displayedGeminiModel || '默认模型'}</div>
                    </div>
                  )}
                  <div className="rounded-lg bg-black/20 border border-white/10 p-3">
                    <div className="text-zinc-500 mb-1">每批图片数</div>
                    <div className="text-zinc-100 font-semibold">{attachedTaskRequest.openai_batch_size ?? '—'}</div>
                  </div>
                  <div className="rounded-lg bg-black/20 border border-white/10 p-3">
                    <div className="text-zinc-500 mb-1">视觉分析并发数</div>
                    <div className="text-zinc-100 font-semibold">{attachedTaskRequest.openai_visual_concurrency ?? '—'}</div>
                  </div>
                  <div className="rounded-lg bg-black/20 border border-white/10 p-3">
                    <div className="text-zinc-500 mb-1">解说分块生成并发数</div>
                    <div className="text-zinc-100 font-semibold">{attachedTaskRequest.commentary_block_concurrency ?? '—'}</div>
                  </div>
                  <div className="rounded-lg bg-black/20 border border-white/10 p-3">
                    <div className="text-zinc-500 mb-1">AI 自动变速</div>
                    <div className="text-zinc-100 font-semibold">{displayedAutoVideoSpeed ? '开启' : '关闭'}</div>
                  </div>
                  {attachedTaskRequest.custom_style_prompt && (
                    <div className="rounded-lg bg-black/20 border border-white/10 p-3 sm:col-span-2">
                      <div className="text-zinc-500 mb-1">自定义风格提示词</div>
                      <div className="line-clamp-3 text-zinc-100">{attachedTaskRequest.custom_style_prompt}</div>
                    </div>
                  )}
                  <div className="rounded-lg bg-black/20 border border-white/10 p-3">
                    <div className="text-zinc-500 mb-1">背景音乐</div>
                    <div className="text-zinc-100 font-semibold">{displayedBackgroundMusicEnabled ? (selectedBackgroundMusicTrack?.label || '已开启') : '关闭'}</div>
                  </div>
                  {displayedBackgroundMusicEnabled && (
                    <div className="rounded-lg bg-black/20 border border-white/10 p-3">
                      <div className="text-zinc-500 mb-1">背景音乐音量</div>
                      <div className="text-zinc-100 font-semibold">{Math.round(Number(displayedBackgroundMusicVolume || 0) * 100)}%</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="mb-4 space-y-2">
              {progressSteps.map((step) => (
                <div key={step.label} className={`rounded-xl border p-3 ${step.state === 'active' ? 'border-cyan-500/30 bg-cyan-500/10' : step.state === 'done' ? 'border-green-500/20 bg-green-500/10' : 'border-white/10 bg-white/[0.03]'}`}>
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <span className={step.state === 'active' ? 'text-cyan-200' : step.state === 'done' ? 'text-green-300' : 'text-zinc-400'}>{step.label}</span>
                    <span className="text-xs text-zinc-500">{step.detail}</span>
                  </div>
                  {typeof step.percent === 'number' && (
                    <div className="mt-2 h-1.5 rounded-full bg-white/10 overflow-hidden">
                      <div className={`h-full rounded-full transition-all ${step.state === 'done' ? 'bg-green-400' : 'bg-cyan-400'}`} style={{ width: `${Math.max(0, Math.min(100, step.percent))}%` }} />
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="rounded-xl overflow-hidden">
              <div className={`flex items-center justify-between gap-3 rounded-t-xl border border-white/5 bg-black/40 px-4 py-3 ${logsExpanded ? 'border-b-0' : 'rounded-b-xl'}`}>
                <div>
                  <div className="text-sm font-medium text-zinc-200">运行日志</div>
                  <div className="text-xs text-zinc-500">{logPanelState.countLabel}</div>
                </div>
                <button
                  type="button"
                  onClick={() => setLogsExpanded((value) => !value)}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10 hover:text-white"
                  title={logPanelState.toggleLabel}
                  aria-label={logPanelState.toggleLabel}
                  aria-expanded={logsExpanded}
                >
                  {logsExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </button>
              </div>
              {logsExpanded && (
                <div className={COMMENTARY_LOG_PANEL_BODY_CLASS}>
                  {logs.length === 0 && <div>{logPanelState.emptyText}</div>}
                  {logs.map((line, idx) => <div key={`${line}-${idx}`}>› {line}</div>)}
                </div>
              )}
            </div>

            {result && (
              <div className="mt-5 space-y-4">
                <div className="p-4 rounded-xl bg-green-500/10 border border-green-500/20">
                  <div className="flex items-center gap-2 text-green-300 font-semibold mb-2">
                    <CheckCircle size={18} /> 生成完成
                  </div>
                  <div className="space-y-3">
                    <div className="rounded-lg border border-white/10 bg-black/20 p-3">
                      <div className="mb-2 flex items-center justify-between gap-3">
                        <span className="text-xs font-medium text-zinc-400">发布标题</span>
                        <div className="flex items-center gap-2">
                          <span className="text-[11px] text-zinc-500">{publishTitle.length}/30</span>
                          <button type="button" onClick={() => copyText(publishTitle)} className="inline-flex h-7 w-7 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10" title="复制发布标题">
                            <Copy size={14} />
                          </button>
                        </div>
                      </div>
                      <div className="text-sm text-zinc-200 font-medium leading-relaxed">{publishTitle}</div>
                    </div>
                    <div className="rounded-lg border border-white/10 bg-black/20 p-3">
                      <div className="mb-2 flex items-center justify-between gap-3">
                        <span className="text-xs font-medium text-zinc-400">发布描述</span>
                        <div className="flex items-center gap-2">
                          <span className="text-[11px] text-zinc-500">{publishDescription.length}/1000</span>
                          <button type="button" onClick={() => copyText(publishDescription)} className="inline-flex h-7 w-7 items-center justify-center rounded-lg border border-white/10 bg-white/5 text-zinc-300 hover:bg-white/10" title="复制发布描述">
                            <Copy size={14} />
                          </button>
                        </div>
                      </div>
                      <p className="whitespace-pre-wrap text-xs text-zinc-400 leading-relaxed">{publishDescription}</p>
                    </div>
                  </div>
                  {speedSummary && (
                    <div className="mt-3 rounded-lg border border-white/10 bg-black/20 p-3 text-xs text-zinc-300">
                      {speedSummary.enabled
                        ? speedSummary.accelerated_count > 0
                          ? `AI 自动变速：${speedSummary.accelerated_count}/${speedSummary.total_blocks} 个片段加速，约节省 ${speedSummary.saved_seconds} 秒。`
                          : 'AI 自动变速：未发现适合加速的慢节奏片段，全部保持 1x。'
                        : 'AI 自动变速：已关闭，全部保持 1x。'}
                    </div>
                  )}
                  {result.background_music_enabled && (
                    <div className="mt-3 flex items-center gap-2 rounded-lg border border-white/10 bg-black/20 p-3 text-xs text-zinc-300">
                      <Music2 size={14} className="text-cyan-300" />
                      背景音乐：{result.background_music_label || selectedBackgroundMusicTrack?.label || '默认 奥德彪专属音乐'} · 音量 {Math.round(Number(result.background_music_volume ?? displayedBackgroundMusicVolume ?? 0) * 100)}%
                    </div>
                  )}
                </div>

                {commentaryCovers.length > 0 && (
                  <div className="space-y-3 rounded-xl border border-white/10 bg-white/[0.03] p-3">
                    <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                      <FileVideo size={16} className="text-cyan-300" /> 发布封面
                    </div>
                    <div className="grid sm:grid-cols-2 gap-3">
                      {commentaryCovers.map((cover) => (
                        <div key={cover.label} className="rounded-lg border border-white/10 bg-black/20 p-3 space-y-2">
                          <div className="flex items-center justify-between gap-3">
                            <div className="text-sm font-medium text-zinc-200">{cover.label}</div>
                            <a href={getApiUrl(cover.url)} download className="inline-flex items-center gap-1 text-xs text-cyan-300 hover:text-cyan-200">
                              <Download size={14} /> 下载
                            </a>
                          </div>
                          <img src={getApiUrl(cover.url)} alt={cover.label} className={`w-full ${cover.className} rounded-md border border-white/10 bg-black object-cover`} />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="space-y-3 rounded-xl border border-white/10 bg-white/[0.03] p-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                    <Film size={16} className="text-cyan-300" /> 完整二创解说总视频
                  </div>
                  <video src={getApiUrl(result.video_url)} controls className="w-full rounded-xl border border-white/10 bg-black" />
                  <div className="grid grid-cols-2 gap-3">
                    <a href={getApiUrl(result.video_url)} download className="btn-primary flex items-center justify-center gap-2 text-sm">
                      <Download size={16} /> 下载总视频
                    </a>
                    <a href={getApiUrl(`/videos/${jobId}/${result.script_path}`)} target="_blank" rel="noopener noreferrer" className="btn-secondary flex items-center justify-center gap-2 text-sm">
                      <FileText size={16} /> 查看脚本
                    </a>
                  </div>
                </div>

                {episodePlan?.should_split && commentaryEpisodes.length > 0 && (
                  <div className="space-y-3 rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-3">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2 text-sm font-semibold text-cyan-100">
                        <FileVideo size={16} className="text-cyan-300" /> AI 分集视频
                      </div>
                      <span className="text-xs text-cyan-300">{commentaryEpisodes.length} 集</span>
                    </div>
                    {episodePlan.reason && <p className="text-xs text-zinc-400">{episodePlan.reason}</p>}
                    <div className="space-y-4">
                      {commentaryEpisodes.map((episode) => (
                        <div key={`${episode.episode_number}-${episode.video_url}`} className="rounded-xl border border-white/10 bg-black/20 p-3 space-y-3">
                          <div>
                            <div className="text-sm font-medium text-zinc-200">{episode.title || `第 ${episode.episode_number} 集`}</div>
                            {episode.summary && <p className="mt-1 text-xs text-zinc-500 line-clamp-2">{episode.summary}</p>}
                            <div className="mt-2 text-[11px] text-zinc-500">
                              解说块 {episode.start_block}-{episode.end_block} · 约 {Math.round(episode.duration || 0)} 秒
                            </div>
                          </div>
                          <video src={getApiUrl(episode.video_url)} controls className="w-full rounded-lg border border-white/10 bg-black" />
                          <a href={getApiUrl(episode.video_url)} download className="btn-secondary flex items-center justify-center gap-2 text-sm">
                            <Download size={16} /> 下载本集
                          </a>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="glass-panel p-6">
            <div className="flex flex-col gap-3 mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <History size={18} className="text-zinc-400" /> 历史任务
              </h2>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={toggleAllCommentaryTaskSelection}
                  disabled={commentaryTasks.length === 0 || isDeletingTasks}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 text-xs text-zinc-300 hover:bg-white/5 disabled:opacity-40"
                >
                  {allVisibleTasksSelected ? <CheckSquare size={14} /> : <Square size={14} />}
                  {allVisibleTasksSelected ? '取消全选' : '全选'}
                </button>
                <button
                  type="button"
                  onClick={() => deleteCommentaryTasks(selectedVisibleTaskIds)}
                  disabled={selectedVisibleTaskIds.length === 0 || isDeletingTasks}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-1.5 text-xs text-red-200 hover:bg-red-500/20 disabled:opacity-40"
                >
                  {isDeletingTasks ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
                  删除选中{selectedVisibleTaskIds.length ? ` ${selectedVisibleTaskIds.length}` : ''}
                </button>
                <button type="button" onClick={refreshCommentaryTasks} className="ml-auto text-xs text-cyan-300 hover:text-cyan-200">
                  {taskListStatus === 'loading' ? '刷新中...' : '刷新'}
                </button>
              </div>
            </div>
            <div className="space-y-3 max-h-[420px] overflow-y-auto custom-scrollbar pr-1">
              {commentaryTasks.length === 0 && (
                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4 text-sm text-zinc-500">
                  暂无历史任务
                </div>
              )}
              {commentaryTasks.map((task) => (
                <div key={task.job_id} className="rounded-xl border border-white/10 bg-white/[0.03] p-4 space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex min-w-0 items-start gap-2">
                      <button
                        type="button"
                        onClick={() => toggleCommentaryTaskSelection(task.job_id)}
                        disabled={isDeletingTasks}
                        className="mt-0.5 shrink-0 text-zinc-400 hover:text-cyan-300 disabled:opacity-40"
                        aria-label={selectedTaskIds.includes(task.job_id) ? '取消选择任务' : '选择任务'}
                      >
                        {selectedTaskIds.includes(task.job_id) ? <CheckSquare size={16} /> : <Square size={16} />}
                      </button>
                      <div className="min-w-0">
                        <div className="truncate text-sm font-medium text-zinc-200">{taskTitle(task)}</div>
                        <div className="mt-1 text-xs text-zinc-500">{task.stage_label || task.status} · {formatTaskTime(task.updated_at || task.created_at)}</div>
                      </div>
                    </div>
                    <span className={`shrink-0 rounded-full px-2 py-1 text-[11px] ${task.status === 'completed' ? 'bg-green-500/10 text-green-300' : task.status === 'failed' ? 'bg-red-500/10 text-red-300' : 'bg-cyan-500/10 text-cyan-300'}`}>
                      {task.status || 'unknown'}
                    </span>
                  </div>
                  {task.error && <div className="line-clamp-2 text-xs text-red-300">{task.error}</div>}
                  <div className="grid grid-cols-2 xl:grid-cols-4 gap-2">
                    <button type="button" onClick={() => attachCommentaryTask(task)} className="btn-secondary text-xs py-2">
                      查看状态
                    </button>
                    <button
                      type="button"
                      onClick={() => retryCommentaryTask(task)}
                      disabled={status === 'processing' || task.status === 'processing' || retryingJobId === task.job_id}
                      className="btn-secondary text-xs py-2 disabled:opacity-50"
                    >
                      {retryingJobId === task.job_id ? '重试中...' : '重试'}
                    </button>
                    {task.result?.video_url ? (
                      <a href={getApiUrl(task.result.video_url)} download className="btn-primary text-xs py-2 text-center">
                        下载结果
                      </a>
                    ) : (
                      <button type="button" disabled className="btn-secondary text-xs py-2 opacity-40">下载结果</button>
                    )}
                    <button
                      type="button"
                      onClick={() => deleteCommentaryTasks([task.job_id])}
                      disabled={isDeletingTasks}
                      className="btn-secondary inline-flex items-center justify-center gap-1 text-xs py-2 text-red-200 hover:border-red-500/40 hover:bg-red-500/10 disabled:opacity-40"
                    >
                      {deletingTaskIds.includes(task.job_id) ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} />}
                      删除
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
          </div>
        </div>
      </div>
    </div>
  )
}
