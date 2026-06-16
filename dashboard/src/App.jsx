import React, { useState, useEffect } from 'react'
import {
  Upload,
  FileVideo,
  Sparkles,
  Youtube,
  Instagram,
  Share2,
  LogOut,
  ChevronDown,
  Check,
  Activity,
  LayoutDashboard,
  Settings,
  PlusCircle,
  History,
  Menu,
  X,
  Terminal,
  Shield,
  LayoutGrid,
  Image,
  Globe,
  RotateCcw,
  Calendar,
  Mic2,
} from 'lucide-react'
import KeyInput from './components/KeyInput'
import MediaInput from './components/MediaInput'
import ResultCard from './components/ResultCard'
import ProcessingAnimation from './components/ProcessingAnimation'
// import Gallery from './components/Gallery';
import ThumbnailStudio from './components/ThumbnailStudio'
import SaaShortsTab from './components/SaaShortsTab'
import CommentaryTab from './components/CommentaryTab'
import UGCGallery from './components/UGCGallery'
import ScheduleWeekModal from './components/ScheduleWeekModal'
import { getApiUrl } from './config'
import {
  buildGeminiConfig,
  buildGeminiHeaders,
  fingerprintGeminiKey,
  getGeminiAccessMissingMessage,
  hasGeminiAccess as hasGeminiConfigAccess,
  normalizeGeminiBaseUrl,
} from './lib/geminiHeaders'
import { useI18n } from './i18n/I18nProvider'
import LanguageSwitcher from './i18n/LanguageSwitcher'

// Enhanced "Encryption" using XOR + Base64 with a Salt
// This is better than plain Base64 but still client-side.
const SECRET_KEY =
  import.meta.env.VITE_ENCRYPTION_KEY || 'OpenShorts-Static-Salt-Change-Me'
const ENCRYPTION_PREFIX = 'ENC:'

const encrypt = (text) => {
  if (!text) return ''
  try {
    const xor = text
      .split('')
      .map((c, i) =>
        String.fromCharCode(
          c.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length),
        ),
      )
      .join('')
    return ENCRYPTION_PREFIX + btoa(xor)
  } catch (e) {
    console.error('Encryption failed', e)
    return text
  }
}

const decrypt = (text) => {
  if (!text) return ''
  if (text.startsWith(ENCRYPTION_PREFIX)) {
    try {
      const raw = text.slice(ENCRYPTION_PREFIX.length)
      // Check if it's plain base64 or our custom XOR (simple try)
      const xor = atob(raw)
      const result = xor
        .split('')
        .map((c, i) =>
          String.fromCharCode(
            c.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length),
          ),
        )
        .join('')
      return result
    } catch (e) {
      // Fallback if decryption fails (might be old plain text)
      return ''
    }
  }
  // Backward compatibility: If no prefix, assume old plain text (or return empty if you want to force re-login)
  // For migration: Return text as is, so it populates the field, and next save will encrypt it.
  return text
}

// Simple TikTok icon sine Lucide might not have it or it varies
const TikTokIcon = ({ size = 16, className = '' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
    className={className}
  >
    <path d="M19.589 6.686a4.793 4.793 0 0 1-3.77-4.245V2h-3.445v13.672a2.896 2.896 0 0 1-5.201 1.743l-.002-.001.002.001a2.895 2.895 0 0 1 3.183-4.51v-3.5a6.329 6.329 0 0 0-5.394 10.692 6.33 6.33 0 0 0 10.857-4.424V8.687a8.182 8.182 0 0 0 4.773 1.526V6.79a4.831 4.831 0 0 1-1.003-.104z" />
  </svg>
)

const UserProfileSelector = ({ profiles, selectedUserId, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false)

  if (!profiles || profiles.length === 0) return null

  const selectedProfile =
    profiles.find((p) => p.username === selectedUserId) || profiles[0]

  return (
    <div className="relative z-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between bg-surface border border-white/10 rounded-lg px-3 py-2 text-sm text-zinc-300 hover:bg-white/5 transition-colors min-w-[180px]"
      >
        <span className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center text-[10px] font-bold text-white">
            {selectedProfile?.username?.substring(0, 1).toUpperCase() || 'U'}
          </div>
          <span className="font-medium text-white truncate max-w-[100px]">
            {selectedProfile?.username || 'Select User'}
          </span>
        </span>
        <ChevronDown
          size={14}
          className={`text-zinc-500 transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {isOpen && (
        <div className="absolute top-full mt-2 right-0 w-64 bg-[#1a1a1a] border border-white/10 rounded-xl shadow-2xl overflow-hidden">
          <div className="max-h-60 overflow-y-auto custom-scrollbar">
            {profiles.map((profile) => (
              <button
                key={profile.username}
                onClick={() => {
                  onSelect(profile.username)
                  setIsOpen(false)
                }}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/5 transition-colors text-left group border-b border-white/5 last:border-0"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-purple-500/20 flex items-center justify-center text-xs font-bold text-white border border-white/10 shrink-0">
                    {profile.username.substring(0, 2).toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-zinc-200 group-hover:text-white transition-colors truncate">
                      {profile.username}
                    </div>
                    <div className="flex gap-2 mt-0.5">
                      {/* Status indicators */}
                      <div
                        className={`flex items-center gap-1 text-[10px] ${profile.connected.includes('tiktok') ? 'text-zinc-300' : 'text-zinc-600'}`}
                      >
                        <TikTokIcon size={10} />
                      </div>
                      <div
                        className={`flex items-center gap-1 text-[10px] ${profile.connected.includes('instagram') ? 'text-pink-400' : 'text-zinc-600'}`}
                      >
                        <Instagram size={10} />
                      </div>
                      <div
                        className={`flex items-center gap-1 text-[10px] ${profile.connected.includes('youtube') ? 'text-red-400' : 'text-zinc-600'}`}
                      >
                        <Youtube size={10} />
                      </div>
                    </div>
                  </div>
                </div>
                {selectedUserId === profile.username && (
                  <Check size={14} className="text-primary shrink-0" />
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

const SESSION_KEY = 'openshorts_session'
const SESSION_MAX_AGE = 3600000 // 1 hour (matches server job retention)

// Mock polling function
const pollJob = async (jobId) => {
  const res = await fetch(getApiUrl(`/api/status/${jobId}`))
  if (!res.ok) throw new Error('Status check failed')
  return res.json()
}

function GeminiAccessModeSettings({
  geminiAccessMode,
  setGeminiAccessMode,
  geminiKeyPoolText,
  setGeminiKeyPoolText,
  geminiKeyPool,
  geminiKeyPoolStats,
  setGeminiKeyPoolStats,
}) {
  return (
    <div className="mt-5 border-t border-white/10 pt-5">
      <div className="flex items-center justify-between gap-4 mb-4">
        <div>
          <h3 className="text-sm font-semibold text-zinc-100">Gemini 访问模式</h3>
          <p className="text-xs text-zinc-500 mt-1">
            访问模式和多 Key 列表会自动保存到当前浏览器。
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5 text-[11px] text-green-300 bg-green-500/10 border border-green-500/20 rounded-full px-2.5 py-1">
            <Check size={12} /> 自动保存
          </span>
          <select
            value={geminiAccessMode}
            onChange={(e) => setGeminiAccessMode(e.target.value)}
            className="input-field max-w-[220px]"
          >
            <option value="custom_proxy">自定义代理 / 单 Key</option>
            <option value="official_pool">官方多 Key 负载均衡</option>
          </select>
        </div>
      </div>
      {geminiAccessMode === 'official_pool' ? (
        <div className="space-y-4">
          <textarea
            value={geminiKeyPoolText}
            onChange={(e) => setGeminiKeyPoolText(e.target.value)}
            placeholder="每行一个官方 Gemini API Key，例如 AIza..."
            className="input-field min-h-[120px] font-mono text-xs resize-y"
          />
          <div className="rounded-xl border border-white/10 overflow-hidden">
            <div className="grid grid-cols-[1fr_90px_90px_1.5fr] gap-3 px-3 py-2 text-xs text-zinc-500 bg-white/5">
              <span>Key</span>
              <span>成功</span>
              <span>429</span>
              <span>最近错误</span>
            </div>
            {geminiKeyPool.length ? (
              geminiKeyPool.map((key) => {
                const fingerprint = fingerprintGeminiKey(key)
                const stat = geminiKeyPoolStats[fingerprint] || {}
                return (
                  <div key={fingerprint} className="grid grid-cols-[1fr_90px_90px_1.5fr] gap-3 px-3 py-2 text-xs border-t border-white/5">
                    <span className="font-mono text-zinc-300">{fingerprint}</span>
                    <span className="text-green-300">{stat.successes || 0}</span>
                    <span className="text-yellow-300">{stat.errors429 || 0}</span>
                    <span className="truncate text-zinc-400">{stat.lastError || '-'}</span>
                  </div>
                )
              })
            ) : (
              <div className="px-3 py-3 text-xs text-zinc-500 border-t border-white/5">
                还没有添加官方 Gemini API Key。
              </div>
            )}
          </div>
          <button
            onClick={() => setGeminiKeyPoolStats({})}
            className="px-4 py-2 rounded-lg border border-white/10 text-sm text-zinc-300 hover:bg-white/5"
          >
            清空 Key 统计
          </button>
        </div>
      ) : (
        <p className="text-xs text-zinc-500">
          单 Key 模式使用上面的 Gemini API Key 和可选 Base URL；修改单 Key/Base URL 后仍需要点击“设置密钥”。
        </p>
      )}
    </div>
  )
}

function App() {
  const { t } = useI18n()
  const [apiKey, setApiKey] = useState(localStorage.getItem('gemini_key') || '')
  const [geminiBaseUrl, setGeminiBaseUrl] = useState(
    localStorage.getItem('gemini_base_url') || '',
  )
  const [geminiAccessMode, setGeminiAccessMode] = useState(
    localStorage.getItem('gemini_access_mode') || 'custom_proxy',
  )
  const [geminiKeyPoolText, setGeminiKeyPoolText] = useState(() => {
    const stored = localStorage.getItem('gemini_key_pool_v1')
    if (!stored) return ''
    return decrypt(stored)
  })
  const [geminiKeyPoolStats, setGeminiKeyPoolStats] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('gemini_key_pool_stats_v1') || '{}')
    } catch {
      return {}
    }
  })
  // Social API State - Load encrypted or plain
  const [uploadPostKey, setUploadPostKey] = useState(() => {
    const stored = localStorage.getItem('uploadPostKey_v3')
    if (stored) return decrypt(stored)
    return ''
  })
  // ElevenLabs API State - Load encrypted
  const [elevenLabsKey, setElevenLabsKey] = useState(() => {
    const stored = localStorage.getItem('elevenLabsKey_v1')
    if (stored) return decrypt(stored)
    return ''
  })

  // fal.ai API State - Load encrypted
  const [falKey, setFalKey] = useState(() => {
    const stored = localStorage.getItem('falKey_v1')
    if (stored) return decrypt(stored)
    return ''
  })

  const [openAICompatibleKey, setOpenAICompatibleKey] = useState(() => {
    const stored = localStorage.getItem('openAICompatibleKey_v1')
    if (stored) return decrypt(stored)
    return ''
  })
  const [openAICompatibleBaseUrl, setOpenAICompatibleBaseUrl] = useState(
    () => localStorage.getItem('openAICompatibleBaseUrl_v1') || '',
  )
  const [openAICompatibleModel, setOpenAICompatibleModel] = useState(
    () => localStorage.getItem('openAICompatibleModel_v1') || '',
  )

  const [uploadUserId, setUploadUserId] = useState(
    () => localStorage.getItem('uploadUserId') || '',
  )
  const [userProfiles, setUserProfiles] = useState([]) // List of {username, connected: []}
  const [showKeyModal, setShowKeyModal] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState('idle') // idle, processing, complete, error
  const [results, setResults] = useState(null)
  const [logs, setLogs] = useState([])
  const [logsVisible, setLogsVisible] = useState(true)
  const [processingMedia, setProcessingMedia] = useState(null)
  const [activeTab, setActiveTab] = useState('dashboard') // dashboard, settings

  const [sessionRecovered, setSessionRecovered] = useState(false)
  const [showScheduleWeek, setShowScheduleWeek] = useState(false)

  // Sync state for original video playback
  const [syncedTime, setSyncedTime] = useState(0)
  const [isSyncedPlaying, setIsSyncedPlaying] = useState(false)
  const [syncTrigger, setSyncTrigger] = useState(0)

  const handleClipPlay = (startTime) => {
    setSyncedTime(startTime)
    setIsSyncedPlaying(true)
    setSyncTrigger((prev) => prev + 1)
  }

  const handleClipPause = () => {
    setIsSyncedPlaying(false)
  }

  // Session Recovery: Restore on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(SESSION_KEY)
      if (!saved) return
      const session = JSON.parse(saved)
      if (Date.now() - session.timestamp > SESSION_MAX_AGE) {
        localStorage.removeItem(SESSION_KEY)
        return
      }
      if (session.jobId && session.status && session.status !== 'idle') {
        setJobId(session.jobId)
        setResults(session.results || null)
        if (session.processingMedia) setProcessingMedia(session.processingMedia)
        if (session.activeTab) setActiveTab(session.activeTab)
        // If was processing, resume polling; if complete/error, just show results
        setStatus(
          session.status === 'processing' ? 'processing' : session.status,
        )
        setSessionRecovered(true)
        setTimeout(() => setSessionRecovered(false), 5000)
      }
    } catch (e) {
      localStorage.removeItem(SESSION_KEY)
    }
  }, [])

  // Session Recovery: Save state changes
  useEffect(() => {
    if (status === 'idle') {
      localStorage.removeItem(SESSION_KEY)
      return
    }
    try {
      const sessionData = {
        jobId,
        status,
        results,
        processingMedia:
          processingMedia?.type === 'url' ? processingMedia : null,
        activeTab,
        timestamp: Date.now(),
      }
      localStorage.setItem(SESSION_KEY, JSON.stringify(sessionData))
    } catch (e) {
      // localStorage full or serialization error - ignore
    }
  }, [jobId, status, results, activeTab])

  useEffect(() => {
    if (apiKey) localStorage.setItem('gemini_key', apiKey)
  }, [apiKey])

  useEffect(() => {
    localStorage.setItem('gemini_access_mode', geminiAccessMode)
  }, [geminiAccessMode])

  useEffect(() => {
    if (geminiKeyPoolText.trim()) {
      localStorage.setItem('gemini_key_pool_v1', encrypt(geminiKeyPoolText))
    } else {
      localStorage.removeItem('gemini_key_pool_v1')
    }
  }, [geminiKeyPoolText])

  useEffect(() => {
    localStorage.setItem('gemini_key_pool_stats_v1', JSON.stringify(geminiKeyPoolStats))
  }, [geminiKeyPoolStats])

  useEffect(() => {
    const normalizedBaseUrl = normalizeGeminiBaseUrl(geminiBaseUrl)
    if (normalizedBaseUrl) {
      localStorage.setItem('gemini_base_url', normalizedBaseUrl)
    } else {
      localStorage.removeItem('gemini_base_url')
    }
  }, [geminiBaseUrl])

  const geminiKeyPool = geminiKeyPoolText
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean)
  const geminiConfig = buildGeminiConfig({
    mode: geminiAccessMode,
    apiKey,
    baseUrl: geminiBaseUrl,
    keyPool: geminiKeyPool,
    stats: geminiKeyPoolStats,
  })
  const hasGeminiAccess = hasGeminiConfigAccess(geminiConfig)
  const geminiAccessMissingMessage = getGeminiAccessMissingMessage(geminiConfig)

  useEffect(() => {
    if (uploadPostKey) {
      localStorage.setItem('uploadPostKey_v3', encrypt(uploadPostKey))
    }
    if (uploadUserId) {
      localStorage.setItem('uploadUserId', uploadUserId)
    }
  }, [uploadPostKey, uploadUserId])

  useEffect(() => {
    if (elevenLabsKey) {
      localStorage.setItem('elevenLabsKey_v1', encrypt(elevenLabsKey))
    }
  }, [elevenLabsKey])

  useEffect(() => {
    if (falKey) {
      localStorage.setItem('falKey_v1', encrypt(falKey))
    }
  }, [falKey])

  useEffect(() => {
    if (openAICompatibleKey) {
      localStorage.setItem('openAICompatibleKey_v1', encrypt(openAICompatibleKey))
    } else {
      localStorage.removeItem('openAICompatibleKey_v1')
    }
  }, [openAICompatibleKey])

  useEffect(() => {
    const normalizedBaseUrl = openAICompatibleBaseUrl.trim().replace(/\/+$/, '')
    if (normalizedBaseUrl) {
      localStorage.setItem('openAICompatibleBaseUrl_v1', normalizedBaseUrl)
    } else {
      localStorage.removeItem('openAICompatibleBaseUrl_v1')
    }
  }, [openAICompatibleBaseUrl])

  useEffect(() => {
    const model = openAICompatibleModel.trim()
    if (model) {
      localStorage.setItem('openAICompatibleModel_v1', model)
    } else {
      localStorage.removeItem('openAICompatibleModel_v1')
    }
  }, [openAICompatibleModel])

  useEffect(() => {
    if (uploadPostKey && userProfiles.length === 0) {
      fetchUserProfiles()
    }
  }, [uploadPostKey])

  useEffect(() => {
    let interval
    if ((status === 'processing' || status === 'completed') && jobId) {
      interval = setInterval(async () => {
        try {
          const data = await pollJob(jobId)
          console.log('Job status:', data)

          // Update results if available (real-time)
          if (data.result) {
            setResults(data.result)
          }

          if (data.status === 'completed') {
            setStatus('complete')
            clearInterval(interval)
          } else if (data.status === 'failed') {
            setStatus('error')
            const errorMsg =
              data.error ||
              (data.logs && data.logs.length > 0
                ? data.logs[data.logs.length - 1]
                : t('clipGenerator.processFailed'))
            setLogs((prev) => [...prev, 'Error: ' + errorMsg])
            clearInterval(interval)
          } else {
            // Update logs if available
            if (data.logs) setLogs(data.logs)
          }
        } catch (e) {
          console.error('Polling error', e)
        }
      }, 2000)
    }
    return () => clearInterval(interval)
  }, [status, jobId, t])

  const fetchUserProfiles = async () => {
    if (!uploadPostKey) return
    try {
      const res = await fetch(getApiUrl('/api/social/user'), {
        headers: { 'X-Upload-Post-Key': uploadPostKey },
      })
      if (!res.ok) throw new Error('Failed to fetch')
      const data = await res.json()
      if (data.profiles && data.profiles.length > 0) {
        setUserProfiles(data.profiles)
        // Auto select first if none selected
        if (!uploadUserId) {
          setUploadUserId(data.profiles[0].username)
        }
      } else {
        alert(t('settings.noProfiles'))
      }
    } catch (e) {
      alert(t('settings.profileFetchError'))
      console.error(e)
    }
  }

  const handleProcess = async (data) => {
    if (!hasGeminiAccess) {
      setShowKeyModal(true)
      return
    }
    setStatus('processing')
    setLogs([t('clipGenerator.startingProcess')])
    setResults(null)
    setProcessingMedia(data)

    try {
      let body
      const headers = buildGeminiHeaders(geminiConfig)

      if (data.type === 'url') {
        headers['Content-Type'] = 'application/json'
        body = JSON.stringify({ url: data.payload })
      } else {
        const formData = new FormData()
        formData.append('file', data.payload)
        body = formData
      }

      const res = await fetch(getApiUrl('/api/process'), {
        method: 'POST',
        headers,
        body,
      })

      if (!res.ok) throw new Error(await res.text())
      const resData = await res.json()
      setJobId(resData.job_id)
    } catch (e) {
      setStatus('error')
      setLogs((l) => [
        ...l,
        t('clipGenerator.startingJobError', { message: e.message }),
      ])
    }
  }

  const handleReset = () => {
    setStatus('idle')
    setJobId(null)
    setResults(null)
    setLogs([])
    setProcessingMedia(null)
    localStorage.removeItem(SESSION_KEY)
  }

  // --- UI Components ---

  const Sidebar = () => (
    <div className="w-20 lg:w-64 bg-surface border-r border-white/5 flex flex-col h-full shrink-0 transition-all duration-300">
      <div className="p-6 flex items-center gap-3">
        <div className="w-8 h-8 bg-white/5 rounded-lg flex items-center justify-center shrink-0 overflow-hidden border border-white/5">
          <img
            src="/logo-openshorts.png"
            alt="Logo"
            className="w-full h-full object-cover"
          />
        </div>
        <span className="font-bold text-lg text-white hidden lg:block tracking-tight">
          OpenShorts
        </span>
      </div>

      <nav className="flex-1 px-4 py-4 space-y-2">
        <button
          onClick={() => setActiveTab('dashboard')}
          className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'dashboard' ? 'bg-primary/10 text-primary' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
        >
          <LayoutDashboard size={20} />
          <span className="font-medium hidden lg:block">
            {t('nav.clipGenerator')}
          </span>
        </button>

        <button
          onClick={() => setActiveTab('saasshorts')}
          className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'saasshorts' ? 'bg-violet-500/10 text-violet-400' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
        >
          <Sparkles size={20} />
          <span className="font-medium hidden lg:block">
            {t('nav.aiShorts')}
          </span>
        </button>

        <button
          onClick={() => setActiveTab('commentary')}
          className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'commentary' ? 'bg-cyan-500/10 text-cyan-400' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
        >
          <Mic2 size={20} />
          <span className="font-medium hidden lg:block">二创解说</span>
        </button>

        <button
          onClick={() => setActiveTab('ugc-gallery')}
          className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'ugc-gallery' ? 'bg-violet-500/10 text-violet-400' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
        >
          <LayoutGrid size={20} />
          <span className="font-medium hidden lg:block">
            {t('nav.ugcGallery')}
          </span>
        </button>

        <button
          onClick={() => setActiveTab('thumbnails')}
          className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'thumbnails' ? 'bg-primary/10 text-primary' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
        >
          <Image size={20} />
          <span className="font-medium hidden lg:block">
            {t('nav.youtubeStudio')}
          </span>
        </button>

        {/* <button
          onClick={() => setActiveTab('gallery')}
          className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'gallery' ? 'bg-primary/10 text-primary' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
        >
          <LayoutGrid size={20} />
          <span className="font-medium hidden lg:block">Gallery</span>
        </button> */}

        <button
          onClick={() => setActiveTab('settings')}
          className={`w-full flex items-center gap-3 px-3 py-3 rounded-xl transition-colors ${activeTab === 'settings' ? 'bg-primary/10 text-primary' : 'text-zinc-400 hover:text-white hover:bg-white/5'}`}
        >
          <Settings size={20} />
          <span className="font-medium hidden lg:block">
            {t('nav.settings')}
          </span>
        </button>
      </nav>

      <div className="p-4 border-t border-white/5 space-y-2">
        <a
          href="#"
          onClick={(e) => {
            e.preventDefault()
            localStorage.removeItem('openshorts_skip_landing')
            window.location.hash = ''
            window.location.reload()
          }}
          className="flex items-center gap-2 p-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors group"
        >
          <div className="w-8 h-8 rounded-full bg-primary/20 text-primary flex items-center justify-center shrink-0">
            <Globe size={16} />
          </div>
          <div className="hidden lg:block overflow-hidden">
            <p className="text-sm font-bold text-white leading-none mb-0.5">
              {t('nav.landingPage')}
            </p>
            <p className="text-[10px] text-zinc-400 group-hover:text-zinc-300 transition-colors truncate">
              {t('nav.viewWebsite')}
            </p>
          </div>
        </a>
        <a
          href="https://github.com/mutonby/openshorts"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 p-3 bg-white/5 hover:bg-white/10 rounded-xl transition-colors group"
        >
          <div className="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center shrink-0">
            <svg
              height="20"
              viewBox="0 0 16 16"
              version="1.1"
              width="20"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
              ></path>
            </svg>
          </div>
          <div className="hidden lg:block overflow-hidden">
            <p className="text-sm font-bold text-white leading-none mb-0.5">
              {t('nav.openSource')}
            </p>
            <p className="text-[10px] text-zinc-400 group-hover:text-zinc-300 transition-colors truncate">
              {t('nav.communityDriven')}
            </p>
          </div>
        </a>
      </div>
    </div>
  )

  return (
    <div className="flex h-screen bg-background overflow-hidden selection:bg-primary/30">
      <Sidebar />

      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Background Gradients */}
        <div className="absolute inset-0 overflow-hidden -z-10 pointer-events-none">
          <div className="absolute -top-[10%] -right-[10%] w-[50%] h-[50%] bg-primary/5 rounded-full blur-[120px]" />
        </div>

        {/* Top Header */}
        <header className="h-16 border-b border-white/5 bg-background/50 backdrop-blur-md flex items-center justify-between px-6 shrink-0 z-10">
          <div className="flex items-center gap-4">
            {status !== 'idle' && (
              <button
                onClick={handleReset}
                className="flex items-center gap-2 text-sm text-zinc-400 hover:text-white transition-colors"
              >
                <PlusCircle size={16} />
                <span className="hidden sm:inline">{t('nav.newProject')}</span>
              </button>
            )}
          </div>

          <div className="flex items-center gap-4">
            <LanguageSwitcher compact />
            {userProfiles.length > 0 && (
              <UserProfileSelector
                profiles={userProfiles}
                selectedUserId={uploadUserId}
                onSelect={setUploadUserId}
              />
            )}

            {!hasGeminiAccess && (
              <span
                className="text-xs text-amber-500 bg-amber-500/10 px-3 py-1 rounded-full border border-amber-500/20"
                title={geminiAccessMissingMessage}
              >
                {geminiConfig.mode === 'official_pool' ? 'Gemini 多 Key 未配置' : t('clipGenerator.apiKeyMissing')}
              </span>
            )}
          </div>
        </header>

        {/* Session Recovery Banner */}
        {sessionRecovered && (
          <div className="mx-6 mt-2 p-3 bg-primary/10 border border-primary/20 rounded-xl flex items-center justify-between animate-[fadeIn_0.3s_ease-out] shrink-0">
            <div className="flex items-center gap-2 text-sm text-primary">
              <RotateCcw size={16} />
              <span className="font-medium">
                {t('clipGenerator.sessionRecovered')}
              </span>
              <span className="text-zinc-400 text-xs">
                {t('clipGenerator.sessionRestored')}
              </span>
            </div>
            <button
              onClick={() => setSessionRecovered(false)}
              className="text-zinc-500 hover:text-white transition-colors"
            >
              <X size={14} />
            </button>
          </div>
        )}

        {/* Main Workspace */}
        <div className="flex-1 overflow-hidden relative">
          {/* View: Settings */}
          {activeTab === 'settings' && (
            <div className="h-full overflow-y-auto p-8 max-w-2xl mx-auto animate-[fadeIn_0.3s_ease-out]">
              <div className="flex items-center justify-between mb-8">
                <h1 className="text-2xl font-bold">{t('settings.title')}</h1>
                <div className="px-3 py-1 bg-green-500/10 border border-green-500/20 rounded-full text-[10px] text-green-400 font-medium flex items-center gap-2">
                  <Shield size={12} /> {t('settings.privacyBadge')}
                </div>
              </div>

              <div className="glass-panel p-6 mb-8">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div>
                    <h2 className="text-lg font-semibold">
                      {t('settings.interfaceLanguage')}
                    </h2>
                    <p className="text-xs text-zinc-500 mt-1">
                      {t('settings.interfaceLanguageHelp')}
                    </p>
                  </div>
                  <LanguageSwitcher />
                </div>
              </div>

              <KeyInput
                onKeySet={setApiKey}
                savedKey={apiKey}
                onBaseUrlSet={setGeminiBaseUrl}
                savedBaseUrl={geminiBaseUrl}
              >
                <GeminiAccessModeSettings
                  geminiAccessMode={geminiAccessMode}
                  setGeminiAccessMode={setGeminiAccessMode}
                  geminiKeyPoolText={geminiKeyPoolText}
                  setGeminiKeyPoolText={setGeminiKeyPoolText}
                  geminiKeyPool={geminiKeyPool}
                  geminiKeyPoolStats={geminiKeyPoolStats}
                  setGeminiKeyPoolStats={setGeminiKeyPoolStats}
                />
              </KeyInput>

              <div className="glass-panel p-6 mt-8">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className="text-lg font-semibold">OpenAI 兼容多模态 API</h2>
                    <p className="text-xs text-zinc-500 mt-1">用于二创解说的“OpenAI 兼容多模态”模式，支持 OpenAI / OpenRouter / 硅基流动 / 火山 / 通义兼容接口。</p>
                  </div>
                  <span className="text-[10px] bg-cyan-500/10 border border-cyan-500/20 px-2 py-0.5 rounded text-cyan-400 uppercase tracking-wider">
                    optional
                  </span>
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm text-zinc-400 mb-2">API Key</label>
                    <input
                      type="password"
                      value={openAICompatibleKey}
                      onChange={(e) => setOpenAICompatibleKey(e.target.value)}
                      className="input-field font-mono"
                      placeholder="sk-..."
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-zinc-400 mb-2">Base URL</label>
                    <input
                      type="url"
                      value={openAICompatibleBaseUrl}
                      onChange={(e) => setOpenAICompatibleBaseUrl(e.target.value)}
                      className="input-field font-mono"
                      placeholder="https://your-openai-compatible-provider/v1"
                    />
                  </div>
                  <div>
                    <label className="block text-sm text-zinc-400 mb-2">Model</label>
                    <input
                      value={openAICompatibleModel}
                      onChange={(e) => setOpenAICompatibleModel(e.target.value)}
                      className="input-field font-mono"
                      placeholder="vision-capable-model-name"
                    />
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    这个模式不会把视频上传给 Gemini；后端会转录完整视频，并按时间线抽取密集画面帧，分批发送到 OpenAI 兼容 chat/completions 多模态接口。
                  </p>
                </div>
              </div>

              <div className="glass-panel p-6 mt-8">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">
                    {t('settings.socialTitle')}
                  </h2>
                  <span className="text-[10px] bg-white/5 border border-white/5 px-2 py-0.5 rounded text-zinc-500 uppercase tracking-wider">
                    {t('common.optional')}
                  </span>
                </div>
                <p className="text-xs text-zinc-500 mb-6 leading-relaxed">
                  {t('settings.socialDescription')}
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-zinc-400">
                    {t('settings.uploadPostKey')}
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={uploadPostKey}
                      onChange={(e) => setUploadPostKey(e.target.value)}
                      className="input-field"
                      placeholder="ey..."
                    />
                    <button
                      onClick={fetchUserProfiles}
                      className="btn-primary py-2 px-4 text-sm"
                    >
                      {t('common.connect')}
                    </button>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    {t('settings.connectUploadPost')}
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2">
                      <a
                        href="https://app.upload-post.com/login"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1"
                      >
                        <span className="text-zinc-400 font-medium">
                          {t('settings.login')}
                        </span>
                        <span className="text-[10px] text-zinc-600">
                          {t('settings.registerAccount')}
                        </span>
                      </a>
                      <a
                        href="https://app.upload-post.com/manage-users"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1"
                      >
                        <span className="text-zinc-400 font-medium">
                          {t('settings.profiles')}
                        </span>
                        <span className="text-[10px] text-zinc-600">
                          {t('settings.createConnect')}
                        </span>
                      </a>
                      <a
                        href="https://app.upload-post.com/api-keys"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1"
                      >
                        <span className="text-zinc-400 font-medium">
                          {t('settings.apiKeyStep')}
                        </span>
                        <span className="text-[10px] text-zinc-600">
                          {t('settings.generateKey')}
                        </span>
                      </a>
                    </div>
                    <br />
                    <span className="text-zinc-600 italic">
                      {t('settings.keysBrowserOnly')}
                    </span>
                  </p>
                </div>
              </div>

              <div className="glass-panel p-6 mt-8">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">
                    {t('settings.translationTitle')}
                  </h2>
                  <span className="text-[10px] bg-white/5 border border-white/5 px-2 py-0.5 rounded text-zinc-500 uppercase tracking-wider">
                    {t('common.optional')}
                  </span>
                </div>
                <p className="text-xs text-zinc-500 mb-6 leading-relaxed">
                  {t('settings.translationDescription')}
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-zinc-400">
                    {t('settings.elevenLabsKey')}
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={elevenLabsKey}
                      onChange={(e) => setElevenLabsKey(e.target.value)}
                      className="input-field"
                      placeholder="sk_..."
                    />
                    <button
                      onClick={() => {
                        if (elevenLabsKey) {
                          localStorage.setItem(
                            'elevenLabsKey_v1',
                            encrypt(elevenLabsKey),
                          )
                          alert(t('settings.savedElevenLabs'))
                        }
                      }}
                      className="btn-primary py-2 px-4 text-sm"
                    >
                      {t('common.save')}
                    </button>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    {t('settings.getElevenLabsKey')}
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <a
                        href="https://elevenlabs.io/sign-up"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1"
                      >
                        <span className="text-zinc-400 font-medium">
                          {t('settings.signUp')}
                        </span>
                        <span className="text-[10px] text-zinc-600">
                          {t('settings.createAccount')}
                        </span>
                      </a>
                      <a
                        href="https://elevenlabs.io/app/settings/api-keys"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1"
                      >
                        <span className="text-zinc-400 font-medium">
                          {t('settings.apiKey')}
                        </span>
                        <span className="text-[10px] text-zinc-600">
                          {t('settings.generateKey')}
                        </span>
                      </a>
                    </div>
                    <br />
                    <span className="text-zinc-600 italic">
                      {t('settings.keysBrowserOnly')}
                    </span>
                  </p>
                </div>
              </div>

              <div className="glass-panel p-6 mt-8">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">
                    {t('settings.aiShortsTitle')}
                  </h2>
                  <span className="text-[10px] bg-violet-500/10 border border-violet-500/20 px-2 py-0.5 rounded text-violet-400 uppercase tracking-wider">
                    {t('common.new')}
                  </span>
                </div>
                <p className="text-xs text-zinc-500 mb-6 leading-relaxed">
                  {t('settings.aiShortsDescription')}
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-zinc-400">
                    {t('settings.falKey')}
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={falKey}
                      onChange={(e) => setFalKey(e.target.value)}
                      className="input-field"
                      placeholder="fal_..."
                    />
                    <button
                      onClick={() => {
                        if (falKey) {
                          localStorage.setItem('falKey_v1', encrypt(falKey))
                          alert(t('settings.savedFal'))
                        }
                      }}
                      className="btn-primary py-2 px-4 text-sm"
                    >
                      {t('common.save')}
                    </button>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    {t('settings.getFalKey')}
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <a
                        href="https://fal.ai/dashboard/keys"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1"
                      >
                        <span className="text-zinc-400 font-medium">
                          {t('settings.signUp')}
                        </span>
                        <span className="text-[10px] text-zinc-600">
                          {t('settings.createAccount')}
                        </span>
                      </a>
                      <a
                        href="https://fal.ai/dashboard/keys"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1"
                      >
                        <span className="text-zinc-400 font-medium">
                          {t('settings.apiKey')}
                        </span>
                        <span className="text-[10px] text-zinc-600">
                          {t('settings.generateKey')}
                        </span>
                      </a>
                    </div>
                    <br />
                    <span className="text-zinc-600 italic">
                      {t('settings.keysBrowserOnly')}
                    </span>
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* View: SaaS Shorts */}
          {activeTab === 'saasshorts' && (
            <SaaShortsTab
              geminiApiKey={apiKey}
              geminiBaseUrl={geminiBaseUrl}
              geminiConfig={geminiConfig}
              elevenLabsKey={elevenLabsKey}
              falKey={falKey}
              uploadPostKey={uploadPostKey}
              uploadUserId={uploadUserId}
            />
          )}

          {/* View: Commentary Remix */}
          {activeTab === 'commentary' && (
            <CommentaryTab
              geminiApiKey={apiKey}
              geminiBaseUrl={geminiBaseUrl}
              geminiConfig={geminiConfig}
              openAICompatibleConfig={{
                apiKey: openAICompatibleKey,
                baseUrl: openAICompatibleBaseUrl,
                model: openAICompatibleModel,
              }}
              elevenLabsKey={elevenLabsKey}
              setGeminiKeyPoolStats={setGeminiKeyPoolStats}
            />
          )}

          {/* View: UGC Gallery */}
          {activeTab === 'ugc-gallery' && <UGCGallery />}

          {/* View: Thumbnails */}
          {activeTab === 'thumbnails' && (
            <ThumbnailStudio
              geminiApiKey={apiKey}
              geminiBaseUrl={geminiBaseUrl}
              geminiConfig={geminiConfig}
              uploadPostKey={uploadPostKey}
              uploadUserId={uploadUserId}
            />
          )}

          {/* View: Gallery */}
          {/* {activeTab === 'gallery' && (
            <Gallery />
          )} */}

          {/* View: Dashboard (Idle) */}
          {activeTab === 'dashboard' && status === 'idle' && (
            <div className="h-full flex flex-col items-center justify-center p-6 animate-[fadeIn_0.3s_ease-out]">
              <div className="max-w-xl w-full text-center space-y-8">
                <div className="space-y-4">
                  <h1 className="text-4xl md:text-5xl font-black bg-gradient-to-b from-white to-white/60 bg-clip-text text-transparent">
                    {t('clipGenerator.title')}
                  </h1>
                  <p className="text-zinc-400 text-lg">
                    {t('clipGenerator.subtitle')}
                  </p>
                </div>

                <MediaInput
                  onProcess={handleProcess}
                  isProcessing={status === 'processing'}
                />

                <div className="flex items-center justify-center gap-8 text-zinc-500 text-sm">
                  <span className="flex items-center gap-2">
                    <Youtube size={16} /> YouTube
                  </span>
                  <span className="flex items-center gap-2">
                    <Instagram size={16} /> Instagram
                  </span>
                  <span className="flex items-center gap-2">
                    <TikTokIcon size={16} /> TikTok
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* View: Processing / Results (Split View) */}
          {activeTab === 'dashboard' &&
            (status === 'processing' ||
              status === 'complete' ||
              status === 'error') && (
              <div className="h-full flex flex-col md:flex-row animate-[fadeIn_0.3s_ease-out]">
                {/* Left Panel: Preview & Status */}
                <div
                  className={`${status === 'complete' ? 'w-full md:w-[30%] lg:w-[25%]' : 'w-full md:w-[55%] lg:w-[60%]'} h-full flex flex-col border-r border-white/5 bg-black/20 p-6 overflow-y-auto custom-scrollbar transition-all duration-700 ease-in-out`}
                >
                  <div className="mb-6 flex items-center justify-between">
                    <h2 className="text-lg font-semibold flex items-center gap-2">
                      <Activity
                        className={`text-primary ${status === 'processing' ? 'animate-pulse' : ''}`}
                        size={20}
                      />
                      {t('clipGenerator.liveAnalysis')}
                    </h2>
                    <span
                      className={`text-xs px-2 py-1 rounded-full border ${
                        status === 'processing'
                          ? 'bg-primary/10 border-primary/20 text-primary'
                          : status === 'complete'
                            ? 'bg-green-500/10 border-green-500/20 text-green-400'
                            : 'bg-red-500/10 border-red-500/20 text-red-400'
                      }`}
                    >
                      {t(
                        `common.status.${status === 'complete' ? 'complete' : status}`,
                      )}
                    </span>
                  </div>

                  {/* Video Preview */}
                  {processingMedia && (
                    <ProcessingAnimation
                      media={processingMedia}
                      isComplete={status === 'complete'}
                      syncedTime={syncedTime}
                      isSyncedPlaying={isSyncedPlaying}
                      syncTrigger={syncTrigger}
                    />
                  )}

                  {/* Logs Terminal */}
                  <div
                    className={`bg-[#0c0c0e] rounded-xl border border-white/10 overflow-hidden flex flex-col transition-all duration-500 ${status === 'complete' ? 'h-32 min-h-0 opacity-50 hover:opacity-100' : 'flex-1 min-h-[200px]'}`}
                  >
                    <div className="px-4 py-2 border-b border-white/5 flex items-center justify-between bg-white/5 shrink-0">
                      <span className="text-xs font-mono text-zinc-400 flex items-center gap-2">
                        <Terminal size={12} /> {t('clipGenerator.systemLogs')}
                      </span>
                      <button
                        onClick={() => setLogsVisible(!logsVisible)}
                        className="text-zinc-500 hover:text-white transition-colors"
                      >
                        {logsVisible ? (
                          <ChevronDown size={14} />
                        ) : (
                          <ChevronDown size={14} className="rotate-180" />
                        )}
                      </button>
                    </div>
                    {logsVisible && (
                      <div className="flex-1 p-4 overflow-y-auto font-mono text-xs space-y-1.5 custom-scrollbar text-zinc-400">
                        {logs.map((log, i) => (
                          <div
                            key={i}
                            className={`flex gap-2 ${log.toLowerCase().includes('error') ? 'text-red-400' : 'text-zinc-400'}`}
                          >
                            <span className="text-zinc-700 shrink-0">
                              {new Date().toLocaleTimeString()}
                            </span>
                            <span>{log}</span>
                          </div>
                        ))}
                        {status === 'processing' && (
                          <div className="animate-pulse text-primary/70">_</div>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Panel: Results Grid */}
                <div
                  className={`${status === 'complete' ? 'w-full md:w-[70%] lg:w-[75%]' : 'w-full md:w-[45%] lg:w-[40%]'} h-full flex flex-col bg-background p-6 transition-all duration-700 ease-in-out`}
                >
                  <h2 className="text-lg font-semibold mb-6 flex items-center gap-2 shrink-0">
                    <Sparkles className="text-yellow-400" size={20} />
                    {t('clipGenerator.generatedShorts')}
                    {results?.clips?.length > 0 && (
                      <span className="text-xs bg-white/10 text-white px-2 py-0.5 rounded-full ml-auto">
                        {t('common.clips', { count: results.clips.length })}
                      </span>
                    )}
                    {results?.cost_analysis && (
                      <span
                        className="text-xs bg-green-500/10 border border-green-500/20 text-green-400 px-2 py-0.5 rounded-full ml-2"
                        title={t('clipGenerator.costTitle', {
                          input: results.cost_analysis.input_tokens,
                          output: results.cost_analysis.output_tokens,
                        })}
                      >
                        ${results.cost_analysis.total_cost.toFixed(5)}
                      </span>
                    )}
                    {results?.clips?.length > 1 && status === 'complete' && (
                      <button
                        onClick={() => setShowScheduleWeek(true)}
                        className="ml-auto flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-purple-500/20 to-indigo-500/20 hover:from-purple-500/30 hover:to-indigo-500/30 border border-purple-500/30 text-purple-300 hover:text-purple-200 rounded-full text-xs font-bold transition-all"
                      >
                        <Calendar size={14} />
                        {t('clipGenerator.scheduleWeek')}
                      </button>
                    )}
                  </h2>

                  <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
                    {results && results.clips && results.clips.length > 0 ? (
                      <div
                        className={`grid gap-4 pb-10 ${status === 'complete' ? 'grid-cols-1 xl:grid-cols-2' : 'grid-cols-1'}`}
                      >
                        {results.clips.map((clip, i) => (
                          <ResultCard
                            key={i}
                            clip={clip}
                            index={i}
                            jobId={jobId}
                            uploadPostKey={uploadPostKey}
                            uploadUserId={uploadUserId}
                            geminiApiKey={apiKey}
                            geminiBaseUrl={geminiBaseUrl}
                            geminiConfig={geminiConfig}
                            elevenLabsKey={elevenLabsKey}
                            onPlay={(time) => handleClipPlay(time)}
                            onPause={handleClipPause}
                          />
                        ))}
                      </div>
                    ) : status === 'processing' ? (
                      <div className="h-full flex flex-col items-center justify-center text-zinc-500 space-y-4 opacity-50">
                        <div className="w-12 h-12 rounded-full border-2 border-zinc-800 border-t-primary animate-spin" />
                        <p className="text-sm">
                          {t('clipGenerator.waitingForClips')}
                        </p>
                      </div>
                    ) : status === 'error' ? (
                      <div className="h-full flex flex-col items-center justify-center text-red-400 space-y-2">
                        <p>{t('clipGenerator.generationFailed')}</p>
                      </div>
                    ) : null}
                  </div>
                </div>
              </div>
            )}
        </div>

        {/* Footer */}
        <div className="h-8 border-t border-white/5 flex items-center justify-center shrink-0">
          <span className="text-[10px] text-zinc-600">
            {t('clipGenerator.footer')}{' '}
            <a
              href="https://www.upload-post.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-500 hover:text-white transition-colors"
            >
              Upload-Post
            </a>
          </span>
        </div>
      </main>

      {/* Missing API Key Modal */}
      {showKeyModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={() => setShowKeyModal(false)}
        >
          <div
            className="bg-[#18181b] border border-white/10 rounded-2xl p-6 max-w-md w-full mx-4 space-y-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-bold text-white">
              {t('keyModal.title')}
            </h2>
            <p className="text-sm text-zinc-400">{t('keyModal.description')}</p>
            <div className="bg-white/5 border border-white/10 rounded-lg p-4 space-y-2">
              <p className="text-xs font-semibold text-zinc-300">
                {t('keyModal.howTo')}
              </p>
              <ol className="text-xs text-zinc-400 space-y-1 list-decimal list-inside">
                <li>
                  {t('keyModal.step1')}{' '}
                  <a
                    href="https://aistudio.google.com/app/apikey"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 underline"
                  >
                    aistudio.google.com/app/apikey
                  </a>
                </li>
                <li>{t('keyModal.step2')}</li>
                <li>{t('keyModal.step3')}</li>
                <li>{t('keyModal.step4')}</li>
              </ol>
            </div>
            <input
              type="text"
              placeholder={t('keyModal.placeholder')}
              className="w-full bg-black/50 border border-white/20 rounded-lg px-4 py-2.5 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-blue-500"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.target.value.trim()) {
                  setApiKey(e.target.value.trim())
                  setShowKeyModal(false)
                }
              }}
            />

            {/* Upload-Post info */}
            <div className="bg-violet-500/5 border border-violet-500/20 rounded-lg p-4 space-y-2">
              <p className="text-xs font-semibold text-violet-300">
                {t('keyModal.uploadPostTitle')}
              </p>
              <p className="text-xs text-zinc-400">
                {t('keyModal.uploadPostDescription')}
              </p>
              <ol className="text-xs text-zinc-400 space-y-1 list-decimal list-inside">
                <li>
                  {t('keyModal.uploadPostStep1')}{' '}
                  <a
                    href="https://app.upload-post.com/login"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-violet-400 underline"
                  >
                    app.upload-post.com
                  </a>
                </li>
                <li>{t('keyModal.uploadPostStep2')}</li>
                <li>{t('keyModal.uploadPostStep3')}</li>
                <li>{t('keyModal.uploadPostStep4')}</li>
              </ol>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => setShowKeyModal(false)}
                className="flex-1 text-sm text-zinc-400 py-2 rounded-lg border border-white/10 hover:bg-white/5 transition-colors"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={() => {
                  setShowKeyModal(false)
                  setActiveTab('settings')
                }}
                className="flex-1 text-sm text-white py-2 rounded-lg bg-blue-600 hover:bg-blue-500 transition-colors font-medium"
              >
                {t('keyModal.goToSettings')}
              </button>
            </div>
          </div>
        </div>
      )}

      <ScheduleWeekModal
        isOpen={showScheduleWeek}
        onClose={() => setShowScheduleWeek(false)}
        clips={results?.clips || []}
        jobId={jobId}
        uploadPostKey={uploadPostKey}
        uploadUserId={uploadUserId}
      />
    </div>
  )
}

export default App
