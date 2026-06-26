import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle,
  CheckSquare,
  Copy,
  Download,
  Eye,
  FileText,
  Link,
  Loader2,
  RefreshCcw,
  Search,
  Square,
  Trash2,
  X,
} from 'lucide-react'
import { getApiUrl } from '../config'
import { useI18n } from '../i18n/I18nProvider'
import { buildOpenAICompatibleHeaders, hasOpenAICompatibleAccess } from '../lib/openaiCompatibleHeaders'
import { COMMENTARY_DEFAULTS } from './commentaryDefaults'
import { mergeStyleIntoLocalStorage } from './commentaryStyleSync'

const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled'])
const TRANSLATED_STATUSES = new Set(['processing', 'completed', 'failed', 'cancelled'])
const TRANSLATED_STAGES = new Set(['queued', 'fetch', 'rank', 'media', 'download', 'transcribe', 'analyze', 'synthesize', 'aggregate', 'completed', 'done', 'failed', 'cancelled'])

function translateStatus(status, t) {
  const value = String(status || '').trim()
  if (TRANSLATED_STATUSES.has(value)) return t(`styleLearning.status.${value}`)
  return value || t('styleLearning.status.unknown')
}

function translateStage(job, t) {
  const stage = String(job?.stage || '').trim()
  if (TRANSLATED_STAGES.has(stage)) return t(`styleLearning.stage.${stage}`)
  return job?.stage_label || stage || t('styleLearning.stage.preparing')
}

async function readErrorMessage(res, fallback) {
  const rawText = await res.text().catch(() => '')
  if (!rawText) return fallback
  try {
    const data = JSON.parse(rawText)
    return data.detail?.message || data.detail || data.message || rawText
  } catch {
    return rawText
  }
}

function formatCount(value) {
  const num = Number(value || 0)
  if (!Number.isFinite(num)) return '0'
  if (num >= 10000) return `${(num / 10000).toFixed(num >= 100000 ? 0 : 1)}w`
  return String(Math.round(num))
}

function compactText(value, max = 90) {
  const text = String(value || '').replace(/\s+/g, ' ').trim()
  if (text.length <= max) return text
  return `${text.slice(0, max - 3)}...`
}

export default function CommentaryStyleLearner({ openAICompatibleConfig }) {
  const { t } = useI18n()
  const [profileUrl, setProfileUrl] = useState('')
  const [styleName, setStyleName] = useState('')
  const [language, setLanguage] = useState(COMMENTARY_DEFAULTS.language)
  const [jobId, setJobId] = useState(null)
  const [job, setJob] = useState(null)
  const [jobs, setJobs] = useState([])
  const [selectedJobIds, setSelectedJobIds] = useState([])
  const [deletingJobIds, setDeletingJobIds] = useState([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)
  const [promptModalOpen, setPromptModalOpen] = useState(false)
  const syncedStyleIdRef = useRef(null)
  const pollingRef = useRef(null)
  const hasOpenAI = hasOpenAICompatibleAccess(openAICompatibleConfig)

  const currentJob = job || jobs.find((item) => item.job_id === jobId) || null
  const learnedStyle = currentJob?.style || currentJob?.result?.style || null
  const selectedVideos = currentJob?.selected_videos || currentJob?.result?.selected_videos || []
  const failedVideos = currentJob?.failed_videos || currentJob?.result?.failed_videos || []
  const totalVideos = Number(currentJob?.total_videos ?? currentJob?.video_count ?? currentJob?.result?.video_count ?? 0)
  const selectedCount = Number(currentJob?.selected_count ?? currentJob?.result?.selected_count ?? selectedVideos.length ?? 0)
  const visibleRankingCount = Math.min(selectedVideos.length, 100)
  const progressValue = Number(currentJob?.stage_progress)
  const hasProgress = Number.isFinite(progressValue)
  const currentStatusLabel = currentJob ? translateStatus(currentJob.status, t) : ''
  const currentStageLabel = currentJob ? translateStage(currentJob, t) : ''
  const visibleJobIds = jobs.map((item) => item.job_id).filter(Boolean)
  const selectedVisibleJobIds = selectedJobIds.filter((id) => visibleJobIds.includes(id))
  const allVisibleJobsSelected = visibleJobIds.length > 0 && visibleJobIds.every((id) => selectedJobIds.includes(id))
  const isDeletingJobs = deletingJobIds.length > 0

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch(getApiUrl('/api/commentary/style-learning/jobs'))
      if (!res.ok) return
      const data = await res.json()
      const list = Array.isArray(data.jobs) ? data.jobs : []
      setJobs(list)
      if (!jobId && list[0]) {
        setJobId(list[0].job_id)
        setJob(list[0])
      }
    } catch {
      // best-effort history
    }
  }, [jobId])

  const fetchJob = useCallback(async (id) => {
    if (!id) return null
    const res = await fetch(getApiUrl(`/api/commentary/style-learning/jobs/${id}`))
    if (!res.ok) throw new Error(await readErrorMessage(res, t('styleLearning.errors.fetchStatus')))
    const data = await res.json()
    setJob(data)
    setJobs((items) => {
      const without = items.filter((item) => item.job_id !== data.job_id)
      return [data, ...without].slice(0, 30)
    })
    return data
  }, [t])

  useEffect(() => {
    fetchJobs()
  }, [fetchJobs])

  useEffect(() => {
    const visibleIds = new Set(jobs.map((item) => item.job_id).filter(Boolean))
    setSelectedJobIds((ids) => {
      const next = ids.filter((id) => visibleIds.has(id))
      return next.length === ids.length ? ids : next
    })
  }, [jobs])

  useEffect(() => {
    if (!jobId) return undefined
    let cancelled = false
    const tick = async () => {
      try {
        const data = await fetchJob(jobId)
        if (!cancelled && data && TERMINAL_STATUSES.has(data.status)) {
          clearInterval(pollingRef.current)
          pollingRef.current = null
          fetchJobs()
        }
      } catch (e) {
        if (!cancelled) setError(e.message)
      }
    }
    tick()
    pollingRef.current = setInterval(tick, 2500)
    return () => {
      cancelled = true
      clearInterval(pollingRef.current)
      pollingRef.current = null
    }
  }, [jobId, fetchJob, fetchJobs])

  useEffect(() => {
    if (!learnedStyle || syncedStyleIdRef.current === learnedStyle.id) return
    const merged = mergeStyleIntoLocalStorage(learnedStyle)
    if (merged) syncedStyleIdRef.current = learnedStyle.id
  }, [learnedStyle])

  const createJob = async () => {
    if (!profileUrl.trim()) {
      setError(t('styleLearning.errors.profileRequired'))
      return
    }
    if (!hasOpenAI) {
      setError(t('styleLearning.errors.openAIRequired'))
      return
    }
    setError('')
    setSubmitting(true)
    setJob(null)
    try {
      const res = await fetch(getApiUrl('/api/commentary/style-learning/jobs'), {
        method: 'POST',
        headers: buildOpenAICompatibleHeaders(openAICompatibleConfig, { 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          profile_url: profileUrl.trim(),
          style_name: styleName.trim() || undefined,
          max_videos: 100,
          language,
        }),
      })
      if (!res.ok) throw new Error(await readErrorMessage(res, t('styleLearning.errors.startFailed')))
      const data = await res.json()
      setJobId(data.job_id)
      fetchJobs()
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  const cancelJob = async () => {
    if (!jobId || currentJob?.status !== 'processing') return
    try {
      await fetch(getApiUrl(`/api/commentary/style-learning/jobs/${jobId}/cancel`), { method: 'POST' })
      fetchJob(jobId)
    } catch (e) {
      setError(e.message)
    }
  }

  const selectJob = (item) => {
    setJobId(item.job_id)
    setJob(item)
    setError('')
  }

  const toggleJobSelection = (id) => {
    setSelectedJobIds((ids) => (
      ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id]
    ))
  }

  const toggleAllJobSelection = () => {
    setSelectedJobIds((ids) => (
      visibleJobIds.length > 0 && visibleJobIds.every((id) => ids.includes(id)) ? [] : visibleJobIds
    ))
  }

  const deleteJobs = async (jobIds) => {
    const ids = Array.from(new Set((jobIds || []).filter(Boolean)))
    if (ids.length === 0 || isDeletingJobs) return
    const message = ids.length === 1
      ? t('styleLearning.deleteConfirmOne')
      : t('styleLearning.deleteConfirmMany', { count: ids.length })
    if (!window.confirm(message)) return

    setError('')
    setDeletingJobIds(ids)
    try {
      let res
      if (ids.length === 1) {
        res = await fetch(getApiUrl(`/api/commentary/style-learning/jobs/${ids[0]}`), { method: 'DELETE' })
      } else {
        res = await fetch(getApiUrl('/api/commentary/style-learning/jobs/delete'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ job_ids: ids }),
        })
      }
      if (!res.ok) throw new Error(await readErrorMessage(res, t('styleLearning.errors.deleteFailed')))
      const data = await res.json().catch(() => ({}))
      if (Array.isArray(data.errors) && data.errors.length > 0) {
        throw new Error(data.errors.map((item) => `${item.job_id}: ${item.error}`).join('\n'))
      }
      const remainingJobs = jobs.filter((item) => !ids.includes(item.job_id))
      setJobs(remainingJobs)
      setSelectedJobIds((selected) => selected.filter((id) => !ids.includes(id)))
      if (jobId && ids.includes(jobId)) {
        const nextJob = remainingJobs[0] || null
        setJobId(nextJob?.job_id || null)
        setJob(nextJob)
        setPromptModalOpen(false)
        setCopied(false)
      }
      fetchJobs()
    } catch (e) {
      setError(e.message || t('styleLearning.errors.deleteFailed'))
    } finally {
      setDeletingJobIds([])
    }
  }

  const copyPrompt = async () => {
    if (!learnedStyle?.prompt) return
    try {
      await navigator.clipboard.writeText(learnedStyle.prompt)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setError(t('styleLearning.errors.copyFailed'))
    }
  }

  const safeFilename = (name, fallback) => {
    const value = String(name || fallback || 'douyin-style')
      .trim()
      .replace(/[\\/:*?"<>|]+/g, '-')
      .replace(/\s+/g, '-')
      .slice(0, 80)
    return value || fallback
  }

  const downloadText = (filename, text, type = 'text/plain;charset=utf-8') => {
    const blob = new Blob([text], { type })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }

  const downloadPrompt = () => {
    if (!learnedStyle?.prompt) return
    downloadText(`${safeFilename(learnedStyle.label, 'douyin-style')}.txt`, learnedStyle.prompt)
  }

  const downloadStyleJson = () => {
    if (!learnedStyle) return
    downloadText(
      `${safeFilename(learnedStyle.label, 'douyin-style')}.json`,
      JSON.stringify(learnedStyle, null, 2),
      'application/json;charset=utf-8',
    )
  }

  const statusBadgeClass = useMemo(() => {
    if (!currentJob) return 'bg-white/5 text-zinc-400'
    if (currentJob.status === 'completed') return 'bg-green-500/10 text-green-300 border-green-500/20'
    if (currentJob.status === 'failed') return 'bg-red-500/10 text-red-300 border-red-500/20'
    if (currentJob.status === 'cancelled') return 'bg-zinc-500/10 text-zinc-300 border-zinc-500/20'
    return 'bg-cyan-500/10 text-cyan-300 border-cyan-500/20'
  }, [currentJob])

  return (
    <div className="h-full overflow-y-auto p-6 lg:p-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">{t('styleLearning.title')}</h1>
            <p className="mt-2 max-w-2xl text-sm text-zinc-500">
              {t('styleLearning.description')}
            </p>
          </div>
          <button type="button" onClick={fetchJobs} className="btn-secondary inline-flex items-center justify-center gap-2 px-4 py-2 text-sm">
            <RefreshCcw size={15} /> {t('styleLearning.refresh')}
          </button>
        </div>

        {error && (
          <div className="flex items-start gap-2 rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-200">
            <AlertTriangle size={18} className="mt-0.5 shrink-0" />
            <span className="break-words">{String(error)}</span>
          </div>
        )}

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="space-y-6">
            <section className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
              <div className="mb-4 flex items-center gap-2">
                <Link size={18} className="text-cyan-300" />
                <h2 className="text-base font-semibold text-zinc-100">{t('styleLearning.profile')}</h2>
              </div>
              <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_220px_260px]">
                <input
                  value={profileUrl}
                  onChange={(event) => setProfileUrl(event.target.value)}
                  className="input-field"
                  placeholder="https://www.douyin.com/user/MS4wLjAB..."
                />
                <select
                  value={language}
                  onChange={(event) => setLanguage(event.target.value)}
                  className="input-field"
                  aria-label={t('styleLearning.language')}
                >
                  <option value="zh">{t('styleLearning.languages.zh')}</option>
                  <option value="en">{t('styleLearning.languages.en')}</option>
                  <option value="es">{t('styleLearning.languages.es')}</option>
                  <option value="ja">{t('styleLearning.languages.ja')}</option>
                </select>
                <input
                  value={styleName}
                  onChange={(event) => setStyleName(event.target.value)}
                  className="input-field"
                  maxLength={40}
                  placeholder={t('styleLearning.styleNamePlaceholder')}
                />
              </div>
              <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center">
                <button
                  type="button"
                  onClick={createJob}
                  disabled={submitting || currentJob?.status === 'processing'}
                  className="btn-primary inline-flex items-center justify-center gap-2 px-5 py-2.5 text-sm disabled:opacity-60"
                >
                  {submitting ? <Loader2 size={16} className="animate-spin" /> : <Search size={16} />}
                  {t('styleLearning.startLearning')}
                </button>
                {!hasOpenAI && (
                  <p className="text-xs text-yellow-300">{t('styleLearning.openAIMissing')}</p>
                )}
              </div>
            </section>

            {currentJob && (
              <section className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
                <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <h2 className="text-base font-semibold text-zinc-100">{t('styleLearning.jobStatus')}</h2>
                      <span className={`rounded-full border px-2.5 py-1 text-xs ${statusBadgeClass}`}>
                        {currentStatusLabel}
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-zinc-500">{currentStageLabel}</p>
                  </div>
                  {currentJob.status === 'processing' && (
                    <button type="button" onClick={cancelJob} className="btn-secondary inline-flex items-center justify-center gap-2 px-3 py-2 text-sm text-red-200">
                      <X size={15} /> {t('common.cancel')}
                    </button>
                  )}
                </div>
                {hasProgress && (
                  <div className="mb-4 h-2 overflow-hidden rounded-full bg-white/10">
                    <div className="h-full rounded-full bg-cyan-400 transition-all" style={{ width: `${Math.max(0, Math.min(100, progressValue))}%` }} />
                  </div>
                )}
                <div className="grid gap-3 sm:grid-cols-4">
                  <div className="rounded-lg border border-white/10 bg-black/20 p-3">
                    <div className="text-xs text-zinc-500">{t('styleLearning.metrics.selected')}</div>
                    <div className="mt-1 text-xl font-semibold text-white">{selectedCount || selectedVideos.length || 0}</div>
                  </div>
                  <div className="rounded-lg border border-white/10 bg-black/20 p-3">
                    <div className="text-xs text-zinc-500">{t('styleLearning.metrics.downloaded')}</div>
                    <div className="mt-1 text-xl font-semibold text-white">{currentJob.downloaded_count || 0}</div>
                  </div>
                  <div className="rounded-lg border border-white/10 bg-black/20 p-3">
                    <div className="text-xs text-zinc-500">{t('styleLearning.metrics.transcribed')}</div>
                    <div className="mt-1 text-xl font-semibold text-white">{currentJob.transcript_count || 0}</div>
                  </div>
                  <div className="rounded-lg border border-white/10 bg-black/20 p-3">
                    <div className="text-xs text-zinc-500">{t('styleLearning.metrics.failed')}</div>
                    <div className="mt-1 text-xl font-semibold text-white">{failedVideos.length}</div>
                  </div>
                </div>
              </section>
            )}

            {learnedStyle && (
              <section className="rounded-xl border border-green-500/20 bg-green-500/[0.04] p-5">
                <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle size={18} className="text-green-300" />
                    <h2 className="text-base font-semibold text-zinc-100">{learnedStyle.label}</h2>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button type="button" onClick={() => setPromptModalOpen(true)} className="btn-secondary inline-flex items-center justify-center gap-2 px-3 py-2 text-sm">
                      <Eye size={15} /> {t('styleLearning.viewFullPrompt')}
                    </button>
                    <button type="button" onClick={copyPrompt} className="btn-secondary inline-flex items-center justify-center gap-2 px-3 py-2 text-sm">
                      <Copy size={15} /> {copied ? t('styleLearning.copied') : t('styleLearning.copyPrompt')}
                    </button>
                    <button type="button" onClick={downloadPrompt} className="btn-secondary inline-flex items-center justify-center gap-2 px-3 py-2 text-sm">
                      <Download size={15} /> {t('styleLearning.downloadPrompt')}
                    </button>
                    <button type="button" onClick={downloadStyleJson} className="btn-secondary inline-flex items-center justify-center gap-2 px-3 py-2 text-sm">
                      <Download size={15} /> {t('styleLearning.downloadJson')}
                    </button>
                  </div>
                </div>
                <div className="max-h-[220px] overflow-y-auto rounded-lg border border-white/10 bg-black/30 p-4 text-sm leading-7 text-zinc-100 whitespace-pre-wrap">
                  {learnedStyle.prompt}
                </div>
                <p className="mt-3 text-xs text-green-300">
                  {t('styleLearning.synced')}
                </p>
              </section>
            )}

            {selectedVideos.length > 0 && (
              <section className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
                <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2">
                    <FileText size={18} className="text-cyan-300" />
                    <h2 className="text-base font-semibold text-zinc-100">{t('styleLearning.videoRanking')}</h2>
                  </div>
                  <div className="text-xs text-zinc-500">
                    {t('styleLearning.rankingCount', {
                      total: totalVideos || selectedCount || selectedVideos.length,
                      selected: selectedCount || selectedVideos.length,
                      shown: visibleRankingCount,
                    })}
                  </div>
                </div>
                <div className="max-h-[420px] overflow-y-auto rounded-lg border border-white/10">
                  {selectedVideos.slice(0, 100).map((video) => (
                    <div key={video.aweme_id} className="grid grid-cols-[48px_minmax(0,1fr)_88px_88px_92px] gap-3 border-b border-white/5 px-3 py-2 text-xs last:border-b-0">
                      <span className="text-zinc-500">#{video.rank_index}</span>
                      <a href={video.video_url} target="_blank" rel="noopener noreferrer" className="truncate text-zinc-200 hover:text-cyan-200">
                        {compactText(video.title || video.aweme_id)}
                      </a>
                      <span className="text-zinc-400">{t('styleLearning.ranking.likes')} {formatCount(video.like_count)}</span>
                      <span className="text-zinc-400">{t('styleLearning.ranking.saves')} {formatCount(video.save_count)}</span>
                      <span className="text-zinc-500">{t('styleLearning.ranking.score')} {formatCount(video.rank_score)}</span>
                    </div>
                  ))}
                </div>
              </section>
            )}
          </div>

          <aside className="space-y-6">
            <section className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
              <div className="mb-3 flex flex-col gap-3">
                <h2 className="text-base font-semibold text-zinc-100">{t('styleLearning.history')}</h2>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={toggleAllJobSelection}
                    disabled={jobs.length === 0 || isDeletingJobs}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 text-xs text-zinc-300 hover:bg-white/5 disabled:opacity-40"
                  >
                    {allVisibleJobsSelected ? <CheckSquare size={14} /> : <Square size={14} />}
                    {allVisibleJobsSelected ? t('styleLearning.unselectAll') : t('styleLearning.selectAll')}
                  </button>
                  <button
                    type="button"
                    onClick={() => deleteJobs(selectedVisibleJobIds)}
                    disabled={selectedVisibleJobIds.length === 0 || isDeletingJobs}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-1.5 text-xs text-red-200 hover:bg-red-500/20 disabled:opacity-40"
                  >
                    {isDeletingJobs ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
                    {t('styleLearning.deleteSelected', { count: selectedVisibleJobIds.length ? ` ${selectedVisibleJobIds.length}` : '' })}
                  </button>
                </div>
              </div>
              <div className="space-y-2">
                {jobs.length === 0 && <div className="rounded-lg border border-white/10 p-4 text-sm text-zinc-500">{t('styleLearning.noJobs')}</div>}
                {jobs.map((item) => {
                  const selected = selectedJobIds.includes(item.job_id)
                  const deleting = deletingJobIds.includes(item.job_id)
                  return (
                    <div
                      key={item.job_id}
                      className={`rounded-lg border p-3 transition-all ${item.job_id === jobId ? 'border-cyan-500/40 bg-cyan-500/10' : 'border-white/10 bg-black/20 hover:bg-white/5'}`}
                    >
                      <div className="flex items-start gap-2">
                        <button
                          type="button"
                          onClick={() => toggleJobSelection(item.job_id)}
                          disabled={isDeletingJobs}
                          className="mt-0.5 shrink-0 text-zinc-400 hover:text-cyan-300 disabled:opacity-40"
                          aria-label={selected ? t('styleLearning.unselectJob') : t('styleLearning.selectJob')}
                        >
                          {selected ? <CheckSquare size={16} /> : <Square size={16} />}
                        </button>
                        <button
                          type="button"
                          onClick={() => selectJob(item)}
                          className="min-w-0 flex-1 text-left"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="truncate text-sm font-medium text-zinc-100">{item.style?.label || item.style_name || t('styleLearning.defaultJobName')}</span>
                            {item.status === 'processing' && <Loader2 size={14} className="shrink-0 animate-spin text-cyan-300" />}
                          </div>
                          <div className="mt-1 truncate text-xs text-zinc-500">{item.stage_label || item.status}</div>
                        </button>
                      </div>
                      <div className="mt-3 flex justify-end">
                        <button
                          type="button"
                          onClick={() => deleteJobs([item.job_id])}
                          disabled={isDeletingJobs}
                          className="inline-flex items-center gap-1.5 rounded-lg border border-red-500/20 px-2.5 py-1.5 text-xs text-red-200 hover:bg-red-500/10 disabled:opacity-40"
                        >
                          {deleting ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} />}
                          {t('styleLearning.delete')}
                        </button>
                      </div>
                    </div>
                  )
                })}
              </div>
            </section>

            {currentJob?.logs?.length > 0 && (
              <section className="rounded-xl border border-white/10 bg-white/[0.03] p-5">
                <h2 className="mb-3 text-base font-semibold text-zinc-100">{t('styleLearning.logs')}</h2>
                <div className="max-h-[360px] overflow-y-auto rounded-lg bg-black/40 p-3 font-mono text-xs leading-6 text-zinc-400">
                  {currentJob.logs.slice(-120).map((line, index) => (
                    <div key={`${index}-${line}`} className="break-words">{line}</div>
                  ))}
                </div>
              </section>
            )}
          </aside>
        </div>
      </div>
      {promptModalOpen && learnedStyle && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4">
          <div className="flex max-h-[88vh] w-full max-w-4xl flex-col rounded-xl border border-white/10 bg-[#111114] shadow-2xl">
            <div className="flex items-center justify-between gap-3 border-b border-white/10 p-4">
              <div>
                <h2 className="text-base font-semibold text-white">{learnedStyle.label}</h2>
                <p className="mt-1 text-xs text-zinc-500">{t('styleLearning.fullPrompt')}</p>
              </div>
              <button type="button" onClick={() => setPromptModalOpen(false)} className="rounded-lg p-2 text-zinc-400 hover:bg-white/10 hover:text-white" aria-label="Close">
                <X size={18} />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto p-4">
              <div className="rounded-lg border border-white/10 bg-black/30 p-4 text-sm leading-7 text-zinc-100 whitespace-pre-wrap">
                {learnedStyle.prompt}
              </div>
            </div>
            <div className="flex flex-wrap justify-end gap-2 border-t border-white/10 p-4">
              <button type="button" onClick={copyPrompt} className="btn-secondary inline-flex items-center justify-center gap-2 px-3 py-2 text-sm">
                <Copy size={15} /> {copied ? t('styleLearning.copied') : t('styleLearning.copyPrompt')}
              </button>
              <button type="button" onClick={downloadPrompt} className="btn-primary inline-flex items-center justify-center gap-2 px-3 py-2 text-sm">
                <Download size={15} /> {t('styleLearning.downloadPrompt')}
              </button>
              <button type="button" onClick={downloadStyleJson} className="btn-secondary inline-flex items-center justify-center gap-2 px-3 py-2 text-sm">
                <Download size={15} /> {t('styleLearning.downloadJson')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
