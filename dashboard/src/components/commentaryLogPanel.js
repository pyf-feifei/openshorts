export const COMMENTARY_LOG_PANEL_BODY_CLASS = 'max-h-[320px] overflow-y-auto custom-scrollbar rounded-b-xl bg-black/30 border-x border-b border-white/5 p-4 font-mono text-xs text-zinc-400 space-y-2'

export const getCommentaryLogPanelState = (logs = [], expanded = true) => {
  const logCount = Array.isArray(logs) ? logs.length : 0
  return {
    logCount,
    countLabel: logCount > 0 ? `${logCount} 条日志` : '暂无日志',
    toggleLabel: expanded ? '收起运行日志' : '展开运行日志',
    emptyText: '等待开始...',
  }
}
