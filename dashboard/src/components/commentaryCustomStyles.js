export const CUSTOM_STYLE_STORAGE_KEY = 'openshorts.commentary.customStyleOptions'
export const CUSTOM_STYLE_PREFIX = 'custom:'

export const createCustomStyleId = () => `${CUSTOM_STYLE_PREFIX}${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`

export const normalizeCustomStyleLabel = (label) => String(label || '').replace(/\s+/g, ' ').trim()
export const normalizeCustomStylePrompt = (prompt) => String(prompt || '').trim()
export const getCustomStyleLabelKey = (label) => normalizeCustomStyleLabel(label).toLowerCase()
export const getCustomStyleKey = (label, prompt) => JSON.stringify([getCustomStyleLabelKey(label), normalizeCustomStylePrompt(prompt)])

export const normalizeCustomStyleOptions = (items) => {
  if (!Array.isArray(items)) return []
  const byLabel = new Map()
  for (const rawItem of items) {
    const item = {
      id: String(rawItem?.id || '').trim(),
      label: normalizeCustomStyleLabel(rawItem?.label),
      prompt: normalizeCustomStylePrompt(rawItem?.prompt),
      custom: true,
    }
    if (!item.id.startsWith(CUSTOM_STYLE_PREFIX) || !item.label || !item.prompt) continue
    byLabel.set(getCustomStyleLabelKey(item.label), item)
  }
  return Array.from(byLabel.values()).slice(-50)
}

export const findCustomStyleOption = (items, label, prompt = '') => {
  const normalized = normalizeCustomStyleOptions(items)
  const targetId = String(label || '').trim()
  const labelKey = getCustomStyleLabelKey(label)
  const promptValue = normalizeCustomStylePrompt(prompt)
  return normalized.find((item) => item.id === targetId)
    || (promptValue ? normalized.find((item) => getCustomStyleKey(item.label, item.prompt) === getCustomStyleKey(label, promptValue)) : null)
    || (labelKey ? normalized.find((item) => getCustomStyleLabelKey(item.label) === labelKey) : null)
    || null
}

export const resolveCommentaryStyleRequest = (style, customStylePrompt, customStyleOptions = []) => {
  const selectedCustomStyle = findCustomStyleOption(customStyleOptions, style)
  if (selectedCustomStyle) {
    return {
      style: selectedCustomStyle.label,
      customStylePrompt: normalizeCustomStylePrompt(customStylePrompt) || selectedCustomStyle.prompt,
      selectedCustomStyle,
      isCustomStyle: true,
    }
  }
  const normalizedStyle = String(style || '').trim() || 'hustle'
  return {
    style: normalizedStyle,
    customStylePrompt: normalizedStyle === 'custom' ? normalizeCustomStylePrompt(customStylePrompt) : '',
    selectedCustomStyle: null,
    isCustomStyle: normalizedStyle === 'custom',
  }
}
