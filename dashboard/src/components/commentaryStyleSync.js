import {
  CUSTOM_STYLE_STORAGE_KEY,
  findCustomStyleOption,
  normalizeCustomStyleOptions,
} from './commentaryCustomStyles.js'

export function loadLocalCustomStyles() {
  if (typeof window === 'undefined') return []
  try {
    return normalizeCustomStyleOptions(JSON.parse(window.localStorage.getItem(CUSTOM_STYLE_STORAGE_KEY) || '[]'))
  } catch {
    return []
  }
}

export function saveLocalCustomStyles(items) {
  if (typeof window === 'undefined') return []
  const normalized = normalizeCustomStyleOptions(items)
  window.localStorage.setItem(CUSTOM_STYLE_STORAGE_KEY, JSON.stringify(normalized))
  window.dispatchEvent(new CustomEvent('openshorts:commentary-styles-updated', { detail: { styles: normalized } }))
  return normalized
}

export function mergeStyleIntoLocalStorage(style) {
  const normalizedStyle = normalizeCustomStyleOptions([style])[0]
  if (!normalizedStyle) return null
  const current = loadLocalCustomStyles()
  const existing = findCustomStyleOption(current, normalizedStyle.id)
    || findCustomStyleOption(current, normalizedStyle.label, normalizedStyle.prompt)
  const next = existing
    ? current.map((item) => (item.id === existing.id ? { ...normalizedStyle, id: existing.id } : item))
    : [...current, normalizedStyle]
  saveLocalCustomStyles(next)
  return existing ? { ...normalizedStyle, id: existing.id } : normalizedStyle
}
