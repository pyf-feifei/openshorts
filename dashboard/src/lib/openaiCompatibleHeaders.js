export function normalizeOpenAICompatibleBaseUrl(baseUrl = '') {
  return String(baseUrl || '').trim().replace(/\/+$/, '');
}

export function buildOpenAICompatibleConfig(config = {}) {
  return {
    apiKey: String(config.apiKey || '').trim(),
    baseUrl: normalizeOpenAICompatibleBaseUrl(config.baseUrl || ''),
    model: String(config.model || '').trim(),
  };
}

export function hasOpenAICompatibleAccess(config = {}) {
  const normalized = buildOpenAICompatibleConfig(config);
  return Boolean(normalized.apiKey && normalized.baseUrl && normalized.model);
}

export function buildOpenAICompatibleHeaders(config = {}, extraHeaders = {}) {
  const normalized = buildOpenAICompatibleConfig(config);
  const headers = { ...extraHeaders };
  if (normalized.apiKey) headers['X-OpenAI-Compatible-Key'] = normalized.apiKey;
  if (normalized.baseUrl) headers['X-OpenAI-Compatible-Base-URL'] = normalized.baseUrl;
  if (normalized.model) headers['X-OpenAI-Compatible-Model'] = normalized.model;
  return headers;
}
