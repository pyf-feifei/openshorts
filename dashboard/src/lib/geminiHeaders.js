export function normalizeGeminiBaseUrl(baseUrl = '') {
  return String(baseUrl || '').trim().replace(/\/+$/, '');
}

export function fingerprintGeminiKey(apiKey = '') {
  const value = String(apiKey || '').trim();
  if (value.length <= 8) return value;
  return `${value.slice(0, 4)}...${value.slice(-4)}`;
}

export function normalizeGeminiKeyPool(keys = []) {
  return Array.from(
    new Set(
      (keys || [])
        .map((key) => String(key || '').trim())
        .filter(Boolean),
    ),
  );
}

export function buildGeminiConfig(optionsOrKey, baseUrl = '', stats = {}) {
  if (typeof optionsOrKey === 'object' && optionsOrKey !== null) {
    const mode = optionsOrKey.mode || 'custom_proxy';
    if (mode === 'official_pool') {
      return {
        mode: 'official_pool',
        keys: normalizeGeminiKeyPool(optionsOrKey.keyPool || optionsOrKey.keys || []),
        stats: optionsOrKey.stats || {},
      };
    }
    return {
      mode: 'custom_proxy',
      apiKey: String(optionsOrKey.apiKey || '').trim(),
      baseUrl: normalizeGeminiBaseUrl(optionsOrKey.baseUrl || ''),
      stats: optionsOrKey.stats || {},
    };
  }
  return {
    mode: 'custom_proxy',
    apiKey: String(optionsOrKey || '').trim(),
    baseUrl: normalizeGeminiBaseUrl(baseUrl),
    stats,
  };
}

export function mergeGeminiEvents(currentStats, events) {
  if (!Array.isArray(events) || events.length === 0) return currentStats;
  const next = { ...currentStats };
  for (const event of events) {
    const fp = event.fingerprint;
    if (!fp) continue;
    const stat = { ...(next[fp] || {}) };
    if (event.status === 'success') {
      stat.successes = (stat.successes || 0) + 1;
      stat.state = 'healthy';
    } else if (event.status === 'cooldown') {
      stat.errors429 = (stat.errors429 || 0) + 1;
      stat.state = 'cooling';
      if (event.summary) stat.lastError = event.summary;
    } else if (event.status === 'disabled') {
      stat.errors403 = (stat.errors403 || 0) + 1;
      if (event.summary) stat.lastError = event.summary;
    } else if (event.status === 'exhausted') {
      stat.state = 'exhausted';
      if (event.summary) stat.lastError = event.summary;
    } else if (event.summary) {
      stat.lastError = event.summary;
    }
    next[fp] = stat;
  }
  return next;
}

export function buildGeminiHeaders(optionsOrKey, baseUrl = '', extraHeaders = {}) {
  const headers = { ...extraHeaders };
  const config = buildGeminiConfig(optionsOrKey, baseUrl);
  if (config.mode === 'official_pool') {
    headers['X-Gemini-Pool'] = JSON.stringify({
      mode: 'official_pool',
      keys: config.keys,
      stats: config.stats || {},
    });
    return headers;
  }

  headers['X-Gemini-Key'] = config.apiKey;
  const normalizedBaseUrl = normalizeGeminiBaseUrl(config.baseUrl);
  if (normalizedBaseUrl) {
    headers['X-Gemini-Base-URL'] = normalizedBaseUrl;
  }
  return headers;
}
