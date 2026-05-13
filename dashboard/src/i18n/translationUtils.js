export const SUPPORTED_LANGUAGES = ['en', 'zh-CN'];
export const DEFAULT_LANGUAGE = 'en';

export function normalizeLanguage(language) {
  return SUPPORTED_LANGUAGES.includes(language) ? language : DEFAULT_LANGUAGE;
}

function getNestedValue(source, key) {
  return key.split('.').reduce((current, part) => {
    if (!current || typeof current !== 'object') return undefined;
    return current[part];
  }, source);
}

function interpolate(template, params = {}) {
  if (typeof template !== 'string') return template;

  return template.replace(/\{(\w+)\}/g, (match, name) => (
    Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : match
  ));
}

export function createTranslator(dictionaries, language) {
  const normalizedLanguage = normalizeLanguage(language);

  return (key, params) => {
    const localized = getNestedValue(dictionaries[normalizedLanguage], key);
    const fallback = getNestedValue(dictionaries[DEFAULT_LANGUAGE], key);
    const value = localized ?? fallback ?? key;

    return interpolate(value, params);
  };
}
