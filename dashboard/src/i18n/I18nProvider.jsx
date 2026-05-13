import React, { createContext, useContext, useMemo, useState } from 'react';
import { createTranslator, DEFAULT_LANGUAGE, normalizeLanguage } from './translationUtils';
import { translations } from './translations';

const LANGUAGE_STORAGE_KEY = 'openshorts_language';

const I18nContext = createContext(null);

function getInitialLanguage() {
  try {
    return normalizeLanguage(localStorage.getItem(LANGUAGE_STORAGE_KEY));
  } catch {
    return DEFAULT_LANGUAGE;
  }
}

export function I18nProvider({ children }) {
  const [language, setLanguageState] = useState(getInitialLanguage);

  const setLanguage = (nextLanguage) => {
    const normalized = normalizeLanguage(nextLanguage);
    setLanguageState(normalized);
    try {
      localStorage.setItem(LANGUAGE_STORAGE_KEY, normalized);
    } catch {
      // Ignore storage failures; the in-memory language still updates.
    }
  };

  const value = useMemo(() => ({
    language,
    setLanguage,
    t: createTranslator(translations, language),
  }), [language]);

  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used inside I18nProvider');
  }
  return context;
}
