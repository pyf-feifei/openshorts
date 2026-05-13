import React from 'react';
import { Languages } from 'lucide-react';
import { useI18n } from './I18nProvider';

const OPTIONS = [
  { code: 'en', label: 'English' },
  { code: 'zh-CN', label: '中文' },
];

export default function LanguageSwitcher({ compact = false }) {
  const { language, setLanguage, t } = useI18n();

  return (
    <div className={`flex ${compact ? 'items-center gap-1' : 'items-center justify-between gap-3'}`}>
      {!compact && (
        <div className="flex items-center gap-2 text-sm text-zinc-300">
          <Languages size={16} className="text-primary" />
          <span>{t('common.language')}</span>
        </div>
      )}
      <div className="inline-flex rounded-xl border border-white/10 bg-black/20 p-1">
        {OPTIONS.map((option) => (
          <button
            key={option.code}
            type="button"
            onClick={() => setLanguage(option.code)}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-colors ${
              language === option.code
                ? 'bg-primary text-white shadow-lg shadow-primary/20'
                : 'text-zinc-400 hover:text-white hover:bg-white/5'
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
