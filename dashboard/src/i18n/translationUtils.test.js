import test from 'node:test';
import assert from 'node:assert/strict';
import { createTranslator, normalizeLanguage, SUPPORTED_LANGUAGES } from './translationUtils.js';

const dictionaries = {
  en: {
    common: { save: 'Save', greeting: 'Hello {name}' },
    onlyEnglish: { label: 'English fallback' },
  },
  'zh-CN': {
    common: { save: '保存', greeting: '你好 {name}' },
  },
};

test('normalizes unsupported languages to English', () => {
  assert.equal(normalizeLanguage('zh-CN'), 'zh-CN');
  assert.equal(normalizeLanguage('fr'), 'en');
  assert.deepEqual(SUPPORTED_LANGUAGES, ['en', 'zh-CN']);
});

test('translates known keys for selected language', () => {
  const t = createTranslator(dictionaries, 'zh-CN');
  assert.equal(t('common.save'), '保存');
});

test('falls back to English for missing selected-language keys', () => {
  const t = createTranslator(dictionaries, 'zh-CN');
  assert.equal(t('onlyEnglish.label'), 'English fallback');
});

test('returns key for unknown keys', () => {
  const t = createTranslator(dictionaries, 'zh-CN');
  assert.equal(t('missing.key'), 'missing.key');
});

test('interpolates provided placeholders and keeps missing placeholders intact', () => {
  const t = createTranslator(dictionaries, 'zh-CN');
  assert.equal(t('common.greeting', { name: '小明' }), '你好 小明');
  assert.equal(t('common.greeting'), '你好 {name}');
});
