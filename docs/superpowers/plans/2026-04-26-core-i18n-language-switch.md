# Core i18n Language Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight English / Simplified Chinese language switch to the core OpenShorts dashboard UI.

**Architecture:** Create a local React i18n provider plus pure translation helper. Store language in `localStorage`, expose `t()` via context, and migrate core UI strings to stable translation keys while leaving long marketing/AI Shorts copy for later.

**Tech Stack:** React 18, Vite 4, Node built-in test runner, local JavaScript modules, no new runtime dependencies.

---

## File Structure

- Create `dashboard/src/i18n/translationUtils.js`: pure helpers for supported language validation, nested key lookup, English fallback, and placeholder interpolation.
- Create `dashboard/src/i18n/translations.js`: English and Simplified Chinese dictionaries for core UI areas.
- Create `dashboard/src/i18n/I18nProvider.jsx`: React context provider and `useI18n()` hook.
- Create `dashboard/src/i18n/LanguageSwitcher.jsx`: compact English / 中文 toggle.
- Create `dashboard/src/i18n/translationUtils.test.js`: Node test-runner tests for helper behavior.
- Modify `dashboard/package.json`: add `test` script using `node --test`.
- Modify `dashboard/src/main.jsx`: wrap app with `I18nProvider`.
- Modify `dashboard/src/App.jsx`: add switcher and translate core dashboard/settings/clip-generator text.
- Modify core modal/components: `ResultCard.jsx`, `SubtitleModal.jsx`, `HookModal.jsx`, `TranslateModal.jsx`, `ScheduleWeekModal.jsx`, `MediaInput.jsx`, `KeyInput.jsx`, `ProcessingAnimation.jsx` where practical for first-phase coverage.

### Task 1: Translation Helper Tests

**Files:**
- Create: `dashboard/src/i18n/translationUtils.test.js`
- Create: `dashboard/src/i18n/translationUtils.js`
- Modify: `dashboard/package.json`

- [ ] **Step 1: Add the failing helper test**

```js
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
```

- [ ] **Step 2: Add npm test script**

```json
"test": "node --test src/**/*.test.js"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd dashboard && npm test`

Expected: FAIL because `translationUtils.js` does not export the required helpers yet.

- [ ] **Step 4: Implement helper minimally**

```js
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
  return template.replace(/\{(\w+)\}/g, (match, name) => {
    return Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : match;
  });
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd dashboard && npm test`

Expected: PASS for all translation helper tests.

### Task 2: Provider, Dictionaries, and Switcher

**Files:**
- Create: `dashboard/src/i18n/translations.js`
- Create: `dashboard/src/i18n/I18nProvider.jsx`
- Create: `dashboard/src/i18n/LanguageSwitcher.jsx`
- Modify: `dashboard/src/main.jsx`

- [ ] **Step 1: Create dictionaries with core keys**

Add `translations.js` exporting `translations` with `en` and `zh-CN` sections for `common`, `nav`, `settings`, `clipGenerator`, `resultCard`, `subtitleModal`, `hookModal`, `translateModal`, `scheduleModal`, `mediaInput`, `keyInput`, `processing`, and `errors`.

- [ ] **Step 2: Create provider**

Create `I18nProvider.jsx` that reads `openshorts_language` from `localStorage`, normalizes it, persists changes, memoizes `t`, and exports `useI18n()`.

- [ ] **Step 3: Create switcher**

Create `LanguageSwitcher.jsx` that calls `useI18n()` and renders two buttons: `English` and `中文`.

- [ ] **Step 4: Wrap the app**

Modify `main.jsx` so `<App />` is wrapped in `<I18nProvider>`.

- [ ] **Step 5: Verify tests still pass**

Run: `cd dashboard && npm test`

Expected: PASS.

### Task 3: Translate Core Shell and Settings

**Files:**
- Modify: `dashboard/src/App.jsx`

- [ ] **Step 1: Import i18n helpers**

Import `useI18n` and `LanguageSwitcher`.

- [ ] **Step 2: Translate app shell text**

Replace hard-coded nav/tab labels and major headings with `t('nav.*')`, `t('settings.*')`, and `t('clipGenerator.*')`.

- [ ] **Step 3: Add language switcher**

Place `<LanguageSwitcher />` in Settings and/or the sidebar header so users can switch language from the core UI.

- [ ] **Step 4: Build check**

Run: `cd dashboard && npm run build`

Expected: Vite build succeeds.

### Task 4: Translate Core Components and Modals

**Files:**
- Modify: `dashboard/src/components/MediaInput.jsx`
- Modify: `dashboard/src/components/KeyInput.jsx`
- Modify: `dashboard/src/components/ProcessingAnimation.jsx`
- Modify: `dashboard/src/components/ResultCard.jsx`
- Modify: `dashboard/src/components/SubtitleModal.jsx`
- Modify: `dashboard/src/components/HookModal.jsx`
- Modify: `dashboard/src/components/TranslateModal.jsx`
- Modify: `dashboard/src/components/ScheduleWeekModal.jsx`

- [ ] **Step 1: Import `useI18n` in each touched component**

Add `const { t } = useI18n();` inside each component function.

- [ ] **Step 2: Replace first-phase user-facing labels**

Translate primary buttons, modal titles, field labels, helper text, and common alerts with dictionary keys.

- [ ] **Step 3: Keep API/data values unchanged**

Do not translate enum values sent to backend, language codes, storage keys, filenames, or API request payload field names.

- [ ] **Step 4: Run tests and build**

Run:

```bash
cd dashboard
npm test
npm run build
```

Expected: tests pass and build succeeds.

### Task 5: Docker Validation

**Files:**
- No source changes expected.

- [ ] **Step 1: Confirm running services**

Run: `docker compose ps`

Expected: `openshorts-frontend`, `openshorts-backend`, and `openshorts-renderer` are Up.

- [ ] **Step 2: Restart frontend if necessary**

Run: `docker compose restart frontend`

Expected: frontend restarts cleanly.

- [ ] **Step 3: Verify page and backend proxy**

Run:

```bash
curl.exe -I --max-time 15 http://localhost:5175/
curl.exe -i --max-time 15 http://localhost:5175/api/status/test-not-exist
```

Expected: first command returns `200 OK`; second returns backend JSON `{"detail":"Job not found"}` through the Vite proxy.

---

## Self-Review

Spec coverage: The plan implements the provider, persistence, English/Chinese dictionaries, switcher, fallback behavior, first-phase core UI translation, tests, build verification, and Docker validation described in the design.

Placeholder scan: The plan has no unresolved implementation placeholders; deferred work is explicitly out of scope.

Type consistency: Helper names are consistent across tests and implementation: `SUPPORTED_LANGUAGES`, `normalizeLanguage`, and `createTranslator`.
