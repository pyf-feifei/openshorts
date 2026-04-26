# Core i18n Language Switch Design

Date: 2026-04-26

## Goal

Add a lightweight English / Simplified Chinese language switch for the main OpenShorts dashboard experience without adding external i18n dependencies. The first release focuses on the core authenticated/product UI that users interact with while creating and editing clips.

## Scope

Included in this phase:

- A small React i18n provider with `language`, `setLanguage`, and `t` helpers.
- Persistent language preference stored in `localStorage`.
- Built-in `en` and `zh-CN` translation dictionaries.
- A visible language switcher in the core dashboard shell/settings area.
- Translation coverage for the primary product flow:
  - Sidebar/top navigation labels.
  - Settings section headings, helper text, API-key labels, and save/status actions.
  - Clip Generator entry points, processing state, logs, and Gemini-key modal.
  - Common clip actions: edit, subtitles, hook overlay, dubbing/translation, posting/scheduling.
  - Frequently used modal labels/buttons in Subtitle, Hook, Translate, and Schedule flows.

Deferred from this phase:

- Full marketing Landing page translation.
- Full AI Shorts long-form copy and generated-script workflow translation.
- Backend-generated AI output language control beyond the existing prompts.
- Changing video subtitle fonts or Docker image fonts for CJK rendering.

## Architecture

Create a local i18n module under `dashboard/src/i18n/`:

- `translations.js` exports nested dictionaries keyed by language code.
- `I18nProvider.jsx` owns language state, loads/saves it from `localStorage`, and exposes a `t(key, params?)` function through React context.
- `useI18n()` provides typed-ish ergonomic access for components.
- Missing keys fall back to English, then to the key string. This keeps the UI usable while translation coverage is expanded incrementally.

The provider wraps the app in `dashboard/src/main.jsx`, so any component can opt in by calling `useI18n()`.

## UI Behavior

The language switcher presents two options: `English` and `中文`. Selecting either updates the UI immediately and persists across refreshes. The switcher should be compact and fit the current dark dashboard style.

The default language is English unless `localStorage` contains a supported language code. Unsupported or malformed stored values fall back to English.

## Translation Strategy

Use stable translation keys such as `nav.clipGenerator`, `settings.title`, and `common.save`. Avoid deriving keys from English phrases. Group dictionaries by feature area:

- `common`
- `nav`
- `settings`
- `clipGenerator`
- `resultCard`
- `subtitleModal`
- `hookModal`
- `translateModal`
- `scheduleModal`
- `errors`

For dynamic labels, use simple interpolation: `t('processing.jobQueued', { jobId })`.

## Error Handling

If a translation key is missing:

1. Return the English translation if present.
2. Return the key string if no English translation exists.

If interpolation parameters are missing, leave the placeholder text unchanged rather than throwing. The UI should never crash because of missing translations.

## Testing

Add focused tests for the pure translation helper behavior:

- English is the default fallback.
- `zh-CN` returns Chinese values for known keys.
- Missing Chinese keys fall back to English.
- Unknown keys return the key string.
- Interpolation replaces provided placeholders.

Run `npm run build` to verify the translated React app compiles.

## Rollout

Implement in small passes:

1. Add i18n module and tests.
2. Wrap the React app and add the switcher.
3. Translate the dashboard shell and Settings.
4. Translate Clip Generator and common result actions/modals.
5. Build and manually verify language switching in Docker/frontend.
