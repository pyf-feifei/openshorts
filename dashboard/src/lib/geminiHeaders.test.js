import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import test from 'node:test';

import {
  buildGeminiConfig,
  buildGeminiHeaders,
  fingerprintGeminiKey,
  mergeGeminiEvents,
  normalizeGeminiBaseUrl,
} from './geminiHeaders.js';

const srcRoot = resolve(import.meta.dirname, '..');

function readSource(relativePath) {
  return readFileSync(resolve(srcRoot, relativePath), 'utf8');
}

test('normalizes Gemini base URL by trimming whitespace and trailing slashes', () => {
  assert.equal(
    normalizeGeminiBaseUrl('  https://gemini-proxy.example.com///  '),
    'https://gemini-proxy.example.com',
  );
});

test('builds Gemini headers with optional base URL and preserves extra headers', () => {
  assert.deepEqual(
    buildGeminiHeaders('AIza-test', 'https://gemini-proxy.example.com/', {
      'Content-Type': 'application/json',
    }),
    {
      'Content-Type': 'application/json',
      'X-Gemini-Key': 'AIza-test',
      'X-Gemini-Base-URL': 'https://gemini-proxy.example.com',
    },
  );
});

test('omits Gemini base URL header when no base URL is configured', () => {
  assert.deepEqual(buildGeminiHeaders('AIza-test', '', {}), {
    'X-Gemini-Key': 'AIza-test',
  });
});

test('fingerprints Gemini keys without exposing the full value', () => {
  assert.equal(fingerprintGeminiKey('AIzaSyBOMrVoq6wAwsfDN2nrbvtEMD_ffK0TfQY'), 'AIza...TfQY');
});

test('merges Gemini pool events into persisted key stats', () => {
  assert.deepEqual(
    mergeGeminiEvents(
      { 'AIza...one1': { successes: 1 } },
      [
        { fingerprint: 'AIza...one1', status: 'success' },
        { fingerprint: 'AIza...two2', status: 'cooldown', summary: 'Gemini rate limit exceeded' },
        { fingerprint: 'AIza...bad3', status: 'disabled', summary: 'Gemini key permission denied or invalid' },
      ],
    ),
    {
      'AIza...one1': { successes: 2, state: 'healthy' },
      'AIza...two2': { errors429: 1, state: 'cooling', lastError: 'Gemini rate limit exceeded' },
      'AIza...bad3': { errors403: 1, lastError: 'Gemini key permission denied or invalid' },
    },
  );
});

test('builds official Gemini key pool config without custom base URL', () => {
  const config = buildGeminiConfig({
    mode: 'official_pool',
    apiKey: 'legacy-key',
    keyPool: [' key-one ', '', 'key-two'],
    stats: { 'key-...-one': { successes: 2 } },
    baseUrl: 'https://proxy.example.com',
  });

  assert.deepEqual(config, {
    mode: 'official_pool',
    keys: ['key-one', 'key-two'],
    stats: { 'key-...-one': { successes: 2 } },
  });
});

test('buildGeminiHeaders sends official pool payload instead of single key headers', () => {
  const headers = buildGeminiHeaders({
    mode: 'official_pool',
    keyPool: ['key-one', 'key-two'],
    stats: {},
  });

  assert.ok(headers['X-Gemini-Pool']);
  assert.equal(headers['X-Gemini-Key'], undefined);
  assert.equal(headers['X-Gemini-Base-URL'], undefined);
});

test('App wires saved Gemini base URL into settings and all Gemini-backed tabs', () => {
  const app = readSource('App.jsx');
  assert.match(app, /onBaseUrlSet=\{setGeminiBaseUrl\}/);
  assert.match(app, /savedBaseUrl=\{geminiBaseUrl\}/);
  assert.match(app, /<SaaShortsTab[\s\S]*geminiBaseUrl=\{geminiBaseUrl\}/);
  assert.match(app, /<ThumbnailStudio[\s\S]*geminiBaseUrl=\{geminiBaseUrl\}/);
});

test('Gemini access mode is rendered inside the Gemini API key settings card with autosave copy', () => {
  const app = readSource('App.jsx');
  const keyInput = readSource('components/KeyInput.jsx');

  assert.match(keyInput, /function KeyInput\(\{[\s\S]*children/);
  assert.match(keyInput, /\{children\}/);
  assert.match(app, /<KeyInput[\s\S]*>\s*<GeminiAccessModeSettings/);
  assert.match(app, /自动保存/);
});

test('top API key warning uses pooled Gemini access state instead of only the legacy single key', () => {
  const app = readSource('App.jsx');

  assert.doesNotMatch(app, /\{!apiKey && \(/);
  assert.match(app, /\{!hasGeminiAccess && \(/);
});

test('Gemini-backed components use shared header builder instead of old inline Gemini key headers', () => {
  const saaShorts = readSource('components/SaaShortsTab.jsx');
  const thumbnailStudio = readSource('components/ThumbnailStudio.jsx');

  assert.doesNotMatch(saaShorts, /'X-Gemini-Key':\s*geminiApiKey/);
  assert.doesNotMatch(thumbnailStudio, /'X-Gemini-Key':\s*geminiApiKey/);
  assert.match(thumbnailStudio, /function ThumbnailStudio\(\{[\s\S]*geminiBaseUrl/);
});

test('App wires OpenAI-compatible commentary settings into CommentaryTab', () => {
  const app = readSource('App.jsx');

  assert.match(app, /openAICompatibleKey/);
  assert.match(app, /openAICompatibleBaseUrl/);
  assert.match(app, /openAICompatibleModel/);
  assert.match(app, /<CommentaryTab[\s\S]*openAICompatibleConfig=\{\{/);
});
