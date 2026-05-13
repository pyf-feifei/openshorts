import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildOpenAICompatibleConfig,
  buildOpenAICompatibleHeaders,
  hasOpenAICompatibleAccess,
  normalizeOpenAICompatibleBaseUrl,
} from './openaiCompatibleHeaders.js';

test('normalizes OpenAI-compatible base URL by trimming whitespace and trailing slashes', () => {
  assert.equal(
    normalizeOpenAICompatibleBaseUrl('  https://provider.example.com/v1///  '),
    'https://provider.example.com/v1',
  );
});

test('builds OpenAI-compatible config from user settings', () => {
  assert.deepEqual(
    buildOpenAICompatibleConfig({
      apiKey: ' sk-test ',
      baseUrl: ' https://provider.example.com/v1/ ',
      model: ' vision-model ',
    }),
    {
      apiKey: 'sk-test',
      baseUrl: 'https://provider.example.com/v1',
      model: 'vision-model',
    },
  );
});

test('requires key, base URL, and model for OpenAI-compatible access', () => {
  assert.equal(hasOpenAICompatibleAccess({ apiKey: 'sk', baseUrl: 'https://provider.example.com/v1', model: 'vision' }), true);
  assert.equal(hasOpenAICompatibleAccess({ apiKey: 'sk', baseUrl: 'https://provider.example.com/v1', model: '' }), false);
});

test('builds OpenAI-compatible headers and preserves extra headers', () => {
  assert.deepEqual(
    buildOpenAICompatibleHeaders(
      { apiKey: 'sk-test', baseUrl: 'https://provider.example.com/v1/', model: 'vision-model' },
      { 'Content-Type': 'application/json' },
    ),
    {
      'Content-Type': 'application/json',
      'X-OpenAI-Compatible-Key': 'sk-test',
      'X-OpenAI-Compatible-Base-URL': 'https://provider.example.com/v1',
      'X-OpenAI-Compatible-Model': 'vision-model',
    },
  );
});
