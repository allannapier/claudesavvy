const PROVIDERS = {
  anthropic: {
    models: ['claude-sonnet-4-6', 'claude-haiku-4-5-20251001', 'claude-opus-4-6'],
  },
  google: {
    models: [
      'gemini-3.1-pro-preview',
      'gemini-3-flash-preview',
      'gemini-3.1-flash-lite-preview',
    ],
  },
};

function getProviderName(model) {
  if (model.startsWith('gemini-')) return 'google';
  if (model.startsWith('claude-')) return 'anthropic';
  for (const [name, config] of Object.entries(PROVIDERS)) {
    if (config.models.includes(model)) return name;
  }
  throw new Error(`Unsupported model: ${model}`);
}

async function callLLM({ model, apiKey, messages, systemPrompt, onDelta, onDone, onError }) {
  let providerName;
  try {
    providerName = getProviderName(model);
  } catch (err) {
    onError(err.message);
    return;
  }

  if (providerName === 'google') {
    await callGemini({ model, apiKey, messages, systemPrompt, onDelta, onDone, onError });
  } else {
    await callAnthropic({ model, apiKey, messages, systemPrompt, onDelta, onDone, onError });
  }
}

// ── Anthropic ────────────────────────────────────────────────────────────────

async function callAnthropic({ model, apiKey, messages, systemPrompt, onDelta, onDone, onError }) {
  const body = {
    model,
    max_tokens: 8096,
    system: systemPrompt,
    messages,
    stream: true,
  };

  let response;
  try {
    response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify(body),
    });
  } catch (err) {
    onError(`Network error: ${err.message}`);
    return;
  }

  if (!response.ok) {
    let errorMsg = `API error ${response.status}`;
    try {
      const errData = await response.json();
      if (errData.error?.message) errorMsg = errData.error.message;
    } catch (_) {}
    if (response.status === 401) errorMsg = 'Invalid API key. Check your key and try again.';
    if (response.status === 429) errorMsg = 'Rate limited. Wait a moment and try again.';
    onError(errorMsg);
    return;
  }

  await readSSEStream(response, (parsed) => {
    if (parsed.type === 'content_block_delta' && parsed.delta?.type === 'text_delta') {
      return { delta: parsed.delta.text };
    }
    if (parsed.type === 'message_stop') return { done: true };
    return null;
  }, onDelta, onDone, onError);
}

// ── Google Gemini ─────────────────────────────────────────────────────────────

function toGeminiMessages(messages) {
  return messages.map((m) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }));
}

async function callGemini({ model, apiKey, messages, systemPrompt, onDelta, onDone, onError }) {
  const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`;

  const body = {
    system_instruction: { parts: [{ text: systemPrompt }] },
    contents: toGeminiMessages(messages),
    generationConfig: { maxOutputTokens: 8192 },
  };

  let response;
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    onError(`Network error: ${err.message}`);
    return;
  }

  if (!response.ok) {
    let errorMsg = `API error ${response.status}`;
    try {
      const errData = await response.json();
      if (errData.error?.message) errorMsg = errData.error.message;
    } catch (_) {}
    if (response.status === 400) errorMsg = 'Bad request — check your API key or model name.';
    if (response.status === 401 || response.status === 403) errorMsg = 'Invalid Google API key. Get one at aistudio.google.com.';
    if (response.status === 429) errorMsg = 'Rate limited. Wait a moment and try again.';
    onError(errorMsg);
    return;
  }

  await readSSEStream(response, (parsed) => {
    const text = parsed.candidates?.[0]?.content?.parts?.[0]?.text;
    if (text) return { delta: text };
    const finishReason = parsed.candidates?.[0]?.finishReason;
    if (finishReason && finishReason !== 'STOP') return null;
    if (finishReason === 'STOP') return { done: true };
    return null;
  }, onDelta, onDone, onError);
}

// ── Shared SSE reader ─────────────────────────────────────────────────────────

async function readSSEStream(response, parseEvent, onDelta, onDone, onError) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = '';
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') { onDone(fullText); return; }

        let parsed;
        try { parsed = JSON.parse(data); } catch (_) { continue; }

        const result = parseEvent(parsed);
        if (!result) continue;
        if (result.done) { onDone(fullText); return; }
        if (result.delta) {
          fullText += result.delta;
          onDelta(result.delta, fullText);
        }
      }
    }
    onDone(fullText);
  } catch (err) {
    onError(`Stream error: ${err.message}`);
  }
}

function extractHtml(text) {
  const match = text.match(/```html\s*([\s\S]*?)```/);
  if (match) return match[1].trim();
  const trimmed = text.trimStart();
  if (trimmed.startsWith('<!DOCTYPE') || trimmed.startsWith('<html')) return trimmed;
  return null;
}
