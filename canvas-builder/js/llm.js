const PROVIDERS = {
  anthropic: {
    endpoint: 'https://api.anthropic.com/v1/messages',
    models: ['claude-sonnet-4-6', 'claude-haiku-4-5-20251001', 'claude-opus-4-6'],
  },
};

function getProvider(model) {
  for (const [name, config] of Object.entries(PROVIDERS)) {
    if (config.models.includes(model)) return { name, ...config };
  }
  // Default to anthropic for unknown claude-* models
  if (model.startsWith('claude-')) return { name: 'anthropic', ...PROVIDERS.anthropic };
  throw new Error(`Unsupported model: ${model}`);
}

async function callLLM({ model, apiKey, messages, systemPrompt, onDelta, onDone, onError }) {
  let provider;
  try {
    provider = getProvider(model);
  } catch (err) {
    onError(err.message);
    return;
  }

  const body = {
    model,
    max_tokens: 8096,
    system: systemPrompt,
    messages,
    stream: true,
  };

  let response;
  try {
    response = await fetch(provider.endpoint, {
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

  // Parse Anthropic SSE stream
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
      buffer = lines.pop(); // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') continue;

        let parsed;
        try {
          parsed = JSON.parse(data);
        } catch (_) {
          continue;
        }

        if (parsed.type === 'content_block_delta' && parsed.delta?.type === 'text_delta') {
          const delta = parsed.delta.text;
          fullText += delta;
          onDelta(delta, fullText);
        }

        if (parsed.type === 'message_stop') {
          onDone(fullText);
          return;
        }
      }
    }
    // Stream ended without message_stop
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
