function canvasApp() {
  return {
    // State
    apiKey: '',
    model: 'claude-sonnet-4-6',
    models: ['claude-sonnet-4-6', 'claude-haiku-4-5-20251001', 'claude-opus-4-6'],
    messages: [],
    history: null,
    isLoading: false,
    promptText: '',
    showKeyModal: false,
    keyInput: '',
    promptHistory: [], // [{role, text, status}] for display
    errorMessage: '',
    streamBuffer: '',

    init() {
      this.history = new HistoryStack();
      this.apiKey = sessionStorage.getItem('canvas_builder_key') || '';
      this.keyInput = this.apiKey;

      // Show placeholder in iframe on init
      const iframe = document.getElementById('canvas');
      showPlaceholder(iframe);

      // Listen for chip clicks from placeholder iframe
      window.addEventListener('message', (e) => {
        if (e.data?.type === 'chip') {
          this.promptText = e.data.text;
          this.$nextTick(() => this.submitPrompt());
        }
      });

      // Keyboard shortcuts
      window.addEventListener('keydown', (e) => {
        const mod = e.metaKey || e.ctrlKey;
        if (mod && e.key === 'z' && !e.shiftKey) { e.preventDefault(); this.undo(); }
        if (mod && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) { e.preventDefault(); this.redo(); }
      });
    },

    get canUndo() { return this.history?.canUndo() ?? false; },
    get canRedo() { return this.history?.canRedo() ?? false; },

    buildSystemPrompt() {
      const currentHtml = getHtml(document.getElementById('canvas'));
      const hasContent = currentHtml && !currentHtml.includes('Type a prompt to start building');
      const currentSection = hasContent
        ? `\n\nCURRENT HTML STATE (build on top of this unless asked to start fresh):\n${currentHtml}`
        : '\n\nCURRENT HTML STATE: (empty canvas — start fresh)';

      return `You are an expert frontend UI builder. Your job is to generate complete, self-contained HTML documents that render immediately in a browser.

Rules you must follow:
1. Always respond with EXACTLY ONE fenced code block: \`\`\`html ... \`\`\`
2. The code block must contain a complete <!DOCTYPE html> document
3. Use only CDN-delivered libraries (Tailwind CSS via https://cdn.tailwindcss.com, Alpine.js, Chart.js, etc.) — no npm, no build tools
4. When the user's prompt says "add", "change", or "update", build on top of the CURRENT HTML STATE shown below
5. When the user's prompt says "make", "build", or "create" with no reference to existing content, you may start fresh
6. Use modern, clean design: good typography, appropriate whitespace, accessible contrast ratios
7. All functionality must work inline (vanilla JS or Alpine.js for interactivity)
8. Never include explanatory text or prose outside the code block
9. Never use document.cookie, fetch(), XMLHttpRequest, or external API calls in generated code
10. If unsure what the user wants, make a reasonable creative decision and build it — do not ask clarifying questions${currentSection}`;
    },

    async submitPrompt() {
      const prompt = this.promptText.trim();
      if (!prompt || this.isLoading) return;

      if (!this.apiKey) {
        this.showKeyModal = true;
        return;
      }

      this.isLoading = true;
      this.errorMessage = '';
      this.promptText = '';
      this.streamBuffer = '';

      // Add to display history
      const userEntry = { role: 'user', text: prompt, status: 'sent' };
      this.promptHistory.push(userEntry);

      // Add to LLM messages
      this.messages.push({ role: 'user', content: prompt });

      // Placeholder assistant entry
      const assistantEntry = { role: 'assistant', text: '', status: 'generating' };
      this.promptHistory.push(assistantEntry);

      this.$nextTick(() => {
        const hist = document.getElementById('prompt-history');
        if (hist) hist.scrollTop = hist.scrollHeight;
      });

      await callLLM({
        model: this.model,
        apiKey: this.apiKey,
        messages: this.messages,
        systemPrompt: this.buildSystemPrompt(),

        onDelta: (delta, full) => {
          this.streamBuffer = full;
          assistantEntry.text = `Generating... (${full.length} chars)`;

          // Render as soon as closing ``` is found
          if (full.includes('```html') && full.includes('```', full.indexOf('```html') + 7)) {
            const html = extractHtml(full);
            if (html) {
              const iframe = document.getElementById('canvas');
              renderHtml(iframe, html);
            }
          }
        },

        onDone: (full) => {
          const html = extractHtml(full);
          if (html) {
            const iframe = document.getElementById('canvas');
            renderHtml(iframe, html);
            this.history.push(html);
            assistantEntry.text = `Generated (${html.length} chars)`;
            assistantEntry.status = 'done';
            this.messages.push({ role: 'assistant', content: full });
          } else {
            assistantEntry.text = 'No HTML found in response';
            assistantEntry.status = 'error';
            // Still add to messages so LLM has context
            this.messages.push({ role: 'assistant', content: full });
          }
          this.isLoading = false;
          this.$nextTick(() => {
            const hist = document.getElementById('prompt-history');
            if (hist) hist.scrollTop = hist.scrollHeight;
          });
        },

        onError: (msg) => {
          assistantEntry.text = msg;
          assistantEntry.status = 'error';
          this.errorMessage = msg;
          this.isLoading = false;
          // Remove the failed user message from LLM history so user can retry
          this.messages.pop();
        },
      });
    },

    undo() {
      if (!this.canUndo) return;
      const html = this.history.undo();
      if (html) renderHtml(document.getElementById('canvas'), html);
    },

    redo() {
      if (!this.canRedo) return;
      const html = this.history.redo();
      if (html) renderHtml(document.getElementById('canvas'), html);
    },

    copyHtml() {
      const html = getHtml(document.getElementById('canvas'));
      if (!html) return;
      navigator.clipboard.writeText(html).then(() => {
        this.showToast('HTML copied!');
      });
    },

    exportHtml() {
      const html = getHtml(document.getElementById('canvas'));
      if (!html) return;
      const blob = new Blob([html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'canvas-export.html';
      a.click();
      URL.revokeObjectURL(url);
    },

    newSession() {
      if (!confirm('Start a new session? This will clear the canvas and prompt history.')) return;
      this.messages = [];
      this.promptHistory = [];
      this.history.clear();
      this.errorMessage = '';
      showPlaceholder(document.getElementById('canvas'));
    },

    saveKey() {
      const key = this.keyInput.trim();
      if (!key) return;
      this.apiKey = key;
      sessionStorage.setItem('canvas_builder_key', key);
      this.showKeyModal = false;
    },

    clearKey() {
      this.apiKey = '';
      this.keyInput = '';
      sessionStorage.removeItem('canvas_builder_key');
      this.showKeyModal = false;
    },

    showToast(msg) {
      // Simple toast via a transient element
      const toast = document.createElement('div');
      toast.textContent = msg;
      toast.style.cssText = 'position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:#333;color:#fff;padding:8px 16px;border-radius:8px;font-size:13px;z-index:9999;pointer-events:none;';
      document.body.appendChild(toast);
      setTimeout(() => toast.remove(), 2000);
    },

    handleKeydown(e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        this.submitPrompt();
      }
    },
  };
}
