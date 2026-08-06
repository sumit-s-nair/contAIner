/**
 * contAIner renderer — SSE pipeline consumer matching the macOS-style UI.
 *
 * SSE events from /run:
 *  system1  running/done  → intent pill + confidence bar
 *  clarify  running/needed → inline question card
 *  mcp      running/done/skipped → doc stage card
 *  system2  running/done  → explanation + steps + code block
 *  error    done          → error card
 */
'use strict';

const BRIDGE_URL = 'http://localhost:5050';

let bridgeReady     = false;
let isProcessing    = false;
let abortController = null;

// ── DOM ───────────────────────────────────────────────────────────────────────
const chatHistory = document.getElementById('chat-history');
const userInput   = document.getElementById('user-input');
const sendBtn     = document.getElementById('send-btn');
const stopBtn     = document.getElementById('stop-btn');
const statusDot   = document.getElementById('status-dot');
const statusText  = document.getElementById('status-text');

// ── Title bar (macOS style — no ipcRenderer needed for these) ─────────────────
document.getElementById('btn-close').addEventListener('click', () => window.close?.());
document.getElementById('btn-min').addEventListener('click',   () => {/* handled by main */});
document.getElementById('btn-max').addEventListener('click',   () => {/* handled by main */});

// ── Bridge status from Electron main ─────────────────────────────────────────
if (window.electronAPI?.onBridgeStatus) {
  window.electronAPI.onBridgeStatus(({ ready, message }) => {
    setBridgeReady(ready, message || (ready ? 'connected' : 'offline'));
  });
}

// Fallback health poll (also works when loading renderer directly)
async function pollBridgeHealth() {
  try {
    const r = await fetch(`${BRIDGE_URL}/health`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) { setBridgeReady(true, 'connected'); return; }
  } catch { /* still loading */ }
  setTimeout(pollBridgeHealth, 2500);
}
pollBridgeHealth();

function setBridgeReady(ready, message) {
  bridgeReady = ready;
  statusDot.className = `status-dot ${ready ? 'ready' : 'error'}`;
  statusText.textContent = message || (ready ? 'connected' : 'offline');
  if (!isProcessing) {
    userInput.disabled = !ready;
    sendBtn.disabled   = !ready;
  }
}

// ── Auto-grow textarea ────────────────────────────────────────────────────────
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 130) + 'px';
});

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
});
sendBtn.addEventListener('click', handleSend);
stopBtn.addEventListener('click', () => {
  if (abortController) { abortController.abort(); setProcessing(false); }
});

// ── Empty state ───────────────────────────────────────────────────────────────
function renderEmptyState() {
  const examples = [
    'install requests for python',
    'update nodejs on linux',
    'remove docker from my system',
    'check if git is installed',
    'create a python virtualenv',
  ];
  const wrap = document.createElement('div');
  wrap.className = 'empty-state';
  wrap.innerHTML = `
    <div class="empty-ai-row">
      <div class="ai-avatar">AI</div>
      <div class="empty-text">Ready. Describe what you need.</div>
    </div>
    <div class="example-chips">
      ${examples.map(e => `<span class="chip">${e}</span>`).join('')}
    </div>`;
  chatHistory.appendChild(wrap);
  wrap.querySelectorAll('.chip').forEach(c => {
    c.addEventListener('click', () => {
      userInput.value = c.textContent;
      userInput.dispatchEvent(new Event('input'));
      handleSend();
    });
  });
}
renderEmptyState();

// ── Send ──────────────────────────────────────────────────────────────────────
async function handleSend() {
  const text = userInput.value.trim();
  if (!text || isProcessing || !bridgeReady) return;

  // remove empty state on first send
  const empty = chatHistory.querySelector('.empty-state');
  if (empty) empty.remove();

  userInput.value = '';
  userInput.style.height = 'auto';
  setProcessing(true);

  appendUserBubble(text);

  const { thinkingPanel, stageContainer, thinkingLabel, aiContent } = createAIRow();

  await runPipeline(text, null, stageContainer, thinkingPanel, thinkingLabel, aiContent);

  setProcessing(false);
}

// ── Pipeline ──────────────────────────────────────────────────────────────────
async function runPipeline(prompt, clarifyAnswer, stageContainer, thinkingPanel, thinkingLabel, aiContent) {
  if (abortController) abortController.abort();
  abortController = new AbortController();

  let resp;
  try {
    resp = await fetch(`${BRIDGE_URL}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: abortController.signal,
      body: JSON.stringify({
        prompt,
        os_hint: 'linux',
        shell_type: 'bash',
        ...(clarifyAnswer ? { clarify_answer: clarifyAnswer } : {}),
      }),
    });
  } catch (err) {
    if (err.name !== 'AbortError') {
      appendToAI(aiContent, renderErrorCard(`Could not reach bridge: ${err.message}`));
    }
    collapseThinking(thinkingPanel, thinkingLabel, false);
    return;
  }

  const reader  = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop();
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith('data:')) continue;
      let event;
      try { event = JSON.parse(line.slice(5).trim()); } catch { continue; }
      handleEvent(event, stageContainer, thinkingPanel, thinkingLabel, aiContent, prompt);
    }
  }
}

// ── Event dispatch ────────────────────────────────────────────────────────────
function handleEvent(event, stageContainer, thinkingPanel, thinkingLabel, aiContent, originalPrompt) {
  const { stage, status } = event;

  if (stage === 'system1' && status === 'running') {
    upsertStage(stageContainer, 'system1', 'running', 'Classifying intent…', '');
  }
  else if (stage === 'system1' && status === 'done') {
    const { intent, confidence, entities } = event;
    const pct = Math.round((confidence || 0) * 100);
    const entityStr = Object.entries(entities || {})
      .map(([k, v]) => `<span class="tag">${k}: ${v}</span>`).join('');
    // Store for later update when grok extracts better entities
    stageContainer.dataset.s1Intent  = intent;
    stageContainer.dataset.s1Pct     = pct;
    stageContainer.dataset.s1Entities = JSON.stringify(entities || {});
    upsertStage(stageContainer, 'system1', 'ok',
      `Intent: <strong>${intent}</strong>`,
      `<div class="conf-badge">confidence <strong>${pct}%</strong></div>
       ${entityStr ? `<div style="margin-top:4px">${entityStr}</div>` : '<div style="margin-top:4px;color:var(--muted);font-size:12px">no entities — awaiting extraction</div>'}`
    );
  }
  else if (stage === 'clarify' && status === 'running') {
    upsertStage(stageContainer, 'clarify', 'running', 'Generating clarifying question…', '');
  }
  else if (stage === 'clarify' && status === 'needed') {
    upsertStage(stageContainer, 'clarify', 'warn', 'Clarification needed', '');
    collapseThinking(thinkingPanel, thinkingLabel, false);
    appendToAI(aiContent, renderClarifyCard(event.question, originalPrompt, stageContainer, thinkingPanel, thinkingLabel, aiContent));
  }
  else if (stage === 'mcp' && status === 'running') {
    upsertStage(stageContainer, 'mcp', 'running', 'Fetching documentation…',
      `<span class="tag">${event.tool}</span><span class="tag">${event.operation}</span>`);
  }
  else if (stage === 'mcp' && status === 'done') {
    const { tool, has_docs, doc_chunk } = event;
    const syntax = doc_chunk?.command_syntax || '';
    upsertStage(stageContainer, 'mcp', 'ok',
      `Docs: <strong>${tool}</strong>`,
      has_docs
        ? `<span class="tag">syntax</span> ${escHtml(syntax.slice(0, 90))}${syntax.length > 90 ? '…' : ''}`
        : '<span style="color:var(--muted)">no docs found</span>');
  }
  else if (stage === 'mcp' && status === 'skipped') {
    upsertStage(stageContainer, 'mcp', 'warn', 'MCP skipped', escHtml(event.reason || ''));
  }
  else if (stage === 'system2' && status === 'running') {
    upsertStage(stageContainer, 'system2', 'running', 'Generating command…', '');
  }
  else if (stage === 'system2' && status === 'done') {
    upsertStage(stageContainer, 'system2', 'ok', 'Command generated', '');
    collapseThinking(thinkingPanel, thinkingLabel, true);

    // Merge grok entities into system1 card if system1 found none
    const grokEntities = event.grok_entities || {};
    const s1Entities   = JSON.parse(stageContainer.dataset.s1Entities || '{}');
    const mergedEntities = Object.keys(s1Entities).length ? s1Entities : grokEntities;
    if (Object.keys(grokEntities).length && !Object.keys(s1Entities).length) {
      const intent = stageContainer.dataset.s1Intent || '';
      const pct    = stageContainer.dataset.s1Pct    || '';
      const entityStr = Object.entries(mergedEntities)
        .map(([k, v]) => `<span class="tag">${k}: ${v}</span>`).join('');
      upsertStage(stageContainer, 'system1', 'ok',
        `Intent: <strong>${intent}</strong>`,
        `<div class="conf-badge">confidence <strong>${pct}%</strong></div>
         <div style="margin-top:4px">${entityStr}</div>`
      );
    }

    const { explanation, steps, command, shell } = event;
    if (explanation || steps?.length) {
      appendToAI(aiContent, renderProse(explanation, steps));
    }
    if (command) {
      appendToAI(aiContent, renderCodeBlock(command, shell || 'bash'));
    }
  }
  else if (stage === 'error') {
    collapseThinking(thinkingPanel, thinkingLabel, false);
    appendToAI(aiContent, renderErrorCard(event.message || 'Unknown error'));
  }
}

// ── Stage card ────────────────────────────────────────────────────────────────
function upsertStage(container, id, state, label, detail) {
  let card = container.querySelector(`[data-stage="${id}"]`);
  if (!card) {
    card = document.createElement('div');
    card.dataset.stage = id;
    container.appendChild(card);
    scrollBottom();
  }
  const iconHTML =
    state === 'running' ? '<div class="stage-icon spinning"></div>' :
    state === 'ok'      ? '<div class="stage-icon ok">✓</div>' :
    state === 'warn'    ? '<div class="stage-icon warn">!</div>' :
                          '<div class="stage-icon warn">✕</div>';
  card.className = `stage-card ${state}`;
  card.innerHTML = `${iconHTML}<div class="stage-body">
    <div class="stage-label">${label}</div>
    ${detail ? `<div class="stage-detail">${detail}</div>` : ''}
  </div>`;
}

// ── Thinking collapse ─────────────────────────────────────────────────────────
function collapseThinking(panel, label, success) {
  if (!panel) return;
  label.textContent = success ? 'Done' : 'Stopped';
  label.style.color = success ? 'var(--success)' : 'var(--muted)';
}

// ── Prose card (explanation + numbered steps) ─────────────────────────────────
function renderProse(explanation, steps) {
  const div = document.createElement('div');
  div.className = 'ai-prose';
  let html = '';
  if (explanation) html += `<div class="explanation">${escHtml(explanation)}</div>`;
  if (steps && steps.length) {
    html += '<div class="ai-steps">';
    steps.forEach((s, i) => {
      const num = String(i + 1).padStart(2, '0');
      html += `<div class="ai-step"><span class="ai-step-num">${num}</span><span>${escHtml(s)}</span></div>`;
    });
    html += '</div>';
  }
  div.innerHTML = html;
  return div;
}

// ── Code block ────────────────────────────────────────────────────────────────
function renderCodeBlock(command, shell) {
  const div = document.createElement('div');
  div.className = 'code-block';
  div.innerHTML = `
    <div class="code-header">
      <span class="code-lang">${escHtml(shell)}</span>
      <button class="copy-btn">copy</button>
    </div>
    <pre class="code-body">${escHtml(command)}</pre>`;
  div.querySelector('.copy-btn').addEventListener('click', (e) => {
    navigator.clipboard.writeText(command).then(() => {
      const btn = e.target;
      btn.textContent = 'copied';
      setTimeout(() => { btn.textContent = 'copy'; }, 1500);
    });
  });
  return div;
}

// ── Clarify card ──────────────────────────────────────────────────────────────
function renderClarifyCard(question, originalPrompt, stageContainer, thinkingPanel, thinkingLabel, aiContent) {
  const card = document.createElement('div');
  card.className = 'clarify-card';
  card.innerHTML = `
    <div class="clarify-q">${escHtml(question)}</div>
    <div class="clarify-row">
      <input class="clarify-input" type="text" placeholder="Type your answer…" autocomplete="off"/>
      <button class="clarify-btn">Send</button>
    </div>`;

  const input = card.querySelector('.clarify-input');
  const btn   = card.querySelector('.clarify-btn');

  const doSubmit = async () => {
    const answer = input.value.trim();
    if (!answer) return;
    card.innerHTML = `<div style="color:var(--muted);font-size:13px">"${escHtml(answer)}"</div>`;
    appendUserBubble(answer);
    setProcessing(true);
    await runPipeline(originalPrompt, answer, stageContainer, thinkingPanel, thinkingLabel, aiContent);
    setProcessing(false);
  };

  btn.addEventListener('click', doSubmit);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') doSubmit(); });
  setTimeout(() => input.focus(), 50);
  return card;
}

function renderErrorCard(msg) {
  const d = document.createElement('div');
  d.className = 'error-card';
  d.textContent = msg;
  return d;
}

// ── DOM helpers ───────────────────────────────────────────────────────────────
function appendUserBubble(text) {
  const row = document.createElement('div');
  row.className = 'msg-row user-row';
  const bubble = document.createElement('div');
  bubble.className = 'bubble-user';
  bubble.textContent = text;
  row.appendChild(bubble);
  chatHistory.appendChild(row);
  scrollBottom();
}

function createAIRow() {
  const row = document.createElement('div');
  row.className = 'msg-row ai-row';

  const avatar = document.createElement('div');
  avatar.className = 'ai-avatar';
  avatar.textContent = 'AI';

  const aiContent = document.createElement('div');
  aiContent.className = 'ai-content';

  // Thinking panel inside ai-content
  const thinkingPanel = document.createElement('div');
  thinkingPanel.className = 'thinking-panel';

  const thinkingLabel = document.createElement('div');
  thinkingLabel.className = 'thinking-label';
  thinkingLabel.textContent = 'Thinking';

  const stageContainer = document.createElement('div');

  thinkingPanel.appendChild(thinkingLabel);
  thinkingPanel.appendChild(stageContainer);
  aiContent.appendChild(thinkingPanel);

  row.appendChild(avatar);
  row.appendChild(aiContent);
  chatHistory.appendChild(row);
  scrollBottom();

  return { thinkingPanel, stageContainer, thinkingLabel, aiContent };
}

function appendToAI(aiContent, el) {
  aiContent.appendChild(el);
  scrollBottom();
}

function scrollBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function setProcessing(active) {
  isProcessing = active;
  if (active) {
    sendBtn.style.display = 'none';
    stopBtn.style.display = 'flex';
    userInput.disabled = true;
  } else {
    sendBtn.style.display = 'flex';
    stopBtn.style.display = 'none';
    userInput.disabled = !bridgeReady;
    sendBtn.disabled   = !bridgeReady;
    abortController    = null;
  }
}

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
