/**
 * contAIner renderer — agent chat logic with SSE pipeline streaming.
 *
 * Flow per message:
 *  1. POST /run  → SSE stream of stage events
 *  2. Each event updates the live "thinking" panel
 *  3. clarify events pause the stream and ask the user a question
 *  4. On clarify answer: POST /run again with clarify_answer
 *  5. system2 done: render the command result card
 */

'use strict';

const BRIDGE_URL = 'http://localhost:5050';

// ── State ─────────────────────────────────────────────────────────────────────
let bridgeReady   = false;
let isProcessing  = false;

// Pending clarification state
let pendingClarify = null; // { originalPrompt, question }

// ── DOM refs ──────────────────────────────────────────────────────────────────
const chatHistory = document.getElementById('chat-history');
const userInput   = document.getElementById('user-input');
const sendBtn     = document.getElementById('send-btn');
const statusDot   = document.getElementById('status-dot');
const statusText  = document.getElementById('status-text');

// ── Title bar controls ────────────────────────────────────────────────────────
document.getElementById('btn-close').addEventListener('click', () => window.close());
document.getElementById('btn-min').addEventListener('click',   () => window.electronAPI?.minimise?.());
document.getElementById('btn-max').addEventListener('click',   () => window.electronAPI?.maximise?.());

// ── Bridge status from main process ──────────────────────────────────────────
if (window.electronAPI?.onBridgeStatus) {
  window.electronAPI.onBridgeStatus(({ ready, message }) => {
    setBridgeReady(ready, message || (ready ? 'Connected' : 'Offline'));
  });
}

// Poll bridge health directly as a fallback (also handles hot-reload in dev)
async function pollBridgeHealth() {
  try {
    const r = await fetch(`${BRIDGE_URL}/health`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) {
      setBridgeReady(true, 'Connected');
      return;
    }
  } catch { /* still loading */ }
  setTimeout(pollBridgeHealth, 2500);
}
pollBridgeHealth();

function setBridgeReady(ready, message) {
  bridgeReady = ready;
  statusDot.className = `status-dot ${ready ? 'ready' : 'error'}`;
  statusText.textContent = message || (ready ? 'Connected' : 'Offline');
  userInput.disabled = !ready;
  sendBtn.disabled   = !ready;
}

// ── Auto-resize textarea ──────────────────────────────────────────────────────
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 140) + 'px';
});

// ── Send on Enter (shift+enter = newline) ─────────────────────────────────────
userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
});
sendBtn.addEventListener('click', handleSend);

// ── Initial empty state ───────────────────────────────────────────────────────
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
    <div class="empty-icon">🤖</div>
    <div class="empty-title">contAIner</div>
    <div class="empty-sub">Describe what you need for your dev environment.<br>I'll classify the intent and generate the right command.</div>
    <div class="empty-chips">
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

// ── Main send handler ────────────────────────────────────────────────────────
async function handleSend() {
  const text = userInput.value.trim();
  if (!text || isProcessing || !bridgeReady) return;

  // Clear empty state on first real send
  const emptyState = chatHistory.querySelector('.empty-state');
  if (emptyState) emptyState.remove();

  userInput.value = '';
  userInput.style.height = 'auto';

  setProcessing(true);
  pendingClarify = null;

  // Render user bubble
  appendUserBubble(text);

  // Create thinking panel and result slot
  const { msgRow, stageContainer, appendResult } = createAssistantRow();

  await runPipeline(text, null, stageContainer, appendResult);

  setProcessing(false);
}

// ── Pipeline runner ───────────────────────────────────────────────────────────
async function runPipeline(prompt, clarifyAnswer, stageContainer, appendResult) {
  clearStageContainer(stageContainer);

  let resp;
  try {
    resp = await fetch(`${BRIDGE_URL}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        os_hint: 'linux',
        shell_type: 'bash',
        ...(clarifyAnswer ? { clarify_answer: clarifyAnswer } : {}),
      }),
    });
  } catch (err) {
    appendResult(renderErrorCard(`Could not reach bridge server: ${err.message}`));
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE lines are delimited by \n\n
    const parts = buffer.split('\n\n');
    buffer = parts.pop(); // keep incomplete chunk

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith('data:')) continue;
      let event;
      try {
        event = JSON.parse(line.slice(5).trim());
      } catch { continue; }

      handlePipelineEvent(event, stageContainer, appendResult, prompt);
    }
  }
}

// ── Event dispatch ────────────────────────────────────────────────────────────
function handlePipelineEvent(event, stageContainer, appendResult, originalPrompt) {
  const { stage, status } = event;

  if (stage === 'system1' && status === 'running') {
    upsertStageCard(stageContainer, 'system1', 'running',
      '🔍 Classifying intent…', '');
  }
  else if (stage === 'system1' && status === 'done') {
    const { intent, confidence, entities } = event;
    const pct = Math.round(confidence * 100);
    const entityStr = Object.entries(entities || {})
      .map(([k, v]) => `<span class="tag">${k}: ${v}</span>`)
      .join('');
    upsertStageCard(stageContainer, 'system1', 'done',
      `Intent: <strong>${intent}</strong>`,
      `<div class="prob-bar-wrap">
         <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:${pct}%"></div></div>
         <span class="prob-label">${pct}%</span>
       </div>
       <div style="margin-top:4px">${entityStr || '<span style="color:var(--muted)">no entities extracted</span>'}</div>`
    );
  }
  else if (stage === 'clarify' && status === 'running') {
    upsertStageCard(stageContainer, 'clarify', 'running',
      '🤔 Confidence too low — asking Grok…', '');
  }
  else if (stage === 'clarify' && status === 'needed') {
    // Replace spinner with question UI
    upsertStageCard(stageContainer, 'clarify', 'warn',
      'Clarification needed', '');
    appendResult(renderClarifyCard(event.question, originalPrompt, stageContainer, appendResult));
  }
  else if (stage === 'mcp' && status === 'running') {
    upsertStageCard(stageContainer, 'mcp', 'running',
      `📚 Fetching docs…`,
      `<span class="tag">${event.tool}</span><span class="tag">${event.operation}</span>`);
  }
  else if (stage === 'mcp' && status === 'done') {
    const { tool, has_docs, doc_chunk } = event;
    const syntax = doc_chunk?.command_syntax || '';
    upsertStageCard(stageContainer, 'mcp', 'done',
      `Docs: <strong>${tool}</strong>`,
      has_docs
        ? `<span class="tag">syntax</span> ${escHtml(syntax.slice(0, 80))}${syntax.length > 80 ? '…' : ''}`
        : '<span style="color:var(--muted)">no docs — stub used</span>');
  }
  else if (stage === 'mcp' && status === 'skipped') {
    upsertStageCard(stageContainer, 'mcp', 'warn',
      'MCP skipped', escHtml(event.reason || ''));
  }
  else if (stage === 'system2' && status === 'running') {
    upsertStageCard(stageContainer, 'system2', 'running',
      '⚙️ Generating command…', '');
  }
  else if (stage === 'system2' && status === 'done') {
    upsertStageCard(stageContainer, 'system2', 'done',
      'Command generated', '');
    appendResult(renderCommandCard(event));
  }
  else if (stage === 'error') {
    appendResult(renderErrorCard(event.message || 'Unknown error'));
  }
}

// ── Stage card upsert ────────────────────────────────────────────────────────
function upsertStageCard(container, stageId, state, label, detail) {
  let card = container.querySelector(`[data-stage="${stageId}"]`);
  if (!card) {
    card = document.createElement('div');
    card.className = 'stage-card';
    card.dataset.stage = stageId;
    container.appendChild(card);
    scrollToBottom();
  }
  const iconHtml = state === 'running'
    ? '<div class="stage-icon spinning"></div>'
    : state === 'done'    ? '<div class="stage-icon ok">✓</div>'
    : state === 'warn'    ? '<div class="stage-icon warn">!</div>'
    : state === 'skipped' ? '<div class="stage-icon warn">–</div>'
    :                       '<div class="stage-icon fail">✕</div>';

  card.className = `stage-card ${state}`;
  card.innerHTML = `
    ${iconHtml}
    <div class="stage-body">
      <div class="stage-label">${label}</div>
      ${detail ? `<div class="stage-detail">${detail}</div>` : ''}
    </div>`;
}

// ── Clarify card ──────────────────────────────────────────────────────────────
function renderClarifyCard(question, originalPrompt, stageContainer, appendResult) {
  const card = document.createElement('div');
  card.className = 'clarify-card';
  card.innerHTML = `
    <div class="clarify-question">💬 ${escHtml(question)}</div>
    <div class="clarify-row">
      <input class="clarify-input" id="clarify-answer-input"
             type="text" placeholder="Type your answer…" autocomplete="off" />
      <button class="clarify-submit" id="clarify-submit-btn">Send</button>
    </div>`;

  const input  = card.querySelector('#clarify-answer-input');
  const submit = card.querySelector('#clarify-submit-btn');

  const doSubmit = async () => {
    const answer = input.value.trim();
    if (!answer) return;
    card.innerHTML = `<div style="color:var(--muted);font-size:13px">📝 "${escHtml(answer)}"</div>`;
    appendUserBubble(answer);
    setProcessing(true);
    await runPipeline(originalPrompt, answer, stageContainer, appendResult);
    setProcessing(false);
  };

  submit.addEventListener('click', doSubmit);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') doSubmit(); });
  setTimeout(() => input.focus(), 50);

  return card;
}

// ── Command result card ───────────────────────────────────────────────────────
function renderCommandCard(event) {
  const plan = event.command_plan;
  const raw = event.raw || '';

  // Extract the command string from plan
  let commandStr = '';
  if (plan) {
    const steps = plan.steps || [];
    if (steps.length > 0) {
      commandStr = steps.map(s => s.command || '').filter(Boolean).join('\n');
    }
    if (!commandStr) commandStr = plan.command || plan.primary_command || '';
  }
  if (!commandStr && raw) {
    // Fallback: show raw with JSON syntax
    commandStr = raw.trim();
  }

  const meta = plan ? [
    plan.intent_type && `intent: ${plan.intent_type}`,
    plan.os          && `os: ${plan.os}`,
    plan.shell       && `shell: ${plan.shell}`,
  ].filter(Boolean) : [];

  const card = document.createElement('div');
  card.className = 'result-card';
  card.innerHTML = `
    <div class="result-header">
      <span>⚡ Generated Command</span>
      <div class="result-actions">
        <button class="result-btn" id="copy-btn">Copy</button>
      </div>
    </div>
    <pre class="result-command" id="result-command-text">${escHtml(commandStr || '(empty output)')}</pre>
    ${meta.length ? `<div class="result-meta">${meta.map(m => `<span><span class="label">${m.split(':')[0]}:</span> ${m.split(':')[1]}</span>`).join('')}</div>` : ''}`;

  card.querySelector('#copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(commandStr).then(() => {
      const btn = card.querySelector('#copy-btn');
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
    });
  });

  return card;
}

function renderErrorCard(message) {
  const card = document.createElement('div');
  card.className = 'error-card';
  card.textContent = `⚠️ ${message}`;
  return card;
}

// ── DOM helpers ───────────────────────────────────────────────────────────────
function appendUserBubble(text) {
  const row = document.createElement('div');
  row.className = 'msg-row';
  const bubble = document.createElement('div');
  bubble.className = 'bubble-user';
  bubble.textContent = text;
  row.appendChild(bubble);
  chatHistory.appendChild(row);
  scrollToBottom();
  return row;
}

function createAssistantRow() {
  const msgRow = document.createElement('div');
  msgRow.className = 'msg-row';

  const panel = document.createElement('div');
  panel.className = 'thinking-panel';

  const header = document.createElement('div');
  header.className = 'thinking-header';
  header.innerHTML = `<span>⟳ Thinking</span>`;

  const stageContainer = document.createElement('div');
  stageContainer.className = 'thinking-steps';

  panel.appendChild(header);
  panel.appendChild(stageContainer);
  msgRow.appendChild(panel);
  chatHistory.appendChild(msgRow);
  scrollToBottom();

  // appendResult: add a DOM element to the row below the panel
  function appendResult(el) {
    msgRow.appendChild(el);
    // Update thinking header once done
    header.innerHTML = '<span style="color:var(--success)">✓ Done</span>';
    scrollToBottom();
  }

  return { msgRow, stageContainer, appendResult };
}

function clearStageContainer(c) {
  c.innerHTML = '';
}

function scrollToBottom() {
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function setProcessing(active) {
  isProcessing = active;
  sendBtn.disabled   = active || !bridgeReady;
  userInput.disabled = active || !bridgeReady;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
