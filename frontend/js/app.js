/**
 * Policy Lens — Frontend Application
 * Hospital Policy Intelligence
 *
 * Architecture:
 *   Browser → fetch('/api/ask') → FastAPI (api.py) → rag.py → Gemini
 *
 * API keys are NEVER accessed here. They live server-side only.
 */

'use strict';

/* ─── Constants ─────────────────────────────────────────────── */
const API_ASK    = '/api/ask';
const API_STATUS = '/api/status';
const STORAGE_KEY = 'policyLensConversations';
const MAX_TITLE_LEN = 50;
const MAX_HISTORY_ITEMS = 20;

const LOADING_MESSAGES = [
  { main: 'Searching policies…',             sub: 'Scanning indexed documents' },
  { main: 'Finding relevant sections…',      sub: 'Running hybrid retrieval' },
  { main: 'Evaluating evidence…',            sub: 'Checking relevance threshold' },
  { main: 'Generating grounded response…',   sub: 'Composing answer with Gemini' },
];

const SUGGESTED_QUESTIONS = [
  { label: 'What are the visiting hours?',           icon: 'clock' },
  { label: 'What is the admission process?',         icon: 'clipboard-list' },
  { label: 'How does cashless insurance work?',      icon: 'shield' },
  { label: 'How can I access my patient records?',   icon: 'folder-open' },
  { label: 'What is the discharge process?',         icon: 'log-out' },
  { label: 'How do I book an outpatient appointment?', icon: 'calendar' },
];

/* ─── State ─────────────────────────────────────────────────── */
const state = {
  currentPage: 'ask',
  isLoading: false,
  loadingInterval: null,
  loadingStep: 0,
  activeConversationId: null,
  // Current session messages (rendered in conversation-area)
  currentMessages: [],
  lastQuestion: '',
};

/* ─── DOM helpers ────────────────────────────────────────────── */
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function sanitize(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

/**
 * Very light markdown→HTML renderer (bold, italic, lists, paragraphs).
 */
function renderMarkdown(raw) {
  if (!raw) return '';
  const blocks = raw.split(/\n{2,}/);
  return blocks.map(block => {
    const lines = block.split('\n');
    if (lines.every(l => /^[\-\*]\s/.test(l.trim()) || l.trim() === '')) {
      const items = lines.filter(l => l.trim()).map(l =>
        `<li>${inlineMd(l.replace(/^[\-\*]\s/, '').trim())}</li>`).join('');
      return `<ul>${items}</ul>`;
    }
    if (lines.every(l => /^\d+\.\s/.test(l.trim()) || l.trim() === '')) {
      const items = lines.filter(l => l.trim()).map(l =>
        `<li>${inlineMd(l.replace(/^\d+\.\s/, '').trim())}</li>`).join('');
      return `<ol>${items}</ol>`;
    }
    const para = lines.map(l => inlineMd(l)).join('<br>');
    return `<p>${para}</p>`;
  }).join('');
}

function inlineMd(text) {
  let s = sanitize(text);
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
  return s;
}

/* ─── Icons ─────────────────────────────────────────────────── */
const ICONS = {
  'clock':          `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>`,
  'clipboard-list': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4M12 16h4M8 11h.01M8 16h.01"/></svg>`,
  'shield':         `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/></svg>`,
  'folder-open':    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/></svg>`,
  'log-out':        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" x2="9" y1="12" y2="12"/></svg>`,
  'calendar':       `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2v4M16 2v4"/><rect width="18" height="18" x="3" y="4" rx="2"/><path d="M3 10h18"/></svg>`,
  'send':           `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>`,
  'refresh-cw':     `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/></svg>`,
  'sun':            `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>`,
  'moon':           `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>`,
  'layers':         `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/></svg>`,
  'sparkles':       `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/></svg>`,
  'chevron-down':   `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>`,
  'file-text':      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8M16 13H8M16 17H8"/></svg>`,
  'book':           `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H19a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6.5a1 1 0 0 1 0-5H20"/></svg>`,
  'map-pin':        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0"/><circle cx="12" cy="10" r="3"/></svg>`,
  'message-circle': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"/></svg>`,
  'alert-triangle': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>`,
  'x-circle':       `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/></svg>`,
  'history':        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/></svg>`,
  'thumbs-up':      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/></svg>`,
  'thumbs-down':    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"/></svg>`,
  'activity':       `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>`,
  'cpu':            `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M20 15h2M2 9h2M20 9h2M9 2v2M9 20v2"/></svg>`,
  'users':          `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>`,
  'database':       `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/></svg>`,
  'check-circle':   `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><path d="m9 11 3 3L22 4"/></svg>`,
};

function iconHTML(name) { return ICONS[name] || ''; }

/* ─── UUID generator ─────────────────────────────────────────── */
function uuid() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

/* ─── Page Navigation ────────────────────────────────────────── */
function navigateTo(page) {
  state.currentPage = page;
  $$('.nav-item').forEach(item => item.classList.toggle('active', item.dataset.page === page));
  $$('.page-section').forEach(s => s.classList.toggle('active', s.id === `page-${page}`));
  closeMobileSidebar();
}

/* ─── Mobile Sidebar ─────────────────────────────────────────── */
function openMobileSidebar() {
  $('#sidebar').classList.add('open');
  $('#sidebar-overlay').classList.add('open');
  $('#mobile-menu-btn').setAttribute('aria-expanded', 'true');
}
function closeMobileSidebar() {
  $('#sidebar').classList.remove('open');
  $('#sidebar-overlay').classList.remove('open');
  $('#mobile-menu-btn').setAttribute('aria-expanded', 'false');
}

/* ─── Theme ──────────────────────────────────────────────────── */
function initTheme() {
  const saved = localStorage.getItem('policyLensTheme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  updateThemeIcon(saved);
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute('data-theme');
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('policyLensTheme', next);
  updateThemeIcon(next);
}
function updateThemeIcon(theme) {
  const btn = $('#theme-toggle-btn');
  if (btn) btn.innerHTML = theme === 'dark' ? iconHTML('sun') : iconHTML('moon');
}

/* ─── Status ─────────────────────────────────────────────────── */
async function fetchStatus() {
  try {
    const res = await fetch(API_STATUS, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return;
    const data = await res.json();
    const docEl = $('#doc-count');
    if (docEl) docEl.textContent = `${data.documents_indexed} documents indexed`;
    const navBadge = $('#nav-doc-count');
    if (navBadge) navBadge.textContent = data.documents_indexed;
    const statusText = $('#status-text');
    if (statusText && data.status === 'operational') {
      statusText.textContent = 'System operational';
    }
    const tooltip = $('#status-tooltip');
    if (tooltip) tooltip.textContent = 'Backend connected';
  } catch {
    const tooltip = $('#status-tooltip');
    if (tooltip) tooltip.textContent = 'Backend unavailable';
    const badge = $('#status-badge');
    if (badge) {
      badge.style.background = 'var(--red-100)';
      const dot = badge.querySelector('.status-dot');
      if (dot) dot.style.background = 'var(--red-500)';
      const txt = badge.querySelector('.status-text');
      if (txt) txt.textContent = 'Offline';
    }
  }
}

/* ─── Loading ────────────────────────────────────────────────── */
function showLoading() {
  state.isLoading = true;
  state.loadingStep = 0;
  const el = $('#loading-state');
  if (el) { el.setAttribute('aria-hidden', 'false'); }
  const btn = $('#submit-btn');
  if (btn) { btn.disabled = true; btn.innerHTML = `${iconHTML('refresh-cw')} <span>Thinking…</span>`; }
  updateLoadingMessage();
  state.loadingInterval = setInterval(() => {
    state.loadingStep = (state.loadingStep + 1) % LOADING_MESSAGES.length;
    updateLoadingMessage();
  }, 2200);
}
function updateLoadingMessage() {
  const msg = LOADING_MESSAGES[state.loadingStep];
  const m = $('#loading-message'); if (m) m.textContent = msg.main;
  const s = $('#loading-sub');    if (s) s.textContent = msg.sub;
}
function hideLoading() {
  state.isLoading = false;
  clearInterval(state.loadingInterval);
  const el = $('#loading-state');
  if (el) el.setAttribute('aria-hidden', 'true');
  const btn = $('#submit-btn');
  if (btn) { btn.disabled = false; btn.innerHTML = `${iconHTML('send')} <span>Ask</span>`; }
}

/* ─── Char counter ───────────────────────────────────────────── */
function updateCharCount(val) {
  const c = $('#char-count');
  if (!c) return;
  c.textContent = `${val.length} / 1000`;
  c.classList.toggle('near-limit', val.length > 850);
}

/* ─── Conversation Storage ───────────────────────────────────── */
function loadAllConversations() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
  catch { return []; }
}
function saveAllConversations(convs) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(convs.slice(0, MAX_HISTORY_ITEMS)));
}
function getActiveConversation() {
  const convs = loadAllConversations();
  return convs.find(c => c.id === state.activeConversationId) || null;
}
function upsertConversation(conv) {
  let convs = loadAllConversations();
  const idx = convs.findIndex(c => c.id === conv.id);
  if (idx >= 0) convs[idx] = conv;
  else convs.unshift(conv);
  saveAllConversations(convs);
}

/* ─── New Conversation ───────────────────────────────────────── */
function createNewConversation() {
  state.activeConversationId = null;
  state.currentMessages = [];
  state.lastQuestion = '';
  // Clear conversation area & show empty state
  const ca = $('#conversation-area');
  if (ca) ca.innerHTML = '';
  const es = $('#empty-state');
  if (es) es.style.display = '';
  // Clear input
  const inp = $('#question-input');
  if (inp) { inp.value = ''; inp.style.height = 'auto'; updateCharCount(''); }
  // Re-render history
  renderSidebarHistory();
  // Focus input
  setTimeout(() => inp && inp.focus(), 50);
}

/* ─── Load Conversation ──────────────────────────────────────── */
function loadConversation(id) {
  const convs = loadAllConversations();
  const conv = convs.find(c => c.id === id);
  if (!conv) return;
  state.activeConversationId = id;
  state.currentMessages = conv.messages || [];

  const ca = $('#conversation-area');
  if (!ca) return;
  ca.innerHTML = '';

  const es = $('#empty-state');
  if (es) es.style.display = 'none';

  // Re-render all messages
  conv.messages.forEach((msg, idx) => {
    if (msg.role === 'user') {
      ca.appendChild(buildUserMsgEl(msg.content));
    } else if (msg.role === 'assistant') {
      if (msg.insufficient_evidence) {
        ca.appendChild(buildInsufficientEl(msg.content));
      } else if (msg.isError) {
        ca.appendChild(buildErrorEl(msg.content));
      } else {
        ca.appendChild(buildAiMsgEl(msg.content, msg.timestamp, msg.feedback, idx, true));
        if (msg.sources && msg.sources.length) {
          ca.appendChild(buildSourcesEl(msg.sources));
        }
        ca.appendChild(buildFeedbackEl(idx, msg.feedback));
      }
    }
    // Separator between rounds
    if (msg.role === 'assistant' && idx < conv.messages.length - 1) {
      const sep = document.createElement('hr');
      sep.className = 'conv-separator';
      ca.appendChild(sep);
    }
  });

  renderSidebarHistory();
  closeMobileSidebar();
  ca.scrollTop = ca.scrollHeight;
}

/* ─── Sidebar History ────────────────────────────────────────── */
function renderSidebarHistory() {
  const container = $('#sidebar-history');
  if (!container) return;
  const convs = loadAllConversations();

  if (!convs.length) {
    container.innerHTML = `<p class="history-empty">No conversations yet.<br>Ask your first question!</p>`;
    return;
  }

  // Group: recent (last 7 days) vs older
  const now = Date.now();
  const weekMs = 7 * 24 * 60 * 60 * 1000;
  const recent = convs.filter(c => now - c.updatedAt < weekMs);
  const older  = convs.filter(c => now - c.updatedAt >= weekMs);

  let html = '';
  if (recent.length) {
    html += `<p class="history-section-label">Recent</p>`;
    html += recent.map(c => historyItemHTML(c)).join('');
  }
  if (older.length) {
    html += `<p class="history-section-label">Older</p>`;
    html += older.map(c => historyItemHTML(c)).join('');
  }
  container.innerHTML = html;

  // Attach click handlers
  container.querySelectorAll('.history-item').forEach(item => {
    item.addEventListener('click', () => loadConversation(item.dataset.id));
    item.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') loadConversation(item.dataset.id); });
  });
}

function historyItemHTML(conv) {
  const active = conv.id === state.activeConversationId ? ' active' : '';
  const title  = sanitize(conv.title || 'Untitled conversation');
  return `<button class="history-item${active}" data-id="${conv.id}" tabindex="0" aria-label="Load conversation: ${title}">
    ${iconHTML('message-circle')}
    <span>${title}</span>
  </button>`;
}

/* ─── Message Building ───────────────────────────────────────── */
function buildUserMsgEl(text) {
  const div = document.createElement('div');
  div.className = 'msg-user';
  div.innerHTML = `<div class="msg-user-label">You</div>
    <div class="msg-user-content">${sanitize(text)}</div>`;
  return div;
}

function buildAiMsgEl(text, timestamp, feedback, msgIndex, isRestored) {
  const ts = timestamp ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
  const div = document.createElement('div');
  div.className = 'msg-ai';
  div.innerHTML = `
    <div class="msg-ai-label">
      ${iconHTML('sparkles')}
      Policy Lens
    </div>
    <div class="msg-ai-content" id="ai-content-${msgIndex}">${renderMarkdown(text)}</div>
    ${ts ? `<div class="msg-timestamp">${ts}</div>` : ''}
  `;
  return div;
}

function buildFeedbackEl(msgIndex, existingFeedback) {
  const div = document.createElement('div');
  div.className = 'feedback-area';
  div.id = `feedback-${msgIndex}`;
  div.innerHTML = `
    <span class="feedback-label">Was this helpful?</span>
    <button class="feedback-btn${existingFeedback === 'positive' ? ' active-positive' : ''}" 
            data-type="positive" data-index="${msgIndex}"
            aria-label="Helpful" aria-pressed="${existingFeedback === 'positive'}">
      ${iconHTML('thumbs-up')} Yes
    </button>
    <button class="feedback-btn${existingFeedback === 'negative' ? ' active-negative' : ''}" 
            data-type="negative" data-index="${msgIndex}"
            aria-label="Not helpful" aria-pressed="${existingFeedback === 'negative'}">
      ${iconHTML('thumbs-down')} No
    </button>
    <span class="feedback-thanks${existingFeedback ? ' visible' : ''}">
      ${existingFeedback === 'positive' ? 'Thanks for your feedback.' : existingFeedback === 'negative' ? 'Thanks. We\'ll use this to improve Policy Lens.' : ''}
    </span>
  `;
  div.querySelectorAll('.feedback-btn').forEach(btn => {
    btn.addEventListener('click', () => handleFeedback(btn.dataset.type, parseInt(btn.dataset.index)));
  });
  return div;
}

function buildInsufficientEl(text) {
  const div = document.createElement('div');
  div.className = 'msg-ai';
  div.innerHTML = `
    <div class="msg-ai-label">${iconHTML('sparkles')} Policy Lens</div>
    <div class="insufficient-msg">
      <h4>Insufficient Policy Evidence</h4>
      <p>${sanitize(text)}</p>
      <p class="tip">Try asking about admission, billing, insurance, visitors, discharge, staff administration, governance, patient records, or outpatient appointments.</p>
    </div>
  `;
  return div;
}

function buildErrorEl(text, retryQuestion) {
  const div = document.createElement('div');
  div.className = 'error-msg';
  div.innerHTML = `
    <h4>Unable to get an answer</h4>
    <p>${sanitize(text)}</p>
    ${retryQuestion ? `<button class="retry-btn" id="retry-btn">${iconHTML('refresh-cw')} Retry</button>` : ''}
  `;
  if (retryQuestion) {
    div.querySelector('#retry-btn')?.addEventListener('click', () => {
      const inp = $('#question-input');
      if (inp) { inp.value = retryQuestion; updateCharCount(retryQuestion); }
      submitQuestion();
    });
  }
  return div;
}

function buildSourcesEl(sources) {
  const div = document.createElement('div');
  div.className = 'sources-section';
  div.innerHTML = `
    <div class="sources-header">
      <div class="sources-title">
        ${iconHTML('layers')}
        Sources
        <span class="sources-count">${sources.length}</span>
      </div>
      <span class="sources-note">Ranked by relevance</span>
    </div>
    <div class="sources-grid">${sources.map((s, i) => sourceCardHTML(s, i + 1)).join('')}</div>
  `;
  // Attach passage toggles
  div.querySelectorAll('.source-passage-toggle').forEach(btn => {
    btn.addEventListener('click', () => togglePassage(btn));
  });
  return div;
}

function sourceCardHTML(src, rank) {
  const hybridPct = (src.hybrid_score * 100).toFixed(1);
  const barWidth  = Math.min(100, src.hybrid_score * 100).toFixed(1);
  const pages     = Array.isArray(src.pages) ? src.pages.join(', ') : (src.pages ?? '—');
  const docName   = sanitize(src.document  || '—');
  const policyId  = sanitize(src.policy_id || '—');
  const section   = sanitize(src.section   || '—');
  const passageId = `passage-${rank}-${Date.now()}`;
  const toggleId  = `toggle-${rank}-${Date.now()}`;

  return `<div class="source-card">
    <div class="source-card-header">
      <div class="source-rank">${String(rank).padStart(2, '0')}</div>
      <div class="source-info">
        <div class="source-document">${docName}</div>
        <div class="source-meta">
          <span class="source-tag">${iconHTML('file-text')} ${policyId}</span>
          <span class="source-tag">${iconHTML('book')} ${section}</span>
          <span class="source-tag">${iconHTML('map-pin')} Pg. ${sanitize(String(pages))}</span>
        </div>
      </div>
      <div class="source-relevance">
        <div class="relevance-label">Relevance</div>
        <div class="relevance-score">${hybridPct}%</div>
        <div class="relevance-bar"><div class="relevance-bar-fill" style="width:${barWidth}%"></div></div>
      </div>
    </div>
    <button class="source-passage-toggle" id="${toggleId}" aria-expanded="false" aria-controls="${passageId}">
      ${iconHTML('chevron-down')} View passage
    </button>
    <div class="source-passage" id="${passageId}" role="region" aria-label="Retrieved passage">
      <blockquote>${sanitize(src.text || '')}</blockquote>
    </div>
  </div>`;
}

function togglePassage(btn) {
  const passageId = btn.getAttribute('aria-controls');
  const passage   = document.getElementById(passageId);
  if (!passage) return;
  const isOpen = passage.classList.toggle('open');
  btn.classList.toggle('open', isOpen);
  btn.setAttribute('aria-expanded', String(isOpen));
  const span = btn.querySelector('span');
  if (span) span.textContent = isOpen ? 'Hide passage' : 'View passage';
}

/* ─── Feedback ───────────────────────────────────────────────── */
function handleFeedback(type, msgIndex) {
  const area = document.getElementById(`feedback-${msgIndex}`);
  if (!area) return;

  // Update buttons
  area.querySelectorAll('.feedback-btn').forEach(btn => {
    btn.classList.remove('active-positive', 'active-negative');
    btn.setAttribute('aria-pressed', 'false');
  });
  const activeBtn = area.querySelector(`[data-type="${type}"]`);
  if (activeBtn) {
    activeBtn.classList.add(type === 'positive' ? 'active-positive' : 'active-negative');
    activeBtn.setAttribute('aria-pressed', 'true');
  }

  // Show thanks
  const thanks = area.querySelector('.feedback-thanks');
  if (thanks) {
    thanks.textContent = type === 'positive'
      ? 'Thanks for your feedback.'
      : "Thanks for your feedback. We'll use this to improve Policy Lens.";
    thanks.classList.add('visible');
  }

  // Persist to localStorage
  if (state.activeConversationId) {
    const convs = loadAllConversations();
    const conv = convs.find(c => c.id === state.activeConversationId);
    if (conv && conv.messages[msgIndex]) {
      conv.messages[msgIndex].feedback = type;
      saveAllConversations(convs);
    }
  }
}

/* ─── Submit Question ────────────────────────────────────────── */
async function submitQuestion() {
  if (state.isLoading) return;
  const input = $('#question-input');
  const question = input?.value.trim();
  if (!question) { input?.focus(); return; }

  state.lastQuestion = question;

  // Clear input
  input.value = '';
  input.style.height = 'auto';
  updateCharCount('');

  // Hide empty state
  const es = $('#empty-state');
  if (es) es.style.display = 'none';

  const ca = $('#conversation-area');
  if (!ca) return;

  // If this is the first message in a new session, create the conversation
  if (!state.activeConversationId) {
    const title = question.length > MAX_TITLE_LEN
      ? question.slice(0, MAX_TITLE_LEN) + '…'
      : question;
    const conv = {
      id: uuid(),
      title,
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    state.activeConversationId = conv.id;
    upsertConversation(conv);
  }

  // Determine msg index
  const userMsgIndex = state.currentMessages.length;
  const aiMsgIndex   = userMsgIndex + 1;

  // Push user message to state & storage
  const userMsg = { role: 'user', content: question, timestamp: Date.now() };
  state.currentMessages.push(userMsg);
  updateConvMessages();

  // Separator if there were previous messages
  if (ca.children.length > 0) {
    const sep = document.createElement('hr');
    sep.className = 'conv-separator';
    ca.appendChild(sep);
  }

  // Render user message
  ca.appendChild(buildUserMsgEl(question));
  ca.scrollTop = ca.scrollHeight;

  // Show loading
  showLoading();

  try {
    const data = await fetchAnswer(question);
    hideLoading();

    const ts = Date.now();
    if (data.insufficient_evidence) {
      const aiMsg = { role: 'assistant', content: data.answer, insufficient_evidence: true, timestamp: ts };
      state.currentMessages.push(aiMsg);
      ca.appendChild(buildInsufficientEl(data.answer));
    } else {
      const aiMsg = { role: 'assistant', content: data.answer, sources: data.sources || [], insufficient_evidence: false, timestamp: ts, feedback: null };
      state.currentMessages.push(aiMsg);
      ca.appendChild(buildAiMsgEl(data.answer, ts, null, aiMsgIndex, false));
      if (data.sources && data.sources.length) {
        ca.appendChild(buildSourcesEl(data.sources));
      }
      ca.appendChild(buildFeedbackEl(aiMsgIndex, null));
    }

    updateConvMessages();
    renderSidebarHistory();

    // Scroll to bottom
    setTimeout(() => ca.scrollTop = ca.scrollHeight, 50);

  } catch (err) {
    hideLoading();
    const msg = err.userMessage || 'Policy Lens is temporarily unavailable. Please try again.';
    const errMsg = { role: 'assistant', content: msg, isError: true, timestamp: Date.now() };
    state.currentMessages.push(errMsg);
    ca.appendChild(buildErrorEl(msg, question));
    updateConvMessages();
    ca.scrollTop = ca.scrollHeight;
  }
}

function updateConvMessages() {
  if (!state.activeConversationId) return;
  const convs = loadAllConversations();
  const conv = convs.find(c => c.id === state.activeConversationId);
  if (!conv) return;
  conv.messages = state.currentMessages;
  conv.updatedAt = Date.now();
  saveAllConversations(convs);
}

/* ─── Fetch ──────────────────────────────────────────────────── */
async function fetchAnswer(question) {
  let response;
  try {
    response = await fetch(API_ASK, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: AbortSignal.timeout(60000),
    });
  } catch (err) {
    if (err.name === 'TimeoutError') {
      const e = new Error('timeout');
      e.userMessage = 'The request timed out. The model may be busy — please try again.';
      throw e;
    }
    const e = new Error('network');
    e.userMessage = 'Network error. Please check your connection and try again.';
    throw e;
  }
  if (!response.ok) {
    let detail = 'An unexpected error occurred.';
    try { const body = await response.json(); detail = body.detail || detail; } catch {}
    const e = new Error(`http-${response.status}`);
    e.userMessage = detail;
    throw e;
  }
  try { return await response.json(); }
  catch {
    const e = new Error('parse');
    e.userMessage = 'Received an invalid response from the server.';
    throw e;
  }
}

/* ─── Suggestion Chips ───────────────────────────────────────── */
function renderSuggestions() {
  const container = $('#suggestions-chips');
  if (!container) return;
  container.innerHTML = SUGGESTED_QUESTIONS.map(q => `
    <button class="suggestion-chip"
            aria-label="Ask: ${sanitize(q.label)}">
      ${iconHTML(q.icon)}
      ${sanitize(q.label)}
    </button>
  `).join('');
  container.querySelectorAll('.suggestion-chip').forEach((btn, i) => {
    btn.addEventListener('click', () => {
      const label = SUGGESTED_QUESTIONS[i].label;
      const inp = $('#question-input');
      if (inp) { inp.value = label; updateCharCount(label); }
      submitQuestion();
    });
  });
}

/* ─── Policy Library ─────────────────────────────────────────── */
const POLICIES = [
  { id:'00', name:'Source Reference Notes',      file:'00_source_reference_notes.pdf',      color:'#6366f1', bg:'#eef2ff', icon:'book' },
  { id:'01', name:'Admission & Registration',    file:'01_admission_registration_policy.pdf',color:'#0ea5e9', bg:'#f0f9ff', icon:'clipboard-list' },
  { id:'02', name:'Billing & Payment',           file:'02_billing_payment_policy.pdf',       color:'#10b981', bg:'#ecfdf5', icon:'activity' },
  { id:'03', name:'Insurance & Cashless',        file:'03_insurance_cashless_policy.pdf',    color:'#3b82f6', bg:'#eff6ff', icon:'shield' },
  { id:'04', name:'Visitor Access',              file:'04_visitor_access_policy.pdf',        color:'#f59e0b', bg:'#fffbeb', icon:'users' },
  { id:'05', name:'Discharge Process',           file:'05_discharge_process_policy.pdf',     color:'#ef4444', bg:'#fef2f2', icon:'log-out' },
  { id:'06', name:'Staff Administration',        file:'06_staff_administrative_policy.pdf',  color:'#8b5cf6', bg:'#f5f3ff', icon:'cpu' },
  { id:'07', name:'Governance & Meetings',       file:'07_governance_meetings_policy.pdf',   color:'#0891b2', bg:'#ecfeff', icon:'layers' },
  { id:'08', name:'Patient Records',             file:'08_patient_records_policy.pdf',       color:'#16a34a', bg:'#f0fdf4', icon:'folder-open' },
  { id:'09', name:'Appointments & Outpatient',   file:'09_appointments_outpatient_policy.pdf',color:'#ea580c', bg:'#fff7ed', icon:'calendar' },
];

function renderPolicyLibrary() {
  const c = $('#policy-grid');
  if (!c) return;
  c.innerHTML = POLICIES.map(p => `
    <div class="policy-card" role="article" aria-label="${sanitize(p.name)}">
      <div class="policy-card-icon" style="background:${p.bg};color:${p.color};">${iconHTML(p.icon)}</div>
      <h3>${sanitize(p.name)}</h3>
      <p>${sanitize(p.file)}</p>
      <div class="policy-card-status"><span></span> Indexed &amp; ready</div>
    </div>
  `).join('');
}

/* ─── How It Works ───────────────────────────────────────────── */
const PIPELINE = [
  { n:'01', title:'Policy Documents',      desc:'10 hospital policy PDFs: admission, billing, insurance, visitor access, discharge, staff administration, governance, patient records, and appointments.', tag:'10 PDF documents' },
  { n:'02', title:'Section Chunking',      desc:'Each document is divided into meaningful sections using a section-aware chunker. Chunks preserve document name, policy ID, section name, and page numbers.', tag:'Section-aware chunking' },
  { n:'03', title:'TF-IDF + Embeddings',   desc:'Questions are processed using TF-IDF (keyword precision) and a Sentence Transformer (semantic similarity) to create two independent relevance signals.', tag:'TF-IDF (40%) + Embeddings (60%)' },
  { n:'04', title:'Hybrid Retrieval',      desc:'Keyword and semantic scores are combined with a weighted formula to rank the most relevant policy chunks for each question.', tag:'Hybrid scoring' },
  { n:'05', title:'Evidence Threshold',    desc:'Only chunks with a hybrid relevance score above the minimum threshold (0.30) are used as evidence. Questions without sufficient evidence are refused.', tag:'Min. score: 0.30' },
  { n:'06', title:'Grounded Generation',   desc:'Top-ranked policy chunks are assembled as evidence and passed to the Gemini model. The model is strictly instructed to use only the provided evidence.', tag:'Gemini — evidence-grounded' },
  { n:'07', title:'Source Attribution',    desc:'Every answer includes the originating document name, policy ID, section name, and page number for each retrieved chunk — making all results transparent and verifiable.', tag:'Document · Section · Page' },
];

function renderHowItWorks() {
  const c = $('#pipeline-steps');
  if (!c) return;
  c.innerHTML = PIPELINE.map(step => `
    <div class="pipeline-step">
      <div class="pipeline-step-left">
        <div class="step-number">${step.n}</div>
        <div class="step-connector"></div>
      </div>
      <div class="pipeline-step-right">
        <h3>${sanitize(step.title)}</h3>
        <p>${sanitize(step.desc)}</p>
        <span class="step-tag">${sanitize(step.tag)}</span>
      </div>
    </div>
  `).join('');
}

/* ─── Evaluation ─────────────────────────────────────────────── */
function renderEvaluation() {
  // Show only real/known system config — no "Not configured" placeholders
  const configMetrics = [
    { label:'Documents Indexed', value:'10',   sub:'Policy PDF documents' },
    { label:'Retrieval Threshold', value:'0.30', sub:'Minimum hybrid score' },
    { label:'TF-IDF Weight',    value:'40%',   sub:'Keyword component' },
    { label:'Embedding Weight', value:'60%',   sub:'Semantic component' },
  ];
  const c = $('#eval-config-grid');
  if (!c) return;
  c.innerHTML = configMetrics.map(m => `
    <div class="eval-config-card">
      <div class="eval-config-label">${sanitize(m.label)}</div>
      <div class="eval-config-value">${sanitize(m.value)}</div>
      <div class="eval-config-sub">${sanitize(m.sub)}</div>
    </div>
  `).join('');
}

/* ─── Init ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  fetchStatus();
  renderSuggestions();
  renderPolicyLibrary();
  renderHowItWorks();
  renderEvaluation();
  renderSidebarHistory();

  // Navigation
  $$('.nav-item').forEach(item => {
    item.addEventListener('click', () => navigateTo(item.dataset.page));
    item.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') navigateTo(item.dataset.page); });
  });

  // Help button navigates to "how"
  $('#help-btn')?.addEventListener('click', () => navigateTo('how'));

  // Mobile menu
  $('#mobile-menu-btn')?.addEventListener('click', openMobileSidebar);
  $('#sidebar-overlay')?.addEventListener('click', closeMobileSidebar);

  // Theme toggle
  $('#theme-toggle-btn')?.addEventListener('click', toggleTheme);

  // New conversation
  $('#new-conv-btn')?.addEventListener('click', createNewConversation);

  // Question input
  const input = $('#question-input');
  if (input) {
    input.addEventListener('input', () => {
      updateCharCount(input.value);
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 160) + 'px';
    });
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitQuestion(); }
    });
  }

  // Submit button
  $('#submit-btn')?.addEventListener('click', submitQuestion);
});
