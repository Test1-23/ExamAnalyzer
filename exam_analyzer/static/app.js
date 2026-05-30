// ─── ExamAnalyzer chat + UI (extracted from index.html) ───

// ─── API helper ─────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ─── State ──────────────────────────────────────────
let polling = false;
let lastStatusUpdate = 0;
const STATUS_UPDATE_INTERVAL = 30000;
let cachedDisplayStatus = '就绪';

// ─── Config ─────────────────────────────────────────
async function loadConfig() {
  const cfg = await api('/api/config');
  document.getElementById('api-url').value = cfg.api_url || '';
  document.getElementById('api-key').value = cfg.api_key || '';
}

async function saveConfig() {
  const cfg = {
    api_url: document.getElementById('api-url').value.trim(),
    api_key: document.getElementById('api-key').value.trim(),
  };
  await api('/api/config', { method: 'POST', body: JSON.stringify(cfg) });
  M.toast({ html: '配置已保存', classes: 'green' });
}

// ─── Toggles ────────────────────────────────────────
function toggleSection(id, arrowId) {
  const body = document.getElementById(id);
  const arrow = document.getElementById(arrowId);
  if (body.style.display === 'none') {
    body.style.display = 'block';
    arrow.textContent = 'expand_less';
  } else {
    body.style.display = 'none';
    arrow.textContent = 'expand_more';
  }
}

function toggleConfig() { toggleSection('config-body', 'config-arrow'); }
function toggleDebug() { toggleSection('debug-body', 'debug-arrow'); }

function toggleTimeline() {
  const body = document.getElementById('timeline-body');
  const arrow = document.getElementById('timeline-arrow');
  if (body.style.display === 'none') {
    body.style.display = 'block';
    arrow.textContent = 'expand_less';
  } else {
    body.style.display = 'none';
    arrow.textContent = 'expand_more';
  }
}

// ─── Files ──────────────────────────────────────────
async function loadFiles() {
  const files = await api('/api/files');
  const emptyEl = document.getElementById('file-empty');
  const listWrap = document.getElementById('file-list-wrap');
  const projChips = document.getElementById('project-chips');

  if (files.length > 0) {
    emptyEl.style.display = 'none';
    listWrap.style.display = 'block';
    document.getElementById('file-count').textContent = files.length;
    projChips.innerHTML = '';
    files.forEach(f => {
      const span = document.createElement('span');
      span.className = 'chip';
      span.textContent = f;
      projChips.appendChild(span);
    });
  } else {
    emptyEl.style.display = 'block';
    listWrap.style.display = 'none';
  }
}

async function clearInput() {
  await api('/api/input-files', { method: 'DELETE' });
  M.toast({ html: '已清空 input 目录', classes: 'blue' });
  await loadFiles();
}

// ─── Analysis ───────────────────────────────────────
async function startAnalysis() {
  const btn = document.getElementById('analyze-btn');
  btn.disabled = true;
  btn.innerHTML = '<i class="material-icons left">hourglass_top</i> 分析中...';

  const res = await api('/api/analyze', { method: 'POST' });
  if (res.error) {
    M.toast({ text: res.error, classes: 'red' });
    btn.disabled = false;
    btn.innerHTML = '<i class="material-icons left">play_arrow</i> 开始分析';
    return;
  }

  document.getElementById('progress-section').style.display = 'block';
  document.getElementById('progress-title').innerHTML =
    '<i class="material-icons card-header-icon">hourglass_top</i> 分析中...';
  document.getElementById('progress-bar').style.width = '0%';
  document.getElementById('progress-text').textContent = '启动中...';
  forceUpdateStatusDisplay('运行中');

  polling = true;
  stopBgConnectionCheck();
  pollStatus();
}

async function startEvaluation() {
  const btn = document.querySelector('#analyze-btn');
  try {
    const res = await api('/api/evaluate', { method: 'POST' });
    if (res.error) {
      M.toast({ text: res.error, classes: 'red' });
    } else {
      M.toast({ text: '质量评估已启动，完成后将在反馈区展示报告', classes: 'teal' });
      pollEvaluation();
    }
  } catch (e) {
    M.toast({ text: '启动评估失败', classes: 'red' });
  }
}

async function pollEvaluation() {
  const card = document.getElementById('eval-card');
  const el = document.getElementById('eval-report');
  if (!el) return;
  card.style.display = 'block';
  try {
    const res = await api('/api/evaluate/status');
    if (res.running) {
      el.textContent = '评估中...';
      setTimeout(pollEvaluation, 3000);
    } else if (res.report) {
      el.textContent = res.report;
      M.toast({ text: '评估完成！', classes: 'teal' });
    } else if (res.error) {
      el.textContent = '评估失败: ' + res.error;
    }
  } catch (e) { /* ignore */ }
}

function showDebugLog(s) {
  const content = document.getElementById('debug-content');
  if (s.debug_log && s.debug_log.length > 0) {
    content.textContent = s.debug_log.join('\n');
  }
}

function updateStatusDisplay(text) {
  const now = Date.now();
  if (now - lastStatusUpdate >= STATUS_UPDATE_INTERVAL) {
    document.getElementById('status-badge').textContent = text;
    lastStatusUpdate = now;
    cachedDisplayStatus = text;
  }
}

function forceUpdateStatusDisplay(text) {
  document.getElementById('status-badge').textContent = text;
  lastStatusUpdate = Date.now();
  cachedDisplayStatus = text;
}

async function pollStatus() {
  if (!polling) return;

  let s;
  try {
    s = await api('/api/status');
  } catch (e) {
    updateStatusDisplay('断开');
    setTimeout(pollStatus, 1500);
    return;
  }

  const badgeText = s.running ? '运行中' : '就绪';
  updateStatusDisplay(badgeText);

  document.getElementById('progress-bar').style.width = s.progress + '%';
  document.getElementById('progress-text').textContent =
    s.running ? `${s.progress}% - ${s.status}` :
    s.error ? `错误: ${s.error}` : s.status;

  if (!s.running) {
    polling = false;
    forceUpdateStatusDisplay('就绪');
    startBgConnectionCheck();
    document.getElementById('analyze-btn').disabled = false;
    document.getElementById('analyze-btn').innerHTML =
      '<i class="material-icons left">play_arrow</i> 开始分析';

    if (!s.error) {
      document.getElementById('progress-title').innerHTML =
        '<i class="material-icons card-header-icon">check_circle</i> 分析完成';
      showDebugLog(s);

      const points = await api('/api/points');
      if (points.content) {
        renderResults(points.content);
      } else if (s.result) {
        renderResults(s.result);
      }
      await loadFiles();
      await loadTimeline();
      await checkChatAvailable();
    } else {
      document.getElementById('progress-title').innerHTML =
        '<i class="material-icons card-header-icon">error</i> 分析失败';
      showDebugLog(s);

    }
    return;
  }

  setTimeout(pollStatus, 1500);
}

// ─── Results rendering ──────────────────────────────
function renderContent(content, prefix) {
  const emptyEl = document.getElementById(prefix + '-empty');
  const contentEl = document.getElementById(prefix + '-content');

  if (!content || !content.trim()) {
    emptyEl.style.display = 'block';
    contentEl.style.display = 'none';
    return;
  }

  emptyEl.style.display = 'none';
  contentEl.style.display = 'block';

  const blocks = content.split(/\n\n(?=\S)/);
  const frag = document.createDocumentFragment();
  let kpCount = 0;

  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;

    const firstNewline = trimmed.indexOf('\n');
    const firstLine = firstNewline === -1 ? trimmed : trimmed.substring(0, firstNewline).trim();
    // Topic line: starts with a letter (not numbered KP, not See also/Related)
    if (/^\d+\./.test(firstLine) || firstLine.startsWith('See also:') || firstLine.startsWith('Related:')) continue;
    const topic = firstLine.replace(/\s+\[.*\]$/, '').trim();

    const subPoints = [];
    let curPoint = null;

    const lines = firstNewline === -1 ? [] : trimmed.substring(firstNewline + 1).split('\n');
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line) continue;

      // Numbered KP: "1. concept text"
      const kpMatch = line.match(/^(\d+)\.\s+(.*)/);
      if (kpMatch) {
        if (curPoint) subPoints.push(curPoint);
        curPoint = { num: kpMatch[1], text: kpMatch[2], detail: '', pitfall: '', scoring: '' };
        continue;
      }

      // Sub-field: "Detail: ...", "Pitfall: ...", "Scoring: ..."
      if (curPoint) {
        if (line.startsWith('Detail:')) curPoint.detail = line.substring(7).trim();
        else if (line.startsWith('Pitfall:')) curPoint.pitfall = line.substring(8).trim();
        else if (line.startsWith('Scoring:')) curPoint.scoring = line.substring(8).trim();
      }
    }
    if (curPoint) subPoints.push(curPoint);

    if (subPoints.length === 0) continue;
    kpCount++;

    const topicDiv = document.createElement('div');
    topicDiv.className = 'result-topic';
    const h6 = document.createElement('h6');
    const icon = document.createElement('i');
    icon.className = 'material-icons left';
    icon.style.cssText = 'font-size:16px;line-height:22px;';
    icon.textContent = 'check_circle';
    h6.appendChild(icon);
    h6.appendChild(document.createTextNode(' ' + topic));
    topicDiv.appendChild(h6);

    for (const sp of subPoints) {
      const pointDiv = document.createElement('div');
      pointDiv.className = 'result-point';
      if (sp.num) {
        const numSpan = document.createElement('span');
        numSpan.className = 'num';
        numSpan.textContent = sp.num + '.';
        pointDiv.appendChild(numSpan);
      }
      const conceptSpan = document.createElement('span');
      conceptSpan.className = 'kp-concept';
      conceptSpan.textContent = sp.text;
      pointDiv.appendChild(conceptSpan);
      if (sp.detail) {
        const d = document.createElement('div');
        d.className = 'kp-detail';
        d.textContent = 'Detail: ' + sp.detail;
        pointDiv.appendChild(d);
      }
      if (sp.pitfall) {
        const p = document.createElement('div');
        p.className = 'kp-pitfall';
        p.textContent = 'Pitfall: ' + sp.pitfall;
        pointDiv.appendChild(p);
      }
      if (sp.scoring) {
        const s = document.createElement('div');
        s.className = 'kp-scoring';
        s.textContent = 'Scoring: ' + sp.scoring;
        pointDiv.appendChild(s);
      }
      topicDiv.appendChild(pointDiv);
    }
    frag.appendChild(topicDiv);
  }

  contentEl.innerHTML = '';
  if (kpCount === 0) {
    contentEl.innerHTML = `<div class="empty-state"><i class="material-icons">info</i><p>未能解析到知识点数据</p></div>`;
  } else {
    contentEl.innerHTML = `<div class="kb-header"><span>共 <strong>${kpCount}</strong> 个知识点</span></div>`;
    contentEl.appendChild(frag);
  }

}

function renderResults(content) { renderContent(content, 'result'); }

async function loadAllResults() {
  const p = await api('/api/points');
  if (p.content) {
    renderResults(p.content);
  } else {
    M.toast({ html: '暂无分析结果', classes: 'blue' });
  }
  loadTimeline();
  checkChatAvailable();
}

// ─── Timeline ───────────────────────────────────────
async function loadTimeline() {
  const entries = await api('/api/timeline');
  if (!entries || entries.length === 0) return;
  const card = document.getElementById('timeline-card');
  card.style.display = 'block';
  const content = document.getElementById('timeline-content');
  content.textContent = entries.map(e => {
    const detail = e.detail ? ` — ${e.detail}` : '';
    return `[${e.time}] ${e.step}${detail}`;
  }).join('\n');
}

// ─── Connection monitor ─────────────────────────────
let bgConnectionInterval = null;

function startBgConnectionCheck() {
  if (bgConnectionInterval) return;
  checkConnection();
  bgConnectionInterval = setInterval(checkConnection, 30000);
}

function stopBgConnectionCheck() {
  if (bgConnectionInterval) {
    clearInterval(bgConnectionInterval);
    bgConnectionInterval = null;
  }
}

// ---- Chat Assistant ----
const CHAT_SESSION = 'exam_analyzer_' + Date.now();

async function checkChatAvailable() {
  const input = document.getElementById('chat-input');
  const btn = document.getElementById('chat-send-btn');
  const hint = document.getElementById('chat-hint');
  try {
    const res = await api('/api/chat/status');
    if (res.available && res.qa_count > 0) {
      input.disabled = false;
      btn.disabled = false;
      input.placeholder = '输入你的问题...';
      hint.textContent = `(${res.qa_count} 条知识点可用)`;
      loadChatHistory();
    } else if (res.available) {
      input.disabled = true;
      btn.disabled = true;
      input.placeholder = '知识库为空，请先运行分析';
      hint.textContent = '(知识库无数据)';
    } else {
      input.disabled = true;
      btn.disabled = true;
      input.placeholder = '知识库尚未建立，请先运行分析';
      hint.textContent = '(Embedding 未就绪或暂无知识库)';
    }
  } catch (e) {
    input.disabled = true;
    btn.disabled = true;
    hint.textContent = '(无法连接)';
  }
}

async function loadChatHistory() {
  try {
    const res = await api(`/api/chat/history?session_id=${CHAT_SESSION}`);
    if (res.history && res.history.length > 0) {
      const container = document.getElementById('chat-messages');
      const empty = container.querySelector('.chat-empty');
      if (empty) empty.remove();
      container.innerHTML = '';
      res.history.forEach(h => {
        if (h.role === 'user') {
          appendChatMessage('user', h.content);
        } else {
          const wrapper = document.createElement('div');
          const answerP = document.createElement('div');
          answerP.style.whiteSpace = 'pre-wrap';
          answerP.textContent = h.content;
          wrapper.appendChild(answerP);
          if (h.sources) {
            try {
              const sources = JSON.parse(h.sources);
              if (sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'chat-sources';
                const strong = document.createElement('strong');
                strong.textContent = '参考知识点:';
                sourcesDiv.appendChild(strong);
                sources.forEach((s, i) => {
                  const line = document.createElement('div');
                  line.textContent = `${i+1}. ${s.topic}: ${(s.question||'').substring(0,80)}...`;
                  sourcesDiv.appendChild(line);
                });
                wrapper.appendChild(sourcesDiv);
              }
            } catch(e) {}
          }
          appendChatMessage('assistant', wrapper, true);
        }
      });
    }
  } catch (e) { /* ignore */ }
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  input.disabled = true;
  document.getElementById('chat-loading').style.display = 'block';
  appendChatMessage('user', question);

  try {
    const res = await api('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question, session_id: CHAT_SESSION}),
    });
    if (res.error) {
      appendChatMessage('assistant', '抱歉：' + res.error);
    } else {
      const wrapper = document.createElement('div');
      // Answer text
      const answerP = document.createElement('div');
      answerP.style.whiteSpace = 'pre-wrap';
      answerP.textContent = res.answer;
      wrapper.appendChild(answerP);
      // Sources
      if (res.sources && res.sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'chat-sources';
        const strong = document.createElement('strong');
        strong.textContent = '参考知识点:';
        sourcesDiv.appendChild(strong);
        res.sources.forEach((s, i) => {
          const line = document.createElement('div');
          const em = document.createElement('em');
          em.textContent = s.topic;
          line.appendChild(document.createTextNode(`${i+1}. `));
          line.appendChild(em);
          line.appendChild(document.createTextNode(`: ${s.question.substring(0,80)}...`));
          sourcesDiv.appendChild(line);
        });
        wrapper.appendChild(sourcesDiv);
      }
      // Follow-up suggestions
      if (res.suggestions && res.suggestions.length > 0) {
        const sugDiv = document.createElement('div');
        sugDiv.style.cssText = 'margin-top:8px;padding-top:6px;border-top:1px solid #e0e0e0;font-size:12px;color:#757575;';
        sugDiv.innerHTML = '<strong>你可能还想问：</strong>';
        res.suggestions.forEach(s => {
          const chip = document.createElement('span');
          chip.style.cssText = 'display:inline-block;margin:3px 6px 3px 0;padding:2px 8px;background:#e8eaf6;border-radius:12px;cursor:pointer;font-size:11px;';
          chip.textContent = s;
          chip.onclick = () => {
            document.getElementById('chat-input').value = s;
            sendChat();
          };
          sugDiv.appendChild(chip);
        });
        wrapper.appendChild(sugDiv);
      }
      appendChatMessage('assistant', wrapper, true);
    }
  } catch (e) {
    appendChatMessage('assistant', '网络错误，请重试');
  }
  input.disabled = false;
  document.getElementById('chat-loading').style.display = 'none';
}

function appendChatMessage(role, content, isDom = false) {
  const container = document.getElementById('chat-messages');
  const empty = container.querySelector('.chat-empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.style.cssText = `margin-bottom:8px;padding:6px 10px;border-radius:8px;font-size:13px;line-height:1.6;max-width:90%;${
    role === 'user'
      ? 'background:#e3f2fd;'
      : 'background:#fff;border:1px solid #e0e0e0;'
  }`;
  if (isDom) {
    div.appendChild(content);
  } else {
    div.style.whiteSpace = 'pre-wrap';
    div.textContent = content;
  }
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

async function checkConnection() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      updateStatusDisplay('就绪');
    } else {
      updateStatusDisplay('断开');
    }
  } catch (e) {
    updateStatusDisplay('断开');
  }
}

// ─── Init ───────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  try {
    await loadConfig();
    await loadFiles();
    await loadTimeline();
    await checkChatAvailable();
    startBgConnectionCheck();
  } catch (e) {
    M.toast({ text: '页面加载异常，请刷新重试', classes: 'red' });
  }
});
