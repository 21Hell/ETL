async function search(text, top_k) {
  const resp = await fetch('/api/search', { method: 'POST', headers: {'content-type':'application/json'}, body: JSON.stringify({text, top_k}) });
  if (!resp.ok) throw new Error('Search failed: ' + resp.status);
  return resp.json();
}

async function getPoint(id) {
  const resp = await fetch('/api/point/' + encodeURIComponent(id));
  if (!resp.ok) throw new Error('Point fetch failed');
  return resp.json();
}

async function chat(text, top_k) {
  const resp = await fetch('/api/chat', { method: 'POST', headers: {'content-type':'application/json'}, body: JSON.stringify({text, top_k}) });
  if (!resp.ok) throw new Error('Chat failed');
  return resp.json();
}

function el(tag, props={}, ...children) {
  const e = document.createElement(tag);
  for (const k of Object.keys(props||{})) e.setAttribute(k, props[k]);
  for (const c of children) {
    if (typeof c === 'string') e.appendChild(document.createTextNode(c)); else e.appendChild(c);
  }
  return e;
}

const btnSearch = document.getElementById('btnSearch');
const queryInput = document.getElementById('query');
const resultsDiv = document.getElementById('results');
const detailList = document.getElementById('detailList');
const topkInput = document.getElementById('topk');
const btnChat = document.getElementById('btnChat');
const chatText = document.getElementById('chatText');
const respuestaDiv = document.getElementById('respuesta');
const contextsDiv = document.getElementById('chatContexts');
const rawResultsPre = document.getElementById('rawResults');
const uiErrorDiv = document.getElementById('uiError');
const llmSpinner = document.getElementById('llmSpinner');
const llmText = document.getElementById('llmText');
const llmRetry = document.getElementById('llmRetry');

async function fetchLlmStatus() {
  try {
    const resp = await fetch('/api/llm_status');
    if (!resp.ok) throw new Error('LLM status fetch failed');
    const j = await resp.json();
    // update UI
    if (j.available) {
      llmSpinner.style.background = '#2b6af6';
      llmSpinner.style.animation = 'spin 1s linear infinite';
      const models = (j.models && j.models.length) ? j.models.join(', ') : (j.ollama_url || 'local');
      llmText.innerText = `LLM: available (${models})`;
    } else {
      llmSpinner.style.background = '#d9534f';
      llmSpinner.style.animation = 'none';
      llmText.innerText = 'LLM: unavailable';
    }
  } catch (e) {
    llmSpinner.style.background = '#d9534f';
    llmSpinner.style.animation = 'none';
    llmText.innerText = 'LLM: error';
  }
}

// wire retry button
llmRetry.addEventListener('click', async () => {
  llmText.innerText = 'LLM: checking…';
  llmSpinner.style.background = '#ddd';
  llmSpinner.style.animation = 'none';
  await fetchLlmStatus();
});

// start polling
fetchLlmStatus();
setInterval(fetchLlmStatus, 10000);

// simple spinner keyframes injection
const s = document.createElement('style');
s.innerHTML = `@keyframes spin { from { transform: rotate(0deg);} to { transform: rotate(360deg);} }`;
document.head.appendChild(s);

btnSearch.addEventListener('click', async () => {
  const q = queryInput.value.trim();
  if (!q) return;
  uiErrorDiv.innerText = '';
  btnSearch.disabled = true;
  resultsDiv.innerHTML = 'Buscando...';
  try {
    const topk = Number.parseInt(topkInput.value || '5') || 5;
    const res = await search(q, topk);
    resultsDiv.innerHTML = '';
    res.results.forEach(r => {
      const d = el('div', {class:'result'});
      const title = el('div', {}, r.payload && r.payload.preview ? r.payload.preview.slice(0,120) : (r.payload && r.payload.source ? r.payload.source : 'result'));
  // show original score and optional rerank score
  const scoreText = (r.score||'') && (typeof r.score === 'number') ? r.score.toFixed(3) : (r.score || '');
  const rer = (r.rerank_score !== undefined) ? (typeof r.rerank_score === 'number' ? r.rerank_score.toFixed(3) : r.rerank_score) : null;
  const sc = el('span', {class:'score'}, rer ? `${scoreText} · rerank ${rer}` : scoreText);
      title.appendChild(sc);
      d.appendChild(title);
      d.addEventListener('click', async () => {
        detailList.innerHTML = '';
        detailList.appendChild(el('div', {}, 'Cargando...'));
        try {
          const p = await getPoint(r.id);
          detailList.innerHTML = '';
          detailList.appendChild(el('h4', {}, p.id || ''));
          const meta = el('div', {}, `Fuente: ${p.payload.source || ''} ${p.payload.page ? '· Página: ' + p.payload.page : ''}`);
          detailList.appendChild(meta);
          if (p.payload.preview) {
            detailList.appendChild(el('div', {style:'margin-top:8px;font-weight:600'}, 'Preview'));
            detailList.appendChild(el('div', {class:'context'}, p.payload.preview));
          }
          if (p.payload.text) {
            detailList.appendChild(el('div', {style:'margin-top:8px;font-weight:600'}, 'Texto (chunk)'));
            detailList.appendChild(el('pre', {class:'context'}, p.payload.text));
          } else {
            detailList.appendChild(el('pre', {class:'context'}, JSON.stringify(p.payload, null, 2)));
          }
        } catch (err) {
          // If fetching point fails, show the raw payload from the search result instead
          detailList.innerHTML = '';
          detailList.appendChild(el('h4', {}, r.id || ''));
          detailList.appendChild(el('div', {style:'margin-top:6px;color:#a33'}, 'No se pudo obtener payload del servidor; mostrando payload retornado por la búsqueda.'));
          try {
            detailList.appendChild(el('pre', {class:'context'}, JSON.stringify(r.payload || {}, null, 2)));
          } catch (e) {
            detailList.appendChild(el('div', {}, 'No hay payload disponible.'));
          }
        }
      });
      resultsDiv.appendChild(d);
    });
  } catch (err) {
    resultsDiv.innerHTML = 'Error: ' + err;
    uiErrorDiv.innerText = String(err);
  }
  finally {
    btnSearch.disabled = false;
  }
});

btnChat.addEventListener('click', async () => {
  uiErrorDiv.innerText = '';
  const q = chatText.value.trim();
  if (!q) return;
  btnChat.disabled = true;
  respuestaDiv.innerText = 'Consultando...';
  contextsDiv.innerText = '';
  rawResultsPre.innerText = '';
  try {
    const topk = Number.parseInt(topkInput.value || '5') || 5;
    const r = await chat(q, topk);
    // Respuesta principal
    respuestaDiv.innerText = r.answer || '(sin respuesta)';

    // Contextos: mostrar previews y metadatos
    if (r.contexts && r.contexts.length) {
      contextsDiv.innerHTML = '';
      r.contexts.forEach((c, i) => {
        const header = el('div', {style:'font-weight:600;margin-top:6px'}, `${i+1}. ${c.source || c.title || 'contexto'}`);
        const preview = el('div', {class:'context'}, c.preview || c.text || '');
        contextsDiv.appendChild(header);
        contextsDiv.appendChild(preview);
      });
    } else {
      contextsDiv.innerText = 'No hay contextos.';
    }

    // Raw results as pretty JSON for debugging
    try {
      rawResultsPre.innerText = JSON.stringify(r.raw_results || r.rawResults || [], null, 2);
    } catch (e) {
      rawResultsPre.innerText = String(r.raw_results || r.rawResults || '');
    }
  } catch (err) {
    respuestaDiv.innerText = 'Error: ' + err;
    contextsDiv.innerText = '';
    rawResultsPre.innerText = '';
    uiErrorDiv.innerText = String(err);
  } finally {
    btnChat.disabled = false;
  }
});

// allow Enter in search box (Enter submits)
queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    btnSearch.click();
  }
});

// chat textarea: Enter submits, Shift+Enter inserts newline
chatText.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    btnChat.click();
  }
});
