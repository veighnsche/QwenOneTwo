'use strict';
const $ = id => document.getElementById(id);
const PHASE = {loading:'Loading model on GPU', preparing:'Preparing the head', scored:'Thinking score computed', generating:'Generating', prefill:'Reading the prompt', queued:'Starting generation', thinking:'Thinking', closing_thinking:'Closing the thinking section', answer:'Writing the answer', complete:'Complete', cancelled:'Stopped', error:'Needs attention', needs_head:'Head not prepared', unloaded:'Model not loaded'};
let state = null, scoreResult = null, run = null, turn = null, pending = false, connected = false;
let lastHeadHash = '', lastHeadKey = '', lastError = '', lastResult = null;
let lastPhase = 'Waiting for a prompt';
const node = (tag, className, text) => { const n = document.createElement(tag); if (className) n.className = className; if (text !== undefined) n.textContent = text; return n; };
const uuid = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join('');

async function api(path, body) {
  // Relative to the document, including any reverse-proxy mount prefix.
  const response = await fetch(new URL(`api/${path}`, document.baseURI), body === undefined
    ? {cache:'no-store'} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok || result.error && path !== 'state' && !path.startsWith('runs/')) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
function message(text) { $('message-text').textContent = text; $('message').hidden = !text; }
function invalidate() {
  scoreResult = null;
  lastResult = null;
  if (turn && !run) { if (!turn.completed) turn.note.textContent = 'Prompt changed.'; turn = null; }
  updateScore();
  setActivity(null);
}
function controls() {
  const busy = pending || !!state?.busy || !!run;
  const action = !state?.loaded ? 'Load model' : !state.ready ? 'Verify weights' : 'Send prompt';
  $('primary').replaceChildren(document.createTextNode(`${action} `), node('span', '', '↗'));
  $('primary').disabled = !connected || busy || (state?.ready && !$('prompt').value.trim());
  $('prompt').disabled = pending || !!run;
  $('override').disabled = busy;
  $('answer-limit').disabled = busy;
  $('prepare').disabled = !state?.loaded || busy;
  $('prepare').textContent = 'Verify trained weights';
  $('stop').hidden = !run;
  $('stop').disabled = !run || run.stopping;
  $('compose-hint').textContent = pending ? 'Preparing the GPU model.'
    : run ? 'Following the chosen budget. Stop keeps the partial output.'
    : !state?.loaded ? 'The local GPU model needs to be loaded first.'
    : !state?.ready ? 'Trained weights must be present before generation.'
    : 'Send once. Qwen scores the prompt and continues into the answer.';
}
function updateScore() {
  $('selection-label').textContent = scoreResult ? 'MODEL SCORE' : 'AWAITING A PROMPT';
  $('model-score').textContent = scoreResult ? scoreResult.score.toFixed(6) : '—';
  $('score-progress').value = scoreResult?.score ?? 0;
  const s = scoreResult?.score ?? 0, fraction = s * s * (3 - 2 * s);
  $('score-point').setAttribute('cx', 4 + 292 * s);
  $('score-point').setAttribute('cy', 76 - 72 * fraction);
  $('score-point').setAttribute('visibility', scoreResult ? 'visible' : 'hidden');
  $('model-score').title = scoreResult ? `Full score: ${scoreResult.score}` : '';
  $('selected-budget').textContent = scoreResult ? scoreResult.predicted_budget.toLocaleString() : '—';
  $('selection-note').textContent = scoreResult ? `${scoreResult.score_ms} ms · same prompt prefill` : 'One scalar from the prompt prefill.';
  $('provenance').textContent = scoreResult ? JSON.stringify({score:scoreResult.score, raw_score:scoreResult.raw_score, predicted_budget:scoreResult.predicted_budget, ...scoreResult.provenance}, null, 2) : 'Send a prompt to inspect the run’s provenance.';
}

function effectiveBudget() {
  if (!scoreResult) return null;
  const value = $('override').value;
  return value === '' ? scoreResult.predicted_budget : Number(value);
}
function setActivity(result) {
  $('adapter-mode').textContent = result?.adapter === 'decision' ? 'Qev enabled' : 'Native Qwen';
  $('decision-calls').replaceChildren(...(result?.decision_calls || []).map(call => {
    const item = node('div', 'decision-call');
    const value = Number.isFinite(call.score) ? call.score.toFixed(4) : Number.isFinite(call.noul) ? call.noul.toFixed(4) : Number.isInteger(call.choice_index) ? `option ${call.choice_index + 1}` : call.status;
    item.textContent = `${call.type.toUpperCase()} · ${call.phase} ${call.at_token} · ${value}`; return item;
  }));
  if (!result?.decision_calls?.length) $('decision-calls').textContent = 'No decisions yet.';
  const budget = result?.budget ?? effectiveBudget(), used = result?.thinking_tokens || 0;
  const limit = result?.answer_limit ?? Number($('answer-limit').value || 256);
  $('thinking-count').textContent = used.toLocaleString();
  $('effective-budget').textContent = budget == null || !Number.isFinite(budget) ? '—' : budget.toLocaleString();
  $('answer-count').textContent = (result?.answer_tokens || 0).toLocaleString();
  $('answer-allowance').textContent = limit;
  $('budget-progress').max = Math.max(1, budget || 1);
  $('budget-progress').value = used;
  $('remaining').textContent = budget == null ? 'Budget not chosen' : budget === 0 ? (result && !['queued','prefill'].includes(result.phase) ? 'Thinking skipped' : 'Will skip thinking') : `${Math.max(0, budget - used).toLocaleString()} tokens remaining`;
  $('budget-source').textContent = (result?.manual_override ?? ($('override').value !== '')) ? 'manual override' : 'model score';
  $('elapsed').textContent = `${((result?.elapsed_ms || 0) / 1000).toFixed(1)}s`;
  if (result) {
    lastPhase = PHASE[result.phase] || result.phase;
  } else if (pending) lastPhase = 'Preparing the GPU model';
  else if (state?.busy && !run) lastPhase = PHASE[state.phase] || state.phase;
  else lastPhase = 'Waiting for a prompt';
  $('phase').textContent = lastPhase;
  $('phase-symbol').textContent = result?.phase === 'complete' ? '✓' : result?.phase === 'cancelled' ? '■' : run || pending ? '◌' : '○';
  $('live-dot').classList.toggle('active', (!!run && !['complete','cancelled','error'].includes(result?.phase)) || pending);
}
function headDetails() {
  const h = state?.head, key = h?.rows_sha256 || '';
  if (key === lastHeadKey) return;
  lastHeadKey = key;
  $('seeds').textContent = h ? JSON.stringify(h.training || {}, null, 2) : 'Train QwenOneTwo on GPU before loading.';
}
function updateState(next) {
  state = next;
  $('override').max = next.max_thinking_tokens;
  if (lastHeadHash && next.head?.rows_sha256 !== lastHeadHash && !run) invalidate();
  lastHeadHash = next.head?.rows_sha256 || '';
  $('connection').textContent = next.busy ? (PHASE[next.phase] || 'Working') : next.ready ? 'Ready' : next.loaded ? 'Needs head' : next.error ? 'GPU unavailable' : 'Not loaded';
  $('connection-dot').className = next.busy ? 'busy' : next.ready ? 'ready' : next.error ? 'error' : '';
  $('gpu-label').textContent = next.gpu ? next.gpu.replace('NVIDIA GeForce ', '') : 'GPU inference only';
  $('detail-gpu').textContent = next.gpu || 'CUDA GPU required';
  $('head-state').textContent = next.head ? 'Qev + trained controls' : 'Not prepared';
  $('prepare-note').textContent = 'GPU-trained routing, latent feedback and scalar regression.';
  if (next.error && next.error !== lastError && !run && !next.error.includes('cancelled')) { message(next.error); lastError = next.error; }
  headDetails(); controls();
  if (!run && !lastResult) setActivity(null);
}
async function refresh() {
  try {
    const next = await api('state'); connected = true; updateState(next);
    if (run) {
      const active = run, result = await api(`runs/${active.id}`);
      if (run === active) renderRun(result);
    }
  } catch (error) {
    connected = false; $('connection').textContent = 'Reconnecting'; $('connection-dot').className = 'error'; controls();
  }
  setTimeout(refresh, state?.busy || pending || run ? 350 : 1200);
}
function newTurn(prompt) {
  $('empty').hidden = true;
  const section = node('article', 'turn'), label = node('div', 'turn-label', 'YOU'), note = node('div', 'decision-note');
  label.append(node('span', 'rule'));
  note.textContent = 'Reading the prompt and computing its thinking score…';
  section.append(label, node('div', 'turn-prompt', prompt), note);
  $('transcript').append(section);
  return {section, note};
}
function followOutput(callback) {
  const box = $('transcript'), nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 90;
  callback();
  if (nearBottom) box.scrollTop = box.scrollHeight;
}
async function prepare() {
  invalidate(); pending = true; controls(); message('');
  try { await api('prepare', {rebuild:!!state?.head}); }
  catch (error) { message(error.message); }
  finally { pending = false; controls(); }
}
function newRunOutput() {
  const reason = node('details', 'reasoning'); reason.open = true;
  const summary = node('summary', '', 'Thinking'), thought = node('pre'), answerLabel = node('div', 'answer-label', 'QWEN'), answer = node('div', 'answer'), footnote = node('div', 'run-footnote');
  reason.append(summary, thought); turn.section.append(reason, answerLabel, answer, footnote);
  return {reason, summary, thought, answer, footnote};
}
async function start() {
  if (pending || run) return;
  const text = $('prompt').value.trim(); if (!text) return;
  const raw = $('override').value, override = raw === '' ? null : Number(raw), answer = Number($('answer-limit').value);
  if (override !== null && (!Number.isInteger(override) || override < 0 || override > state.max_thinking_tokens)) return message(`Choose a thinking override from 0 to ${state.max_thinking_tokens.toLocaleString()} tokens.`);
  if (!Number.isInteger(answer) || answer < 1 || answer > 512) return message('Choose an answer allowance from 1 to 512 tokens.');
  message(''); lastResult = null; scoreResult = null; updateScore();
  turn = newTurn(text);
  const id = uuid();
  run = {id, ...newRunOutput(), stopping:false};
  controls(); setActivity({phase:'queued', answer_limit:answer, manual_override:override !== null});
  $('transcript').scrollTop = $('transcript').scrollHeight;
  try { await api('generate', {text, run_id:id, budget_override:override, answer_tokens:answer}); }
  catch (error) { renderRun({phase:'error', error:error.message}); }
}
function renderRun(r) {
  if (!run) return;
  const output = run;
  lastResult = r;
  if (Number.isFinite(r.score)) {
    scoreResult = r;
    updateScore();
    turn.note.replaceChildren(
      document.createTextNode(`Thinking score ${r.score.toFixed(6)} → ${r.predicted_budget} tokens${r.manual_override ? `; using override ${r.budget}` : ''}.`),
      node('small', '', `One prompt prefill · same inference cache · ${r.score_ms} ms to compute the score`));
  }
  followOutput(() => {
    output.thought.textContent = r.thinking || (r.budget === 0 ? 'Skipped. The thinking section was closed in the same inference cache.' : '');
    output.summary.textContent = r.budget === 0 ? 'Thinking skipped' : `Thinking · ${r.thinking_tokens || 0} tokens${r.thinking_end === 'natural_close' ? ' · finished early' : r.thinking_end === 'budget_exhausted' ? ' · budget reached' : ''}`;
    output.answer.textContent = r.answer || '';
    if (r.phase === 'complete' || r.phase === 'cancelled' || r.phase === 'error') {
      const ending = r.answer_end === 'answer_limit' ? 'Answer allowance reached.' : r.answer_end === 'eos' ? 'Answer complete.' : '';
      output.footnote.textContent = r.error || `${r.prefill_count} prompt prefill · ${r.cache_continuous ? 'same cache' : 'cache error'} · ${r.thinking_tokens} thinking · ${r.answer_tokens} answer · ${r.sampled_control_tokens} sampled control · ${r.forced_control_tokens} injected control tokens · ${((r.elapsed_ms || 0) / 1000).toFixed(1)}s. ${ending}`;
      if (r.budget === 0) output.reason.open = false;
    }
  });
  setActivity(r); lastResult = r;
  if (['complete','cancelled','error'].includes(r.phase)) {
    if (turn) turn.completed = true;
    if (r.phase === 'error') message(r.error || 'Generation failed.');
    run = null; controls();
  }
}
async function stop() {
  if (!run || run.stopping) return;
  run.stopping = true; controls();
  try { await api('stop', {operation_id:run.id}); }
  catch (error) { message(error.message); if (run) run.stopping = false; controls(); }
}
async function primary(event) {
  event?.preventDefault();
  if ($('primary').disabled) return;
  if (!state?.loaded) {
    pending = true; controls(); message('');
    try { await api('load', {}); } catch (error) { message(error.message); }
    finally { pending = false; controls(); }
  } else if (!state.ready) await prepare();
  else await start();
}
$('composer').addEventListener('submit', primary);
$('prompt').addEventListener('input', () => { invalidate(); controls(); });
$('prompt').addEventListener('keydown', event => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') primary(event); });
$('override').addEventListener('input', () => { lastResult = null; setActivity(null); });
$('answer-limit').addEventListener('input', () => { lastResult = null; setActivity(null); });
$('stop').addEventListener('click', stop);
$('prepare').addEventListener('click', prepare);
$('message-dismiss').addEventListener('click', () => message(''));
for (const id of ['about-open','details-open']) $(id).addEventListener('click', () => $('details').showModal());
$('about-close').addEventListener('click', () => $('details').close());
for (const sample of document.querySelectorAll('[data-prompt]')) sample.addEventListener('click', () => { if (run || pending) return; $('prompt').value = sample.dataset.prompt; invalidate(); controls(); $('prompt').focus(); });
refresh();
