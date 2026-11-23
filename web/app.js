/* In-band telemetry console.
 *
 * Vanilla JS + canvas: no build step, no dependencies.  Every chart is drawn
 * from an API payload, so the numbers on screen are the numbers the collector
 * produced -- nothing is recomputed in the browser.
 */

const $ = (sel) => document.querySelector(sel);
const state = {
  topo: null,
  slow: [],
  selectedTxn: null,
  timeUs: 1000000,
  polling: null,
};

const fmt = (v, digits = 1) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : Number(v).toFixed(digits);
const usToMs = (v) => (v === null || v === undefined) ? '—' : (v / 1000).toFixed(2);
const pct = (v) => (v === null || v === undefined || Number.isNaN(v))
  ? '—' : (v * 100).toFixed(0) + '%';

/* ------------------------------------------------------------------ */
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.error || r.statusText);
  }
  return r.json();
}

function setStatus(text, cls) {
  const el = $('#status-pill');
  el.textContent = text;
  el.className = 'pill' + (cls ? ' ' + cls : '');
}

/* ------------------------------------------------------------------ */
async function boot() {
  setStatus('正在加载拓扑…');
  state.topo = await api('/api/topology');
  drawTopology({});
  await waitReady();
  await loadAll();
  setStatus('就绪', 'ok');
  // refresh the heat map while the user drags the time slider
  $('#time-slider').addEventListener('input', debounce(onTimeChange, 90));
  $('#only-captured').addEventListener('change', renderSlowTable);
  $('#rebuild').addEventListener('click', rebuild);
}

function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

async function waitReady() {
  for (let i = 0; i < 400; i++) {
    const s = await api('/api/status');
    if (s.ready) return s;
    if (s.error) throw new Error(s.error);
    setStatus('构建仿真中… ' + (i * 0.4).toFixed(0) + 's');
    await new Promise((r) => setTimeout(r, 400));
  }
  throw new Error('仿真超时');
}

async function loadAll() {
  const [overview, evaluation, sampling] = await Promise.all([
    api('/api/overview'), api('/api/evaluation'), api('/api/sampling'),
  ]);
  renderKpis(overview, evaluation);
  renderEvaluation(evaluation);
  renderSampling(sampling);
  await loadQueueSeries();
  await loadSlow();
  const dur = overview.duration_us;
  const slider = $('#time-slider');
  slider.max = String(dur);
  slider.step = String(Math.round(dur / 100));
  state.timeUs = Number(slider.value);
  await onTimeChange();
}

/* ------------------------------------------------------------------ */
function renderKpis(o, ev) {
  const h = o.evaluation_headline;
  const kpis = [
    { label: '慢请求（网络原因）', value: (h.net_slow_transactions || 0),
      hint: '端到端 ≥ 5 ms 共 ' + (o.slow_transactions || 0) + ' 条',
      cls: 'warn' },
    { label: 'P99 端到端时延', value: fmt(o.latency_ms.p99, 2) + ' ms',
      hint: 'P999 ' + fmt(o.latency_ms.p999, 1) + ' ms' },
    { label: '自适应采样捕获率', value: pct(h.adaptive_capture),
      hint: '同数据量固定采样 ' + pct(h.fixed_capture), cls: 'good' },
    { label: '归因准确率（网络原因）', value: pct(h.attribution_accuracy_net),
      hint: '全部慢请求 ' + pct(h.attribution_accuracy), cls: 'good' },
    { label: '逐跳时延误差', value: fmt(h.delay_mae_us, 1) + ' µs',
      hint: '平均绝对误差' },
    { label: '收集器数据量', value: fmt(ev.adaptive.volume_mbps, 2) + ' Mb/s',
      hint: '固定采样同量 ' + fmt(ev.fixed.volume_mbps, 2) + ' Mb/s' },
  ];
  $('#kpis').innerHTML = kpis.map((k) => `
    <div class="kpi">
      <div class="label">${k.label}</div>
      <div class="value ${k.cls || ''}">${k.value}</div>
      <div class="hint">${k.hint}</div>
    </div>`).join('');
}

/* ------------------------------------------------------------------ */
/* topology: switches placed by the backend layout, coloured by queue  */
function drawTopology(queueBySwitch) {
  const svg = $('#topo');
  const t = state.topo;
  if (!t) return;
  const NS = 'http://www.w3.org/2000/svg';
  svg.innerHTML = '';
  const gLinks = document.createElementNS(NS, 'g');
  const gNodes = document.createElementNS(NS, 'g');
  svg.append(gLinks, gNodes);

  const pos = new Map(t.switches.map((s) => [s.idx, s]));
  for (const l of t.links) {
    const a = pos.get(l.src), b = pos.get(l.dst);
    if (!a || !b) continue;
    const line = document.createElementNS(NS, 'line');
    line.setAttribute('x1', a.x); line.setAttribute('y1', a.y);
    line.setAttribute('x2', b.x); line.setAttribute('y2', b.y);
    line.setAttribute('class', 'link-line');
    gLinks.appendChild(line);
  }

  const colorFor = (qus) => {
    if (qus === undefined || qus === null) return '#2b3a4a';
    const v = Math.min(1, qus / 4000);
    // blue -> green -> amber -> red
    const stops = [
      [0.0, [43, 108, 176]],
      [0.35, [47, 133, 90]],
      [0.7, [183, 121, 31]],
      [1.0, [197, 48, 48]],
    ];
    let a = stops[0], b = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
      if (v >= stops[i][0] && v <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; break; }
    }
    const f = (v - a[0]) / Math.max(1e-9, b[0] - a[0]);
    const c = a[1].map((x, i) => Math.round(x + f * (b[1][i] - x)));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };

  for (const s of t.switches) {
    const g = document.createElementNS(NS, 'g');
    g.setAttribute('class', 'sw-node');
    g.setAttribute('data-idx', s.idx);
    const r = s.kind === 'core' || s.kind === 'super' ? 9
      : s.kind === 'spine' ? 8 : 7;
    const circ = document.createElementNS(NS, 'circle');
    circ.setAttribute('cx', s.x); circ.setAttribute('cy', s.y);
    circ.setAttribute('r', r);
    circ.setAttribute('fill', colorFor(queueBySwitch[s.id]));
    g.appendChild(circ);
    const label = document.createElementNS(NS, 'text');
    label.setAttribute('x', s.x); label.setAttribute('y', s.y - r - 4);
    label.setAttribute('text-anchor', 'middle');
    label.textContent = s.id;
    g.appendChild(label);
    g.addEventListener('click', () => focusSwitch(s.id));
    gNodes.appendChild(g);
  }
}

async function onTimeChange() {
  const v = Number($('#time-slider').value);
  state.timeUs = v;
  $('#time-label').textContent = 't = ' + (v / 1e6).toFixed(3) + ' s';
  const h = await api('/api/queues?t=' + Math.round(v));
  state.lastHeat = h.switches || {};
  drawTopology(state.lastHeat);
  const top = (h.ports || []).slice(0, 6);
  // annotate the data point on the queue chart is unnecessary; the marker
  // already sits at the same x position
  if (top.length) {
    $('#time-label').textContent += ' · 最深 ' + fmt(top[0].queue_us, 0) + ' µs @ ' + top[0].switch + '/' + top[0].port;
  }
}

async function focusSwitch(swId) {
  const h = await api('/api/queues?t=' + Math.round(state.timeUs));
  const rows = (h.ports || []).filter((p) => p.switch === swId);
  if (!rows.length) return;
  $('#time-label').textContent = swId + ' 端口最深队列 ' +
    fmt(rows[0].queue_us, 1) + ' µs（端口 ' + rows[0].port + '）';
}

/* ------------------------------------------------------------------ */
async function loadQueueSeries() {
  const d = await api('/api/timeseries');
  drawArea('#queue-chart', d.t_us, d.qdepth_bytes.map((v) => v / 125), {
    color: '#4aa3df', yLabel: 'µs', xMax: d.t_us[d.t_us.length - 1],
    marker: state.timeUs,
  });
}

/* ------------------------------------------------------------------ */
async function loadSlow() {
  const d = await api('/api/slow');
  state.slow = d.slow || [];
  renderSlowTable();
  // deep link: /?txn=997 opens that attribution straight away
  const want = new URLSearchParams(location.search).get('txn');
  if (want) selectTxn(Number(want));
}

function renderSlowTable() {
  const only = $('#only-captured').checked;
  const rows = state.slow.filter((r) => !only || r.observed);
  $('#slow-count').textContent = rows.length + ' / ' + state.slow.length + ' 条已捕获';
  const tb = $('#slow-table tbody');
  if (!rows.length) {
    tb.innerHTML = '<tr><td colspan="7" class="muted">没有已捕获的慢请求——'
      + '提高预算或采样比例再试</td></tr>';
    return;
  }
  tb.innerHTML = rows.map((r) => {
    const ok = r.pred_dominant_hop !== null
      && r.pred_dominant_hop === r.true_dominant_hop;
    const cls = r.observed ? (ok ? 'good' : 'bad') : 'dim';
    const verdict = !r.observed ? '未捕获' : (ok ? '正确' : '偏差');
    return `<tr data-txn="${r.txn_id}" class="${r.txn_id === state.selectedTxn ? 'sel' : ''}">
      <td>#${r.txn_id}</td>
      <td class="name">${r.cls}</td>
      <td class="num">${usToMs(r.latency_us)}</td>
      <td class="num">${usToMs(r.net_us)}</td>
      <td class="num">${r.pred_dominant_hop === null ? '—'
        : '跳 ' + r.pred_dominant_hop + ' <span class="muted">' + (r.pred_switch || '') + '</span>'}</td>
      <td class="num muted">跳 ${r.true_dominant_hop}</td>
      <td class="num">${r.pred_queue_us === null ? '—' : fmt(r.pred_queue_us, 0) + ' µs'}</td>
      <td><span class="tag ${cls}">${verdict}</span></td>
    </tr>`;
  }).join('');
  tb.querySelectorAll('tr[data-txn]').forEach((tr) => {
    tr.addEventListener('click', () => selectTxn(Number(tr.dataset.txn)));
  });
}

/* ------------------------------------------------------------------ */
async function selectTxn(txnId) {
  state.selectedTxn = txnId;
  renderSlowTable();
  const url = new URL(location.href);
  url.searchParams.set('txn', String(txnId));
  history.replaceState(null, '', url);
  const d = await api('/api/attribute?txn_id=' + txnId);
  $('#detail-empty').classList.add('hidden');
  $('#detail-body').classList.remove('hidden');
  $('#detail-txn').textContent = '事务 #' + txnId + ' · ' + d.cls + ' · ' + d.src + ' → ' + d.dst;
  const truth = d.truth || {};
  const pred = d.dominant_hop;
  const hit = pred && truth.dominant_hop === pred.hop;
  $('#detail-verdict').innerHTML = `<span class="tag ${hit ? 'good' : 'bad'}">`
    + (hit ? '归因正确' : '归因偏差') + '</span>';

  // path with the culprit highlighted
  $('#detail-path').innerHTML = (d.hop_breakdown || []).map((h, i) => {
    const bad = pred && pred.hop === h.hop;
    const name = bad ? `<span class="bad-hop">${h.switch}</span>` : `<b>${h.switch}</b>`;
    const sep = i ? ' → ' : '';
    return sep + name + `<span class="muted">/${h.out_port}</span>`;
  }).join('');

  drawHopChart(d);
  renderHopTable(d);
  renderCoFlows(d);
  const tl = [];
  if (d.txn_start_us !== undefined) {
    tl.push('事务起始 ' + (d.txn_start_us / 1e6).toFixed(4) + ' s');
    tl.push('服务器处理 ' + fmt(d.server_thinking_us, 1) + ' µs');
    tl.push('端到端 ' + fmt(truth.latency_us / 1000, 2) + ' ms，其中网络 '
      + fmt(truth.net_us / 1000, 2) + ' ms');
    tl.push('真实最拥塞跳 ' + truth.dominant_hop + '，排队 '
      + fmt(truth.dominant_queue_us, 0) + ' µs');
  }
  tl.push('遥测缺失跳数 ' + d.missing_hops + '（元数据长度受限被截断）');
  if (d.txn_summary && d.txn_summary.five_tuple) {
    tl.push('五元组 ' + d.txn_summary.five_tuple.join(' : '));
  }
  $('#timeline').innerHTML = tl.map((x) => '<div>· ' + x + '</div>').join('');
  if (d.packet_timeline && d.packet_timeline.t0_us !== undefined) {
    drawTimeline(d.packet_timeline);
  }
}

function renderHopTable(d) {
  const pred = d.dominant_hop;
  $('#hop-table tbody').innerHTML = (d.hop_breakdown || []).map((h) => {
    const bad = pred && pred.hop === h.hop;
    const state = !h.observed ? '<span class="tag dim">截断</span>'
      : (h.retx ? '<span class="tag warn">重传</span>' : '<span class="tag good">观测</span>');
    return `<tr class="${bad ? 'sel' : ''}">
      <td>${h.hop}</td><td class="name">${h.switch}</td>
      <td>${h.in_port} / ${h.out_port}</td>
      <td class="num">${h.mean_queue_us === null || h.mean_queue_us === undefined
        ? (h.queue_us === null ? '—' : fmt(h.queue_us, 1))
        : fmt(h.mean_queue_us, 1)}</td>
      <td class="num">${h.residence_us === null ? '—' : fmt(h.residence_us, 1)}</td>
      <td class="num">${h.qdepth_bytes === null ? '—' : fmt(h.qdepth_bytes, 0)}</td>
      <td>${state}</td></tr>`;
  }).join('');
}

function renderCoFlows(d) {
  const rows = d.co_flows || [];
  const tb = $('#coflow-table tbody');
  if (!rows.length) {
    tb.innerHTML = '<tr><td colspan="4" class="muted">窗口内没有其他观测到的流'
      + '（可能它们的遥测未被采样）</td></tr>';
  } else {
    tb.innerHTML = rows.map((r) => `<tr>
      <td class="name">${r.cls}</td>
      <td class="name">${r.src} → ${r.dst}</td>
      <td class="num">${r.packets}</td>
      <td class="num">${fmt(r.queue_sum_us, 0)} µs</td></tr>`).join('');
  }
  const trig = d.trigger_events || [];
  $('#trigger-list').innerHTML = trig.length
    ? trig.map((t) => `<span class="chip">${t.switch}/${t.port} @ `
      + `${(t.t_us / 1e6).toFixed(3)}s (${t.dt_us > 0 ? '+' : ''}${fmt(t.dt_us / 1000, 1)}ms)</span>`).join('')
    : '<span class="muted" style="font-size:12px">该时刻附近没有触发器</span>';
}

/* ------------------------------------------------------------------ */
function drawHopChart(d) {
  const c = $('#hop-chart');
  const ctx = setupCanvas(c);
  const W = c.clientWidth, H = c.clientHeight;
  const hops = d.hop_breakdown || [];
  const pad = { l: 66, r: 16, t: 28, b: 32 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const maxQ = Math.max(1, ...hops.map((h) => (h.mean_queue_us ?? h.queue_us) || 0));
  ctx.clearRect(0, 0, W, H);
  ctx.font = '11px ui-monospace, monospace';

  const barW = Math.min(70, iw / Math.max(1, hops.length) - 10);
  const step = iw / Math.max(1, hops.length);
  const pred = d.dominant_hop;

  // grid
  ctx.strokeStyle = '#212c3a';
  ctx.fillStyle = '#7f8fa3';
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + ih - (i / 4) * ih;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    const lab = Math.round((maxQ * i) / 4) + ' µs';
    ctx.fillText(lab, pad.l - 8 - ctx.measureText(lab).width, y + 4);
  }

  hops.forEach((h, i) => {
    const x = pad.l + i * step + (step - barW) / 2;
    const qv = (h.mean_queue_us === null || h.mean_queue_us === undefined)
      ? h.queue_us : h.mean_queue_us;
    if (qv === null || qv === undefined) {
      ctx.fillStyle = '#2a3442';
      ctx.fillRect(x, pad.t + ih - 3, barW, 3);
      ctx.fillStyle = '#5a6b7e';
      ctx.fillText('?', x + barW / 2 - 3, pad.t + ih - 9);
    } else {
      const bh = Math.max(1.5, (qv / maxQ) * ih);
      const hit = pred && pred.hop === h.hop;
      ctx.fillStyle = hit ? '#e05252' : (h.retx ? '#e0a13a' : '#4aa3df');
      ctx.fillRect(x, pad.t + ih - bh, barW, bh);
      ctx.fillStyle = hit ? '#ff9a9a' : '#9fb2c6';
      const lab = fmt(qv, qv >= 100 ? 0 : 1);
      ctx.fillText(lab, Math.min(W - pad.r - ctx.measureText(lab).width,
        Math.max(pad.l, x + barW / 2 - ctx.measureText(lab).width / 2)),
        Math.max(12, pad.t + ih - bh - 5));
    }
    ctx.fillStyle = '#7f8fa3';
    ctx.fillText(String(h.hop), pad.l + i * step + step / 2 - 3, H - 11);
  });

}

/* ------------------------------------------------------------------ */
/* one transaction: when each response packet left the server and arrived */
function drawTimeline(tl) {
  const c = $('#timeline-chart');
  const ctx = setupCanvas(c);
  const W = c.clientWidth, H = c.clientHeight;
  const pad = { l: 8, r: 8, t: 22, b: 18 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const n = Math.max(1, tl.send_us.length);
  const maxT = Math.max(...tl.deliver_us, 1);
  ctx.clearRect(0, 0, W, H);
  ctx.font = '10.5px ui-monospace, monospace';
  const X = (v) => pad.l + (v / maxT) * iw;
  ctx.fillStyle = '#7f8fa3';
  ctx.fillText('0 s', pad.l, H - 5);
  const lab = (maxT / 1000).toFixed(1) + ' ms';
  ctx.fillText(lab, W - pad.r - ctx.measureText(lab).width, H - 5);
  // one horizontal lane per packet, send -> deliver
  const laneH = Math.max(1.5, ih / n);
  tl.send_us.forEach((s, i) => {
    const y = pad.t + (i + 0.5) * laneH;
    const x1 = X(s), x2 = X(tl.deliver_us[i]);
    ctx.strokeStyle = '#2a3b4e'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.strokeStyle = tl.queue_us[i] > 2000 ? '#e05252'
      : tl.queue_us[i] > 200 ? '#e0a13a' : '#4aa3df';
    ctx.lineWidth = Math.max(1.6, laneH * 0.55);
    ctx.beginPath(); ctx.moveTo(x1, y); ctx.lineTo(Math.max(x2, x1 + 1), y); ctx.stroke();
  });
  ctx.fillStyle = '#7f8fa3';
  ctx.fillText(tl.send_us.length + ' 个响应包 · 左端发出 → 右端到达', pad.l, 12);
}

function renderSampling(d) {
  drawLines('#sampling-chart', d.t_us, [
    { data: d.adaptive_rate.map((v) => v * 100), color: '#48bb78', label: '自适应' },
    { data: d.fixed_rate.map((v) => v * 100), color: '#7f8fa3', label: '固定' },
    { data: d.boost_share.map((v) => v * 100), color: '#e0a13a', label: '触发器全量' },
  ], { yLabel: '%', xMax: d.t_us[d.t_us.length - 1], triggers: d.triggers,
       tUs: d.t_us });
  const ntrig = d.triggers.reduce((a, b) => a + b, 0);
  const budget = (d.budget_bytes_per_s || 0) / 1e6;
  const actual = (d.actual_bytes_per_s || 0) / 1e6;
  $('#sampling-summary').innerHTML =
    `基线 <b>${(d.base_prob * 100).toFixed(2)}%</b> · 规则 ${d.n_rules} 条 · 触发器 ${ntrig} 次<br>`
    + `预算 ${budget.toFixed(2)} → 实际 <b>${actual.toFixed(2)}</b> MB/s`;
  const rows = (d.rules || []).slice(0, 8)
    .map((r) => `${r.cls} ${r.path.split('-').slice(0, 2).join('-')}… → ${(r.p * 100).toFixed(1)}%`);
  $('#sampling-summary').title = '采样规则（部分）:\n' + rows.join('\n');
}

/* ------------------------------------------------------------------ */
function renderEvaluation(ev) {
  const a = ev.adaptive, f = ev.fixed;
  $('#eval-sub').textContent = `${ev.scenario.packets.toLocaleString()} 包 · `
    + `${ev.scenario.slow_transactions} 慢请求 · 元数据上限 `
    + `${ev.scenario.int_budget_bytes} B（${ev.scenario.max_records} 跳）`;
  const cards = [
    {
      t: '慢请求捕获率（同数据量）',
      big: pct(a.capture.capture_rate),
      rows: [
        ['自适应采样', pct(a.capture.capture_rate) + ` (${a.capture.captured}/${a.capture.slow_txns})`],
        ['固定比例采样', pct(f.capture.capture_rate) + ` (${f.capture.captured}/${f.capture.slow_txns})`],
        ['固定采样概率', (f.params.base_prob * 100).toFixed(2) + '%'],
        ['两者数据量（实测）', fmt(a.volume_mbps, 2) + ' Mb/s'],
      ],
    },
    {
      t: '慢请求归因准确率',
      big: pct(a.attribution_accuracy),
      rows: [
        ['全部已观测慢请求', pct(a.attribution_accuracy) + ` (n=${a.attribution_txns})`],
        ['排队 >1 ms 的慢请求', pct(a.attribution_accuracy_big) + ` (n=${a.attribution_txns_big})`],
        ['网络原因慢请求（≥5 ms）', pct(a.attribution_accuracy_net) + ` (n=${a.attribution_txns_net})`],
        ['固定采样：网络原因', pct(f.attribution_accuracy_net)],
        ['截断造成的记录丢失', pct(a.collector.truncation_loss)],
      ],
    },
    {
      t: '逐跳排队时延估计误差',
      big: fmt(a.delay_mae_us, 1) + ' µs',
      rows: [
        ['平均绝对误差', fmt(a.delay_mae_us, 1) + ' µs'],
        ['P95 绝对误差', fmt(a.delay_p95_abs_us, 1) + ' µs'],
        ['样本数', a.delay_samples.toLocaleString()],
        ['时钟偏差中位误差', fmt(a.clock_median_abs_error_us, 2) + ' µs'],
      ],
    },
  ];
  $('#eval-cards').innerHTML = cards.map((c) => `
    <div class="eval-card">
      <div class="t">${c.t}</div>
      <div class="big">${c.big}</div>
      ${c.rows.map((r) => `<div class="row"><span>${r[0]}</span><span>${r[1]}</span></div>`).join('')}
    </div>`).join('');

  const bh = a.per_hop || [];
  drawGroupedBars('#delay-chart', bh);
  renderVolume(ev);
}

function renderVolume(ev) {
  const a = ev.adaptive;
  const v = a.volume || {};
  const aPackets = a.sampled_packets;
  const parts = [
    ['跳记录 12 B × N', v.record_bytes_per_s || 0, '#4aa3df'],
    ['基础头 24 B × N', v.header_bytes_per_s || 0, '#2f855a'],
    ['收集器索引 8 B × N', v.index_bytes_per_s || 0, '#b7791f'],
  ];
  const records = v.records || 0;
  const total = parts.reduce((s, p) => s + p[1], 0) || 1;
  $('#volume-breakdown').innerHTML = `
    <div class="bar">${parts.map((p) =>
      `<i style="width:${(p[1] / total) * 100}%;background:${p[2]}"></i>`).join('')}</div>
    ${parts.map((p) => `<div class="row" style="display:flex;justify-content:space-between;
        font-family:var(--mono);font-size:12px;color:#9fb2c6;padding:2px 0">
        <span><i style="display:inline-block;width:8px;height:8px;border-radius:2px;
          background:${p[2]};margin-right:6px"></i>${p[0]}</span>
        <span style="color:var(--text)">${(p[1] / 1e6).toFixed(2)} MB/s</span></div>`).join('')}
    <div class="row" style="display:flex;justify-content:space-between;
        font-family:var(--mono);font-size:12px;padding-top:7px;border-top:1px solid var(--line);margin-top:5px">
      <span>合计</span><span>${(total / 1e6).toFixed(2)} MB/s</span></div>
    <div class="row" style="display:flex;justify-content:space-between;
        font-family:var(--mono);font-size:12px;color:var(--muted)">
      <span>采样报文数</span><span>${aPackets.toLocaleString()}</span></div>
    <div class="row" style="display:flex;justify-content:space-between;
        font-family:var(--mono);font-size:12px;color:var(--muted)">
      <span>保留的跳记录数</span><span>${records.toLocaleString()}</span></div>`;
}

/* ------------------------------------------------------------------ */
/* canvas helpers                                                      */
function setupCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth || 600;
  const h = Number(c.getAttribute('height')) || 260;
  c.width = w * dpr;
  c.height = h * dpr;
  c.style.height = h + 'px';
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clientHeight = h;
  return ctx;
}

function drawArea(sel, xs, ys, opt) {
  const c = $(sel);
  const ctx = setupCanvas(c);
  const W = c.clientWidth, H = c.clientHeight;
  const pad = { l: 50, r: 14, t: 14, b: 26 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const xMax = opt.xMax || xs[xs.length - 1];
  const yMax = Math.max(1, ...ys);
  ctx.clearRect(0, 0, W, H);
  ctx.font = '11px ui-monospace, monospace';
  const X = (v) => pad.l + (v / xMax) * iw;
  const Y = (v) => pad.t + ih - (v / yMax) * ih;

  ctx.strokeStyle = '#212c3a'; ctx.fillStyle = '#7f8fa3';
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + ih - (i / 4) * ih;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(Math.round((yMax * i) / 4) + ' ' + (opt.yLabel || ''), 3, y + 4);
  }
  // seconds on x, nudged inward so the first label clears the y axis
  for (let i = 0; i <= 4; i++) {
    const x = pad.l + (i / 4) * iw;
    const lab = (xMax * i / 4 / 1e6).toFixed(1) + 's';
    const w = ctx.measureText(lab).width;
    const cx = Math.min(W - pad.r - w, Math.max(pad.l, x - w / 2));
    ctx.fillText(lab, cx, H - 8);
  }

  ctx.beginPath();
  ctx.moveTo(X(xs[0]), Y(0));
  for (let i = 0; i < xs.length; i++) ctx.lineTo(X(xs[i]), Y(ys[i]));
  ctx.lineTo(X(xs[xs.length - 1]), Y(0));
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ih);
  grad.addColorStop(0, 'rgba(74,163,223,0.42)');
  grad.addColorStop(1, 'rgba(74,163,223,0.02)');
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  for (let i = 0; i < xs.length; i++) {
    const x = X(xs[i]), y = Y(ys[i]);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.strokeStyle = opt.color || '#4aa3df';
  ctx.lineWidth = 1.4; ctx.stroke();

  if (opt.marker !== undefined) {
    const x = X(Math.min(opt.marker, xMax));
    ctx.strokeStyle = 'rgba(224,161,58,0.85)'; ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ih); ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawLines(sel, xs, series, opt) {
  const c = $(sel);
  const ctx = setupCanvas(c);
  const W = c.clientWidth, H = c.clientHeight;
  const pad = { l: 46, r: 14, t: 14, b: 26 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  const xMax = opt.xMax || xs[xs.length - 1];
  const yMax = Math.max(0.001, ...series.flatMap((s) => s.data));
  ctx.clearRect(0, 0, W, H);
  ctx.font = '11px ui-monospace, monospace';
  const X = (v) => pad.l + (v / xMax) * iw;
  const Y = (v) => pad.t + ih - (v / yMax) * ih;

  ctx.strokeStyle = '#212c3a'; ctx.fillStyle = '#7f8fa3';
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + ih - (i / 4) * ih;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText((yMax * i / 4).toFixed(yMax < 2 ? 2 : 0) + opt.yLabel, 3, y + 4);
  }
  for (let i = 0; i <= 4; i++) {
    const lab = (xMax * i / 4 / 1e6).toFixed(1) + 's';
    const w = ctx.measureText(lab).width;
    ctx.fillText(lab, Math.min(W - pad.r - w,
      Math.max(pad.l, pad.l + (i / 4) * iw - w / 2)), H - 8);
  }
  // trigger markers along the top
  if (opt.triggers) {
    const maxT = Math.max(1, ...opt.triggers);
    ctx.fillStyle = 'rgba(224,161,58,0.45)';
    opt.triggers.forEach((v, i) => {
      if (!v) return;
      const x = X(opt.tUs[i]);
      const h = 4 + (v / maxT) * 16;
      ctx.fillRect(x, pad.t - h + 2, 2, h);
    });
  }
  for (const s of series) {
    ctx.beginPath();
    for (let i = 0; i < xs.length; i++) {
      const x = X(xs[i]), y = Y(s.data[i]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.stroke();
  }
  // legend along the top, inside the plot area
  let lx = pad.l + 4;
  for (const s of series) {
    ctx.fillStyle = s.color; ctx.fillRect(lx, pad.t + 1, 9, 3);
    ctx.fillStyle = '#9fb2c6'; ctx.fillText(s.label, lx + 13, pad.t + 5);
    lx += ctx.measureText(s.label).width + 26;
  }
}

function drawGroupedBars(sel, rows) {
  const c = $(sel);
  const ctx = setupCanvas(c);
  const W = c.clientWidth, H = c.clientHeight;
  const pad = { l: 60, r: 12, t: 14, b: 74 };
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
  ctx.clearRect(0, 0, W, H);
  ctx.font = '10.5px ui-monospace, monospace';
  if (!rows.length) {
    ctx.fillStyle = '#7f8fa3';
    ctx.fillText('没有可对比的跳', pad.l, pad.t + 20);
    return;
  }
  // Log scale: a single 16 ms incast port would otherwise flatten every
  // other hop to invisibility.
  const vals = rows.flatMap((r) => [r.true_p99_us, r.est_p99_us]).filter((v) => v > 0);
  const yMin = Math.max(1e-2, Math.min(...vals) / 2);
  const yMax = Math.max(...vals) * 2.2;
  const L = (v) => Math.log10(Math.max(v, yMin) / yMin) / Math.log10(yMax / yMin);
  const Y = (v) => pad.t + ih - L(v) * ih;

  ctx.strokeStyle = '#212c3a'; ctx.fillStyle = '#7f8fa3';
  let lastLab = 1e9;
  for (let e = Math.floor(Math.log10(yMin)); e <= Math.ceil(Math.log10(yMax)); e++) {
    for (const m of [1, 3]) {
      const v = m * Math.pow(10, e);
      if (v < yMin || v > yMax) continue;
      const y = Y(v);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
      if (lastLab - y < 15) continue;      // skip labels that would collide
      lastLab = y;
      const lab = v >= 1000 ? (v / 1000) + ' ms' : v + ' µs';
      ctx.fillText(lab, pad.l - 6 - ctx.measureText(lab).width, y + 3);
    }
  }

  const step = iw / rows.length;
  const bw = Math.min(13, step / 3);
  rows.forEach((r, i) => {
    const x = pad.l + i * step + step / 2;
    ctx.fillStyle = '#4aa3df';
    ctx.fillRect(x - bw - 1, Y(r.true_p99_us), bw,
      Math.max(1, pad.t + ih - Y(r.true_p99_us)));
    ctx.fillStyle = '#48bb78';
    ctx.fillRect(x + 1, Y(r.est_p99_us), bw,
      Math.max(1, pad.t + ih - Y(r.est_p99_us)));
    ctx.save();
    ctx.translate(x + 2, pad.t + ih + 6);
    ctx.rotate(Math.PI / 3.6);
    ctx.fillStyle = '#7f8fa3';
    ctx.textAlign = 'left';
    ctx.fillText(r.switch.replace(/^(leaf|agg|spine)-/, '$1 ') + ':' + r.out_port, 0, 0);
    ctx.restore();
  });

  // legend, bottom-left, clear of the rotated labels
  const ly = H - 12;
  ctx.fillStyle = '#4aa3df'; ctx.fillRect(pad.l, ly - 4, 9, 3);
  ctx.fillStyle = '#9fb2c6'; ctx.fillText('仿真真值 P99', pad.l + 13, ly);
  ctx.fillStyle = '#48bb78'; ctx.fillRect(pad.l + 108, ly - 4, 9, 3);
  ctx.fillStyle = '#9fb2c6'; ctx.fillText('收集器估计 P99', pad.l + 121, ly);
  ctx.fillStyle = '#5d6d80';
  ctx.textAlign = 'right';
  ctx.fillText('y 轴：对数', W - pad.r, ly);
  ctx.textAlign = 'left';
}

/* ------------------------------------------------------------------ */
async function rebuild() {
  const seed = Number($('#seed').value || 7);
  const budget = Number($('#budget').value || 0.8) * 1e6;
  const btn = $('#rebuild');
  btn.disabled = true;
  setStatus('重新构建仿真…');
  await api('/api/rebuild', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seed, budget_bytes_per_s: budget }),
  });
  state.selectedTxn = null;
  $('#detail-body').classList.add('hidden');
  $('#detail-empty').classList.remove('hidden');
  try {
    await waitReady();
    await loadAll();
    setStatus('就绪', 'ok');
  } catch (e) {
    setStatus('构建失败: ' + e.message, 'err');
  }
  btn.disabled = false;
}

window.addEventListener('resize', debounce(() => {
  if (state.topo) drawTopology(state.lastHeat || {});
  loadQueueSeries();
  api('/api/sampling').then(renderSampling);
  api('/api/evaluation').then(renderEvaluation);
}, 250));

boot().catch((e) => {
  console.error(e);
  setStatus('加载失败: ' + e.message, 'err');
});
