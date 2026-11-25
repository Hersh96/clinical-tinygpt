// Interactive guide for Clinical TinyGPT. Every chart reads a JSON file from ./data,
// written by `make export` from the MLflow runs. A missing file shows a message
// in that chapter only; the rest of the page keeps working.
import { loadMerges } from "./bpe.js";

const DATA = "data/";
const REPO = "https://github.com/Hersh96/clinical-tinygpt";
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const fmtParams = (n) => (n >= 1e9 ? (n / 1e9).toFixed(1) + "B" : n >= 1e6 ? (n / 1e6).toFixed(n < 1e7 ? 1 : 0) + "M" : (n / 1e3).toFixed(0) + "K");
// Label each model by its measured parameter count (filled in from the data files).
const PARAMS = {};
const SIZE_LABEL = (run) => (PARAMS[run] ? fmtParams(PARAMS[run]) : run.replace(/^gpt_/, "").toUpperCase());

const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage blocked: progress just isn't remembered */ } },
};

async function load(name) {
  try {
    const r = await fetch(DATA + name + ".json");
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

function missing(key) {
  for (const el of $$(`[data-needs="${key}"]`)) {
    el.innerHTML = `<p class="missing">Data not available yet. This chart fills in once the <code>${key}</code> results are exported from the training runs.</p>`;
  }
}

// ---------------------------------------------------------------- theming
function theme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return { ink: v("--ink"), muted: v("--muted"), line: v("--line"), accent: v("--accent"), paper: v("--paper"), canvas: v("--canvas") };
}
const SERIES = ["#a93620", "#2f6f8f", "#6b8e23", "#8a5a9e", "#c58b12", "#555"];
const SERIES_DARK = ["#ffa78e", "#7cc4e8", "#b5d67a", "#d1a6e6", "#f0c35a", "#bbb"];
const series = (i) => (matchMedia("(prefers-color-scheme: dark)").matches ? SERIES_DARK : SERIES)[i % SERIES.length];

const charts = [];
function chart(canvas, config) {
  if (!window.Chart || !canvas) return null;
  const t = theme();
  Chart.defaults.color = t.muted;
  Chart.defaults.borderColor = t.line;
  Chart.defaults.font.family = 'Inter, ui-sans-serif, system-ui, sans-serif';
  config.options = { responsive: true, maintainAspectRatio: false, animation: false, ...config.options };
  const c = new Chart(canvas, config);
  charts.push({ canvas, config, c });
  return c;
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => location.reload());

// ---------------------------------------------------------------- glossary
const GLOSSARY = {
  token: "A token is a piece of text, often a word or part of a word, that the model treats as one unit. Each token has an ID number.",
  bpe: "Byte pair encoding starts from single bytes and repeatedly merges the most frequent neighbouring pair into a new token, until the vocabulary is the size you want.",
  block: "A transformer block has two parts. Attention lets each token gather information from earlier tokens, and a small neural network (the MLP) then processes each token on its own. Stacking blocks gives the model depth.",
  loss: "Loss (cross-entropy) measures how surprised the model is by the real next token. A model guessing at random over 8,192 tokens scores about 9.0. Lower is better.",
  optimizer: "The optimizer (AdamW here) uses the gradient, which says which direction reduces the loss, to update every weight a little after each batch.",
  powerlaw: "A power law L(N) = a·N^(−α) + c. On log-log axes it's close to a straight line. α says how fast loss falls as size grows, and c is the floor.",
  kv: "In attention, every token has a query (what it's looking for), a key (what it contains) and a value (what it passes on). Keys and values of earlier tokens never change, so they can be cached.",
  bpb: "Bits per byte is the model's total surprise divided by the number of bytes of text. It doesn't depend on the tokenizer, so different models can be compared fairly. Random guessing is 8.0.",
  embedding: "An embedding is a learned list of numbers (a vector) for each token. Tokens used in similar ways end up with similar vectors.",
  attention: "Attention weights say how much each token looks at each earlier token. Every row adds up to 100%.",
};
function setupGlossary() {
  const pop = $("#glossary-pop"), text = $("#glossary-text");
  let current = null;
  const close = () => { pop.hidden = true; current?.focus(); current = null; };
  document.addEventListener("click", (e) => {
    const b = e.target.closest("button.term");
    if (b) {
      text.textContent = GLOSSARY[b.dataset.term] || "";
      pop.hidden = false;
      const r = b.getBoundingClientRect();
      pop.style.left = Math.max(16, Math.min(r.left, innerWidth - pop.offsetWidth - 16)) + "px";
      pop.style.top = (r.bottom + 8 + pop.offsetHeight > innerHeight ? r.top - pop.offsetHeight - 8 : r.bottom + 8) + "px";
      current = b;
      $("#glossary-close").focus();
      return;
    }
    if (!pop.hidden && !e.target.closest("#glossary-pop")) close();
  });
  $("#glossary-close").addEventListener("click", close);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !pop.hidden) close(); });
  addEventListener("scroll", () => { if (!pop.hidden) close(); }, { passive: true });
}

// ---------------------------------------------------------------- progress rail
function setupRail() {
  const read = new Set(store.get("ctg-read") || []);
  const links = new Map($$("#rail a").map((a) => [a.dataset.ch, a]));
  const paint = () => {
    for (const [ch, a] of links) a.classList.toggle("read", read.has(ch));
    $("#rail-count").textContent = `${read.size} of ${links.size} read`;
  };
  paint();
  const active = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) {
      for (const a of links.values()) a.classList.remove("active");
      const a = links.get(e.target.id);
      a?.classList.add("active");
      a?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
    }
  }, { rootMargin: "-40% 0px -55% 0px" });
  const done = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) {
      read.add(e.target.parentElement.id); store.set("ctg-read", [...read]); paint();
    }
  });
  for (const sec of $$("section.chapter")) {
    active.observe(sec);
    const end = document.createElement("span");
    end.setAttribute("aria-hidden", "true");
    sec.appendChild(end);
    done.observe(end);
  }
}

// ---------------------------------------------------------------- start here
function renderSummary(s) {
  if (!s) return missing("summary");
  if (s.preview) $("#preview-note").hidden = false;
  const set = (k, v) => { const el = $(`[data-stat="${k}"]`); if (el) el.textContent = v; };
  if (s.tokens_per_word) set("tpw", `${fmt(s.tokens_per_word.ours)} vs ${fmt(s.tokens_per_word.gpt2)}`);
  if (s.best_bpb) set("bpb", `${fmt(s.best_bpb.bpb, 2)}${s.gpt2_bpb ? ` vs ${fmt(s.gpt2_bpb, 2)}` : ""}`);
  if (s.kv_speedup_512) set("kv", `${fmt(s.kv_speedup_512, 1)}×`);
  if (s.kv_speedup_512_by_device?.cpu) { const el = $('[data-stat="kv"] + span'); if (el) el.textContent = "faster generation with the KV cache (CPU, 512 tokens)"; }
  if (s.scaling_alpha != null) set("alpha", `α = ${fmt(s.scaling_alpha, 2)}`);
}

// ---------------------------------------------------------------- 1. tokenization
const mergeCache = {};
async function tokenizerFor(v) {
  if (!mergeCache[v]) mergeCache[v] = load(`merges_${v}`).then((d) => (d ? loadMerges(d) : null));
  return mergeCache[v];
}
function chipsHTML(pieces, cls = "chip") {
  return pieces.map((p, i) => {
    const shown = p.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥");
    const esc = shown.replace(/&/g, "&amp;").replace(/</g, "&lt;");
    return `<span class="${cls} c${i % 5}" data-piece="${p.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;")}">${esc}</span>`;
  }).join("");
}
async function setupPlayground() {
  const input = $("#tok-input"), select = $("#tok-vocab");
  if (!(await tokenizerFor(select.value))) { missing("merges"); return; }
  let pending = 0;
  const run = async () => {
    const id = ++pending;
    const tok = await tokenizerFor(select.value);
    if (id !== pending || !tok) return;
    const text = input.value;
    const pieces = tok.tokens(text);
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    $("#tok-count").textContent = `${pieces.length} tokens for ${words} words, ${text.length} characters`;
    $("#tok-chips").innerHTML = chipsHTML(pieces);
  };
  input.addEventListener("input", run);
  select.addEventListener("change", run);
  await run();
}
function renderTokenizerCompare(d) {
  if (!d) return missing("tokenizer_compare");
  const t = d.tokenizers;
  const names = Object.keys(t).sort((a, b) => (a === "gpt2") - (b === "gpt2") || a.localeCompare(b, undefined, { numeric: true }));
  const label = (n) => (n === "gpt2" ? "GPT-2 (web text)" : `Ours, ${Number(n.split("_")[1]).toLocaleString()} vocab`);
  chart($("#chart-tokenizer"), {
    type: "bar",
    data: { labels: names.map(label), datasets: [{ label: "Tokens per word (lower is better)", data: names.map((n) => t[n].tokens_per_word),
      backgroundColor: names.map((n) => (n === "gpt2" ? theme().muted : series(0))) }] },
    options: { indexAxis: "y", plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${fmt(c.raw, 3)} tokens per word` } } },
      scales: { x: { beginAtZero: true, title: { display: true, text: "Tokens per word on held-out notes (lower is better)" } } } },
  });
  const terms = t[names[0]].examples.map((e) => e.term);
  const head = `<thead><tr><th>Term</th>${names.map((n) => `<th>${label(n)}</th>`).join("")}</tr></thead>`;
  const rows = terms.map((term, i) => `<tr><td>${term}</td>${names.map((n) => {
    const ex = t[n].examples[i];
    const p = ex.pieces ? ex.pieces.join(" · ") : "";
    return `<td title="${p.replace(/"/g, "&quot;")}">${ex.n_tokens}</td>`;
  }).join("")}</tr>`).join("");
  $("#term-table").innerHTML = `<table>${head}<tbody>${rows}</tbody></table><p class="code-link">Number of tokens for each clinical term. Hover a number to see the pieces.</p>`;
  const ours = names.filter((n) => n !== "gpt2");
  if (t.gpt2 && ours.length) {
    const best = ours.reduce((a, b) => (t[a].tokens_per_word < t[b].tokens_per_word ? a : b));
    const pct = (1 - t[best].tokens_per_word / t.gpt2.tokens_per_word) * 100;
    $("#tok-means").textContent = `Our ${Number(best.split("_")[1]).toLocaleString()}-token vocabulary needs ${fmt(Math.abs(pct), 0)}% ${pct >= 0 ? "fewer" : "more"} tokens per word than GPT-2 on clinical notes. ` + $("#tok-means").textContent;
  }
}

// ---------------------------------------------------------------- 2. transformer
function heat(v, t) {
  // 0 -> paper, 1 -> accent
  return `color-mix(in srgb, ${t.accent} ${Math.round(Math.max(0, Math.min(1, v)) * 100)}%, ${t.paper})`;
}
function renderTransformer(a) {
  if (!a) return missing("attention_example");
  const panel = $("#tf-panel");
  const t = theme();
  let sel = a.tokens.length - 1, layer = a.attention.length - 1, head = 0;
  const steps = [
    () => `<p>The sentence <i>"${a.sentence}"</i> becomes ${a.tokens.length} tokens. Each one is just an ID number to the model.</p>
      <div class="tok-row">${chipsHTML(a.tokens)}</div>`,
    () => `<p>Each token ID looks up a learned <button class="term" data-term="embedding">embedding</button> vector with ${a.model.n_embd} numbers. These are the first 16 for each token. Darker means a larger value, positive or negative.</p>
      <div class="emb">${a.tokens.map((tok, i) => `<div class="emb-row"><span>${tok.replace(/ /g, "␣")}</span>${a.embedding_preview[i].map((x) => `<i style="background:${heat(Math.abs(x), t)}" title="${fmt(x, 3)}"></i>`).join("")}</div>`).join("")}</div>`,
    () => `<p>In <button class="term" data-term="attention">attention</button>, every token decides how much to look at each earlier token. Click a token to see what it attends to. The model has ${a.model.n_layer} layers with ${a.model.n_head} heads each, and each head learns a different pattern.</p>
      <div class="attn-controls"><label>Layer <select id="attn-layer">${a.attention.map((_, i) => `<option value="${i}" ${i === layer ? "selected" : ""}>${i + 1}</option>`).join("")}</select></label>
      <label>Head <select id="attn-head">${a.attention[0].map((_, i) => `<option value="${i}" ${i === head ? "selected" : ""}>${i + 1}</option>`).join("")}</select></label></div>
      <div class="tok-row" id="attn-tokens">${chipsHTML(a.tokens)}</div>
      <div class="bars" id="attn-bars"></div>
      <p class="code-link">The bars show the top 5 tokens the selected token looks at. In the grid below, each row is a token in sentence order and each column is a token it can look at. Darker means more attention, and the outlined row is the token you selected. The empty upper right is the causal mask: no token can look at the future.</p>
      <div class="attn-wrap"><canvas id="attn-canvas" aria-label="Attention heatmap: rows are tokens, columns are the tokens they attend to"></canvas></div>`,
    () => `<p>After the last layer, the model scores every token in its vocabulary as the possible next one. Here are its top guesses after <b>"${a.tokens.slice(0, sel + 1).join("")}"</b>. Click a token to change the position.</p>
      <div class="tok-row" id="nt-tokens">${chipsHTML(a.tokens)}</div><div class="bars" id="nt-bars"></div>`,
  ];
  const bars = (el, items) => {
    const max = Math.max(...items.map((x) => x.p), 1e-9);
    el.innerHTML = items.map((x) => `<div class="bar"><span>${String(x.token).replace(/ /g, "␣").replace(/</g, "&lt;")}</span><i style="width:${(x.p / max) * 100}%"></i><span>${(x.p * 100).toFixed(1)}%</span></div>`).join("");
  };
  const drawAttn = () => {
    const m = a.attention[layer][head];
    const n = m.length, cell = Math.max(12, Math.min(28, Math.floor((panel.clientWidth - 20) / n)));
    const cv = $("#attn-canvas");
    const dpr = devicePixelRatio || 1;
    cv.width = n * cell * dpr; cv.height = n * cell * dpr;
    cv.style.width = n * cell + "px"; cv.style.height = n * cell + "px";
    const ctx = cv.getContext("2d");
    ctx.scale(dpr, dpr);
    const probe = document.createElement("span"); probe.style.color = t.accent; document.body.appendChild(probe);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      probe.style.backgroundColor = heat(m[i][j] ** 0.5, t);
      ctx.fillStyle = getComputedStyle(probe).backgroundColor;
      ctx.fillRect(j * cell, i * cell, cell - 1, cell - 1);
    }
    probe.remove();
    ctx.strokeStyle = t.ink; ctx.lineWidth = 2; ctx.strokeRect(1, sel * cell + 1, n * cell - 2, cell - 2);
    const row = m[sel].map((p, j) => ({ token: a.tokens[j], p })).filter((x, j) => j <= sel).sort((x, y) => y.p - x.p).slice(0, 5);
    bars($("#attn-bars"), row);
    $$("#attn-tokens .chip").forEach((c, i) => c.classList.toggle("sel", i === sel));
  };
  const show = (k) => {
    $$(".steps button").forEach((b) => b.setAttribute("aria-selected", String(+b.dataset.step === k)));
    panel.innerHTML = steps[k]();
    if (k === 2) {
      $("#attn-layer").onchange = (e) => { layer = +e.target.value; drawAttn(); };
      $("#attn-head").onchange = (e) => { head = +e.target.value; drawAttn(); };
      $$("#attn-tokens .chip").forEach((c, i) => (c.onclick = () => { sel = i; drawAttn(); }));
      drawAttn();
    }
    if (k === 3) {
      const draw = () => { bars($("#nt-bars"), a.next_token[sel]); $$("#nt-tokens .chip").forEach((c, i) => c.classList.toggle("sel", i === sel)); };
      $$("#nt-tokens .chip").forEach((c, i) => (c.onclick = () => { sel = i; show(3); }));
      draw();
    }
  };
  $$(".steps button").forEach((b) => (b.onclick = () => show(+b.dataset.step)));
  show(0);
}

// ---------------------------------------------------------------- 3. training
function renderLoss(d) {
  if (!d) return missing("loss_curves");
  const runs = Object.entries(d.runs).sort((x, y) => x[1].n_params - y[1].n_params);
  const datasets = [];
  runs.forEach(([name, r], i) => {
    datasets.push({ label: `${SIZE_LABEL(name)} train`, data: r.train.map(([x, y]) => ({ x, y })), borderColor: series(i), backgroundColor: series(i), borderWidth: 1.5, pointRadius: 0, tension: 0.2 });
    datasets.push({ label: `${SIZE_LABEL(name)} validation`, data: r.val.map(([x, y]) => ({ x, y })), borderColor: series(i), backgroundColor: series(i), borderDash: [5, 4], borderWidth: 2, pointRadius: 2 });
  });
  chart($("#chart-loss"), {
    type: "line", data: { datasets },
    options: { parsing: false, interaction: { mode: "nearest", intersect: false },
      plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: loss ${fmt(c.parsed.y, 3)} at step ${c.parsed.x}` } } },
      scales: { x: { type: "linear", title: { display: true, text: "Training step" } }, y: { title: { display: true, text: "Loss (lower is better)" } } } },
  });
}

// ---------------------------------------------------------------- 4. scaling
function renderScaling(d) {
  if (!d) return missing("scaling");
  const { a, alpha, c } = d.fit;
  const pred = (n) => a * n ** -alpha + c;
  const xs = d.points.map((p) => p.n_params);
  const lo = Math.log10(Math.min(...xs)) - 0.5, hi = 9;
  const curve = Array.from({ length: 60 }, (_, i) => { const n = 10 ** (lo + ((hi - lo) * i) / 59); return { x: n, y: pred(n) }; });
  const slider = $("#scale-slider");
  slider.min = lo.toFixed(2); slider.max = hi; slider.value = Math.log10(Math.max(...xs)).toFixed(2);
  const ch = chart($("#chart-scaling"), {
    type: "scatter",
    data: { datasets: [
      { label: "Trained models", data: d.points.map((p) => ({ x: p.n_params, y: p.val_loss, run: p.run })), backgroundColor: series(0), pointRadius: 6 },
      { label: `Fit: L = a·N^(−${fmt(alpha, 3)}) + ${fmt(c, 2)}`, data: curve, type: "line", borderColor: theme().muted, borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 },
      { label: "Your pick", data: [], backgroundColor: theme().ink, pointStyle: "rectRot", pointRadius: 8 },
    ] },
    options: { parsing: false,
      plugins: { tooltip: { callbacks: { label: (ctx) => `${ctx.raw.run ? SIZE_LABEL(ctx.raw.run) + ": " : ""}${fmtParams(ctx.raw.x)} params, loss ${fmt(ctx.raw.y, 3)}` } } },
      scales: { x: { type: "logarithmic", title: { display: true, text: "Parameters (log scale, excluding embeddings)" }, ticks: { callback: (v) => ([1e5, 1e6, 1e7, 1e8, 1e9].includes(v) ? fmtParams(v) : "") } },
        y: { title: { display: true, text: "Validation loss" } } } },
  });
  const update = () => {
    const n = 10 ** +slider.value;
    const L = pred(n);
    $("#scale-out").textContent = `${fmtParams(n)} parameters → predicted loss ${fmt(L, 3)}`;
    if (ch) { ch.data.datasets[2].data = [{ x: n, y: L }]; ch.update("none"); }
  };
  slider.addEventListener("input", update);
  update();
  $("#scale-means").textContent = `Across ${d.points.length} sizes, loss follows the fitted curve with α = ${fmt(alpha, 2)} (R² = ${fmt(d.fit.r2, 3)}). Each time the model gets 10 times bigger, the reducible part of the loss shrinks by about ${fmt((1 - 10 ** -alpha) * 100, 0)}%. ` + $("#scale-means").textContent.split(". ").slice(1).join(". ");
}

// ---------------------------------------------------------------- 5. KV cache
function setupKvDemo() {
  const N = 16, noRow = $("#kv-nocache"), yesRow = $("#kv-cache");
  let timer = null;
  const reset = () => { noRow.innerHTML = ""; yesRow.innerHTML = ""; $("#kv-work-no").textContent = "0 computed"; $("#kv-work-yes").textContent = "0 computed"; };
  const play = () => {
    clearInterval(timer); reset();
    let step = 0, workNo = 0, workYes = 0;
    timer = setInterval(() => {
      step += 1; workNo += step; workYes += 1;
      noRow.innerHTML = Array.from({ length: step }, (_, i) => `<i class="${i === step - 1 ? "new" : "re"}"></i>`).join("");
      yesRow.innerHTML = Array.from({ length: step }, (_, i) => `<i class="${i === step - 1 ? "new" : ""}"></i>`).join("");
      $("#kv-work-no").textContent = `${workNo} computed`;
      $("#kv-work-yes").textContent = `${workYes} computed`;
      if (step >= N) clearInterval(timer);
    }, matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 280);
  };
  $("#kv-play").addEventListener("click", play);
}
function renderKv(d) {
  if (!d) return missing("kv_bench");
  // Older files hold one run at the top level; newer ones hold one run per device.
  const runs = d.runs || [{ device: d.device || "cpu", rows: d.rows }];
  const lengths = [...new Set(runs.flatMap((r) => r.rows.map((x) => x.length)))].sort((a, b) => a - b);
  const get = (run, cache) => lengths.map((L) => run.rows.find((r) => r.length === L && r.cache === cache)?.tokens_per_sec ?? null);
  const name = (dev) => (dev === "cuda" ? "GPU" : "CPU");
  const datasets = runs.flatMap((run, i) => [
    { label: `${name(run.device)}, with cache`, data: get(run, true), borderColor: series(i), backgroundColor: series(i), pointRadius: 4, spanGaps: true },
    { label: `${name(run.device)}, without cache`, data: get(run, false), borderColor: series(i), backgroundColor: series(i), borderDash: [5, 4], pointRadius: 4, spanGaps: true },
  ]);
  chart($("#chart-kv"), {
    type: "line",
    data: { labels: lengths, datasets },
    options: { plugins: { tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmt(c.raw, 0)} tokens/s` } } },
      scales: { x: { title: { display: true, text: "Tokens generated" } }, y: { beginAtZero: true, title: { display: true, text: `Tokens per second (${d.model || "model"})` } } } },
  });
  const at = (run) => {
    const ys = get(run, true), ns = get(run, false);
    const i = lengths.indexOf(512) >= 0 && ys[lengths.indexOf(512)] ? lengths.indexOf(512) : ys.findLastIndex((v) => v);
    return ys[i] && ns[i] ? { L: lengths[i], x: ys[i] / ns[i] } : null;
  };
  const parts = runs.map((run) => [run.device, at(run)]).filter(([, r]) => r);
  const cpu = parts.find(([dev]) => dev === "cpu"), gpu = parts.find(([dev]) => dev === "cuda");
  let text = "";
  if (cpu) text += `On the CPU, the cache makes generating ${cpu[1].L} tokens ${fmt(cpu[1].x, 1)} times faster. `;
  if (gpu) text += `On the laptop GPU it's ${fmt(gpu[1].x, 2)} times. With one sequence at a time, each step is so small that the fixed cost of starting GPU work dominates, so recomputing the earlier tokens costs the GPU almost nothing extra. The cache saves the same work on both; it only shows up as speed when the arithmetic is the bottleneck, as it is on a CPU or when serving many users at once. `;
  if (!cpu && !gpu && parts.length) text += `At ${parts[0][1].L} tokens the cache makes generation ${fmt(parts[0][1].x, 1)} times faster. `;
  $("#kv-means").textContent = text + $("#kv-means").textContent;
}

// ---------------------------------------------------------------- 6. results
function renderBpb(d) {
  if (!d) return missing("bpb");
  const m = Object.entries(d.models).sort((x, y) => x[1].n_params - y[1].n_params);
  chart($("#chart-bpb"), {
    type: "bar",
    data: { labels: m.map(([k, v]) => k === "gpt2_small" ? `GPT-2 small (${fmtParams(v.n_params)})` : `Ours, ${SIZE_LABEL(k)}`),
      datasets: [{ label: "Bits per byte", data: m.map(([, v]) => v.bpb), backgroundColor: m.map(([k]) => (k === "gpt2_small" ? theme().muted : series(0))) }] },
    options: { plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${fmt(c.raw, 3)} bits per byte` } } },
      scales: { y: { beginAtZero: true, title: { display: true, text: "Bits per byte on held-out notes (lower is better)" } } } },
  });
}
function renderSamples(d) {
  if (!d) return missing("samples");
  const sel = $("#sample-model");
  const names = Object.keys(d.models);
  sel.innerHTML = names.map((n) => `<option value="${n}">${SIZE_LABEL(n)} parameters</option>`).join("");
  sel.value = names.at(-1);
  const show = () => {
    $("#samples").innerHTML = d.models[sel.value].map((s) => {
      const esc = (x) => x.replace(/&/g, "&amp;").replace(/</g, "&lt;");
      const rest = s.output.startsWith(s.prompt) ? s.output.slice(s.prompt.length) : s.output;
      return `<div class="sample"><b>${esc(s.prompt)}</b>${esc(rest)}</div>`;
    }).join("");
  };
  sel.addEventListener("change", show);
  show();
}

// ---------------------------------------------------------------- boot
async function main() {
  setupGlossary();
  setupRail();
  setupKvDemo();
  const names = ["summary", "tokenizer_compare", "attention_example", "loss_curves", "scaling", "kv_bench", "bpb", "samples"];
  const data = Object.fromEntries(await Promise.all(names.map(async (n) => [n, await load(n)])));
  for (const [k, v] of Object.entries(data.loss_curves?.runs || {})) if (v.n_params) PARAMS[k] = v.n_params;
  for (const [k, v] of Object.entries(data.bpb?.models || {})) if (v.n_params) PARAMS[k] = v.n_params;
  const steps = [
    () => renderSummary(data.summary), () => setupPlayground(), () => renderTokenizerCompare(data.tokenizer_compare),
    () => renderTransformer(data.attention_example), () => renderLoss(data.loss_curves), () => renderScaling(data.scaling),
    () => renderKv(data.kv_bench), () => renderBpb(data.bpb), () => renderSamples(data.samples),
  ];
  for (const s of steps) {
    try { await s(); } catch (e) { console.error(e); }
  }
  document.body.dataset.ready = "1";
}
main();
