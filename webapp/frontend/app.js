const API = "/api";

function el(id) { return document.getElementById(id); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function parseNames(raw) { return raw.split(",").map(s => s.trim()).filter(Boolean); }
function normalizeLite(s) { return (s || "").trim().toLowerCase().replace(/\.$/, ""); }

// ---------------- tabs ----------------
el("tab-btn-normal").onclick = () => switchTab("normal");
el("tab-btn-attack").onclick = () => switchTab("attack");

function switchTab(name) {
  for (const t of ["normal", "attack"]) {
    el(`tab-${t}`).classList.toggle("active", t === name);
    el(`tab-btn-${t}`).classList.toggle("active", t === name);
  }
}

// ---------------- rendering ----------------
function renderWeights(container, weights, advNames) {
  container.innerHTML = "";
  const advSet = new Set(Array.isArray(advNames) ? advNames : (advNames ? [advNames] : []));
  const entries = Object.entries(weights);
  const max = Math.max(0.01, ...entries.map(([, v]) => v));
  for (const [name, val] of entries) {
    const isAdv = advSet.has(name);
    const row = document.createElement("div");
    row.className = "weight-row";
    row.innerHTML = `
      <div class="weight-name">${name}${isAdv ? ' <span class="badge adv">adv</span>' : ""}</div>
      <div class="weight-bar-bg"><div class="weight-bar-fill ${isAdv ? "adv" : ""}" style="width:${(val / max * 100).toFixed(1)}%"></div></div>
      <div class="weight-val">${val.toFixed(3)}</div>
    `;
    container.appendChild(row);
  }
}

function answersCell(answers, manual, filtered) {
  return Object.entries(answers)
    .map(([name, ans]) => `<div><span class="muted">${name}:</span> ${ans}${manual[name] ? ' <span class="badge">manual</span>' : ""}${filtered && filtered[name] ? ' <span class="badge adv">filtered by AlignScore</span>' : ""}</div>`)
    .join("");
}

function renderCalHistory(tbody, rounds) {
  tbody.innerHTML = "";
  rounds.forEach((r, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i + 1}</td><td>${r.query}</td><td>${answersCell(r.answers, r.manual, r.filtered)}</td>`;
    tbody.appendChild(tr);
  });
}

function renderInfHistory(tbody, rounds) {
  tbody.innerHTML = "";
  rounds.forEach((r, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i + 1}</td><td>${r.query}</td><td>${r.chosen_sources.join(", ")}</td><td>${answersCell(r.answers, r.manual)}</td><td>${r.consensus ?? ""}</td>`;
    tbody.appendChild(tr);
  });
}

function buildContextBlocks(container, names, placeholder) {
  container.innerHTML = "";
  for (const name of names) {
    const block = document.createElement("div");
    block.className = "source-block";
    block.innerHTML = `
      <div class="name">${name}</div>
      <label>${placeholder}</label>
      <textarea data-source="${name}" placeholder="Paste the passage this source would retrieve..."></textarea>
    `;
    container.appendChild(block);
  }
}

function readContexts(container) {
  const contexts = {};
  container.querySelectorAll("textarea").forEach(t => { contexts[t.dataset.source] = t.value; });
  return contexts;
}

function setContexts(container, contexts) {
  container.querySelectorAll("textarea").forEach(t => { t.value = contexts[t.dataset.source] || ""; });
}

// ---------------- API ----------------
async function apiPost(path, body) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail);
  return data;
}
const apiCreateSession = (names, topK) => apiPost("/session", { source_names: names, top_k_src: topK });
const apiAddCalibration = (sid, query, contexts, manual) => apiPost(`/session/${sid}/calibration_round`, { query, contexts, manual_answers: manual });
const apiLock = (sid) => apiPost(`/session/${sid}/lock`);
const apiAddInference = (sid, query, contexts, manual) => apiPost(`/session/${sid}/inference_round`, { query, contexts, manual_answers: manual });

// ==================== NORMAL TAB ====================
let normalSessionId = null;
let normalSources = [];

async function createNormalSession(names, topK) {
  normalSources = names;
  const data = await apiCreateSession(names, topK);
  normalSessionId = data.session_id;
  el("normal-session-status").textContent = `Session ${normalSessionId} — ${names.length} sources, top-${data.top_k_src} consulted at inference.`;
  buildContextBlocks(el("normal-cal-contexts"), names, "Context / document this source has for the query");
  el("normal-calibration-panel").classList.remove("hidden");
  el("normal-inference-panel").classList.add("hidden");
  renderWeights(el("normal-cal-weights"), data.weights, null);
  renderCalHistory(el("normal-cal-history"), []);
}

async function addNormalCalibrationRound(query, contexts) {
  el("normal-cal-query").value = query;
  setContexts(el("normal-cal-contexts"), contexts);
  const data = await apiAddCalibration(normalSessionId, query, contexts, {});
  renderWeights(el("normal-cal-weights"), data.weights, null);
  renderCalHistory(el("normal-cal-history"), data.calibration_rounds);
}

async function lockNormal() {
  const data = await apiLock(normalSessionId);
  renderWeights(el("normal-frozen-weights"), data.weights, null);
  buildContextBlocks(el("normal-inf-contexts"), normalSources, "Context / document this source has for the query");
  el("normal-inference-panel").classList.remove("hidden");
  el("normal-add-calibration").disabled = true;
  el("normal-lock").disabled = true;
  return data;
}

async function runNormalInference(query, contexts) {
  el("normal-inf-query").value = query;
  setContexts(el("normal-inf-contexts"), contexts);
  const data = await apiAddInference(normalSessionId, query, contexts, {});
  renderInfHistory(el("normal-inf-history"), data.inference_rounds);
  return data;
}

el("normal-create-session").onclick = async () => {
  const names = parseNames(el("normal-source-names").value);
  const topK = parseInt(el("normal-top-k").value, 10) || names.length;
  if (names.length < 2) { el("normal-session-status").textContent = "Need at least 2 sources."; return; }
  try { await createNormalSession(names, topK); }
  catch (e) { el("normal-session-status").textContent = e.message; }
};

el("normal-add-calibration").onclick = async () => {
  const query = el("normal-cal-query").value.trim();
  if (!query) return;
  const btn = el("normal-add-calibration");
  btn.disabled = true; btn.textContent = "Running...";
  try {
    await addNormalCalibrationRound(query, readContexts(el("normal-cal-contexts")));
    el("normal-cal-query").value = "";
  } catch (e) { alert(e.message); }
  finally { btn.disabled = false; btn.textContent = "Add calibration round"; }
};

el("normal-lock").onclick = async () => {
  try { await lockNormal(); } catch (e) { alert(e.message); }
};

el("normal-run-inference").onclick = async () => {
  const query = el("normal-inf-query").value.trim();
  if (!query) return;
  const btn = el("normal-run-inference");
  btn.disabled = true; btn.textContent = "Running (semantic clustering)...";
  try {
    await runNormalInference(query, readContexts(el("normal-inf-contexts")));
    el("normal-inf-query").value = "";
  } catch (e) { alert(e.message); }
  finally { btn.disabled = false; btn.textContent = "Run inference round"; }
};

// ---- Normal presets ----
const NORMAL_PRESETS = [
  {
    label: "Calibrate then ask a repeat + a new question",
    sources: ["Wikipedia", "StackOverflow", "CompanyWiki"],
    topK: 3,
    calibration: [
      {
        query: "what is the boiling point of water at sea level in celsius",
        contexts: {
          Wikipedia: "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard sea-level atmospheric pressure of 1 atmosphere.",
          StackOverflow: "At standard atmospheric pressure, water reaches its boiling point at 100 degrees Celsius.",
          CompanyWiki: "Internal lab testing confirms water boils at approximately 100C under normal sea-level conditions.",
        },
      },
      {
        query: "what gas do plants absorb during photosynthesis",
        contexts: {
          Wikipedia: "During photosynthesis, plants absorb carbon dioxide from the atmosphere and release oxygen as a byproduct.",
          StackOverflow: "Photosynthesis uses carbon dioxide (CO2) taken in through the leaves, combined with water and sunlight.",
          CompanyWiki: "This document discusses quarterly sales figures and does not cover biology topics.",
        },
      },
    ],
    inference: [
      {
        query: "what is the chemical symbol for gold",
        contexts: {
          Wikipedia: "Gold's chemical symbol is Au, from the Latin word aurum.",
          StackOverflow: "The element gold is represented by the symbol Au on the periodic table.",
          CompanyWiki: "Au is the periodic table symbol for the element gold.",
        },
      },
    ],
  },
];

async function runNormalPreset(preset) {
  el("normal-source-names").value = preset.sources.join(", ");
  el("normal-top-k").value = preset.topK;
  await createNormalSession(preset.sources, preset.topK);
  for (const r of preset.calibration) { await addNormalCalibrationRound(r.query, r.contexts); await sleep(200); }
  await lockNormal();
  for (const r of preset.inference) { await runNormalInference(r.query, r.contexts); await sleep(200); }
}

function buildPresetButtons(containerId, presets, runFn) {
  const container = el(containerId);
  presets.forEach((preset) => {
    const btn = document.createElement("button");
    btn.className = "ghost";
    btn.textContent = preset.label;
    btn.style.marginRight = "8px";
    btn.style.marginBottom = "8px";
    btn.onclick = async () => {
      container.querySelectorAll("button").forEach(b => b.disabled = true);
      btn.textContent = "Running...";
      try { await runFn(preset); }
      catch (e) { alert(e.message); }
      finally {
        container.querySelectorAll("button").forEach(b => b.disabled = false);
        btn.textContent = preset.label;
      }
    };
    container.appendChild(btn);
  });
}

// ==================== ATTACK TAB ====================
// attackAdvNames: one or more colluding adversarial source names. A lone
// adversary's frozen weight is capped at 1.0 and can never outvote two
// honest sources that agree, so a genuinely flipping attack needs either
// multiple colluding sources, or honest sources whose own reliability is
// imperfect. Lowering top-kappa also demonstrates the paper's actual
// defense: a low-weight adversary can be excluded from inference entirely.
let attackSessionId = null;
let attackHonestNames = [];
let attackAdvNames = ["Adversary"];

el("attack-adv-mode-cal").onchange = () => {
  const mode = el("attack-adv-mode-cal").value;
  el("attack-adv-answer-wrap-cal").classList.toggle("hidden", mode !== "mimic");
  // context is always needed now (both honest and mimic), since AlignScore
  // filters ungrounded answers to "i don't know" regardless of mode.
};
el("attack-adv-mode-inf").onchange = () => {
  const mode = el("attack-adv-mode-inf").value;
  el("attack-adv-context-wrap-inf").classList.toggle("hidden", mode !== "honest");
  el("attack-adv-payload-wrap-inf").classList.toggle("hidden", mode !== "payload");
};

async function createAttackSession(honestNames, advNames, topK) {
  attackHonestNames = honestNames;
  attackAdvNames = advNames;
  const allSources = [...honestNames, ...advNames];
  const data = await apiCreateSession(allSources, topK);
  attackSessionId = data.session_id;
  el("attack-session-status").textContent = `Session ${attackSessionId} — ${allSources.length} sources (${advNames.length} adversarial), top-${data.top_k_src} consulted at inference.`;
  el("attack-adv-label-cal").textContent = advNames.join(", ");
  el("attack-adv-label-inf").textContent = advNames.join(", ");

  buildContextBlocks(el("attack-cal-contexts"), honestNames, "Context / document this honest source has for the query");
  el("attack-calibration-panel").classList.remove("hidden");
  el("attack-inference-panel").classList.add("hidden");
  el("attack-history-panel").classList.remove("hidden");
  el("attack-result-banner").className = "result-banner";
  renderWeights(el("attack-cal-weights"), data.weights, advNames);
  renderCalHistory(el("attack-cal-history"), []);
}

async function addAttackCalibrationRound({ query, contexts = {}, mode, advContext = "", advAnswer = "", honestAnswers = null }) {
  el("attack-cal-query").value = query;
  setContexts(el("attack-cal-contexts"), contexts);
  el("attack-adv-mode-cal").value = mode;
  el("attack-adv-context-cal").value = advContext;
  el("attack-adv-answer-cal").value = advAnswer;
  el("attack-adv-answer-wrap-cal").classList.toggle("hidden", mode !== "mimic");

  const roundContexts = { ...contexts };
  const manual = {};
  // Context is sent either way now: AlignScore scores every answer (honest
  // or mimicked) against whatever context was supplied, and filters it to
  // "i don't know" if it isn't actually grounded in it.
  for (const n of attackAdvNames) roundContexts[n] = advContext;
  if (mode === "mimic") {
    for (const n of attackAdvNames) manual[n] = advAnswer;
  }
  // honest: no manual override — Gemini generates the answer from advContext

  // honestAnswers (preset-only): scripts a specific honest source's answer
  // instead of a live Gemini call. Calibration has no semantic clustering
  // (unlike inference), so exact wording varies between Gemini calls even
  // when they mean the same thing ("edison" vs "thomas edison") — which
  // would silently break a demo that depends on an exact agreement/disagreement
  // pattern. The interactive UI never sets this; only scripted presets do.
  if (honestAnswers) {
    for (const [n, ans] of Object.entries(honestAnswers)) manual[n] = ans;
  }

  const data = await apiAddCalibration(attackSessionId, query, roundContexts, manual);
  renderWeights(el("attack-cal-weights"), data.weights, attackAdvNames);
  renderCalHistory(el("attack-cal-history"), data.calibration_rounds);
  return data;
}

async function lockAttack() {
  const data = await apiLock(attackSessionId);
  renderWeights(el("attack-frozen-weights"), data.weights, attackAdvNames);
  buildContextBlocks(el("attack-inf-contexts"), attackHonestNames, "Context / document this honest source has for the query");
  el("attack-inference-panel").classList.remove("hidden");
  el("attack-add-calibration").disabled = true;
  el("attack-lock").disabled = true;
  return data;
}

async function runAttackInference({ query, contexts = {}, mode, advContext = "", advPayload = "", honestAnswers = null }) {
  el("attack-inf-query").value = query;
  setContexts(el("attack-inf-contexts"), contexts);
  el("attack-adv-mode-inf").value = mode;
  el("attack-adv-context-inf").value = advContext;
  el("attack-adv-payload-inf").value = advPayload;
  el("attack-adv-context-wrap-inf").classList.toggle("hidden", mode !== "honest");
  el("attack-adv-payload-wrap-inf").classList.toggle("hidden", mode !== "payload");

  const roundContexts = { ...contexts };
  const manual = {};
  let payload = null;
  if (mode === "payload") {
    payload = advPayload.trim();
    for (const n of attackAdvNames) manual[n] = payload;
  } else {
    for (const n of attackAdvNames) roundContexts[n] = advContext;
  }
  // See addAttackCalibrationRound — same reasoning, scripted presets only.
  if (honestAnswers) {
    for (const [n, ans] of Object.entries(honestAnswers)) manual[n] = ans;
  }

  const data = await apiAddInference(attackSessionId, query, roundContexts, manual);
  renderInfHistory(el("attack-inf-history"), data.inference_rounds);

  const latest = data.inference_rounds[data.inference_rounds.length - 1];
  const advConsulted = attackAdvNames.filter(n => latest.chosen_sources.includes(n));
  const banner = el("attack-result-banner");

  if (mode === "payload") {
    if (advConsulted.length === 0) {
      banner.className = "result-banner fail";
      banner.textContent = `Attack didn't even get a hearing: none of [${attackAdvNames.join(", ")}] made the top-${data.top_k_src} sources consulted this round (kappa-RRSS filtered them out). Consulted: ${latest.chosen_sources.join(", ")}.`;
    } else {
      const succeeded = latest.consensus === normalizeLite(payload);
      banner.className = `result-banner ${succeeded ? "ok" : "fail"}`;
      banner.textContent = succeeded
        ? `Attack succeeded: consensus is now "${latest.consensus}" (adversary payload). Consulted sources: ${latest.chosen_sources.join(", ")}.`
        : `Attack failed: consensus is still "${latest.consensus}", not the payload "${payload}". Consulted sources: ${latest.chosen_sources.join(", ")}.`;
    }
  } else {
    banner.className = "result-banner ok";
    banner.textContent = `Honest inference round recorded. Consulted sources: ${latest.chosen_sources.join(", ")}.`;
  }
  return data;
}

el("attack-create-session").onclick = async () => {
  const honestNames = parseNames(el("attack-source-names").value);
  const advNames = parseNames(el("attack-adv-name").value);
  const topK = parseInt(el("attack-top-k").value, 10) || (honestNames.length + advNames.length);
  if (honestNames.length < 1 || advNames.length < 1) {
    el("attack-session-status").textContent = "Need at least 1 honest source and at least 1 adversary.";
    return;
  }
  try { await createAttackSession(honestNames, advNames, topK); }
  catch (e) { el("attack-session-status").textContent = e.message; }
};

el("attack-add-calibration").onclick = async () => {
  const query = el("attack-cal-query").value.trim();
  if (!query) return;
  const btn = el("attack-add-calibration");
  btn.disabled = true; btn.textContent = "Running...";
  try {
    await addAttackCalibrationRound({
      query,
      contexts: readContexts(el("attack-cal-contexts")),
      mode: el("attack-adv-mode-cal").value,
      advContext: el("attack-adv-context-cal").value,
      advAnswer: el("attack-adv-answer-cal").value,
    });
    el("attack-cal-query").value = "";
  } catch (e) { alert(e.message); }
  finally { btn.disabled = false; btn.textContent = "Add calibration round"; }
};

el("attack-lock").onclick = async () => {
  try { await lockAttack(); } catch (e) { alert(e.message); }
};

el("attack-run-inference").onclick = async () => {
  const query = el("attack-inf-query").value.trim();
  if (!query) return;
  const btn = el("attack-run-inference");
  btn.disabled = true; btn.textContent = "Running (semantic clustering)...";
  try {
    await runAttackInference({
      query,
      contexts: readContexts(el("attack-inf-contexts")),
      mode: el("attack-adv-mode-inf").value,
      advContext: el("attack-adv-context-inf").value,
      advPayload: el("attack-adv-payload-inf").value,
    });
    el("attack-inf-query").value = "";
  } catch (e) { alert(e.message); }
  finally { btn.disabled = false; btn.textContent = "Run inference round"; }
};

// ---- Attack presets ----
const CAPITAL_HONEST_CONTEXTS = {
  "Source A": "Paris is the capital and most populous city of France.",
  "Source B": "France's capital city is Paris, located on the Seine river.",
};
const PLANET_CONTEXTS = {
  "Source A": "Jupiter is the largest planet in the Solar System by both mass and volume.",
  "Source B": "Among all planets orbiting the Sun, Jupiter has the greatest mass and diameter.",
};
const GOLD_CONTEXTS = {
  "Source A": "Gold's chemical symbol is Au, from the Latin word aurum.",
  "Source B": "The element gold is represented by the symbol Au on the periodic table.",
};
function trim(contexts, honestNames) {
  const out = {};
  for (const n of honestNames) if (contexts[n]) out[n] = contexts[n];
  return out;
}

const ATTACK_PRESETS = [
  {
    label: "2 honest vs 1 adversary — attack fails (outvoted)",
    honest: ["Source A", "Source B"],
    adv: ["Adversary"],
    topK: 3,
    calibration: (h) => [
      { query: "what is the capital of France", contexts: trim(CAPITAL_HONEST_CONTEXTS, h), mode: "honest", advContext: "Paris has served as the capital of France since its founding." },
      { query: "what is the largest planet in the solar system", contexts: trim(PLANET_CONTEXTS, h), mode: "mimic", advAnswer: "jupiter", advContext: "Jupiter is the largest planet in the Solar System." },
      { query: "what is the chemical symbol for gold", contexts: trim(GOLD_CONTEXTS, h), mode: "mimic", advAnswer: "au", advContext: "The chemical symbol for gold is Au." },
    ],
    inference: (h) => [
      { query: "what is the capital of France", contexts: trim(CAPITAL_HONEST_CONTEXTS, h), mode: "payload", advPayload: "berlin" },
    ],
  },
  {
    label: "1 honest vs 2 colluding adversaries — attack succeeds",
    honest: ["Source A"],
    adv: ["Adversary 1", "Adversary 2"],
    topK: 3,
    calibration: (h) => [
      { query: "what is the capital of France", contexts: trim(CAPITAL_HONEST_CONTEXTS, h), mode: "honest", advContext: "Paris has served as the capital of France since its founding." },
      { query: "what is the largest planet in the solar system", contexts: trim(PLANET_CONTEXTS, h), mode: "mimic", advAnswer: "jupiter", advContext: "Jupiter is the largest planet in the Solar System." },
      { query: "what is the chemical symbol for gold", contexts: trim(GOLD_CONTEXTS, h), mode: "mimic", advAnswer: "au", advContext: "The chemical symbol for gold is Au." },
    ],
    inference: (h) => [
      { query: "what is the capital of France", contexts: trim(CAPITAL_HONEST_CONTEXTS, h), mode: "payload", advPayload: "berlin" },
    ],
  },
  {
    label: "1 honest vs 1 adversary, top-1 kappa — adversary gets filtered out",
    honest: ["Source A"],
    adv: ["Adversary"],
    topK: 1,
    calibration: (h) => [
      { query: "what is the capital of France", contexts: trim(CAPITAL_HONEST_CONTEXTS, h), mode: "honest", advContext: "Paris has served as the capital of France since its founding." },
      { query: "what is the largest planet in the solar system", contexts: trim(PLANET_CONTEXTS, h), mode: "mimic", advAnswer: "wrong on purpose" },
    ],
    inference: (h) => [
      { query: "what is the capital of France", contexts: trim(CAPITAL_HONEST_CONTEXTS, h), mode: "payload", advPayload: "berlin" },
    ],
  },
  {
    // A single, non-colluding adversary beats every individual honest source
    // without ever needing to know a true answer: it just sides with
    // whichever pair of honest sources already agrees on genuinely disputed
    // trivia, banking a perfect track record while the honest sources'
    // reliability gets ground down by disagreeing with EACH OTHER. Once each
    // honest source is individually weaker than the adversary, top-kappa
    // selection (meant to filter out low-reliability sources) actually helps
    // the attacker: it excludes the OTHER honest sources that would otherwise
    // have outvoted it, leaving a 1-on-1 matchup the adversary wins outright.
    label: "1 non-colluding adversary vs 3 disagreeing honest sources — wins via kappa",
    honest: ["Source A", "Source B", "Source C"],
    adv: ["Adversary"],
    topK: 2,
    calibration: () => [
      {
        query: "what is the capital of France",
        contexts: {
          "Source A": "Paris is the capital of France.",
          "Source B": "Paris is the capital of France.",
          "Source C": "Paris is the capital of France.",
        },
        // Every source's answer is scripted (honestAnswers), not left to a
        // live Gemini call — see the note on addAttackCalibrationRound. This
        // preset demonstrates a specific, verified numeric outcome, and
        // calibration has no semantic clustering to smooth over wording
        // differences between sources the way inference does.
        honestAnswers: { "Source A": "paris", "Source B": "paris", "Source C": "paris" },
        mode: "honest",
        advContext: "Paris is the capital of France.",
      },
      {
        // odd one out: Source C
        query: "who invented the telephone",
        contexts: {
          "Source A": "Alexander Graham Bell is officially credited with inventing and patenting the telephone in 1876.",
          "Source B": "Alexander Graham Bell is officially credited with inventing and patenting the telephone in 1876.",
          "Source C": "Antonio Meucci developed an early voice-communication device years before Bell and is credited by some historians as the telephone's true inventor.",
        },
        honestAnswers: { "Source A": "bell", "Source B": "bell", "Source C": "meucci" },
        mode: "mimic", advAnswer: "bell",
        advContext: "Alexander Graham Bell is officially credited with inventing and patenting the telephone in 1876.",
      },
      {
        // odd one out: Source B
        query: "who invented the light bulb",
        contexts: {
          "Source A": "Thomas Edison invented the practical incandescent light bulb and patented it in the United States in 1879.",
          "Source B": "Joseph Swan invented an early incandescent light bulb in Britain before Thomas Edison patented his version in the United States.",
          "Source C": "Thomas Edison invented the practical incandescent light bulb and patented it in the United States in 1879.",
        },
        honestAnswers: { "Source A": "edison", "Source B": "swan", "Source C": "edison" },
        mode: "mimic", advAnswer: "edison",
        advContext: "Thomas Edison invented the practical incandescent light bulb and patented it in the United States in 1879.",
      },
      {
        // odd one out: Source A — each honest source has now been "wrong"
        // exactly once (3/4 = 0.75), symmetric, none of them undefeated.
        query: "who was first to reach the south pole",
        contexts: {
          "Source A": "Robert Falcon Scott led the British Terra Nova expedition that was first to reach the South Pole in January 1912.",
          "Source B": "Roald Amundsen led the Norwegian expedition that was first to reach the South Pole, arriving in December 1911.",
          "Source C": "Roald Amundsen led the Norwegian expedition that was first to reach the South Pole, arriving in December 1911.",
        },
        honestAnswers: { "Source A": "scott", "Source B": "amundsen", "Source C": "amundsen" },
        mode: "mimic", advAnswer: "amundsen",
        advContext: "Roald Amundsen led the Norwegian expedition that was first to reach the South Pole, arriving in December 1911.",
      },
    ],
    inference: () => [
      {
        query: "what is the capital of France, one more time",
        contexts: {
          "Source A": "Paris is the capital of France.",
          "Source B": "Paris is the capital of France.",
          "Source C": "Paris is the capital of France.",
        },
        honestAnswers: { "Source A": "paris", "Source B": "paris", "Source C": "paris" },
        mode: "payload", advPayload: "berlin",
      },
    ],
  },
];

async function runAttackPreset(preset) {
  el("attack-source-names").value = preset.honest.join(", ");
  el("attack-adv-name").value = preset.adv.join(", ");
  el("attack-top-k").value = preset.topK;
  await createAttackSession(preset.honest, preset.adv, preset.topK);
  for (const r of preset.calibration(preset.honest)) { await addAttackCalibrationRound(r); await sleep(200); }
  await lockAttack();
  for (const r of preset.inference(preset.honest)) { await runAttackInference(r); await sleep(200); }
}

buildPresetButtons("normal-presets", NORMAL_PRESETS, runNormalPreset);
buildPresetButtons("attack-presets", ATTACK_PRESETS, runAttackPreset);
