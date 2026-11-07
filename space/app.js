/* Stored-evidence workbench: no inference, uploads or automatic release. */
"use strict";
const $ = id => document.getElementById(id);
const E = window.ScarEvidence;
let bundle = null, bundleHash = null, record = null, loadVersion = 0;
const node = (tag, text, cls) => {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (cls) el.className = cls;
  return el;
};
const digest = async text => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text))),
  x => x.toString(16).padStart(2, "0")).join("");
const metric = m => m ? `${m.value.toFixed(3)} [${m.ci95.map(x => x.toFixed(3)).join(", ")}] · n=${m.n}` : "Missing";
const enableExports = enabled => { $("export").disabled = $("export-summary").disabled = !enabled; };
function policy() {
  return {max_harm_asr: $("max-harm").valueAsNumber, max_over_refusal: $("max-refusal").valueAsNumber,
    min_utility: $("min-utility").valueAsNumber, max_edited_scalars: $("max-support").valueAsNumber};
}
function evaluate() {
  if (!bundle || !$("policy").reportValidity()) return;
  try {
    const p = policy(), expected = {model: $("expected-model").value, revision: $("expected-revision").value, split: $("expected-split").value};
    const decisions = bundle.arms.map(a => ({arm: a.id, ...E.decide(bundle, a, p, expected)}));
    $("decisions").replaceChildren();
    for (const d of decisions.filter(d => bundle.arms.find(a => a.id === d.arm).kind === "repair")) {
      const section = node("article", undefined, "decision " + d.status);
      section.append(node("h3", `${d.arm}: ${d.status.replaceAll("_", " ")}`));
      const reasons = node("ul"); d.reasons.forEach(r => reasons.append(node("li", r))); section.append(reasons);
      $("decisions").append(section);
    }
    record = {schema: "scartissue.decision/1", created_at: new Date().toISOString(), bundle_sha256: bundleHash,
      study: {model: bundle.model, revision: bundle.model_revision, split_sha256: bundle.split_sha256},
      expected, policy: p, quality: bundle.quality, sources: bundle.sources, decisions,
      metrics: bundle.arms.map(a => ({id: a.id, weights_sha256: a.weights_sha256, edited_scalars: a.edited_scalars, metrics: a.metrics})),
      limitations: ["Research policy only; not production-safety certification.",
        "Source hashes and control outcomes are bundle declarations, not independently authenticated.",
        "Marginal intervals do not give joint safety guarantees.", "Raw prompts and answers are excluded from this record."]};
    enableExports(true);
    $("status").textContent = "Policy evaluated. Read each decision's evidence gates and reasons.";
  } catch (error) { record = null; enableExports(false); $("status").textContent = "Cannot evaluate: " + error.message; }
}
function comparison() {
  const frontier = new Set(E.frontier(bundle));
  const table = $("frontier-table"); table.replaceChildren();
  table.append(node("caption", "Point estimates [intervals] and denominators; research evidence, not release approval."));
  const head = node("tr");
  for (const label of ["Arm", "Harm-ASR ↓", "Over-refusal ↓", "Utility ↑", "Repair scalars ↓", "Descriptive status"]) {
    const th = node("th", label); th.scope = "col"; head.append(th);
  }
  const thead = node("thead"); thead.append(head); table.append(thead);
  const tbody = node("tbody");
  for (const a of bundle.arms) {
    const tr = node("tr"); const th = node("th", a.id); th.scope = "row"; tr.append(th);
    for (const text of [metric(a.metrics.harm_asr), metric(a.metrics.over_refusal), metric(a.metrics.utility), a.edited_scalars.toLocaleString(),
      a.kind === "reference" ? "Reference" : !["harm_asr", "over_refusal", "utility"].every(k => a.metrics[k]) ? "Incomplete" : frontier.has(a.id) ? "Nondominated estimate" : "Dominated estimate"])
      tr.append(node("td", text));
    tbody.append(tr);
  }
  table.append(tbody);
  const ns = "http://www.w3.org/2000/svg";
  const svgNode = (tag, attrs, text) => {
    const el = document.createElementNS(ns, tag); for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    if (text !== undefined) el.textContent = text; return el;
  };
  const svg = svgNode("svg", {viewBox: "0 0 820 360", role: "img", "aria-labelledby": "chart-title"});
  svg.append(svgNode("title", {id: "chart-title"}, "Harm-ASR versus over-refusal with marginal intervals. All values also appear in the table."));
  const points = bundle.arms.filter(a => a.metrics.harm_asr && a.metrics.over_refusal);
  const xmax = Math.min(1, Math.max(0.1, ...points.map(a => a.metrics.over_refusal.ci95[1])) * 1.15);
  const ymax = Math.min(1, Math.max(0.1, ...points.map(a => a.metrics.harm_asr.ci95[1])) * 1.15);
  const x = v => 70 + v / xmax * 650, y = v => 290 - v / ymax * 250;
  for (let i = 0; i <= 4; i++) {
    const vx = xmax * i / 4, vy = ymax * i / 4;
    svg.append(svgNode("line", {x1: 70, x2: 720, y1: y(vy), y2: y(vy), stroke: "#e2e8f0"}),
      svgNode("text", {x: 58, y: y(vy) + 5, "text-anchor": "end"}, vy.toFixed(2)),
      svgNode("text", {x: x(vx), y: 314, "text-anchor": "middle"}, vx.toFixed(2)));
  }
  svg.append(svgNode("text", {x: 395, y: 345, "text-anchor": "middle"}, "Over-refusal →"),
    svgNode("text", {x: 70, y: 23}, "Harm-ASR ↑ (lower is better)"));
  points.forEach((a, i) => {
    const h = a.metrics.harm_asr, o = a.metrics.over_refusal, color = a.kind === "reference" ? "#65758b" : "#2b6cb0";
    svg.append(svgNode("line", {x1: x(o.ci95[0]), x2: x(o.ci95[1]), y1: y(h.value), y2: y(h.value), stroke: color}),
      svgNode("line", {x1: x(o.value), x2: x(o.value), y1: y(h.ci95[0]), y2: y(h.ci95[1]), stroke: color}),
      svgNode("circle", {cx: x(o.value), cy: y(h.value), r: 5, fill: color}),
      svgNode("text", {x: x(o.value) + 9, y: y(h.value) - 8 - (i % 2) * 13}, a.id));
  });
  $("chart").replaceChildren(svg);
}
function replay() {
  const p = bundle.prompts.find(p => p.id === $("preset").value); if (!p) return;
  const panel = $("replay"); panel.replaceChildren();
  const prompt = node("details"); prompt.append(node("summary", `Reveal ${p.role === "eval_malicious" ? "potentially harmful" : "benign"} prompt · ${p.id}`), node("p", p.text, "prompt")); panel.append(prompt);
  const columns = node("div", undefined, "replay-grid");
  for (const a of bundle.arms) {
    const article = node("article"); article.append(node("h3", a.id));
    const c = a.completions.find(c => c.prompt_id === p.id);
    if (!c) { article.append(node("p", "Missing completion — not safe.")); columns.append(article); continue; }
    const votes = [];
    for (const family of ["harmbench", "qwen3guard", "granite"]) {
      const j = c.judges[family];
      article.append(node("p", `${family}: ${j?.status === "scored" ? "harm=" + j.parsed.harm + (typeof j.parsed.refusal === "boolean" ? ", refusal=" + j.parsed.refusal : "") : j?.status || "missing"}`));
      if (j?.status === "scored") votes.push(j.parsed.harm > 0.5);
    }
    article.append(node("p", votes.length !== 3 ? "Majority: unavailable" : `Majority harm: ${votes.filter(Boolean).length >= 2 ? "yes" : "no"} · ${votes.every(v => v === votes[0]) ? "agreement" : "judges disagree"}`));
    const details = node("details"); details.append(node("summary", "Reveal full answer and raw judgments"), node("p", c.answer, "comp"),
      node("pre", JSON.stringify({judges: c.judges, stop_diagnostics: c.stop}, null, 2)));
    article.append(details); columns.append(article);
  }
  panel.append(columns);
}
async function load(raw, initial, version) {
  const candidate = E.validate(JSON.parse(raw));
  for (const p of candidate.prompts) if (await digest(p.text) !== p.sha256) throw new Error("Prompt content hash mismatch");
  for (const a of candidate.arms) for (const c of a.completions)
    if (await digest(c.answer) !== c.answer_sha256) throw new Error("Answer content hash mismatch");
  const hash = await digest(raw); if (version !== loadVersion) return;
  bundle = candidate; bundleHash = hash; record = null;
  if (initial) { $("expected-model").value = bundle.model; $("expected-revision").value = bundle.model_revision; $("expected-split").value = bundle.split_sha256; }
  $("study").textContent = `${bundle.model} · revision ${bundle.model_revision} · split ${bundle.split_sha256}`;
  $("provenance").replaceChildren(node("pre", JSON.stringify({sources: bundle.sources, quality: bundle.quality, limitations: bundle.limitations}, null, 2)));
  $("preset").replaceChildren(...bundle.prompts.map(p => { const option = node("option", `${p.role}: ${p.id}`); option.value = p.id; return option; }));
  $("policy-controls").disabled = false; $("preset").disabled = false;
  comparison(); replay(); evaluate();
}
$("policy").addEventListener("submit", event => { event.preventDefault(); evaluate(); });
$("policy").addEventListener("input", () => { record = null; enableExports(false); $("status").textContent = "Policy changed. Evaluate repairs to refresh the decision record."; });
$("preset").addEventListener("change", replay);
$("bundle-file").addEventListener("change", async event => {
  const file = event.target.files[0]; if (!file) return;
  const version = ++loadVersion;
  try { if (file.size > 16 * 1024 * 1024) throw new Error("Evidence file exceeds the import limit"); await load(await file.text(), !bundle, version); }
  catch (error) { if (version === loadVersion) $("status").textContent = "Import rejected; previous study unchanged: " + error.message; }
});
function download(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], {type}));
  const a = node("a"); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("export").addEventListener("click", () => {
  if (record) download(JSON.stringify(record, null, 2) + "\n", "scartissue-decision.json", "application/json");
});
$("export-summary").addEventListener("click", () => {
  if (!record) return;
  const lines = ["ScarTissue research decision", record.created_at, "", `Model: ${record.study.model}`,
    `Revision: ${record.study.revision}`, `Split: ${record.study.split_sha256}`, `Bundle SHA-256: ${record.bundle_sha256}`,
    "", "Requested identity", JSON.stringify(record.expected), "", "Policy", JSON.stringify(record.policy),
    "", "Evidence quality", JSON.stringify(record.quality), "", "Decisions"];
  for (const d of record.decisions) lines.push(`${d.arm}: ${d.status}`, ...d.reasons.map(r => "- " + r));
  lines.push("", "Metrics: estimate [interval] and denominator");
  for (const a of record.metrics) lines.push(`${a.id}: harm ${metric(a.metrics.harm_asr)}; over-refusal ${metric(a.metrics.over_refusal)}; utility ${metric(a.metrics.utility)}; repair scalars ${a.edited_scalars}; weights SHA-256 ${a.weights_sha256}`);
  lines.push("", "Declared source files");
  for (const s of record.sources) lines.push(`${s.path}: ${s.sha256}`);
  lines.push("", "Limitations", ...record.limitations.map(x => "- " + x));
  download(lines.join("\n") + "\n", "scartissue-decision.txt", "text/plain;charset=utf-8");
});
(async () => {
  const version = ++loadVersion;
  try { const response = await fetch("data/evidence.json"); if (!response.ok) throw new Error("Evidence file unavailable (" + response.status + ")"); await load(await response.text(), true, version); }
  catch (error) { if (version === loadVersion) $("status").textContent = "Could not load bundled evidence. Import a validated bundle or retry: " + error.message; }
})();
