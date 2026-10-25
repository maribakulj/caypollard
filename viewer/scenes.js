// Scene-graph page: two readers side by side on the relations bench. Reads data/derived/viewer/scenes/scenes.json
// (scripts/build_scene_viewer.py); computes nothing but counts.
const DATA = "../data/derived/viewer/scenes/";
const FR = {
  // roles
  "man": "homme", "woman": "femme", "old man": "vieillard", "old woman": "vieille femme", "child": "enfant",
  "infant": "nourrisson", "angel": "ange", "soldier": "soldat", "crowd": "foule", "animal": "animal",
  // postures
  "standing": "debout", "seated": "assis", "kneeling": "agenouillé", "lying": "couché", "bending": "penché",
  "flying": "en vol", "riding": "à cheval",
  // relations
  "holds": "tient", "carries": "porte", "embraces": "étreint", "touches": "touche", "reaches toward": "tend la main vers",
  "points at": "désigne", "looks at": "regarde", "kneels before": "s'agenouille devant", "bows to": "s'incline devant",
  "gives to": "donne à", "blesses": "bénit", "supports": "soutient", "mourns over": "pleure sur", "strikes": "frappe",
  "leads": "conduit", "lies before": "est couché devant", "is above": "est au-dessus de", "stands beside": "est à côté de",
};
const SUBJ = {
  "annonciation": "Annonciation", "visitation": "Visitation", "nativite": "Nativité", "bergers": "Adoration des bergers",
  "mages": "Adoration des mages", "presentation": "Présentation au Temple", "fuite-egypte": "Fuite en Égypte",
  "sainte-famille": "Sainte Famille", "gethsemani": "Gethsémani", "ecce-homo": "Ecce homo", "portement": "Portement de croix",
  "crucifixion": "Crucifixion", "descente": "Descente de croix", "deploration": "Déploration",
  "mise-au-tombeau": "Mise au tombeau", "resurrection": "Résurrection",
};
const fr = (w) => FR[w] || w;
const subj = (s) => SUBJ[s] || s;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const $ = (id) => document.getElementById(id);

let D, index = {}, filter = "all", current = null;

function pref(key, fallback) {
  try { return localStorage.getItem("scenes." + key) || fallback; } catch { return fallback; }
}
function keep(key, value) {
  try { localStorage.setItem("scenes." + key, value); } catch { /* private window */ }
}

function outcome(i) {
  const a = D.readers[$("ra").value].rows[i].hit, b = D.readers[$("rb").value].rows[i].hit;
  return a && b ? "both" : a ? "onlyA" : b ? "onlyB" : "none";
}

function visible() {
  const s = $("subj").value;
  return D.items.map((_, i) => i).filter((i) => (!s || D.items[i].subject === s) && (filter === "all" || outcome(i) === filter));
}

function drawFilters() {
  const s = $("subj").value;
  const pool = D.items.map((_, i) => i).filter((i) => !s || D.items[i].subject === s);
  const n = { all: pool.length, both: 0, onlyA: 0, onlyB: 0, none: 0 };
  pool.forEach((i) => n[outcome(i)]++);
  const A = D.readers[$("ra").value].label, B = D.readers[$("rb").value].label;
  const names = { all: "toutes", both: "les deux réussissent", onlyA: `seul ${A} réussit`, onlyB: `seul ${B} réussit`, none: "les deux échouent" };
  $("filters").innerHTML = Object.keys(names).map((k) =>
    `<button data-f="${k}" class="${k === filter ? "on" : ""}">${esc(names[k])} · ${n[k]}</button>`).join("") +
    `<span class="muted">réussite = l'image la plus proche par le graphe a le même sujet</span>`;
  $("filters").querySelectorAll("button").forEach((b) => (b.onclick = () => { filter = b.dataset.f; keep("filter", filter); draw(); }));
}

function drawBySubject() {
  const ra = D.readers[$("ra").value], rb = D.readers[$("rb").value];
  const by = {};
  D.items.forEach((it, i) => {
    const r = (by[it.subject] ||= { n: 0, a: 0, b: 0 });
    r.n++; r.a += ra.rows[i].hit; r.b += rb.rows[i].hit;
  });
  const rows = Object.entries(by).sort((x, y) => (x[1].a + x[1].b) / x[1].n - (y[1].a + y[1].b) / y[1].n);
  $("bysubject").innerHTML = `<table><tr><th>sujet</th><th>${esc(ra.label)}</th><th>${esc(rb.label)}</th></tr>` +
    rows.map(([s, r]) => `<tr data-s="${s}"><td>${esc(subj(s))}</td><td>${r.a}/${r.n}</td><td>${r.b}/${r.n}</td></tr>`).join("") +
    `</table>`;
  $("bysubject").querySelectorAll("tr[data-s]").forEach((tr) => (tr.onclick = () => { $("subj").value = tr.dataset.s; draw(); }));
}

function dots(i) {
  const a = D.readers[$("ra").value].rows[i].hit, b = D.readers[$("rb").value].rows[i].hit;
  return `<span class="dots"><span class="dot ${a ? "ok" : "ko"}">A</span><span class="dot ${b ? "ok" : "ko"}">B</span></span>`;
}

function drawGrid() {
  $("grid").innerHTML = visible().map((i) => {
    const id = D.items[i].id;
    return `<a href="#${esc(id)}" class="${id === current ? "sel" : ""}" title="${esc(subj(D.items[i].subject))}">` +
      `<img loading="lazy" src="${DATA}thumbs/${esc(id)}.jpg" alt="">${dots(i)}</a>`;
  }).join("") || `<p class="muted">Aucune image pour ce filtre.</p>`;
}

function card(key, i) {
  const R = D.readers[key], r = R.rows[i], me = D.items[i].subject;
  const figs = r.figures.map((f) => `<li><b>${esc(f.id || "?")}</b> ${esc(fr(f.role || "?"))} · ${esc(fr(f.posture || "?"))}</li>`).join("");
  const rels = r.relations.map((x) => `<li>${esc(x.a)} <b>${esc(fr(x.rel))}</b> ${esc(x.b)}</li>`).join("");
  const near = r.nearest.map((id) => {
    const same = D.items[index[id]].subject === me;
    return `<a href="#${esc(id)}"><img class="${same ? "same" : "other"}" src="${DATA}thumbs/${esc(id)}.jpg" alt="">` +
      `${esc(subj(D.items[index[id]].subject))}</a>`;
  }).join("");
  return `<div class="card"><h3>${esc(R.label)} <span class="badge ${r.hit ? "ok" : "ko"}">${r.hit ? "réussi" : "échec"}</span></h3>` +
    `<div class="muted">sujet nommé : « ${esc(r.subject || "—")} » · premier même sujet au rang ${r.rank}` +
    `${r.seconds ? ` · ${Math.round(r.seconds)} s` : ""}</div>` +
    `<div><b>Figures</b> (${r.figures.length})</div><ul>${figs || "<li class='muted'>aucune</li>"}</ul>` +
    `<div><b>Relations</b> (${r.relations.length})</div><ul>${rels || "<li class='muted'>aucune</li>"}</ul>` +
    `<div><b>Images les plus proches</b> (vert = même sujet)</div><div class="near">${near}</div></div>`;
}

function drawDetail() {
  if (!current || !(current in index)) return;
  const i = index[current];
  $("detail").innerHTML = `<div><b>${esc(subj(D.items[i].subject))}</b> <span class="muted">${esc(current)}</span></div>` +
    `<img class="big" src="${DATA}thumbs/${esc(current)}.jpg" alt="">` +
    `<div class="cols">${card($("ra").value, i)}${card($("rb").value, i)}</div>`;
}

function draw() {
  keep("ra", $("ra").value); keep("rb", $("rb").value); keep("subj", $("subj").value);
  drawFilters(); drawBySubject(); drawGrid(); drawDetail();
}

window.addEventListener("hashchange", () => { current = decodeURIComponent(location.hash.slice(1)); drawGrid(); drawDetail(); });

fetch(DATA + "scenes.json").then((r) => r.json()).then((data) => {
  D = data;
  D.items.forEach((it, i) => (index[it.id] = i));
  const keys = Object.keys(D.readers);
  for (const sel of ["ra", "rb"]) $(sel).innerHTML = keys.map((k) => `<option value="${k}">${esc(D.readers[k].label)}</option>`).join("");
  $("ra").value = keys.includes(pref("ra", "")) ? pref("ra", "") : keys[0];
  $("rb").value = keys.includes(pref("rb", "")) ? pref("rb", "") : keys[1] || keys[0];
  [...new Set(D.items.map((it) => it.subject))].sort().forEach((s) => $("subj").insertAdjacentHTML("beforeend", `<option value="${s}">${esc(subj(s))}</option>`));
  $("subj").value = pref("subj", "");
  filter = pref("filter", "all");
  for (const sel of ["ra", "rb", "subj"]) $(sel).onchange = draw;
  current = decodeURIComponent(location.hash.slice(1)) || null;
  draw();
});
