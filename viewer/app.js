/* caypollard viewer — shows the pool as thumbnails and, for one picture, the
   neighbours each representation returns. Reads data/derived/viewer/, computes
   nothing but comparisons of attributes between a query and its neighbours. */
"use strict";

const DATA = "../data/derived/viewer/";
const BATCH = 240;
const BASE_RE = /[(\[]/;

const state = {
  items: [], iconclass: {}, hubs: new Set(), tables: [], cache: new Map(),
  filtered: [], rendered: 0, order: "id", seed: 1,
  selected: -1, history: [], table: null, k: 30, regimes: new Set(),
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  for (const child of children) if (child !== null && child !== undefined) node.append(child);
  return node;
};

/* ---------- loading ---------- */

async function load() {
  const [items, tables] = await Promise.all([
    fetch(DATA + "items.json").then((r) => r.json()),
    fetch(DATA + "tables.json").then((r) => r.json()),
  ]);
  state.items = items.items;
  state.iconclass = items.iconclass || {};
  state.hubs = new Set(items.hubs || []);
  state.tables = tables;
  state.table = tables.length ? tables[0].slug : null;
  buildFilters();
  applyFilters();
  const fromHash = decodeURIComponent(location.hash.replace(/^#/, ""));
  if (fromHash) {
    const index = state.items.findIndex((it) => it.id === fromHash);
    if (index >= 0) select(index, false);
  }
}

async function loadTable(slug) {
  if (state.cache.has(slug)) return state.cache.get(slug);
  const meta = state.tables.find((t) => t.slug === slug);
  const [ids, nb, sc] = await Promise.all([
    fetch(DATA + `ids-${slug}.json`).then((r) => r.json()),
    fetch(DATA + `nb-${slug}.bin`).then((r) => r.arrayBuffer()).then((b) => new Uint32Array(b)),
    fetch(DATA + `sc-${slug}.bin`).then((r) => r.arrayBuffer()).then((b) => new Float32Array(b)),
  ]);
  const rowOf = new Int32Array(state.items.length).fill(-1);
  ids.forEach((itemIndex, row) => { rowOf[itemIndex] = row; });
  const table = { meta, ids, nb, sc, rowOf, k: meta.k };
  state.cache.set(slug, table);
  return table;
}

/* ---------- filters and grid ---------- */

function buildFilters() {
  const counts = { corpus: new Map(), col: new Map(), typ: new Map(), gen: new Map(), cen: new Map() };
  const bump = (map, key) => { if (key !== null && key !== undefined && key !== "") map.set(key, (map.get(key) || 0) + 1); };
  for (const it of state.items) {
    bump(counts.corpus, it.c); bump(counts.col, it.col); bump(counts.cen, it.cen);
    for (const t of it.typ) bump(counts.typ, t);
    for (const g of it.gen) bump(counts.gen, g);
  }
  const fill = (id, map, sortNumeric = false) => {
    const select = $(id);
    const entries = [...map.entries()].sort(sortNumeric ? (a, b) => a[0] - b[0] : (a, b) => b[1] - a[1]);
    for (const [key, n] of entries) {
      select.append(el("option", { value: key, text: `${sortNumeric ? key + "e s." : key} (${n})` }));
    }
  };
  fill("#f-corpus", counts.corpus); fill("#f-col", counts.col); fill("#f-typ", counts.typ);
  fill("#f-gen", counts.gen); fill("#f-cen", counts.cen, true);
}

function haystack(it) {
  if (!it._hay) {
    it._hay = [it.id, it.col, ...it.typ, ...it.gen, ...it.cre, ...it.mat, ...it.ic, it.mot || "", it.book || "", ...(it.nodes || [])]
      .join(" ").toLowerCase();
  }
  return it._hay;
}

function applyFilters() {
  const q = $("#q").value.trim().toLowerCase();
  const corpus = $("#f-corpus").value, col = $("#f-col").value, typ = $("#f-typ").value;
  const gen = $("#f-gen").value, cen = $("#f-cen").value;
  const out = [];
  state.items.forEach((it, i) => {
    if (corpus && it.c !== corpus) return;
    if (col && it.col !== col) return;
    if (typ && !it.typ.includes(typ)) return;
    if (gen && !it.gen.includes(gen)) return;
    if (cen && String(it.cen) !== cen) return;
    if (q && !haystack(it).includes(q)) return;
    out.push(i);
  });
  if (state.order === "random") shuffle(out, state.seed);
  state.filtered = out;
  state.rendered = 0;
  $("#grid").replaceChildren();
  $("#count").textContent = `${out.length.toLocaleString("fr-FR")} sur ${state.items.length.toLocaleString("fr-FR")}`;
  renderMore();
}

function shuffle(array, seed) {
  let s = seed >>> 0 || 1;
  const rand = () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function renderMore() {
  const grid = $("#grid");
  const end = Math.min(state.rendered + BATCH, state.filtered.length);
  const frag = document.createDocumentFragment();
  for (let n = state.rendered; n < end; n++) {
    const i = state.filtered[n];
    const it = state.items[i];
    const ratio = it.w && it.h ? it.w / it.h : 1;
    const tile = el("button", { class: "tile" + (it.c === "emblème" ? " emb" : "") + (i === state.selected ? " selected" : ""), "data-i": i, type: "button", title: it.id,
      style: `--r:${ratio.toFixed(3)}` },
      el("img", { src: DATA + it.th, loading: "lazy", alt: "" }),
      el("span", { class: "cap", text: shortId(it) }));
    frag.append(tile);
  }
  grid.append(frag);
  state.rendered = end;
}

const shortId = (it) => it.id.replace(/^rijksmuseum:/, "").replace(/^emblematica:/, "");

/* ---------- attributes shared between two pictures ---------- */

const base = (notation) => notation.split(BASE_RE)[0].trim();
const intersect = (a, b) => a.filter((x) => b.includes(x));

function shared(query, other) {
  // A notation most of the pool carries (86, mottoes; 31A, the human figure…) is a category:
  // sharing it is not sharing a subject. The build lists them, with the regime scripts' cut.
  const subjects = (it) => it.ic.map(base).filter((n) => !state.hubs.has(n));
  const subject = [...new Set(intersect(subjects(query), subjects(other)))];
  return {
    subject,
    corpus: query.c === other.c,
    col: query.col && query.col === other.col,
    typ: intersect(query.typ, other.typ).length > 0,
    cen: query.cen !== null && query.cen === other.cen,
    gen: intersect(query.gen, other.gen).length > 0,
    cre: intersect(query.cre, other.cre).length > 0,
    mat: intersect(query.mat, other.mat).length > 0,
    book: query.book && query.book === other.book,
  };
}

/* Regime chips: each is a predicate on what the query and the neighbour share. */
const REGIMES = [
  { key: "sujet+", label: "même sujet", test: (s) => s.subject.length > 0, excludes: "sujet-" },
  { key: "sujet-", label: "aucun sujet commun", test: (s) => s.subject.length === 0, excludes: "sujet+" },
  { key: "corpus-", label: "autre corpus", test: (s) => !s.corpus, excludes: "corpus+" },
  { key: "corpus+", label: "même corpus", test: (s) => s.corpus, excludes: "corpus-" },
  { key: "col-", label: "autre collection", test: (s) => !s.col, excludes: "col+" },
  { key: "col+", label: "même collection", test: (s) => s.col, excludes: "col-" },
  { key: "typ-", label: "autre type", test: (s) => !s.typ, excludes: "typ+" },
  { key: "typ+", label: "même type", test: (s) => s.typ, excludes: "typ-" },
  { key: "cen-", label: "autre siècle", test: (s) => !s.cen, excludes: "cen+" },
  { key: "cen+", label: "même siècle", test: (s) => s.cen, excludes: "cen-" },
  { key: "gen+", label: "même genre", test: (s) => s.gen, excludes: null },
  { key: "cre+", label: "même main", test: (s) => s.cre, excludes: null },
  { key: "book-", label: "autre livre", test: (s) => !s.book, excludes: null },
];

/* ---------- detail pane ---------- */

function select(index, push = true) {
  if (push && state.selected >= 0 && state.selected !== index) state.history.push(state.selected);
  state.selected = index;
  const it = state.items[index];
  location.hash = encodeURIComponent(it.id);
  document.body.classList.add("pane-open");
  $("#detail").hidden = false;
  $("#back").disabled = state.history.length === 0;
  $("#crumb").textContent = it.id;
  for (const tile of document.querySelectorAll(".tile.selected")) tile.classList.remove("selected");
  const tile = document.querySelector(`.tile[data-i="${index}"]`);
  if (tile) tile.classList.add("selected");
  renderDetail(it);
}

function closePane() {
  state.selected = -1;
  state.history = [];
  document.body.classList.remove("pane-open");
  $("#detail").hidden = true;
  history.replaceState(null, "", location.pathname);
  for (const tile of document.querySelectorAll(".tile.selected")) tile.classList.remove("selected");
}

function notationChip(notation) {
  const b = base(notation);
  let label = state.iconclass[b];
  let approx = false;
  if (!label) {
    // Fall back to the nearest labelled ancestor, marked as such.
    for (let cut = b.length - 1; cut >= 1 && !label; cut--) {
      const prefix = b.slice(0, cut);
      if (state.iconclass[prefix]) { label = state.iconclass[prefix]; approx = true; }
    }
  }
  const hub = state.hubs.has(b);
  return el("span", { class: "chip" + (hub ? " hub" : ""), title: (hub ? "catégorie portée par une large part du pool, ignorée pour « même sujet »\n" : "") + (label ? (approx ? `≈ ${label} (libellé de l'ancêtre)` : label) : "") },
    el("code", { text: notation }), label ? el("span", { text: (approx ? "≈ " : "") + label }) : null);
}

function describe(it) {
  const parts = [];
  if (it.typ.length) parts.push(it.typ.slice(0, 2).join(", "));
  if (it.gen.length) parts.push(it.gen[0]);
  if (it.cen) parts.push(`${it.cen}e s.`);
  if (it.col) parts.push(it.col);
  return parts.join(" · ");
}

function renderDetail(it) {
  const body = $("#detail-body");
  const meta = el("dl", { class: "meta" });
  const row = (key, value) => { if (value) meta.append(el("dt", { text: key }), el("dd", { text: value })); };
  row("corpus", it.c);
  row("collection", it.col);
  row("type", it.typ.join(", "));
  row("genre", it.gen.join(", "));
  row("siècle", it.cen ? `${it.cen}e` : "");
  row("créateur", it.cre.join(", "));
  row("matière", it.mat.join(", "));
  row("livre", it.book);

  // replaceChildren renders a null as the text "null"; el() filters them, this must too.
  body.replaceChildren(...[
    el("div", { class: "hero" }, el("img", { src: "../" + it.img, alt: it.id })),
    el("div", { class: "idline" }, el("span", { text: it.id }),
      it.url ? el("a", { href: it.url, target: "_blank", rel: "noopener", text: "source ↗" }) : null,
      el("button", { type: "button", class: "linkish", text: "sous le capot ↘", onclick: () => window.demo && window.demo.open(state.selected) })),
    it.mot ? el("p", { class: "motto", text: it.mot }) : null,
    meta,
    el("h2", { text: `Notations Iconclass (${it.ic.length})` }),
    el("div", { class: "chips" }, ...it.ic.map(notationChip)),
    it.nodes ? el("h2", { text: `Nœuds nommés (${it.nodes.length})` }) : null,
    it.nodes ? el("div", { class: "chips" }, ...it.nodes.map((n) => el("span", { class: "chip node", text: n }))) : null,
    el("h2", { text: "Voisins" }),
    neighbourControls(),
    el("div", { class: "regimes" }, ...REGIMES.map((r) => el("button", {
      type: "button", class: state.regimes.has(r.key) ? "on" : "", text: r.label,
      onclick: () => { toggleRegime(r); renderNeighbours(it); renderRegimeChips(); },
    }))),
    el("p", { class: "nbnote", id: "nbnote" }),
    el("div", { class: "nbs", id: "nbs" }),
  ].filter((node) => node !== null));
  renderNeighbours(it);
}

function renderRegimeChips() {
  const buttons = document.querySelectorAll(".regimes button");
  REGIMES.forEach((r, i) => buttons[i].classList.toggle("on", state.regimes.has(r.key)));
}

function toggleRegime(r) {
  if (state.regimes.has(r.key)) state.regimes.delete(r.key);
  else { state.regimes.add(r.key); if (r.excludes) state.regimes.delete(r.excludes); }
}

function neighbourControls() {
  const tableSelect = el("select", { onchange: (e) => { state.table = e.target.value; renderNeighbours(state.items[state.selected]); } },
    ...state.tables.map((t) => el("option", { value: t.slug, text: `${t.name} · ${t.rows.toLocaleString("fr-FR")}`, selected: t.slug === state.table ? "" : null })));
  const kSelect = el("select", { onchange: (e) => { state.k = Number(e.target.value); renderNeighbours(state.items[state.selected]); } },
    ...[10, 30, 100].map((k) => el("option", { value: k, text: `${k} premiers`, selected: k === state.k ? "" : null })));
  return el("div", { class: "nbctl" }, tableSelect, kSelect);
}

async function renderNeighbours(it) {
  const list = $("#nbs"), note = $("#nbnote");
  if (!list) return;
  if (!state.table) { note.textContent = "aucune représentation construite"; return; }
  note.textContent = "…";
  const table = await loadTable(state.table);
  if (state.selected < 0 || state.items[state.selected] !== it) return;
  const row = table.rowOf[state.items.indexOf(it)];
  list.replaceChildren();
  if (row < 0) {
    note.textContent = `Cette image n'est pas dans le pool de « ${table.meta.name} » (${table.meta.rows.toLocaleString("fr-FR")} images).`;
    return;
  }
  const active = REGIMES.filter((r) => state.regimes.has(r.key));
  const k = Math.min(state.k, table.k);
  const rows = [];
  let kept = 0;
  for (let n = 0; n < table.k && kept < k; n++) {
    const nbRow = table.nb[row * table.k + n];
    const other = state.items[table.ids[nbRow]];
    const s = shared(it, other);
    if (active.every((r) => r.test(s))) {
      rows.push({ rank: n + 1, score: table.sc[row * table.k + n], other, s });
      kept++;
    }
  }
  const scanned = active.length ? `dans les ${table.k} premiers voisins` : "";
  note.textContent = active.length
    ? `${rows.length} voisin${rows.length > 1 ? "s" : ""} ${scanned} satisfont le régime · ${table.meta.method || ""}`
    : `${table.meta.name} · pool ${table.meta.rows.toLocaleString("fr-FR")} · ${table.meta.dimension} dimensions · ${table.meta.method || ""}`;
  if (!rows.length) list.append(el("p", { class: "empty", text: "Rien dans les " + table.k + " premiers voisins." }));
  for (const r of rows) list.append(neighbourRow(r));
}

function neighbourRow({ rank, score, other, s }) {
  const badges = [];
  if (s.subject.length) badges.push(el("span", { class: "badge subject", text: "sujet " + s.subject.slice(0, 3).join(" ") + (s.subject.length > 3 ? " …" : "") }));
  if (s.cre) badges.push(el("span", { class: "badge", text: "même main" }));
  if (s.book) badges.push(el("span", { class: "badge", text: "même livre" }));
  if (s.typ) badges.push(el("span", { class: "badge", text: "type" }));
  if (s.gen) badges.push(el("span", { class: "badge", text: "genre" }));
  if (s.col) badges.push(el("span", { class: "badge", text: "collection" }));
  if (s.cen) badges.push(el("span", { class: "badge", text: "siècle" }));
  if (s.mat) badges.push(el("span", { class: "badge", text: "matière" }));
  if (!s.corpus) badges.push(el("span", { class: "badge", text: "autre corpus" }));
  const index = state.items.indexOf(other);
  return el("button", { class: "nb", type: "button", onclick: () => select(index) },
    el("span", { class: "rank" }, el("b", { text: String(rank) }), el("span", { text: score.toFixed(3) })),
    el("span", { class: "th" }, el("img", { src: DATA + other.th, loading: "lazy", alt: "" })),
    el("span", { class: "who" },
      el("div", { class: "id", text: other.id }),
      el("div", { class: "desc", text: describe(other) || (other.mot || "") }),
      el("div", { class: "badges" }, ...badges)));
}

/* ---------- wiring ---------- */

$("#grid").addEventListener("click", (e) => {
  const tile = e.target.closest(".tile");
  if (tile) select(Number(tile.dataset.i));
});
$("#close").addEventListener("click", closePane);
$("#back").addEventListener("click", () => { if (state.history.length) select(state.history.pop(), false); $("#back").disabled = state.history.length === 0; });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.selected >= 0) closePane();
  if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && state.selected >= 0 && document.activeElement.tagName !== "INPUT") {
    const pos = state.filtered.indexOf(state.selected);
    if (pos < 0) return;
    const next = pos + (e.key === "ArrowRight" ? 1 : -1);
    if (next >= 0 && next < state.filtered.length) {
      while (state.rendered <= next) renderMore();
      select(state.filtered[next], false);
      document.querySelector(`.tile[data-i="${state.filtered[next]}"]`)?.scrollIntoView({ block: "nearest" });
    }
  }
});
let debounce;
$("#q").addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(applyFilters, 180); });
for (const id of ["#f-corpus", "#f-col", "#f-typ", "#f-gen", "#f-cen"]) $(id).addEventListener("change", applyFilters);
$("#order").addEventListener("change", (e) => { state.order = e.target.value; state.seed = (Math.random() * 2 ** 31) | 0; applyFilters(); });
$("#reset").addEventListener("click", () => {
  $("#q").value = "";
  for (const id of ["#f-corpus", "#f-col", "#f-typ", "#f-gen", "#f-cen"]) $(id).value = "";
  applyFilters();
});
new IntersectionObserver((entries) => { if (entries[0].isIntersecting && state.rendered < state.filtered.length) renderMore(); }, { rootMargin: "1200px" })
  .observe($("#sentinel"));

window.addEventListener("hashchange", () => {
  const id = decodeURIComponent(location.hash.replace(/^#/, ""));
  const index = id ? state.items.findIndex((it) => it.id === id) : -1;
  if (index >= 0 && index !== state.selected) select(index);
});

load().catch((error) => { $("#count").textContent = "erreur : " + error.message; console.error(error); });
