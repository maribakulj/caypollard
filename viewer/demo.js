/* Sous le capot — analyse one picture through every representation and search by channel.
   Talks to /api/demo/*; reuses the grid's items, `el`, `select`, `shared` from app.js. */
"use strict";

(function () {
  const ORDER = ["source", "renders", "traits", "pixels", "silhouette", "segmentation", "descriptors", "formes", "relations",
    "composition", "repetition", "groups", "record", "palette", "signal", "nommes", "pose", "mix", "search"];
  const SWATCH = { noir: "#000", blanc: "#fff", rouge: "#c42021", vert: "#2a8c3c", jaune: "#f0c828", bleu: "#244aa8",
    brun: "#784a26", orange: "#de7820", rose: "#e896ac", violet: "#763694", gris: "#808080" };
  const CHANNEL_STEP = { croquis: "traits", traits: "traits", pixels: "pixels", silhouette: "silhouette", formes: "formes", relations: "relations", composition: "composition",
    repetition: "repetition", record: "record", palette: "palette", signal: "signal", nommes: "nommes", pose: "pose", mix: "mix" };
  const PRESETS = {
    image: { pixels: 1 },
    trait: { croquis: 1 },
    "traits (signes)": { traits: 1 },
    symbolique: { record: 1 },
    grammaire: { formes: 1, relations: 1 },
    pose: { pose: 1 },
    semantique: { nommes: 1 },
    "les deux": { pixels: 1, nommes: 1 },
    surface: { palette: 1, signal: 1 },
    mix: { mix: 1 },
  };

  const d = {
    meta: null, file: null, item: null, params: {}, analysis: null,
    weights: { pixels: 1 }, k: 20, preset: "image",
  };

  const fmt = (v, n = 3) => (typeof v === "number" ? v.toFixed(n) : String(v));
  const signColour = (sign) => { const h = (sign * 0.6180339887) % 1; return `hsl(${Math.round(h * 360)} 65% 60%)`; };

  async function meta() {
    if (!d.meta) d.meta = await fetch("/api/demo/explain").then((r) => r.json());
    return d.meta;
  }

  /* ---------- open / close ---------- */
  async function open(itemIndex) {
    await meta();
    d.item = itemIndex >= 0 && itemIndex !== null && itemIndex !== undefined ? state.items[itemIndex] : null;
    $("#demo-item").textContent = d.item ? d.item.id : "aucune";
    if (d.item) {
      d.file = null;
      $("#drop-preview").src = DATA + d.item.th;
      $("#drop-preview").hidden = false;
      $("#drop-text").hidden = true;
    }
    $("#namer").disabled = !d.meta.namer_available;
    if (!d.meta.namer_available) $("#namer").checked = false;
    buildParams();
    document.body.classList.add("demo-open");
    $("#demo").hidden = false;
    if (state.selected >= 0) closePane();
    window.scrollTo({ top: 0 });
  }
  function close() {
    document.body.classList.remove("demo-open");
    $("#demo").hidden = true;
  }

  /* ---------- parameters ---------- */
  function buildParams() {
    const box = $("#params");
    box.replaceChildren();
    const docs = {};
    for (const step of Object.values(d.meta.steps)) for (const [k, v] of Object.entries(step.params || {})) docs[k] = v;
    for (const [key, value] of Object.entries(d.meta.defaults)) {
      const current = key in d.params ? d.params[key] : value;
      const input = el("input", { type: "number", step: "any", value: current, "data-key": key,
        oninput: (e) => { const v = Number(e.target.value); if (Number.isFinite(v)) d.params[key] = v; e.target.parentElement.classList.toggle("changed", v !== value); paramsState(); } });
      box.append(el("label", { class: current !== value ? "changed" : "", title: docs[key] || key },
        el("span", { text: `${key} — ${docs[key] || ""}` }), input));
    }
    paramsState();
  }
  function paramsState() {
    const changed = Object.entries(d.params).filter(([k, v]) => v !== d.meta.defaults[k]).length;
    $("#params-state").textContent = changed ? `· ${changed} modifié${changed > 1 ? "s" : ""} : les canaux touchés ne sont plus ceux du pool` : "· valeurs du pool";
  }

  /* ---------- analyse ---------- */
  async function run() {
    const status = $("#demo-status");
    if (!d.file && !d.item) { status.textContent = "choisis une image, ou une image du pool"; return; }
    $("#run").disabled = true;
    status.textContent = "analyse en cours… (DINOv2, découpe, canaux" + ($("#namer").checked ? ", nommeur Sonnet" : "") + ")";
    const query = new URLSearchParams({ params: JSON.stringify(d.params), namer: $("#namer").checked ? "sonnet" : "none" });
    let response;
    try {
      if (d.file) {
        response = await fetch("/api/demo/analyse?" + query, { method: "POST", body: d.file, headers: { "Content-Type": "application/octet-stream" } });
      } else {
        query.set("item", d.item.id);
        response = await fetch("/api/demo/analyse?" + query);
      }
      d.analysis = await response.json();
      if (d.analysis.error) throw new Error(d.analysis.error);
    } catch (error) {
      status.textContent = "erreur : " + error.message;
      $("#run").disabled = false;
      return;
    }
    $("#run").disabled = false;
    const t = d.analysis.timings;
    status.textContent = `analysé en ${Object.values(t).reduce((a, b) => a + b, 0).toFixed(1)} s`;
    renderSteps();
    search();
  }

  /* ---------- steps ---------- */
  function renderSteps() {
    const a = d.analysis;
    const byKey = Object.fromEntries(a.steps.map((s) => [s.key, s]));
    const container = $("#steps");
    container.replaceChildren();
    ORDER.forEach((key, n) => {
      const doc = d.meta.steps[key];
      const step = byKey[key] || {};
      container.append(card(n + 1, key, doc, step, a));
    });
  }

  function tag(text, kind = "") { return el("span", { class: "tag " + kind, text }); }

  function card(n, key, doc, step, a) {
    const tags = [];
    if (step.dimension) tags.push(tag(`${step.dimension} nombres`));
    if ("available" in step && !step.available) tags.push(tag("non calculable pour cette image", "bad"));
    if ("faithful" in step) tags.push(step.faithful ? tag("paramètres du pool", "ok") : tag("paramètres modifiés", "warn"));
    if ("comparable" in step && !step.comparable) tags.push(tag("dimension différente : incomparable au pool", "bad"));
    const channel = Object.keys(CHANNEL_STEP).find((c) => CHANNEL_STEP[c] === key);
    if (channel && a.reproduction && a.reproduction[channel] !== undefined && a.reproduction[channel] !== null) {
      const c = a.reproduction[channel];
      tags.push(tag(`reproduction du pool : cos ${c.toFixed(4)}`, c > 0.9999 ? "ok" : c > 0.9 ? "warn" : "bad"));
    }
    if (a.timings && a.timings[key] !== undefined) tags.push(tag(`${a.timings[key]} s`));
    const explain = el("div", { class: "explain" },
      el("p", { text: doc.what }),
      doc.why ? el("details", { class: "more" }, el("summary", { text: "pourquoi, et ce que ça vaut" }), el("p", { text: doc.why })) : null,
      el("p", {}, el("span", { class: "muted", text: "code : " }), el("code", { text: doc.code || "" })));
    const body = VISUALS[key] ? VISUALS[key](step, a) : null;
    return el("section", { class: "card", id: "step-" + key },
      el("h3", {}, el("span", { class: "no", text: String(n).padStart(2, "0") }), el("span", { text: doc.title }), el("span", { class: "tags" }, ...tags)),
      explain, body);
  }

  const figure = (src, label, big = false) => el("div", { class: "figure" + (big ? " big" : "") }, el("img", { src, alt: label }), el("span", { text: label }));
  const bars = (rows, { max, labelWidth } = {}) => {
    const top = max || Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
    return el("div", { class: "bars" }, ...rows.map((r) => el("div", { class: "brow" },
      el("span", { text: r.name, title: r.name, style: labelWidth ? `width:${labelWidth}px` : "" }),
      el("span", {}, el("i", { style: `width:${Math.max(1, (Math.abs(r.value) / top) * 100)}%` })),
      el("span", { class: "v", text: fmt(r.value) }))));
  };
  const hist = (values) => { const top = Math.max(...values, 1e-9); return el("div", { class: "hist" }, ...values.map((v) => el("i", { style: `height:${(v / top) * 100}%`, title: fmt(v) }))); };
  const table = (head, rows) => el("table", { class: "small" }, el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h })))),
    el("tbody", {}, ...rows.map((r) => el("tr", {}, ...r.map((c) => c instanceof Node ? el("td", {}, c) : el("td", { text: String(c) }))))));

  const VISUALS = {
    source: (s) => el("div", { class: "visuals" }, figure(s.image, `${s.size[0]} × ${s.size[1]} px`, true)),
    renders: (s) => el("div", { class: "visuals" }, ...Object.entries(s.images || {}).map(([m, src]) => figure(src, { gray: "gris", edges: "contours (Sobel)", shape: "forme (48×48 étirée)", silhouette: "silhouette (seuil médian)", mass: "masse (Otsu, encre locale)" }[m] || m))),
    traits: (s) => el("div", {},
      el("div", { class: "visuals" }, ...Object.entries(s.images || {}).map(([m, src]) => figure(src, { contours: `contours (${((s.ink || 0) * 100).toFixed(1)} % d'encre)`, traits: `${s.counts.traits} traits`, croquis: `croquis : ${s.counts.croquis} traits les plus longs`, pictogramme: `pictogramme : ${s.counts.pictogramme} traits` }[m] || m, true))),
      el("p", { class: "muted", text: "les 12 premiers traits du croquis, décrits par : " + (s.features || []).join(" · ") }),
      table(["trait", "signe", "points", "longueur px", ...(s.features || []).slice(0, 8)], (s.strokes || []).map((t, i) => [t.n, s.signs ? s.signs[i] : "—", t.points, t.length, ...t.descriptor.slice(0, 8).map((v) => v.toFixed(2))])),
      el("p", { class: "muted", text: "deux canaux de recherche en sortent : « croquis », le dessin au trait passé dans DINOv2 ; « traits », l'histogramme des signes de trait pondéré par la longueur." })),
    pixels: (s) => el("div", { class: "visuals" }, el("div", {}, el("p", { class: "muted", text: `${s.model || ""} · ${s.device || ""} · norme ${fmt(s.norm || 0, 3)}` }),
      el("p", { class: "muted", text: "les 32 premières des 768 valeurs (elles n'ont pas de nom : c'est ce qu'un réseau rend)" }), hist((s.preview || []).map((v) => Math.abs(v))))),
    silhouette: (s) => el("div", { class: "visuals" }, el("div", {}, el("p", { class: "muted", text: "32 premières valeurs du plongement de la silhouette" }), hist((s.preview || []).map((v) => Math.abs(v))))),
    segmentation: (s) => {
      if (!s.regions) return null;
      const rows = s.regions.map((r) => [r.n, r.tone, fmt(r.area, 4), fmt(r.elongation, 2), fmt(r.solidity, 2), r.holes, fmt(r.scale_vs_median, 2),
        el("span", {}, el("span", { class: "dot", style: `background:${signColour(r.sign_v2)}` }), " " + r.sign_v2), r.sign_pool]);
      return el("div", {},
        el("div", { class: "visuals" }, figure(s.smoothed, "lissé et étiré (ce qui est seuillé)"), figure(s.overlay, `${s.count} régions, colorées par signe v2` + (s.paired ? "" : " (appariement incertain)")),
          s.reconstruction ? figure(s.reconstruction, "reconstruction depuis les descripteurs (profil radial orienté)") : null),
        table(["n°", "ton", "aire", "élong.", "rempl.", "trous", "échelle/méd.", "signe v2", "signe pool"], rows));
    },
    descriptors: (s) => el("p", { class: "muted", text: (s.names || []).join(" · ") }),
    formes: (s) => s.available ? el("div", { class: "visuals" }, el("div", {}, el("p", { class: "muted", text: "poids des signes (aire cumulée, normalisée) — le vecteur a 256 cases, presque toutes à zéro" }),
      bars((s.entries || []).map((e) => ({ name: "signe " + e.name, value: e.value })))),
      el("div", {}, el("p", { class: "muted", text: "distance de chaque région à son signe (espace standardisé)" }), bars((s.distance_to_sign || []).map((v, i) => ({ name: "région " + (i + 1), value: v }))))) : null,
    composition: (s) => el("div", { class: "visuals" }, figure(s.heatmap, "grille 8×8 de la masse d'encre (sombre = plus)"),
      el("div", {}, el("p", { class: "muted", text: "profil des lignes (haut → bas)" }), hist(s.rows || []), el("p", { class: "muted", text: "profil des colonnes (gauche → droite)" }), hist(s.columns || [])),
      bars(Object.entries(s.scalars || {}).map(([k, v]) => ({ name: k, value: v })), { labelWidth: 170 })),
    repetition: (s) => s.available ? el("div", { class: "visuals" }, bars(Object.entries(s.buckets || {}).map(([k, v]) => ({ name: "signes vus " + k, value: v }))),
      s.repeated && s.repeated.length ? table(["signe (pool)", "occurrences"], s.repeated.map((r) => [el("span", {}, el("span", { class: "dot", style: `background:${signColour(r.sign)}` }), " " + r.sign), r.count])) : el("p", { class: "muted", text: "aucun signe ne revient : le profil est tout dans « une fois »" })) : null,
    groups: (s) => s.groups && s.groups.length ? table(["amas (n° de régions)", "signe composite"], s.groups.map((g) => [g.members.join(" + "), g.composite])) : el("p", { class: "muted", text: "aucun amas d'au moins deux parties : pas de signe composite, donc pas de record" }),
    record: (s) => el("p", { class: "muted", text: s.available ? "128 composites ×1 · 87 composition ×2 · 1 028 répétition ×½, normalisés puis concaténés" : "indisponible : il manque un des trois canaux" }),
    palette: (s) => el("div", { class: "visuals" }, figure(s.balanced, "après équilibrage grey-world (ce qui est classé)"),
      el("div", { class: "bars" }, ...(s.terms || []).map((t) => el("div", { class: "brow" },
        el("span", {}, el("span", { class: "swatch", style: `background:${SWATCH[t.name] || "#888"}` }), t.name),
        el("span", {}, el("i", { style: `width:${Math.max(1, t.value * 100)}%` })),
        el("span", { class: "v", text: fmt(t.value) }))))),
    signal: (s) => el("div", { class: "visuals" }, el("div", {}, el("p", { class: "muted", text: "histogramme de luminance, 12 cases (sombre → clair)" }), hist(s.histogram || [])), bars(Object.entries(s.scalars || {}).map(([k, v]) => ({ name: k, value: v })), { labelWidth: 120 })),
    nommes: (s) => {
      if (s.error && !(s.nodes || []).length) return el("p", { class: "muted", text: "pas de nœuds : " + s.error });
      const grid = el("div", { class: "cellgrid" }, ...(s.cells || []).map((c, i) => el("div", {}, el("b", { text: c }), ...(s.nodes || []).filter((n) => n.cell === i).map((n) => el("span", { class: "chip node", text: n.name })))));
      const counted = (s.nodes || []).slice(0, s.counted || 3).map((n) => n.name);
      return el("div", { class: "visuals" }, grid, el("div", {}, el("p", { class: "muted", text: `nommeur : ${s.model || "?"} · ${(s.nodes || []).length} nœuds, les ${s.counted} premiers comptés : ${counted.join(", ")}` }),
        el("details", { class: "more" }, el("summary", { text: "réponse brute" }), el("pre", { text: s.raw || "" }))));
    },
    mix: (s) => el("p", { class: "muted", text: s.available ? "nommés 116 + pixels 768 + palette 11 + signal 16 + formes v4 256, à poids égaux" : "indisponible : il faut les nœuds nommés (active le nommeur)" }),
    relations: (s) => {
      if (!s.triples || !s.triples.length) return el("p", { class: "muted", text: "moins de deux régions : aucune relation" });
      const rel = { "au-dessus": "est au-dessus de", "en-dessous": "est en dessous de", droite: "est à droite de", gauche: "est à gauche de", touche: "touche", contient: "contient", dans: "est dans" };
      return el("div", {}, el("p", { class: "muted", text: `${s.triples.length} paires ordonnées entre les ${Math.min(10, (s.triples.length ? Math.round((1 + Math.sqrt(1 + 4 * s.triples.length)) / 2) : 0))} plus grandes régions · ${s.kept} triplets font partie des ${s.dimension} que le pool garde (support ≥ 30) · signes sur ${s.signs}` }),
        table(["région", "signe", "relation", "région", "signe", "gardé"], s.triples.map((t) => [t.a, t.sign_a, rel[t.relation] || t.relation, t.b, t.sign_b, t.kept ? "oui" : "—"])));
    },
    pose: (s) => el("div", { class: "visuals" }, figure(s.skeleton, `${s.detected} personne(s) détectée(s) · ${s.usable} exploitable(s) · ${s.device || ""}`, true),
      el("div", {}, ...(s.figures || []).map((f, i) => el("div", {}, el("p", { class: "muted", text: `figure ${i + 1} · score ${f.score} · épaules/torse ${f.shoulders_over_torso}` }),
        table(["segment", "angle au torse", ""], f.segments.map((g) => [g.name, g.present ? `${g.angle}°` : "absent", g.present ? "●" : "○"])))),
        (s.figures || []).length ? null : el("p", { class: "muted", text: "aucune figure avec les deux épaules et une hanche : pas de vecteur de pose (c'est le cas de 72 % des images)" }))),
    search: () => searchPanel(),
  };

  /* ---------- search ---------- */
  function searchPanel() {
    const a = d.analysis;
    const available = new Set(Object.keys(a.channels));
    const regimes = el("div", { class: "regimes-demo" }, ...Object.keys(PRESETS).map((p) => el("button", { type: "button", class: d.preset === p ? "on" : "", text: p,
      onclick: () => { d.preset = p; d.weights = { ...PRESETS[p] }; renderSteps(); search(); } })));
    const weights = el("div", { class: "weights" }, ...d.meta.channels.map((c) => {
      const on = available.has(c.key);
      const value = d.weights[c.key] || 0;
      return el("label", { class: on ? "" : "off", title: c.what },
        el("span", { text: `${c.key} · ${c.regime}` + (on ? "" : " (indisponible)"), title: c.name + " — " + c.what }),
        el("input", { type: "range", min: 0, max: 2, step: 0.25, value, disabled: on ? null : "", oninput: (e) => { d.weights[c.key] = Number(e.target.value); d.preset = "personnalisé"; e.target.nextSibling.textContent = e.target.value; } }),
        el("span", { text: String(value) }));
    }));
    const k = el("select", { onchange: (e) => { d.k = Number(e.target.value); search(); } }, ...[10, 20, 50].map((n) => el("option", { value: n, text: `${n} premiers`, selected: n === d.k ? "" : null })));
    return el("div", {}, regimes, weights, el("div", { class: "nbctl" }, k, el("button", { type: "button", class: "primary", text: "Chercher", onclick: search })),
      el("p", { class: "nbnote", id: "search-note" }), el("div", { class: "results", id: "results" }));
  }

  async function search() {
    const a = d.analysis;
    if (!a) return;
    const note = $("#search-note"), box = $("#results");
    if (!note) return;
    const active = Object.entries(d.weights).filter(([k, w]) => w > 0 && a.channels[k]);
    if (!active.length) { note.textContent = "aucun canal actif (ou ses vecteurs manquent pour cette image)"; box.replaceChildren(); return; }
    note.textContent = "recherche…";
    const columns = [...active.map(([k, w]) => ({ title: k, weights: { [k]: 1 } }))];
    if (active.length > 1) columns.push({ title: "mélange pondéré", weights: Object.fromEntries(active) });
    const query = d.item ? d.item : null;
    const answers = await Promise.all(columns.map((c) => fetch("/api/demo/search", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: a.token, weights: c.weights, k: d.k, exclude: query ? query.id : null }) }).then((r) => r.json())));
    box.replaceChildren(...columns.map((c, i) => column(c, answers[i], query)));
    note.textContent = `cosinus exact sur le pool, ${d.k} premiers par colonne · ` + columns.map((c, i) => `${c.title} : pool ${(answers[i].pool || 0).toLocaleString("fr-FR")}`).join(" · ");
  }

  function column(c, answer, query) {
    const list = el("div", { class: "list" });
    if (answer.error) list.append(el("p", { class: "empty", text: answer.error }));
    for (const [n, r] of (answer.results || []).entries()) {
      const it = state.items[r.i];
      const badges = [];
      if (query) {
        const s = shared(query, it);
        if (s.subject.length) badges.push(el("span", { class: "badge subject", text: "sujet " + s.subject.slice(0, 2).join(" ") }));
        if (s.cre) badges.push(el("span", { class: "badge", text: "main" }));
        if (s.typ) badges.push(el("span", { class: "badge", text: "type" }));
        if (s.gen) badges.push(el("span", { class: "badge", text: "genre" }));
        if (s.col) badges.push(el("span", { class: "badge", text: "coll." }));
        if (!s.corpus) badges.push(el("span", { class: "badge", text: "autre corpus" }));
      }
      list.append(el("button", { class: "hit", type: "button", onclick: () => select(r.i) },
        el("span", { class: "r" }, el("b", { text: String(n + 1) }), el("span", { text: r.score.toFixed(3) })),
        el("span", { class: "t" }, el("img", { src: DATA + it.th, loading: "lazy", alt: "" })),
        el("span", { class: "w" }, el("div", { class: "id", text: it.id }), el("div", { text: describe(it) || (it.mot || "").slice(0, 40) }), el("div", { class: "badges" }, ...badges))));
    }
    const label = d.meta.channels.find((x) => x.key === c.title);
    return el("div", { class: "column" }, el("h4", {}, el("span", { text: label ? label.name : c.title }), c.weights && Object.keys(c.weights).length > 1 ? el("span", { text: " " + Object.entries(c.weights).map(([k, w]) => `${k}×${w}`).join(" + ") }) : null), list);
  }

  /* ---------- wiring ---------- */
  $("#open-demo").addEventListener("click", () => open(state.selected));
  $("#close-demo").addEventListener("click", close);
  $("#run").addEventListener("click", run);
  $("#params-reset").addEventListener("click", () => { d.params = {}; buildParams(); });
  const drop = $("#drop");
  const setFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    d.file = file; d.item = null;
    $("#demo-item").textContent = "aucune (image déposée)";
    $("#drop-preview").src = URL.createObjectURL(file); $("#drop-preview").hidden = false; $("#drop-text").hidden = true;
  };
  $("#file").addEventListener("change", (e) => setFile(e.target.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); setFile(e.dataTransfer.files[0]); });

  window.demo = { open, close };
})();
