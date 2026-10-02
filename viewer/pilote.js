// Pilot viewer: one picture, its proposed figures and objects, and the relations computed
// from them. Points can be dragged, placed or removed; saving sends the corrections to the
// server, which recomputes the record with caypollard.figures.build_record.
(() => {
  const DATA = "../data/derived/pilote/";
  const COLORS = ["#d1495b", "#2e86ab", "#3f8f29", "#c98a00", "#7a4fa3"];
  const OBJ = "#8a5a00";
  const BONES = [
    ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"], ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
    ["left_shoulder", "right_shoulder"], ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"], ["left_hip", "right_hip"],
    ["left_hip", "left_knee"], ["left_knee", "left_ankle"], ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
    ["nose", "left_eye"], ["nose", "right_eye"], ["left_eye", "left_ear"], ["right_eye", "right_ear"],
  ];
  const POINT_FR = {
    nose: "nez", left_eye: "œil gauche", right_eye: "œil droit", left_ear: "oreille gauche", right_ear: "oreille droite",
    left_shoulder: "épaule gauche", right_shoulder: "épaule droite", left_elbow: "coude gauche", right_elbow: "coude droit",
    left_wrist: "poignet gauche", right_wrist: "poignet droit", left_hip: "hanche gauche", right_hip: "hanche droite",
    left_knee: "genou gauche", right_knee: "genou droit", left_ankle: "cheville gauche", right_ankle: "cheville droite",
  };
  const FACING = { "-1": "tournée vers la gauche de l'image", "1": "tournée vers la droite de l'image", "0": "de face, ou indécidable" };

  let index = [], record = null, current = null, editing = false, pending = {}, placing = null;
  const show = { image: true, contours: true, squelettes: true, boites: false };
  const $ = (s, r = document) => r.querySelector(s);
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, kids = []) => {
    const svg = ["svg", "line", "circle", "polygon", "rect", "text", "g", "path", "marker", "defs", "title", "polyline"].includes(tag);
    const n = svg ? document.createElementNS(NS, tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") n.textContent = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v);
    }
    for (const c of [].concat(kids)) if (c) n.append(c);
    return n;
  };
  const color = (id) => (id.startsWith("F") ? COLORS[(parseInt(id.slice(1), 10) - 1) % COLORS.length] : OBJ);
  const byId = () => Object.fromEntries([...record.figures, ...record.objects].map((x) => [x.id, x]));
  const centre = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];

  async function load() {
    index = await (await fetch(DATA + "index.json", { cache: "no-store" })).json();
    const list = $("#list");
    for (const it of index) {
      list.append(el("button", { "data-id": it.id, onclick: () => open(it.id) }, [
        el("img", { src: `../data/raw/emblematica/full/${it.id}.jpg`, loading: "lazy", alt: it.id }),
        el("span", { text: it.id }),
      ]));
    }
    const first = decodeURIComponent(location.hash.slice(1)) || (index[0] && index[0].id);
    if (first) open(first);
  }

  async function open(id) {
    current = id; editing = false; pending = {}; placing = null;
    location.hash = id;
    document.querySelectorAll(".plist button").forEach((b) => b.classList.toggle("on", b.dataset.id === id));
    record = await (await fetch(DATA + `records/${id}.json`, { cache: "no-store" })).json();
    render();
  }

  function keypoints(f) {
    return { ...f.keypoints, ...(pending[f.id] || {}) };
  }

  // ---- drawing ----
  function drawFigure(g, f, { image }) {
    const c = color(f.id);
    if (show.contours && f.polygon) g.append(el("polygon", { points: f.polygon.map((p) => p.join(",")).join(" "), fill: c, "fill-opacity": image ? 0.18 : 0.12, stroke: c, "stroke-width": 3 }));
    if (show.boites && image) g.append(el("rect", { x: f.box[0], y: f.box[1], width: f.box[2] - f.box[0], height: f.box[3] - f.box[1], fill: "none", stroke: c, "stroke-dasharray": "8 6", "stroke-width": 2 }));
    const k = keypoints(f);
    if (show.squelettes || !image) {
      for (const [a, b] of BONES) {
        if (k[a] && k[b]) g.append(el("line", { x1: k[a][0], y1: k[a][1], x2: k[b][0], y2: k[b][1], stroke: c, "stroke-width": a.startsWith("left") || b.startsWith("left") ? 7 : 4, "stroke-linecap": "round" }));
      }
      for (const [name, p] of Object.entries(k)) {
        if (!p) continue;
        const corrected = (f.corrected || []).includes(name) || (pending[f.id] && name in pending[f.id]);
        const dot = el("circle", { cx: p[0], cy: p[1], r: editing && image ? 9 : 6, fill: name.startsWith("left") ? c : "#fff", stroke: corrected ? "#000" : c, "stroke-width": corrected ? 4 : 3, class: editing && image ? "kp" : "" },
          [el("title", { text: `${f.name} — ${POINT_FR[name]}${name.startsWith("left") ? " (plein = gauche de la figure)" : ""}` })]);
        if (editing && image) attachDrag(dot, f, name);
        g.append(dot);
      }
    }
    const [x, y] = [f.box[0], f.box[1]];
    if (!image) g.append(el("text", { x: x + 4, y: y - 8, fill: c, "font-size": 28, "font-weight": 600, text: f.name }));
  }

  function drawObject(g, o, { image }) {
    if (show.contours || !image) {
      const pts = o.polygon || [[o.box[0], o.box[1]], [o.box[2], o.box[1]], [o.box[2], o.box[3]], [o.box[0], o.box[3]]];
      g.append(el("polygon", { points: pts.map((p) => p.join(",")).join(" "), fill: OBJ, "fill-opacity": image ? 0.15 : 0.25, stroke: OBJ, "stroke-width": 3, "stroke-dasharray": o.polygon ? "" : "6 4" }));
    }
    if (show.boites && image) g.append(el("rect", { x: o.box[0], y: o.box[1], width: o.box[2] - o.box[0], height: o.box[3] - o.box[1], fill: "none", stroke: OBJ, "stroke-dasharray": "8 6", "stroke-width": 2 }));
    if (!image) g.append(el("text", { x: o.box[0], y: o.box[3] + 26, fill: OBJ, "font-size": 24, text: o.name }));
  }

  function drawRelations(g) {
    const all = byId();
    for (const r of record.relations) {
      if (!["touche", "tient", "tend le bras vers", "regarde vers", "pose le pied sur", "met le pied sur"].includes(r.rel)) continue;
      const a = all[r.a], b = all[r.b];
      let from = centre(a.box);
      const k = keypoints(a);
      const PART = { "main gauche": "left_wrist", "main droite": "right_wrist", "pied gauche": "left_ankle", "pied droit": "right_ankle" };
      if (r.par) from = k[PART[r.par.split(" et ")[0]]] || from;
      if (r.rel === "regarde vers") from = k.nose || from;
      const to = centre(b.box);
      const c = color(r.a);
      const dash = r.rel === "regarde vers" ? "4 8" : r.rel === "tend le bras vers" ? "14 6" : "";
      g.append(el("line", { x1: from[0], y1: from[1], x2: to[0], y2: to[1], stroke: c, "stroke-width": 3, "stroke-dasharray": dash, "marker-end": `url(#arrow-${r.a})`, opacity: 0.85 }));
    }
  }

  function svgFor(image) {
    const [w, h] = record.size;
    const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, class: image ? "over" : "abstract", preserveAspectRatio: "xMidYMid meet" });
    const defs = el("defs");
    for (const f of record.figures) {
      defs.append(el("marker", { id: `arrow-${f.id}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
        [el("path", { d: "M0,0 L10,5 L0,10 z", fill: color(f.id) })]));
    }
    svg.append(defs);
    const g = el("g");
    for (const o of record.objects) drawObject(g, o, { image });
    for (const f of record.figures) drawFigure(g, f, { image });
    if (!image) drawRelations(g);
    svg.append(g);
    if (image && editing) {
      svg.addEventListener("click", (ev) => {
        if (!placing || ev.target.classList.contains("kp")) return;
        const p = toSvg(svg, ev);
        (pending[placing.fig] ||= {})[placing.point] = p;
        renderPanes();
      });
    }
    return svg;
  }

  function toSvg(svg, ev) {
    const pt = svg.createSVGPoint();
    pt.x = ev.clientX; pt.y = ev.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    return [Math.round(p.x), Math.round(p.y)];
  }

  function attachDrag(dot, f, name) {
    dot.addEventListener("pointerdown", (ev) => {
      ev.preventDefault(); ev.stopPropagation();
      if (ev.altKey) { (pending[f.id] ||= {})[name] = null; renderPanes(); return; }
      const svg = dot.ownerSVGElement;
      dot.classList.add("drag");
      dot.setPointerCapture(ev.pointerId);
      const move = (e) => { const p = toSvg(svg, e); dot.setAttribute("cx", p[0]); dot.setAttribute("cy", p[1]); };
      const up = (e) => {
        dot.removeEventListener("pointermove", move); dot.removeEventListener("pointerup", up);
        (pending[f.id] ||= {})[name] = toSvg(svg, e);
        renderPanes();
      };
      dot.addEventListener("pointermove", move); dot.addEventListener("pointerup", up);
    });
  }

  // ---- text ----
  function sentence(r) {
    const all = byId();
    const ev = [];
    if (r.par) ev.push(r.par);
    if (r.distance !== undefined) ev.push(r.distance === 0 ? "dedans" : `à ${r.distance} px`);
    if (r["écart"] !== undefined) ev.push(`écart ${r["écart"]}°`);
    if (r.rapport !== undefined) ev.push(`×${r.rapport}`);
    return el("li", {}, [
      el("span", { class: "chip", style: `background:${color(r.a)}`, text: all[r.a].name }), " ",
      el("strong", { text: r.rel }), " ",
      el("span", { class: "chip", style: `background:${color(r.b)}`, text: all[r.b].name }),
      ev.length ? el("span", { class: "ev", text: `  (${ev.join(", ")})` }) : null,
    ]);
  }

  function stick(f) {
    const s = f.skeleton || {};
    const svg = el("svg", { viewBox: "-2 -2.6 4 5.2" });
    for (const [a, b] of BONES) {
      if (s[a] && s[b]) svg.append(el("line", { x1: s[a][0], y1: s[a][1], x2: s[b][0], y2: s[b][1], stroke: color(f.id), "stroke-width": a.startsWith("left") || b.startsWith("left") ? 0.12 : 0.07, "stroke-linecap": "round" }));
    }
    return svg;
  }

  function figureCard(f) {
    const rows = Object.entries(f.pose || {}).map(([k, v]) => el("tr", {}, [el("td", { text: k }), el("td", { text: v === null ? "—" : `${v}°` })]));
    const missing = Object.entries(f.keypoints).filter(([, p]) => !p).map(([k]) => POINT_FR[k]);
    return el("div", { class: "pane fig" }, [
      stick(f),
      el("h3", { style: `color:${color(f.id)}`, text: `${f.id} · ${f.name}` }),
      f.attributes && f.attributes.length ? el("p", { text: `attributs : ${f.attributes.join(", ")}` }) : null,
      el("p", { class: "muted", text: `tête ${FACING[String(f.facing)]}` }),
      el("p", { class: "muted", text: "squelette normalisé (à droite) : hanches à l'origine, torse vertical, taille unité — trait épais = côté gauche de la figure" }),
      missing.length ? el("p", { class: "warn", text: `points absents : ${missing.join(", ")}` }) : null,
      el("table", { class: "pose" }, [el("tr", {}, [el("th", { text: "membre" }), el("th", { text: "angle au torse" })]), ...rows]),
    ]);
  }

  // ---- layout ----
  function renderPanes() {
    const stage = $("#stage");
    stage.replaceChildren(
      show.image ? el("img", { src: "../" + record.image, alt: record.id }) : el("div", { style: `aspect-ratio:${record.size[0]}/${record.size[1]};background:#fffdf8` }),
      svgFor(true),
    );
    $("#abstract").replaceChildren(svgFor(false));
    const changed = Object.values(pending).reduce((n, o) => n + Object.keys(o).length, 0);
    const status = $("#status");
    if (status) status.textContent = changed ? `${changed} point(s) modifié(s), non enregistré(s)` : "";
  }

  function render() {
    const said = record.objects.filter((o) => o.held_by_said).map((o) => `${o.name} ← ${byId()[o.held_by_said]?.name || o.held_by_said}`);
    const toggles = el("div", { class: "toggles" }, Object.keys(show).map((k) =>
      el("label", {}, [el("input", { type: "checkbox", ...(show[k] ? { checked: "" } : {}), onchange: (e) => { show[k] = e.target.checked; renderPanes(); } }), k])));
    const pointSel = el("select", { onchange: (e) => { const [fig, point] = e.target.value.split("|"); placing = fig ? { fig, point } : null; } },
      [el("option", { value: "", text: "placer un point : —" }), ...record.figures.flatMap((f) => Object.keys(POINT_FR).map((p) => el("option", { value: `${f.id}|${p}`, text: `${f.name} · ${POINT_FR[p]}` })))]);
    const editBar = el("div", { class: "edit" }, editing
      ? [el("button", { class: "primary", text: "enregistrer", onclick: save }), el("button", { text: "annuler", onclick: () => open(current) }), pointSel,
         el("span", { class: "muted", text: "glisser un point · alt-clic pour l'effacer · choisir puis cliquer pour placer" }), el("span", { id: "status", class: "warn" })]
      : [el("button", { text: "corriger les points", onclick: () => { editing = true; render(); } })]);

    $("#view").replaceChildren(
      el("div", { class: "ptitle" }, [el("h2", { text: record.id }), el("span", { class: "muted", text: `proposition : ${record.model || "?"}` })]),
      el("div", { class: "panes" }, [
        el("div", { class: "pane" }, [el("h3", { text: "L'image et la proposition" }), toggles, el("div", { id: "stage", class: "stage" }), editBar]),
        el("div", { class: "pane" }, [el("h3", { text: "La représentation, sans les pixels" }), el("div", { id: "abstract" }),
          el("p", { class: "muted", text: "flèche pleine : touche / tient · tirets longs : tend le bras vers · pointillés : regarde vers" })]),
      ]),
      el("div", { class: "panes", style: "margin-top:16px" }, [
        el("div", { class: "pane" }, [el("h3", { text: "Relations calculées par la géométrie" }),
          record.relations.length ? el("ul", { class: "rels" }, record.relations.map(sentence)) : el("p", { class: "muted", text: "aucune" })]),
        el("div", { class: "pane" }, [el("h3", { text: "Ce que dit Claude (en mots)" }), el("p", { class: "phrase", text: record.phrase || "—" }),
          said.length ? el("p", { class: "muted", text: `objets tenus, selon Claude : ${said.join(" · ")}` }) : null,
          el("p", { class: "muted", text: `objets : ${record.objects.map((o) => o.name).join(", ") || "aucun"}` })]),
      ]),
      el("div", { class: "figs" }, record.figures.map(figureCard)),
    );
    renderPanes();
  }

  async function save() {
    const corrections = JSON.parse(JSON.stringify(record.corrections || {}));
    corrections.items ||= {};
    for (const [fig, pts] of Object.entries(pending)) {
      const item = (corrections.items[fig] ||= {});
      item.keypoints = { ...(item.keypoints || {}), ...pts };
    }
    const res = await fetch("/api/pilote/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id: current, corrections }) });
    const out = await res.json();
    if (out.error) { $("#status").textContent = out.error; return; }
    record = out; pending = {}; editing = false; render();
  }

  window.addEventListener("hashchange", () => {
    const id = decodeURIComponent(location.hash.slice(1));
    if (id && id !== current) open(id);
  });
  load();
})();
