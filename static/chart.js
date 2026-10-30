/* Petits graphiques SVG, ecrits a la main.
   Pas de bibliotheque : la page doit rester lisible hors ligne et sans CDN, et ces
   quatre formes suffisent. Tout est en coordonnees viewBox pour rester net a
   n'importe quelle taille. */

const SVG = 'http://www.w3.org/2000/svg';
const el = (n, a = {}, t = null) => {
  const e = document.createElementNS(SVG, n);
  for (const k in a) if (a[k] !== null && a[k] !== undefined) e.setAttribute(k, a[k]);
  if (t !== null) e.textContent = t;
  return e;
};
const pcTxt = v => (v * 100).toFixed(1).replace('.', ',') + '%';
const nbTxt = n => n.toLocaleString('fr-FR');

function cadre(hote, w, h) {
  hote.innerHTML = '';
  const svg = el('svg', {viewBox: `0 0 ${w} ${h}`, class: 'graph',
                         preserveAspectRatio: 'xMidYMid meet'});
  hote.appendChild(svg);
  return svg;
}

/* Courbe : series = [{nom, couleur, points:[{x,y}]}], y en fraction [0,1]. */
function courbe(hote, series, opts = {}) {
  const w = 760, h = opts.hauteur || 260;
  const mg = {g: 46, d: 12, ht: 16, b: 34};
  const svg = cadre(hote, w, h);
  const tous = series.flatMap(s => s.points);
  if (!tous.length) return;
  const xmin = Math.min(...tous.map(p => p.x)), xmax = Math.max(...tous.map(p => p.x));
  const ymax = Math.max(opts.ymax || 0, ...tous.map(p => p.y)) * 1.08 || 1;
  const X = x => mg.g + (x - xmin) / Math.max(1e-9, xmax - xmin) * (w - mg.g - mg.d);
  const Y = y => h - mg.b - (y / ymax) * (h - mg.ht - mg.b);

  for (let i = 0; i <= 4; i++) {
    const v = ymax * i / 4;
    svg.appendChild(el('line', {x1: mg.g, x2: w - mg.d, y1: Y(v), y2: Y(v), class: 'grille-l'}));
    svg.appendChild(el('text', {x: mg.g - 7, y: Y(v) + 4, class: 'axe fin'}, pcTxt(v)));
  }
  (opts.reperes || []).forEach(r => {
    svg.appendChild(el('line', {x1: mg.g, x2: w - mg.d, y1: Y(r.y), y2: Y(r.y),
                                class: 'repere', stroke: r.couleur || null}));
    svg.appendChild(el('text', {x: w - mg.d - 4, y: Y(r.y) - 5, class: 'axe fin',
                                'text-anchor': 'end', fill: r.couleur || null}, r.nom));
  });
  (opts.marques || []).forEach(mq => {
    svg.appendChild(el('line', {x1: X(mq.x), x2: X(mq.x), y1: mg.ht, y2: h - mg.b,
                                class: 'marque'}));
    svg.appendChild(el('text', {x: X(mq.x) + 4, y: mg.ht + 11, class: 'axe fin'}, mq.nom));
  });
  series.forEach(s => {
    const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(' ');
    svg.appendChild(el('path', {d, fill: 'none', stroke: s.couleur, 'stroke-width': 2.2,
                                'stroke-linejoin': 'round', 'stroke-linecap': 'round'}));
    s.points.forEach(p => svg.appendChild(
      el('circle', {cx: X(p.x), cy: Y(p.y), r: 2.4, fill: s.couleur})));
  });
  [xmin, Math.round((xmin + xmax) / 2), xmax].forEach(v =>
    svg.appendChild(el('text', {x: X(v), y: h - 12, class: 'axe fin',
                                'text-anchor': 'middle'}, nbTxt(v))));
  svg.appendChild(el('text', {x: mg.g, y: h - 1, class: 'axe fin'}, opts.x || ''));
  legende(hote, series);
}

/* Barres horizontales : items = [{nom, v, couleur, note}] , v en fraction. */
function barres(hote, items, opts = {}) {
  const w = 760, lh = opts.hauteur_ligne || 30;
  const h = items.length * lh + 14;
  const svg = cadre(hote, w, h);
  const lab = opts.largeur_label || 250;
  const vmax = Math.max(opts.vmax || 0, ...items.map(i => i.v)) || 1;
  items.forEach((it, i) => {
    const y = 8 + i * lh;
    svg.appendChild(el('text', {x: 0, y: y + 14, class: 'axe'}, it.nom));
    svg.appendChild(el('rect', {x: lab, y: y + 3, width: w - lab - 58, height: 14,
                                rx: 4, class: 'fond-barre'}));
    svg.appendChild(el('rect', {x: lab, y: y + 3, width: Math.max(1, (it.v / vmax) * (w - lab - 58)),
                                height: 14, rx: 4, fill: it.couleur || 'var(--accent)'}));
    svg.appendChild(el('text', {x: w, y: y + 15, class: 'axe val', 'text-anchor': 'end'},
                       it.texte || pcTxt(it.v)));
  });
}

/* Histogramme depuis {bords, effectifs}. */
function histo(hote, hist, opts = {}) {
  const w = 760, h = opts.hauteur || 170, mg = {g: 40, d: 10, ht: 12, b: 28};
  const svg = cadre(hote, w, h);
  const e = hist.effectifs || [];
  if (!e.length) return;
  const emax = Math.max(...e) || 1;
  const bw = (w - mg.g - mg.d) / e.length;
  e.forEach((v, i) => {
    const ht = (v / emax) * (h - mg.ht - mg.b);
    svg.appendChild(el('rect', {x: mg.g + i * bw + 0.7, y: h - mg.b - ht,
                                width: Math.max(1, bw - 1.4), height: ht, rx: 1.5,
                                fill: opts.couleur || 'var(--accent)', opacity: 0.85}));
  });
  svg.appendChild(el('text', {x: mg.g - 6, y: mg.ht + 8, class: 'axe fin',
                              'text-anchor': 'end'}, nbTxt(emax)));
  const b = hist.bords;
  [[mg.g, b[0]], [w - mg.d, b[b.length - 1]]].forEach(([x, v], i) =>
    svg.appendChild(el('text', {x, y: h - 9, class: 'axe fin',
                                'text-anchor': i ? 'end' : 'start'}, nbTxt(Math.round(v)))));
  if (opts.legende) svg.appendChild(
    el('text', {x: w / 2, y: h - 9, class: 'axe fin', 'text-anchor': 'middle'}, opts.legende));
}

/* Profil trie (une valeur par neurone, decroissant). */
function profil(hote, valeurs, opts = {}) {
  const w = 760, h = opts.hauteur || 150, mg = {g: 44, d: 10, ht: 12, b: 24};
  const svg = cadre(hote, w, h);
  if (!valeurs.length) return;
  const vmax = Math.max(...valeurs) || 1;
  const bw = (w - mg.g - mg.d) / valeurs.length;
  valeurs.forEach((v, i) => {
    const ht = (v / vmax) * (h - mg.ht - mg.b);
    svg.appendChild(el('rect', {x: mg.g + i * bw + 0.5, y: h - mg.b - ht,
                                width: Math.max(1, bw - 1), height: Math.max(0.5, ht),
                                fill: v >= (opts.seuil || 0) ? (opts.couleur || 'var(--accent)')
                                                             : 'var(--rouge)'}));
  });
  if (opts.seuil) {
    const y = h - mg.b - (opts.seuil / vmax) * (h - mg.ht - mg.b);
    svg.appendChild(el('line', {x1: mg.g, x2: w - mg.d, y1: y, y2: y, class: 'repere'}));
    svg.appendChild(el('text', {x: w - mg.d, y: y - 4, class: 'axe fin',
                                'text-anchor': 'end'}, opts.nom_seuil || ''));
  }
  svg.appendChild(el('text', {x: mg.g - 6, y: mg.ht + 8, class: 'axe fin',
                              'text-anchor': 'end'}, nbTxt(vmax)));
  svg.appendChild(el('text', {x: mg.g, y: h - 7, class: 'axe fin'}, opts.legende || ''));
}

function legende(hote, series) {
  const d = document.createElement('div');
  d.className = 'legende';
  d.innerHTML = series.map(s =>
    `<span><i style="background:${s.couleur}"></i>${s.nom}</span>`).join('');
  hote.appendChild(d);
}

window.G = {courbe, barres, histo, profil, pcTxt, nbTxt};
