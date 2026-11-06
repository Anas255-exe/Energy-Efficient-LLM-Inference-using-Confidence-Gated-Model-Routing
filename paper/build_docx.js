// paper/draft.md -> paper/greengate_paper.docx, research-paper layout (Letter, Times, numbered figures/tables/equations).
// Handles the markdown subset draft.md uses: headings, paragraphs, **bold**, *italic*, `code`, X_sub, X^sup, lists,
// pipe tables, <!-- FIG:key --> figure markers, and standalone equation lines.
// usage (from greengate/): NODE_PATH=<dir with node_modules> node paper/build_docx.js
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, Header, Footer, AlignmentType,
  WidthType, BorderStyle, LevelFormat, TabStopType, PageNumber, VerticalAlign,
} = require("docx");

const FONT = "Times New Roman", BODY = 22, SMALL = 17, CAP = 19; // half-points
const TEXT_W = 9360; // 6.5in in DXA
const md = fs.readFileSync("paper/draft.md", "utf8").split("\n");

const FIGS = {
  fig1: ["paper/figures/fig1_architecture.png", 6.3, "Cascade pipeline and energy accounting. The draft model answers every query; the gate escalates the top fraction r of queries by risk score to the target model. Escalated queries pay for both models, so cascade energy relative to always-large is ρ + r."],
  fig2: ["paper/figures/fig2_energy_scaling.png", 4.4, "Measured energy per query (NVML gross, batch 1, 100 real queries per model) against parameter count on 2x T4. Labels give the local exponent β between neighbouring sizes: small fp16 models are dominated by fixed overheads (β 0.15-0.25), while from 3B upward β is 0.86-1.13."],
  fig3: ["paper/figures/fig3_four_cell.png", 6.3, "The four-cell model. Left: share of questions in each joint-correctness cell for a small pair (0.5B→7B, T4 fp16 test split) and a large pair (14B→32B, T4 4-bit). Right: binormal gate scores per cell at the mean measured cell AUROCs over the nine pairs. The rescuable and both-wrong distributions nearly coincide, so the highest-risk queries include many that neither model answers."],
  fig4: ["paper/figures/fig4_hardness.png", 5.2, "Gates track question hardness. Token-entropy AUROC of the rescuable cell (filled) and the both-wrong cell (ring), each against the both-right cell. Both score about equally high on every pair, so the gate cannot tell which errors the target can fix."],
  fig5: ["paper/figures/fig5_curves.png", 5.2, "Accuracy retained against cascade energy relative to always-large as the escalation rate rises (entropy gate, measured replay at every escalation count). Dots mark the first point at 95% retention; points left of the dotted line save energy."],
  phase: ["results_kaggle/phase_diagram.png", 5.4, "Phase diagram. Saving at 95% retention predicted by the four-cell model over the energy ratio ρ and coverage; the solid and dashed lines are break-even for the entropy and oracle gates. Points are the nine measured pairs (circles: small pairs, diamonds: large 4-bit pairs), labelled measured / predicted saving."],
  scale: ["results_olb/scale_map.png", 4.6, "Projected saving at 95% retention for all Qwen2.5 pairs on MMLU-Pro (Open LLM Leaderboard per-question logs, max-probability gate, ρ = (N_S/N_L)^0.86, the conservative end of the measured exponent range)."],
  dash: ["paper/figures/fig8_dashboard.png", 6.3, "The interactive dashboard (offline HTML). Choosing a pair, an accuracy-retention level and an electricity grid updates each tile, which compares the large model alone with the cascade (green: better, red: worse) for energy, hallucination rate, latency and CO2, plus the share escalated. Left chart: saving against retention for the chosen pair; right chart: saving for every measured pair at that retention. Shown: 14B→32B on T4 4-bit at 95% retention, India grid."],
};

// ---------- inline formatting
function scripts(text, base) { // X_sub and X^sup outside code
  const out = [];
  const re = /([A-Za-zΦ̄]+)_([A-Za-z0-9]+)|\^(β|[0-9]+(?:\.[0-9]+)?)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ ...base, text: text.slice(last, m.index) }));
    if (m[1]) {
      out.push(new TextRun({ ...base, text: m[1] }));
      out.push(new TextRun({ ...base, text: m[2], subScript: true }));
    } else out.push(new TextRun({ ...base, text: m[3], superScript: true }));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(new TextRun({ ...base, text: text.slice(last) }));
  return out;
}
function runs(text, opt = {}) {
  const base = { font: FONT, size: opt.size || BODY, bold: opt.bold, italics: opt.italics, color: opt.color };
  const out = [];
  const re = /`([^`]+)`|\*\*([^*]+)\*\*|\*([^*]+)\*/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(...scripts(text.slice(last, m.index), base));
    if (m[1] !== undefined) out.push(new TextRun({ ...base, font: "Courier New", size: (opt.size || BODY) - 3, text: m[1] }));
    else if (m[2] !== undefined) out.push(...scripts(m[2], { ...base, bold: true }));
    else out.push(...scripts(m[3], { ...base, italics: !opt.italics }));
    last = re.lastIndex;
  }
  if (last < text.length) out.push(...scripts(text.slice(last), base));
  return out;
}
const para = (text, o = {}) => new Paragraph({
  children: runs(text, o), alignment: o.align ?? AlignmentType.JUSTIFIED,
  spacing: { before: o.before ?? 0, after: o.after ?? 120, line: o.line ?? 252 },
  indent: o.indent, keepNext: o.keepNext, border: o.border, numbering: o.numbering,
});

// ---------- figures
function pngSize(f) { const b = fs.readFileSync(f); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }
let figNo = 0;
function figure(key) {
  const [file, inches, caption] = FIGS[key];
  const [w, h] = pngSize(file);
  const pw = Math.round(inches * 96);
  figNo += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: pw, height: Math.round(pw * h / w) } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200, line: 240 }, indent: { left: 360, right: 360 },
      children: [new TextRun({ font: FONT, size: CAP, bold: true, text: `Figure ${figNo}: ` }), ...runs(caption, { size: CAP })] }),
  ];
}

// ---------- tables (booktabs style)
function table(rows) {
  const cells = rows.map(r => r.replace(/^\||\|$/g, "").split("|").map(c => c.trim()));
  const head = cells[0], body = cells.slice(2);
  const n = head.length;
  // width: every column gets its longest unbreakable token, the rest is shared by how much longer its text runs
  const plain = c => (c || "").replace(/\*|`/g, "");
  const colMax = f => [...Array(n).keys()].map(i => Math.max(...[head, ...body].map(r => f(plain(r[i])))));
  const minW = colMax(c => Math.max(...c.split(/\s+/).map(t => t.length))).map(x => x * 110 + 160);
  const desW = colMax(c => Math.min(c.length, 40)).map(x => x * 85 + 160);
  const sumMin = minW.reduce((a, b) => a + b, 0);
  let widths;
  if (sumMin >= TEXT_W) widths = minW.map(x => Math.floor(x * TEXT_W / sumMin));
  else {
    const extra = desW.map((d, i) => Math.max(0, d - minW[i])), te = extra.reduce((a, b) => a + b, 0) || 1;
    widths = minW.map((m, i) => Math.floor(m + (TEXT_W - sumMin) * extra[i] / te));
  }
  widths[n - 1] += TEXT_W - widths.reduce((a, b) => a + b, 0);
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const rule = sz => ({ style: BorderStyle.SINGLE, size: sz, color: "000000" });
  const mk = (r, ri, isHead) => new TableRow({
    tableHeader: isHead, cantSplit: true,
    children: r.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
      margins: { top: 40, bottom: 40, left: 70, right: 70 },
      borders: { left: none, right: none, top: isHead ? rule(12) : none, bottom: isHead ? rule(6) : (ri === body.length - 1 ? rule(12) : none) },
      children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, spacing: { after: 0, line: 220 },
        children: runs(c, { size: SMALL, bold: isHead || undefined }) })],
    })),
  });
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths,
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
    rows: [mk(head, -1, true), ...body.map((r, ri) => mk(r, ri, false))],
  });
}

// ---------- walk the markdown
const out = [];
const marks = []; // out indices where a figure or table caption starts (for PROBE renders)
let i = 0, inAbstract = false, inRefs = false, eqNo = 0, listInst = 0, prevList = false;
const H1 = t => new Paragraph({ children: runs(t, { size: 26, bold: true }), keepNext: true, spacing: { before: 280, after: 120 } });
const H2 = t => new Paragraph({ children: runs(t, { size: BODY, bold: true }), keepNext: true, spacing: { before: 200, after: 80 } });
while (i < md.length) {
  const line = md[i];
  const isList = /^\s*(- |\d+\. )/.test(line);
  if (!isList) prevList = false;
  if (line.startsWith("# ")) {
    out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 240, line: 300 },
      border: { top: { style: BorderStyle.SINGLE, size: 24, color: "000000", space: 10 }, bottom: { style: BorderStyle.SINGLE, size: 8, color: "000000", space: 10 } },
      children: runs(line.slice(2), { size: 34, bold: true }) }));
  } else if (/^\*\*Mohammad/.test(line)) {
    out.push(para(line, { align: AlignmentType.CENTER, after: 40, size: 24 }));
    i++;
    while (md[i] !== undefined && !md[i].startsWith("---")) {
      if (md[i].trim()) out.push(para(md[i], { align: AlignmentType.CENTER, after: 40 }));
      i++;
    }
    continue;
  } else if (line.startsWith("---") || !line.trim()) {
    // separators and blank lines carry no content
  } else if (line.startsWith("## ")) {
    const t = line.slice(3).trim();
    inAbstract = t === "Abstract"; inRefs = t === "References";
    if (inAbstract) out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 280, after: 100 }, children: runs("Abstract", { size: 24, bold: true }) }));
    else out.push(H1(t));
  } else if (line.startsWith("### ")) {
    out.push(H2(line.slice(4)));
  } else if (line.startsWith("<!-- FIG:")) {
    marks.push(out.length); out.push(...figure(line.match(/FIG:(\w+)/)[1]));
  } else if (/^\*Table [A-Z0-9]+\./.test(line)) {
    const m = line.replace(/^\*|\*$/g, "").match(/^(Table [A-Z0-9]+)\.\s*(.*)$/);
    marks.push(out.length);
    out.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 80 },
      children: [new TextRun({ font: FONT, size: CAP, bold: true, text: `${m[1]}: ` }), ...runs(m[2], { size: CAP })] }));
  } else if (line.startsWith("|")) {
    const rows = [];
    while (md[i] && md[i].startsWith("|")) rows.push(md[i++]);
    out.push(table(rows), new Paragraph({ spacing: { after: 120 }, children: [] }));
    continue;
  } else if (/^(E_cascade|acc\(r\) =)/.test(line)) {
    eqNo += 1;
    out.push(new Paragraph({ tabStops: [{ type: TabStopType.CENTER, position: TEXT_W / 2 }, { type: TabStopType.RIGHT, position: TEXT_W }],
      spacing: { before: 80, after: 160 },
      children: [new TextRun({ font: FONT, size: BODY, text: "\t" }), ...runs(line.replace(/[,.]$/, ""), { italics: true }),
        new TextRun({ font: FONT, size: BODY, text: `\t(${eqNo})` })] }));
  } else if (isList) {
    const m = line.match(/^(\s*)(- |\d+\. )(.*)$/);
    const level = m[1].length >= 2 ? 1 : 0, numbered = /\d/.test(m[2]);
    if (numbered && !prevList) listInst += 1;
    prevList = true;
    out.push(para(m[3], { after: 60, size: inAbstract ? 20 : BODY,
      numbering: numbered ? { reference: "numbers", level: 0, instance: listInst } : { reference: "bullets", level } }));
  } else if (inRefs) {
    out.push(para(line, { size: 18, after: 80, line: 230, indent: { left: 440, hanging: 440 }, align: AlignmentType.LEFT }));
  } else if (inAbstract) {
    out.push(para(line, { size: 20, indent: { left: 720, right: 720 }, after: 100 }));
  } else {
    out.push(para(line));
  }
  i++;
}

const mkDoc = children => new Document({
  creator: "greengate", title: "Energy-Efficient LLM Inference using Confidence-Gated Model Routing",
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 240 } } } }] },
    { reference: "numbers", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440, header: 720, footer: 720 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ font: FONT, size: 16, italics: true, color: "555555", text: "Energy-Efficient LLM Inference using Confidence-Gated Model Routing  ·  Preprint" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ font: FONT, size: 18, children: [PageNumber.CURRENT] })] })] }) },
    children,
  }],
});
Packer.toBuffer(mkDoc(out)).then(b => { fs.writeFileSync("paper/greengate_paper.docx", b); console.log("wrote paper/greengate_paper.docx", figNo, "figures", eqNo, "equations"); });
// PROBE=dir: one extra docx per figure/table starting at it, so a first-page renderer (Quick Look) can show each
if (process.env.PROBE) marks.forEach((k, n) => Packer.toBuffer(mkDoc(out.slice(k))).then(b => fs.writeFileSync(`${process.env.PROBE}/probe_${String(n).padStart(2, "0")}.docx`, b)));
