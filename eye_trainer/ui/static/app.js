"use strict";
// Local labelling page. Talks only to the local server (same origin).
const TOKEN = document.querySelector('meta[name="token"]').content;
const $ = (id) => document.getElementById(id);
let current = null;        // line being labelled
let direction = "rtl";
let history = [];          // ids saved in this session, for "previous"

async function api(path, body) {
  const opt = { headers: { "X-Token": TOKEN } };
  if (body !== undefined) {
    opt.method = "POST";
    opt.headers["Content-Type"] = "application/json";
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => null);
  if (!r.ok) throw new Error((data && data.error) || r.statusText);
  return data;
}

// ------------------------------------------------------------------ tabs
function showTab(name) {
  document.querySelectorAll(".panel[id^=tab-]").forEach((p) => p.classList.toggle("hidden", p.id !== "tab-" + name));
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  try { localStorage.setItem("tab", name); } catch (e) { /* storage may be blocked */ }
  if (name === "label" && !current) loadNext();
}
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ------------------------------------------------------------------ labelling
function setDirection(d) {
  direction = d;
  $("dir-label").textContent = d === "rtl" ? "من اليمين ←" : "→ من اليسار";
  $("text").dir = d;
  renderPreview();
}

function renderPreview() {
  const p = $("preview");
  p.textContent = $("text").value;
  p.className = "preview " + direction;
}

function showLine(line) {
  current = line;
  $("label-empty").classList.toggle("hidden", !!line);
  $("label-box").classList.toggle("hidden", !line);
  if (!line) return;
  $("crop").src = "/crop/" + line.id + "?t=" + encodeURIComponent(TOKEN);
  $("text").value = line.label || "";
  setDirection(line.direction);
  const alt = line.alt_text && line.alt_text !== line.label ? line.alt_text : "";
  $("btn-alt").classList.toggle("hidden", !alt);
  $("btn-alt").textContent = alt ? "قراءة بديلة: " + alt : "";
  const score = line.score == null ? "—" : Math.round(line.score * 100) + "%";
  const states = { todo: "بانتظار التصحيح", done: "محفوظ", skipped: "متخطّى", flagged: "غير مقروء" };
  $("meta").textContent = `ثقة النموذج: ${score} · صفحة ${line.page} · الحالة: ${states[line.status] || line.status}`;
  $("text").focus();
  $("text").setSelectionRange($("text").value.length, $("text").value.length);
}

async function loadNext() {
  const lines = await api("/api/next?n=1");
  showLine(lines[0] || null);
  refreshStatus();
}

async function save(status) {
  if (!current) return;
  const label = $("text").value.trim();
  if (status === "done" && !label) { alert("السطر فارغ. استخدم «تخطٍّ» أو «غير مقروء»."); return; }
  try {
    await api("/api/label", { id: current.id, status, label, direction,
                              persian_to_arabic: $("persian").checked });
  } catch (e) { alert("لم يُحفظ: " + e.message); return; }
  history.push(current.id);
  loadNext();
}

async function previous() {
  const id = history.pop();
  if (id === undefined) return;
  showLine(await api("/api/line/" + id));
}

$("text").addEventListener("input", renderPreview);
$("btn-save").addEventListener("click", () => save("done"));
$("btn-skip").addEventListener("click", () => save("skipped"));
$("btn-flag").addEventListener("click", () => save("flagged"));
$("btn-prev").addEventListener("click", previous);
$("btn-reset").addEventListener("click", () => { if (current) { $("text").value = current.suggestion || ""; renderPreview(); } });
$("btn-dir").addEventListener("click", () => setDirection(direction === "rtl" ? "ltr" : "rtl"));
$("btn-alt").addEventListener("click", () => { if (current) { $("text").value = current.alt_text; renderPreview(); $("text").focus(); } });
document.querySelectorAll("[data-ins]").forEach((b) => b.addEventListener("click", () => {
  const t = $("text"), s = t.selectionStart, e = t.selectionEnd;
  t.value = t.value.slice(0, s) + b.dataset.ins + t.value.slice(e);
  t.setSelectionRange(s + b.dataset.ins.length, s + b.dataset.ins.length);
  t.focus(); renderPreview();
}));

document.addEventListener("keydown", (ev) => {
  if ($("tab-label").classList.contains("hidden") || !current) return;
  if (ev.key === "Enter" && !ev.shiftKey && document.activeElement === $("text")) { ev.preventDefault(); save("done"); }
  else if (ev.key === "F2") { ev.preventDefault(); save("skipped"); }
  else if (ev.key === "F3") { ev.preventDefault(); setDirection(direction === "rtl" ? "ltr" : "rtl"); }
  else if (ev.key === "F4") { ev.preventDefault(); save("flagged"); }
  else if (ev.key === "PageUp") { ev.preventDefault(); previous(); }
  else if (ev.key === "Escape") { ev.preventDefault(); $("btn-reset").click(); }
});

// ------------------------------------------------------------------ jobs
async function startJob(body) {
  try {
    const r = await api("/api/job", body);
    if (!r.started) alert("هناك عملية أخرى قيد التشغيل. انتظر حتى تنتهي.");
  } catch (e) { alert(e.message); }
  refreshStatus();
}
$("btn-import").addEventListener("click", () => startJob({ job: "import", folder: $("folder").value.trim() }));
$("btn-synth").addEventListener("click", () => startJob({ job: "synth", lines: +$("synth-lines").value }));
$("btn-train").addEventListener("click", () => startJob({ job: "train", run: $("run").value, steps: +$("steps").value,
  batch_size: +$("bs").value, use_synthetic: $("use-syn").checked }));
$("btn-stop").addEventListener("click", () => api("/api/job/stop", {}).then(refreshStatus));
$("btn-eval").addEventListener("click", () => startJob({ job: "evaluate", run: $("eval-run").value }));
$("btn-export").addEventListener("click", () => startJob({ job: "export", run: $("eval-run").value }));
$("eval-run").addEventListener("change", loadReport);

function fmtTime(s) {
  if (s == null) return "—";
  const m = Math.round(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} ساعة و${m % 60} دقيقة` : `${m} دقيقة`;
}
const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");

async function loadReport() {
  const run = $("eval-run").value;
  if (!run) { $("report").textContent = ""; return; }
  const rep = await api("/api/report/" + encodeURIComponent(run));
  $("report").textContent = Object.values(rep).join("\n\n") || "لا يوجد تقرير بعد. اضغط «قيِّم».";
}

let lastJobStatus = null;
async function refreshStatus() {
  let s;
  try { s = await api("/api/status"); } catch (e) { return; }
  const p = s.progress;
  $("label-progress").textContent =
    `الوثائق: ${p.documents} · الأسطر: ${p.lines} · صُحّحت: ${p.done} (منها مُعدّلة: ${p.corrected}) · متخطّاة: ${p.skipped} · غير مقروءة: ${p.flagged} · متبقية: ${p.todo}`;
  $("import-stats").textContent = `الوثائق المستوردة: ${p.documents} · الأسطر: ${p.lines}`;
  const sp = p.splits;
  $("split-info").textContent =
    `تقسيم الوثائق (حسب الوثيقة وليس السطر): تدريب ${sp.train.documents} وثيقة (${sp.train.labelled_lines} سطر مصحّح) · ` +
    `تحقق ${sp.val.documents} (${sp.val.labelled_lines}) · اختبار ${sp.test.documents} (${sp.test.labelled_lines})`;
  const t = s.training[$("run").value];
  if (t) {
    $("train-bar").style.width = t.percent + "%";
    const states = { running: "يعمل", paused: "متوقف مؤقتاً", finished: "انتهى" };
    $("train-info").textContent =
      `الحالة: ${states[t.status] || t.status} · الخطوة ${t.step} من ${t.max_steps} (${t.percent}%) · الوقت المتبقي تقريباً: ${fmtTime(t.eta_seconds)}\n` +
      `الجهاز: ${t.device === "gpu" ? "بطاقة رسومات" : "المعالج"} · نسبة خطأ الأحرف: الأصلي ${pct(t.base_val_cer)} ← أفضل نتيجة ${pct(t.best_val_cer)}`;
  }
  const sel = $("eval-run"), cur = sel.value;
  if (sel.options.length !== s.runs.length) {
    sel.innerHTML = "";
    s.runs.forEach((r) => sel.add(new Option(r, r)));
    if (s.runs.includes(cur)) sel.value = cur;
    loadReport();
  }
  $("exports").textContent = s.exports.length ? "النماذج المصدَّرة:\n" + s.exports.join("\n") : "";
  const j = s.job;
  const jstates = { idle: "", running: "⏳ يعمل", finished: "✓ انتهى", failed: "✗ فشل", stopping: "⏸ يتوقف..." };
  $("job-status").textContent = j.name ? `${j.name} — ${jstates[j.status] || j.status}` : "";
  $("log").textContent = j.log.join("\n");
  $("log").scrollTop = $("log").scrollHeight;
  if (lastJobStatus === "running" && j.status !== "running") {
    loadReport();
    if (!current) loadNext();
  }
  lastJobStatus = j.status;
}

let startTab = "label";
try { startTab = localStorage.getItem("tab") || "label"; } catch (e) { /* ignore */ }
showTab(startTab);
refreshStatus();
setInterval(refreshStatus, 3000);
