// Диагностика бетінің клиенттік логикасы
const $ = (id) => document.getElementById(id);
const file = $("file"), drop = $("drop"), btn = $("btn");
let chosen = null;

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  chosen = f;
  const url = URL.createObjectURL(f);
  $("preview").src = url; $("preview").hidden = false; $("dropText").hidden = true;
  btn.disabled = false;
}

file.addEventListener("change", () => setFile(file.files[0]));
["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => setFile(ev.dataTransfer.files[0]));

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const list = (arr) => arr && arr.length ? `<ul>${arr.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";

// Тарихтан қайта талдау: /?img=<файл атауы>&model=<модель>
(async () => {
  const q = new URLSearchParams(location.search);
  const name = q.get("img");
  if (!name || !/^[\w.-]+$/.test(name)) return;
  if (q.get("model")) $("model").value = q.get("model");
  const blob = await (await fetch("/uploads/" + name)).blob();
  setFile(new File([blob], name, { type: blob.type || "image/jpeg" }));
  $("form").requestSubmit();
})();

$("form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!chosen) return;
  const fd = new FormData();
  fd.append("image", chosen); fd.append("model", $("model").value);
  $("empty").hidden = true; $("out").hidden = true; $("loading").hidden = false; btn.disabled = true;
  try {
    const r = await fetch("/api/predict", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Қате");
    render(d);
  } catch (e) {
    $("empty").hidden = false; $("empty").innerHTML = `<p class="err">${esc(e.message)}</p>`;
  } finally {
    $("loading").hidden = true; btn.disabled = false;
  }
});

function render(d) {
  const best = d.top[0], info = d.info, pct = Math.round(best.prob * 100);
  $("rPlant").textContent = info.plant;
  $("rName").textContent = info.name;
  $("rKind").textContent = info.kind;
  $("rKind").className = "tag " + (info.healthy ? "ok" : "bad");
  $("rSev").textContent = "Қауіптілік: " + info.severity_text;
  $("rSev").className = "tag sev" + info.severity;
  $("rConf").textContent = pct + "%";
  $("gauge").style.setProperty("--p", pct);
  $("warn").hidden = !d.low_confidence;
  $("rPath").textContent = info.pathogen;
  $("rImg").src = d.image_url;
  $("rCam").src = d.heatmap || d.image_url;
  $("bars").innerHTML = d.top.map((t) => `
    <div class="bar"><div class="bar-l"><span>${esc(t.plant)} — ${esc(t.name)}</span><b>${(t.prob * 100).toFixed(1)}%</b></div>
    <div class="bar-t"><div class="bar-f ${t.healthy ? "ok" : ""}" style="width:${(t.prob * 100).toFixed(1)}%"></div></div></div>`).join("");
  $("details").innerHTML = `
    <h3>Белгілері</h3><p>${esc(info.symptoms)}</p>
    ${info.treatment && info.treatment.length ? `<h3>Емдеу шаралары</h3>${list(info.treatment)}` : ""}
    <h3>Алдын алу</h3>${list(info.prevention)}`;
  $("rMeta").textContent = `${d.model_title} · өңдеу уақыты ${d.time_ms} мс · класс: ${best.cls}`;
  $("out").hidden = false;
}
