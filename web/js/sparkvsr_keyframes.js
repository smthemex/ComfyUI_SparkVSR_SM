// SparkVSR Native - interactive keyframe picker for LoadVideoKeyframes.
// Renders source-video thumbnails; clicking toggles the frame index in the
// node's `selected_indices` widget (which is also exported downstream).

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "SparkVSR_Native_LoadVideoKeyframes";

function parseList(text) {
  return String(text || "")
    .split(/[,，\s]+/)
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n) && n >= 0);
}

function buildPicker(node) {
  const idxWidget = node.widgets?.find((w) => w.name === "selected_indices");
  const fileWidget = node.widgets?.find((w) => w.name === "video_file");
  if (!idxWidget || !fileWidget) return;

  const root = document.createElement("div");
  root.className = "sparkvsr-picker";
  Object.assign(root.style, {
    padding: "6px",
    overflowY: "auto",
    maxHeight: "260px",
    minHeight: "70px",
    boxSizing: "border-box",
    background: "var(--comfy-menu-bg, rgba(30,30,30,0.35))",
    borderRadius: "6px",
  });

  const bar = document.createElement("div");
  Object.assign(bar.style, {
    display: "flex", gap: "4px", flexWrap: "wrap",
    alignItems: "center", marginBottom: "6px",
  });
  const status = document.createElement("span");
  Object.assign(status.style, {
    fontSize: "11px", opacity: "0.8", marginRight: "auto",
  });
  status.textContent = "Keyframes: -";
  bar.appendChild(status);

  const mkBtn = (label, fn) => {
    const b = document.createElement("button");
    b.textContent = label;
    Object.assign(b.style, {
      fontSize: "11px", padding: "2px 6px", cursor: "pointer",
      border: "1px solid var(--border-color, #666)",
      borderRadius: "4px", background: "transparent",
      color: "var(--input-text, inherit)",
    });
    b.onclick = fn;
    bar.appendChild(b);
    return b;
  };

  const grid = document.createElement("div");
  Object.assign(grid.style, {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(88px, 1fr))",
    gap: "4px",
  });
  root.appendChild(bar);
  root.appendChild(grid);

  let thumbs = [];
  const selected = new Set(parseList(idxWidget.value));

  const sync = () => {
    const list = [...selected].sort((a, b) => a - b);
    idxWidget.value = list.join(",");
    status.textContent = `Keyframes: ${list.join(",") || "-"}`;
    for (const child of grid.children) {
      const i = Number(child.dataset.index);
      const on = selected.has(i);
      child.style.outline = on ? "3px solid #e0553b" : "1px solid rgba(128,128,128,0.4)";
      child.style.opacity = on ? "1" : "0.72";
      child.querySelector(".sv-badge").textContent = on ? `✓ ${i}` : `${i}`;
    }
    if (idxWidget.callback) idxWidget.callback(idxWidget.value);
    node.setDirtyCanvas(true, true);
  };

  const render = () => {
    grid.innerHTML = "";
    for (const t of thumbs) {
      const cell = document.createElement("div");
      cell.dataset.index = String(t.index);
      Object.assign(cell.style, {
        position: "relative", cursor: "pointer",
        border: "1px solid rgba(128,128,128,0.4)", borderRadius: "4px",
        overflow: "hidden", lineHeight: "0",
      });
      const img = document.createElement("img");
      img.src = "data:image/jpeg;base64," + t.data;
      Object.assign(img.style, { width: "100%", display: "block" });
      const badge = document.createElement("span");
      badge.className = "sv-badge";
      Object.assign(badge.style, {
        position: "absolute", left: "2px", bottom: "2px",
        fontSize: "10px", lineHeight: "1.2", padding: "0 3px",
        background: "rgba(0,0,0,0.65)", color: "#fff", borderRadius: "3px",
      });
      badge.textContent = String(t.index);
      cell.appendChild(img);
      cell.appendChild(badge);
      cell.onclick = () => {
        if (selected.has(t.index)) selected.delete(t.index);
        else selected.add(t.index);
        sync();
      };
      grid.appendChild(cell);
    }
    sync();
  };

  let lastFile = "";
  const load = async () => {
    const file = fileWidget.value;
    if (!file || file === lastFile) return;
    lastFile = file;
    status.textContent = "Loading frames...";
    try {
      const resp = await api.fetchApi(
        `/sparkvsr/frames?file=${encodeURIComponent(file)}&max=120`
      );
      const data = await resp.json();
      if (data.error) throw new Error(data.error);
      thumbs = data.frames || [];
      status.textContent = `${data.count || thumbs.length} frames (step ${data.step || 1})`;
      render();
    } catch (e) {
      status.textContent = "Frames unavailable: " + e.message;
    }
  };

  mkBtn("Refresh", () => { lastFile = ""; load(); });
  mkBtn("Clear", () => { selected.clear(); sync(); });
  mkBtn("Every 8", () => {
    selected.clear();
    for (const t of thumbs) selected.add(t.index);
    // keep every ~8th source frame, based on rendered step
    const step = Math.max(1, Math.round(8 / Math.max(1, thumbs[1]?.index - thumbs[0]?.index || 1)));
    selected.clear();
    for (let k = 0; k < thumbs.length; k += step) selected.add(thumbs[k].index);
    sync();
  });
  mkBtn("From text", () => {
    selected.clear();
    for (const i of parseList(idxWidget.value)) selected.add(i);
    sync();
  });

  const domWidget = node.addDOMWidget("keyframes", "div", root, {
    serialize: false,
    hideOnZoom: false,
  });
  domWidget.computeSize = () => [320, 200];
  node.setSize([360, 320]);

  const oldFileCb = fileWidget.callback;
  fileWidget.callback = function (...args) {
    const r = oldFileCb ? oldFileCb.apply(this, args) : undefined;
    lastFile = "";
    load();
    return r;
  };

  // Initial sync from the serialized widget value, then fetch thumbnails.
  setTimeout(() => { sync(); load(); }, 120);
}

app.registerExtension({
  name: "SparkVSR.Native.KeyframePicker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = orig ? orig.apply(this, arguments) : undefined;
      try { buildPicker(this); } catch (e) { console.error("[SparkVSR]", e); }
      return r;
    };
  },
});
