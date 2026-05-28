import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from typing import AsyncGenerator, Dict, Any, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agents import run_researcher, run_writer, run_reviewer, PipelineConfig

app = FastAPI()

# Serve static files (logos)
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory store for active pipeline runs
_runs: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Live metrics helpers
# ---------------------------------------------------------------------------

# NUMA topology: populated once at startup
_numa_map: Dict[int, List[int]] = {}  # node_id -> [cpu_ids]


def _parse_numa_topology() -> Dict[int, List[int]]:
    """Parse NUMA node -> CPU mapping from lscpu."""
    result: Dict[int, List[int]] = {}
    try:
        out = subprocess.check_output("lscpu", text=True, timeout=5)
        for line in out.splitlines():
            m = re.match(r"NUMA node(\d+) CPU\(s\):\s+(.+)", line)
            if not m:
                continue
            node = int(m.group(1))
            cpus: List[int] = []
            for part in m.group(2).split(","):
                part = part.strip()
                if "-" in part:
                    lo, hi = part.split("-", 1)
                    cpus.extend(range(int(lo), int(hi) + 1))
                else:
                    cpus.append(int(part))
            result[node] = cpus
    except Exception:
        pass
    return result


def _read_proc_stat() -> Dict[int, List[int]]:
    """Read per-CPU jiffies from /proc/stat. Returns {cpu_id: [user,nice,sys,idle,...]}."""
    cpus: Dict[int, List[int]] = {}
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu") and line[3] != " ":
                    parts = line.split()
                    cpu_id = int(parts[0][3:])
                    cpus[cpu_id] = [int(x) for x in parts[1:]]
    except Exception:
        pass
    return cpus


# Previous snapshot for delta computation
_prev_stat: Dict[int, List[int]] = {}
_prev_time: float = 0.0


def _compute_cpu_utilization() -> Dict[str, Any]:
    """Compute overall and per-NUMA CPU utilization as percentages."""
    global _prev_stat, _prev_time, _numa_map

    if not _numa_map:
        _numa_map = _parse_numa_topology()

    cur = _read_proc_stat()
    now = time.monotonic()

    if not _prev_stat:
        _prev_stat = cur
        _prev_time = now
        return {"overall": 0.0, "per_numa": {str(k): 0.0 for k in sorted(_numa_map)}}

    def _usage(prev_vals: List[int], cur_vals: List[int]) -> float:
        d = [c - p for c, p in zip(cur_vals, prev_vals)]
        total = sum(d)
        if total == 0:
            return 0.0
        idle = d[3] + (d[4] if len(d) > 4 else 0)  # idle + iowait
        return round(100.0 * (1 - idle / total), 1)

    # Overall
    all_prev = [0] * 10
    all_cur = [0] * 10
    for cpu_id in cur:
        if cpu_id in _prev_stat:
            for i in range(min(len(cur[cpu_id]), 10)):
                all_prev[i] += _prev_stat[cpu_id][i]
                all_cur[i] += cur[cpu_id][i]
    overall = _usage(all_prev, all_cur)

    # Per-NUMA
    per_numa: Dict[str, float] = {}
    for node, cpu_ids in sorted(_numa_map.items()):
        np = [0] * 10
        nc = [0] * 10
        for cid in cpu_ids:
            if cid in cur and cid in _prev_stat:
                for i in range(min(len(cur[cid]), 10)):
                    np[i] += _prev_stat[cid][i]
                    nc[i] += cur[cid][i]
        per_numa[str(node)] = _usage(np, nc)

    _prev_stat = cur
    _prev_time = now
    return {"overall": overall, "per_numa": per_numa}


def _get_memory_info() -> Dict[str, Any]:
    """Get system RAM usage from /proc/meminfo."""
    info: Dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal:", "MemAvailable:", "MemFree:")):
                    parts = line.split()
                    info[parts[0].rstrip(":")] = int(parts[1])  # in kB
    except Exception:
        pass
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", info.get("MemFree", 0))
    used = total - avail
    return {
        "total_gb": round(total / (1024 * 1024), 1),
        "used_gb": round(used / (1024 * 1024), 1),
        "pct": round(100.0 * used / total, 1) if total else 0.0,
    }


def _get_gpu_info() -> Dict[str, Any]:
    """Get GPU memory and utilization from nvidia-smi."""
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader,nounits 2>/dev/null | head -1",
            shell=True, text=True, timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        util = parts[0] if parts[0] != "[N/A]" else None
        mem_used = int(parts[1]) if parts[1] != "[N/A]" else None
        mem_total = int(parts[2]) if parts[2] != "[N/A]" else None
        return {
            "util_pct": float(util) if util else None,
            "mem_used_mb": mem_used,
            "mem_total_mb": mem_total,
            "mem_pct": round(100.0 * mem_used / mem_total, 1) if mem_used and mem_total else None,
        }
    except Exception:
        return {"util_pct": None, "mem_used_mb": None, "mem_total_mb": None, "mem_pct": None}


INDEX_HTML = '''
<!doctype html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Pipelined Multi-Agent (2 Endpoints)</title>
    <style>
      body {
        font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        margin: 20px;
        background: radial-gradient(circle at 20% 0%, #eef3ff, #ffffff 60%);
      }
      h2 { margin-bottom: 10px; }
      .controls { display:flex; gap:12px; align-items:center; margin-bottom:18px; }
      input[type="text"] {
        flex:1; padding:10px; border-radius:12px;
        border:1px solid #dcdcdc; font-size:14px;
      }
      button {
        padding:10px 14px; border-radius:12px; border:none;
        font-weight:600; cursor:pointer;
      }
      #run { background: linear-gradient(180deg,#3b82f6,#2563eb); color:white; }
      #stop { background: linear-gradient(180deg,#ef4444,#dc2626); color:white; }
      .pill {
        padding:4px 10px; border-radius:999px; background:#f3f4f6; font-size:12px;
      }
      .row { display:flex; gap:16px; }
      .panel {
        flex:1; border-radius:18px; padding:14px;
        box-shadow:0 8px 24px rgba(0,0,0,.06);
        display:flex; flex-direction:column;
      }
      .researcher { background: linear-gradient(180deg,#fff4d6,#ffffff); border:1px solid #facc15; }
      .writer     { background: linear-gradient(180deg,#e0f2fe,#ffffff); border:1px solid #38bdf8; }
      .reviewer   { background: linear-gradient(180deg,#dcfce7,#ffffff); border:1px solid #22c55e; }
      textarea {
        flex:1; border-radius:12px; border:1px solid rgba(0,0,0,.1);
        padding:10px; resize:none; background:rgba(255,255,255,.7); min-height:260px;
      }
      #log {
        margin-top:18px; border-radius:16px; padding:12px;
        background: linear-gradient(180deg,#f5f3ff,#ffffff);
        border:1px solid #ddd; height:220px; overflow:auto;
        font-family: ui-monospace, monospace;
      }
      .logline {
        padding:6px 8px; margin-bottom:6px; border-radius:10px;
        background:rgba(255,255,255,.8); border-left:6px solid #aaa;
      }
      .logline.researcher { border-left-color:#facc15; }
      .logline.writer { border-left-color:#3b82f6; }
      .logline.reviewer { border-left-color:#22c55e; }
      .logline.status { border-left-color:#a855f7; }

      /* Logo styles */
      .logo-container {
        position: fixed;
        bottom: 16px;
        right: 20px;
        display: flex;
        gap: 14px;
        align-items: center;
        opacity: 0.9;
      }
      .logo-container img {
        height: 30px;
      }

      /* Tech stack section */
      .sysinfo-bar {
        margin-bottom: 18px;
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        align-items: stretch;
      }
      .sysinfo-card {
        flex: 1;
        min-width: 180px;
        border-radius: 14px;
        padding: 12px 16px;
        background: linear-gradient(180deg, #f8fafc, #ffffff);
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,.04);
      }
      .sysinfo-card .label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-bottom: 4px;
      }
      .sysinfo-card .value {
        font-size: 15px;
        font-weight: 600;
        color: #1e293b;
      }
      .sysinfo-card .sub {
        font-size: 12px;
        color: #64748b;
        margin-top: 2px;
      }

      /* Utilization dashboard */
      .util-section {
        margin-top: 18px;
        border-radius: 18px;
        padding: 16px;
        background: linear-gradient(180deg, #f0f9ff, #ffffff);
        border: 1px solid #bae6fd;
        box-shadow: 0 4px 16px rgba(0,0,0,.04);
      }
      .util-section h3 {
        margin: 0 0 12px 0;
        font-size: 14px;
        color: #475569;
      }
      .util-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
      }
      .util-card {
        border-radius: 12px;
        padding: 10px 14px;
        background: rgba(255,255,255,.85);
        border: 1px solid #e2e8f0;
      }
      .util-card .util-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
      }
      .util-card .util-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.3px;
      }
      .util-card .util-pct {
        font-size: 18px;
        font-weight: 700;
        color: #1e293b;
      }
      .util-bar-bg {
        height: 8px;
        border-radius: 4px;
        background: #e2e8f0;
        overflow: hidden;
      }
      .util-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease, background 0.5s ease;
      }
      .util-bar-fill.cpu  { background: linear-gradient(90deg, #3b82f6, #2563eb); }
      .util-bar-fill.mem  { background: linear-gradient(90deg, #8b5cf6, #7c3aed); }
      .util-bar-fill.gpu  { background: linear-gradient(90deg, #10b981, #059669); }
      .util-card .util-sub {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
      }
      .numa-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 6px;
        margin-top: 8px;
      }
      .numa-cell {
        border-radius: 8px;
        padding: 6px 8px;
        background: rgba(255,255,255,.9);
        border: 1px solid #e2e8f0;
        text-align: center;
      }
      .numa-cell .numa-id {
        font-size: 10px;
        color: #94a3b8;
        text-transform: uppercase;
      }
      .numa-cell .numa-pct {
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
      }
      .numa-cell .numa-bar-bg {
        height: 4px;
        border-radius: 2px;
        background: #e2e8f0;
        margin-top: 3px;
        overflow: hidden;
      }
      .numa-cell .numa-bar-fill {
        height: 100%;
        border-radius: 2px;
        background: #3b82f6;
        transition: width 0.5s ease;
      }
    </style>
  </head>
  <body>
    <h2>Pipelined Multi-Agent (Researcher &rarr; Writer &rarr; Reviewer)</h2>

    <div class="controls">
      <input id="task" type="text"
        value="Explain how pipelining helps multi-agent latency while preserving dependencies."/>
      <button id="run">Run</button>
      <button id="stop">Stop</button>
      <span id="mode" class="pill">mode: ...</span>
      <span id="routing" class="pill">routing: researcher/reviewer&rarr;CPU, writer&rarr;GPU</span>
    </div>

    <div id="sysinfo" class="sysinfo-bar" style="display:none;"></div>

    <div class="row">
      <div class="panel researcher">
        <h3>Researcher (CPU {{CPU_MODEL}})</h3>
        <textarea id="research" readonly></textarea>
      </div>
      <div class="panel writer">
        <h3>Writer (GPU {{GPU_MODEL}})</h3>
        <textarea id="writer" readonly></textarea>
      </div>
      <div class="panel reviewer">
        <h3>Reviewer (CPU {{CPU_MODEL}})</h3>
        <textarea id="review" readonly></textarea>
      </div>
    </div>

    <div id="log"></div>

    <!-- Utilization Dashboard -->
    <div class="util-section">
      <h3>Live System Utilization</h3>
      <div class="util-grid">
        <div class="util-card">
          <div class="util-header">
            <span class="util-label">CPU Overall</span>
            <span class="util-pct" id="cpu-pct">--</span>
          </div>
          <div class="util-bar-bg"><div class="util-bar-fill cpu" id="cpu-bar" style="width:0%"></div></div>
          <div class="util-sub" id="cpu-sub"></div>
        </div>
        <div class="util-card">
          <div class="util-header">
            <span class="util-label">System RAM</span>
            <span class="util-pct" id="ram-pct">--</span>
          </div>
          <div class="util-bar-bg"><div class="util-bar-fill mem" id="ram-bar" style="width:0%"></div></div>
          <div class="util-sub" id="ram-sub"></div>
        </div>
        <div class="util-card">
          <div class="util-header">
            <span class="util-label">GPU VRAM</span>
            <span class="util-pct" id="gpu-pct">--</span>
          </div>
          <div class="util-bar-bg"><div class="util-bar-fill gpu" id="gpu-bar" style="width:0%"></div></div>
          <div class="util-sub" id="gpu-sub"></div>
        </div>
      </div>
      <div style="margin-top:12px;">
        <div style="font-size:12px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.3px;margin-bottom:6px;">Per-NUMA Node CPU Utilization</div>
        <div class="numa-grid" id="numa-grid"></div>
      </div>
    </div>

    <!-- Logo -->
    <div class="logo-container">
      <img src="/static/intel_logo.png">
    </div>

    <script>
      let runId = null;
      let pollTimer = null;
      let cursor = 0;

      const panelMap = {researcher: "research", writer: "writer", reviewer: "review"};

      function appendText(id, text) {
        const el = document.getElementById(id);
        el.value += text;
        el.scrollTop = el.scrollHeight;
      }

      function logLine(obj) {
        const log = document.getElementById("log");
        const div = document.createElement("div");
        div.className = "logline";
        if (obj.stage) div.classList.add(obj.stage);
        if (obj.event === "status") div.classList.add("status");
        div.textContent = JSON.stringify(obj);
        log.appendChild(div);
        log.scrollTop = log.scrollHeight;
      }

      function showError(stage, msg) {
        const id = panelMap[stage];
        if (!id) return;
        const el = document.getElementById(id);
        el.value += "\\n[ERROR] " + msg;
        el.style.borderColor = "#ef4444";
      }

      async function fetchMode() {
        const r = await fetch("/mode");
        const j = await r.json();
        document.getElementById("mode").textContent = "mode: " + j.mode;
      }

      function resetPanels() {
        document.getElementById("research").value = "";
        document.getElementById("writer").value = "";
        document.getElementById("review").value = "";
        document.getElementById("log").innerHTML = "";
      }

      function processEvent(evt) {
        if (evt.event === "status") {
          logLine({event: "status", ...evt.data});
        } else if (evt.event === "update") {
          const data = evt.data;
          logLine(data);
          if (data.type === "error") showError(data.stage, data.error);
          if (data.type === "text") {
            const id = panelMap[data.stage];
            if (id) appendText(id, data.delta);
          }
        }
      }

      async function poll() {
        if (!runId) return;
        try {
          const r = await fetch("/poll?run_id=" + runId + "&cursor=" + cursor);
          const j = await r.json();
          if (j.events) {
            for (const evt of j.events) processEvent(evt);
            cursor += j.events.length;
          }
          if (j.done) {
            stopPolling();
            return;
          }
        } catch(e) {
          logLine({event: "status", state: "poll error", error: e.message});
        }
      }

      function stopPolling() {
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
      }

      document.getElementById("run").onclick = async () => {
        resetPanels();
        stopPolling();
        Object.values(panelMap).forEach(id => {
          document.getElementById(id).style.borderColor = "";
        });
        try { await fetchMode(); } catch(e) {
          logLine({event: "status", state: "error", error: "Failed to reach server"});
          return;
        }
        const task = document.getElementById("task").value;
        try {
          const r = await fetch("/start", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({task: task})
          });
          const j = await r.json();
          runId = j.run_id;
          cursor = 0;
          logLine({event: "status", state: "started", run_id: runId});
          pollTimer = setInterval(poll, 400);
        } catch(e) {
          logLine({event: "status", state: "error", error: e.message});
        }
      };

      document.getElementById("stop").onclick = () => {
        stopPolling();
        runId = null;
        logLine({event: "status", state: "stopped"});
      };

      fetchMode();

      // Fetch and render hardware info
      (async () => {
        try {
          const r = await fetch("/sysinfo");
          const info = await r.json();
          const bar = document.getElementById("sysinfo");
          const cards = [
            {label: "Azure Instance", value: info.vm_size || "N/A", sub: ""},
            {label: "CPU", value: info.cpu_model, sub: info.cpu_cores + " cores / " + info.numa_nodes + " NUMA nodes"},
            {label: "GPU", value: info.gpu_name || "N/A", sub: info.gpu_vram || ""},
            {label: "Memory", value: info.ram_total, sub: ""},
          ];
          bar.innerHTML = cards.map(c =>
            '<div class="sysinfo-card">' +
              '<div class="label">' + c.label + '</div>' +
              '<div class="value">' + c.value + '</div>' +
              (c.sub ? '<div class="sub">' + c.sub + '</div>' : '') +
            '</div>'
          ).join("");
          bar.style.display = "flex";
        } catch(e) { /* silently skip if sysinfo unavailable */ }
      })();

      // Live utilization polling
      function updateBar(barId, pctId, pct, subId, subText) {
        const bar = document.getElementById(barId);
        const label = document.getElementById(pctId);
        const sub = document.getElementById(subId);
        if (pct != null) {
          bar.style.width = pct + "%";
          label.textContent = pct + "%";
        } else {
          bar.style.width = "0%";
          label.textContent = "N/A";
        }
        if (sub && subText) sub.textContent = subText;
      }

      function renderNuma(perNuma) {
        const grid = document.getElementById("numa-grid");
        const nodes = Object.keys(perNuma).sort((a,b) => +a - +b);
        // Build cells only once, then update
        if (grid.children.length !== nodes.length) {
          grid.innerHTML = nodes.map(n =>
            '<div class="numa-cell" id="numa-' + n + '">' +
              '<div class="numa-id">Node ' + n + '</div>' +
              '<div class="numa-pct" id="numa-pct-' + n + '">--</div>' +
              '<div class="numa-bar-bg"><div class="numa-bar-fill" id="numa-bar-' + n + '" style="width:0%"></div></div>' +
            '</div>'
          ).join("");
        }
        for (const n of nodes) {
          const pct = perNuma[n];
          document.getElementById("numa-pct-" + n).textContent = pct + "%";
          const bar = document.getElementById("numa-bar-" + n);
          bar.style.width = pct + "%";
          // Color by intensity
          if (pct > 70) bar.style.background = "#ef4444";
          else if (pct > 40) bar.style.background = "#f59e0b";
          else bar.style.background = "#3b82f6";
        }
      }

      async function fetchMetrics() {
        try {
          const r = await fetch("/metrics");
          const m = await r.json();
          // CPU
          updateBar("cpu-bar", "cpu-pct", m.cpu.overall, "cpu-sub", "");
          // RAM
          updateBar("ram-bar", "ram-pct", m.memory.pct, "ram-sub",
            m.memory.used_gb + " / " + m.memory.total_gb + " GB");
          // GPU
          const gpuPct = m.gpu.mem_pct;
          updateBar("gpu-bar", "gpu-pct", gpuPct, "gpu-sub",
            m.gpu.mem_used_mb != null ? (m.gpu.mem_used_mb + " / " + m.gpu.mem_total_mb + " MiB") : "");
          // NUMA
          if (m.cpu.per_numa) renderNuma(m.cpu.per_numa);
        } catch(e) { /* skip */ }
      }

      fetchMetrics();
      setInterval(fetchMetrics, 2000);
    </script>
  </body>
</html>
'''


@app.get("/", response_class=HTMLResponse)
def index():
    cfg = PipelineConfig.from_env()
    html = INDEX_HTML
    html = html.replace("{{CPU_MODEL}}", cfg.cpu_model.split("/")[-1])
    html = html.replace("{{GPU_MODEL}}", cfg.gpu_model.split("/")[-1])
    return html


def _run(cmd: str) -> str:
    """Run a shell command and return stripped stdout, or '' on failure."""
    try:
        return subprocess.check_output(cmd, shell=True, text=True, timeout=5).strip()
    except Exception:
        return ""


_sysinfo_cache: Dict[str, Any] | None = None


@app.get("/sysinfo")
def sysinfo():
    """Return hardware specs (cached after first call)."""
    global _sysinfo_cache
    if _sysinfo_cache is not None:
        return _sysinfo_cache

    cpu_model = _run("lscpu | grep 'Model name' | sed 's/.*: *//'")
    cpu_cores = _run("lscpu | grep '^CPU(s):' | awk '{print $2}'")
    numa_nodes = _run("lscpu | grep 'NUMA node(s)' | awk -F: '{print $2}' | tr -d ' '")
    ram_bytes = _run("grep MemTotal /proc/meminfo | awk '{print $2}'")
    try:
        ram_total = f"{int(ram_bytes) // (1024 * 1024)} GB"
    except (ValueError, TypeError):
        ram_total = ram_bytes

    gpu_name = _run(
        "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1"
    )
    gpu_vram = _run(
        "nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1"
    )

    # Azure VM size via Instance Metadata Service
    vm_size = _run(
        "curl -s -H 'Metadata: true'"
        " 'http://169.254.169.254/metadata/instance/compute/vmSize"
        "?api-version=2021-02-01&format=text'"
    )

    _sysinfo_cache = {
        "cpu_model": cpu_model or "unknown",
        "cpu_cores": cpu_cores or "?",
        "numa_nodes": numa_nodes or "?",
        "ram_total": ram_total or "unknown",
        "gpu_name": gpu_name or None,
        "gpu_vram": gpu_vram or None,
        "vm_size": vm_size or None,
    }
    return _sysinfo_cache


@app.get("/metrics")
def metrics():
    """Return live CPU, memory, and GPU utilization."""
    return {
        "cpu": _compute_cpu_utilization(),
        "memory": _get_memory_info(),
        "gpu": _get_gpu_info(),
    }


@app.get("/mode")
def mode():
    cfg = PipelineConfig.from_env()
    return {
        "mode": "dry-run" if cfg.dry_run else "vllm",
        "cpu_url": cfg.cpu_url,
        "gpu_url": cfg.gpu_url,
    }


@app.post("/start")
async def start_run(body: Dict[str, Any]):
    """Kick off pipeline in background, return run_id for polling."""
    task = body.get("task", "")
    cfg = PipelineConfig.from_env()
    run_id = uuid.uuid4().hex[:12]

    run_state = {
        "events": [],
        "done": False,
    }
    _runs[run_id] = run_state

    async def run_pipeline():
        research_q: asyncio.Queue = asyncio.Queue()
        writer_q: asyncio.Queue = asyncio.Queue()
        ui_q: asyncio.Queue = asyncio.Queue()

        async def bridge_research():
            try:
                async for delta in run_researcher(task, cfg):
                    await research_q.put(delta)
                    await ui_q.put({"stage": "researcher", "type": "text", "delta": delta})
            except Exception as exc:
                await ui_q.put({"stage": "researcher", "type": "error", "error": str(exc)})
            finally:
                await research_q.put(None)
                await ui_q.put({"stage": "researcher", "type": "done"})

        async def bridge_writer():
            try:
                async for delta in run_writer(task, research_q, cfg):
                    await writer_q.put(delta)
                    await ui_q.put({"stage": "writer", "type": "text", "delta": delta})
            except Exception as exc:
                await ui_q.put({"stage": "writer", "type": "error", "error": str(exc)})
            finally:
                await writer_q.put(None)
                await ui_q.put({"stage": "writer", "type": "done"})

        async def bridge_reviewer():
            try:
                async for delta in run_reviewer(task, writer_q, cfg):
                    await ui_q.put({"stage": "reviewer", "type": "text", "delta": delta})
            except Exception as exc:
                await ui_q.put({"stage": "reviewer", "type": "error", "error": str(exc)})
            finally:
                await ui_q.put({"stage": "reviewer", "type": "done"})

        bg_tasks = [
            asyncio.create_task(bridge_research()),
            asyncio.create_task(bridge_writer()),
            asyncio.create_task(bridge_reviewer()),
        ]

        run_state["events"].append({
            "event": "status",
            "data": {
                "state": "started",
                "mode": "dry-run" if cfg.dry_run else "vllm",
                "routing": "researcher/reviewer->CPU, writer->GPU",
            },
        })

        done = {"researcher": False, "writer": False, "reviewer": False}
        stall_timeout = max(cfg.writer_timeout_s, 300) + 60
        while True:
            try:
                msg = await asyncio.wait_for(ui_q.get(), timeout=stall_timeout)
            except asyncio.TimeoutError:
                run_state["events"].append({
                    "event": "status",
                    "data": {"state": "error", "error": "Pipeline stalled."},
                })
                break
            if msg.get("type") == "done":
                done[msg["stage"]] = True
            run_state["events"].append({"event": "update", "data": msg})
            if all(done.values()):
                break

        await asyncio.gather(*bg_tasks, return_exceptions=True)
        run_state["events"].append({
            "event": "status",
            "data": {"state": "finished"},
        })
        run_state["done"] = True

    asyncio.create_task(run_pipeline())
    return {"run_id": run_id}


@app.get("/poll")
async def poll(run_id: str, cursor: int = 0):
    """Return new events since cursor."""
    run_state = _runs.get(run_id)
    if not run_state:
        return JSONResponse({"error": "unknown run_id"}, status_code=404)

    events = run_state["events"][cursor:]
    done = run_state["done"]

    # Clean up finished runs after client has seen all events
    if done and cursor + len(events) >= len(run_state["events"]):
        _runs.pop(run_id, None)

    return {"events": events, "done": done}
