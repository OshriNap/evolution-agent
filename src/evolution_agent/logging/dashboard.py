"""Web dashboard served via existing Caddy instance as static files.

Writes index.html + _data/data.json to a web root directory. A background
thread periodically refreshes data.json from the JSONL evolution log.
Caddy serves these as static files under /evolution/*.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any

logger = logging.getLogger(__name__)

_WEB_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent / ".dashboard")
_CADDY_CONFIG = "/etc/caddy/Caddyfile"
_REFRESH_INTERVAL = 3  # seconds


def _json_safe(obj):
    """Replace inf/nan floats with None for valid JSON."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evolution Agent Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0a0a0a;color:#d0d0d0;padding:16px 20px}
h1{color:#00ff88;margin-bottom:4px;font-size:1.5em;font-weight:600}
h2{color:#6cb4ee;margin:0 0 8px;font-size:1em;font-weight:500}
h3{color:#888;font-size:.85em;font-weight:400;margin:8px 0 4px}
.sub{color:#666;font-size:.8em;margin-bottom:16px}
.row{display:flex;gap:16px;margin-bottom:16px}
.row>*{flex:1;min-width:0}
.card{background:#141414;border:1px solid #252525;border-radius:6px;padding:14px;overflow:hidden}
.stats{display:flex;gap:24px;flex-wrap:wrap}
.st{text-align:center}.st .v{font-size:1.8em;color:#00ff88;font-weight:700;font-family:'Courier New',monospace}.st .l{color:#666;font-size:.75em}
canvas{width:100%;display:block}
.chart-wrap{height:220px;position:relative}
table{width:100%;border-collapse:collapse;font-size:.82em}
th{color:#6cb4ee;font-weight:400;text-align:left;padding:5px 8px;border-bottom:1px solid #252525}
td{padding:5px 8px;border-bottom:1px solid #1a1a1a}
.err{color:#f44}.ok{color:#00ff88}
pre{background:#0e0e0e;padding:10px;border-radius:4px;font-size:.8em;max-height:350px;overflow:auto;line-height:1.5;white-space:pre-wrap;word-break:break-all}
.tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:.72em;font-weight:500;margin-right:4px}
.tag-obs{background:#1a2a1a;color:#4a4}.tag-conc{background:#1a1a2a;color:#66f}.tag-sug{background:#2a2a1a;color:#da0}.tag-pat{background:#2a1a2a;color:#a4a}.tag-phase{background:#1a2a2a;color:#4aa}
.entry{margin:3px 0;font-size:.82em;line-height:1.4;padding:3px 0;border-bottom:1px solid #1a1a1a}
.entry b{color:#888;font-weight:400}
.scroll{max-height:320px;overflow-y:auto}
#status{color:#555;font-size:.72em;position:fixed;top:8px;right:16px}
@media(max-width:900px){.row{flex-direction:column}}
</style>
</head>
<body>
<h1>Evolution Agent</h1>
<div class="sub">Evolutionary Code Optimization</div>
<span id="status">loading...</span>

<div class="row">
  <div class="card">
    <h2>Population</h2>
    <div class="stats">
      <div class="st"><div class="v" id="gen">-</div><div class="l">Generation</div></div>
      <div class="st"><div class="v" id="best">-</div><div class="l">Best</div></div>
      <div class="st"><div class="v" id="avg">-</div><div class="l">Avg</div></div>
      <div class="st"><div class="v" id="div">-</div><div class="l">Diversity</div></div>
      <div class="st"><div class="v" id="pop">-</div><div class="l">Pop Size</div></div>
    </div>
  </div>
</div>

<div class="row">
  <div class="card" style="flex:2">
    <h2>Fitness</h2>
    <div class="chart-wrap"><canvas id="chart"></canvas></div>
  </div>
  <div class="card" style="flex:1">
    <h2>Diversity</h2>
    <div class="chart-wrap"><canvas id="divchart"></canvas></div>
  </div>
</div>

<div class="row">
  <div class="card" style="flex:2">
    <h2>Meta-Analysis</h2>
    <div id="analysis" class="scroll"><span style="color:#555">No analysis yet</span></div>
  </div>
  <div class="card" style="flex:1">
    <h2>Detected Patterns</h2>
    <div id="patterns" class="scroll"><span style="color:#555">None</span></div>
  </div>
</div>

<div class="row">
  <div class="card">
    <h2>Scratchpad</h2>
    <div id="scratchpad" class="scroll" style="max-height:250px"><span style="color:#555">Empty</span></div>
  </div>
</div>

<div class="row">
  <div class="card" style="flex:2">
    <h2>Best Code</h2>
    <pre id="best-code">loading...</pre>
  </div>
  <div class="card" style="flex:1">
    <h2>Recent Evaluations</h2>
    <div class="scroll" style="max-height:350px">
    <table>
      <thead><tr><th>ID</th><th>Fitness</th><th>Gen</th><th>Type</th><th>Error</th></tr></thead>
      <tbody id="evals"></tbody>
    </table>
    </div>
  </div>
</div>

<script>
const base = window.location.pathname.replace(/\/index\.html$/, '').replace(/\/$/, '');

function drawLine(canvas, datasets, opts={}) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const wrap = canvas.parentElement;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(dpr, dpr);
  const pad = {l:50, r:12, t:12, b:24};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;

  // compute bounds
  let allY = [];
  datasets.forEach(ds => ds.data.forEach(v => { if(v!=null && isFinite(v)) allY.push(v); }));
  if(!allY.length) return;
  let yMin = Math.min(...allY), yMax = Math.max(...allY);
  if(yMax-yMin < 1e-10) { yMin -= 0.5; yMax += 0.5; }
  const yPad = (yMax-yMin)*0.08;
  yMin -= yPad; yMax += yPad;
  const n = datasets[0].data.length;

  // grid
  ctx.strokeStyle = '#1e1e1e'; ctx.lineWidth = 1;
  const nGrid = 5;
  ctx.font = '10px system-ui'; ctx.fillStyle = '#555'; ctx.textAlign = 'right';
  for(let i=0;i<=nGrid;i++){
    const y = pad.t + ph - (i/nGrid)*ph;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+pw,y); ctx.stroke();
    const val = yMin + (i/nGrid)*(yMax-yMin);
    ctx.fillText(val.toFixed(3), pad.l-4, y+3);
  }
  // x labels
  ctx.textAlign='center'; ctx.fillStyle='#555';
  const xStep = Math.max(1, Math.floor(n/8));
  for(let i=0;i<n;i+=xStep){
    const x = pad.l + (i/(n-1||1))*pw;
    ctx.fillText(i+1, x, H-4);
  }

  // lines
  datasets.forEach(ds => {
    ctx.strokeStyle = ds.color;
    ctx.lineWidth = ds.width || 2;
    ctx.beginPath();
    let started = false;
    ds.data.forEach((v,i) => {
      if(v==null || !isFinite(v)) return;
      const x = pad.l + (i/(n-1||1))*pw;
      const y = pad.t + ph - ((v-yMin)/(yMax-yMin))*ph;
      if(!started){ctx.moveTo(x,y);started=true} else ctx.lineTo(x,y);
    });
    ctx.stroke();

    // dots
    if(ds.dots !== false){
      ctx.fillStyle = ds.color;
      ds.data.forEach((v,i) => {
        if(v==null || !isFinite(v)) return;
        const x = pad.l + (i/(n-1||1))*pw;
        const y = pad.t + ph - ((v-yMin)/(yMax-yMin))*ph;
        ctx.beginPath(); ctx.arc(x,y,2.5,0,Math.PI*2); ctx.fill();
      });
    }
  });

  // legend
  let lx = pad.l + 8;
  ctx.font = '11px system-ui';
  datasets.forEach(ds => {
    ctx.fillStyle = ds.color;
    ctx.fillRect(lx, pad.t+4, 14, 3);
    ctx.fillStyle = '#888';
    ctx.textAlign = 'left';
    ctx.fillText(ds.label, lx+18, pad.t+10);
    lx += ctx.measureText(ds.label).width + 38;
  });

  // analysis markers
  if(opts.markers){
    opts.markers.forEach(gen => {
      const i = gen - 1;
      if(i<0||i>=n) return;
      const x = pad.l + (i/(n-1||1))*pw;
      ctx.strokeStyle = 'rgba(218,170,0,0.4)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+ph); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#da0';
      ctx.font = '9px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('A', x, pad.t+ph+12);
    });
  }
}

async function refresh() {
  try {
    const r = await fetch(base + '/_data/data.json?' + Date.now());
    const D = await r.json();
    document.getElementById('status').textContent = new Date().toLocaleTimeString();

    const gens = D.generations || [];
    if(gens.length) {
      const L = gens[gens.length-1];
      document.getElementById('gen').textContent = L.number||'-';
      document.getElementById('best').textContent = L.best_fitness!=null?L.best_fitness.toFixed(4):'-';
      document.getElementById('avg').textContent = L.avg_fitness!=null?L.avg_fitness.toFixed(4):'-';
      document.getElementById('div').textContent = L.diversity!=null?L.diversity.toFixed(3):'-';
      document.getElementById('pop').textContent = L.population_size||'-';

      // analysis gen markers
      const aGens = (D.analyses||[]).map(a=>a.generation).filter(g=>g);

      // Fitness chart
      drawLine(document.getElementById('chart'), [
        {label:'Best', data:gens.map(g=>g.best_fitness), color:'#00ff88', width:2.5},
        {label:'Average', data:gens.map(g=>g.avg_fitness), color:'#3388dd', width:1.5},
      ], {markers: aGens});

      // Diversity chart
      drawLine(document.getElementById('divchart'), [
        {label:'Diversity', data:gens.map(g=>g.diversity), color:'#aa44ff', width:2},
      ]);
    }

    if(D.best_code) document.getElementById('best-code').textContent = D.best_code;

    // Evaluations
    const tbody = document.getElementById('evals');
    tbody.innerHTML='';
    (D.recent_evals||[]).slice(-25).reverse().forEach(e=>{
      const tr=document.createElement('tr');
      tr.innerHTML=
        '<td>'+(e.id||'-').slice(0,6)+'</td>'+
        '<td class="'+(e.error?'err':'ok')+'">'+(e.fitness!=null?e.fitness.toFixed(4):'-')+'</td>'+
        '<td>'+(e.generation||'-')+'</td>'+
        '<td>'+(e.mutation_type||'seed')+'</td>'+
        '<td class="err">'+(e.error||'').slice(0,30)+'</td>';
      tbody.appendChild(tr);
    });

    // Meta-Analysis
    const aDiv = document.getElementById('analysis');
    const analyses = D.analyses||[];
    if(analyses.length){
      let h='';
      analyses.slice(-3).reverse().forEach(a=>{
        h+='<div style="margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1e1e1e">';
        h+='<div><span class="tag tag-phase">Gen '+a.generation+'</span> <span class="tag tag-phase">'+a.phase+'</span></div>';
        (a.observations||[]).forEach(o=>{h+='<div class="entry"><span class="tag tag-obs">OBS</span> '+o+'</div>'});
        (a.conclusions||[]).forEach(c=>{h+='<div class="entry"><span class="tag tag-conc">CONC</span> '+c+'</div>'});
        (a.suggestions||[]).forEach(s=>{
          const t=typeof s==='string'?s:(s.content||JSON.stringify(s));
          const p=typeof s==='object'?s.priority||'':'';
          h+='<div class="entry"><span class="tag tag-sug">SUG'+(p?' '+p:'')+'</span> '+t+'</div>';
        });
        if(a.mutation_guidance) h+='<div class="entry" style="color:#888"><b>Guidance:</b> '+a.mutation_guidance.slice(0,200)+'</div>';
        h+='</div>';
      });
      aDiv.innerHTML=h;
    }

    // Patterns
    const pDiv = document.getElementById('patterns');
    if(analyses.length){
      const allPats={};
      analyses.forEach(a=>{(a.detected_patterns||[]).forEach(p=>{allPats[p]=(allPats[p]||0)+1})});
      if(Object.keys(allPats).length){
        let h='';
        Object.entries(allPats).sort((a,b)=>b[1]-a[1]).forEach(([p,c])=>{
          h+='<div class="entry"><span class="tag tag-pat">'+c+'x</span> '+p.replace(/_/g,' ')+'</div>';
        });
        pDiv.innerHTML=h;
      }
    }

    // Scratchpad — build from analysis observations/conclusions/suggestions
    const spDiv = document.getElementById('scratchpad');
    if(analyses.length){
      let h='';
      analyses.forEach(a=>{
        const g = a.generation||'?';
        (a.observations||[]).forEach(o=>{h+='<div class="entry"><b>gen '+g+'</b> <span class="tag tag-obs">obs</span> '+o+'</div>'});
        (a.conclusions||[]).forEach(c=>{h+='<div class="entry"><b>gen '+g+'</b> <span class="tag tag-conc">conc</span> '+c+'</div>'});
        (a.suggestions||[]).forEach(s=>{
          const t=typeof s==='string'?s:(s.content||'');
          h+='<div class="entry"><b>gen '+g+'</b> <span class="tag tag-sug">sug</span> '+t+'</div>';
        });
      });
      // Meta-optimizer events
      (D.meta_events||[]).forEach(e=>{
        const d=e.data||{};
        let txt=JSON.stringify(d).slice(0,120);
        h+='<div class="entry"><span class="tag" style="background:#222;color:#888">META</span> '+txt+'</div>';
      });
      spDiv.innerHTML=h||'<span style="color:#555">Empty</span>';
    }

  } catch(err) {
    document.getElementById('status').textContent='Error: '+err.message;
    document.getElementById('best-code').textContent='JS Error: '+err.stack;
  }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


def _load_run_data(run_dir: str) -> dict[str, Any]:
    """Load evolution data from JSONL log."""
    log_path = Path(run_dir) / "evolution.jsonl"
    generations: list[dict[str, Any]] = []
    recent_evals: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []
    meta_events: list[dict[str, Any]] = []
    best_code: str = ""

    if not log_path.exists():
        return {"generations": [], "recent_evals": [], "best_code": "",
                "analyses": [], "meta_events": []}

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("type") == "generation":
                generations.append(entry.get("data", {}))
            elif entry.get("type") == "evaluation":
                data = entry.get("data", {})
                recent_evals.append(data)
                if len(recent_evals) > 100:
                    recent_evals = recent_evals[-100:]
            elif entry.get("type") == "summary":
                s = entry.get("data", {})
                if s.get("best_code"):
                    best_code = s["best_code"]
            elif entry.get("type") == "analysis":
                a = entry.get("data", {})
                a["generation"] = entry.get("generation")
                analyses.append(a)
            elif entry.get("type") in ("meta_optimizer", "library_update"):
                meta_events.append({
                    "type": entry["type"],
                    "data": entry.get("data", {}),
                })

    if not best_code:
        pop_files = sorted(Path(run_dir).glob("population_gen*.json"))
        if pop_files:
            try:
                pop = json.loads(pop_files[-1].read_text(encoding="utf-8"))
                if pop:
                    best_ind = max(pop, key=lambda x: x.get("fitness", float("-inf")))
                    best_code = best_ind.get("code", "")
            except Exception:
                pass

    return {
        "generations": generations,
        "recent_evals": recent_evals,
        "best_code": best_code,
        "analyses": analyses,
        "latest_analysis": analyses[-1] if analyses else None,
        "meta_events": meta_events,
    }


def _refresh_data_json(run_dir: str, web_root: str, stop: Event) -> None:
    """Background thread: periodically write _data/data.json from JSONL log."""
    api_dir = Path(web_root) / "_data"
    api_dir.mkdir(parents=True, exist_ok=True)
    data_path = api_dir / "data.json"

    while not stop.is_set():
        try:
            data = _load_run_data(run_dir)
            tmp = data_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_json_safe(data), default=str), encoding="utf-8")
            tmp.rename(data_path)  # atomic on same filesystem
        except Exception as e:
            logger.warning("Failed to refresh data.json: %s", e)
        stop.wait(_REFRESH_INTERVAL)



def deploy_static(run_dir: str, web_root: str | None = None) -> None:
    """Write index.html + _data/data.json to the web root (one-shot).

    Call this to initialize or update the static dashboard files.
    Requires write access to web_root (may need sudo).
    """
    wr = Path(web_root or _WEB_ROOT)
    wr.mkdir(parents=True, exist_ok=True)
    api_dir = wr / "_data"
    api_dir.mkdir(exist_ok=True)

    (wr / "index.html").write_text(_DASHBOARD_HTML, encoding="utf-8")

    data = _load_run_data(run_dir)
    (api_dir / "data.json").write_text(
        json.dumps(_json_safe(data), default=str), encoding="utf-8",
    )
    logger.info("Deployed dashboard to %s", wr)


def serve_dashboard(run_dir: str, port: int = 8050) -> None:
    """Refresh dashboard data for existing Caddy static site.

    Caddy serves /var/www/evolution/ at http://192.168.50.114/evolution.
    This command keeps _data/data.json updated from the JSONL evolution log.
    """
    web_root = os.environ.get("EVOL_WEB_ROOT", _WEB_ROOT)
    wr = Path(web_root)

    if not wr.exists():
        print(f"Web root {wr} does not exist. Run setup first:")
        print(f"  sudo mkdir -p {wr}/api")
        print(f"  sudo chown $USER {wr} {wr}/api")
        return

    # Write/update index.html
    (wr / "index.html").write_text(_DASHBOARD_HTML, encoding="utf-8")

    # Initial data refresh
    api_dir = wr / "_data"
    api_dir.mkdir(exist_ok=True)
    data = _load_run_data(run_dir)
    (api_dir / "data.json").write_text(json.dumps(_json_safe(data), default=str), encoding="utf-8")

    print(f"Dashboard: http://192.168.50.114/evolution")
    print(f"Watching:  {run_dir}/evolution.jsonl")
    print(f"Refresh:   every {_REFRESH_INTERVAL}s")
    print("Press Ctrl+C to stop\n")

    # Background refresh loop
    stop = Event()
    refresh_thread = Thread(
        target=_refresh_data_json,
        args=(run_dir, web_root, stop),
        daemon=True,
    )
    refresh_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop.set()
        refresh_thread.join(timeout=5)
        print("Stopped (files remain at {})".format(web_root))
