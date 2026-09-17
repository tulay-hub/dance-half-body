"""Small local web UI for Lens110 ankle PR/UL conversion.

Run:
    python -u pitchRoll2UpperLower/model_eval_vis/ankle_mlp_converter_ui.py --port 7862

Then open:
    http://127.0.0.1:7862
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from ankle_conversion_backends import (
    DEFAULT_OLD_PKL_DIR,
    DEFAULT_POLY_DEGREE3_DIR,
    DEFAULT_SKLEARN_PYTHON,
    DEFAULT_TORCH_PYTHON,
    DEFAULT_WEIGHTED_MLP_DIR,
    ConversionBackends,
    PR_POS,
    PR_VEL,
    UL_POS,
    UL_VEL,
)


ROOT = Path(__file__).resolve().parent


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Lens110 Ankle PR/UL Converter</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; background: #f6f7f9; color: #1f2328; }
    main { max-width: 1080px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 24px; margin: 0 0 16px; }
    .bar { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }
    select, button, input { font: inherit; }
    select, button { height: 36px; padding: 0 12px; border: 1px solid #c7ccd1; border-radius: 6px; background: white; }
    button { background: #1f6feb; color: white; border-color: #1f6feb; cursor: pointer; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .panel { background: white; border: 1px solid #d8dee4; border-radius: 8px; padding: 16px; }
    h2 { font-size: 16px; margin: 0 0 12px; }
    label { display: grid; grid-template-columns: 1fr 120px; gap: 12px; align-items: center; margin: 8px 0; }
    input { height: 30px; padding: 0 8px; border: 1px solid #c7ccd1; border-radius: 6px; text-align: right; }
    table { width: 100%; border-collapse: collapse; }
    td, th { border-bottom: 1px solid #d8dee4; padding: 8px; text-align: right; }
    td:first-child, th:first-child { text-align: left; }
    .hint { color: #57606a; font-size: 13px; margin: 8px 0 0; }
    @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>Lens110 Ankle PR/UL Converter</h1>
  <div class="bar">
    <select id="backend">
      <option value="weighted_mlp">weighted_mlp</option>
      <option value="poly_degree3">poly_degree3</option>
      <option value="old_pkl">old_pkl</option>
    </select>
    <select id="mode">
      <option value="pr_to_ul">PR -> UL</option>
      <option value="ul_to_pr">UL -> PR</option>
    </select>
    <button id="convert">Convert</button>
    <button id="urdf_default" type="button">URDF PR Default</button>
    <button id="deploy_default" type="button">Deploy UL 0.29</button>
    <button id="deploy_stand" type="button">Deploy UL 0.26</button>
  </div>
  <div class="grid">
    <section class="panel">
      <h2>Input</h2>
      <div id="inputs"></div>
      <p class="hint">只看角度转换时，速度填 0。模型会同时输出位置和速度。</p>
      <p class="hint" id="model_info"></p>
    </section>
    <section class="panel">
      <h2>Output</h2>
      <table>
        <thead><tr><th>Name</th><th>Value</th></tr></thead>
        <tbody id="outputs"></tbody>
      </table>
    </section>
  </div>
</main>
<script>
const names = {
  pr_to_ul: ["left_pitch_pos","left_roll_pos","right_pitch_pos","right_roll_pos","left_pitch_vel","left_roll_vel","right_pitch_vel","right_roll_vel"],
  ul_to_pr: ["left_upper_pos","left_lower_pos","right_upper_pos","right_lower_pos","left_upper_vel","left_lower_vel","right_upper_vel","right_lower_vel"]
};

function renderInputs(values=null) {
  const mode = document.getElementById("mode").value;
  const box = document.getElementById("inputs");
  box.innerHTML = "";
  names[mode].forEach((name, i) => {
    const label = document.createElement("label");
    const span = document.createElement("span");
    const input = document.createElement("input");
    span.textContent = name;
    input.id = "in_" + i;
    input.type = "number";
    input.step = "0.000001";
    input.value = values ? values[i] : "0";
    label.appendChild(span);
    label.appendChild(input);
    box.appendChild(label);
  });
}

async function convert() {
  const backend = document.getElementById("backend").value;
  const mode = document.getElementById("mode").value;
  const values = names[mode].map((_, i) => Number(document.getElementById("in_" + i).value || 0));
  const resp = await fetch("/convert", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({backend, mode, values})
  });
  const data = await resp.json();
  if (!resp.ok) {
    document.getElementById("outputs").innerHTML = `<tr><td>Error</td><td>${data.error}</td></tr>`;
    return;
  }
  document.getElementById("model_info").textContent = data.backend_info || "";
  const out = document.getElementById("outputs");
  out.innerHTML = "";
  data.names.forEach((name, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${name}</td><td>${Number(data.values[i]).toFixed(6)}</td>`;
    out.appendChild(tr);
  });
}

document.getElementById("backend").addEventListener("change", convert);
document.getElementById("mode").addEventListener("change", () => renderInputs());
document.getElementById("convert").addEventListener("click", convert);
document.getElementById("urdf_default").addEventListener("click", () => {
  document.getElementById("mode").value = "pr_to_ul";
  renderInputs([-0.15, 0, -0.15, 0, 0, 0, 0, 0]);
  convert();
});
document.getElementById("deploy_default").addEventListener("click", () => {
  document.getElementById("mode").value = "ul_to_pr";
  renderInputs([0.29, 0.29, 0.29, 0.29, 0, 0, 0, 0]);
  convert();
});
document.getElementById("deploy_stand").addEventListener("click", () => {
  document.getElementById("mode").value = "ul_to_pr";
  renderInputs([0.26, 0.26, 0.26, 0.26, 0, 0, 0, 0]);
  convert();
});
renderInputs();
</script>
</body>
</html>
"""


def make_handler(converter: ConversionBackends):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlparse(self.path).path != "/":
                self.send_error(404)
                return
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if urlparse(self.path).path != "/convert":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                backend = payload.get("backend", "weighted_mlp")
                names, values = converter.convert(backend, payload["mode"], payload["values"])
                body = json.dumps(
                    {"names": names, "values": values, "backend_info": converter.describe(backend)}
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, fmt, *args):
            print(fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Local web UI for Lens110 ankle PR/UL conversion.")
    parser.add_argument("--weighted_mlp_dir", default=str(DEFAULT_WEIGHTED_MLP_DIR))
    parser.add_argument("--poly_degree3_dir", default=str(DEFAULT_POLY_DEGREE3_DIR))
    parser.add_argument("--old_pkl_dir", default=str(DEFAULT_OLD_PKL_DIR))
    parser.add_argument("--weighted_mlp_python", default=str(DEFAULT_TORCH_PYTHON))
    parser.add_argument("--old_pkl_python", default=str(DEFAULT_SKLEARN_PYTHON))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()

    converter = ConversionBackends(
        args.weighted_mlp_dir,
        args.poly_degree3_dir,
        args.old_pkl_dir,
        args.weighted_mlp_python,
        args.old_pkl_python,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(converter))
    print(f"open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
