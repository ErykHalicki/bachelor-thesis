"""The stream page's javascript, run under node against a stub DOM.

Everything else about the stream is checked from the python side, which cannot see a
rendering bug at all: the server can serve perfectly correct json to a page that throws
on the first chart and draws nothing. This drives the real chart renderer over a real
payload and checks what it actually built.

Skipped where node is unavailable, so it never blocks a run on the robot.
"""

import json
import re
import shutil
import subprocess

import pytest

from thesis.utils.stream.page import PAGE

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

# appendChild moves an existing child rather than duplicating it, as the real
# one does, which is what makes the leak check below meaningful
DOM_STUB = """
const LOG = [];
function makeEl(cls) {
  return {
    className: cls || "", dataset: {}, children: [], style: {}, clientWidth: 800,
    textContent: "",
    appendChild(c) {
      const i = this.children.indexOf(c);
      if (i >= 0) this.children.splice(i, 1);
      this.children.push(c); return c;
    },
    replaceChildren(...c) { this.children = c; },
    querySelectorAll(sel) {           // recursive, so the count is layout-independent
      const want = sel.replace(".", "");
      const found = [];
      for (const c of this.children) {
        if (c.className.split(" ").includes(want)) found.push(c);
        found.push(...c.querySelectorAll(sel));
      }
      return found;
    },
    querySelector() { return makeEl(); },
    set innerHTML(v) { this.children = []; },
    getContext() { return new Proxy({}, {get: () => () => {}}); },
  };
}
globalThis.document = {createElement: () => makeEl(), getElementById: () => makeEl("root")};
globalThis.window = {devicePixelRatio: 1, addEventListener() {}};
globalThis.uPlot = class {
  constructor(o) { LOG.push(o.title); HEIGHTS.push(o.height); this.title = o.title; }
  setData() {} destroy() {} setSize() {}
};
const HEIGHTS = [];
let TICK = null;
globalThis.setInterval = fn => { TICK = fn; };   // captured so the driver can re-poll
globalThis.fetch = async () => ({ok: true, json: async () => ({charts: CHARTS})});
"""

DRIVER = """
(async () => {
  const root = document.createElement();
  root.className = "panel";
  let failed = null;
  process.on("unhandledRejection", e => { failed = String(e && e.message || e); });
  RENDERERS.chart(root, {name: "observation", src: "/panel/observation"});
  await new Promise(r => setTimeout(r, 20));
  const first = root.querySelectorAll(".chart").length;
  for (let i = 0; i < 3; i++) { await TICK(); }
  console.log(JSON.stringify({
    boxes_after_one_poll: first,
    boxes_after_four_polls: root.querySelectorAll(".chart").length,
    rendered: LOG,
    heights: HEIGHTS,
    failed,
  }));
})();
"""


def render(tmp_path, charts):
    """Run the page's chart renderer over `charts` and report what it built."""
    js = re.search(r"<script>(.*?)</script>", PAGE.decode(), re.S).group(1)
    js = js.replace("main();", "")
    script = tmp_path / "page.js"
    script.write_text(
        DOM_STUB + f"\nconst CHARTS = {json.dumps(charts)};\n" + js + DRIVER
    )
    out = subprocess.run(
        ["node", str(script)], capture_output=True, text=True, timeout=60, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def chart(name, labels=("j1", "j2")):
    return {
        "name": name,
        "spec": {"title": name, "precision": 1,
                 "series": [{}] + [{"label": label} for label in labels]},
        "data": [[0.0, 0.1], *[[1.0, 2.0] for _ in labels]],
    }


def test_every_chart_in_the_payload_is_drawn(tmp_path):
    """A payload of four charts must produce four plots. Drawing only the first and
    throwing on the rest still serves correct json, so nothing python-side catches it."""
    names = ["action.pos", "state.pos", "state.torq", "state.vel"]
    result = render(tmp_path, [chart(n) for n in names])

    assert result["failed"] is None
    assert sorted(result["rendered"]) == sorted(names)


def test_polling_does_not_accumulate_chart_boxes(tmp_path):
    """Each poll re-sends every chart, and the page must reuse the boxes it already
    has -- otherwise it grows a blank stripe per tick for as long as it is open."""
    charts = [chart(n) for n in ["action.pos", "state.pos", "state.torq"]]
    result = render(tmp_path, charts)

    assert result["boxes_after_one_poll"] == len(charts)
    assert result["boxes_after_four_polls"] == len(charts)
    assert len(result["rendered"]) == len(charts)


def test_every_chart_is_built_at_the_configured_height(tmp_path):
    """uPlot takes pixels, so the height is set in two places -- construction and the
    resize handler -- and they have to agree. The width comes from the grid cell, which
    only a real browser resolves."""
    result = render(tmp_path, [chart("state.pos"), chart("state.torq")])
    assert result["heights"] == [300, 300]


def test_a_chart_arriving_late_still_gets_drawn(tmp_path):
    """Series appear as data does -- a torque chart shows up only once a torque value
    has been pushed -- so a chart added after the first poll must render too."""
    result = render(tmp_path, [chart("state.pos"), chart("late.torq")])
    assert result["failed"] is None
    assert sorted(result["rendered"]) == ["late.torq", "state.pos"]
