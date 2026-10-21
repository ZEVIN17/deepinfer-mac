# -*- coding: utf-8 -*-
"""Build a dependency-free, relocatable static showcase from reviewed evidence."""
from __future__ import annotations
import argparse
import html
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
DIST = SITE / "dist"


def inline(value):
    value = html.escape(value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
    return value


def markdown(text):
    output, fence, table = [], False, False
    for line in text.splitlines():
        if line.startswith("```"):
            output.append("</code></pre>" if fence else "<pre><code>")
            fence = not fence
            continue
        if fence:
            output.append(html.escape(line) + "\n")
            continue
        if table and not line.startswith("|"):
            output.append("</tbody></table>")
            table = False
        if line.startswith("|"):
            if re.match(r"^\|[\s:|\-]+\|$", line):
                continue
            tag = "td" if table else "th"
            if not table:
                output.append("<table><tbody>")
                table = True
            output.append("<tr>" + "".join(f"<{tag}>{inline(x.strip())}</{tag}>" for x in line.strip("|").split("|")) + "</tr>")
        elif line.startswith("#"):
            level = min(4, len(line) - len(line.lstrip("#")))
            output.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif line.startswith("- "):
            output.append(f"<p>• {inline(line[2:])}</p>")
        elif line.strip():
            output.append(f"<p>{inline(line)}</p>")
    if table:
        output.append("</tbody></table>")
    return "\n".join(output)


def document_links(body):
    return re.sub(r'href="((?!https?://)[^"#]+\.(?:md|json|png))"',
                  lambda match: 'href="docs/' + match[1].removeprefix('docs/') + '"', body)


def shell(title, body, lang="en"):
    return f'''<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="Independent DeepInfer-inspired local inference for Apple Silicon, powered by MLX. Real evidence, exact prefix reuse, transparent comparisons."><title>{html.escape(title)} · DeepInfer-Mac</title><link rel="stylesheet" href="assets/style.css"><link rel="icon" href="assets/icon.svg" type="image/svg+xml"></head><body><header><nav class="wrap" aria-label="Main navigation"><a class="brand" href="index.html">deepinfer<i>/</i>mac</a><div class="navlinks"><a href="architecture.html">Architecture</a><a href="benchmarks.html">Evidence</a><a href="contribute.html">Contribute</a><a class="language" href="{'index.html' if lang == 'zh-CN' else 'zh.html'}">{'English' if lang == 'zh-CN' else '中文'}</a><a href="https://github.com/ZEVIN17/deepinfer-mac">GitHub ↗</a></div><span class="edition">EXPERIMENTAL ALPHA<br>0.1.0a1 / APPLE SILICON</span></nav></header><main class="wrap">{body}</main><footer class="wrap"><div>Independent community project · Powered by MLX<br>DeepInfer-inspired. No official DSpark or full V4.1 port.</div><div class="footlinks"><a href="docs/LICENSE">Apache-2.0</a><a href="docs/NOTICE">Attribution</a><a href="compatibility.html">Compatibility</a></div></footer></body></html>'''


def evidence(report):
    if report is None:
        return '<div class="notice">Controlled benchmark results are being verified. No throughput advantage is claimed without the complete report. / 受控基准核验中，不填模拟性能。</div>'
    summary = report["warm_summary"]
    groups = summary.get("durations_all_successful_runs", {})
    scale = max((v["median_total_ms"] for v in groups.values()), default=1)
    plots, rows = [], []
    for mode in dict.fromkeys([*groups, *summary.get("failures", {})]):
        label = html.escape(mode)
        values = groups.get(mode)
        if values is not None:
            value = values["median_total_ms"]
            plots.append(f'<div class="plotrow"><span>{label}</span><span class="bar" style="width:{value / scale * 100:.2f}%"></span><span>{value:.1f} ms</span></div>')
            median, count = f"{value:.2f}", values["n"]
        else:
            median, count = "—", 0
        rows.append(f'<tr><td>{label}</td><td>{count}</td><td>{median}</td><td>{summary.get("failures", {}).get(mode, 0)}</td></tr>')
    cfg, env = report["configuration"], report.get("environment", {})
    status = ("Complete token equivalence passed on this corpus; broader claims need more hardware and workloads."
              if summary.get("equivalent_speedup_claim_supported") else
              "No equivalent speedup is established. Adaptive did not improve on the tuned fixed-length control. Every failure or divergent output remains in the report; elapsed times may compare different work.")
    return f'''<p class="muted">{html.escape(str(env.get("hardware", "See raw report")))} · {html.escape(str(cfg.get("profile")))} · {report.get("repeats")} repeats · {cfg.get("max_tokens")} token limit</p><div class="notice">{status}<br>Full token equality: {summary.get("equal_token_samples", 0)} / {summary.get("paired_samples", 0)} comparison groups.</div><div class="benchplot" role="img" aria-label="Median total request duration in milliseconds; lower is faster. Durations include all successful runs.">{''.join(plots)}</div><table class="results"><thead><tr><th>Variant</th><th>Runs</th><th>Median request ms</th><th>Failures</th></tr></thead><tbody>{''.join(rows)}</tbody></table><p class="muted">Warm loaded models; empty prompt cache before each run. All successful durations shown; mismatches are not removed. Lower is faster. Short fixed corpus, no p95 or general significance claim.</p><a class="textlink" href="data/benchmark.json">Download complete raw JSON ↗</a> · <a class="textlink" href="data/source.tar.gz">Exact benchmark source archive</a> · <a class="textlink" href="data/source-manifest.json">Source SHA-256 manifest</a>'''


def cache_evidence():
    path = SITE / "data/cache-qwen3-4b.json"
    if not path.exists():
        return ""
    report = json.loads(path.read_text())
    rows = []
    for pair in report["pairs"]:
        cold, hot = pair["cold"]["metrics"], pair["hot"]["metrics"]
        rows.append(f'<tr><td>{pair["repeat"] + 1}</td><td>{cold["total_ms"]:.2f}</td><td>{hot["total_ms"]:.2f}</td><td>{hot["cache_hit_tokens"]}</td><td>{"yes" if pair["hot_equal"] else "NO"}</td></tr>')
    return '<section class="section"><span class="index">EXACT PREFIX REUSE / 精确前缀复用</span><h2>Reuse the work, keep the tokens.</h2><p class="muted">Qwen3-4B baseline, warm loaded model. The exact repeated prompt has 1740 input tokens; output is only 2 tokens including EOS. Cold/hot below means prompt-cache state, not model loading. This demonstrates repeated-prefix cost reduction, not long-generation throughput or speculative acceleration.</p><table class="results"><thead><tr><th>Repeat</th><th>Cold KV request ms</th><th>Hot KV request ms</th><th>Prefix hit tokens</th><th>Full output equal</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table><p class="muted">Three repeated pairs match complete tokens and text. Eight branch rounds preserve output equality with 512-token prefix reuse; allocator active bytes rise only 6 KiB and RSS about 0.26 MiB across those rounds, with zero observed swap delta per request. A 1 KiB snapshot budget refuses admission and recomputation remains equal. These are this short, controlled workload only.</p><a class="textlink" href="data/cache-qwen3-4b.json">Complete cache evidence JSON ↗</a></section>'


def recorded_demo():
    if not (ROOT / "docs/evidence/local-inference-qwen3-4b.png").exists():
        return ""
    return '<section class="section"><span class="index">RECORDED LOCAL DEMO / 实际本地演示</span><h2>A real request, on this Mac.</h2><p class="muted">Qwen3-4B baseline · public Chinese prompt · actual bundled UI. A single interactive request including model loading, not a controlled benchmark or a 27B result. Model output may contain factual errors.</p><img class="capture" src="assets/local-inference.png" alt="Actual Qwen3-4B local generation with its output and measured request metrics"><p><a class="textlink" href="data/local-ui-demo.json">Exported result and exact public prompt ↗</a></p></section>'


def homepage(report, zh=False):
    headline = '让推理留在<br>你的 <span>Mac。</span>' if zh else 'Your model.<br>Your <span>silicon.</span>'
    lead = ('受 DeepInfer 启发，由 MLX 驱动。让小模型推测、大模型验证，精确复用上下文，并用真实结果判断是否值得。' if zh else 'DeepInfer-inspired local inference, powered by MLX. Explore speculative decoding and exact prefix reuse. Measure what actually helps.')
    return f'''<section class="hero"><div><div class="eyebrow">LOCAL INFERENCE / OPEN SOURCE / MLX</div><h1>{headline}</h1><p class="lead">{lead}</p><div class="actions"><a class="button primary" href="#install">{'本地运行' if zh else 'Run it locally'} ↓</a><a class="button" href="benchmarks.html">{'查看实测' if zh else 'Read the evidence'} ↗</a></div><p class="muted" style="font-size:11px;margin-top:20px">Apple Silicon · Greedy decoding · Experimental alpha</p></div><div class="terminal" aria-label="Real installation commands, not simulated inference"><div class="terminalbar"><span>QUICK START / YOUR TERMINAL</span><span>● LOCAL</span></div><pre><span class="dim"># Install from the repository</span>
<span class="green">$</span> uv sync --frozen --extra mlx
<span class="green">$</span> uv run deepinfer-mac doctor
<span class="green">$</span> uv run deepinfer-mac models \\
    --download qwen3-4b
<span class="green">$</span> uv run deepinfer-mac serve

<span class="orange">→ http://127.0.0.1:8765</span>
<span class="dim">Chat · Compare · Export real JSON</span></pre><div class="diagram"><b>draft</b><span>→ propose →</span><b>target</b><span>→</span><b>verify</b></div></div></section><div class="notice">Priority showcase candidate / 主展示候选: Qwen3.8-27B 4-bit · hardware validation pending, not the default. Baseline/exact cache only. Smaller Qwen3 speculative results are separate. / 待实机验证，未设默认；不沿用旧草稿。 <a href="qwen27.html">27B resource evidence / 资源与低比特候选 ↗</a></div><div class="strap"><span><b>01</b> Real MLX execution</span><span><b>02</b> Exact prefix cache</span><span><b>03</b> Honest benchmarks</span><span><b>04</b> Apache-2.0</span></div><section class="section"><div class="sectionhead"><div><span class="index">01 / THE EXPERIMENT</span><h2>{'为可复现而构建。' if zh else 'Built to be inspected.'}</h2></div><p class="muted">{'不是模型越大越好，也不是草稿越长越快。首版围绕一台 Mac，验证推测收益、缓存复用与可靠服务。' if zh else 'A small, inspectable engine for one Mac. Keep the baseline honest, make the draft budget observable, and reuse work without changing the prompt.'}</p></div><div class="rows"><div class="row"><span class="index">A / DECODE</span><h3>Three paths. One comparison.</h3><p>Ordinary, fixed speculative and adaptive greedy decoding. Native MLX-LM stays in the benchmark as a reference.</p></div><div class="row"><span class="index">B / REUSE</span><h3>Keep the exact context.</h3><p>Identity-isolated prompt snapshots, cloned before use. A byte-budget LRU evicts cached state, never silently truncates your input.</p></div><div class="row"><span class="index">C / OBSERVE</span><h3>See every outcome.</h3><p>Timings, accepted drafts, memory, fallbacks and complete token mismatches. Slowdowns and failures remain visible.</p></div></div><div class="notice">Independent community experiment. Official DSpark, specialized confidence heads and full DeepSeek V4.1 are outside this release. Adaptive decoding may be slower.</div></section><section class="section" id="results"><div class="sectionhead"><div><span class="index">02 / MEASURED, NOT ASSUMED</span><h2>{'证据先于倍数。' if zh else 'Evidence before a multiplier.'}</h2></div><p class="muted">{'仅展示真实模型与完整报告。数字受模型、机器和工作负载约束；没有数据时不填占位成绩。' if zh else 'Only real model measurements, with their full context. Every number must lead back to raw data, a locked revision and an explicit limitation.'}</p></div>{evidence(report)}<p><a class="textlink" href="benchmarks.html">Measurement method / 测量口径 ↗</a></p></section>{cache_evidence()}{recorded_demo()}<section class="section" id="install"><div class="sectionhead"><div><span class="index">03 / TRY IT ON YOUR MAC</span><h2>{'从 doctor 开始。' if zh else 'Start with a diagnosis.'}</h2></div><p class="muted">Python 3.12 / 3.13 and Apple Silicon Metal. The web UI is bundled with the wheel; no Node runtime needed. Weights download separately.</p></div><div class="install"><pre class="code">git clone https://github.com/ZEVIN17/deepinfer-mac.git
cd deepinfer-mac
uv sync --frozen --extra mlx
uv run deepinfer-mac doctor
uv run deepinfer-mac models --download qwen3-4b
uv run deepinfer-mac run --mode baseline 'Hello'
uv run deepinfer-mac serve</pre><div><div class="step"><b>01</b><div><strong>Check the machine</strong><p>Doctor reports Python, Metal, disk and memory. Keep room for the OS, weights and exact KV.</p></div></div><div class="step"><b>02</b><div><strong>Generate locally</strong><p>Open 127.0.0.1:8765 to chat and compare. This public page does not send prompts or connect to your localhost.</p></div></div><div class="step"><b>03</b><div><strong>Contribute a complete report</strong><p>Run the fixed corpus and publish the whole result, including failures. Help extend the hardware matrix.</p></div></div></div></div><a class="textlink" href="docs/{'README.zh-CN.md' if zh else 'README.md'}">{'中文安装指南' if zh else 'Full installation guide'} ↗</a></section><section class="section"><span class="index">04 / OPEN RESEARCH, PRACTICAL SOFTWARE</span><h2>Make the next result reproducible.</h2><p class="muted">Inspect the <a href="architecture.html">architecture</a>, check the <a href="compatibility.html">support boundaries</a>, or contribute a <a href="contribute.html">hardware report</a>. Improvements follow measured bottlenecks.</p></section>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="Reviewed real benchmark summary.json")
    args = parser.parse_args()
    source = args.report or SITE / "data" / "benchmark.json"
    if args.report is not None and not source.exists():
        raise FileNotFoundError(source)
    report = json.loads(source.read_text()) if source.exists() else None
    if report is not None and (report.get("schema_version") != 2 or not report.get("samples") or not report.get("environment")):
        raise ValueError("Expected a real schema-2 benchmark report with environment and full samples")
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "assets").mkdir(parents=True)
    shutil.copytree(SITE / "assets", DIST / "assets", dirs_exist_ok=True)
    (DIST / "docs").mkdir()
    for path in [*ROOT.glob("README*.md"), ROOT / "LICENSE", ROOT / "NOTICE", ROOT / "CONTRIBUTING.md", ROOT / "models.lock.json", ROOT / "upstream.lock.json", *(p for p in (ROOT / "docs").glob("*.md") if not p.name.startswith("CODEX_") and p.name != "PROGRESS.md")]:
        if path.exists():
            shutil.copy2(path, DIST / "docs" / path.name)
    for path in (ROOT / "docs").rglob("*.json"):
        target = DIST / "docs" / path.relative_to(ROOT / "docs")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (DIST / ".nojekyll").touch()
    if (ROOT / "docs/evidence/local-inference-qwen3-4b.png").exists():
        shutil.copy2(ROOT / "docs/evidence/local-inference-qwen3-4b.png", DIST / "assets/local-inference.png")
        (DIST / "data").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "benchmarks/reports/local-ui-demo-qwen3-4b.json", DIST / "data/local-ui-demo.json")
    for zh, name in [(False, "index.html"), (True, "zh.html")]:
        (DIST / name).write_text(shell("Local inference, honestly measured", homepage(report, zh), "zh-CN" if zh else "en"))
    for name, doc in [("architecture", "docs/architecture.md"), ("compatibility", "docs/compatibility.md"), ("contribute", "CONTRIBUTING.md"), ("qwen27", "docs/qwen27.md")]:
        body = markdown((ROOT / doc).read_text())
        body = document_links(body)
        (DIST / f"{name}.html").write_text(shell(name.title(), '<article class="article">' + body + '</article>'))
    benchmarks = '<article class="article"><div class="eyebrow">REAL HARDWARE / COMPLETE RESULTS</div><h1>Benchmark evidence</h1>' + evidence(report) + cache_evidence() + document_links(markdown((ROOT / "docs/benchmarking.md").read_text())) + '</article>'
    (DIST / "benchmarks.html").write_text(shell("Benchmark evidence", benchmarks))
    if report is not None:
        (DIST / "data").mkdir(exist_ok=True)
        (DIST / "data" / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        for name in ("source.tar.gz", "source-manifest.json", "cache-qwen3-4b.json"):
            artifact = source.parent / name
            if artifact.exists():
                shutil.copy2(artifact, DIST / "data" / name)
    print(f"Built {DIST}: 7 pages; {'real report included' if report else 'no performance data published'}")

if __name__ == "__main__":
    main()
