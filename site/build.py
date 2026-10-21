"""Build the architecture-led, relocatable GitHub Pages showcase."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
DIST = SITE / "dist"
REPO = "https://github.com/ZEVIN17/deepinfer-mac"
BRANCH = REPO + "/tree/codex/phase2-core"
SOURCE_COMMIT = "66ae444f535c63b43003ba6febeb42228d4d607a"
ADAPTER_SOURCE_COMMIT = "43d967652683817daf7eff0b7553ae5d5293739e"
JOINT_SOURCE_COMMIT = "d54f09ec709b3cb65513057e0d6f1eb54e13a1ae"
JOINT_SOURCE = REPO + "/blob/" + JOINT_SOURCE_COMMIT
ADAPTER_SOURCE = REPO + "/blob/" + ADAPTER_SOURCE_COMMIT
SOURCE_TREE = "bfe1c16058038bfb2867fcadf8e59f2f47b39150"
BUNDLE = SITE / "data/phase2"
ROOT_DOCUMENTS = {"README.md", "README.zh-CN.md", "LICENSE", "NOTICE", "CONTRIBUTING.md",
                  "models.lock.json", "upstream.lock.json"}
REPORTS = (
    "phase2-ui-qwen35-9b8.json",
    "phase2-service-qwen3.5-9b-8bit.json",
    "phase2-service-sampling-qwen35-9b8.json",
    "phase2-message-checkpoints-qwen35-9b8.json",
    "phase2-hybrid-lagged-dev-qwen35-9b8.json",
    "phase2-hybrid-lagged-dev-qwen35-9b8-supplemental-verification.json",
    "phase2-experimental-api-qwen35-9b8.json",
    "context-4204-256-qwen35-9b8.json",
)


def load_report(name):
    return json.loads((BUNDLE / "benchmarks/reports" / name).read_text())


def source_link(name, label="Source JSON ↗"):
    return f'<a class="textlink" href="benchmarks/reports/{name}">{label}</a>'


def pinned_document_links(markup):
    """Link research docs to their actual published Phase2 revision."""
    def replace(match):
        name = match[1]
        path = name if name in ROOT_DOCUMENTS else "docs/" + name
        return f'href="{REPO}/blob/{SOURCE_COMMIT}/{path}"'
    return re.sub(r'href="docs/([^"#]+)"', replace, markup)


def verify_bundle():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    if (manifest["source_commit"] != SOURCE_COMMIT or manifest["source_repository"] != REPO
            or manifest["source_tree"] != SOURCE_TREE):
        raise ValueError("Showcase bundle source identity differs from the pinned published revision")
    expected = {"benchmarks/reports/" + name for name in REPORTS}
    expected.add("docs/evidence/phase2-local-ui-qwen35-9b8.png")
    actual = {str(path.relative_to(BUNDLE)) for path in BUNDLE.rglob("*") if path.is_file()}
    if set(manifest["files"]) != expected or actual != expected | {"manifest.json"}:
        raise ValueError("Unexpected or missing member in the curated showcase bundle")
    for name, details in manifest["files"].items():
        data = (BUNDLE / name).read_bytes()
        if details["source_path"] != name or len(data) != details["bytes"]:
            raise ValueError(f"Showcase bundle length/path mismatch: {name}")
        if hashlib.sha256(data).hexdigest() != details["sha256"]:
            raise ValueError(f"Showcase bundle SHA-256 mismatch: {name}")
    return manifest


def shell(title, body, zh=False):
    nav = [("architecture.html", "架构" if zh else "Architecture"),
           ("benchmarks.html", "已实现" if zh else "Milestones"),
           ("phase2.html", "演示" if zh else "Demo"),
           ("qwen27.html", "技术文档" if zh else "Docs")]
    return pinned_document_links(f'''<!doctype html>
<html lang="{'zh-CN' if zh else 'en'}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="DeepInfer-Mac: adaptive inference for Apple Silicon. Local models, an MLX execution backend and inspectable DeepInfer-inspired scheduling."><meta name="theme-color" content="#f5f4ee"><title>{html.escape(title)} · DeepInfer-Mac</title><link rel="stylesheet" href="assets/style.css"><link rel="icon" href="assets/icon.svg" type="image/svg+xml"><script src="assets/showcase.js" defer></script></head>
<body><a class="skip" href="#main">{'跳至正文' if zh else 'Skip to content'}</a><header><nav class="wrap" aria-label="{'主导航' if zh else 'Main navigation'}"><a class="brand" href="{'zh.html' if zh else 'index.html'}">deepinfer<i>/</i>mac<span class="brand-dot"></span></a><div class="navlinks">{''.join(f'<a href="{url}">{label}</a>' for url, label in nav)}<a class="language" href="{'index.html' if zh else 'zh.html'}">{'EN' if zh else '中文'}</a><a class="github" href="{BRANCH}">GitHub ↗</a></div></nav></header>
<main id="main" class="wrap">{body}</main><footer class="wrap"><div><a class="brand" href="index.html">deepinfer<i>/</i>mac</a><p>Independent community project · Powered by MLX<br>Apache-2.0 · Apple Silicon · Experimental 0.1.0a2</p></div><div class="footlinks"><a href="qwen27.html">Technical docs</a><a href="contribute.html">Contribute</a><a href="docs/LICENSE">License</a><a href="docs/NOTICE">Attribution</a></div></footer></body></html>''')


def architecture_visual(zh=False):
    # The diagram describes implemented paths; it is not an animated execution trace.
    t = lambda en, cn: cn if zh else en
    return f'''<div class="architecture-visual" aria-label="{t('Implemented request and adaptive decoding architecture', '已实现的请求与自适应解码架构')}"><div class="visual-top"><span>ENGINE / APPLE SILICON</span><span class="status"><i></i> MLX + METAL</span></div><div class="interface-line"><span>CLI</span><span>OpenAI API</span><span>Local UI</span></div><div class="vertical-arrow">↓</div><div class="visual-cache"><span class="node-index">01</span><div><strong>{t('Exact context', '精确上下文')}</strong><small>{t('Bounded cache · isolated snapshots', '限额缓存 · 独立状态快照')}</small></div><span class="node-icon">⌘</span></div><div class="vertical-arrow">↓</div><div class="decode-loop"><div class="loop-title">{t('ADAPTIVE DECODE LOOP', '自适应解码循环')}<span>EXPERIMENTAL</span></div><div class="loop-nodes"><div><b>{t('Draft', '草稿')}</b><small>propose</small></div><span>→</span><div class="target-node"><b>{t('Target', '目标模型')}</b><small>verify</small></div><span>→</span><div><b>{t('Recover', '精确恢复')}</b><small>replay</small></div></div><div class="feedback"><span>↳ {t('measured cycle cost + acceptance', '实测周期成本 + 接受率')}</span><strong>{t('next budget', '下一轮预算')} ↺</strong></div></div><div class="vertical-arrow">↓</div><div class="visual-output"><span>{t('Text stream', '文本流')}</span><span>+ {t('observable decisions', '可核查决策')}</span></div><div class="visual-bottom">LOCAL EXECUTION. INSPECTABLE BY DESIGN.</div></div>'''


def milestones(zh=False):
    t = lambda en, cn: cn if zh else en
    return f'''<div class="milestones"><a href="benchmarks.html#scheduler"><strong>28<span>/28</span></strong><span>{t('Scheduler decisions matched', '调度决策逐次一致')}</span><small>{t('Original C++ reference · real 9B trace', '原始 C++ 参考 · 真实 9B 轨迹')}</small></a><a href="benchmarks.html#cache"><strong>16<span>/16</span></strong><span>{t('Cache token checks passed', '缓存完整 token 检查通过')}</span><small>{t('20 real requests · two sampler settings', '20 次真实请求 · 两种采样设置')}</small></a><a href="benchmarks.html#context"><strong>4,204<span> + 256</span></strong><span>{t('Context + generation completed', '输入 + 输出完整运行')}</span><small>{t('Qwen3.5-9B-8bit · M4 Pro / 24 GiB', 'Qwen3.5-9B-8bit · M4 Pro / 24 GiB')}</small></a></div>'''


def capabilities(zh=False):
    t = lambda en, cn: cn if zh else en
    rows = [
        ("01 / ADAPT", t("Make the budget a decision.", "让推测预算，随执行而变。"), t("A DeepInfer MTPVarlen-derived scheduler uses observed acceptance and complete-cycle cost to choose the next draft budget. Decisions can be replayed against the original C++ reference.", "将 DeepInfer MTPVarlen 方法接入真实执行，以接受率和完整周期成本选择下一轮草稿预算；每次决策都可由原始 C++ 参考重放核查。")),
        ("02 / RECOVER", t("Preserve the exact state.", "把混合状态，精确接续。"), t("Hybrid attention and recurrent state travel together. Checkpoint, restore and replay the consumed prefix when a draft is rejected, keeping mutable branches isolated.", "同时管理注意力与循环状态。草稿被拒绝时，通过快照、恢复和已消费前缀重放继续执行；可变分支保持独立。")),
        ("03 / REUSE", t("Carry context forward.", "让已有上下文，继续发挥作用。"), t("Native MLX-LM LRU reuse and exact hybrid snapshots share a bounded cache interface. Verified message boundaries create reusable checkpoints for repeated and changed-tail requests.", "原生 MLX-LM LRU 与混合状态精确快照，共用限额缓存接口；经 token 核验的消息边界，为重复请求与修改末尾的请求建立复用点。")),
    ]
    return '<div class="capabilities">' + ''.join(f'<div class="capability"><span class="index">{index}</span><h3>{title}</h3><p>{text}</p></div>' for index, title, text in rows) + '</div>'


def quickstart(zh=False):
    t = lambda en, cn: cn if zh else en
    command = '''git clone --branch codex/phase2-core --single-branch https://github.com/ZEVIN17/deepinfer-mac.git
cd deepinfer-mac
uv sync --frozen --extra mlx
uv run deepinfer-mac doctor
uv run deepinfer-mac models --download qwen3.5-9b-8bit --target-only
uv run deepinfer-mac serve'''
    return f'''<section class="section" id="install"><div class="sectionhead"><div><span class="index">GET STARTED</span><h2>{t('Your next inference runs here.', '下一次推理，就在你的 Mac。')}</h2></div><p>{t('Apple Silicon · Python 3.12 / 3.13 · uv. Start the bundled local UI and API with the locked 9B model.', 'Apple Silicon · Python 3.12 / 3.13 · uv。用锁定版本的 9B 模型，启动内置本地 UI 与 API。')}</p></div><div class="install"><div class="code-panel"><div class="code-head"><span>TERMINAL / EXPERIMENTAL BRANCH</span><button class="copy-button" data-copy="install-command">{t('Copy', '复制')}</button></div><pre id="install-command"><code>{html.escape(command)}</code></pre><span class="copy-status" role="status"></span></div><div class="steps"><div class="step"><b>01</b><div><strong>{t('Check your Mac', '检查本机环境')}</strong><p>{t('Doctor reports Metal, Python, available memory and disk.', 'Doctor 检查 Metal、Python、可用内存与磁盘。')}</p></div></div><div class="step"><b>02</b><div><strong>{t('Open the local workspace', '打开本地工作台')}</strong><p>{t('Visit http://127.0.0.1:8765 and select qwen3.5-9b-8bit. The default path is baseline generation.', '访问 http://127.0.0.1:8765，选择 qwen3.5-9b-8bit；默认使用 baseline 生成路径。')}</p></div></div><div class="step"><b>03</b><div><strong>{t('Build on the same engine', '在同一个引擎上开发')}</strong><p>{t('Use CLI, streamed API or the bundled browser UI. Explore the explicit adaptive entry in the architecture guide.', '使用 CLI、流式 API 或内置浏览器 UI；自适应实验入口见架构指南。')}</p></div></div><a class="textlink" href="compatibility.html">{t('Run guide & API example', '运行指南与 API 示例')} ↗</a></div></div></section>'''



def model_capabilities(zh=False):
    t = lambda en, cn: cn if zh else en
    rows = [
        ("Qwen3.5-9B-8bit", t("Verified CLI / API / UI", "已验证 CLI / API / UI"), t("Hybrid · explicit experimental adaptive", "Hybrid · 显式实验自适应")),
        ("Qwen3.5-9B-4bit", t("Verified baseline / native controls", "已验证 baseline / native 控制"), t("Hybrid · quality / memory control", "Hybrid · 质量 / 内存对照")),
        ("Qwen3-14B-4bit", t("Verified baseline / native controls", "已验证 baseline / native 控制"), t("Dense · engineering control", "Dense · 工程对照")),
        ("Qwen3-4B", t("Verified regression path", "已验证回归路径"), t("Dense · experimental fixed / adaptive", "Dense · 实验 fixed / adaptive")),
        ("SmolLM2-135M-Instruct-8bit", t("Verified baseline / cache / CLI / API / SSE", "已验证 baseline / 缓存 / CLI / API / SSE"), t("Dense · configurable model adapter", "Dense · 可配置模型适配器")),
        (t("Other custom models", "其它自定义模型"), t("Per-model acceptance required", "按模型独立验收"), t("Capability-gated model path / repo integration", "按能力接入模型路径 / repo")),
        (t("Experimental joint service", "实验联合执行服务"), t("35/35 real-model service checks", "35/35 项真实模型服务检查"), t("360M target + 135M draft · dense research pair", "360M 目标 + 135M 草稿 · Dense 研发模型对")),
    ]
    body = ''.join(f'<tr><th scope="row">{name}</th><td>{status}</td><td>{scope}</td></tr>' for name, status, scope in rows)
    return f'<section class="section" id="models"><div class="sectionhead"><div><span class="index">MODEL CAPABILITIES / 模型能力</span><h2>{t("One framework.<br>Explicit capabilities.", "一个框架，<br>清晰的能力边界。")}</h2></div><p>{t("The 9B model is our primary demonstration. Baseline compatibility and optimized execution are validated separately, across dense and hybrid state structures.", "9B 是首要演示载体。Baseline 兼容性与优化执行分别验证，按 Dense 和 Hybrid 状态结构区分能力。")}</p></div><div class="table-scroll"><table class="capability-table"><thead><tr><th>{t("Model / path", "模型 / 路径")}</th><th>{t("Status", "状态")}</th><th>{t("Scope", "范围")}</th></tr></thead><tbody>{body}</tbody></table></div><p class="measurement">{t("Verified describes the listed path. Public 9B UI/API use baseline; hybrid adaptive decoding is explicit experimental opt-in. Bounded multi-request execution and joint verification have separate explicit opt-in entries. The small-model joint path validates service behavior.", "已验证仅指表中具体路径。公开 9B UI/API 使用 baseline；Hybrid 自适应需显式开启实验入口。有界多请求执行与联合验证分别提供显式实验入口；小模型联合路径已验证服务行为。")}</p><a class="textlink" href="docs/compatibility.md">{t("Compatibility and execution modes", "兼容性与执行模式")} ↗</a></section>'



def adapter_milestone(zh=False):
    t = lambda en, cn: cn if zh else en
    return f'''<section class="section milestone-detail" id="model-adapter"><span class="index">MODEL ADAPTER / 模型可替换</span><div><h2>{t('A new model family.<br>The same working interface.', '接入新的模型家族，<br>复用同一执行接口。')}</h2><p>{t('SmolLM2-135M-Instruct-8bit has completed real baseline/native token matching for English and Chinese inputs, dense cache branch reuse, CLI generation, API and SSE checks. Registration, explicit fallback, cancellation and recovery were exercised with the actual model.', 'SmolLM2-135M-Instruct-8bit 已完成中英文输入的真实 baseline/native token 一致性、Dense 缓存分支复用、CLI 生成、API 与 SSE 验证，并以真实模型核验注册、显式回退、取消与恢复。')}</p><p class="measurement">{t('Scope: baseline integration of this 135M dense model. Qwen3.5-9B-8bit remains the primary demonstration. These checks establish integration correctness.', '范围：该 135M Dense 模型的 baseline 接入正确性。Qwen3.5-9B-8bit 仍是首要演示载体。')}</p><div class="actions"><a class="textlink" href="{ADAPTER_SOURCE}/benchmarks/reports/phase3-model-adapter-integration.json">{t('Real-model acceptance', '真实模型验收')} ↗</a><a class="textlink" href="{ADAPTER_SOURCE}/benchmarks/reports/phase3-smollm2-weight-audit.json">{t('Weight identity', '权重身份核验')} ↗</a><a class="textlink" href="{ADAPTER_SOURCE}/docs/MODEL_ADAPTERS.md">{t('Adapter guide', '适配器指南')} ↗</a></div></div></section>'''

def joint_milestone(zh=False):
    t = lambda en, cn: cn if zh else en
    return f'''<section class="section milestone-detail" id="joint-service"><span class="index">MULTI-REQUEST / 多请求执行</span><div><h2>{t('Many requests.<br>One accountable lifecycle.', '多个请求，<br>清晰可控的执行生命周期。')}</h2><p>{t('Bounded admission, per-request state and cancellation recovery now connect to real joint target verification. The experimental service passed 35/35 functional checks with distinct SmolLM2 360M target and 135M draft models, including concurrent requests, actual token SSE and recovery after cancellation.', '有界准入、逐请求状态与取消恢复已接入真实联合目标验证。实验服务使用独立的 SmolLM2 360M 目标模型与 135M 草稿模型，通过 35/35 项功能检查，覆盖并发请求、真实 token SSE 与取消后恢复。')}</p><p class="measurement">{t('330 portable tests + 48 real Metal tests passed at this checkpoint. The small models are engineering controls; Qwen3.5-9B remains the primary demonstration. Default service uses baseline; joint execution is explicit experimental opt-in.', '本次检查点通过 330 项 CPU 测试与 48 项真实 Metal 测试。小模型用于工程验证；Qwen3.5-9B 仍是首要演示载体。默认服务使用 baseline，联合执行需显式开启实验入口。')}</p><div class="actions"><a class="textlink" href="{JOINT_SOURCE}/benchmarks/reports/phase3-joint-service.json">{t('35-check service report', '35 项服务验收报告')} ↗</a><a class="textlink" href="{JOINT_SOURCE}/docs/evidence/phase3-joint-service-verification.json">{t('Independent verification', '独立验证证据')} ↗</a><a class="textlink" href="{JOINT_SOURCE}/docs/JOINT_VERIFICATION.md">{t('Execution guide & research scope', '执行指南与研究范围')} ↗</a></div></div></section>'''


def budget_illustration():
    return '''<section class="section" id="joint-budget"><div class="sectionhead"><div><span class="index">MECHANISM ILLUSTRATION / 机制示意 · PHASE 3</span><h2>A shared budget.<br>Different request depths.</h2></div><p>Explore the allocation mechanism: cumulative prefix survival ranks candidate verification positions across requests. The numbers below are illustrative inputs, not model measurements or benchmark results.</p></div><div class="budget-demo"><div class="budget-control"><label for="budget-q">Candidate position budget / 候选位置预算</label><input id="budget-q" type="range" min="0" max="9" value="5"><output id="budget-value" for="budget-q">5 / 9</output></div><div id="budget-rows" aria-label="Illustrative request allocation"></div><p id="budget-summary" role="status" class="measurement"></p><p class="muted">Fixed example conditional acceptance inputs: A [0.95, 0.90, 0.85], B [0.80, 0.60, 0.40], C [0.60, 0.50, 0.30]. Each position's score is the cumulative product. Equal scores use request order then position order. This view illustrates allocation only; cost-based total-budget selection, real verification and recovery remain separate responsibilities.</p></div><p class="measurement">This browser-only mechanism illustration executes no model and makes no latency claim. The separate joint-service report records actual model execution, streaming and cancellation checks.</p></section>'''


def homepage(zh=False):
    t = lambda en, cn: cn if zh else en
    return f'''<section class="hero"><div class="hero-copy"><div class="eyebrow"><span class="tiny-square"></span> LOCAL INFERENCE / OPEN SOURCE / MLX</div><h1>{t('Adaptive inference.<br><span>For Apple Silicon.</span>', '让本地模型，<br><span>更懂你的 Mac。</span>')}</h1><p class="lead">{t('A local inference framework with MLX at its core. Run through CLI, Python or streamed chat API. Explore DeepInfer-inspired scheduling with a clear view of every execution cycle.', '以 MLX 为执行后端的本地推理框架。连接 CLI、Python 与流式聊天 API，以 DeepInfer 方法探索调度，让每个执行周期都有据可循。')}</p><div class="actions"><a class="button primary" href="#install">{t('Run on your Mac', '在 Mac 上运行')} ↗</a><a class="button" href="phase2.html">{t('Explore the demo', '查看真实演示')} →</a></div><p class="hero-meta">{t('MLX BACKEND', 'MLX 执行后端')} <span>·</span> CLI / API / UI <span>·</span> Apache-2.0</p></div>{architecture_visual(zh)}</section><div class="strap"><span>{t('Native MLX execution', '原生 MLX 执行')}</span><span>{t('Cost-aware draft scheduling', '成本驱动草稿调度')}</span><span>{t('Exact hybrid-state recovery', '混合状态精确恢复')}</span><span>{t('Bounded context reuse', '限额上下文复用')}</span></div><section class="section"><div class="sectionhead"><div><span class="index">01 / ENGINEERING THE EXECUTION</span><h2>{t('Local workflows.<br>Inspectable execution.', '本地工作流，<br>可核查的执行方式。')}</h2></div><p>{t('Models are execution objects, not the framework definition. Start with the verified 9B demonstration, reuse context and explore an adaptive cycle that accounts for drafting, verification and recovery.', '模型是执行对象，不是框架的定义。从已验证的 9B 演示开始，复用上下文，并探索同时计入草稿、验证与恢复成本的自适应周期。')}</p></div>{capabilities(zh)}<a class="textlink" href="architecture.html">{t('Inside the architecture', '深入了解架构')} ↗</a></section><section class="section"><div class="sectionhead"><div><span class="index">02 / BUILT AND VERIFIED</span><h2>{t('From design to running code.', '从架构设计，到真实运行。')}</h2></div><p>{t('Real Qwen3.5-9B-8bit execution on an M4 Pro. These milestones validate scheduler correspondence, token preservation and a practical local context workload.', '在 M4 Pro 上运行真实 Qwen3.5-9B-8bit。已完成调度方法对应、完整 token 保持与本地上下文工作负载验证。')}</p></div>{milestones(zh)}<div class="section-tail"><span>{t('Measured 2026-10-21 · source-linked engineering checks', '实测日期 2026-10-21 · 每项成绩均关联来源')}</span><a class="textlink" href="benchmarks.html">{t('Explore the milestones', '查看已实现成绩')} ↗</a></div></section><section class="section demo-teaser"><div><span class="index">03 / A WORKING LOCAL EXPERIENCE</span><h2>{t('Write. Stream. Inspect.', '输入、生成、洞察。')}</h2><p>{t('A single engine connects the terminal, streamed chat API and local browser workspace. Watch a recorded 9B response and inspect the real request behind it.', '同一引擎连接终端、流式聊天 API 与本地浏览器工作台。查看真实 9B 响应回放，并追溯对应请求。')}</p><a class="button" href="phase2.html">{t('Watch recorded demo', '查看录制演示')} →</a></div><a class="capture-link" href="phase2.html"><img class="capture" src="assets/phase2-local-ui-qwen35-9b8.png" alt="{t('Actual local Qwen3.5-9B UI repairing a Python clamp function', '本地 Qwen3.5-9B UI 修复 Python clamp 函数的真实截图')}" loading="lazy"></a></section>{joint_milestone(zh)}{quickstart(zh)}{adapter_milestone(zh)}{model_capabilities(zh)}<section class="closing"><span class="index">OPEN SOURCE / OPEN TO INSPECTION</span><h2>{t('A small engine.<br>A clear line of sight.', '轻量的引擎，<br>清晰的执行链路。')}</h2><p>{t('Trace the method to its source, inspect the implementation and make the next contribution.', '追溯方法来源，阅读实现，参与下一次改进。')}</p><div class="actions"><a class="button primary" href="{BRANCH}">GitHub ↗</a><a class="button" href="qwen27.html">{t('Technical documentation', '技术文档')} ↗</a></div></section>'''


def architecture_page():
    return f'''<section class="page-heading"><span class="index">ARCHITECTURE / 架构</span><h1>One execution cycle.<br><span>Every part accounted for.</span></h1><p class="lead">DeepInfer-derived scheduling meets native MLX execution and exact hybrid-state recovery. 从请求入口到下一轮预算，保持清晰的执行边界。</p></section><section class="architecture-wide">{architecture_visual()}<div><span class="index">THE REQUEST PATH</span><h2>One engine, three entrances.</h2><p>CLI, local UI and a supported OpenAI-compatible chat API share a strict request schema and a bounded FIFO worker. A single inference owner executes MLX on Metal and returns text events with request metrics.</p><p>Baseline generation uses MLX-LM sampling. The explicit experimental path adds a locked draft model, cost-aware budget selection and exact recovery for Qwen3.5 hybrid state.</p><a class="textlink" href="docs/architecture.md">Execution layers and source attribution ↗</a></div></section><section class="section"><div class="sectionhead"><div><span class="index">EXPLORE THE CYCLE / 探索执行周期</span><h2>Choose. Verify. Recover.</h2></div><p>Select a stage to explore its implemented responsibility. This diagram explains the architecture; the measured trace is linked in the milestones.</p></div><div class="architecture-explorer"><div class="stage-tabs" role="tablist" aria-label="Execution stages"><button id="tab-budget" role="tab" aria-selected="true" aria-controls="panel-budget" data-stage="budget">01 Budget</button><button id="tab-verify" role="tab" aria-selected="false" aria-controls="panel-verify" tabindex="-1" data-stage="verify">02 Verify</button><button id="tab-recover" role="tab" aria-selected="false" aria-controls="panel-recover" tabindex="-1" data-stage="recover">03 Recover</button><button id="tab-reuse" role="tab" aria-selected="false" aria-controls="panel-reuse" tabindex="-1" data-stage="reuse">04 Reuse</button></div><div id="panel-budget" class="stage-panel" role="tabpanel" aria-labelledby="tab-budget"><span class="index">ACCEPTANCE × COMPLETE-CYCLE COST</span><h3>Choose a budget from observed work.</h3><p>The MTPVarlen-derived core selects an expected-yield / cost budget. A lagged mean recovery adapter feeds measured complete-cycle costs, keeping scheduling history distinct from partial, interrupted rounds.</p><code>observed cycle → cost curve → next draft budget</code><a class="textlink" href="docs/DS_DESIGN_MAPPING.md">Core method and adaptation ↗</a></div><div id="panel-verify" class="stage-panel" role="tabpanel" aria-labelledby="tab-verify" hidden><span class="index">DRAFT / TARGET</span><h3>Keep the target in control.</h3><p>The locked 0.8B draft proposes tokens. The 9B target verifies the proposal in the greedy experimental path; accepted tokens and the target correction determine the continuation.</p><code>draft proposal → target verification → accepted prefix</code><a class="textlink" href="benchmarks.html#scheduler">128-token exact-output control ↗</a></div><div id="panel-recover" class="stage-panel" role="tabpanel" aria-labelledby="tab-recover" hidden><span class="index">SNAPSHOT / RESTORE / REPLAY</span><h3>Recover the state actually consumed.</h3><p>Qwen3.5 combines attention KV with recurrent ArraysCache state. The adapter restores an exact checkpoint and replays the consumed prefix after rejection, EOS or close, rather than assuming every state supports trimming.</p><code>isolated checkpoint → restore → consumed-prefix replay</code><a class="textlink" href="docs/DS_DESIGN_MAPPING.md">State invariants and controls ↗</a></div><div id="panel-reuse" class="stage-panel" role="tabpanel" aria-labelledby="tab-reuse" hidden><span class="index">EXACT IDENTITY / BYTE BUDGET</span><h3>Reuse a verified prefix.</h3><p>Model revision, template, quantization, thinking mode and draft identity define cache identity. Byte-budgeted snapshots preserve branch isolation. Optional message checkpoints use verified token-prefix boundaries.</p><code>exact prefix → bounded snapshot → remaining suffix</code><a class="textlink" href="benchmarks.html#cache">Real cache controls ↗</a></div></div></section><section class="section">{capabilities()}<p class="muted">MLX / MLX-LM supply model execution, sampling and native LRU reuse. The scheduler reimplements pinned DeepInfer methods; hybrid recovery, service integration and measurement are Mac adaptations. Full attribution lives in the source mapping and NOTICE.</p></section>{joint_milestone()}{budget_illustration()}<section class="closing compact"><h2>Follow the implementation.</h2><div class="actions"><a class="button primary" href="{BRANCH}/src/deepinfer_mac">Browse the engine ↗</a><a class="button" href="docs/DS_DESIGN_MAPPING.md">Method mapping ↗</a></div></section>'''


def milestones_page():
    return f'''<section class="page-heading"><span class="index">ENGINEERING MILESTONES / 已取得成绩</span><h1>Designed on paper.<br><span>Verified on silicon.</span></h1><p class="lead">Qwen3.5-9B-8bit · Apple M4 Pro · 24 GiB · 2026-10-21. Source-linked execution checks, with the workload attached to every result.</p></section>{milestones()}<section class="section milestone-detail" id="scheduler"><span class="index">01 / COST-AWARE SCHEDULING</span><div><h2>28 decisions.<br>28 original-reference matches.</h2><p>A real hybrid decoding trace produced 58 cycles and 28 mature scheduler decisions. Independent chronological reconstruction and the unchanged original C++ MTPVarlen reference matched all 28. The candidate and baseline generated identical complete sequences of 128 tokens on this development request.</p><p class="measurement">Scope: one real 9B development trace, with a separate CPU supplemental verification. This milestone measures scheduler correspondence and output identity.</p>{source_link('phase2-hybrid-lagged-dev-qwen35-9b8-supplemental-verification.json', 'Supplemental verification JSON ↗')} · {source_link('phase2-hybrid-lagged-dev-qwen35-9b8.json', 'Immutable execution trace ↗')}</div></section><section class="section milestone-detail" id="cache"><span class="index">02 / EXACT CONTEXT REUSE</span><div><h2>20 requests.<br>16 complete-token checks.</h2><p>Repeated prompts and changed-tail branches passed all 16 full-output token equality checks across 20 real requests. The control spans greedy and recommended non-thinking sampling, cache on/off, and optional message checkpoints.</p><p>For the changed-tail request in both sampler settings, the verified reused prefix increased from 512 to 2,014 tokens with message checkpoints.</p><p class="measurement">Scope: one 2K development session, three request shapes and two sampler settings. The token count describes reusable prefix coverage.</p>{source_link('phase2-message-checkpoints-qwen35-9b8.json')}</div></section><section class="section milestone-detail" id="context"><span class="index">03 / PRACTICAL LOCAL CONTEXT</span><div><h2>4,204 input tokens.<br>256 generated tokens.</h2><p>The primary 9B8bit model completed the full configured input and output in a single baseline resource check. Streamed text matched the final result. The request-scoped Metal peak was 10.65 GB; the measured system swap increase from loading through completion was zero.</p><p class="measurement">Scope: one baseline request on this M4 Pro / 24 GiB machine. Peak is decimal GB; swap is a system-wide before/after measurement for this run.</p>{source_link('context-4204-256-qwen35-9b8.json')}</div></section><section class="section milestone-detail"><span class="index">04 / CONNECTED LOCAL EXPERIENCE</span><div><h2>CLI. API. Browser.<br>A single working engine.</h2><p>Real local UI generation, seeded baseline sampling, streamed chat, cancellation and recovery have exercised the service. A separate explicit experimental API control matched baseline tokens, returned strict JSON in both requests and completed the SSE stream.</p><div class="actions"><a class="button" href="phase2.html">Watch recorded execution →</a><a class="textlink" href="benchmarks/reports/phase2-experimental-api-qwen35-9b8.json">Experimental API control JSON ↗</a></div></div></section>{adapter_milestone()}{joint_milestone()}<section class="closing compact"><h2>Trace every milestone.</h2><p>The full technical record includes reproducible commands, protocols, source revisions and experiment history.</p><a class="button primary" href="qwen27.html">Open technical documentation ↗</a><a class="textlink" href="data/showcase-provenance.json">Showcase source hashes ↗</a></section>'''


def demo_page(ui):
    metrics = ui['run']['result']['metrics']
    output = html.escape(ui['run']['result']['text'])
    return f'''<section class="page-heading"><span class="index">LOCAL DEMO / 本地演示</span><h1>See the engine.<br><span>Follow the response.</span></h1><p class="lead">Real Qwen3.5-9B-8bit execution, from a repaired Python function to a structured streamed response. 真实本地运行，可查看截图、回放和来源。</p></section><section class="section demo-split"><div><span class="index">RECORDED LOCAL UI / 真实截图</span><h2>A practical code repair.</h2><p>The local browser workspace repaired a Python clamp function and produced the same token sequence as direct engine execution.</p><pre class="code output-code">{output}</pre><div class="demo-metrics"><div><strong>{metrics['ttft_ms']/1000:.2f}s</strong><span>first token</span></div><div><strong>{metrics['total_ms']/1000:.2f}s</strong><span>request total</span></div><div><strong>{metrics['decode_tps']:.2f}</strong><span>decode tok/s</span></div></div><p class="measurement">Single recorded interactive request, 62 input tokens and 26 generated tokens including EOS; measured on M4 Pro / 24 GiB. Request timing describes this interaction.</p>{source_link('phase2-ui-qwen35-9b8.json')}</div><a href="assets/phase2-local-ui-qwen35-9b8.png"><img class="capture" src="assets/phase2-local-ui-qwen35-9b8.png" alt="Real Safari local UI showing a completed clamp function repair and request metrics"></a></section><section class="section"><div class="sectionhead"><div><span class="index">RECORDED SSE REPLAY / 录制回放</span><h2>Watch a real text stream.</h2></div><p>A structured extraction request from the recorded baseline API control. Playback follows captured content-event timestamps at your selected speed. All inference ran locally when recorded.</p></div><div class="replay"><div class="replay-head"><span><i class="record-dot"></i> RECORDED REPLAY · QWEN3.5-9B-8BIT / BASELINE</span><span>2,048 input tokens</span></div><div class="replay-task">Extract project IDs, integer costs and owners into a strict JSON object.</div><pre class="replay-output" id="replay-output" aria-label="Recorded response output">Press play to replay the captured response.</pre><div class="replay-controls"><button class="button primary" id="replay-play">Play recording</button><button class="button" id="replay-reset">Reset</button><label>Playback <select id="replay-speed"><option value="1">1× recorded time</option><option value="4" selected>4× recorded time</option></select></label><span id="replay-status" role="status">Ready · recorded baseline response</span></div></div><p class="measurement">Recorded replay uses the real baseline content events and elapsed timestamps, including the recorded prefill wait. Playback speed changes presentation only.</p>{source_link('phase2-experimental-api-qwen35-9b8.json', 'Source request, SSE events and token controls ↗')}</section>{quickstart()}'''


def guide_page():
    return f'''<section class="page-heading"><span class="index">RUN LOCALLY / 本地运行</span><h1>From checkout<br><span>to local inference.</span></h1><p class="lead">Start with the locked primary 9B text model, then connect through your preferred interface.</p></section>{quickstart()}{model_capabilities()}<section class="section"><span class="index">THE SAME MODEL, THREE INTERFACES</span><h2>Start with one request.</h2><div class="interface-examples"><div><h3>Terminal</h3><pre class="code">uv run deepinfer-mac run \\
  --profile qwen3.5-9b-8bit \\
  --mode baseline 'Explain KV caching in two sentences.'</pre></div><div><h3>Chat API</h3><pre class="code">curl http://127.0.0.1:8765/v1/chat/completions \\
  -H 'Content-Type: application/json' \\
  -d '{{"model":"qwen3.5-9b-8bit","messages":[{{"role":"user","content":"Hello"}}],"max_tokens":128,"temperature":0,"stream":true}}'</pre></div></div><h3>Local browser workspace</h3><p>Open http://127.0.0.1:8765 and choose the 9B8bit profile. The wheel includes the UI assets; the local application runs with Python.</p><a class="textlink" href="docs/compatibility.md">Complete interface and model reference ↗</a></section><section class="closing compact"><h2>Explore the adaptive path.</h2><p>Cost-aware hybrid decoding uses an explicit experimental engine entry. The architecture guide links its model identity, state handling and controls.</p><a class="button" href="architecture.html">Architecture →</a></section>'''


def docs_page():
    groups = [
        ("Use the engine", [("README.md", "Installation and API"), ("README.zh-CN.md", "中文使用指南"), ("compatibility.md", "Interface and model reference")]),
        ("Understand the design", [("architecture.md", "Execution architecture"), ("DS_DESIGN_MAPPING.md", "DeepInfer method mapping"), ("technical-note.md", "Technical note and attribution")]),
        ("Explore the evidence", [("PHASE2_MODEL_SELECTION.md", "Model selection and quality"), ("PHASE2_CACHE_COMPARISON.md", "Service comparison protocol"), ("PHASE2_QUALITY_PROTOCOL.md", "Quality evaluation protocol"), ("evidence.md", "Experiment and release history")]),
    ]
    cards = ''.join('<section class="doc-group"><span class="index">'+heading+'</span>'+''.join(f'<a href="docs/{path}">{label}<span>↗</span></a>' for path,label in links)+'</section>' for heading,links in groups)
    return f'''<section class="page-heading"><span class="index">TECHNICAL DOCUMENTATION / 技术文档</span><h1>Go deeper.<br><span>Follow the source.</span></h1><p class="lead">Complete design notes, evaluation protocols and experiment records, preserved alongside the implementation. 完整技术资料与研究记录，可追溯、可复现。</p></section><div class="doc-grid">{cards}</div><section class="section"><span class="index">LOCKED INPUTS / 可追溯来源</span><h2>Implementation and attribution.</h2><div class="actions"><a class="button primary" href="{BRANCH}">Source code ↗</a><a class="button" href="docs/models.lock.json">Model locks ↗</a><a class="button" href="docs/upstream.lock.json">Upstream locks ↗</a><a class="button" href="docs/NOTICE">NOTICE ↗</a></div><p class="muted">Independent community implementation, with MLX / MLX-LM execution and a DeepInfer-derived scheduler. Exact responsibilities and source revisions are recorded in the method mapping.</p></section>'''


def contribute_page():
    return f'''<section class="page-heading"><span class="index">OPEN SOURCE / 共同构建</span><h1>Make local inference<br><span>more inspectable.</span></h1><p class="lead">Contribute a focused improvement, a reproducible report or a well-defined model integration.</p></section><section class="section">{capabilities()}<div class="actions"><a class="button primary" href="{REPO}/issues">Discuss an improvement ↗</a><a class="button" href="docs/CONTRIBUTING.md">Contribution guide ↗</a></div></section><section class="section"><span class="index">A SHARED ENGINEERING LANGUAGE</span><h2>Code. Context. Reproduction.</h2><p>Pin model and source revisions, keep complete token controls, and publish the command behind the result. Exact state transitions and cancellation behavior are valuable contributions alongside new execution paths.</p><pre class="code">uv sync --frozen --extra dev
uv run pytest -q -m 'not metal'
uv run ruff check src tests scripts benchmarks/experiments benchmarks/quality</pre><a class="textlink" href="docs/CONTRIBUTING.md">Full development setup ↗</a></section>'''


def verify_and_prepare_data():
    """Bind headline milestones and recorded replay to actual report fields."""
    ui = load_report("phase2-ui-qwen35-9b8.json")
    cache = load_report("phase2-message-checkpoints-qwen35-9b8.json")
    context = load_report("context-4204-256-qwen35-9b8.json")
    supplement = load_report("phase2-hybrid-lagged-dev-qwen35-9b8-supplemental-verification.json")
    observed = supplement["real_probe_observations"]
    assert observed["mature_core_decisions"] == 28 and observed["all_token_ids_exact"]
    assert observed["baseline_output_tokens"] == observed["lagged_output_tokens"] == 128
    assert supplement["corrected_verifier_json_result"]["decisions"] == 28
    assert supplement["corrected_verifier_json_result"]["matches"]
    assert len(cache["records"]) == 20 and len(cache["checks"]) == 16
    assert cache["all_token_checks_equal"] and all(check["equal"] for check in cache["checks"])
    for sampling in ("greedy", "official"):
        branches = {record["variant"]: record["result"]["metrics"]["cache_hit_tokens"]
                    for record in cache["records"]
                    if record["sampling"] == sampling and record["kind"] == "changed_tail_branch"}
        assert branches["off_cached"] == 512 and branches["on_cached"] == 2014
    assert round(context["result"]["metrics"]["peak_memory_bytes"] / 1e9, 2) == 10.65
    assert context["actual_input_tokens"] == 4204 and context["actual_output_tokens"] == 256
    assert context["completed_configured_length"] and context["stream_text_matches"]
    assert context["system_before"]["swap_used_bytes"] == context["load"]["swap_used_bytes"] == context["system_after"]["swap_used_bytes"]
    api = load_report("phase2-experimental-api-qwen35-9b8.json")
    baseline = next(record for record in api["records"] if record["request"]["mode"] == "baseline")
    events = []
    for event in baseline["events"]:
        payload = event.get("payload")
        if isinstance(payload, dict):
            for choice in payload.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    events.append({"elapsed_ms": event["elapsed_ms"], "text": content})
    assert events and all(a["elapsed_ms"] <= b["elapsed_ms"] for a, b in pairwise(events))
    assert json.loads("".join(event["text"] for event in events)) == baseline["case"]["expected_json"]
    replay = {"kind": "recorded_sse_replay", "source": "phase2-experimental-api-qwen35-9b8.json", "model": "qwen3.5-9b-8bit", "mode": "baseline", "prompt_tokens": 2048, "events": events}
    (DIST / "data").mkdir(exist_ok=True)
    (DIST / "data/recorded-response.json").write_text(json.dumps(replay, indent=2) + "\n")
    sources = {name: hashlib.sha256((BUNDLE / "benchmarks/reports" / name).read_bytes()).hexdigest() for name in REPORTS}
    (DIST / "data/showcase-provenance.json").write_text(json.dumps({"date": "2026-10-21", "source_commit": SOURCE_COMMIT, "bundle_manifest": "phase2-source-manifest.json", "scope": "Source-linked engineering milestones and recorded interactions; checkout runtime is independent", "reports_sha256": sources}, indent=2) + "\n")
    return ui


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="Optional schema-2 historical report retained as a downloadable record")
    args = parser.parse_args()
    source = args.report or SITE / "data/benchmark.json"
    if source.exists():
        report = json.loads(source.read_text())
        if report.get("schema_version") != 2 or not report.get("samples") or not report.get("environment"):
            raise ValueError("Expected a schema-2 report with environment and complete samples")
    elif args.report:
        raise FileNotFoundError(source)
    manifest = verify_bundle()
    if DIST.exists():
        shutil.rmtree(DIST)
    shutil.copytree(SITE / "assets", DIST / "assets")
    target = DIST / "benchmarks/reports"
    target.mkdir(parents=True)
    for name in REPORTS:
        shutil.copy2(BUNDLE / "benchmarks/reports" / name, target / name)
    shutil.copy2(BUNDLE / "docs/evidence/phase2-local-ui-qwen35-9b8.png",
                 DIST / "assets/phase2-local-ui-qwen35-9b8.png")
    (DIST / "data").mkdir()
    (DIST / "data/phase2-source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ui = verify_and_prepare_data()
    pages = {
        "index.html": ("Adaptive inference for Apple Silicon", homepage(), False),
        "zh.html": ("让本地模型，更懂你的 Mac", homepage(True), True),
        "architecture.html": ("Architecture", architecture_page(), False),
        "benchmarks.html": ("Engineering milestones", milestones_page(), False),
        "phase2.html": ("Local demo", demo_page(ui), False),
        "compatibility.html": ("Run locally", guide_page(), False),
        "contribute.html": ("Contribute", contribute_page(), False),
        "qwen27.html": ("Technical documentation", docs_page(), False),
    }
    for name, (title, body, zh) in pages.items():
        (DIST / name).write_text(shell(title, body, zh))
    (DIST / ".nojekyll").touch()
    # Historical source bytes remain untouched; keep them reachable from raw docs.
    if source.exists():
        shutil.copy2(source, DIST / "data/benchmark.json")
        for name in ("source.tar.gz", "source-manifest.json", "cache-qwen3-4b.json"):
            artifact = source.parent / name
            if artifact.exists():
                shutil.copy2(artifact, DIST / "data" / name)
    print(f"Built {DIST}: {len(pages)} pages; source-linked architecture and milestones")


if __name__ == "__main__":
    main()
