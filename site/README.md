# Static architecture showcase

Dependency-free HTML/CSS/JavaScript generated with the Python standard library. The existing GitHub Pages artifact stays relocatable under `/deepinfer-mac/`.

```sh
.venv/bin/python site/build.py
.venv/bin/python site/check.py
python3 -m http.server 8088 --directory site/dist
```

For GitHub Pages prefix QA, serve a temporary directory containing `deepinfer-mac -> site/dist`, then open `/deepinfer-mac/zh.html` or `/deepinfer-mac/`.

The eight existing URLs present the architecture, implemented capabilities, verified engineering milestones, real recorded demo, installation and source documentation. English and Chinese homepages are complete entry points. `qwen27.html` is retained as a legacy URL for the technical documentation hub. Technical-document links are GitHub permalinks to the published Phase2 source commit `66ae444f535c63b43003ba6febeb42228d4d607a`; the active checkout runtime is not presented as that evidence source. Historical raw site artifacts remain byte-identical.

The site build checks the real source report fields behind the milestone numbers. It emits `data/showcase-provenance.json` with source report SHA-256 digests. The SSE replay is extracted from actual baseline content events in `phase2-experimental-api-qwen35-9b8.json`; playback uses captured timestamps, with an explicit 1× / 4× presentation speed. The source raw report remains linked. The screenshot and single-interaction UI metrics come from the recorded real 9B request.

The build is self-contained under `site/`: `site/data/phase2/manifest.json` binds an exact allowlist of eight report files and one screenshot to their published Phase2 source paths, commit and SHA-256 digests. Every build verifies exact bundle membership, source identity, length and digest, then copies the permitted report bytes. It reads no root runtime, model lock, README, research document or report. A Pages-only source update can therefore build against the alpha `main` checkout without importing the experimental engine. Restricted dataset tasks and generations are excluded. The static showcase sends no prompts, runs no model, collects no visitor data, loads no remote fonts and never automatically connects to visitor localhost. JavaScript only controls local presentation, clipboard copying and the same-origin recorded-response JSON.

The architecture is attributed to its actual components: MLX/MLX-LM execution and native LRU, DeepInfer-derived scheduling, and the Mac hybrid recovery/service adaptations. Milestones describe correctness and a specific local resource workload; they are not competitor throughput comparisons.

`site/dist` is generated and excluded from Git. The GitHub Actions workflow uploads its contents as the deployable artifact. Building locally does not deploy the website. Historical alpha release artifacts and `site/data/` source bytes stay untouched. An optional `--report <schema-2-summary.json>` retains an explicitly supplied historical report as downloadable data.

## Phase 3 product presentation (local candidate)

The product entry now presents an Apple Silicon inference framework with MLX as
its execution backend; 9B remains the verified demonstration, not the product
definition. Both homepages and the run guide include the same capability matrix.
The adapter interface is labelled experimental / acceptance in progress;
cross-request/pipeline entries remain labelled building until their separate
acceptance completes. Existing real milestones and the recorded
SSE replay retain their frozen source bytes and attribution.

`architecture.html#joint-budget` adds a browser-only allocation illustration.
Its three conditional-probability arrays are hand-set teaching inputs, explicitly
labelled **mechanism illustration**, not model predictions or measurements. The
slider selects candidate positions by cumulative prefix survival with stable
request/position ties. It does not implement cost-based total-budget selection,
target/golden-token accounting, real verification, or state recovery. It is not a
performance animation and does not announce Phase 3 execution as complete.

Build and link checks remain the commands above. Local source changes and a
successful build do not update the deployed website or publish a release.
