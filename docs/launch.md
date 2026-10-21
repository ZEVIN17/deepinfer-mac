# Release checklist and announcement draft

Run the commands in PROGRESS; inspect wheel/sdist, static assets, all links and real evidence. Keep weights, private prompts, credentials and caches out of Git and archives. Publish a tagged alpha only after checks pass. This draft is not evidence of publication.

GitHub Pages uses `.github/workflows/pages.yml`, builds `site/dist/`, then uploads the complete artifact. Configure the repository Pages source to **GitHub Actions**. The initial repository commit must be pushed before a workflow can execute. Expected repository URL after successful deployment: https://zevin17.github.io/deepinfer-mac/ (do not call it live until verified).

Announcement: “DeepInfer-Mac is an independent experiment in local Apple Silicon inference powered by MLX. Try the bundled local UI, reproduce ordinary/fixed/adaptive comparisons, and contribute complete hardware reports. We publish failures and numerical mismatches along with timings; no universal speedup claim. See README, benchmark evidence and limitations.”

中文发布文案：“DeepInfer-Mac 是受 DeepInfer 启发、由 MLX 驱动的独立社区实验。欢迎运行本地 UI、复现三模式与缓存比较、贡献完整硬件报告。性能结果包含失败与数值差异，不承诺普遍加速。安装、基准与限制见仓库。”

External community posting needs explicit user instruction; no posts are sent by this build task.
